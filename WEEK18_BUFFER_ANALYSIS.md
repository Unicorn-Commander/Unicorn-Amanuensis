# Week 18: Buffer Pool Architecture Analysis

**Date**: November 2, 2025
**Team**: Buffer Management Team Lead
**Status**: Phase 1 Analysis Complete

---

## Executive Summary

Week 17 testing revealed a critical buffer pool size limitation that prevents long-form audio transcription beyond ~7.7 seconds. This analysis documents the current buffer architecture, identifies the specific size limits, and proposes a configuration strategy to support 30s, 60s, and 120s audio clips.

**Key Finding**: The audio buffer pool in `transcription_pipeline.py` is configured for only **122,880 samples** (~7.7s at 16kHz), but 30s audio requires **480,000 samples**.

---

## Current Buffer Architecture

### Buffer Pool Locations

The Unicorn-Amanuensis service uses **two separate buffer pool configurations**:

#### 1. Server Buffer Pool (`xdna2/server.py`)
**Location**: Lines 756-780
**Usage**: Sequential mode transcription
**Configuration**:
```python
buffer_manager.configure({
    'mel': {
        'size': 960 * 1024,      # 960KB
        'count': 10,
        'max_count': 20,
        'dtype': np.float32,
        'shape': (3000, 80),     # 3000 frames × 80 mels (30s audio)
        'zero_on_release': False
    },
    'audio': {
        'size': 480 * 1024,      # 480KB for audio
        'count': 5,
        'max_count': 15,
        'dtype': np.float32,
        'zero_on_release': False
    },
    'encoder_output': {
        'size': 3 * 1024 * 1024, # 3MB
        'count': 5,
        'max_count': 15,
        'dtype': np.float32,
        'shape': (3000, 512),    # 3000 frames × 512 hidden
        'zero_on_release': False
    }
})
```

**Audio Buffer Capacity**:
- Size: 480 KB
- Samples: 480KB / 4 bytes = **120,000 samples**
- Duration: 120,000 / 16,000 Hz = **7.5 seconds**

#### 2. Pipeline Buffer Pool (`transcription_pipeline.py`)
**Location**: Uses GlobalBufferManager.instance() (singleton)
**Usage**: Pipeline mode transcription (concurrent processing)
**Configuration**: **INHERITS** from server configuration

**Problem**: Pipeline acquires audio buffer at line 408:
```python
audio_buffer = self.buffer_manager.acquire('audio')
```

Then tries to copy audio at line 426:
```python
np.copyto(audio_buffer[:len(audio)], audio)
```

For 30s audio:
- `len(audio)` = 480,000 samples
- `audio_buffer.shape[0]` = 122,880 samples ← **MISMATCH!**

---

## Buffer Size Limits Identified

### Current Limits

| Buffer Type | Configured Size | Max Duration | Issue |
|-------------|----------------|--------------|-------|
| **Mel** | 960 KB (3000 frames × 80 mels) | **30s** | ✅ OK |
| **Audio** | 122,880 samples | **7.7s** | ❌ TOO SMALL |
| **Encoder Output** | 3 MB (3000 frames × 512) | **30s** | ✅ OK |

### Why Audio Buffer is Only 122,880 Samples

Looking at the server.py configuration (line 766):
```python
'audio': {
    'size': 480 * 1024,      # 480KB for audio
```

**But wait - where does 122,880 come from?**

Let me check if there's a shape configuration:

Looking at buffer_pool.py (lines 138-140):
```python
self.shape = shape or (buffer_size // dtype(1).itemsize,)
```

If `shape` is not provided:
- `buffer_size = 480 * 1024 = 491,520 bytes`
- `dtype.itemsize = 4` (for float32)
- `shape = (491,520 / 4,) = (122,880,)`

**FOUND IT!** The audio buffer pool doesn't specify a `shape`, so it defaults to `buffer_size // 4` samples.

### Week 17 Test Failure

**Test**: 30-second audio (`test_30s.wav`)
**Error**:
```
Pipeline processing failed: Load/Mel failed: could not broadcast input array
from shape (480000,) into shape (122880,)
```

**Root Cause**: Audio buffer pool has only 122,880 samples, but 30s audio is 480,000 samples.

**Location**: `transcription_pipeline.py`, line 426
```python
np.copyto(audio_buffer[:len(audio)], audio)  # FAILS when len(audio) > 122,880
```

---

## Memory Budget Analysis

### Current Configuration (10s max audio)

| Buffer Type | Per-Buffer Size | Initial Count | Max Count | Total Memory |
|-------------|-----------------|---------------|-----------|--------------|
| Mel | 960 KB | 10 | 20 | 19.2 MB |
| Audio | 480 KB | 5 | 15 | 7.2 MB |
| Encoder | 3 MB | 5 | 15 | 45 MB |
| **Total** | - | - | - | **71.4 MB** |

### Proposed Configuration (120s max audio)

**Target**: Support 120-second audio with <100 MB total memory

**Calculation**:
- 120s audio @ 16kHz = **1,920,000 samples** = **7.68 MB** per buffer
- Mel for 120s = ~12,000 frames × 80 mels = **3.84 MB** per buffer
- Encoder output = ~12,000 frames × 512 hidden = **24.6 MB** per buffer

**New configuration**:

| Buffer Type | Per-Buffer Size | Initial Count | Max Count | Total Memory |
|-------------|-----------------|---------------|-----------|--------------|
| Mel | 3.84 MB | 10 | 20 | 76.8 MB |
| Audio | 7.68 MB | 3 | 10 | 76.8 MB |
| Encoder | 24.6 MB | 3 | 10 | 246 MB |
| **Total** | - | - | - | **~400 MB** |

**Concern**: This exceeds the 100 MB target significantly!

---

## Configuration Strategy

### Option 1: Environment Variable (RECOMMENDED)

Make buffer pool size **configurable** via environment variable:

```python
import os

# Configuration
MAX_AUDIO_DURATION = int(os.getenv('MAX_AUDIO_DURATION', '120'))  # seconds
SAMPLE_RATE = 16000

# Calculate buffer sizes
MAX_AUDIO_SAMPLES = MAX_AUDIO_DURATION * SAMPLE_RATE
MAX_MEL_FRAMES = (MAX_AUDIO_SAMPLES // 160) * 2  # hop_length=160, conv1d stride=2 reduces by 2×
MAX_ENCODER_FRAMES = MAX_MEL_FRAMES

# Buffer pool configuration
buffer_manager.configure({
    'mel': {
        'size': MAX_MEL_FRAMES * 80 * 4,  # frames × mels × sizeof(float32)
        'count': 10,
        'max_count': 20,
        'dtype': np.float32,
        'shape': (MAX_MEL_FRAMES, 80),
        'zero_on_release': False
    },
    'audio': {
        'size': MAX_AUDIO_SAMPLES * 4,  # samples × sizeof(float32)
        'count': 5,
        'max_count': 15,
        'dtype': np.float32,
        'shape': (MAX_AUDIO_SAMPLES,),  # ← CRITICAL: Must specify shape!
        'zero_on_release': False
    },
    'encoder_output': {
        'size': MAX_ENCODER_FRAMES * 512 * 4,  # frames × hidden × sizeof(float32)
        'count': 5,
        'max_count': 15,
        'dtype': np.float32,
        'shape': (MAX_ENCODER_FRAMES, 512),
        'zero_on_release': False
    }
})
```

**Usage**:
```bash
# Default: 120s audio
python -m uvicorn xdna2.server:app --port 9000

# Custom: 60s audio
MAX_AUDIO_DURATION=60 python -m uvicorn xdna2.server:app --port 9000

# Short audio only: 10s
MAX_AUDIO_DURATION=10 python -m uvicorn xdna2.server:app --port 9000
```

**Advantages**:
- ✅ User controls memory usage
- ✅ Backward compatible (default 120s)
- ✅ No code changes needed for different durations
- ✅ Easy to document

**Disadvantages**:
- ⚠️  Users must know to set environment variable
- ⚠️  No runtime adjustment (requires restart)

---

### Option 2: Dynamic Buffer Allocation

Allocate buffers **on-demand** based on actual audio length:

```python
def _process_load_mel(self, item: WorkItem) -> WorkItem:
    # Load audio first
    audio = load_audio(tmp_path)
    audio_len = len(audio)

    # Calculate required buffer size
    required_mel_frames = (audio_len // 160) * 2

    # Acquire buffer of appropriate size
    if audio_len <= 160000:  # 10s
        audio_buffer = self.buffer_manager.acquire('audio_10s')
        mel_buffer = self.buffer_manager.acquire('mel_10s')
    elif audio_len <= 480000:  # 30s
        audio_buffer = self.buffer_manager.acquire('audio_30s')
        mel_buffer = self.buffer_manager.acquire('mel_30s')
    else:  # 120s
        audio_buffer = self.buffer_manager.acquire('audio_120s')
        mel_buffer = self.buffer_manager.acquire('mel_120s')

    # Process as normal
    ...
```

**Advantages**:
- ✅ Optimal memory usage (small buffers for short audio)
- ✅ Supports unlimited audio lengths
- ✅ No user configuration needed

**Disadvantages**:
- ❌ Complex implementation
- ❌ More pool management overhead
- ❌ Harder to tune pool sizes

---

### Option 3: Streaming Approach

Process audio **in chunks** instead of loading entirely:

```python
def _process_load_mel_streaming(self, item: WorkItem) -> WorkItem:
    # Stream audio in 30s chunks
    CHUNK_SIZE = 30 * 16000  # 30s

    audio_chunks = []
    for offset in range(0, len(audio), CHUNK_SIZE):
        chunk = audio[offset:offset+CHUNK_SIZE]

        # Process chunk with fixed-size buffer
        audio_buffer = self.buffer_manager.acquire('audio')
        np.copyto(audio_buffer[:len(chunk)], chunk)

        mel_chunk = compute_mel_spectrogram_zerocopy(
            audio_buffer[:len(chunk)],
            self.feature_extractor,
            output=mel_buffer
        )

        audio_chunks.append(mel_chunk)
        self.buffer_manager.release('audio', audio_buffer)

    # Concatenate chunks
    mel = np.concatenate(audio_chunks, axis=0)
    ...
```

**Advantages**:
- ✅ Supports unlimited audio length
- ✅ Fixed memory usage
- ✅ Better for very long audio (hours)

**Disadvantages**:
- ❌ Complex implementation
- ❌ Chunk boundary handling
- ❌ May impact accuracy (mel spectrogram windowing)

---

## Recommended Approach

**For Week 18**: **Option 1 (Environment Variable)**

**Rationale**:
1. **Simple**: ~50 lines of code changes
2. **Fast**: 30 minutes to implement and test
3. **Flexible**: Users can tune for their use case
4. **Backward Compatible**: Default to 120s (safe for most cases)
5. **Documented**: Easy to explain in user guide

**For Future (Week 20+)**: Consider **Option 3 (Streaming)** for unlimited audio

---

## Implementation Plan

### Step 1: Add Environment Variable Support (10 minutes)

**File**: `xdna2/server.py`
**Lines**: Before line 756 (buffer pool configuration)

```python
import os

# Buffer pool configuration (user-configurable via environment)
MAX_AUDIO_DURATION = int(os.getenv('MAX_AUDIO_DURATION', '120'))  # seconds (default: 120s)
SAMPLE_RATE = 16000

# Calculate buffer sizes dynamically
MAX_AUDIO_SAMPLES = MAX_AUDIO_DURATION * SAMPLE_RATE
MAX_MEL_FRAMES = (MAX_AUDIO_SAMPLES // 160) * 2  # hop_length=160, conv1d stride=2
MAX_ENCODER_FRAMES = MAX_MEL_FRAMES

logger.info(f"[BufferPool] Configured for audio up to {MAX_AUDIO_DURATION}s")
logger.info(f"  Audio buffer: {MAX_AUDIO_SAMPLES:,} samples ({MAX_AUDIO_SAMPLES*4/1024/1024:.1f} MB)")
logger.info(f"  Mel buffer: {MAX_MEL_FRAMES} frames ({MAX_MEL_FRAMES*80*4/1024/1024:.1f} MB)")
logger.info(f"  Encoder buffer: {MAX_ENCODER_FRAMES} frames ({MAX_ENCODER_FRAMES*512*4/1024/1024:.1f} MB)")
```

### Step 2: Update Buffer Pool Configuration (10 minutes)

**File**: `xdna2/server.py`
**Lines**: 756-780 (replace hardcoded values)

```python
buffer_manager.configure({
    'mel': {
        'size': MAX_MEL_FRAMES * 80 * 4,
        'count': 10,
        'max_count': 20,
        'dtype': np.float32,
        'shape': (MAX_MEL_FRAMES, 80),
        'zero_on_release': False
    },
    'audio': {
        'size': MAX_AUDIO_SAMPLES * 4,
        'count': 5,
        'max_count': 15,
        'dtype': np.float32,
        'shape': (MAX_AUDIO_SAMPLES,),  # ← CRITICAL FIX!
        'zero_on_release': False
    },
    'encoder_output': {
        'size': MAX_ENCODER_FRAMES * 512 * 4,
        'count': 5,
        'max_count': 15,
        'dtype': np.float32,
        'shape': (MAX_ENCODER_FRAMES, 512),
        'zero_on_release': False
    }
})
```

### Step 3: Test Configuration (10 minutes)

```bash
# Test 1: 30s audio with default config (120s)
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis
python -m pytest tests/integration_test_week15.py::test_30s_audio -v

# Test 2: 30s audio with minimal config (30s)
MAX_AUDIO_DURATION=30 python -m pytest tests/integration_test_week15.py::test_30s_audio -v

# Test 3: Memory usage check
MAX_AUDIO_DURATION=120 python -c "
import os
os.environ['MAX_AUDIO_DURATION'] = '120'
from xdna2.server import startup_event
import asyncio
asyncio.run(startup_event())
"
```

---

## Expected Outcomes

### With Default Configuration (120s)

**Memory Usage**:
- Audio pool: 5 × 7.68 MB = **38.4 MB**
- Mel pool: 10 × 3.84 MB = **38.4 MB**
- Encoder pool: 5 × 24.6 MB = **123 MB**
- **Total**: **~200 MB** (exceeds 100 MB target)

**Supported Audio**:
- ✅ 1s audio (test_1s.wav)
- ✅ 5s audio (test_5s.wav)
- ✅ 30s audio (test_30s.wav)
- ✅ 60s audio (test_60s.wav)
- ✅ 120s audio (test_120s.wav)

### With Conservative Configuration (30s)

**Memory Usage**:
- Audio pool: 5 × 1.92 MB = **9.6 MB**
- Mel pool: 10 × 960 KB = **9.6 MB**
- Encoder pool: 5 × 6.14 MB = **30.7 MB**
- **Total**: **~50 MB** ✅ Under 100 MB target!

**Supported Audio**:
- ✅ 1s audio
- ✅ 5s audio
- ✅ 30s audio
- ❌ 60s audio (too large)
- ❌ 120s audio (too large)

---

## Buffer Reuse Analysis

**Current Implementation**: ✅ **Already Implemented**

Looking at `transcription_pipeline.py` lines 463-466:
```python
# Release buffers on error
if mel_buffer is not None:
    self.buffer_manager.release('mel', mel_buffer)
if audio_buffer is not None:
    self.buffer_manager.release('audio', audio_buffer)
```

And in `xdna2/server.py` lines 1137-1147:
```python
finally:
    # CRITICAL: Always release buffers back to pool
    if mel_buffer is not None:
        buffer_manager.release('mel', mel_buffer)
    if audio_buffer is not None:
        buffer_manager.release('audio', audio_buffer)
    if encoder_buffer is not None:
        buffer_manager.release('encoder_output', encoder_buffer)
```

**Verdict**: Buffer reuse is already operational via the pool's acquire/release mechanism.

**From Week 8 Implementation**:
- Buffer hit rate: 100% (all buffers reused)
- No memory leaks detected
- Proper cleanup in all code paths (success and error)

**No additional work needed** for buffer reuse optimization.

---

## Memory Optimization Strategies

### Strategy 1: Reduce Pool Pre-allocation

**Current**: 10 mel buffers pre-allocated
**Proposed**: 3 mel buffers pre-allocated, grow to 10 on demand

```python
'mel': {
    'count': 3,      # ← Reduce from 10
    'max_count': 10, # Keep max the same
    ...
}
```

**Savings**: 7 × 3.84 MB = **~27 MB** initial memory reduction

### Strategy 2: Share Buffers Between Stages

**Observation**: Audio buffer is only used in Stage 1, then released before Stage 2.

**Optimization**: Reuse audio buffer space for encoder output:
```python
# Stage 1: Use audio buffer
audio_buffer = self.buffer_manager.acquire('audio')
# ... compute mel ...
self.buffer_manager.release('audio', audio_buffer)  # Free immediately

# Stage 2: Reuse same memory for encoder
encoder_buffer = self.buffer_manager.acquire('audio')  # ← Renamed pool
# ... encode ...
```

**Savings**: Eliminate separate encoder pool = **~123 MB**

**Risk**: ⚠️ Complexity, potential for bugs

---

## Recommendations

### Week 18 (Immediate)

1. **Implement Option 1 (Environment Variable)** ← **DO THIS**
   - Simple, fast, effective
   - 30 minutes implementation
   - Solves 30s audio issue immediately

2. **Set Default to 30s** (not 120s)
   - Keeps memory usage under 100 MB target
   - Users can increase if needed
   - Safer default for resource-constrained systems

3. **Document Configuration**
   - Add to DEPLOYMENT_GUIDE.md
   - Include memory usage table
   - Provide tuning guidelines

### Week 19-20 (Future)

4. **Implement Strategy 1 (Reduce Pre-allocation)**
   - Further reduce initial memory footprint
   - 2-3 hours implementation
   - ~30% memory savings

5. **Investigate Streaming (Option 3)**
   - For unlimited audio length support
   - 1 week research + implementation
   - Enable transcription of hours-long audio

---

## Risk Assessment

### Implementation Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Memory exhaustion** | Medium | Default to 30s (not 120s) |
| **Configuration confusion** | Low | Clear documentation + examples |
| **Backward compatibility** | Low | Default preserves current behavior |
| **Performance impact** | Very Low | No runtime overhead (config-time only) |

### Testing Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **30s audio still fails** | Low | Calculation validated above |
| **Memory leak** | Low | Buffer pool already tested (Week 8) |
| **Accuracy degradation** | Very Low | No algorithmic changes |

---

## Conclusion

**Current Status**: Buffer pool is correctly implemented with reuse, but sized too small for long-form audio.

**Root Cause**: Audio buffer pool defaults to `buffer_size // dtype.itemsize` when `shape` not specified, resulting in only 122,880 samples (~7.7s).

**Solution**: Add environment variable `MAX_AUDIO_DURATION` to configure buffer sizes dynamically.

**Expected Outcome**:
- ✅ 30s audio working
- ✅ Configurable up to 120s
- ✅ Memory usage under 100 MB (with 30s default)
- ✅ No code changes for users (environment variable)

**Next Step**: Implement the fix and test with long-form audio.

---

**Analysis Completed**: November 2, 2025
**Team Lead**: Buffer Management Team
**Status**: Ready to implement fix (Phase 1 complete)

**Built with 🦄 by Magic Unicorn Unconventional Technology & Stuff Inc**
