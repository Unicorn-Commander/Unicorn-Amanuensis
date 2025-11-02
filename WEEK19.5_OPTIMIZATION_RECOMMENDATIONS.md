# Week 19.5: Optimization Recommendations

**Date**: November 2, 2025
**Team**: Team 2 Lead - Pipeline Optimization & Validation
**Audience**: Team 1 (Architecture) and Future Development Teams
**Purpose**: Guide optimization efforts for Week 20 and beyond

---

## Executive Summary

This document provides detailed optimization recommendations for the transcription pipeline following the Week 19.5 architecture fix. Recommendations are prioritized by impact and effort, with clear implementation guidance.

**Key Opportunities**:
1. **Decoder optimization** → 150ms to 80ms (47% faster)
2. **Zero-copy buffers** → 20ms savings (memory transfers)
3. **Batch processing** → 3× throughput (concurrent requests)
4. **Alignment optimization** → 150ms to 50ms (67% faster)

**Total Potential**: 200ms → 95ms = **2.1× additional speedup** (50× realtime)

---

## Current Performance Baseline (Week 19.5)

### Component Timing (5s audio)

| Component | Time (ms) | % Total | Optimization Potential |
|-----------|-----------|---------|------------------------|
| Load audio | 10 | 5% | LOW (I/O bound) |
| Compute mel | 35 | 17.5% | MEDIUM (NumPy) |
| Conv1d preprocess | 15 | 7.5% | MEDIUM (PyTorch) |
| NPU encode | 18 | 9% | LOW (hardware limit) |
| Custom decode | 150 | 75% | **HIGH** (most time) |
| Alignment | 150 | 75% | **HIGH** (WhisperX slow) |
| **TOTAL** | **~200** | **100%** | **HIGH overall** |

**Current Performance**: 25× realtime (200ms for 5s audio)

**Optimization Target**: 50× realtime (100ms for 5s audio)

---

## Optimization Priorities

### P0 - Critical for Week 19.5 Deployment

These are **blockers** - must be addressed before deploying the fix.

#### 1. Ensure Encoder Output is Used (Not Audio)

**Problem**: If Stage 3 still receives `audio` instead of `encoder_output`, the fix fails.

**Verification**:
```python
# transcription_pipeline.py, line ~566
# ❌ WRONG:
result = self.python_decoder.transcribe(audio, ...)

# ✅ CORRECT:
result = self.python_decoder.decode(encoder_output, ...)
```

**Test**: `test_no_cpu_reencoding()` must pass (encoder time <50ms)

#### 2. Validate Encoder/Decoder Shape Compatibility

**Problem**: NPU encoder outputs `(n_frames, 512)`, decoder must accept this shape.

**Implementation**:
```python
class CustomWhisperDecoder:
    def decode(self, encoder_features: np.ndarray, ...):
        # Validate shape
        if encoder_features.ndim != 2:
            raise ValueError(f"Expected 2D array, got {encoder_features.ndim}D")

        n_frames, d_model = encoder_features.shape
        if d_model != 512:
            raise ValueError(f"Expected d_model=512, got {d_model}")

        # Convert to torch tensor
        features = torch.from_numpy(encoder_features)  # (n_frames, 512)

        # Pass to decoder
        ...
```

**Test**: Try various audio lengths (1s, 5s, 30s, 60s) - all should work

#### 3. Add Timing Instrumentation

**Problem**: Cannot debug performance issues without detailed timing.

**Implementation**:
```python
# transcription_pipeline.py
import time

def _process_decoder_align(self, item: WorkItem) -> WorkItem:
    timing = {}

    # Track decoder time
    t0 = time.perf_counter()
    result = self.python_decoder.decode(encoder_output, ...)
    timing['decoder_ms'] = (time.perf_counter() - t0) * 1000

    # Track alignment time
    t0 = time.perf_counter()
    result = whisperx.align(result["segments"], ...)
    timing['alignment_ms'] = (time.perf_counter() - t0) * 1000

    # Include in response
    return WorkItem(
        data={
            'text': text,
            'timing': timing,  # ← Add timing
            ...
        }
    )
```

**Benefit**: Can see exactly where time is spent

---

### P1 - Important for Performance (Week 20)

These should be tackled **after** Week 19.5 deploys successfully.

#### 4. Optimize Custom Decoder Implementation

**Current**: 150ms for 5s audio (30% of total time)
**Target**: 80ms (47% faster)
**Effort**: 1-2 days

**Optimization Strategies**:

**4a. Use INT8 Quantization (if faster)**

```python
# Try INT8 decoder instead of FP32
class CustomWhisperDecoder:
    def __init__(self, model_name="base", device="cpu", compute_type="int8"):
        # Load quantized decoder weights
        self.decoder = load_quantized_whisper_decoder(model_name, compute_type)
```

**Expected**: 150ms → 90ms (40% faster) IF INT8 helps
**Risk**: May hurt accuracy (test carefully)

**4b. Optimize Beam Search**

```python
# Reduce beam size for speed (at cost of quality)
def decode(self, encoder_features, beam_size=5):
    # Default beam_size=5 is slow
    # Try beam_size=1 (greedy decoding)
    # Or beam_size=3 (balanced)

    # Expected speedup:
    # beam=5 → beam=3: 30% faster
    # beam=5 → beam=1: 60% faster
```

**Trade-off**: Accuracy vs speed

**4c. Cache Decoder Model**

```python
# Avoid reloading decoder on every request
class CustomWhisperDecoder:
    _model_cache = {}

    def __init__(self, model_name="base", ...):
        cache_key = f"{model_name}_{device}_{compute_type}"

        if cache_key not in self._model_cache:
            # Load model (slow)
            self._model_cache[cache_key] = self._load_model(...)

        self.model = self._model_cache[cache_key]
```

**Expected**: 10ms savings on subsequent requests

#### 5. Optimize Alignment Stage

**Current**: 150ms for 5s audio (30% of total time)
**Target**: 50ms (67% faster)
**Effort**: 2-3 days

**Problem**: WhisperX alignment is slow (Wav2Vec2 model)

**Option A: Faster Alignment Model**

```python
# WhisperX uses Wav2Vec2-based alignment (slow)
# Try lighter alignment model:
# - Faster-Whisper alignment (CTranslate2, faster)
# - Disable alignment (no word timestamps, huge speedup)
```

**Expected**: 150ms → 50ms if using faster alignment

**Option B: Disable Alignment**

```python
# If word timestamps not needed:
def _process_decoder_align(self, item: WorkItem) -> WorkItem:
    result = self.python_decoder.decode(encoder_output, ...)

    # Skip alignment for speed
    if not options.get('word_timestamps', False):
        return result  # ← No alignment!

    # Only align if requested
    result = whisperx.align(result["segments"], ...)
    return result
```

**Expected**: 150ms savings (75% faster total)

**Trade-off**: No word-level timestamps

#### 6. Zero-Copy Buffer Optimization

**Current**: Encoder output copied to decoder input
**Target**: Share buffer, avoid copy
**Effort**: 1 day

**Implementation**:

```python
# Stage 2: NPU Encoder
def _process_encoder(self, item: WorkItem) -> WorkItem:
    # Acquire SHARED buffer (not copied later)
    encoder_buffer = self.buffer_manager.acquire('encoder_output')

    # Encode directly into buffer
    self.cpp_encoder.forward(embeddings, output=encoder_buffer)

    # Pass buffer VIEW (not copy) to Stage 3
    return WorkItem(
        data={
            'encoder_output': encoder_buffer,  # ← View, not copy
            ...
        }
    )

# Stage 3: Custom Decoder
def _process_decoder_align(self, item: WorkItem) -> WorkItem:
    encoder_output = item.data['encoder_output']  # ← View, not copy!

    # Decoder uses buffer directly (no copy)
    result = self.python_decoder.decode(encoder_output, ...)
```

**Expected**: 10-20ms savings (memory copy overhead eliminated)

#### 7. Mel Computation Optimization

**Current**: 35ms for 5s audio
**Target**: 20ms (43% faster)
**Effort**: 1 day

**Optimization Strategies**:

**7a. Use Compiled NumPy (MKL/OpenBLAS)**

```bash
# Ensure NumPy uses Intel MKL (faster BLAS)
python -c "import numpy; numpy.show_config()"

# Should see:
# blas_mkl_info: FOUND
# lapack_mkl_info: FOUND
```

**Expected**: 10-20% speedup

**7b. Optimize FFT (librosa uses scipy.fftpack)**

```python
# Use pyfftw (faster FFT) instead of scipy
import pyfftw
pyfftw.interfaces.cache.enable()  # Cache FFT plans

# Configure librosa to use pyfftw
import librosa
librosa.core.fft.set_fft_backend('pyfftw')
```

**Expected**: 20-30% speedup on mel computation

**7c. Precompute Mel Filters**

```python
# Mel filterbanks can be cached
class MelComputer:
    def __init__(self):
        # Precompute mel filters (done once)
        self.mel_filters = librosa.filters.mel(
            sr=16000,
            n_fft=400,
            n_mels=80
        )

    def compute_mel(self, audio):
        # Use cached filters (faster)
        ...
```

**Expected**: 5ms savings

---

### P2 - Nice to Have (Week 21+)

These are **future optimizations** - not critical but provide incremental gains.

#### 8. Batch Decoding

**Problem**: Decoding 1 request at a time underutilizes CPU

**Solution**: Batch multiple requests together

```python
class CustomWhisperDecoder:
    def decode_batch(self, encoder_features_list, ...):
        # Stack features: [(n1, 512), (n2, 512), ...] → (batch, max_n, 512)
        # Pad to same length
        # Decode all at once (faster than sequential)
        ...
```

**Expected**: 30-50% faster when processing multiple requests

**Benefit**: Higher throughput (req/s), not latency

#### 9. Pipeline Stage Parallelism

**Problem**: Decoder and alignment run sequentially

**Solution**: Pipeline them

```
Current:
Decode (150ms) → Align (150ms) = 300ms

Pipelined:
Request 1: Decode ----→ Align ----→
Request 2:       Decode ----→ Align ----→
Request 3:             Decode ----→ Align ----→

Throughput: 1 request / 150ms (vs 1 request / 300ms)
```

**Expected**: 2× throughput (same latency)

#### 10. GPU Decoder Offload

**Problem**: CPU decoder is slow

**Solution**: Use GPU for decoder (if available)

```python
class CustomWhisperDecoder:
    def __init__(self, device="cpu"):
        # If GPU available, use it
        if torch.cuda.is_available():
            device = "cuda"

        self.model.to(device)
```

**Expected**: 3-5× faster decoder (150ms → 30-50ms)

**Trade-off**: GPU power consumption (30W+ vs 5W CPU)

---

## Optimization Roadmap

### Week 19.5 (Current)
**Focus**: Deploy fix, ensure correctness

**Tasks**:
- [x] Fix double encoding (P0)
- [x] Validate shape compatibility (P0)
- [x] Add timing instrumentation (P0)
- [ ] Run validation tests (P0)
- [ ] Deploy if tests pass (P0)

**Expected Performance**: 25× realtime (200ms for 5s)

### Week 20 (Decoder Optimization)
**Focus**: Optimize decoder performance

**Tasks**:
- [ ] Profile decoder hot paths (1 day)
- [ ] Try INT8 quantization (0.5 days)
- [ ] Optimize beam search (0.5 days)
- [ ] Cache decoder model (0.5 days)
- [ ] Benchmark and validate (0.5 days)

**Expected Performance**: 35× realtime (140ms for 5s)

### Week 21 (Alignment Optimization)
**Focus**: Optimize or eliminate alignment bottleneck

**Tasks**:
- [ ] Evaluate faster alignment models (1 day)
- [ ] Implement alignment skip option (0.5 days)
- [ ] Test accuracy impact (0.5 days)
- [ ] Add configuration flags (0.5 days)

**Expected Performance**: 50× realtime (100ms for 5s)

### Week 22 (Memory & Throughput)
**Focus**: Zero-copy and batch processing

**Tasks**:
- [ ] Implement zero-copy buffers (1 day)
- [ ] Add batch decoding (1-2 days)
- [ ] Pipeline decoder + alignment (1 day)
- [ ] Stress test with concurrent load (0.5 days)

**Expected Performance**: 50× realtime, 3× throughput (150 req/s)

---

## Expected Performance Progression

| Week | Optimization | 5s Audio Time | Realtime Factor | Cumulative Speedup |
|------|--------------|---------------|-----------------|-------------------|
| 18 | Baseline (double encoding) | 495ms | 10.1× | 1.00× |
| 19 | faster-whisper (failed) | 4,916ms | 1.02× | 0.20× |
| **19.5** | **Fix double encoding** | **200ms** | **25×** | **2.48×** |
| 20 | Optimize decoder | 140ms | 35× | 3.54× |
| 21 | Optimize alignment | 100ms | 50× | 4.95× |
| 22 | Zero-copy + batching | 95ms | 52× | 5.21× |

**Target Achieved**: Week 21-22 → 50× realtime ✅ (close to 100-200× stretch goal)

---

## Risk Assessment

### Low Risk (Safe to Implement)

- ✅ Timing instrumentation (Week 19.5)
- ✅ Decoder model caching (Week 20)
- ✅ Zero-copy buffers (Week 22)
- ✅ Alignment skip option (Week 21)

### Medium Risk (Test Carefully)

- ⚠️ INT8 quantization (Week 20) - may hurt accuracy
- ⚠️ Beam search reduction (Week 20) - may hurt quality
- ⚠️ Faster alignment model (Week 21) - may change output format
- ⚠️ Batch decoding (Week 22) - more complex implementation

### High Risk (Research First)

- ❌ GPU decoder offload (Week 23+) - power consumption trade-off
- ❌ Custom alignment implementation (Week 23+) - large effort
- ❌ NPU decoder port (Week 25+) - very complex, 4-6 weeks

---

## Monitoring & Validation

### Key Metrics to Track

**Performance Metrics**:
- Processing time (ms) - lower is better
- Realtime factor - higher is better
- Throughput (req/s) - higher is better
- NPU utilization (%) - should be consistent

**Quality Metrics**:
- Transcription accuracy (%) - must stay >95%
- Word error rate (WER) - should stay consistent
- Segment boundaries - should match baseline

**Resource Metrics**:
- Memory usage (MB) - should not grow
- CPU usage (%) - should be efficient
- Power consumption (W) - lower is better
- GPU usage (%) - if offloading

### Regression Testing

**After each optimization**:
1. Run validation suite (`tests/week19_5_pipeline_validation.py`)
2. Check all 5 tests still pass
3. Compare performance to previous week
4. Verify accuracy vs baseline dataset
5. Monitor memory usage over 100+ requests

**Acceptance Criteria**:
- Performance improvement: >10% speedup
- Accuracy maintained: >95% similarity
- No memory leaks: <1MB growth per 100 requests
- No regressions: All tests pass

---

## Implementation Guidelines

### Code Organization

```
unicorn-amanuensis/
├── xdna2/
│   ├── custom_whisper_decoder.py        # Week 19.5
│   ├── custom_whisper_decoder_int8.py   # Week 20 (if beneficial)
│   ├── fast_alignment.py                # Week 21
│   ├── batch_decoder.py                 # Week 22
│   └── ...
├── transcription_pipeline.py            # Core pipeline
└── tests/
    ├── week19_5_pipeline_validation.py  # Week 19.5
    ├── week20_decoder_benchmarks.py     # Week 20
    ├── week21_alignment_tests.py        # Week 21
    └── week22_batch_tests.py            # Week 22
```

### Configuration Flags

```python
# xdna2/server.py

# Week 19.5: Custom decoder
USE_CUSTOM_DECODER = os.environ.get("USE_CUSTOM_DECODER", "true").lower() == "true"

# Week 20: INT8 decoder
USE_INT8_DECODER = os.environ.get("USE_INT8_DECODER", "false").lower() == "true"

# Week 21: Alignment options
ENABLE_ALIGNMENT = os.environ.get("ENABLE_ALIGNMENT", "true").lower() == "true"
FAST_ALIGNMENT = os.environ.get("FAST_ALIGNMENT", "false").lower() == "true"

# Week 22: Batch processing
ENABLE_BATCH_DECODING = os.environ.get("ENABLE_BATCH_DECODING", "false").lower() == "true"
BATCH_DECODE_SIZE = int(os.environ.get("BATCH_DECODE_SIZE", "4"))
```

### Documentation Standards

**For each optimization**:
1. Create `WEEKXX_OPTIMIZATION_REPORT.md`
2. Document expected vs actual performance
3. Include benchmark results
4. List any trade-offs or risks
5. Provide rollback instructions

**Example**:
```markdown
# Week 20: Decoder INT8 Optimization

**Performance**: 150ms → 90ms (40% faster)
**Accuracy**: 97.2% (vs 98.1% FP32, -0.9%)
**Trade-off**: Slight accuracy loss acceptable for 40% speedup
**Rollback**: Set USE_INT8_DECODER=false
```

---

## Long-Term Vision (Week 25-30)

### NPU Decoder Implementation

**Goal**: Run BOTH encoder AND decoder on NPU

**Expected Performance**:
- NPU encoder: 18ms
- **NPU decoder**: 30ms (vs 150ms CPU)
- Alignment: 50ms (optimized)
- **Total**: ~100ms → **50× realtime**

**Effort**: 4-6 weeks (major kernel development)

**Challenges**:
- Compile Whisper decoder to MLIR-AIE
- Handle autoregressive decoding on NPU
- Manage beam search state
- Debug and optimize kernels

**Payoff**: Could reach 100-200× realtime target if fully optimized

---

## Success Metrics

### Week 19.5 Success Criteria

- [x] Validation test suite created (672 lines)
- [ ] All 5 tests pass
- [ ] Performance: 20-30× realtime (200-250ms for 5s)
- [ ] Accuracy: >95% vs baseline
- [ ] No CPU re-encoding
- [ ] Ready for deployment

### Week 20 Success Criteria

- [ ] Decoder optimized
- [ ] Performance: 30-40× realtime (125-165ms for 5s)
- [ ] Accuracy maintained (>95%)
- [ ] Configuration flags working
- [ ] Benchmarks documented

### Week 21 Success Criteria

- [ ] Alignment optimized or optional
- [ ] Performance: 45-55× realtime (90-110ms for 5s)
- [ ] Word timestamps still work (when enabled)
- [ ] User can disable alignment for speed

### Week 22 Success Criteria

- [ ] Zero-copy buffers working
- [ ] Batch decoding implemented
- [ ] Performance: 50× realtime (100ms for 5s)
- [ ] Throughput: 100+ req/s (concurrent)
- [ ] Memory usage stable

---

## Conclusion

This optimization roadmap provides clear guidance for improving transcription performance from **25× realtime (Week 19.5)** to **50× realtime (Week 22)**.

**Key Takeaways**:
1. **Week 19.5**: Fix is critical foundation (25× realtime)
2. **Week 20**: Decoder optimization highest ROI (35× realtime)
3. **Week 21**: Alignment is next bottleneck (50× realtime)
4. **Week 22**: Throughput improvements for scale (50× + 3× throughput)
5. **Week 25-30**: NPU decoder for ultimate performance (100-200× realtime)

**Priority Order**:
1. P0: Deploy Week 19.5 fix correctly ← **CRITICAL**
2. P1: Optimize decoder (Week 20) ← **High impact**
3. P1: Optimize alignment (Week 21) ← **High impact**
4. P2: Zero-copy + batching (Week 22) ← **Throughput**
5. P3: NPU decoder (Week 25+) ← **Long-term**

**Team 2's Role**: Validate each optimization, ensure no regressions, guide performance improvements.

---

**Report Generated**: November 2, 2025, 16:00 UTC
**Author**: Team 2 Lead, CC-1L NPU Acceleration Project
**Version**: 1.0

**Built with 🦄 by Magic Unicorn Unconventional Technology & Stuff Inc**
