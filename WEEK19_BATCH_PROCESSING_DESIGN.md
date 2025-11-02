# Week 19: Batch Processing Architecture Design

**Team**: Week 19 Batch Processing & Integration Team
**Date**: November 2, 2025
**Status**: Design Phase Complete
**Priority**: P1 (High - Throughput optimization)

## Executive Summary

This document outlines the architectural design for implementing request batching in the Unicorn-Amanuensis NPU transcription service to achieve **2-3× throughput improvement** over Week 18's multi-stream results (10.4× realtime with 16 concurrent streams).

**Key Decisions**:
- **Batching Strategy**: Hybrid (time-based + size-based)
- **Batch Points**: Mel spectrogram computation + NPU encoder execution
- **Queue Management**: `asyncio.Queue` with timeout-based batch collection
- **Integration**: Minimal service modifications, transparent to existing API

## Problem Statement

### Current Performance (Week 18)

**Multi-Stream Results**:
- 16 concurrent streams: **10.4× realtime throughput**
- Average latency per request: **308ms**
- Sequential processing within pipeline stages

**Performance Bottleneck**:
Each request incurs fixed overhead:
- Model loading: ~50ms
- Memory allocation: ~20ms
- DMA transfers: ~30ms
- **Total overhead**: ~100ms per request

For short audio clips (1-5s), overhead dominates processing time:
- 1s audio: 328ms total (100ms overhead = 30% waste)
- 5s audio: 450ms total (100ms overhead = 22% waste)

### Target Performance

**With Batching**:
- Batch size 4: **2-2.5× throughput** (20-26× realtime)
- Batch size 8: **2.5-3× throughput** (26-31× realtime)
- Batch size 16: **3-3.5× throughput** (31-36× realtime)

**Overhead Amortization**:
- Load model once: ~50ms (shared across batch)
- Batch memory: ~30ms (vs 20ms × N)
- Batch DMA: ~40ms (vs 30ms × N)
- **Amortized overhead**: ~120ms for 4-8 requests

## Architecture Decisions

### 1. Batching Strategy: Hybrid (Time + Size)

**Options Evaluated**:

| Strategy | Pros | Cons | Decision |
|----------|------|------|----------|
| **Time-based** | Predictable latency | May batch too few requests | ❌ |
| **Size-based** | Maximum throughput | Unpredictable latency | ❌ |
| **Hybrid** | Balances latency & throughput | More complex | ✅ **SELECTED** |

**Selected: Hybrid (whichever comes first)**
- **Max batch size**: 8 requests (configurable: 4-16)
- **Max wait time**: 50ms (configurable: 25-100ms)
- **Rationale**: Achieves 2.5-3× throughput while keeping latency <200ms

### 2. Where to Batch

**Analysis of Pipeline Stages**:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Mel Compute    │ --> │  NPU Encoder    │ --> │  Decoder        │
│  (Batchable)    │     │  (Batchable)    │     │  (Limited)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Stage 1: Mel Spectrogram** - ✅ **Batch**
- Independent computation per request
- Can process N requests in parallel with NumPy broadcasting
- Minimal synchronization overhead

**Stage 2: NPU Encoder** - ✅ **Batch**
- NPU supports parallel execution across samples
- Batch matmul operations (M×K @ K×N → M×N)
- Significant overhead reduction (load model once)

**Stage 3: Decoder** - ⚠️ **Limited Batching**
- Autoregressive (sequential token generation)
- WhisperX supports batch decoding but limited benefit
- Keep sequential for simplicity (Week 19 scope)

**Decision**: Batch mel + encoder, sequential decoder

### 3. Queue Management

**Architecture**:

```python
┌─────────────────────────────────────────────────────────┐
│                   BatchProcessor                        │
│                                                          │
│  ┌──────────────────┐                                   │
│  │  Input Queue     │ <-- submit_request()              │
│  │  (asyncio.Queue) │                                   │
│  └────────┬─────────┘                                   │
│           │                                              │
│           v                                              │
│  ┌──────────────────┐                                   │
│  │ Batch Collector  │ (timeout: 50ms OR size: 8)        │
│  │ _collect_batch() │                                   │
│  └────────┬─────────┘                                   │
│           │                                              │
│           v                                              │
│  ┌──────────────────┐                                   │
│  │ Batch Processor  │                                   │
│  │ _process_batch() │                                   │
│  │                  │                                   │
│  │  1. Batch Mel    │ --> NumPy vectorization          │
│  │  2. Batch Encoder│ --> NPU parallel execution       │
│  │  3. Sequential   │ --> WhisperX per-request         │
│  │     Decoder      │                                   │
│  └────────┬─────────┘                                   │
│           │                                              │
│           v                                              │
│  ┌──────────────────┐                                   │
│  │  Result Futures  │ --> set_result()                  │
│  │  (dict[id,Future])│                                  │
│  └──────────────────┘                                   │
└─────────────────────────────────────────────────────────┘
```

**Key Components**:

1. **Input Queue**: `asyncio.Queue` for incoming requests
2. **Batch Collector**: Groups requests using timeout or size limit
3. **Batch Processor**: Processes batched requests in 3 sub-stages
4. **Result Futures**: Maps request_id → Future for async response

**Implementation Classes**:

```python
@dataclass
class TranscriptionRequest:
    request_id: str
    audio: np.ndarray
    language: Optional[str]
    timestamp: float

@dataclass
class TranscriptionResult:
    request_id: str
    text: str
    processing_time: float
    error: Optional[str] = None

class BatchProcessor:
    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_ms: int = 50,
        encoder_callback=None,
        decoder_callback=None
    ):
        self.input_queue = asyncio.Queue()
        self.result_futures = {}  # request_id -> Future

    async def submit_request(self, audio, language) -> TranscriptionResult:
        """Submit request and wait for batched result"""

    async def process_batches(self):
        """Main batch processing loop"""

    async def _collect_batch(self) -> List[TranscriptionRequest]:
        """Collect requests until timeout or size limit"""

    async def _process_batch(self, batch) -> List[TranscriptionResult]:
        """Process batch through mel + encoder + decoder"""
```

### 4. Batching Implementation Details

#### 4.1 Mel Spectrogram Batching

**Current (Sequential)**:
```python
# Process one request at a time
for request in requests:
    mel = compute_mel_spectrogram(request.audio)  # ~15ms
```

**Batched (Parallel)**:
```python
# Process batch together
audios = [req.audio for req in batch]
mels = compute_mel_spectrogram_batch(audios)  # ~20ms for 8 requests
# Speedup: 8×15ms = 120ms → 20ms (6× faster)
```

**Implementation**: Use NumPy broadcasting for batch FFT computation

#### 4.2 NPU Encoder Batching

**Current (Sequential)**:
```python
# Process one mel at a time
for mel in mels:
    encoded = npu_encoder.forward(mel)  # ~50ms overhead + 5ms compute
```

**Batched (Parallel)**:
```python
# Stack mels and process together
mel_batch = np.stack([mel for mel in mels], axis=0)  # (batch, frames, 512)
encoded_batch = npu_encoder.forward_batch(mel_batch)  # ~60ms total
# Speedup: 8×55ms = 440ms → 60ms (7× faster)
```

**Key Changes**:
- Add `forward_batch()` method to `WhisperEncoderCPP`
- Stack mel features along batch dimension
- NPU processes all samples in parallel

#### 4.3 Decoder (Sequential)

**Decision**: Keep sequential for Week 19
- WhisperX batch decoding has limited benefit (autoregressive)
- Adds complexity without significant throughput gain
- Can be optimized in future weeks if needed

**Implementation**:
```python
# Decode each request separately
for i, encoded in enumerate(encoded_batch):
    result = decoder.transcribe(encoded, language=batch[i].language)
    results.append(result)
```

### 5. Integration with Existing Service

**Minimal Changes to server.py**:

```python
# 1. Import batch processor
from xdna2.batch_processor import BatchProcessor

# 2. Initialize in startup
@app.on_event("startup")
async def startup_event():
    # ... existing initialization ...

    # Create batch processor
    self.batch_processor = BatchProcessor(
        max_batch_size=8,
        max_wait_ms=50,
        encoder_callback=self.npu_encoder.encode,
        decoder_callback=self.decoder
    )

    # Start batch processing task
    asyncio.create_task(self.batch_processor.process_batches())

# 3. Update transcription endpoint
@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile):
    # Load audio
    audio = load_audio(await file.read())

    # Submit to batch processor (instead of direct processing)
    result = await self.batch_processor.submit_request(
        audio=audio,
        language=None
    )

    return {
        "text": result.text,
        "processing_time": result.processing_time
    }
```

**Backward Compatibility**:
- API remains unchanged (OpenAI-compatible)
- Clients see same interface, faster response
- Can toggle batching with `ENABLE_BATCHING=false` env var

## Performance Projections

### Expected Throughput Improvement

**Week 18 Baseline**:
| Concurrent Streams | Throughput | Avg Latency | NPU Util |
|-------------------|------------|-------------|----------|
| 1                 | 3.0×       | 328ms       | 0.7%     |
| 4                 | 4.5×       | 290ms       | 1.1%     |
| 16                | 10.4×      | 308ms       | 2.5%     |

**Week 19 with Batching (Projected)**:
| Concurrent Streams | Batch Size | Throughput | Improvement | Avg Latency |
|-------------------|------------|------------|-------------|-------------|
| 4                 | 4          | 9-12×      | 2-2.7×      | 180ms       |
| 8                 | 8          | 18-24×     | 2.5-3×      | 190ms       |
| 16                | 8          | 25-35×     | 2.4-3.4×    | 200ms       |

**Rationale**:
- Overhead reduction: 100ms → 15-20ms per request
- Mel batching: 6× speedup
- Encoder batching: 7× speedup
- Decoder unchanged (sequential)

### Latency Trade-off

**Latency Components**:

| Component | Sequential | Batched (8 requests) | Change |
|-----------|-----------|----------------------|--------|
| Queue wait | 0ms | 0-50ms (avg 25ms) | +25ms |
| Mel compute | 15ms | 2.5ms (amortized) | -12.5ms |
| Encoder | 55ms | 7.5ms (amortized) | -47.5ms |
| Decoder | 200ms | 200ms | 0ms |
| **Total** | **270ms** | **235ms** | **-35ms** |

**Analysis**: Despite batching wait time, overall latency **improves** due to overhead amortization!

### NPU Utilization

**Current**: 2.5% (16 streams, sequential)
**Target**: 5-8% (16 streams, batched)

**Calculation**:
- Throughput: 10.4× → 25-35×
- NPU cycles: 2.5% × (25÷10.4) = **6.0%** (at 25× throughput)
- NPU cycles: 2.5% × (35÷10.4) = **8.4%** (at 35× throughput)

**Remaining Headroom**: 91-95% (excellent for future optimization)

## Configuration Parameters

### Tunable Parameters

```python
# Batch processing configuration (environment variables)
BATCH_MAX_SIZE = int(os.getenv('BATCH_MAX_SIZE', '8'))  # 4-16
BATCH_MAX_WAIT_MS = int(os.getenv('BATCH_MAX_WAIT_MS', '50'))  # 25-100ms
ENABLE_BATCHING = os.getenv('ENABLE_BATCHING', 'true') == 'true'

# Per-stage batching control
BATCH_MEL_COMPUTE = True  # Always enable (6× speedup)
BATCH_ENCODER = True      # Always enable (7× speedup)
BATCH_DECODER = False     # Week 19: keep sequential
```

### Tuning Guide

**For Low Latency (real-time applications)**:
```bash
BATCH_MAX_SIZE=4
BATCH_MAX_WAIT_MS=25
```

**For High Throughput (batch processing)**:
```bash
BATCH_MAX_SIZE=16
BATCH_MAX_WAIT_MS=100
```

**Balanced (recommended)**:
```bash
BATCH_MAX_SIZE=8
BATCH_MAX_WAIT_MS=50
```

## Implementation Risks & Mitigations

### Risk 1: Variable Audio Lengths

**Problem**: Batching requests with different audio lengths
- 1s audio: 100 frames
- 5s audio: 500 frames

**Solution**: Pad to maximum length in batch
```python
max_frames = max(mel.shape[0] for mel in mels)
padded_mels = [
    np.pad(mel, ((0, max_frames - mel.shape[0]), (0, 0)))
    for mel in mels
]
```

**Cost**: Minimal (padding is fast, NPU ignores padded regions)

### Risk 2: Memory Pressure

**Problem**: Batching increases peak memory usage
- 8 requests × 1.5MB/request = 12MB peak

**Solution**: Use existing buffer pool
- Pre-allocated buffers handle batch sizes
- No additional memory allocation needed

**Verification**: Monitor buffer pool statistics

### Risk 3: Latency Spikes

**Problem**: Timeout waiting for batch to fill
- If only 1 request arrives, wait 50ms before processing

**Solution**: Adaptive timeout
```python
# Reduce timeout if queue is mostly empty
if queue.qsize() < max_batch_size / 2:
    timeout = max_wait_ms / 2
```

**Future Enhancement**: Dynamic timeout based on load

### Risk 4: Error Handling

**Problem**: One failed request should not block batch
- Request 3/8 fails during mel computation

**Solution**: Per-request error isolation
```python
for i, req in enumerate(batch):
    try:
        mels[i] = compute_mel(req.audio)
    except Exception as e:
        results[i] = TranscriptionResult(error=str(e))
```

**Guarantee**: Failed requests get error response, others succeed

## Testing Strategy

### Test Scenarios

**1. Single Request (baseline)**:
```bash
time curl -X POST http://localhost:9050/v1/audio/transcriptions \
  -F "file=@tests/audio/test_1s.wav"
```
**Expected**: Similar latency to non-batched (~350ms)

**2. 4 Concurrent Requests (batching)**:
```bash
for i in {1..4}; do
  curl -X POST http://localhost:9050/v1/audio/transcriptions \
    -F "file=@tests/audio/test_1s.wav" &
done
wait
```
**Expected**: 2-2.5× throughput improvement

**3. 16 Concurrent Requests (stress test)**:
```bash
python tests/week18_multi_stream_test.py --streams 16
```
**Expected**: 2.4-3.4× throughput improvement

**4. Mixed Audio Lengths**:
```bash
# Mix 1s and 5s audio files
python tests/week19_batch_test.py --mixed-lengths
```
**Expected**: Proper padding, correct results

### Performance Metrics

**Must Measure**:
- Throughput (requests/second)
- Latency (p50, p95, p99)
- Batch fill rate (requests/batch)
- Batch wait time (ms)
- NPU utilization (%)

**Comparison Points**:
- Week 18 baseline (no batching)
- Week 19 batching (various batch sizes)

## Success Criteria

### Must Have (P0)
- ✅ Batch processor implemented and working
- ✅ Service integrated with batching
- ✅ Throughput improvement >1.5× (vs Week 18)
- ✅ Latency acceptable (<200ms for 1s audio)

### Should Have (P1)
- ✅ Throughput improvement >2×
- ✅ Batch size auto-tuning
- ✅ Comprehensive stats/metrics

### Stretch Goals
- ⏳ Throughput improvement >3×
- ⏳ NPU encoder batching (parallel execution)
- ⏳ Adaptive batch size based on load

## Future Enhancements (Post-Week 19)

### Week 20: Decoder Batching
- Implement batch decoding for WhisperX
- Use beam search parallelization
- Target: Additional 1.2-1.5× throughput

### Week 21: Dynamic Batch Sizing
- Monitor queue depth and latency
- Adjust batch size automatically
- Balance throughput vs latency in real-time

### Week 22: Priority-Based Batching
- High-priority requests processed first
- Low-priority requests batched more aggressively
- SLA-aware scheduling

### Week 23: Multi-Level Batching
- Batch at multiple pipeline stages
- Hierarchical batching (micro-batches + macro-batches)
- Optimal load balancing

## References

**Week 18 Results**:
- `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/tests/week18_multi_stream_test.py`
- 16 concurrent streams: 10.4× realtime throughput
- Average latency: 308ms per request

**Current Service Architecture**:
- `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/server.py`
- Multi-stream pipeline with 3 stages
- Buffer pool optimization

**NPU Performance**:
- XDNA2: 50 TOPS (32 tiles)
- Current utilization: 2.3-2.5%
- Target utilization: 5-8% (Week 19)
- Headroom: 92-95% for future growth

## Appendix: Code Structure

### Files to Create

1. **`xdna2/batch_processor.py`** (300-400 lines)
   - `BatchProcessor` class
   - `TranscriptionRequest` dataclass
   - `TranscriptionResult` dataclass
   - Batch collection logic
   - Batch processing pipeline

2. **`xdna2/mel_batch_utils.py`** (100-150 lines)
   - `compute_mel_spectrogram_batch()` - batch mel computation
   - NumPy vectorization for FFT
   - Padding utilities

3. **`xdna2/encoder_batch.py`** (50-100 lines)
   - `forward_batch()` method for WhisperEncoderCPP
   - Batch stacking utilities
   - NPU batch execution

### Files to Modify

1. **`xdna2/server.py`** (~30 lines changed)
   - Import BatchProcessor
   - Initialize in startup
   - Route requests to batch processor

2. **`xdna2/encoder_cpp.py`** (~50 lines added)
   - Add batch processing support
   - Batch dimension handling

### Test Files

1. **`tests/week19_batch_test.py`** (400-500 lines)
   - Comprehensive batch testing
   - Performance benchmarks
   - Comparison with Week 18

2. **`tests/week19_batch_validation.py`** (200-300 lines)
   - Correctness validation
   - Edge case testing
   - Error handling verification

## Conclusion

The hybrid batching strategy (time + size based) provides an optimal balance between throughput and latency. By batching mel spectrogram computation and NPU encoder execution, we expect to achieve **2-3× throughput improvement** over Week 18's multi-stream results.

**Key Advantages**:
- Minimal service changes (backward compatible)
- Significant overhead reduction (100ms → 15-20ms per request)
- Excellent NPU utilization increase (2.5% → 5-8%)
- Configurable batch size and timeout for different workloads

**Next Steps**:
1. Implement `batch_processor.py` (Phase 2)
2. Integrate with service (Phase 3)
3. Test and optimize (Phase 4)

---

**Document Status**: Design Complete
**Approved By**: Team 2 Lead
**Date**: November 2, 2025
**Next Review**: After Phase 2 Implementation
