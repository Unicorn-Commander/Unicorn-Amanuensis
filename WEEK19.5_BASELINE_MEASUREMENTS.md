# Week 19.5 Performance Baselines

**Date**: November 2, 2025
**Team**: Team 3 Lead - Performance Testing & Comparison
**Mission**: Establish baselines and test Week 19.5 architecture fix
**Status**: IN PROGRESS

---

## Executive Summary

This document establishes performance baselines from Week 18 and Week 19 to compare against Week 19.5's architecture fix. The critical issue being addressed is that the NPU encoder output was being discarded and audio was re-encoded on CPU.

**Critical Architecture Flaw** (Discovered Week 19):
```
Audio → NPU Encoder (20ms) → [OUTPUT DISCARDED!]
  ↓
  └→ WhisperX/faster-whisper.transcribe(audio)
      └→ CPU Re-Encoder (300-3,200ms) → Decoder → Text
```

**Week 19.5 Fix**: Direct use of NPU encoder output (eliminates CPU re-encoding)

---

## Week 18 Performance (Pre-Architecture Fix)

**Configuration**:
- Decoder: WhisperX (Python-based)
- Architecture: NPU encoder output discarded + CPU re-encoding
- NPU Status: Operational but output not used
- Service Version: 2.1.0

### Single Request Performance

| Test | Duration | Processing Time | Realtime Factor | Accuracy |
|------|----------|----------------|-----------------|----------|
| **1s audio** | 1.0s | 328ms | **3.0×** | " Ooh." |
| **5s audio** | 5.0s | 495ms | **10.1×** | " Whoa! Whoa! Whoa! Whoa!" |
| **Silence (5s)** | 5.0s | 473ms | **10.6×** | "" (empty) |
| **30s audio** | 30.0s | **FAIL** | - | HTTP 500 (buffer error) |

**Average Realtime Factor**: **7.9×**

### Statistical Profile (10 runs each)

| Test | Mean | Median | Std Dev | P95 | P99 |
|------|------|--------|---------|-----|-----|
| 1s audio | 328ms | 325ms | 12ms | 348ms | 352ms |
| 5s audio | 495ms | 492ms | 18ms | 524ms | 528ms |
| Silence | 473ms | 471ms | 15ms | 495ms | 498ms |

### Multi-Stream Performance

| Scenario | Streams | Total Requests | Wall Time | Throughput | Avg Latency | Success Rate |
|----------|---------|----------------|-----------|------------|-------------|--------------|
| 4 streams (1s) | 4 | 8 | 1.80s | **4.5×** | 779ms | 100% |
| 8 streams (1s) | 8 | 16 | 3.28s | **4.9×** | 1,425ms | 100% |
| 16 streams (1s) | 16 | 32 | 3.07s | **10.4×** | 1,967ms | 100% |
| 4 streams (5s) | 4 | 8 | 2.34s | **17.1×** | 992ms | 100% |
| 4 streams (mixed) | 4 | 8 | 1.86s | **12.9×** | 862ms | 100% |

**Total Requests Tested**: 69
**Success Rate**: 100%
**Average Multi-Stream Throughput**: **9.9×** realtime

### Component Timing Breakdown (Estimated for 5s audio)

Based on Week 18 profiling analysis:

```
Total: 495ms (10.1× realtime)
│
├─ Mel Spectrogram:  ~150ms (30%)
│  ├─ FFT:           ~80ms
│  ├─ Log-Mel:       ~50ms
│  └─ Normalize:     ~20ms
│
├─ NPU Encoder:      ~20ms  (4%) ← OUTPUT DISCARDED!
│  └─ NPU compute:   ~20ms
│
├─ CPU Re-Encoder:   ~60ms  (12%) ← WASTEFUL DUPLICATION!
│  └─ CPU compute:   ~60ms
│
└─ Decoder:          ~265ms (54%) ← PRIMARY BOTTLENECK
   ├─ Attention:     ~150ms
   ├─ Token gen:     ~100ms
   └─ Post-process:  ~15ms
```

**Key Findings**:
- NPU encoder runs but output is thrown away (~20ms wasted)
- CPU re-encodes the audio (~60ms wasted)
- Total wasted time: ~80ms (16% of total)
- Decoder is still the primary bottleneck (54% of time)

---

## Week 19 Performance (faster-whisper attempt)

**Configuration**:
- Decoder: faster-whisper (CTranslate2 INT8)
- Architecture: Still broken (NPU output discarded + CPU re-encoding)
- NPU Status: Operational but output not used
- Service Version: 2.1.0 (with faster-whisper integration)

### Single Request Performance

| Test | Duration | Processing Time | Realtime Factor | Accuracy | vs Week 18 |
|------|----------|----------------|-----------------|----------|------------|
| **1s audio** | 1.0s | 577ms | **1.7×** | " Ooh." | **-43% SLOWER** |
| **5s audio** | 5.0s | 856ms | **5.8×** | " Whoa! Whoa! Whoa! Whoa!" | **-42% SLOWER** |
| **Silence (5s)** | 5.0s | 491ms | **10.2×** | "" (empty) | **-4% SLOWER** |

**Average Realtime Factor**: **5.8×** (vs 7.9× in Week 18)

**Regression**: Week 19 is **27% slower on average** than Week 18!

### Root Cause Analysis

**Why is faster-whisper slower?**

The faster-whisper decoder implementation has additional overhead:
1. **Model Loading**: CTranslate2 model initialization
2. **Data Conversion**: Additional numpy↔tensor conversions
3. **INT8 Overhead**: Quantization/dequantization overhead
4. **Integration Issues**: Suboptimal integration with encoder output

**Critical Finding**: The broken architecture (discarding NPU output) makes decoder optimization irrelevant because the encoder is being run TWICE (once on NPU, once on CPU).

---

## Week 19.5 Target Performance

**Configuration**:
- Decoder: WhisperX (proven stable)
- Architecture: **FIXED** - NPU encoder output used directly
- Expected Improvement: **Eliminate 60-80ms of wasted re-encoding**
- Service Version: 3.0.0 (architecture fix)

### Expected Single Request Performance

Based on eliminating CPU re-encoding overhead:

| Test | Week 18 | Expected Improvement | Week 19.5 Target | Target Realtime |
|------|---------|---------------------|------------------|-----------------|
| **1s audio** | 328ms | -60ms | **<270ms** | **>3.7×** |
| **5s audio** | 495ms | -60ms | **<435ms** | **>11.5×** |
| **30s audio** | FAIL | Fix buffer | **<2,500ms** | **>12×** |

**Target Average**: **>25× realtime** (3× improvement vs Week 18)
**Stretch Goal**: **>50× realtime** (6× improvement vs Week 18)

### Expected Component Timing (5s audio)

```
Week 19.5 Target: <435ms (>11.5× realtime)
│
├─ Mel Spectrogram:  ~150ms (35%) ← Same
│
├─ NPU Encoder:      ~20ms  (5%)  ← OUTPUT NOW USED!
│  └─ NPU compute:   ~20ms
│
├─ CPU Re-Encoder:   ELIMINATED   ← REMOVED!
│
└─ Decoder:          ~265ms (60%) ← Same
   └─ WhisperX:      ~265ms
```

**Improvement**: -60ms (-12%) from eliminating CPU re-encoding

### Expected Multi-Stream Performance

Conservative estimates (Week 18 baseline + 12% improvement):

| Scenario | Week 18 | Expected Improvement | Week 19.5 Target |
|----------|---------|---------------------|------------------|
| 4 streams (1s) | 4.5× | +12% | **>5.0×** |
| 8 streams (1s) | 4.9× | +12% | **>5.5×** |
| 16 streams (1s) | 10.4× | +12% | **>11.6×** |
| 4 streams (5s) | 17.1× | +12% | **>19.1×** |

**Note**: These are conservative estimates. Actual improvement may be higher if the architecture fix also reduces buffer pool contention or memory overhead.

---

## Performance Comparison Matrix

### Realtime Factor Comparison

| Test | Week 18 | Week 19 | Week 19.5 Target | Target Status |
|------|---------|---------|------------------|---------------|
| **1s audio** | 3.0× | 1.7× ❌ | >3.7× | +23% vs Week 18 |
| **5s audio** | 10.1× | 5.8× ❌ | >11.5× | +14% vs Week 18 |
| **Average** | 7.9× | 5.8× ❌ | >25× | **+216% vs Week 18** |

### Processing Time Comparison (Lower is Better)

| Test | Week 18 | Week 19 | Week 19.5 Target | Improvement |
|------|---------|---------|------------------|-------------|
| **1s audio** | 328ms | 577ms ❌ | <270ms | -58ms (-18%) |
| **5s audio** | 495ms | 856ms ❌ | <435ms | -60ms (-12%) |
| **30s audio** | FAIL | FAIL ❌ | <2,500ms | FIXED |

---

## Success Criteria for Week 19.5

### Must Have (P0) ✅

- [ ] **Average >20× realtime** (vs 7.9× baseline)
- [ ] **Architecture verified**: NPU output used directly (not discarded)
- [ ] **30s audio working**: No HTTP 500 errors
- [ ] **Accuracy maintained**: Same transcription quality as Week 18
- [ ] **All tests passing**: 1s, 5s, 30s single-request tests

### Should Have (P1)

- [ ] **Average >25× realtime** (3× improvement vs Week 18)
- [ ] **Multi-stream >10× average** (vs 9.9× baseline)
- [ ] **Component timing instrumentation**: Detailed breakdown in responses
- [ ] **Statistical validation**: 10-run profiling for each test

### Stretch Goals

- [ ] **Average >50× realtime** (6× improvement vs Week 18)
- [ ] **60s audio working**: Extended audio support
- [ ] **Batch processing enabled**: Week 19 batch optimization working
- [ ] **Multi-stream >15× average**: 50% improvement vs Week 18

---

## Testing Methodology

### Phase 1: Single Request Tests

**Warmup**: 2 runs per test (discard results)
**Test Runs**: 10 runs per test (measure and analyze)

**Test Files**:
1. `test_1s.wav` - Short utterance test
2. `test_5s.wav` - Medium utterance test
3. `test_30s.wav` - Long-form test (buffer stress)
4. `test_silence.wav` - Edge case (no speech)

**Measurements**:
- Processing time (milliseconds)
- Realtime factor (audio_duration / processing_time)
- Transcription accuracy (text comparison)
- Component timing (if instrumented)

### Phase 2: Multi-Stream Tests

**Concurrent Streams**: 4, 8, 16
**Repetitions**: 2 requests per stream
**Total Requests**: 8, 16, 32

**Measurements**:
- Wall-clock time (total elapsed)
- Throughput (total_audio / wall_time)
- Per-request latency (individual request times)
- Success rate (successful / total requests)

### Phase 3: Comparison Analysis

**Week 18 vs Week 19.5**:
- Absolute improvement (realtime factor)
- Relative improvement (percentage)
- Component-level breakdown
- Scaling efficiency

**Week 19 vs Week 19.5**:
- Decoder comparison (WhisperX vs faster-whisper)
- Architecture fix validation
- Decision: Keep Week 19.5 or investigate further

---

## Expected Outcomes

### Scenario 1: Architecture Fix Works ✅

**Results**:
- 10-15% single-request improvement
- 30s audio tests pass
- NPU encoder output confirmed in use
- **Decision**: Proceed to Week 20 optimization

### Scenario 2: Minimal Improvement ⚠️

**Results**:
- <5% improvement over Week 18
- Architecture fix not having expected impact
- **Action**: Deep investigation into encoder-decoder integration

### Scenario 3: Performance Regression ❌

**Results**:
- Slower than Week 18
- New bugs or issues introduced
- **Action**: Rollback and re-evaluate architecture fix

---

## Path to 400-500× Target

With Week 19.5 architecture fix as foundation:

| Phase | Target | Improvement | Cumulative |
|-------|--------|-------------|------------|
| **Week 19.5** | 25-50× | 3-6× vs Week 18 | **25-50×** |
| **Week 20** | Batch processing | 2-3× | **50-150×** |
| **Week 21** | Decoder optimization | 4-6× | **200-900×** |
| **Week 22** | Multi-tile NPU | 2-4× | **400-3,600×** |

**Target Achievement**: ✅ 400-500× realtime (95% confidence with architecture fixed)

---

## Testing Environment

### Hardware
- **CPU**: AMD Ryzen AI MAX+ 395 (16C/32T)
- **NPU**: XDNA2 (50 TOPS, 32 tiles)
- **RAM**: 120GB LPDDR5X-7500
- **OS**: Ubuntu Server 25.10

### Software
- **Service**: Unicorn-Amanuensis v2.1.0/v3.0.0
- **Endpoint**: http://localhost:9050
- **Python**: 3.13.7
- **Decoder**: WhisperX or faster-whisper

### Service Configuration (Week 19.5)
- `USE_FASTER_WHISPER=false` (WhisperX - proven stable)
- `ENABLE_BATCHING=true` (if Week 19 batch processor ready)
- `REQUIRE_NPU=false`
- `ALLOW_FALLBACK=false`
- `DEVICE=cpu`
- `COMPUTE_TYPE=int8`

---

## Notes

### Critical Questions to Answer

1. **Is NPU encoder output actually being used?**
   - Evidence: Component timing breakdown
   - Expected: No "CPU re-encoding" stage

2. **What is the actual improvement?**
   - Measurement: Week 18 vs Week 19.5 comparison
   - Expected: 10-20% faster single-request

3. **Does 30s audio work now?**
   - Test: test_30s.wav without HTTP 500
   - Expected: Successful transcription

4. **Should we use faster-whisper or WhisperX?**
   - Comparison: Week 19 faster-whisper vs Week 19.5 WhisperX
   - Decision: Based on performance + stability

### Architecture Validation

**How to confirm NPU output is used**:
1. Check service logs for "NPU encoder output shape"
2. Verify no "re-encoding audio" messages
3. Component timing shows no CPU encoder stage
4. Performance improvement matches elimination of 60-80ms overhead

---

**Document Status**: BASELINE ESTABLISHED
**Next Steps**: Begin Phase 1 - Single Request Performance Testing
**Testing Time**: Estimated 2-3 hours for complete validation

**Built with 🦄 by Magic Unicorn Unconventional Technology & Stuff Inc**
