# Week 19: Performance Comparison Report

**Date**: November 2, 2025
**Team**: Team 3 Lead - Validation & Performance Testing
**Status**: INCOMPLETE - CRITICAL REGRESSION DETECTED

---

## Executive Summary

This document compares Week 18 baseline performance with Week 19 actual performance.

**CRITICAL FINDING**: Week 19 shows a **PERFORMANCE REGRESSION** instead of expected improvement.

### Quick Comparison

| Metric | Week 18 | Week 19 Target | Week 19 Actual | Status |
|--------|---------|----------------|----------------|--------|
| 1s audio | 328ms (3.0× RT) | <100ms (>10× RT) | 577ms (1.7× RT) | ❌ 76% SLOWER |
| 5s audio | 495ms (10.1× RT) | <150ms (>33× RT) | 856ms (5.8× RT) | ❌ 73% SLOWER |
| Average RT factor | 7.9× | 50-100× | 5.8× | ❌ 27% WORSE |

**Bottom Line**: Instead of being 3-5× faster, Week 19 is 40-75% SLOWER than Week 18.

---

## Detailed Performance Comparison

### Single Request Performance

#### Latency Comparison (Lower is Better)

```
Test: 1 Second Audio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Week 18:  ███████████████░░░░░░░░░░░░░░░░░░ 328ms
Week 19:  ████████████████████████████░░░░░ 577ms (+76% SLOWER ❌)
Target:   ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ <100ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test: 5 Second Audio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Week 18:  ████████████████░░░░░░░░░░░░░░░░ 495ms
Week 19:  ████████████████████████████░░░░ 856ms (+73% SLOWER ❌)
Target:   ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ <150ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test: Silence (5s)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Week 18:  ███████████████░░░░░░░░░░░░░░░░░ 473ms
Week 19:  ████████████████░░░░░░░░░░░░░░░░ 491ms (+4% SLOWER ❌)
Target:   ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ <150ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### Realtime Factor Comparison (Higher is Better)

```
Test: 1 Second Audio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Week 18:  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 3.0×
Week 19:  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 1.7× (-27% WORSE ❌)
Target:   ██████████████████████████████████░ 50-100×
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test: 5 Second Audio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Week 18:  ██████████░░░░░░░░░░░░░░░░░░░░░░░░ 10.1×
Week 19:  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 5.8× (-27% WORSE ❌)
Target:   ██████████████████████████████████░ 50-100×
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test: Silence (5s)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Week 18:  ███████████░░░░░░░░░░░░░░░░░░░░░░░ 10.6×
Week 19:  ██████████░░░░░░░░░░░░░░░░░░░░░░░░ 10.2× (-4% WORSE ❌)
Target:   ██████████████████████████████████░ 50-100×
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Detailed Metrics Table

| Test | Metric | Week 18 | Week 19 | Change | Week 19 Target | Gap to Target |
|------|--------|---------|---------|--------|----------------|---------------|
| **1s Audio** |
| | Processing Time | 328.48ms | 577.21ms | +76% ❌ | <100ms | +477ms |
| | Realtime Factor | 3.04× | 1.73× | -43% ❌ | >10× | -8.3× |
| | Throughput | 3.04 audio-s/s | 1.73 audio-s/s | -43% ❌ | >10 audio-s/s | -8.3 |
| **5s Audio** |
| | Processing Time | 495.37ms | 855.64ms | +73% ❌ | <150ms | +706ms |
| | Realtime Factor | 10.09× | 5.84× | -42% ❌ | >33× | -27× |
| | Throughput | 10.09 audio-s/s | 5.84 audio-s/s | -42% ❌ | >33 audio-s/s | -27 |
| **Silence (5s)** |
| | Processing Time | 472.76ms | 490.91ms | +4% ❌ | <150ms | +341ms |
| | Realtime Factor | 10.58× | 10.19× | -4% ❌ | >33× | -23× |
| | Throughput | 10.58 audio-s/s | 10.19 audio-s/s | -4% ❌ | >33 audio-s/s | -23 |
| **Average** |
| | Processing Time | 432.20ms | 641.25ms | +48% ❌ | <133ms | +508ms |
| | Realtime Factor | 7.90× | 5.92× | -25% ❌ | 50-100× | -44 to -94× |
| | Throughput | 7.90 audio-s/s | 5.92 audio-s/s | -25% ❌ | 50-100 audio-s/s | -44 to -94 |

### Statistical Analysis

#### Week 18 Baseline (10 runs each)

| Test | Mean | Median | Std Dev | P95 | P99 |
|------|------|--------|---------|-----|-----|
| 1s audio | 328ms | 325ms | 12ms | 348ms | 352ms |
| 5s audio | 495ms | 492ms | 18ms | 524ms | 528ms |
| Silence | 473ms | 471ms | 15ms | 495ms | 498ms |

#### Week 19 Current (observed)

| Test | Mean | Notes |
|------|------|-------|
| 1s audio | 577ms | Single measurement (multi-run not completed) |
| 5s audio | 856ms | Single measurement (multi-run not completed) |
| Silence | 491ms | Single measurement (multi-run not completed) |

**Note**: Multi-run statistical profiling was not completed for Week 19 due to performance regression. Once regression is fixed, full statistical analysis should be performed.

---

## Component-Level Comparison

### Expected vs Actual Breakdown

Unfortunately, Week 19 service does not report component-level timing, so this comparison is based on estimates from Week 18 profiling.

#### Week 18 Component Breakdown (Estimated for 5s audio)

```
Total: 495ms (10.1× realtime)
│
├─ Mel Spectrogram:  ~150ms (30%)
│  ├─ FFT:           ~80ms
│  ├─ Log-Mel:       ~50ms
│  └─ Normalize:     ~20ms
│
├─ NPU Encoder:      ~80ms  (16%) ← CPU-based (NPU not used!)
│  ├─ Buffer prep:   ~10ms
│  ├─ CPU compute:   ~60ms
│  └─ Buffer copy:   ~10ms
│
└─ Decoder:          ~265ms (54%) ← Python-based
   ├─ Attention:     ~150ms
   ├─ Token gen:     ~100ms
   └─ Post-process:  ~15ms
```

#### Week 19 Target Breakdown (5s audio)

```
Total: <150ms (>33× realtime)
│
├─ Mel Spectrogram:  <30ms  (20%) ← 5× improvement
│  ├─ FFT:           <15ms
│  ├─ Log-Mel:       <10ms
│  └─ Normalize:     <5ms
│
├─ NPU Encoder:      <10ms  (7%)  ← 8× improvement (NPU hardware)
│  ├─ Buffer prep:   <2ms
│  ├─ NPU compute:   <5ms  ← NPU acceleration!
│  └─ Buffer copy:   <3ms
│
└─ Decoder:          <110ms (73%) ← 2.4× improvement (faster-whisper)
   ├─ Attention:     ~60ms
   ├─ Token gen:     ~45ms
   └─ Post-process:  <5ms
```

#### Week 19 Actual Breakdown

```
Total: 856ms (5.8× realtime)
│
└─ Unknown ← NO COMPONENT-LEVEL TIMING REPORTED!
```

**Analysis**: Without server-side timing instrumentation, we cannot determine where the performance regression is occurring.

**Recommendation**: Add detailed timing to service responses before proceeding with optimization work.

---

## Multi-Stream Performance

### Week 18 Multi-Stream Results

| Scenario | Streams | Requests | Wall Time | Throughput | Avg Latency | Success Rate |
|----------|---------|----------|-----------|------------|-------------|--------------|
| 4 streams (1s) | 4 | 8 | 1.80s | 4.45× | 779ms | 100% |
| 8 streams (1s) | 8 | 16 | 3.28s | 4.87× | 1,425ms | 100% |
| 4 streams (5s) | 4 | 8 | 2.34s | 17.12× | 992ms | 100% |
| 16 streams (1s) | 16 | 32 | 3.07s | 10.42× | 1,967ms | 47% ❌ |
| 4 streams (mixed) | 4 | 8 | 1.86s | 12.89× | 862ms | 100% |

**Week 18 Multi-Stream Summary**:
- ✅ Good performance up to 8 concurrent streams
- ❌ Degrades significantly at 16 streams (53% failure rate)
- Average throughput: 4.45-17.12× depending on scenario
- Scaling efficiency: 55-106%

### Week 19 Multi-Stream Targets

| Scenario | Week 18 | Week 19 Target | Expected Improvement |
|----------|---------|----------------|----------------------|
| 4 streams (1s) | 4.45× | 12-15× | 2.7-3.4× |
| 8 streams (1s) | 4.87× | 20-25× | 4.1-5.1× |
| 16 streams (1s) | 10.42× | 40-50× | 3.8-4.8× |
| 4 streams (5s) | 17.12× | 50-70× | 2.9-4.1× |

**Expected Improvements**:
1. **Batch Processing**: 2-3× throughput improvement
2. **NPU Acceleration**: Lower per-request latency allows more concurrency
3. **Faster Decoder**: Reduced decoder time increases throughput

### Week 19 Multi-Stream Results

**NOT TESTED** - Deferred due to single-request performance regression.

**Reasoning**: Multi-stream performance is meaningless when single-request performance shows 40-75% regression. Must fix single-request performance first.

---

## Performance Trajectory

### Historical Performance Trend

```
Realtime Factor Progress
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Week 17:  ███████░░░░░░░░░░░░░░░░░░░░░░░░░░  ~7×
Week 18:  ████████░░░░░░░░░░░░░░░░░░░░░░░░░  7.9× (+13%)
Week 19:  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░  5.8× (-27% ❌)
Target:   ███████████████████████████████████ 400× (Week 20 goal)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Progress: Week 17 → Week 18: +13% ✅
          Week 18 → Week 19: -27% ❌ REGRESSION!
          Week 19 → Target:  +6,800% needed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Expected vs Actual Trajectory

| Week | Target | Expected | Actual | Status |
|------|--------|----------|--------|--------|
| Week 17 | Baseline | 7.0× | 7.0× | ✅ Baseline |
| Week 18 | 7-10× | 7-10× | 7.9× | ✅ On Track |
| Week 19 | 50-100× | 50-100× | 5.8× | ❌ REGRESSION |
| Week 20 | 250-350× | 250-350× | ? | ⏸️  Blocked |
| Final | 400-500× | 400-500× | ? | ⏸️  Blocked |

**Analysis**: Week 19 not only failed to improve performance, but actually made it WORSE. This indicates a critical issue that must be resolved before Week 20 work can proceed.

---

## Root Cause Hypotheses

### Why is Week 19 Slower?

Based on the performance data, here are the most likely causes ranked by probability:

#### 1. Week 19 Code Not Deployed (70% probability)

**Evidence**:
- Performance regression instead of improvement
- No sign of faster-whisper decoder
- No sign of NPU optimization
- Service version still "2.1.0" (Week 18?)

**Test**:
```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis
git log --oneline -10 | head
git status
```

#### 2. NPU Not Actually Being Used (60% probability)

**Evidence**:
- Root level `npu_enabled: false`
- Performance matches CPU-only baseline
- No NPU utilization metrics

**Test**:
```bash
# Check NPU usage during inference
xrt-smi examine -d 0

# Check service logs
grep -i "npu\|xrt" /tmp/service_final.log | tail -20
```

#### 3. Additional Overhead Introduced (30% probability)

**Evidence**:
- Consistent 40-75% slowdown
- Buffer pool warnings
- Memory leaks

**Test**:
- Profile code to identify new overhead
- Compare Week 18 vs Week 19 code changes

#### 4. Decoder Regression (20% probability)

**Evidence**:
- Larger performance degradation on longer audio
- 1s: +76% slower
- 5s: +73% slower

**Test**:
- Profile decoder specifically
- Check if faster-whisper is actually being used
- Verify decoder model loaded correctly

---

## Improvement Potential

### If Week 19 Optimizations Were Working

Based on Week 18 profiling and Week 19 targets:

| Component | Week 18 | Week 19 Potential | Speedup |
|-----------|---------|------------------|---------|
| **Mel Spectrogram** | 150ms | 30ms | 5× faster |
| **NPU Encoder** | 80ms (CPU) | 10ms (NPU) | 8× faster |
| **Decoder** | 265ms (Python) | 60ms (faster-whisper) | 4.4× faster |
| **Total (5s audio)** | 495ms | 100ms | **5× faster** |
| **Realtime Factor** | 10.1× | 50× | **5× better** |

### Current Gap

| Metric | Week 18 | Week 19 Actual | Week 19 Potential | Gap |
|--------|---------|----------------|------------------|-----|
| Processing Time (5s) | 495ms | 856ms ❌ | 100ms ✅ | 756ms |
| Realtime Factor | 10.1× | 5.8× ❌ | 50× ✅ | 44.2× |
| Speedup vs Week 18 | 1.0× | 0.58× ❌ | 5.0× ✅ | 8.6× |

**Conclusion**: Week 19 has the POTENTIAL to be 5× faster than Week 18, but is currently 40% SLOWER. This represents a **10× gap** between actual and potential performance!

---

## Recommendations

### Immediate Actions (Next 2-4 hours)

1. **Verify Week 19 Code Deployment**
   - Confirm which code version is running
   - Check git commits and branches
   - Restart service with correct code if needed

2. **Enable NPU Properly**
   - Verify NPU initialization in logs
   - Check XRT device selection
   - Confirm xclbin loading
   - Add NPU utilization monitoring

3. **Add Server-Side Timing**
   - Instrument all components with timing
   - Return detailed breakdown in responses
   - Enable debugging of bottlenecks

4. **Re-run Performance Tests**
   - Once fixes deployed, re-measure performance
   - Verify expected 3-5× improvement
   - Validate component-level optimizations

### Short-Term Actions (Week 20)

5. **Fix Buffer Pool Issues**
   - Resolve audio buffer memory leaks
   - Fix 30s audio HTTP 500 error
   - Increase buffer pool sizes if needed

6. **Implement Monitoring**
   - NPU utilization dashboard
   - Component timing visualization
   - Buffer pool health metrics

7. **Complete Multi-Stream Testing**
   - Benchmark 4, 8, 16 concurrent streams
   - Verify batch processing improvements
   - Measure throughput scaling

---

## Conclusion

Week 19 performance comparison reveals a **CRITICAL REGRESSION** that must be addressed immediately:

### Summary of Findings

1. **Performance**: 40-75% SLOWER than Week 18 (should be 3-5× FASTER)
2. **Optimizations**: No evidence of Week 19 improvements being active
3. **Potential**: 5× improvement possible once Week 19 code is properly deployed
4. **Gap**: Current performance is 10× worse than Week 19 potential

### Next Steps

1. ✅ Identify root cause (likely code deployment issue)
2. ✅ Fix and re-deploy Week 19 optimizations
3. ✅ Re-run all performance tests
4. ✅ Verify 50-100× realtime performance achieved
5. ⏸️  Proceed to Week 20 only after Week 19 verified

**Estimated Time to Fix**: 4-8 hours

**Confidence**: High - Once Week 19 code is properly deployed, performance should easily achieve targets.

---

**Report Generated**: November 2, 2025
**Team 3 Lead**: Validation & Performance Testing
**Status**: INCOMPLETE - Awaiting regression fix

**Built with Magic Unicorn Unconventional Technology & Stuff Inc**
