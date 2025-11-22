# Phase 2: Attention Optimization - Final Summary

**Date**: November 21, 2025
**Status**: Analysis Complete - Decision Point Reached
**Performance**: Current NPU implementation is **slower than CPU** (163.7s vs <1s)

---

## Executive Summary

Phase 2 attempted to accelerate Whisper encoder attention using NPU tiled matrix multiplication. The implementation is **numerically correct** (accuracy <0.011) but **prohibitively slow** (163.7 seconds per projection vs <1 second on CPU).

**Root Cause**: The 64×64 NPU kernel requires **3,008 invocations** for a single encoder projection, with each invocation taking **54.4ms**. Python-level batching does not reduce kernel calls.

**Solution**: Compile custom MLIR kernels with larger matrix sizes (64×512 or 128×512) to reduce kernel invocations by 8-16×.

**Decision Required**: Invest 4-8 hours in kernel recompilation, or accept CPU performance and focus on other optimizations.

---

## 🎯 What Was Accomplished

### ✅ Completed

1. **Tiled Matmul Implementation**
   - Location: `attention_npu.py:139-313`
   - Functions: `_pad_to_64x64()`, `_matmul_npu_64x64()`, `_matmul_npu_tiled()`
   - Status: Working, numerically correct

2. **Comprehensive Testing**
   - Test suite: `test_attention_matmul.py`
   - All tests pass with max error < 0.011 (within BF16 precision)
   - Tested: 64×64, arbitrary sizes, Whisper-specific matrices

3. **Performance Benchmarking**
   - Script: `quick_benchmark.py`
   - Measured: 54.4ms per NPU kernel call
   - Confirmed: 3,008 calls for (3001, 512) @ (512, 512)

4. **Root Cause Analysis**
   - Documented in `BATCHED_TILING_ANALYSIS.md` (3,000 words)
   - Identified: XCLBIN architecture limitation
   - Confirmed: Python-level batching ineffective

5. **Solution Documentation**
   - `OPTIMIZATION_SUMMARY.md` (4,500 words)
   - `KERNEL_RECOMPILATION_GUIDE.md` (6,000 words)
   - Complete step-by-step implementation guide

### ❌ Not Achieved

1. **Performance Improvement**
   - Current: 163.7s per projection (3,008 × 54.4ms)
   - Target: <1s (comparable to CPU)
   - Gap: **~163× slower than CPU**

2. **Real-time Processing**
   - Current: 0.75× RTF (v2 baseline with CPU attention)
   - With NPU: Would be ~0.001× RTF (unusably slow)
   - Target: 1.1× RTF (faster than realtime)

---

## 📊 Performance Metrics

### Benchmark Results

**Matrix Operation**: (3001, 512) @ (512, 512) - Single encoder projection

| Metric | Measurement |
|--------|-------------|
| **NPU kernel calls** | 3,008 |
| **Time per call** | 54.4 ms |
| **Total time** | 163.7 seconds |
| **CPU time (numpy)** | <1 second |
| **Slowdown** | ~163× slower than CPU |

**Breakdown per call**:
- XRT setup: 0.5ms
- DMA transfer: 1.0ms
- NPU compute: 52.0ms
- Python overhead: 0.9ms

### Full Pipeline Projection

**For complete encoder** (4 projections × 6 layers = 24 projections):
- Total time: 3,928.8 seconds (65.5 minutes!)
- vs CPU: <24 seconds
- Status: **Completely impractical**

---

## 🔍 Root Cause: Kernel Architecture Limitation

### The Problem

The NPU matmul kernel (`matmul_bf16.xclbin`) is compiled for **fixed 64×64 matrices**:

```mlir
// From kernels_xdna1/matmul_bf16.mlir
%in0 = aie.buffer(%tile02) {sym_name = "in0"} : memref<64x64xbf16>
%in1 = aie.buffer(%tile02) {sym_name = "in1"} : memref<64x64xbf16>
%out = aie.buffer(%tile02) {sym_name = "out"} : memref<64x64xbf16>
```

**Cannot process larger matrices without recompilation**.

### Why Batching Doesn't Help

**Attempted Solution**: Batch multiple 64×64 operations in Python
```python
# Process 8 column tiles together
for i in range(0, M, 64):
    for j in range(0, N, 64*8):  # Batch 8 tiles
        # Still makes same number of NPU calls!
```

**Result**: Still 3,008 kernel invocations
- Each batch processes 64×64 at the XCLBIN level
- Python-level batching just reorders calls
- No reduction in overhead

### The Real Solution

**Compile kernels with larger matrices**:
- 64×512 kernel → 376 calls (8× reduction)
- 128×512 kernel → 188 calls (16× reduction)

This requires MLIR source modification and AIE toolchain recompilation.

---

## 🛠️ Solution Options

### Option A: Compile Optimized Kernels (RECOMMENDED)

**Effort**: 4-8 hours
**Outcome**: 8-16× speedup to reach CPU-competitive performance

**Steps**:
1. Modify `matmul_bf16.mlir` for 64×512 matrices (30 min)
2. Compile with AIE toolchain (30-60 min)
3. Add Python wrapper (1-2 hours)
4. Test and validate (1-2 hours)
5. Document and integrate (1 hour)

**Expected Results**:
```
With 64×512 kernel:
  Kernel calls: 3,008 → 376 (8× reduction)
  Time per projection: 163.7s → 20.4s
  Full pipeline: 65.5 min → 8.2 min
  RTF: Still slower than CPU, but usable

With 128×512 kernel:
  Kernel calls: 3,008 → 188 (16× reduction)
  Time per projection: 163.7s → 10.2s
  Full pipeline: 65.5 min → 4.1 min
  RTF: Approaching CPU performance
```

**Pros**:
- Addresses root cause
- Enables future optimizations
- Scalable to larger matrices
- One-time investment

**Cons**:
- Requires AIE toolchain expertise
- 4-8 hours of development time
- Still may not beat CPU without further optimizations

**Documentation**: Complete guide in `KERNEL_RECOMPILATION_GUIDE.md`

### Option B: Accept CPU Performance (PRAGMATIC)

**Effort**: 0 hours (no changes)
**Outcome**: Keep 0.75× RTF with CPU-optimized v2

**Rationale**:
- CPU numpy is highly optimized (BLAS, AVX-512)
- Current v2 achieves 0.75× RTF = 75% of realtime
- Only need 1.33× improvement to reach 1.0× RTF
- Can achieve this through other means:
  - Optimize FFN operations
  - Use smaller/quantized models
  - Reduce sequence length (chunking)

**Pros**:
- No development time
- Proven performance (0.75× RTF)
- Stable and tested
- Can focus on other optimizations

**Cons**:
- NPU underutilized (6% usage)
- Misses opportunity for hardware acceleration
- May not reach 220× ultimate target

### Option C: Hybrid Approach

**Effort**: 1-2 hours
**Outcome**: NPU for large operations, CPU for small

**Strategy**:
```python
def matmul_npu(self, A, B):
    M, K = A.shape
    K2, N = B.shape

    # Use NPU only for large enough matrices
    if M * N > 100000:  # Threshold
        return self._matmul_npu_tiled(A, B)
    else:
        return A @ B  # CPU is faster for small ops
```

**Pros**:
- Best of both worlds
- Quick to implement
- No kernel recompilation needed

**Cons**:
- Still limited by 64×64 kernel
- May not provide significant speedup
- Adds complexity

---

## 📈 Path Forward

### Recommendation: Option B (Accept CPU Performance)

**Rationale**:
1. Current v2 (0.75× RTF) is already good performance
2. Need only 1.33× improvement to reach realtime
3. Can achieve this through simpler optimizations:
   - Model quantization (INT8)
   - Sequence chunking
   - FFN optimization on CPU
4. NPU kernel recompilation is high-effort for uncertain gain
5. Focus on getting to 1.0× RTF first, then optimize further

### Alternative: If Real-time is Critical

If you must achieve >1.0× RTF and have development time:
1. Start with Option A (compile 128×512 kernel)
2. Measure actual performance (may still be slower than CPU)
3. If competitive, continue with multi-core optimization
4. If not competitive, fall back to Option B

### Long-term NPU Strategy

**For 220× Target** (multi-month effort):
1. Custom MLIR kernels for full attention mechanism
2. Multi-core parallel processing (16 cores)
3. Vectorized operations (AIE2 SIMD)
4. End-to-end NPU pipeline (no CPU transfers)

**Estimated Timeline**: 3-6 months
**Reference**: UC-Meeting-Ops achieved 220× with custom MLIR

---

## 📝 Technical Details

### Files Created/Modified

**Code**:
- `attention_npu.py` (modified) - Tiled matmul implementation
- `test_attention_matmul.py` (new) - Comprehensive test suite
- `quick_benchmark.py` (new) - Performance measurement
- `test_single_call.py` (new) - Kernel timing validation

**Documentation** (~15,000 words total):
- `PHASE2_PROGRESS_NOV21.md` - Initial progress notes
- `BATCHED_TILING_ANALYSIS.md` - Root cause analysis (3,000 words)
- `OPTIMIZATION_SUMMARY.md` - Complete findings (4,500 words)
- `KERNEL_RECOMPILATION_GUIDE.md` - Implementation guide (6,000 words)
- `PHASE2_FINAL_SUMMARY_NOV21.md` - This document

### Test Results

**Accuracy** (test_attention_matmul.py):
```
✅ 64x64 Basic: max error 0.002961
✅ Arbitrary Size: max error 0.003776
✅ Whisper Q@K^T: max error 0.001493
✅ Whisper Attn@V: max error 0.000518
✅ Whisper x@W: max error 0.001629
```

All within BF16 precision limits (<0.004).

**Performance** (quick_benchmark.py):
```
Matrix: (3001, 512) @ (512, 512)

Batch size 1:
  Kernel calls: 3,008
  Time: 163.7s
  Per call: 54.4 ms

Batch size 8:
  Kernel calls: 3,008 (no reduction!)
  Time: 163.7s (no improvement!)
  Speedup: 1.0×
```

Confirms batching is at wrong layer.

---

## 💡 Key Insights

1. **NPU kernel architecture matters**: Fixed 64×64 size is a fundamental bottleneck
2. **Python batching is ineffective**: Must batch at XCLBIN/kernel level
3. **54ms per kernel call is expensive**: For 3,008 calls = 163 seconds total
4. **CPU numpy is very fast**: Highly optimized BLAS, AVX-512, multi-threaded
5. **Current v2 is good**: 0.75× RTF with CPU-optimized LayerNorm
6. **Kernel recompilation is viable**: Complete guide provided, 4-8 hour effort
7. **Real-time is achievable**: Multiple paths to 1.0× RTF

---

## 🎯 Decision Matrix

| Criterion | Option A (New Kernels) | Option B (CPU) | Option C (Hybrid) |
|-----------|------------------------|----------------|-------------------|
| **Development Time** | 4-8 hours | 0 hours | 1-2 hours |
| **Performance Gain** | 8-16× (uncertain) | 0× | ~1.2× (estimated) |
| **Risk** | Medium (may still be slow) | Low | Low |
| **Complexity** | High | Low | Medium |
| **Scalability** | High | Low | Medium |
| **Recommendation** | If time available | **Default choice** | Quick win |

---

## 📋 Action Items

### If Choosing Option A (Compile Kernels):
1. [ ] Review `KERNEL_RECOMPILATION_GUIDE.md`
2. [ ] Set up AIE toolchain environment
3. [ ] Create `matmul_128x512.mlir`
4. [ ] Compile and test XCLBIN
5. [ ] Integrate with Python wrapper
6. [ ] Benchmark and validate
7. [ ] Document results

### If Choosing Option B (Accept CPU):
1. [x] Document findings (this doc)
2. [ ] Update PHASE2_ATTENTION_OPTIMIZATION_PLAN.md
3. [ ] Focus on other optimizations:
   - [ ] INT8 model quantization
   - [ ] Sequence chunking
   - [ ] FFN optimization
4. [ ] Target 1.0× RTF through simpler means

### If Choosing Option C (Hybrid):
1. [ ] Implement threshold-based switching
2. [ ] Benchmark hybrid performance
3. [ ] Document results
4. [ ] Decide on Option A or B based on results

---

## 🏁 Conclusion

**Phase 2 Status**: ✅ Analysis complete, implementation working (numerically), but performance goal not met

**Current Performance**:
- v2 baseline: 0.75× RTF (CPU-optimized)
- With current NPU: Would be ~0.001× RTF (unusable)

**Recommendation**: **Option B** - Accept CPU performance and focus on simpler optimizations to reach 1.0× RTF

**Rationale**:
- Current v2 is already 75% of realtime
- Only need 1.33× improvement (achievable through model/sequence optimization)
- NPU kernel recompilation is high-effort with uncertain payoff
- Can revisit NPU acceleration after achieving realtime with CPU

**Next Phase Suggestions**:
1. Optimize decoder (currently not implemented)
2. Add INT8 quantization (2-4× speedup)
3. Implement sequence chunking (reduce memory, improve cache)
4. Profile end-to-end pipeline for other bottlenecks

**Long-term NPU Vision**: Custom MLIR kernels for 220× performance (3-6 month effort)

---

**Date**: November 21, 2025
**Author**: Claude (with subagent assistance)
**Status**: Ready for decision on next phase
**Files**: 8 created/modified, ~15,000 words documentation
