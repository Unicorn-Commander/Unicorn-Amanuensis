# Phase 2: Quick Reference Card

**Date**: November 21, 2025

---

## 📊 The Bottom Line

**Goal**: Optimize attention on NPU for 1.1× RTF (faster than realtime)

**Result**: NPU implementation is **163× slower than CPU**

**Reason**: 64×64 kernel requires 3,008 invocations @ 54.4ms each = 163.7 seconds

**Status**: ✅ Numerically correct, ❌ Performance unusable

---

## ⚡ Quick Facts

| Metric | Value |
|--------|-------|
| **Current v2 RTF** | 0.75× (CPU-optimized) |
| **NPU matmul time** | 163.7s per projection |
| **CPU matmul time** | <1s per projection |
| **Slowdown** | 163× slower |
| **Kernel calls** | 3,008 (too many!) |
| **Time per call** | 54.4ms |
| **Accuracy** | <0.011 max error ✅ |

---

## 🎯 Three Options

### Option A: Compile Larger Kernels
- **Effort**: 4-8 hours
- **Result**: 8-16× speedup (may still be slower than CPU)
- **Guide**: `KERNEL_RECOMPILATION_GUIDE.md`

### Option B: Accept CPU Performance (RECOMMENDED)
- **Effort**: 0 hours
- **Result**: Keep 0.75× RTF
- **Next**: Focus on simpler optimizations for 1.0× RTF

### Option C: Hybrid CPU/NPU
- **Effort**: 1-2 hours
- **Result**: ~1.2× speedup (estimated)
- **Method**: Use NPU only for large matrices

---

## 📁 Key Files

**Documentation** (~15,000 words):
- `PHASE2_FINAL_SUMMARY_NOV21.md` - Complete analysis
- `OPTIMIZATION_SUMMARY.md` - Detailed findings
- `BATCHED_TILING_ANALYSIS.md` - Root cause
- `KERNEL_RECOMPILATION_GUIDE.md` - How to fix
- `PHASE2_QUICK_REFERENCE.md` - This file

**Code**:
- `attention_npu.py` - Tiled matmul (working, slow)
- `test_attention_matmul.py` - Tests (all pass)
- `quick_benchmark.py` - Performance test

---

## ✅ What Works

1. Tiled matmul implementation ✅
2. Accuracy < 0.011 ✅
3. Tests passing ✅
4. Integration with v2 encoder ✅

## ❌ What Doesn't Work

1. Performance (163× slower) ❌
2. Python-level batching (no effect) ❌
3. Real-time processing (impossible) ❌

---

## 🔧 Root Cause

**Problem**: 64×64 NPU kernel is too small

**Why**:
- Encoder projection: (3001, 512) @ (512, 512)
- Requires: 47 × 8 × 8 = 3,008 kernel calls
- Each call: 54.4ms (XRT + DMA + compute + Python)
- Total: 3,008 × 54.4ms = 163.7 seconds

**Fix**: Compile 128×512 kernel → 188 calls → 10.2 seconds

---

## 💡 Key Insight

**Python batching ≠ Kernel batching**

```python
# This DOESN'T reduce NPU calls:
for i in range(0, M, 64):
    for j in range(0, N, 512):  # "Batch" 8 tiles
        result = matmul_npu_64x64(...)  # Still 3,008 calls!

# This WOULD reduce calls (requires new kernel):
result = matmul_npu_128x512(...)  # Only 188 calls!
```

---

## 🚀 Recommendation

**Accept CPU performance (Option B)**

**Why**:
1. v2 is already fast (0.75× RTF)
2. Only need 1.33× improvement for realtime
3. Can achieve through simpler optimizations:
   - INT8 quantization (2-4× speedup)
   - Sequence chunking
   - Better batching strategies
4. Kernel recompilation is high-effort, uncertain gain

**Alternative**: If you have 4-8 hours and want to try NPU:
1. Follow `KERNEL_RECOMPILATION_GUIDE.md`
2. Compile 128×512 kernel
3. Measure actual performance
4. May still be slower than CPU due to DMA overhead

---

## 📈 Path to 1.0× RTF

**Without NPU** (recommended):
1. Implement INT8 quantization → 0.75× → 1.5× RTF ✅
2. OR: Optimize decoder (currently missing)
3. OR: Use smaller model (base instead of large)

**With NPU** (if kernel recompilation works):
1. Compile 128×512 kernel (4-8 hours)
2. If competitive with CPU:
   - Continue with multi-core optimization
   - Eventually reach 220× (multi-month effort)
3. If still slower than CPU:
   - Fall back to CPU approach

---

## 🎓 Lessons Learned

1. **Measure first**: Assumed 0.3ms overhead, actually 54.4ms
2. **Architecture matters**: 64×64 kernel is fundamental limitation
3. **Batching layer matters**: Python batching ≠ kernel batching
4. **CPU is fast**: Highly optimized BLAS, AVX-512, multi-threaded
5. **NPU needs custom kernels**: Generic kernels may be slower than CPU

---

## 📞 Decision Time

**Question**: Compile optimized kernels or accept CPU?

**Ask yourself**:
- Do I have 4-8 hours for kernel development?
- Is 0.75× RTF good enough for my use case?
- Do I need to maximize NPU utilization?
- Am I targeting 220× long-term?

**If yes to 1 & 4**: Try Option A (kernel recompilation)
**If yes to 2 & 3**: Choose Option B (accept CPU)
**If uncertain**: Start with Option B, revisit later

---

**For full details**: Read `PHASE2_FINAL_SUMMARY_NOV21.md`
