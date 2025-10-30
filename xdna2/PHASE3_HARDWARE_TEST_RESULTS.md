# Phase 3: Hardware Validation Results

**Date**: October 30, 2025, 03:40 UTC
**Test**: Full 6-layer Whisper Encoder on XDNA2 NPU
**Hardware**: AMD Strix Halo (XDNA2, 50 TOPS)
**Status**: TESTS COMPLETE - Issues Identified

---

## Executive Summary

Hardware validation tests completed successfully but revealed **significant performance and accuracy issues** that must be addressed:

- **Performance**: 5.97x realtime (2.7% of 220x baseline, 1.3% of 450x target)
- **Accuracy**: 7.7% relative error (FAIL - exceeds 2% tolerance)
- **Single Layer**: 282.76 ms average latency
- **Full Encoder**: 1,713.81 ms (1.71 seconds)

**Root Causes Identified**:
1. K-dimension chunking overhead is massive (4x expected)
2. Attention uses 512x512x512 kernel with chunking (inefficient)
3. Quantization error accumulation across 6 layers
4. Possible kernel performance issues

---

## Test Configuration

```
Model: Whisper Base (6 layers)
Sequence Length: 512 tokens
Hidden Dimension: 512
Attention Heads: 8 (64 dims each)
FFN Dimension: 2048
Kernel: 4-tile INT8 matmul
Quantization: Symmetric INT8
Test Date: Oct 30, 2025
```

---

## Test Results

### TEST 1: Single Encoder Layer

**Input**: (512, 512) random FP32 tensor
**Runs**: 3 iterations after warmup

| Metric | Value |
|--------|-------|
| Run 1 | 290.64 ms |
| Run 2 | 280.34 ms |
| Run 3 | 277.29 ms |
| **Average** | **282.76 ms** |
| Std Dev | 5.71 ms |
| Output Shape | (512, 512) |
| Output Range | [-9.491, 6.240] |

**Analysis**:
- Consistent timing (~2% variance)
- Single layer taking ~280ms is **very slow**
- Expected: ~47ms (6 layers @ 280ms = 1680ms vs 1714ms actual)

---

### TEST 2: Full 6-Layer Encoder

**Input**: (512, 512) random FP32 tensor
**Runs**: 3 iterations after warmup

| Metric | Value |
|--------|-------|
| Run 1 | 1688.20 ms |
| Run 2 | 1711.01 ms |
| Run 3 | 1742.22 ms |
| **Average** | **1713.81 ms** |
| Std Dev | 22.14 ms |
| Output Shape | (512, 512) |
| Output Range | [-16.391, 21.147] |

**Performance Metrics**:
- Latency: **1,713.81 ms** (1.71 seconds)
- Audio duration: 10.24 seconds (512 frames @ 20ms/frame)
- **Realtime factor: 5.97x** ⚠️

**Comparison**:
- XDNA1 Baseline: 220x realtime
- Current: 5.97x realtime
- Gap: 214.03x (2.7% of baseline!)
- Target: 450x realtime
- Gap to target: 444.03x (75.31x speedup needed)

---

### TEST 3: Layer-by-Layer Profiling

**Per-Layer Latencies** (2 runs each, averaged):

| Layer | Avg Latency | Run 1 | Run 2 |
|-------|-------------|-------|-------|
| Layer 0 | 325.93 ms | 330.60 ms | 321.27 ms |
| Layer 1 | 319.36 ms | 317.64 ms | 321.08 ms |
| Layer 2 | 271.88 ms | 270.99 ms | 272.76 ms |
| Layer 3 | 313.36 ms | 308.06 ms | 318.66 ms |
| Layer 4 | 340.30 ms | 320.43 ms | 360.16 ms |
| Layer 5 | 320.05 ms | 343.73 ms | 296.38 ms |

**Statistics**:
- Total: 1,890.88 ms
- Average: 315.15 ms
- Min: 271.88 ms (Layer 2)
- Max: 340.30 ms (Layer 4)
- Variance: 68.42 ms (max - min)

**Analysis**:
- Layers are reasonably consistent (~21% variance)
- No outlier layers (all within expected range)
- Layer 4 slightly slower (possible cache effects)

---

### TEST 4: Accuracy Validation

**Comparison**: NPU vs CPU FP32 Reference

| Metric | Value | Status |
|--------|-------|--------|
| MSE | 0.010143 | - |
| MAE | 0.075235 | - |
| Max Diff | 2.769601 | - |
| **Relative Error** | **7.716%** | **FAIL** |
| Tolerance | 2.0% | - |

**Output Statistics**:

| Stat | NPU | CPU |
|------|-----|-----|
| Range | [-16.391, 21.147] | [-16.625, 21.678] |
| Mean | -0.010 | -0.009 |
| Std | 1.426 | 1.429 |

**Status**: **FAIL** - 7.716% error exceeds 2% tolerance

**Analysis**:
- Quantization error is accumulating across layers
- INT8 quantization causing ~7.7% error after 6 layers
- Each layer contributes ~1.3% error
- Needs better quantization strategy or per-layer calibration

---

### TEST 5: Operation Breakdown

**Single Layer Components** (Layer 0, 3 runs averaged):

| Operation | Latency | Percentage |
|-----------|---------|------------|
| Attention | 256.87 ms | 75.1% |
| FFN | 85.13 ms | 24.9% |
| **Total** | **342.00 ms** | **100%** |

**Analysis**:
- Attention dominates (3x more time than FFN)
- Attention has 4 matmuls: Q, K, V, Out (each 512x512x512)
- FFN has 2 matmuls: FC1 (512x512x2048), FC2 (512x2048x512)
- **Attention is using chunking** (512x512x512 kernel, no direct support)
- **FFN FC1 is using 512x512x2048 kernel** (direct support, faster!)

**Bottleneck**: Attention Q/K/V projections using chunked execution

---

## Performance Breakdown

### Matmul Operations Per Layer

**Attention** (4 matmuls):
- Q projection: 512x512x512 (chunked)
- K projection: 512x512x512 (chunked)
- V projection: 512x512x512 (chunked)
- Out projection: 512x512x512 (chunked)

**FFN** (2 matmuls):
- FC1: 512x512x2048 (direct kernel!)
- FC2: 512x2048x512 (chunked: 4 chunks)

**Total per layer**: 6 matmuls
**Total for 6 layers**: 36 matmuls

### Chunking Analysis

**Current Kernels**:
1. `512x512x512`: Direct execution
2. `512x512x2048`: Direct execution for FC1

**Chunking Required**:
- Attention Q/K/V/Out: All use 512x512x512 → NO chunking needed!
- FFN FC1: 512x512x2048 → Direct kernel (GOOD!)
- FFN FC2: 512x2048x512 → K=2048, chunked into 4x512 (4× overhead)

**Wait, this doesn't add up!**

Let me check the actual kernel usage...

---

## Critical Issue Identified

Looking at the operation breakdown:
- Attention: 256.87 ms (should be 4× single matmul)
- FFN: 85.13 ms (should be 2× matmuls)

**Expected single matmul timing**:
- 512x512x512: ~64ms (256ms / 4 for attention)
- 512x512x2048: ~43ms (if FC1 is using direct kernel)
- 512x2048x512: ~43ms × 4 chunks = ~172ms? (But we see 85ms total for FFN)

**Something is wrong with our assumptions!**

Possible issues:
1. Kernels are slower than expected (64ms vs projected 5-10ms)
2. Chunking overhead is higher than 4× (maybe 10-20×?)
3. CPU-NPU transfer overhead is significant
4. Quantization/dequantization overhead

---

## Root Cause Analysis

### Issue 1: Kernel Performance

**Observed**: Single 512x512x512 matmul takes ~64ms
**Expected**: ~5-10ms (based on 1,183x baseline)

**Gap**: 6-13x slower than expected!

**Possible causes**:
- 4-tile kernel is much slower than 32-tile
- AIE2 compilation not optimized
- Memory bandwidth bottleneck
- Wrong kernel parameters

### Issue 2: Quantization Error

**Observed**: 7.7% error after 6 layers
**Expected**: <2% error

**Causes**:
- Error accumulation across layers
- Symmetric quantization not ideal for activations
- Need per-layer calibration
- Attention softmax/layernorm in FP32 but surrounding ops in INT8

### Issue 3: Chunking Overhead

**FFN FC2** should be chunked:
- 512x2048x512 → 4 chunks of 512x512x512
- Expected: 4× single matmul latency
- Observed: FFN total is 85ms, but FC1 alone should be ~43ms

**This suggests FC1 is NOT using the 512x512x2048 kernel!**

Need to check kernel selection logic.

---

## Comparison to Baseline

| Metric | XDNA1 Baseline | XDNA2 Current | Ratio |
|--------|----------------|---------------|-------|
| Realtime Factor | 220x | 5.97x | 0.027× |
| Encoder Latency | ~46ms | 1,714ms | 37× slower |
| Single Layer | ~7.7ms | 283ms | 37× slower |
| Audio Duration | 10.24s | 10.24s | 1× |

**XDNA2 is 37× SLOWER than XDNA1!**

This is **completely unexpected** and indicates a serious implementation issue.

---

## Optimization Opportunities

### Short-Term (Phase 4): 2-4 hours

1. **Verify Kernel Selection**
   - Add logging to `_run_matmul_npu` to show which kernel is used
   - Check that 512x512x2048 kernel is actually being selected for FC1
   - Verify chunking logic for 512x2048x512 (FC2)

2. **Profile Kernel Performance**
   - Measure individual kernel execution time (not including transfers)
   - Compare to CC-1L baseline (1,183x kernel)
   - Check if 4-tile vs 32-tile is the issue

3. **Reduce Quantization Error**
   - Implement per-layer calibration (collect activation stats)
   - Try asymmetric quantization (with zero point)
   - Experiment with INT16 for critical layers

4. **Optimize Memory Transfers**
   - Check buffer registration (should be once at init)
   - Measure transfer overhead separately
   - Investigate NPU memory pinning

**Expected improvement**: 2-5× speedup (to 12-30x realtime)

### Medium-Term (Phase 5): 10-15 hours

1. **Compile Optimized Kernels**
   - Build 32-tile kernels (vs current 4-tile)
   - Tune tile mappings for XDNA2 (32 tiles vs XDNA1's 16)
   - Implement kernel for 512x2048x512 (eliminate chunking)

2. **Attention Optimization**
   - Fuse Q/K/V projections into single kernel
   - Implement attention on NPU (not just matmuls)
   - Use flash attention patterns

3. **Advanced Quantization**
   - Group quantization (per-head for attention)
   - Mixed precision (INT8 + INT16)
   - Quantization-aware training (if needed)

**Expected improvement**: 20-50× speedup (to 100-250x realtime)

### Long-Term (Phase 6+): 20-40 hours

1. **Full NPU Pipeline**
   - Mel spectrogram on NPU
   - Entire encoder on NPU (no CPU fallback)
   - Decoder on NPU

2. **Architecture Changes**
   - Custom attention kernels
   - Tiled execution for long sequences
   - Streaming inference

**Expected improvement**: 100-200× speedup (to 400-500x target)

---

## Next Steps

### Immediate (Next Session)

1. **Add Kernel Selection Logging**
   ```python
   logger.info(f"Using kernel: {kernel_name} for matmul {M}x{K}x{N}")
   ```

2. **Profile Individual Matmul**
   - Measure just kernel execution (not transfers)
   - Compare 512x512x512 vs 512x512x2048 performance

3. **Investigate 4-Tile Performance**
   - Check if 4-tile is inherently slower
   - Consider compiling 32-tile kernel
   - Review CC-1L kernel parameters

4. **Debug Quantization Error**
   - Test single layer accuracy (should be ~1.3% error)
   - Check scale values (print to log)
   - Verify dequantization logic

### Phase 4 Goals

- **Target**: 20-50x realtime (from current 5.97x)
- **Approach**: Fix implementation issues, optimize kernels
- **Accuracy**: <2% error (from current 7.7%)

---

## Conclusions

### Summary

Hardware validation **revealed critical issues**:
1. **Performance 37× slower than XDNA1 baseline** (completely unexpected!)
2. **Accuracy fails at 7.7% error** (vs 2% target)
3. **Kernel performance far below expectations** (~64ms vs ~5-10ms)
4. **Possible kernel selection bugs** (FC1 may not be using 512x512x2048)

### Critical Findings

1. **4-Tile Kernel May Be the Bottleneck**
   - Single matmul: 64ms (should be 5-10ms)
   - 6-13× slower than expected
   - Need to test 32-tile kernel

2. **Quantization Strategy Needs Work**
   - 7.7% error is too high
   - Need per-layer calibration
   - Consider mixed precision

3. **Implementation Issues to Debug**
   - Kernel selection logic
   - Transfer overhead
   - Chunking implementation

### Confidence in Target

**Current Confidence: 40%** (was 95% before hardware testing!)

**Risks**:
- 4-tile kernel may be fundamentally slower
- XDNA2 may have different performance characteristics
- Quantization error harder to solve than expected

**Path Forward**:
1. Debug kernel selection (immediate)
2. Compile and test 32-tile kernel (Phase 4)
3. Fix quantization error (Phase 4)
4. Re-evaluate target after Phase 4 (2-4 hours)

### Recommendation

**Proceed to Phase 4** with focus on:
1. Debugging current implementation
2. Profiling individual operations
3. Testing 32-tile kernel
4. Fixing quantization

**Re-assess target** after Phase 4 debugging complete.

---

## Appendix: Raw Test Output

```
======================================================================
  WHISPER ENCODER HARDWARE VALIDATION
  XDNA2 NPU - Full 6-Layer Test
======================================================================

Test Configuration:
  Model: Whisper Base (6 layers)
  Sequence length: 512 tokens
  Hidden dimension: 512
  Kernel: 4-tile INT8 matmul
  Quantization: Symmetric INT8

[... full test output as shown above ...]
```

---

**Generated**: October 30, 2025, 03:50 UTC
**Next Phase**: Phase 4 - Debugging and Optimization
**Est. Time**: 2-4 hours
**Priority**: CRITICAL - Debug kernel selection and performance
