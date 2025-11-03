# Attention Kernel Accuracy Findings

**Date**: November 3, 2025
**Test**: `test_attention_accuracy.py`
**Status**: CRITICAL ISSUE IDENTIFIED

## Executive Summary

The NPU attention kernel executes successfully and produces non-zero output, but **accuracy is very poor** compared to PyTorch reference:

- **Correlation**: 0.176 (target: >0.95)
- **MAE**: 31.78 (target: <2.0)
- **RMSE**: 36.74
- **Within ±5 tolerance**: Only 8.7% (target: >95%)

## Key Finding: Output Range Mismatch

**PyTorch Reference**: Output range [-64, +63] (full INT8 range)
**NPU Kernel**: Output range [-15, +14] (compressed to ~12% of full range)

This suggests the NPU kernel is not correctly implementing the full attention mechanism or is applying incorrect scaling.

## Visual Comparison

### PyTorch Reference (8x8 corner):
```
[[-15 -52 -27  61 -42 -29  -8  23]
 [-50 -17  45 -49   3 -15  56 -14]
 [ 37  16 -28   7 -24  13  38  40]
 ...]
```

### NPU Output (8x8 corner):
```
[[ 3 -6 -1  3 -4 -1  8 -8]
 [ 3 -6 -1  0 -3 -1  7 -8]
 [ 2 -3 -2  2 -2 -1  9 -6]
 ...]
```

**Observation**: NPU values are much smaller and appear to lack the dynamic range of the reference.

## Potential Root Causes

### 1. Incorrect Softmax Implementation
The attention mechanism requires:
```
attention = softmax(Q @ K^T / sqrt(d)) @ V
```

If softmax is not working correctly, the attention weights won't sum to 1.0, leading to incorrect output scaling.

### 2. Missing or Incorrect Scaling Factor
- Scale factor should be sqrt(64) = 8
- If scale is missing or wrong, QK^T scores will be too large/small
- This affects softmax distribution and final output

### 3. Integer Overflow in QK^T Computation
- Q @ K^T can produce values up to ±262,144 (64 * 64 * 64)
- INT8 saturation at ±128 would clip intermediate results
- Need INT32 accumulation for QK^T, then scale back to INT8

### 4. Quantization Issues
- INT8 quantization of attention weights may lose precision
- Softmax on INT8 is difficult (exponentials and normalization)
- May need higher precision intermediate computations

## Comparison with Day 1 Report

**Day 1 Status**:
- ✅ Attention returns non-zero output (89% non-zero)
- ✅ Executes without errors
- ⏳ Accuracy validation pending

**Today's Finding**:
- ✅ Confirmed: Non-zero output (91% non-zero)
- ✅ Confirmed: No execution errors
- ❌ **NEW**: Accuracy is very poor (0.18 correlation vs target 0.95)

## Impact on Week 1 Goals

**Original Plan**:
- Day 1: ✅ Validate attention kernel (DONE)
- Day 2: ❌ Attention accuracy validation (FAILED)
- Days 3-4: Implement batched matmul
- Day 5: Benchmark

**Revised Plan**:
- Day 2: Investigate attention kernel accuracy issue
- Days 3-4: Either fix attention OR focus on matmul (which is working)
- Day 5: Reassess based on progress

## Recommendations

### Option A: Debug Attention Kernel (2-3 days)
1. Review C kernel implementation (`attention_int8_64x64_tiled.c`)
2. Check softmax implementation
3. Verify scaling factors
4. Add INT32 accumulation for QK^T
5. Recompile and test

**Pros**: Gets attention working correctly
**Cons**: Time-consuming, may require MLIR changes

### Option B: Use CPU Attention Temporarily (immediate)
1. Skip NPU attention for now
2. Use PyTorch attention on CPU
3. Focus on batched matmul (which is likely more impactful)
4. Come back to attention in Phase 2

**Pros**: Quick progress on other optimizations
**Cons**: Leaves attention unoptimized

### Option C: Hybrid Approach (recommended)
1. Document attention issue thoroughly
2. File as "known issue" for Phase 2
3. Focus on matmul batching (Quick Win #2)
4. Focus on decoder fixes (Quick Wins #3-4)
5. Return to attention with more time allocated

**Pros**: Makes progress on multiple fronts
**Cons**: Attention remains unoptimized for now

## Next Steps (Immediate)

1. ✅ Document findings (this file)
2. Update `ATTENTION_VALIDATION_RESULTS.md` with accuracy data
3. Move to Quick Win #2: Batched MatMul implementation
4. Move to Quick Win #3: Decoder fixes

## Long-term Action Items

1. **Week 2**: Deep dive into attention C kernel
2. **Week 2**: Compare with working attention implementations
3. **Week 2**: Consider using higher precision (INT16/INT32) for intermediate computations
4. **Week 3**: Recompile with fixes and retest

## Success Criteria (Updated)

**Minimum** (Must achieve by Week 2):
- ⏳ Attention correlation >0.70
- ⏳ MAE <10.0
- ⏳ 80% within ±5 tolerance

**Good** (Target by Week 3):
- ⏳ Attention correlation >0.90
- ⏳ MAE <5.0
- ⏳ 90% within ±5 tolerance

**Excellent** (Stretch goal):
- ⏳ Attention correlation >0.95
- ⏳ MAE <2.0
- ⏳ 95% within ±5 tolerance

## Technical Deep Dive Needed

To fix this issue, we need to examine:

1. **C Kernel Source**: `attention_int8_64x64_tiled.c`
   - Check QK^T computation (is it using INT32 accumulation?)
   - Check scaling operation (is divide-by-8 correct?)
   - Check softmax implementation (is it numerically stable?)
   - Check final attention application (is V multiplication correct?)

2. **MLIR Lowering**: `attention_64x64.mlir`
   - Check buffer sizes (are intermediate buffers large enough?)
   - Check data flow (is data being copied correctly?)
   - Check quantization (are scales being applied correctly?)

3. **Compiler Flags**: `compile_attention_64x64.sh`
   - Check optimization level
   - Check vectorization settings
   - Check fixed-point arithmetic settings

## Conclusion

The attention kernel **works** (no crashes, produces output) but does **not work correctly** (poor accuracy). This is actually a more complex problem than if it simply crashed, because the issue is algorithmic/numerical rather than syntactic.

**Recommendation**: Proceed with Option C (Hybrid Approach) to make progress on other optimizations while this is investigated in parallel.

---

**Report By**: Encoder/Decoder Phase 1 Team Lead
**Date**: November 3, 2025
**Status**: CRITICAL ACCURACY ISSUE IDENTIFIED
**Priority**: HIGH (but defer to Week 2 for deep fix)
**Impact**: Attention will need CPU fallback until fixed
