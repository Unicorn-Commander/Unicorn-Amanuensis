# FFT Debugging Findings - October 29, 2025

## Critical Discovery: Lookup Tables Work, FFT Algorithm Broken

### Tests Performed

1. **✅ Passthrough Test** - Data path works perfectly
2. **✅ Hann Window Test** - 100% accurate (80/80 matches)
3. **✅ Minimal 2-Point FFT** - Basic math works (returns 75, 25 as expected)
4. **✅ Lookup Table Access Test** - ALL lookup tables readable on NPU:
   - `bit_reverse_lut[]` - ✅ Works perfectly
   - `twiddle_cos_q15[]` - ✅ Works perfectly
   - `twiddle_sin_q15[]` - ✅ Works perfectly
5. **❌ Full 512-Point FFT** - Returns all zeros

### What We Know Works on NPU

- ✅ Data transfer (input → NPU → output)
- ✅ Basic arithmetic operations
- ✅ Loops (for loops execute correctly)
- ✅ Array indexing
- ✅ Reading from const arrays (lookup tables)
- ✅ Simple FFT butterfly computations

### What Doesn't Work

- ❌ Full `fft_radix2_512_fixed()` function returns zeros
- ❌ All output bins are zero or near-zero
- ❌ No recognizable FFT output pattern

### Root Cause Analysis

The problem is **NOT**:
- ❌ Missing lookup tables
- ❌ Inaccessible const arrays
- ❌ Data path issues
- ❌ Basic computation failures

The problem **IS**:
- ✅ Something in the FFT algorithm itself
- ✅ Likely in complex multiplication or scaling
- ✅ Possibly in nested loop structure
- ✅ Possibly in bit-reverse permutation with large arrays

### Evidence

**Lookup Table Test Output:**
```
[0] = 0    <- bit_reverse_lut[0] ✅
[1] = 1    <- bit_reverse_lut[1] >> 8 ✅
[2] = 1    <- bit_reverse_lut[256] ✅
[3] = 127  <- twiddle_cos_q15[0] >> 8 ✅
[4] = -126 <- twiddle_cos_q15[64] & 0xFF ✅
[5] = 0    <- twiddle_sin_q15[0] ✅
[6] = -91  <- twiddle_sin_q15[64] >> 8 ✅
[7] = 127  <- Loop sum of twiddle values ✅
[8-15] = [16, 18, 20, 22, 24, 26, 28, 30] <- Pattern ✅
```

All values match expectations perfectly!

**Minimal FFT Test Output:**
```
[0] = 75  <- (100 + 50) / 2 ✅
[1] = 25  <- (100 - 50) / 2 ✅
```

Basic FFT butterfly computation works!

### Hypothesis for FFT Failure

Possible causes (in order of likelihood):

1. **Complex Multiplication Overflow** (lines 90 in `fft_fixed_point.c`):
   - `cmul_q15()` might overflow when processing real data
   - INT16 multiplication → INT32 intermediate → INT16 result
   - Possible overflow on specific input patterns

2. **Scaling Issues** (lines 101-104 in `fft_fixed_point.c`):
   - Per-stage scaling `>>1` might cause underflow
   - After 9 stages (512-point FFT), total scaling is `÷512`
   - Small input values might become zero

3. **Nested Loop Complexity**:
   - 3 nested loops with 512 iterations total
   - NPU might have issues with this structure
   - Possible timeout or resource exhaustion

4. **Bit-Reverse Array Access**:
   - Reading `output[bit_reverse_lut[i]]` works in isolation
   - But might fail in the full algorithm context
   - Possible memory access pattern issue

### Next Steps

1. **Create 4-Point FFT Test**:
   - Use same `fft_radix2_512_fixed()` structure
   - But only process 4 points instead of 512
   - Should reveal if issue is scaling or complexity

2. **Test Individual FFT Stages**:
   - Run just stage 0 (2-point butterflies)
   - Run stages 0-1 (4-point)
   - Run stages 0-2 (8-point)
   - Identify at which stage it breaks

3. **Test Without Scaling**:
   - Remove the `>>1` per-stage scaling
   - See if output is non-zero (even if wrong)
   - Would confirm if underflow is the issue

4. **Simplify Complex Multiplication**:
   - Replace `cmul_q15()` with simpler version
   - Use fixed-point with less precision
   - Test if arithmetic is the issue

### Current Code

**Working:**
- `mel_kernel_PASSTHROUGH.c` - Simple copy test
- `mel_kernel_DEBUG_STAGES.c` - Lookup table test
- `fft_fixed_point.c` - FFT implementation (compiles correctly)

**Not Working:**
- Full FFT in `fft_radix2_512_fixed()` when executed on NPU

### Compilation Status

- ✅ All code compiles without errors
- ✅ Object files contain all symbols
- ✅ XCLBIN generation successful
- ✅ XRT loads and executes kernel
- ❌ FFT returns zeros instead of correct values

### Time Invested

- Stage 1 (Hann window): 30 minutes
- Stage 2 (FFT debug): 2 hours
- Lookup table verification: 1 hour
- **Total**: 3.5 hours of systematic debugging

### Confidence Level

- **Lookup Tables**: 100% confirmed working ✅
- **Data Path**: 100% confirmed working ✅
- **Basic Computation**: 100% confirmed working ✅
- **FFT Algorithm**: 0% working on NPU ❌
- **Root Cause**: 80% confident it's overflow/underflow in complex math

### Recommendation

**Next debugging session should:**
1. Create simplified 4-point FFT test (30 minutes)
2. Test individual FFT stages (1 hour)
3. Compare NPU output with CPU output stage-by-stage
4. Identify exact stage/operation that fails

**Estimated time to fix**: 2-4 hours once root cause identified

---

**Date**: October 29, 2025
**Status**: Lookup tables confirmed working, FFT algorithm broken
**Files**: All test code in `mel_kernels/` directory
