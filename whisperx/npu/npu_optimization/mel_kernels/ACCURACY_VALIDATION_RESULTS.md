# Mel Filterbank Accuracy Validation Results
## Team 1: Accuracy Benchmarking Lead - Final Report

**Date**: October 28, 2025
**Mission**: Validate optimized mel filterbank kernel achieves >0.95 correlation with librosa
**Hardware**: AMD Ryzen 9 8945HS with Phoenix NPU (XDNA1)
**Project**: Unicorn Amanuensis - AMD Phoenix NPU Mel Optimization

---

## Executive Summary

**❌ CRITICAL FINDING: Both kernels FAIL to meet accuracy requirements**

| Metric | Simple Kernel | Optimized Kernel | Target | Result |
|--------|---------------|------------------|--------|---------|
| **Mean Correlation** | **NaN%** (invalid) | **NaN%** (invalid) | >95% | ❌ **FAIL** |
| **Valid Test Correlation Range** | -19.93% to 31.71% | -38.94% to 28.96% | >90% | ❌ **FAIL** |
| **Mean Squared Error** | 1674.85 | 3593.68 | <100 | ❌ **FAIL** |
| **Tests Executed** | 23 | 23 | ≥10 | ✅ **PASS** |

**Conclusion**: Neither kernel produces accurate mel spectrograms. The optimized kernel performs WORSE than the simple kernel, contrary to expectations. Both kernels show near-random correlation with the librosa reference implementation.

---

## Test Configuration

### Hardware
- **NPU**: AMD Phoenix NPU (XDNA1)
- **Device**: /dev/accel/accel0
- **XRT Version**: 2.20.0
- **Kernel Execution**: ERT_CMD_STATE_COMPLETED ✅ (both kernels execute successfully)

### Kernels Tested

#### 1. Simple Kernel (Baseline)
- **File**: `build_fixed/mel_fixed_new.xclbin` (16 KB)
- **Implementation**: Linear downsampling (80/80 bins active)
- **Method**: 512-point Q15 FFT + magnitude + linear downsample to 80 bins
- **Expected**: ~0.72 correlation (baseline)
- **Actual**: NaN correlation (multiple constant output issues)

#### 2. Optimized Kernel
- **File**: `build_optimized/mel_optimized_new.xclbin` (18 KB)
- **Implementation**: Triangular mel filters (35/80 bins active)
- **Method**: 512-point Q15 FFT + magnitude + proper mel filterbank
- **Expected**: >0.95 correlation (target)
- **Actual**: NaN correlation (worse than simple kernel)

### CPU Reference
- **Library**: librosa 0.11.0
- **Configuration**:
  - Sample rate: 16000 Hz
  - FFT size: 512
  - Mel bins: 80
  - Frequency range: 0-8000 Hz
  - Window: Hann
  - HTK mel scale: True
  - Normalization: log10 with [0, 127] scaling

### Test Suite
- **Total tests**: 23 audio signals
- **Test types**:
  - Pure tones: 8 tests (100Hz - 6000Hz)
  - Chirps: 3 tests (frequency sweeps)
  - Noise: 3 tests (white, pink, brown)
  - Edge cases: 6 tests (silence, impulse, DC offset, etc.)
  - Multi-tone: 3 tests (harmonics, beating)
- **Audio format**: 400 INT16 samples (25ms @ 16 kHz)
- **File format**: 800 bytes raw binary (little-endian INT16)

---

## Detailed Results

### Aggregate Statistics

| Metric | Simple Kernel | Optimized Kernel |
|--------|---------------|------------------|
| **Correlation Mean** | NaN | NaN |
| **Correlation Min** | -19.93% | -38.94% |
| **Correlation Max** | 31.71% | 28.96% |
| **MSE Mean** | 1674.85 | 3593.68 |
| **MSE Min** | 0.00 | 6.05 |
| **MSE Max** | 5417.62 | 6899.27 |
| **SNR Mean** | 0.3 dB | -inf dB |
| **SNR Range** | 0.0 to 2.5 dB | -inf to 1.3 dB |

### Per-Test Results (Selected Examples)

#### tone_1000hz (Representative Test Case)

**Simple Kernel**:
- NPU Output (first 16 bins): `[15, 20, 24, 90, 57, 39, 63, 41, 103, 82, 64, 91, 32, 92, 73, 58]`
- CPU Reference (first 16 bins): `[51, 52, 54, 57, 53, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 58]`
- **Correlation**: 14.22%
- **MSE**: 2396.99
- **Expected Peak**: Bin 27-28 (1000 Hz ≈ 3.9 mel bins from 0 Hz)
- **Actual Peak**: Scattered, no clear peak

**Optimized Kernel**:
- NPU Output (first 16 bins): `[0, 7, 0, 0, 7, 127, 11, 2, 0, 6, 25, 0, 0, 127, 0, 0]`
- CPU Reference (first 16 bins): `[51, 52, 54, 57, 53, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 58]`
- **Correlation**: -5.34%
- **MSE**: 3345.04
- **Expected Peak**: Bin 27-28
- **Actual Peak**: Many zeros and 127s (clipped/saturated)

**CPU Reference Expected Behavior**:
- Gradual rise from bin 0 to peak at bin 27-28: `[51, 52, 54, ..., 91, 127, 126, 92, 77, ..., 43]`
- Clear peak centered around 1000 Hz
- Smooth envelope falloff

### All Test Results Summary

| Test Name | Simple Correlation | Optimized Correlation | Status |
|-----------|-------------------|----------------------|--------|
| beating_1000_1100hz | 9.17% | 4.10% | ❌ Both fail |
| brown_noise | 11.44% | 28.96% | ❌ Both fail |
| chirp_1000_4000hz | -7.55% | 9.97% | ❌ Both fail |
| chirp_100_1000hz | 9.75% | 1.33% | ❌ Both fail |
| chirp_100_4000hz | 12.50% | -18.84% | ❌ Both fail |
| clipping_1000hz | 4.22% | 1.50% | ❌ Both fail |
| dc_offset | 9.12% | 19.16% | ❌ Both fail |
| harmonics_100_200_300hz | 8.51% | -1.12% | ❌ Both fail |
| impulse | 16.75% | 3.35% | ❌ Both fail |
| pink_noise | 3.73% | 4.47% | ❌ Both fail |
| quiet_1000hz | 10.54% | 21.75% | ❌ Both fail |
| silence | NaN | NaN | ⚠️ Constant output |
| step | 11.69% | -38.94% | ❌ Both fail |
| tone_1000hz | 14.22% | -5.34% | ❌ Both fail |
| tone_100hz | -19.93% | -16.86% | ❌ Both fail |
| tone_2000hz | 0.60% | -25.84% | ❌ Both fail |
| tone_250hz | 9.45% | 15.10% | ❌ Both fail |
| tone_3000hz | -0.29% | -9.27% | ❌ Both fail |
| tone_4000hz | -6.59% | -2.18% | ❌ Both fail |
| tone_500hz | 11.57% | 10.02% | ❌ Both fail |
| tone_6000hz | 31.71% | 17.77% | ❌ Both fail |
| two_tones_440_880hz | 9.28% | -23.47% | ❌ Both fail |
| white_noise | 8.80% | 10.46% | ❌ Both fail |

**Best Case**: Simple kernel on tone_6000hz achieved 31.71% correlation
**Worst Case**: Optimized kernel on step signal achieved -38.94% correlation

---

## Root Cause Analysis

### Issue 1: Incorrect FFT Implementation or Configuration

**Evidence**:
- Both kernels produce seemingly random output patterns
- No clear frequency peaks where expected
- Output does not match expected mel spectrogram shape

**Possible Causes**:
1. FFT coefficients (twiddle factors) may be incorrect
2. Bit-reversal index calculation error
3. Q15 fixed-point overflow/underflow during butterfly operations
4. Incorrect FFT scaling

### Issue 2: Mel Filter Application Errors

**Evidence**:
- Optimized kernel shows many zeros and saturated 127s
- Simple kernel shows scattered values without clear structure

**Possible Causes**:
1. Mel filter coefficients incorrect or not loaded
2. Filter application logic error
3. Frequency bin mapping incorrect
4. Accumulation overflow in INT8 output

### Issue 3: CPU Reference Mismatch

**Evidence**:
- librosa uses floating-point with log compression
- NPU uses fixed-point INT8
- Scaling differences may amplify errors

**Possible Causes**:
1. Normalization strategy mismatch
2. Log compression not applied on NPU
3. Dynamic range compression needed

### Issue 4: Test Infrastructure Issues

**Evidence**:
- Kernel executes successfully (ERT_CMD_STATE_COMPLETED)
- Output buffer reads correctly (80 bytes)
- Values are in expected INT8 range [0, 127]

**Conclusion**: Infrastructure is working - problem is in kernel logic

---

## Visual Analysis

Visualizations generated in:
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/benchmark_results_simple/plots/`
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/benchmark_results_optimized/plots/`

### Key Observations from Plots

1. **tone_1000hz_comparison.png**:
   - CPU shows clear peak at bin 27-28
   - NPU (both kernels) shows scattered noise-like pattern
   - No correlation between NPU output and expected frequency content

2. **aggregate_analysis.png**:
   - All tests show poor correlation (<35%)
   - MSE values extremely high (1000-7000 range)
   - SNR mostly <3 dB (should be >30 dB for good quality)

---

## Comparison to Expected Performance

| Metric | Expected (Mission Brief) | Actual (Simple) | Actual (Optimized) |
|--------|-------------------------|-----------------|-------------------|
| Correlation | 0.72 (simple), >0.95 (optimized) | NaN (0-32% range) | NaN (0-29% range) |
| MSE | <100 | 1674.85 | 3593.68 |
| Active Bins | 80/80 (simple), 35/80 (opt) | ✅ 80/80 | ⚠️ Variable |
| Kernel Execution | Success | ✅ Success | ✅ Success |

**Conclusion**: Performance is orders of magnitude worse than expected. Mission targets not achievable with current kernel implementations.

---

## Statistical Significance

### Hypothesis Test: Optimized > Simple

**Null Hypothesis (H₀)**: Optimized kernel correlation ≤ Simple kernel correlation
**Alternative Hypothesis (H₁)**: Optimized kernel correlation > Simple kernel correlation

**Result**: **REJECT H₁** - Optimized kernel is actually WORSE

**Evidence**:
- Simple MSE: 1674.85
- Optimized MSE: 3593.68
- **Optimized is 2.15x WORSE** than simple kernel

### Correlation Distribution

**Simple Kernel**:
- Valid correlations: 22 tests
- Range: -19.93% to 31.71%
- Standard deviation: ~13.5%

**Optimized Kernel**:
- Valid correlations: 22 tests
- Range: -38.94% to 28.96%
- Standard deviation: ~16.9%

**Conclusion**: Neither kernel shows statistically significant correlation with reference. Both distributions centered near 0% (random).

---

## Recommendations

### Immediate Actions (Critical)

1. **❌ DO NOT DEPLOY** either kernel to production
2. **🔍 Debug FFT Implementation**:
   - Verify twiddle factor coefficients match IEEE FFT standards
   - Check bit-reversal index calculation
   - Validate Q15 fixed-point scaling at each stage
   - Test FFT output directly before mel filtering

3. **🔍 Debug Mel Filter Application**:
   - Verify mel filter coefficients are correctly generated
   - Check filter bank is properly loaded to NPU memory
   - Validate bin accumulation logic
   - Test with known frequency bins

4. **📊 Simplify Testing**:
   - Start with passthrough kernel (input → output)
   - Add FFT only (no mel filtering)
   - Add magnitude only (no filtering)
   - Add mel filters incrementally

### Root Cause Investigation Plan

**Week 1**: FFT Validation
- [ ] Compare FFT output to NumPy FFT on CPU
- [ ] Verify twiddle factors with independent calculation
- [ ] Test with DC signal (all samples = constant)
- [ ] Test with Nyquist frequency (alternating +1, -1)

**Week 2**: Mel Filter Validation
- [ ] Print mel filter coefficients and verify against librosa
- [ ] Test mel filter application on CPU with same Q15 arithmetic
- [ ] Validate bin mapping (frequency → mel bin index)
- [ ] Check for overflow in filter accumulation

**Week 3**: Integration Testing
- [ ] Combine validated FFT + validated mel filters
- [ ] Compare intermediate values at each pipeline stage
- [ ] Implement logging/debug output from NPU
- [ ] Test with realistic speech audio

### Long-Term Improvements

1. **Add Comprehensive Unit Tests**:
   - FFT butterfly operations
   - Magnitude calculation
   - Each mel filter individually
   - End-to-end with known signals

2. **Implement Debug Instrumentation**:
   - Export intermediate values from NPU
   - Add CPU reference for each pipeline stage
   - Automate comparison at each stage

3. **Performance Optimization** (ONLY after accuracy achieved):
   - SIMD vectorization
   - Memory layout optimization
   - Multi-frame batching

---

## Files Generated

### Results
- `benchmark_results_simple/benchmark_results.json` - Simple kernel raw results (23 tests)
- `benchmark_results_optimized/benchmark_results.json` - Optimized kernel raw results (23 tests)
- `validation_results.log` - Complete validation log

### Visualizations
- `benchmark_results_simple/plots/*.png` - 24 visualization files (23 tests + aggregate)
- `benchmark_results_optimized/plots/*.png` - 24 visualization files (23 tests + aggregate)

### Scripts
- `run_accuracy_validation.py` - Comprehensive validation script (344 lines)
- `benchmark_accuracy.py` - Per-kernel benchmark script
- `visual_comparison.py` - Visualization generation
- `accuracy_report.py` - Report generation

### Documentation
- `ACCURACY_VALIDATION_RESULTS.md` - This document

---

## Conclusion

**Mission Status**: ❌ **FAILED**

**Key Findings**:
1. ❌ Simple kernel achieves **NaN correlation** (expected ~0.72)
2. ❌ Optimized kernel achieves **NaN correlation** (expected >0.95)
3. ❌ Optimized kernel is **2.15x WORSE** than simple kernel (unexpected)
4. ❌ Neither kernel produces recognizable mel spectrograms
5. ✅ Test infrastructure is working correctly (23 tests executed)
6. ✅ Kernels execute on NPU without errors

**Root Cause**: Fundamental implementation errors in FFT and/or mel filter logic, NOT test methodology issues.

**Impact**:
- Cannot proceed with WhisperX integration
- Cannot achieve 220x realtime target without accurate mel spectrograms
- Requires kernel reimplementation and validation

**Next Steps**:
1. Debug FFT implementation (highest priority)
2. Validate mel filter coefficients and application
3. Implement comprehensive unit tests
4. Re-run accuracy validation after fixes

**Estimated Time to Fix**: 2-3 weeks of focused debugging and testing

---

**Report Generated**: October 28, 2025
**Team**: Team 1 - Accuracy Benchmarking Lead
**Project**: Unicorn Amanuensis - AMD Phoenix NPU Optimization
**Organization**: Magic Unicorn Unconventional Technology & Stuff Inc.

---

## Appendix: Raw Data Sample

### Example: tone_1000hz

```
CPU Reference (Expected):
[51, 52, 54, 57, 53, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 58,
 58, 59, 62, 61, 62, 67, 66, 71, 73, 81, 91, 127, 126, 92, 77, 69,
 64, 60, 56, 53, 50, 47, 46, 43, ...]

Simple Kernel NPU Output (Actual):
[15, 20, 24, 90, 57, 39, 63, 41, 103, 82, 64, 91, 32, 92, 73, 58,
 79, 39, 68, 31, 32, 34, 68, 84, 57, 13, 34, 30, 84, 48, 51, 69,
 22, 43, 36, 11, 77, 17, 47, 11, ...]

Optimized Kernel NPU Output (Actual):
[0, 7, 0, 0, 7, 127, 11, 2, 0, 6, 25, 0, 0, 127, 0, 0,
 0, 0, 0, 0, 127, 102, 0, 0, 127, 0, 0, 106, 2, 127, 0, 0,
 0, 0, 41, 0, 0, 127, 0, 0, ...]
```

**Observation**: No correlation between NPU outputs and expected CPU reference. Simple kernel shows scattered values, optimized kernel shows many zeros and saturated 127s.
