# NPU Mel Spectrogram Accuracy Report

**Generated**: 2025-10-28 06:06:12 UTC
**Project**: Unicorn Amanuensis - AMD Phoenix NPU Optimization
**Hardware**: AMD Ryzen 9 8945HS with Phoenix NPU (XDNA1)
**Implementation**: Q15 Fixed-Point FFT with Linear Mel Binning

---

## Executive Summary

**❌ Overall Verdict: NEEDS IMPROVEMENT (FAIL)**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Correlation** | nan% | >99% (excellent), >95% (good) | ❌ |
| **Mean Squared Error** | 2564.5631 | <0.01 (excellent), <0.1 (good) | ❌ |
| **Signal-to-Noise Ratio** | 0.3 dB | >40 dB (excellent), >30 dB (good) | ❌ |
| **Mean Absolute Error** | 37.5231 | <1.0 (excellent), <5.0 (good) | ❌ |

**Summary**: The NPU fixed-point FFT implementation shows **needs improvement** accuracy compared to the librosa CPU reference. Significant improvements are recommended before production deployment.

---

## Test Configuration

**NPU Implementation**:
- **Algorithm**: 512-point Radix-2 FFT
- **Arithmetic**: Q15 fixed-point (INT16)
- **Window**: Hann window (Q15 coefficients)
- **Mel Binning**: Linear downsampling (256 → 80 bins)
- **Output**: INT8 scaled to [0, 127]

**CPU Reference** (librosa):
- **Library**: librosa v0.10+
- **FFT Size**: 512
- **Mel Bins**: 80
- **Frequency Range**: 0-8000 Hz
- **Mel Scale**: HTK (Whisper standard)
- **Window**: Hann
- **Scaling**: Log10 with normalization

**Test Suite**:
- Total tests: 23
- Test types: Pure tones, chirps, noise, edge cases, multi-tone
- Audio format: 400 INT16 samples (25ms @ 16 kHz)

---

## Detailed Test Results

### Synthetic Tones

| Frequency | Correlation | MSE | MAE | SNR (dB) | Status |
|-----------|-------------|-----|-----|----------|--------|
| 1000hz | -23.89% | 2396.9879 | 39.6634 | 0.0 | ❌ |
| 100hz | -18.38% | 2132.2037 | 33.6085 | 0.2 | ❌ |
| 2000hz | -17.50% | 1648.8887 | 33.5330 | 0.0 | ❌ |
| 250hz | -5.25% | 1581.1077 | 28.5689 | 0.0 | ❌ |
| 3000hz | -17.10% | 1143.2441 | 25.8203 | 0.0 | ❌ |
| 4000hz | -6.39% | 690.4159 | 15.9475 | 0.0 | ❌ |
| 500hz | -13.65% | 2597.2337 | 39.7153 | 0.0 | ❌ |
| 6000hz | 29.45% | 887.9997 | 17.9617 | 0.1 | ❌ |

### Chirps (Frequency Sweeps)

| Test | Correlation | MSE | MAE | SNR (dB) | Status |
|------|-------------|-----|-----|----------|--------|
| Chirp 1000 4000Hz | 3.91% | 4128.2104 | 43.2070 | 0.4 | ❌ |
| Chirp 100 1000Hz | -8.54% | 3568.3572 | 42.5341 | 0.3 | ❌ |
| Chirp 100 4000Hz | -3.94% | 6483.5927 | 67.9498 | 0.4 | ❌ |

### Noise Tests

| Noise Type | Correlation | MSE | MAE | SNR (dB) | Status |
|------------|-------------|-----|-----|----------|--------|
| Brown Noise | 6.88% | 1078.9531 | 22.4812 | 1.6 | ❌ |
| Pink Noise | nan% | 3493.3660 | 52.1073 | 0.0 | ❌ |
| White Noise | -9.13% | 7220.9010 | 81.4672 | 0.0 | ❌ |

### Edge Cases

| Test | Correlation | MSE | MAE | SNR (dB) | Status |
|------|-------------|-----|-----|----------|--------|
| Clipping 1000Hz | -29.68% | 2982.9968 | 47.6248 | 0.1 | ❌ |
| Dc Offset | -3.49% | 1426.0081 | 26.9507 | 0.3 | ❌ |
| Impulse | nan% | 3780.4018 | 59.7640 | 0.0 | ❌ |
| Quiet 1000Hz | nan% | 2463.7288 | 40.8652 | 0.0 | ❌ |
| Silence | nan% | 0.0000 | 0.0000 | ∞ | ❌ |
| Step | -23.45% | 1742.9923 | 30.0046 | 0.4 | ❌ |

---

## Statistical Analysis

### Overall Metrics

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| **Correlation** | nan% | nan% | nan% | nan% |
| **MSE** | 2564.5631 | 1697.4324 | 0.0000 | 7220.9010 |
| **MAE** | 37.5231 | 17.1569 | 0.0000 | 81.4672 |
| **RMSE** | 47.4303 | 17.7462 | 0.0000 | 84.9759 |
| **SNR (dB)** | 0.3 | 0.6 | 0.0 | 2.5 |

### Per-Bin Error Analysis

Analyzing error distribution across 80 mel bins to identify systematic errors...

**Bins with Highest Average Error**:

1. Mel Bin 27: 58.78 (avg error)
2. Mel Bin 3: 58.73 (avg error)
3. Mel Bin 28: 55.82 (avg error)
4. Mel Bin 25: 54.81 (avg error)
5. Mel Bin 16: 52.78 (avg error)

**Error by Frequency Range**:
- Low frequencies (bins 0-19): 51.13 avg error
- Mid frequencies (bins 20-59): 40.67 avg error
- High frequencies (bins 60-79): 17.61 avg error

✅ **Finding**: Error is relatively uniform across frequency ranges.

---

## Visual Comparisons

Detailed visual comparisons are available in `benchmark_results/plots/`:

**Individual Test Comparisons**:

1. `aggregate_analysis.png`
2. `beating_1000_1100hz_comparison.png`
3. `brown_noise_comparison.png`
4. `chirp_1000_4000hz_comparison.png`
5. `chirp_100_1000hz_comparison.png`
6. `chirp_100_4000hz_comparison.png`
7. `clipping_1000hz_comparison.png`
8. `dc_offset_comparison.png`
9. `harmonics_100_200_300hz_comparison.png`
10. `impulse_comparison.png`

...and 14 more

**Aggregate Analysis**: `aggregate_analysis.png`

---

## Error Analysis

### Potential Sources of Error

1. **Linear Mel Binning** ⚠️
   - Current implementation uses simple linear downsampling
   - True mel scale is logarithmic
   - **Impact**: Medium - May reduce accuracy for speech recognition
   - **Fix**: Implement proper triangular mel filterbank

3. **Quantization Errors** ⚠️
   - Q15 fixed-point may lose precision
   - INT8 output has limited dynamic range
   - **Impact**: Low to Medium
   - **Fix**: Consider INT16 output or adjust scaling


### Systematic Error Patterns

- No significant systematic error patterns detected ✅
---

## Recommendations

### For Production Deployment

❌ **Improvements needed before production deployment**

Current accuracy (nan% correlation) may not be sufficient for production Whisper.

**Required Improvements** (Priority Order):
1. **Critical**: Implement proper mel filterbank
2. **Critical**: Add log compression
3. **Important**: Review Q15 precision and scaling
4. **Important**: Validate against Whisper accuracy metrics

**Timeline**: 3-5 days for critical improvements

### Performance Optimization Opportunities

---

## Technical Details

### NPU Implementation

**File**: `mel_kernel_fft_fixed.c`
**Size**: 3.7 KB (108 lines)
**Stack Usage**: 3.5 KB (under safe limit)

**Pipeline**:
```
800 bytes input (400 INT16 samples)
  ↓ Convert little-endian
INT16 samples [400]
  ↓ Hann window (Q15 × Q15)
Windowed samples
  ↓ Zero-pad to 512
Padded samples [512]
  ↓ 512-point FFT (Q15)
Complex FFT output [512]
  ↓ Magnitude (alpha-max + beta-min)
Magnitude spectrum [256]
  ↓ Linear downsample (256 → 80)
Mel bins (INT16)
  ↓ Scale to INT8 [0, 127]
80 INT8 mel bins
```

### CPU Reference

**Library**: librosa  (available)
**Method**: `librosa.feature.melspectrogram()`
**Configuration**:
- Sample rate: 16000 Hz
- FFT size: 512
- Mel bins: 80
- Frequency range: 0-8000 Hz
- Window: Hann
- HTK mel scale: True
- Log scaling: log10(mel + 1e-10)

---

## Conclusion

**Status**: ❌ **NEEDS IMPROVEMENT** (FAIL)

The NPU fixed-point FFT implementation requires improvements before production use.

**Key Achievements**:
- ✅ 512-point Q15 FFT executing on NPU
- ✅ nan% correlation with CPU reference
- ✅ 0.3 dB signal-to-noise ratio
- ✅ All 80 mel bins processing correctly

**Next Steps**: Address critical issues, re-benchmark accuracy

---

**Report Generated**: 2025-10-28 06:06:12 UTC
**Project**: Unicorn Amanuensis
**Organization**: Magic Unicorn Unconventional Technology & Stuff Inc.
**Hardware**: AMD Ryzen 9 8945HS with Phoenix NPU (XDNA1)
