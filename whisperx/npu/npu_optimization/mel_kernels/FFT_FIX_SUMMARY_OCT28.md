# FFT Fix Summary - October 28, 2025 Evening

## Problems Found

### 1. FFT Scaling Error (FIXED ✅)

**Problem**: No scaling in FFT butterfly operations causes 512x overflow
**Location**: `fft_fixed_point.c` lines 92-98
**Impact**: Massive magnitude errors, wrong peak bins, 0.44 correlation

**Root Cause**:
```c
// OLD CODE (BROKEN):
output[idx_even].real = even.real + t.real;  // No scaling!
output[idx_odd].real = even.real - t.real;
```

Each FFT stage doubles the magnitude. After 9 stages (512-point), output is 512x larger, causing INT16 overflow.

**Fix Applied**:
```c
// NEW CODE (FIXED):
int32_t sum_real = (int32_t)even.real + (int32_t)t.real;
int32_t diff_real = (int32_t)even.real - (int32_t)t.real;

// Scale down by 2 (with rounding) to prevent overflow
output[idx_even].real = (int16_t)((sum_real + 1) >> 1);
output[idx_odd].real = (int16_t)((diff_real + 1) >> 1);
```

**Result**:
- FFT correlation: 0.44 → **1.0000** ✅
- Peak bin: 480 (wrong) → **32 (correct)** ✅
- No overflow warnings ✅

---

### 2. Mel Filterbank Error (NOT FIXED ❌)

**Problem**: Uses simple linear binning instead of proper HTK mel-scale filterbanks
**Location**: `mel_kernel_fft_fixed.c` lines 67-105
**Impact**: Wrong mel bin energies, 4.68% correlation with librosa

**Current Code** (INCORRECT):
```c
// Step 6: Downsample 256 bins → 80 mel bins via averaging
// Simple approach: average ~3.2 bins per mel bin
for (int mel_bin = 0; mel_bin < 80; mel_bin++) {
    int start_bin = (mel_bin * 256) / 80;      // LINEAR binning ❌
    int end_bin = ((mel_bin + 1) * 256) / 80;

    // Accumulate energy from FFT bins
    for (int bin = start_bin; bin < end_bin; bin++) {
        energy += magnitude[bin];  // Simple averaging ❌
    }
}
```

**What's Wrong**:
1. Uses **linear** frequency binning instead of **mel-scale** (logarithmic)
2. Uses simple averaging instead of **triangular mel filters**
3. No proper HTK formula: `mel = 2595 * log10(1 + f/700)`
4. Whisper absolutely requires proper mel filterbanks!

**What's Needed**:
```c
// Proper mel filterbank (HTK formula with triangular filters)
for (int mel_bin = 0; mel_bin < 80; mel_bin++) {
    // Convert mel bin to frequency range
    float mel_low = mel_bin * (mel_high - mel_low) / n_mels + mel_low;
    float mel_high = (mel_bin + 1) * (mel_high - mel_low) / n_mels + mel_low;
    float mel_center = (mel_bin + 0.5) * (mel_high - mel_low) / n_mels + mel_low;

    // Convert mel to Hz using HTK formula
    float f_low = 700 * (pow(10, mel_low / 2595) - 1);
    float f_center = 700 * (pow(10, mel_center / 2595) - 1);
    float f_high = 700 * (pow(10, mel_high / 2595) - 1);

    // Convert Hz to FFT bin index
    int bin_low = (int)(f_low * n_fft / sr);
    int bin_center = (int)(f_center * n_fft / sr);
    int bin_high = (int)(f_high * n_fft / sr);

    // Apply triangular filter
    for (int bin = bin_low; bin < bin_high; bin++) {
        float weight = (bin < bin_center)
            ? (bin - bin_low) / (bin_center - bin_low)  // Rising edge
            : (bin_high - bin) / (bin_high - bin_center);  // Falling edge

        energy += magnitude[bin] * weight;
    }
}
```

---

## Test Results

### FFT-Only Test (test_fft_cpu.py):
- ✅ DC signal: Correct bin 0
- ✅ 1000 Hz sine: Correct peak bin 32
- ✅ Correlation: **1.0000**
- ✅ No overflow

### Complete Mel Pipeline (test_mel_with_fixed_fft.py):
- ❌ Correlation: **-0.0275** (still broken)
- ❌ Peak mel bin: 10 vs expected 28
- **Reason**: Linear binning instead of proper mel filterbanks

---

## Files Modified

1. **fft_fixed_point.c** - Added FFT scaling fix (lines 92-104)
2. **test_fft_cpu.py** - Python test showing FFT fix works
3. **test_mel_with_fixed_fft.py** - Python test showing mel filterbanks still broken

---

## Next Steps

### Week 1, Day 2-3: Fix Mel Filterbanks

1. **Create mel filter coefficient tables** (Q15 fixed-point)
   - Precompute 80 triangular mel filters
   - Store as lookup table in `mel_coeffs_fixed.h`
   - Use HTK formula for mel scale

2. **Update mel_kernel_fft_fixed.c**
   - Replace linear binning (lines 67-105)
   - Use proper mel filterbank multiplication
   - Maintain Q15 fixed-point arithmetic

3. **Test accuracy**
   - Target: >95% correlation with librosa
   - Verify peak mel bins match

### Week 1, Day 4: Rebuild NPU Kernel

Once both fixes are complete:
1. Recompile C kernels
2. Rebuild XCLBINs
3. Test on actual NPU hardware
4. Validate with existing benchmark suite

---

## Summary

**Fixed**:
- ✅ FFT overflow (correlation 0.44 → 1.00)

**Still Broken**:
- ❌ Mel filterbanks (using linear binning)

**Total Effort**:
- FFT fix: 2 hours (complete)
- Mel fix: Est. 4-6 hours (pending)
- Testing: 2 hours
- **Total: ~8-10 hours to 95% accuracy**

**Confidence**: Very high. FFT fix works perfectly. Mel filterbank fix is well-understood.

---

**Status**: FFT computation fixed, mel filterbank implementation needed
**Next**: Implement proper HTK mel-scale triangular filters in Q15 fixed-point
**Timeline**: Week 1 Day 2-4 (per MASTER_CHECKLIST)

Magic Unicorn Unconventional Technology & Stuff Inc.
