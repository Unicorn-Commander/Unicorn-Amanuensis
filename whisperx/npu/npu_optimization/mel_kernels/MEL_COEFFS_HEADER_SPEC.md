# mel_coeffs_fixed.h Header Specification

**Purpose**: Define HTK mel filterbank coefficients for Whisper mel spectrogram computation
**Format**: Q15 fixed-point (int16_t)
**Target**: AMD Phoenix NPU (AIE2) - no floating point

---

## Required Structure Definition

```c
#ifndef MEL_COEFFS_FIXED_H
#define MEL_COEFFS_FIXED_H

#include <stdint.h>

// Mel filter structure (Q15 fixed-point)
// Each filter is a triangular window over a range of FFT bins
typedef struct {
    int16_t start_bin;      // First FFT bin in filter's range [0-255]
    int16_t end_bin;        // Last FFT bin (exclusive) [1-256]
    int16_t num_weights;    // Number of weight coefficients
    int16_t weights[32];    // Triangular filter weights (Q15 format)
                            // Array size 32 assumes max ~30 FFT bins per filter
} mel_filter_q15_t;

// Array of 80 mel filters for Whisper
// Covers frequency range 0-8000 Hz (16 kHz sample rate)
// Filters overlap by 50% (standard HTK)
extern const mel_filter_q15_t mel_filters_q15[80];

#endif // MEL_COEFFS_FIXED_H
```

---

## Whisper Mel Parameters

**Must match these exact specifications**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Number of mel bins | 80 | Whisper standard |
| Sample rate | 16000 Hz | Whisper audio input |
| FFT size | 512 | From fft_fixed_point.c |
| FFT bins used | 256 | First half (0-8000 Hz) |
| Min frequency | 0 Hz | DC component |
| Max frequency | 8000 Hz | Nyquist (sr/2) |
| Mel scale | HTK | Use HTK formula, not Slaney |
| Filter shape | Triangular | 50% overlap |
| Normalization | Area = 1.0 | Sum of weights per filter |

---

## HTK Mel Scale Formula

**Frequency to Mel**:
```
mel = 2595 * log10(1 + freq / 700)
```

**Mel to Frequency**:
```
freq = 700 * (10^(mel / 2595) - 1)
```

**In C (for reference)**:
```c
// Convert Hz to mel scale (HTK formula)
double hz_to_mel_htk(double freq_hz) {
    return 2595.0 * log10(1.0 + freq_hz / 700.0);
}

// Convert mel to Hz (HTK formula)
double mel_to_hz_htk(double mel) {
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}
```

---

## Filter Generation Algorithm

### Step 1: Compute Mel Bin Edges
```python
import numpy as np

def mel_filterbank_centers_htk(n_mels=80, sr=16000, n_fft=512):
    """Compute mel bin edges using HTK formula."""
    # Frequency range
    fmin = 0.0
    fmax = sr / 2.0  # 8000 Hz

    # Convert to mel scale
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)

    # Create n_mels + 2 points (includes edges)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

    # Convert back to Hz
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    # Convert Hz to FFT bin indices
    fft_bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    return fft_bins, hz_points
```

### Step 2: Generate Triangular Filters
```python
def generate_mel_filter_q15(start_hz, center_hz, end_hz, sr=16000, n_fft=512):
    """Generate a single triangular mel filter in Q15 format."""
    # Convert Hz to FFT bin indices
    start_bin = int(np.floor((n_fft + 1) * start_hz / sr))
    center_bin = int(np.floor((n_fft + 1) * center_hz / sr))
    end_bin = int(np.floor((n_fft + 1) * end_hz / sr))

    # Ensure valid range
    start_bin = max(0, min(start_bin, 255))
    center_bin = max(start_bin + 1, min(center_bin, 255))
    end_bin = max(center_bin + 1, min(end_bin, 256))

    num_bins = end_bin - start_bin
    weights_q15 = []

    for bin_idx in range(start_bin, end_bin):
        if bin_idx < center_bin:
            # Ascending slope: 0.0 to 1.0
            slope = (bin_idx - start_bin) / (center_bin - start_bin)
        else:
            # Descending slope: 1.0 to 0.0
            slope = (end_bin - 1 - bin_idx) / (end_bin - 1 - center_bin)

        # Convert to Q15 (range 0-32767)
        weight_q15 = int(slope * 32767)
        weight_q15 = max(0, min(32767, weight_q15))  # Clamp
        weights_q15.append(weight_q15)

    return {
        'start_bin': start_bin,
        'end_bin': end_bin,
        'center_bin': center_bin,
        'num_weights': len(weights_q15),
        'weights': weights_q15
    }
```

### Step 3: Generate All 80 Filters
```python
def generate_mel_filterbank_htk(n_mels=80, sr=16000, n_fft=512):
    """Generate all 80 mel filters for Whisper."""
    fft_bins, hz_points = mel_filterbank_centers_htk(n_mels, sr, n_fft)

    filters = []
    for i in range(n_mels):
        # Each filter is a triangle from edge[i] to edge[i+2]
        # with peak at edge[i+1]
        start_hz = hz_points[i]
        center_hz = hz_points[i + 1]
        end_hz = hz_points[i + 2]

        filter_data = generate_mel_filter_q15(start_hz, center_hz, end_hz, sr, n_fft)
        filters.append(filter_data)

    return filters
```

---

## Example Filter Output

**Filter #0** (Lowest frequency, ~0-100 Hz):
```c
{
    .start_bin = 0,
    .end_bin = 4,
    .num_weights = 4,
    .weights = {0, 10922, 21845, 32767}  // Ascending triangle
}
```

**Filter #40** (Mid-range, ~2000 Hz):
```c
{
    .start_bin = 85,
    .end_bin = 105,
    .num_weights = 20,
    .weights = {
        0, 3276, 6553, 9830, 13107, 16384, 19661, 22938, 26214, 29491,  // Ascending
        32767,                                                            // Peak
        29491, 26214, 22938, 19661, 16384, 13107, 9830, 6553, 3276      // Descending
    }
}
```

**Filter #79** (Highest frequency, ~7500-8000 Hz):
```c
{
    .start_bin = 248,
    .end_bin = 256,
    .num_weights = 8,
    .weights = {0, 4681, 9362, 14043, 18725, 23406, 28087, 32767}
}
```

---

## C Header File Template

```c
// mel_coeffs_fixed.h
// HTK Mel Filterbank Coefficients for Whisper (Q15 Fixed-Point)
// Auto-generated - do not edit manually

#ifndef MEL_COEFFS_FIXED_H
#define MEL_COEFFS_FIXED_H

#include <stdint.h>

// Mel filter structure (Q15 fixed-point)
typedef struct {
    int16_t start_bin;      // First FFT bin [0-255]
    int16_t end_bin;        // Last FFT bin (exclusive) [1-256]
    int16_t num_weights;    // Number of coefficients
    int16_t weights[32];    // Triangular weights (Q15)
} mel_filter_q15_t;

// 80 mel filters for Whisper (16 kHz, 0-8000 Hz)
const mel_filter_q15_t mel_filters_q15[80] = {
    // Filter 0: 0.0 Hz - 100.0 Hz (center: 50.0 Hz)
    {
        .start_bin = 0,
        .end_bin = 4,
        .num_weights = 4,
        .weights = {0, 10922, 21845, 32767, 0, 0, ...}  // Pad to 32
    },

    // Filter 1: 50.0 Hz - 150.0 Hz (center: 100.0 Hz)
    {
        .start_bin = 2,
        .end_bin = 6,
        .num_weights = 4,
        .weights = {0, 10922, 21845, 32767, 0, 0, ...}
    },

    // ... (78 more filters)

    // Filter 79: 7500.0 Hz - 8000.0 Hz (center: 7750.0 Hz)
    {
        .start_bin = 248,
        .end_bin = 256,
        .num_weights = 8,
        .weights = {0, 4681, 9362, 14043, 18725, 23406, 28087, 32767, 0, ...}
    }
};

#endif // MEL_COEFFS_FIXED_H
```

---

## Validation Checks

### 1. Filter Coverage
```python
# All FFT bins should be covered by at least one filter
covered_bins = set()
for filter in filters:
    covered_bins.update(range(filter['start_bin'], filter['end_bin']))

assert len(covered_bins) == 256, "Not all FFT bins covered"
```

### 2. Overlap
```python
# Adjacent filters should overlap
for i in range(len(filters) - 1):
    assert filters[i]['end_bin'] > filters[i+1]['start_bin'], \
        f"Filters {i} and {i+1} do not overlap"
```

### 3. Weight Range
```python
# All weights should be in valid Q15 range [0, 32767]
for filter in filters:
    for weight in filter['weights'][:filter['num_weights']]:
        assert 0 <= weight <= 32767, f"Weight {weight} out of Q15 range"
```

### 4. Filter Monotonicity
```python
# Filters should be in ascending frequency order
for i in range(len(filters) - 1):
    assert filters[i]['start_bin'] <= filters[i+1]['start_bin'], \
        f"Filters {i} and {i+1} not in ascending order"
```

---

## Memory Requirements

**Total Size**:
```
sizeof(mel_filter_q15_t) = 2 + 2 + 2 + (32 × 2) = 70 bytes
80 filters × 70 bytes = 5,600 bytes (~5.5 KB)
```

**Placement**:
- Should be in `.rodata` section (read-only)
- NPU can DMA from DDR if needed
- Consider placing in tile memory for performance

---

## Integration Test

```python
import numpy as np
import librosa

# Generate reference mel filterbank
ref_mel_fb = librosa.filters.mel(
    sr=16000,
    n_fft=512,
    n_mels=80,
    fmin=0.0,
    fmax=8000.0,
    htk=True,  # CRITICAL: Must use HTK formula
    norm='slaney'  # Standard normalization
)

# Compare with generated coefficients
for i in range(80):
    filter = mel_filters_q15[i]

    # Extract weights
    npu_weights = np.array(filter['weights'][:filter['num_weights']]) / 32767.0

    # Extract reference weights
    ref_weights = ref_mel_fb[i, filter['start_bin']:filter['end_bin']]

    # Normalize
    npu_weights = npu_weights / npu_weights.sum()
    ref_weights = ref_weights / ref_weights.sum()

    # Compute correlation
    correlation = np.corrcoef(npu_weights, ref_weights)[0, 1]

    # Should be > 0.99 for each filter
    print(f"Filter {i}: correlation = {correlation:.4f}")
    assert correlation > 0.99, f"Filter {i} correlation too low"
```

---

## Common Pitfalls

### 1. Using Slaney Instead of HTK
- Whisper uses HTK formula
- Slaney has different mel conversion
- **Solution**: Ensure `htk=True` in librosa

### 2. Wrong Frequency Range
- Must be 0-8000 Hz (not 0-8000 mel)
- Sample rate must be 16000 Hz
- **Solution**: Verify fmin=0, fmax=8000

### 3. Incorrect Normalization
- Filters must sum to 1.0 (area normalization)
- Q15 scaling must preserve this
- **Solution**: Normalize before Q15 conversion

### 4. Array Size Too Small
- Some filters may span 30+ FFT bins at high frequencies
- Array size of 32 is safe but verify
- **Solution**: Check max num_weights across all filters

### 5. Off-by-One Errors
- end_bin is EXCLUSIVE (like Python ranges)
- FFT bins are 0-255 (256 total)
- **Solution**: Careful with loop bounds

---

## Performance Considerations

**Memory Access Pattern**:
- Sequential reads of mel_filters_q15[] array
- Random reads of magnitude[] array (within filter range)
- Sequential writes to mel_output[] array

**Optimization Opportunities**:
- Prefetch mel_filters_q15[m+1] while processing m
- Use SIMD for weight × magnitude multiplications
- Unroll inner loop if filter size is fixed

**Expected Execution Time**:
- 80 filters × 12 bins/filter × 3 ops/bin = ~2880 operations
- On NPU: ~100-200 cycles total
- Estimated: 50-100 µs

---

## Questions for Coefficient Generator

1. **What is the maximum num_weights across all 80 filters?**
   - If > 32, increase weights[] array size
   - Current assumption: max ~30 bins per filter

2. **Are filters normalized (sum to 1.0)?**
   - Important for energy preservation
   - Should match librosa default

3. **What Q15 conversion method is used?**
   - Simple rounding: `int(weight * 32767)`
   - Or banker's rounding for accuracy?

4. **Are zero weights included or skipped?**
   - Current implementation expects all weights[0..num_weights-1]
   - Including leading/trailing zeros is OK

---

## Deliverables Checklist

- [ ] `mel_coeffs_fixed.h` file created
- [ ] Contains `mel_filter_q15_t` structure definition
- [ ] Contains `mel_filters_q15[80]` array
- [ ] All 80 filters defined with correct parameters
- [ ] Weights in Q15 format (0-32767)
- [ ] HTK mel scale formula used
- [ ] Frequency range 0-8000 Hz
- [ ] Sample rate 16000 Hz
- [ ] Filters validated against librosa (correlation > 0.99)
- [ ] Header guards present
- [ ] Documentation comments included

---

**Generated**: October 28, 2025
**For**: mel_kernel_fft_fixed.c integration
**Status**: Specification complete, awaiting implementation
