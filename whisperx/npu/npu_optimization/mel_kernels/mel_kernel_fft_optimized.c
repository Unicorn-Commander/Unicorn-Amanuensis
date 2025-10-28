// MEL Spectrogram Kernel - Optimized with Proper Mel Filterbank
// For AMD Phoenix NPU (AIE2) - Integer-only arithmetic
//
// IMPROVEMENTS OVER mel_kernel_fft_fixed.c:
//   - Proper triangular mel filters (80 filters)
//   - Log-spaced mel scale (HTK formula)
//   - Overlapping filters (~50% overlap)
//   - Matches Whisper/librosa expectations
//
// Pipeline:
//   800 bytes (400 INT16 samples)
//   → Hann window (Q15)
//   → Zero-pad to 512
//   → 512-point FFT (Q15)
//   → Magnitude spectrum (256 bins)
//   → Apply 80 triangular mel filters  ← NEW!
//   → Output as INT8

#include <stdint.h>
#include "mel_filterbank_coeffs.h"  // Auto-generated mel filters

// Forward declarations from fft_fixed_point.c
typedef struct {
    int16_t real;
    int16_t imag;
} complex_q15_t;

// External coefficients (from fft_coeffs_fixed.h)
extern const int16_t hann_window_q15[400];

// Fixed-point multiply helper (Q15 × Q15 → Q15)
static inline int16_t mul_q15(int16_t a, int16_t b) {
    int32_t product = (int32_t)a * (int32_t)b;
    return (int16_t)((product + (1 << 14)) >> 15);  // Round and scale
}

// Apply single mel filter to magnitude spectrum
// Returns mel energy in Q15 format
static inline int16_t apply_mel_filter_optimized(
    const int16_t* magnitude,
    const mel_filter_t* filter
) {
    int32_t energy = 0;  // Accumulate in Q30 (Q15 × Q15)

    // Apply left slope (rising edge: 0 → 1.0)
    for (uint16_t i = 0; i < filter->left_width; i++) {
        uint16_t bin_idx = filter->start_bin + i;
        int16_t mag = magnitude[bin_idx];
        int16_t weight = filter->left_slopes[i];

        // Q15 × Q15 = Q30
        energy += (int32_t)mag * (int32_t)weight;
    }

    // Apply right slope (falling edge: 1.0 → 0)
    for (uint16_t i = 0; i < filter->right_width; i++) {
        uint16_t bin_idx = filter->peak_bin + i;
        int16_t mag = magnitude[bin_idx];
        int16_t weight = filter->right_slopes[i];

        // Q15 × Q15 = Q30
        energy += (int32_t)mag * (int32_t)weight;
    }

    // Convert Q30 back to Q15 (divide by 2^15)
    // Add rounding: (energy + 2^14) >> 15
    int32_t rounded = (energy + (1 << 14)) >> 15;

    // Clamp to INT16 range
    if (rounded > 32767) rounded = 32767;
    if (rounded < -32768) rounded = -32768;

    return (int16_t)rounded;
}

// Kernel entry point
// Input: 800 bytes (400 INT16 samples, little-endian)
// Output: 80 INT8 mel bins
extern "C" {

// C functions from fft_fixed_point.c
void fft_radix2_512_fixed(int16_t* input, complex_q15_t* output);
void compute_magnitude_fixed(complex_q15_t* fft_output, int16_t* magnitude, uint32_t size);
void apply_hann_window_fixed(int16_t* samples, const int16_t* window, uint32_t size);
void zero_pad_to_512(int16_t* samples, uint32_t input_size);

void mel_kernel_simple(int8_t *input, int8_t *output) {
    // Working buffers (stack allocation)
    // Total: 1024 + 2048 + 512 = 3584 bytes (~3.5KB)

    int16_t samples[512];        // 1024 bytes - zero-padded audio
    complex_q15_t fft_out[512];  // 2048 bytes - FFT output
    int16_t magnitude[256];      // 512 bytes - magnitude spectrum

    // Step 1: Convert 800 bytes to 400 INT16 samples (little-endian)
    for (int i = 0; i < 400; i++) {
        int byte_idx = i * 2;
        samples[i] = ((int16_t)(uint8_t)input[byte_idx]) |
                    (((int16_t)(int8_t)input[byte_idx + 1]) << 8);
    }

    // Step 2: Apply Hann window (Q15 × Q15 → Q15)
    apply_hann_window_fixed(samples, hann_window_q15, 400);

    // Step 3: Zero-pad to 512 samples
    zero_pad_to_512(samples, 400);

    // Step 4: Compute 512-point FFT
    fft_radix2_512_fixed(samples, fft_out);

    // Step 5: Compute magnitude spectrum (first 256 bins only, due to symmetry)
    compute_magnitude_fixed(fft_out, magnitude, 256);

    // Step 6: Apply 80 triangular mel filters (NEW - proper mel filterbank!)
    for (int mel_bin = 0; mel_bin < NUM_MEL_FILTERS; mel_bin++) {
        const mel_filter_t* filter = &mel_filters[mel_bin];

        // Apply filter and get mel energy (Q15 format)
        int16_t mel_energy = apply_mel_filter_optimized(magnitude, filter);

        // Step 7: Convert from Q15 to INT8 range [0, 127]
        // Q15 max is 32767, scale to 127
        int32_t scaled = ((int32_t)mel_energy * 127) / 32767;

        // Optional: Apply log-like compression for better dynamic range
        // log2(x+1) approximation: x / (1 + x/K)
        // For now, use linear scaling (can add log later if needed)

        // Clamp to INT8 range [0, 127]
        if (scaled > 127) scaled = 127;
        if (scaled < 0) scaled = 0;

        output[mel_bin] = (int8_t)scaled;
    }
}

// Alternative: Output INT16 for higher precision (160 bytes output)
void mel_kernel_simple_int16(int8_t *input, int16_t *output) {
    // Working buffers
    int16_t samples[512];
    complex_q15_t fft_out[512];
    int16_t magnitude[256];

    // Same steps 1-5 as above
    for (int i = 0; i < 400; i++) {
        int byte_idx = i * 2;
        samples[i] = ((int16_t)(uint8_t)input[byte_idx]) |
                    (((int16_t)(int8_t)input[byte_idx + 1]) << 8);
    }

    apply_hann_window_fixed(samples, hann_window_q15, 400);
    zero_pad_to_512(samples, 400);
    fft_radix2_512_fixed(samples, fft_out);
    compute_magnitude_fixed(fft_out, magnitude, 256);

    // Step 6: Apply mel filters and output Q15 directly
    for (int mel_bin = 0; mel_bin < NUM_MEL_FILTERS; mel_bin++) {
        const mel_filter_t* filter = &mel_filters[mel_bin];
        output[mel_bin] = apply_mel_filter_optimized(magnitude, filter);
    }
}

} // extern "C"
