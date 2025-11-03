// Copy of mel_kernel_fft_fixed.c with CORRECTED INT8 scaling
// The issue: FFT scaling reduces magnitude by 512x, so mel_energy is much smaller than 32767
// Fix: Scale based on realistic mel_energy range, not theoretical Q15 maximum

#include <stdint.h>
#include <string.h>

// Include coefficient tables
#include "mel_coeffs_fixed.h"

// FFT size and constants
#define FFT_SIZE 512
#define MEL_BINS 80

// Complex number in Q15 fixed-point
typedef struct {
    int16_t real;
    int16_t imag;
} complex_q15_t;

// External functions
extern void fft_radix2_512_fixed(int16_t* input, complex_q15_t* output);
extern const int16_t hann_window_q15[400];

// Helper functions
void apply_hann_window_fixed(int16_t* samples, const int16_t* window, int size) {
    for (int i = 0; i < size; i++) {
        int32_t product = (int32_t)samples[i] * (int32_t)window[i];
        samples[i] = (int16_t)(product >> 15);
    }
}

void zero_pad_to_512(int16_t* samples, int original_size) {
    for (int i = original_size; i < 512; i++) {
        samples[i] = 0;
    }
}

void compute_magnitude(const complex_q15_t* fft_output, int16_t* magnitude, int size) {
    for (int i = 0; i < size; i++) {
        int32_t real_sq = (int32_t)fft_output[i].real * fft_output[i].real;
        int32_t imag_sq = (int32_t)fft_output[i].imag * fft_output[i].imag;
        int32_t mag_sq = (real_sq + imag_sq) >> 15;
        
        // Simple sqrt approximation for Q15
        magnitude[i] = (int16_t)(mag_sq >> 7);
    }
}

// FIXED mel filterbank application with CORRECTED scaling
void apply_mel_filters_q15(
    const int16_t* magnitude,
    int8_t* mel_output,
    uint32_t n_mels
) {
    for (uint32_t m = 0; m < n_mels; m++) {
        const mel_filter_q15_t* filter = &mel_filters_q15[m];
        int32_t mel_energy = 0;

        // Sparse loop: only non-zero weights
        for (int bin = filter->start_bin; bin < filter->end_bin; bin++) {
            if (filter->weights[bin] != 0) {
                // Q15 × Q15 = Q30
                int32_t weighted = (int32_t)magnitude[bin] * filter->weights[bin];
                mel_energy += weighted >> 15;
            }
        }

        // Clamp to prevent negative values
        if (mel_energy < 0) mel_energy = 0;

        // **FIXED SCALING**: Account for FFT scaling factor
        // After FFT with per-stage scaling, magnitudes are ~512x smaller
        // Realistic mel_energy range: [0, 64] instead of [0, 32767]
        // So we scale from [0, 64] to [0, 127] with headroom
        
        // Scale up by 16x to use more of INT8 range, then scale to [0, 127]
        int32_t scaled = (mel_energy * 16 * 127) / 1024;  // Adjusted scaling
        
        // Clamp to INT8 range [0, 127]
        if (scaled > 127) scaled = 127;
        if (scaled < 0) scaled = 0;

        mel_output[m] = (int8_t)scaled;
    }
}

void mel_kernel_simple(int8_t *input, int8_t *output) {
    int16_t samples[512];
    complex_q15_t fft_out[512];
    int16_t magnitude[256];

    // Step 1: Convert 800 bytes to 400 INT16 samples
    for (int i = 0; i < 400; i++) {
        int byte_idx = i * 2;
        samples[i] = ((int16_t)(uint8_t)input[byte_idx]) |
                    (((int16_t)(int8_t)input[byte_idx + 1]) << 8);
    }

    // Step 2: Apply Hann window
    apply_hann_window_fixed(samples, hann_window_q15, 400);

    // Step 3: Zero-pad to 512
    zero_pad_to_512(samples, 400);

    // Step 4: FFT (with per-stage scaling)
    fft_radix2_512_fixed(samples, fft_out);

    // Step 5: Compute magnitude
    compute_magnitude(fft_out, magnitude, 256);

    // Step 6: Apply mel filterbanks with FIXED scaling
    apply_mel_filters_q15(magnitude, output, MEL_BINS);
}
