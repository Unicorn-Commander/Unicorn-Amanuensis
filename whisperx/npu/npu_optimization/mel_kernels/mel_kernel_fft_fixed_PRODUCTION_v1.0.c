// Mel kernel with CORRECTED INT8 scaling + extern "C" linkage

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <string.h>
#include "mel_coeffs_fixed.h"

#define FFT_SIZE 512
#define MEL_BINS 80

typedef struct {
    int16_t real;
    int16_t imag;
} complex_q15_t;

extern void fft_radix2_512_fixed(int16_t* input, complex_q15_t* output);
extern const int16_t hann_window_q15[400];

static void apply_hann_window_fixed(int16_t* samples, const int16_t* window, int size) {
    for (int i = 0; i < size; i++) {
        int32_t product = (int32_t)samples[i] * (int32_t)window[i];
        samples[i] = (int16_t)(product >> 15);
    }
}

static void zero_pad_to_512(int16_t* samples, int original_size) {
    for (int i = original_size; i < 512; i++) {
        samples[i] = 0;
    }
}

static void compute_magnitude(const complex_q15_t* fft_output, int16_t* magnitude, int size) {
    for (int i = 0; i < size; i++) {
        int32_t real_sq = (int32_t)fft_output[i].real * fft_output[i].real;
        int32_t imag_sq = (int32_t)fft_output[i].imag * fft_output[i].imag;
        // FIXED: FFT already scaled down by 512, so don't scale again!
        // Just compute magnitude squared directly
        int32_t mag_sq = real_sq + imag_sq;
        // Clip to int16_t range
        if (mag_sq > 32767) mag_sq = 32767;
        magnitude[i] = (int16_t)mag_sq;
    }
}

// Fast integer square root approximation
static uint16_t isqrt(uint32_t n) {
    if (n == 0) return 0;
    uint32_t x = n;
    uint32_t y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return (uint16_t)x;
}

static void apply_mel_filters_q15(const int16_t* magnitude, int8_t* mel_output, uint32_t n_mels) {
    for (uint32_t m = 0; m < n_mels; m++) {
        const mel_filter_q15_t* filter = &mel_filters_q15[m];
        int32_t mel_energy = 0;

        for (int bin = filter->start_bin; bin < filter->end_bin; bin++) {
            if (filter->weights[bin] != 0) {
                int32_t weighted = (int32_t)magnitude[bin] * filter->weights[bin];
                mel_energy += weighted >> 15;  // Scale down Q15 weights
            }
        }

        if (mel_energy < 0) mel_energy = 0;

        // Take square root to compress dynamic range (similar to dB scale)
        uint16_t mel_sqrt = isqrt((uint32_t)mel_energy);

        // Scale to [0, 127] - sqrt range is roughly [0, 80]
        int32_t scaled = (mel_sqrt * 127) / 80;

        if (scaled > 127) scaled = 127;
        if (scaled < 0) scaled = 0;

        mel_output[m] = (int8_t)scaled;
    }
}

void mel_kernel_simple(int8_t *input, int8_t *output) {
    int16_t samples[512];
    complex_q15_t fft_out[512];
    int16_t magnitude[256];

    for (int i = 0; i < 400; i++) {
        int byte_idx = i * 2;
        samples[i] = ((int16_t)(uint8_t)input[byte_idx]) |
                    (((int16_t)(int8_t)input[byte_idx + 1]) << 8);
    }

    apply_hann_window_fixed(samples, hann_window_q15, 400);
    zero_pad_to_512(samples, 400);
    fft_radix2_512_fixed(samples, fft_out);
    compute_magnitude(fft_out, magnitude, 256);
    apply_mel_filters_q15(magnitude, output, MEL_BINS);
}

#ifdef __cplusplus
}
#endif
