// Real 512-point FFT Implementation with Twiddle Factors
// Optimized for AIE2 NPU - Uses precomputed coefficients

#include <stdint.h>
#include "fft_coeffs.h"

// Fast inverse square root approximation (Quake III algorithm)
static inline float fast_inv_sqrt(float number) {
    union {
        float f;
        uint32_t i;
    } conv = { .f = number };
    conv.i = 0x5f3759df - (conv.i >> 1);
    conv.f *= 1.5f - (number * 0.5f * conv.f * conv.f);
    return conv.f;
}

// Fast square root using inverse square root
static inline float fast_sqrt(float number) {
    return number * fast_inv_sqrt(number);
}

#define FFT_SIZE 512
#define LOG2_SIZE 9

// Bit-reversal permutation
static inline uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t reversed = 0;
    for (uint32_t i = 0; i < log2n; i++) {
        reversed = (reversed << 1) | (x & 1);
        x >>= 1;
    }
    return reversed;
}

// 512-point FFT with real twiddle factors
void fft_radix2_512_real(int16_t* input, complex_t* output) {
    // Step 1: Bit-reversal permutation and convert to complex
    for (uint32_t i = 0; i < FFT_SIZE; i++) {
        uint32_t rev = bit_reverse(i, LOG2_SIZE);
        output[rev].real = (float)input[i];
        output[rev].imag = 0.0f;
    }

    // Step 2: FFT butterfly stages with twiddle factors
    for (uint32_t stage = 0; stage < LOG2_SIZE; stage++) {
        uint32_t m = 1 << (stage + 1);  // 2^(stage+1)
        uint32_t half_m = m >> 1;       // m/2

        // Twiddle factor step for this stage
        uint32_t twid_step = FFT_SIZE / m;

        // Process all butterflies in this stage
        for (uint32_t k = 0; k < FFT_SIZE; k += m) {
            for (uint32_t j = 0; j < half_m; j++) {
                uint32_t idx_even = k + j;
                uint32_t idx_odd = k + j + half_m;

                // Get twiddle factor for this butterfly
                uint32_t twid_idx = j * twid_step;
                complex_t W = twiddle_factors[twid_idx];

                // Get even and odd samples
                complex_t even = output[idx_even];
                complex_t odd = output[idx_odd];

                // Complex multiplication: t = W * odd
                complex_t t;
                t.real = W.real * odd.real - W.imag * odd.imag;
                t.imag = W.real * odd.imag + W.imag * odd.real;

                // Butterfly: even ± t
                output[idx_even].real = even.real + t.real;
                output[idx_even].imag = even.imag + t.imag;
                output[idx_odd].real = even.real - t.real;
                output[idx_odd].imag = even.imag - t.imag;
            }
        }
    }
}

// Compute magnitude spectrum (only first half needed due to symmetry)
void compute_magnitude_real(complex_t* fft_output, float* magnitude, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        float real = fft_output[i].real;
        float imag = fft_output[i].imag;

        // Magnitude = sqrt(real^2 + imag^2)
        // Use fast sqrt approximation for AIE2 (no math.h dependency)
        float mag_squared = real * real + imag * imag;
        magnitude[i] = fast_sqrt(mag_squared);
    }
}
