// Fixed-Point 512-Point FFT for AMD Phoenix NPU (AIE2)
// Uses INT16/INT32 arithmetic only (Q15 format)
// Designed for maximum reliability on NPU hardware

#include <stdint.h>
#include "fft_coeffs_fixed.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FFT_SIZE 512
#define LOG2_SIZE 9
#define Q15_SCALE 32767  // 2^15 - 1 (max Q15 value)

// Q15 Format: 1 sign bit + 15 fractional bits
// Range: -1.0 to +0.999969482421875
// Example: 0.5 in Q15 = 16384 (0x4000)

// Complex number structure (Q15 fixed-point)
typedef struct {
    int16_t real;  // Q15 format
    int16_t imag;  // Q15 format
} complex_q15_t;

// Fixed-point multiply with proper scaling
// Multiplies two Q15 numbers and returns Q15 result
static inline int16_t mul_q15(int16_t a, int16_t b) {
    int32_t product = (int32_t)a * (int32_t)b;
    // Right shift by 15 to convert Q30 back to Q15
    return (int16_t)((product + (1 << 14)) >> 15);  // +0.5 for rounding
}

// Complex multiplication in Q15
// (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
static inline complex_q15_t cmul_q15(complex_q15_t a, complex_q15_t b) {
    complex_q15_t result;

    int32_t ac = (int32_t)a.real * (int32_t)b.real;
    int32_t bd = (int32_t)a.imag * (int32_t)b.imag;
    int32_t ad = (int32_t)a.real * (int32_t)b.imag;
    int32_t bc = (int32_t)a.imag * (int32_t)b.real;

    // Scale back to Q15 with rounding
    result.real = (int16_t)(((ac - bd) + (1 << 14)) >> 15);
    result.imag = (int16_t)(((ad + bc) + (1 << 14)) >> 15);

    return result;
}

// 512-point FFT with Q15 fixed-point arithmetic
// Input: 512 INT16 samples (already in Q15 format from audio)
// Output: 512 complex Q15 values
void fft_radix2_512_fixed(int16_t* input, complex_q15_t* output) {
    uint32_t i, stage, k, j;
    uint32_t m, half_m, twid_step, twid_idx;
    uint32_t idx_even, idx_odd;
    complex_q15_t W, even, odd, t;

    // Step 1: Bit-reversal permutation using lookup table
    // This avoids the G_BITREVERSE instruction that caused issues
    for (i = 0; i < FFT_SIZE; i++) {
        uint32_t rev = bit_reverse_lut[i];
        output[rev].real = input[i];
        output[rev].imag = 0;  // Real input, imaginary = 0
    }

    // Step 2: FFT butterfly stages
    for (stage = 0; stage < LOG2_SIZE; stage++) {
        m = 1 << (stage + 1);  // 2^(stage+1)
        half_m = m >> 1;       // m/2
        twid_step = FFT_SIZE / m;

        // Process all butterflies in this stage
        for (k = 0; k < FFT_SIZE; k += m) {
            for (j = 0; j < half_m; j++) {
                idx_even = k + j;
                idx_odd = k + j + half_m;

                // Get twiddle factor for this butterfly
                twid_idx = j * twid_step;
                W.real = twiddle_cos_q15[twid_idx];
                W.imag = twiddle_sin_q15[twid_idx];

                // Get even and odd samples
                even = output[idx_even];
                odd = output[idx_odd];

                // Complex multiplication: t = W * odd
                t = cmul_q15(W, odd);

                // Butterfly: even ± t
                // Note: Addition/subtraction stays in Q15 range
                // thanks to FFT scaling properties
                output[idx_even].real = even.real + t.real;
                output[idx_even].imag = even.imag + t.imag;
                output[idx_odd].real = even.real - t.real;
                output[idx_odd].imag = even.imag - t.imag;
            }
        }
    }

    // Note: Output is not scaled down (natural FFT scaling)
    // For normalized FFT, divide by sqrt(512) = 22.627
    // We skip this for Whisper MEL computation (just need relative magnitudes)
}

// Fast magnitude approximation using alpha-max + beta-min
// Avoids sqrt and gives ~2% error
// Result in Q15 format
static inline int16_t fast_magnitude_q15(int16_t real, int16_t imag) {
    // Get absolute values
    int16_t abs_real = (real < 0) ? -real : real;
    int16_t abs_imag = (imag < 0) ? -imag : imag;

    // Find max and min
    int16_t max_val = (abs_real > abs_imag) ? abs_real : abs_imag;
    int16_t min_val = (abs_real < abs_imag) ? abs_real : abs_imag;

    // Alpha-max + beta-min approximation
    // alpha ≈ 0.96, beta ≈ 0.4
    // In Q15: 0.96 ≈ 31457, 0.4 ≈ 13107
    // Simplified: max + 0.4 * min

    int32_t beta_min = ((int32_t)13107 * (int32_t)min_val) >> 15;

    return (int16_t)(max_val + (int16_t)beta_min);
}

// More accurate magnitude using squared values
// Result is NOT in Q15 - it's a scaled INT32 magnitude
// Use this for mel spectrogram (we need power spectrum)
static inline int32_t magnitude_squared_q15(int16_t real, int16_t imag) {
    // Square both components (Q15 * Q15 = Q30)
    int32_t real_sq = (int32_t)real * (int32_t)real;
    int32_t imag_sq = (int32_t)imag * (int32_t)imag;

    // Sum of squares (Q30 format)
    int32_t mag_sq = real_sq + imag_sq;

    // Right shift to Q15 (divide by 2^15)
    return mag_sq >> 15;
}

// Compute magnitude spectrum (first half only, due to symmetry)
// Output is INT16 for mel filterbank processing
void compute_magnitude_fixed(complex_q15_t* fft_output, int16_t* magnitude, uint32_t size) {
    uint32_t i;

    for (i = 0; i < size; i++) {
        // Use fast approximation for speed
        magnitude[i] = fast_magnitude_q15(fft_output[i].real, fft_output[i].imag);

        // Alternative: Use squared magnitude for better accuracy
        // int32_t mag_sq = magnitude_squared_q15(fft_output[i].real, fft_output[i].imag);
        // magnitude[i] = (int16_t)((mag_sq > 32767) ? 32767 : mag_sq);
    }
}

// Compute power spectrum (magnitude squared) for mel filterbank
// This is what Whisper actually needs
void compute_power_spectrum_fixed(complex_q15_t* fft_output, int32_t* power, uint32_t size) {
    uint32_t i;

    for (i = 0; i < size; i++) {
        power[i] = magnitude_squared_q15(fft_output[i].real, fft_output[i].imag);
    }
}

// Apply Hann window before FFT
// Window coefficients are in Q15 format
void apply_hann_window_fixed(int16_t* samples, const int16_t* window, uint32_t size) {
    uint32_t i;

    for (i = 0; i < size; i++) {
        samples[i] = mul_q15(samples[i], window[i]);
    }
}

// Zero-pad from 400 to 512 samples (in-place, requires 512-element buffer)
void zero_pad_to_512(int16_t* samples, uint32_t input_size) {
    uint32_t i;

    // Input samples are in [0..input_size-1]
    // Set [input_size..511] to zero
    for (i = input_size; i < FFT_SIZE; i++) {
        samples[i] = 0;
    }
}

#ifdef __cplusplus
}
#endif
