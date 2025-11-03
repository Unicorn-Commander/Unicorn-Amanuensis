// Test: Can NPU read from lookup tables?
// This will verify if bit_reverse_lut and twiddle_cos_q15 are accessible

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "fft_coeffs_fixed.h"

typedef struct {
    int16_t real;
    int16_t imag;
} complex_q15_t;

void mel_kernel_simple(int8_t *input_unused, int8_t *output) {
    // Test 1: Can we read from bit_reverse_lut?
    // bit_reverse_lut[0] should be 0
    // bit_reverse_lut[1] should be 256
    // bit_reverse_lut[256] should be 1
    output[0] = (int8_t)(bit_reverse_lut[0] & 0xFF);        // Should be 0
    output[1] = (int8_t)((bit_reverse_lut[1] >> 8) & 0xFF); // Should be 1 (256 >> 8)
    output[2] = (int8_t)(bit_reverse_lut[256] & 0xFF);      // Should be 1

    // Test 2: Can we read from twiddle_cos_q15?
    // twiddle_cos_q15[0] should be 32767 (cos(0) = 1.0)
    // twiddle_cos_q15[64] should be 0 (cos(π/2) = 0)
    output[3] = (int8_t)((twiddle_cos_q15[0] >> 8) & 0xFF);  // Should be 127 (32767 >> 8)
    output[4] = (int8_t)(twiddle_cos_q15[64] & 0xFF);        // Should be 0

    // Test 3: Can we read from twiddle_sin_q15?
    // twiddle_sin_q15[0] should be 0 (sin(0) = 0)
    // twiddle_sin_q15[64] should be -32767 (sin(π/2) = -1.0)
    output[5] = (int8_t)(twiddle_sin_q15[0] & 0xFF);         // Should be 0
    output[6] = (int8_t)((twiddle_sin_q15[64] >> 8) & 0xFF); // Should be 128 (-32767 >> 8 with wrap)

    // Test 4: Can we use lookup in a simple loop?
    int16_t sum = 0;
    for (int i = 0; i < 8; i++) {
        sum += twiddle_cos_q15[i * 32];  // Sample every 32nd value
    }
    output[7] = (int8_t)((sum >> 8) & 0xFF);  // Should be non-zero

    // Fill rest with pattern for verification
    for (int i = 8; i < 80; i++) {
        output[i] = (int8_t)(i * 2);
    }
}

#ifdef __cplusplus
}
#endif
