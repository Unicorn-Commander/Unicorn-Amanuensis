// Basic Radix-2 FFT Implementation for AIE2
// 512-point FFT for Whisper mel spectrogram
// Reference: Cooley-Tukey FFT algorithm

#include <stdint.h>
#include <math.h>

#define FFT_SIZE 512
#define PI 3.14159265358979323846

// Complex number structure
typedef struct {
    float real;
    float imag;
} complex_t;

// Bit-reversal permutation for FFT
static inline uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t reversed = 0;
    for (uint32_t i = 0; i < log2n; i++) {
        reversed = (reversed << 1) | (x & 1);
        x >>= 1;
    }
    return reversed;
}

// Cooley-Tukey Radix-2 Decimation-in-Time FFT
// Input: time-domain samples (real values from windowed audio)
// Output: frequency-domain complex values
void fft_radix2_512(int16_t* input, complex_t* output) {
    const uint32_t log2n = 9;  // log2(512) = 9
    const uint32_t n = FFT_SIZE;

    // Step 1: Bit-reversal permutation
    // Reorder input samples for in-place FFT
    for (uint32_t i = 0; i < n; i++) {
        uint32_t rev = bit_reverse(i, log2n);
        if (i < rev) {
            // Swap elements
            int16_t temp = input[i];
            input[i] = input[rev];
            input[rev] = temp;
        }
    }

    // Initialize output with input values (convert int16 to complex)
    for (uint32_t i = 0; i < n; i++) {
        output[i].real = (float)input[i];
        output[i].imag = 0.0f;
    }

    // Step 2: FFT butterfly operations
    // Process log2(n) stages
    for (uint32_t stage = 0; stage < log2n; stage++) {
        uint32_t m = 1 << (stage + 1);  // 2^(stage+1)
        uint32_t half_m = m >> 1;       // m/2

        // Twiddle factor angle increment
        float theta = -2.0f * PI / (float)m;

        // Process all butterflies in this stage
        for (uint32_t k = 0; k < n; k += m) {
            for (uint32_t j = 0; j < half_m; j++) {
                // Twiddle factor W_m^j = e^(-2πij/m)
                float angle = theta * (float)j;
                complex_t w;
                w.real = cosf(angle);
                w.imag = sinf(angle);

                // Butterfly operation indices
                uint32_t idx_even = k + j;
                uint32_t idx_odd = k + j + half_m;

                // Get even and odd samples
                complex_t even = output[idx_even];
                complex_t odd = output[idx_odd];

                // Complex multiplication: t = W * odd
                complex_t t;
                t.real = w.real * odd.real - w.imag * odd.imag;
                t.imag = w.real * odd.imag + w.imag * odd.real;

                // Butterfly: even + t, even - t
                output[idx_even].real = even.real + t.real;
                output[idx_even].imag = even.imag + t.imag;
                output[idx_odd].real = even.real - t.real;
                output[idx_odd].imag = even.imag - t.imag;
            }
        }
    }
}

// Compute magnitude spectrum (for Phase 2.1)
// Output: magnitude of each frequency bin
void compute_magnitude(complex_t* fft_output, int32_t* magnitude, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        float real = fft_output[i].real;
        float imag = fft_output[i].imag;

        // Magnitude = sqrt(real^2 + imag^2)
        // For now, use squared magnitude to avoid sqrt (optimization)
        float mag_sq = real * real + imag * imag;

        // Convert to int32 (scaling for fixed-point representation)
        magnitude[i] = (int32_t)(mag_sq * 1000.0f);
    }
}
