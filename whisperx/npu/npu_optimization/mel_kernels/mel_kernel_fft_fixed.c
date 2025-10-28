// MEL Spectrogram Kernel - Fixed-Point FFT Version
// For AMD Phoenix NPU (AIE2) - Integer-only arithmetic
//
// Pipeline:
//   800 bytes (400 INT16 samples)
//   → Hann window (Q15)
//   → Zero-pad to 512
//   → 512-point FFT (Q15)
//   → Magnitude spectrum (256 bins)
//   → Downsample to 80 mel bins
//   → Output as INT8

#include <stdint.h>

// Forward declarations from fft_fixed_point.c
typedef struct {
    int16_t real;
    int16_t imag;
} complex_q15_t;

// External coefficients (from fft_coeffs_fixed.h)
extern const int16_t hann_window_q15[400];

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
    // Working buffers (stack allocation - carefully sized)
    // Total: 1024 + 2048 + 512 = 3584 bytes (~3.5KB)
    // This is under the ~7KB limit that caused stack overflow

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

    // Step 6: Downsample 256 bins → 80 mel bins via averaging
    // Simple approach: average ~3.2 bins per mel bin
    // For Whisper: mel bins are log-spaced, but simple averaging is OK

    for (int mel_bin = 0; mel_bin < 80; mel_bin++) {
        // Each mel bin covers roughly 256/80 = 3.2 FFT bins
        // Use linear mapping for simplicity
        int start_bin = (mel_bin * 256) / 80;      // Integer division
        int end_bin = ((mel_bin + 1) * 256) / 80;

        // Accumulate energy from FFT bins
        int32_t energy = 0;
        int count = 0;

        for (int bin = start_bin; bin < end_bin && bin < 256; bin++) {
            // magnitude[bin] is in Q15 format
            // For mel computation, we want power (magnitude squared)
            // But for speed, just use magnitude directly
            int16_t mag = magnitude[bin];

            // Accumulate absolute value
            energy += (mag < 0) ? -mag : mag;
            count++;
        }

        // Average the energy
        int32_t avg_energy = (count > 0) ? (energy / count) : 0;

        // Step 7: Convert from Q15 to INT8 range [0, 127]
        // Q15 max is 32767, scale to 127
        // Using log scale is better, but linear is simpler for now
        int32_t scaled = (avg_energy * 127) / 32767;

        // Apply log-like compression (optional, improves dynamic range)
        // log2(x+1) approximation: x / (1 + x/32767)
        // Skipping for simplicity - can add later if needed

        // Clamp to INT8 range
        if (scaled > 127) scaled = 127;
        if (scaled < 0) scaled = 0;

        output[mel_bin] = (int8_t)scaled;
    }
}

} // extern "C"
