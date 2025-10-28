// MEL kernel with real FFT computation
// Processes 400 INT16 audio samples -> 80 INT8 mel bins

#include <stdint.h>
#include "fft_coeffs.h"

// External FFT functions from fft_real_simple.o (C linkage)
extern "C" {
void fft_radix2_512_real(int16_t* input, complex_t* output);
void compute_magnitude_real(complex_t* fft_output, float* magnitude, uint32_t size);

void mel_kernel_simple(int8_t *input, int8_t *output) {
    // Buffer for audio samples (400 INT16)
    int16_t audio[400];

    // Buffer for windowed samples (will be zero-padded to 512)
    int16_t padded[512];

    // FFT output (512 complex numbers)
    complex_t fft_output[512];

    // Magnitude spectrum (256 bins - only positive frequencies)
    float magnitude[256];

    // Step 1: Convert input bytes to INT16 samples
    for (int i = 0; i < 400; i++) {
        // Little-endian: low byte first, high byte second
        audio[i] = ((int16_t)(uint8_t)input[i*2]) |
                   (((int16_t)(int8_t)input[i*2+1]) << 8);
    }

    // Step 2: Apply Hann window and zero-pad to 512
    for (int i = 0; i < 400; i++) {
        // Multiply by Hann window coefficient
        float windowed_sample = (float)audio[i] * hann_window[i];
        padded[i] = (int16_t)windowed_sample;
    }

    // Zero-padding (samples 400-511)
    for (int i = 400; i < 512; i++) {
        padded[i] = 0;
    }

    // Step 3: Compute 512-point real FFT
    fft_radix2_512_real(padded, fft_output);

    // Step 4: Compute magnitude spectrum (first 256 bins)
    compute_magnitude_real(fft_output, magnitude, 256);

    // Step 5: Downsample to 80 mel bins
    // Simple linear downsampling: 256 / 80 ≈ 3.2 bins per output
    // For now, use simple averaging (more sophisticated mel filtering later)
    for (int i = 0; i < 80; i++) {
        // Map output bin to input range
        float start_idx = (float)i * 256.0f / 80.0f;
        float end_idx = (float)(i + 1) * 256.0f / 80.0f;

        int start = (int)start_idx;
        int end = (int)end_idx;
        if (end > 256) end = 256;

        // Average magnitude over this range
        float sum = 0.0f;
        int count = 0;
        for (int j = start; j < end; j++) {
            sum += magnitude[j];
            count++;
        }

        float avg = (count > 0) ? sum / (float)count : 0.0f;

        // Convert to INT8 with scaling
        // Typical FFT magnitude range is 0-10000, scale to 0-127
        int8_t scaled = (int8_t)(avg * 0.0127f);
        if (scaled > 127) scaled = 127;
        if (scaled < 0) scaled = 0;

        output[i] = scaled;
    }
}

} // extern "C"