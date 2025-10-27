/*
 * Phase 2.2: Basic Mel Spectrogram Kernel for AMD Phoenix NPU
 *
 * Implements:
 * - Hann window application
 * - 512-point FFT (Cooley-Tukey radix-2)
 * - Magnitude spectrum computation
 * - Mel filterbank application
 *
 * Uses precomputed lookup tables from mel_luts.h
 */

#include <stdint.h>
#include "mel_luts.h"

// FFT helper: bit-reversal permutation
static void bit_reverse_int16(int16_t* real, int16_t* imag, uint32_t n) {
    uint32_t j = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (i < j) {
            // Swap real parts
            int16_t temp = real[i];
            real[i] = real[j];
            real[j] = temp;

            // Swap imaginary parts
            temp = imag[i];
            imag[i] = imag[j];
            imag[j] = temp;
        }

        // Bit-reversal index calculation
        uint32_t m = n >> 1;
        while (m > 0 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

// Simplified FFT using precomputed twiddle factors (Q7 format)
static void fft_512_q7(int16_t* real, int16_t* imag) {
    const uint32_t N = 512;

    // Bit-reversal permutation
    bit_reverse_int16(real, imag, N);

    // FFT butterfly computations
    for (uint32_t s = 1; s <= 9; s++) {  // log2(512) = 9 stages
        uint32_t m = 1 << s;  // 2^s
        uint32_t m2 = m >> 1;

        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < m2; j++) {
                // Get twiddle factor
                uint32_t twiddle_idx = (j * 256) / m2;  // Map to [0, 255]
                int8_t cos_val = twiddle_cos_q7[twiddle_idx];
                int8_t sin_val = twiddle_sin_q7[twiddle_idx];

                uint32_t idx1 = k + j;
                uint32_t idx2 = k + j + m2;

                // Butterfly computation (Q7 multiply)
                int32_t t_real = ((int32_t)cos_val * real[idx2] - (int32_t)sin_val * imag[idx2]) >> 7;
                int32_t t_imag = ((int32_t)sin_val * real[idx2] + (int32_t)cos_val * imag[idx2]) >> 7;

                // Butterfly outputs
                int32_t u_real = real[idx1];
                int32_t u_imag = imag[idx1];

                real[idx1] = (int16_t)(u_real + t_real);
                imag[idx1] = (int16_t)(u_imag + t_imag);
                real[idx2] = (int16_t)(u_real - t_real);
                imag[idx2] = (int16_t)(u_imag - t_imag);
            }
        }
    }
}

// Compute magnitude spectrum (first 256 bins, Nyquist symmetry)
static void compute_magnitude_spectrum(const int16_t* real, const int16_t* imag,
                                       int16_t* magnitude, uint32_t n_bins) {
    for (uint32_t i = 0; i < n_bins; i++) {
        // Magnitude squared (to avoid sqrt)
        int32_t mag_sq = ((int32_t)real[i] * real[i] + (int32_t)imag[i] * imag[i]) >> 8;
        magnitude[i] = (int16_t)(mag_sq > 32767 ? 32767 : mag_sq);
    }
}

// Apply mel filterbank
static void apply_mel_filterbank(const int16_t* spectrum, int8_t* mel_output, uint32_t n_mels) {
    for (uint32_t mel_idx = 0; mel_idx < n_mels; mel_idx++) {
        int32_t mel_energy = 0;

        // Dot product with mel filter weights
        for (uint32_t bin = 0; bin < 256; bin++) {
            int8_t weight = mel_filter_weights_q7[mel_idx][bin];
            if (weight > 0) {
                mel_energy += (int32_t)spectrum[bin] * weight;
            }
        }

        // Normalize and clip to INT8
        mel_energy >>= 14;  // Scale down
        mel_output[mel_idx] = (int8_t)(mel_energy > 127 ? 127 : (mel_energy < -128 ? -128 : mel_energy));
    }
}

// Main mel spectrogram kernel
void mel_spectrogram_kernel(int16_t* restrict audio_in,     // Input: [num_frames, 400] INT16
                           int8_t* restrict mel_out,        // Output: [num_frames, 80] INT8
                           uint32_t num_frames) {

    // Buffers for FFT computation (512 samples per frame)
    int16_t fft_real[512];
    int16_t fft_imag[512];
    int16_t magnitude[256];

    for (uint32_t frame = 0; frame < num_frames; frame++) {
        int16_t* frame_audio = audio_in + (frame * 400);
        int8_t* frame_mel = mel_out + (frame * 80);

        // Step 1: Apply Hann window and prepare FFT input
        for (uint32_t i = 0; i < 400; i++) {
            // Apply Hann window (Q7 multiply)
            int32_t windowed = ((int32_t)frame_audio[i] * hann_window_q7[i]) >> 7;
            fft_real[i] = (int16_t)windowed;
            fft_imag[i] = 0;  // Real input
        }

        // Zero-pad to 512 samples
        for (uint32_t i = 400; i < 512; i++) {
            fft_real[i] = 0;
            fft_imag[i] = 0;
        }

        // Step 2: Compute 512-point FFT
        fft_512_q7(fft_real, fft_imag);

        // Step 3: Compute magnitude spectrum (first 256 bins)
        compute_magnitude_spectrum(fft_real, fft_imag, magnitude, 256);

        // Step 4: Apply mel filterbank
        apply_mel_filterbank(magnitude, frame_mel, 80);
    }
}
