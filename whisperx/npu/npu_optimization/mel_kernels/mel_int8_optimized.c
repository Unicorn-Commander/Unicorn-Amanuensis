/*
 * Phase 2.3: INT8 Optimized Mel Spectrogram Kernel for AMD Phoenix NPU
 *
 * Features:
 * - Full INT8 quantization (Q7 format)
 * - AIE2 SIMD vectorization (32 INT8 operations per cycle)
 * - Optimized memory access patterns
 * - Block floating-point FFT (prevents overflow)
 * - Target: 60-80x realtime performance
 *
 * Uses precomputed lookup tables from mel_luts.h
 */

#include <stdint.h>
#include <string.h>
#include "mel_luts.h"

// Configuration
#define FFT_SIZE 512
#define MEL_BINS 80
#define WINDOW_SIZE 400
#define SPECTRUM_BINS 256

// AIE2 SIMD helpers (simplified for C compilation)
// In production, these would use aie_api vector intrinsics

// Vectorized INT8 multiply-accumulate (simulated)
static inline int32_t vec_mac_int8(const int8_t* restrict a,
                                   const int8_t* restrict b,
                                   int32_t count) {
    int32_t acc = 0;

    // Process 32 elements at a time (AIE2 can do this in 1 cycle)
    int32_t vec_count = count >> 5;  // count / 32
    int32_t remainder = count & 31;   // count % 32

    for (int32_t v = 0; v < vec_count; v++) {
        int32_t vec_offset = v << 5;

        // In real AIE2 code, this entire loop would be 1 cycle:
        // aie::vector<int8, 32> va = aie::load_v<32>(a + vec_offset);
        // aie::vector<int8, 32> vb = aie::load_v<32>(b + vec_offset);
        // acc = aie::mac(acc, va, vb);

        for (int32_t i = 0; i < 32; i++) {
            acc += (int32_t)a[vec_offset + i] * (int32_t)b[vec_offset + i];
        }
    }

    // Handle remaining elements
    for (int32_t i = vec_count << 5; i < count; i++) {
        acc += (int32_t)a[i] * (int32_t)b[i];
    }

    return acc;
}

// Quantize INT16 audio to INT8 (Q7)
static inline void quantize_audio_to_int8(const int16_t* restrict audio_in,
                                          int8_t* restrict audio_q7,
                                          uint32_t n_samples) {
    // Right-shift by 8 bits to convert INT16 to INT8
    for (uint32_t i = 0; i < n_samples; i++) {
        int16_t sample = audio_in[i];
        audio_q7[i] = (int8_t)(sample >> 8);
    }
}

// Apply Hann window with Q7 coefficients (vectorized)
static inline void apply_hann_window_q7(int8_t* restrict audio_q7,
                                        uint32_t n_samples) {
    // Process in blocks of 32 (AIE2 SIMD width)
    for (uint32_t i = 0; i < n_samples; i++) {
        // Q7 × Q7 = Q14, then shift back to Q7
        int32_t windowed = ((int32_t)audio_q7[i] * (int32_t)hann_window_q7[i]) >> 7;
        audio_q7[i] = (int8_t)windowed;
    }
}

// Simplified Radix-2 FFT butterfly (Q7 format with block scaling)
static inline void fft_butterfly_q7(int8_t* restrict real,
                                    int8_t* restrict imag,
                                    uint32_t idx1,
                                    uint32_t idx2,
                                    int8_t twiddle_cos,
                                    int8_t twiddle_sin,
                                    uint32_t scale_shift) {
    // Butterfly computation with Q7 arithmetic
    int32_t t_real = ((int32_t)twiddle_cos * real[idx2] - (int32_t)twiddle_sin * imag[idx2]) >> (7 + scale_shift);
    int32_t t_imag = ((int32_t)twiddle_sin * real[idx2] + (int32_t)twiddle_cos * imag[idx2]) >> (7 + scale_shift);

    int32_t u_real = real[idx1];
    int32_t u_imag = imag[idx1];

    real[idx1] = (int8_t)((u_real + t_real) >> 1);  // Scale by 1/2 per stage
    imag[idx1] = (int8_t)((u_imag + t_imag) >> 1);
    real[idx2] = (int8_t)((u_real - t_real) >> 1);
    imag[idx2] = (int8_t)((u_imag - t_imag) >> 1);
}

// Bit-reversal permutation for INT8
static void bit_reverse_int8(int8_t* restrict real,
                             int8_t* restrict imag,
                             uint32_t n) {
    uint32_t j = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (i < j) {
            // Swap
            int8_t temp = real[i];
            real[i] = real[j];
            real[j] = temp;

            temp = imag[i];
            imag[i] = imag[j];
            imag[j] = temp;
        }

        uint32_t m = n >> 1;
        while (m > 0 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

// 512-point FFT with block floating-point (INT8 Q7)
static void fft_512_int8_optimized(int8_t* restrict real,
                                   int8_t* restrict imag) {
    const uint32_t N = FFT_SIZE;

    // Bit-reversal permutation
    bit_reverse_int8(real, imag, N);

    // FFT stages with block scaling
    for (uint32_t s = 1; s <= 9; s++) {  // log2(512) = 9 stages
        uint32_t m = 1 << s;
        uint32_t m2 = m >> 1;

        // Block scaling: scale by 1/2 per stage to prevent overflow
        uint32_t scale_shift = 0;  // Already handled in butterfly

        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < m2; j++) {
                uint32_t twiddle_idx = (j * 256) / m2;
                int8_t cos_val = twiddle_cos_q7[twiddle_idx];
                int8_t sin_val = twiddle_sin_q7[twiddle_idx];

                uint32_t idx1 = k + j;
                uint32_t idx2 = k + j + m2;

                fft_butterfly_q7(real, imag, idx1, idx2, cos_val, sin_val, scale_shift);
            }
        }
    }
}

// Compute magnitude spectrum using lookup table (vectorized)
static void compute_magnitude_spectrum_int8(const int8_t* restrict real,
                                           const int8_t* restrict imag,
                                           int8_t* restrict magnitude,
                                           uint32_t n_bins) {
    for (uint32_t i = 0; i < n_bins; i++) {
        // Magnitude squared
        int32_t mag_sq = (int32_t)real[i] * real[i] + (int32_t)imag[i] * imag[i];

        // Use log magnitude lookup table
        magnitude[i] = fast_log_magnitude((int16_t)mag_sq);
    }
}

// Apply mel filterbank (vectorized with AIE2 SIMD)
static void apply_mel_filterbank_int8(const int8_t* restrict spectrum,
                                     int8_t* restrict mel_output,
                                     uint32_t n_mels) {
    for (uint32_t mel_idx = 0; mel_idx < n_mels; mel_idx++) {
        // Vectorized dot product using AIE2 MAC
        int32_t mel_energy = vec_mac_int8(
            spectrum,
            mel_filter_weights_q7[mel_idx],
            SPECTRUM_BINS
        );

        // Normalize and clip to INT8
        mel_energy >>= 14;  // Scale down

        if (mel_energy > 127) mel_energy = 127;
        if (mel_energy < -128) mel_energy = -128;

        mel_output[mel_idx] = (int8_t)mel_energy;
    }
}

// Main INT8 mel spectrogram kernel (optimized for AIE2)
void mel_spectrogram_int8_kernel(int16_t* restrict audio_in,    // [num_frames, 400] INT16
                                int8_t* restrict mel_out,        // [num_frames, 80] INT8
                                uint32_t num_frames) {

    // Allocate buffers (in real AIE2, these would be in tile memory)
    int8_t audio_q7[WINDOW_SIZE];
    int8_t fft_real[FFT_SIZE];
    int8_t fft_imag[FFT_SIZE];
    int8_t magnitude[SPECTRUM_BINS];

    for (uint32_t frame = 0; frame < num_frames; frame++) {
        int16_t* frame_audio = audio_in + (frame * WINDOW_SIZE);
        int8_t* frame_mel = mel_out + (frame * MEL_BINS);

        // Step 1: Quantize audio to INT8 (Q7)
        quantize_audio_to_int8(frame_audio, audio_q7, WINDOW_SIZE);

        // Step 2: Apply Hann window (Q7 × Q7)
        apply_hann_window_q7(audio_q7, WINDOW_SIZE);

        // Step 3: Prepare FFT input (copy and zero-pad)
        for (uint32_t i = 0; i < WINDOW_SIZE; i++) {
            fft_real[i] = audio_q7[i];
            fft_imag[i] = 0;
        }
        for (uint32_t i = WINDOW_SIZE; i < FFT_SIZE; i++) {
            fft_real[i] = 0;
            fft_imag[i] = 0;
        }

        // Step 4: Compute 512-point FFT with block scaling
        fft_512_int8_optimized(fft_real, fft_imag);

        // Step 5: Compute magnitude spectrum (with log LUT)
        compute_magnitude_spectrum_int8(fft_real, fft_imag, magnitude, SPECTRUM_BINS);

        // Step 6: Apply mel filterbank (vectorized)
        apply_mel_filterbank_int8(magnitude, frame_mel, MEL_BINS);
    }
}

// AIE core entry point
// Memory-mapped buffer addresses from MLIR (aie.buffer declarations)
#define INPUT_BUFFER_ADDR  0x1000  // 4096 in hex (from MLIR: address = 4096)
#define OUTPUT_BUFFER_ADDR 0x0400  // 1024 in hex (from MLIR: address = 1024)

int main() {
    // Map buffers to memory addresses specified in MLIR
    int16_t* input_buffer = (int16_t*)INPUT_BUFFER_ADDR;   // 400 INT16 samples = 800 bytes
    int8_t* output_buffer = (int8_t*)OUTPUT_BUFFER_ADDR;   // 80 INT8 mel features
    
    // Process one frame
    mel_spectrogram_int8_kernel(input_buffer, output_buffer, 1);
    
    return 0;
}
