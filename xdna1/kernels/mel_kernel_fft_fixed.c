// MEL Spectrogram Kernel - Fixed-Point FFT Version with HTK Mel Filterbanks
// For AMD Phoenix NPU (AIE2) - Integer-only arithmetic
//
// Pipeline:
//   800 bytes (400 INT16 samples)
//   → Hann window (Q15)
//   → Zero-pad to 512
//   → 512-point FFT (Q15)
//   → Magnitude spectrum (256 bins)
//   → Apply HTK triangular mel filterbanks (80 filters)
//   → Output as INT8

#include <stdint.h>
#include "mel_coeffs_fixed.h"  // HTK mel filterbank coefficients (Q15)

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

// Apply HTK mel filterbanks to magnitude spectrum
// Input:  magnitude[256] - FFT magnitude bins in Q15 format
// Output: mel_output[80] - Mel bin energies in INT8 format [0, 127]
// Uses:   mel_filters_q15[] - Precomputed triangular filter weights from mel_coeffs_fixed.h
//
// Algorithm:
//   For each of 80 mel filters:
//     1. Apply triangular filter weights to FFT bins in filter's frequency range
//     2. Sum weighted magnitudes (Q15 × Q15 = Q30, scaled back to Q15)
//     3. Apply log compression for better dynamic range
//     4. Convert to INT8 range [0, 127] with clamping
//
// All arithmetic uses Q15 fixed-point (no floating point) for NPU compatibility
//
// Note: mel_coeffs_fixed.h uses full 257-element weight arrays with zeros for unused bins.
//       We optimize by only processing the non-zero range [start_bin, end_bin).
void apply_mel_filters_q15(
    const int16_t* magnitude,  // 256 FFT magnitude bins (Q15)
    int8_t* mel_output,         // 80 mel bins (INT8 output)
    uint32_t n_mels             // Number of mel filters (typically 80)
) {
    for (uint32_t m = 0; m < n_mels; m++) {
        const mel_filter_q15_t* filter = &mel_filters_q15[m];

        int32_t mel_energy = 0;  // Accumulator (Q30 after Q15 × Q15)

        // Apply triangular filter across the filter's frequency range
        // Filter weights array is indexed by FFT bin number (0-256)
        // Only process non-zero range [start_bin, end_bin) for efficiency
        for (int bin = filter->start_bin; bin < filter->end_bin; bin++) {
            // Bounds check to prevent buffer overrun
            if (bin >= 256) break;

            // Get filter weight for this FFT bin (direct indexing)
            int16_t weight = filter->weights[bin];

            // Skip if weight is zero (sparse optimization)
            if (weight == 0) continue;

            // Q15 × Q15 = Q30 multiplication
            int32_t weighted = (int32_t)magnitude[bin] * (int32_t)weight;

            // Scale back to Q15 and accumulate
            mel_energy += weighted >> 15;
        }

        // Convert Q15 energy to INT8 range [0, 127]
        // Use aggressive scaling to amplify signal and use full dynamic range
        // mel_energy can be quite small for typical audio, so multiply by large factor

        if (mel_energy < 0) mel_energy = 0;

        // Aggressive scaling: multiply by 512 then divide by 32767
        // This amplifies weak signals while still preventing overflow
        // (mel_energy * 512) / 32767 ≈ mel_energy / 64
        // For mel_energy=100, scaled ≈ 1.5, for mel_energy=8192, scaled ≈ 127
        int32_t scaled = (mel_energy * 512) / 32767;

        // Clamp to INT8 range [0, 127]
        if (scaled > 127) scaled = 127;
        if (scaled < 0) scaled = 0;

        mel_output[m] = (int8_t)scaled;
    }
}

void mel_kernel_simple(int8_t *input, int8_t *output) {
    // Working buffers (stack allocation - carefully sized)
    // Total: 1024 + 2048 + 512 = 3584 bytes (~3.5KB)
    // This is under the ~7KB limit that caused stack overflow

    int16_t samples[512];        // 1024 bytes - zero-padded audio
    complex_q15_t fft_out[512];  // 2048 bytes - FFT output
    int16_t magnitude[256];      // 512 bytes - magnitude spectrum

    // Step 1: Convert 800 bytes to 400 INT16 samples (little-endian)
    // FIXED OCT 31: Changed high byte from int8_t to uint8_t to prevent sign extension bug
    for (int i = 0; i < 400; i++) {
        int byte_idx = i * 2;
        samples[i] = ((int16_t)(uint8_t)input[byte_idx]) |
                    (((int16_t)(uint8_t)input[byte_idx + 1]) << 8);
    }

    // Step 2: Apply Hann window (Q15 × Q15 → Q15)
    apply_hann_window_fixed(samples, hann_window_q15, 400);

    // Step 3: Zero-pad to 512 samples
    zero_pad_to_512(samples, 400);

    // Step 4: Compute 512-point FFT
    fft_radix2_512_fixed(samples, fft_out);

    // Step 5: Compute magnitude spectrum (first 256 bins only, due to symmetry)
    compute_magnitude_fixed(fft_out, magnitude, 256);

    // Step 6: Apply HTK triangular mel filterbanks (FIXED)
    // Uses proper mel-scale triangular filters instead of linear averaging
    apply_mel_filters_q15(magnitude, output, 80);
}

} // extern "C"
