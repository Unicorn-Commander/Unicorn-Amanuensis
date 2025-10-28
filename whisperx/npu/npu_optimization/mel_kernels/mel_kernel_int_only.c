// MEL kernel - Integer-only version (no floating-point)
// Step 1: Convert bytes to INT16, basic processing
// No large arrays on stack

#include <stdint.h>

extern "C" {

void mel_kernel_simple(int8_t *input, int8_t *output) {
    // Process 400 INT16 samples -> 80 INT8 mel bins
    // Using integer arithmetic only, no stack arrays

    // Strategy: Process in streaming fashion
    // Accumulate energy per mel bin directly

    // For now: Simple energy binning (placeholder for FFT)
    // Each mel bin averages 10 INT16 samples (400/80 = 5, but we'll use 10 for overlap)

    for (int mel_bin = 0; mel_bin < 80; mel_bin++) {
        int32_t energy = 0;

        // Process 10 samples per mel bin (with overlap)
        int start_sample = mel_bin * 5;  // 80 * 5 = 400 samples

        for (int s = 0; s < 10 && (start_sample + s) < 400; s++) {
            int sample_idx = start_sample + s;
            int byte_idx = sample_idx * 2;

            // Bounds check
            if (byte_idx + 1 >= 800) break;

            // Convert bytes to INT16 (little-endian)
            int16_t sample = ((int16_t)(uint8_t)input[byte_idx]) |
                            (((int16_t)(int8_t)input[byte_idx + 1]) << 8);

            // Accumulate absolute value (energy)
            int16_t abs_sample = (sample < 0) ? -sample : sample;
            energy += abs_sample;
        }

        // Average and scale to INT8 (0-127 range)
        int32_t avg = energy / 10;

        // Scale: assume max input is 16000, max avg is 16000
        // Scale to 0-127: divide by ~125
        int8_t mel_value = (int8_t)(avg / 125);
        if (mel_value > 127) mel_value = 127;
        if (mel_value < 0) mel_value = 0;

        output[mel_bin] = mel_value;
    }
}

} // extern "C"
