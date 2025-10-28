// Minimal MEL kernel - just passthrough to test infrastructure
// No large arrays, no floating-point, no FFT

#include <stdint.h>

extern "C" {

void mel_kernel_simple(int8_t *input, int8_t *output) {
    // Simple passthrough with basic processing
    // This tests: linkage, memory access, basic operations

    // Just copy first 80 bytes from input to output
    for (int i = 0; i < 80; i++) {
        output[i] = input[i];
    }
}

} // extern "C"
