// Simple mel kernel - pure computation, NO main(), NO loops
// The infinite loop and synchronization is in MLIR (Python-generated)

#include <stdint.h>

extern "C" {

// Simple test: write sequential pattern
void mel_kernel_simple(int8_t *input, int8_t *output) {
    // Simple passthrough + transformation to verify execution
    for (int i = 0; i < 80; i++) {
        // Read from input (800 bytes, 400 INT16 samples as bytes)
        // Write sequential pattern to output for now
        output[i] = (int8_t)i;
    }
}

}
