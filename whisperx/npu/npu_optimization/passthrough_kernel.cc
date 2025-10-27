//===- passthrough_kernel.cc -------------------------------------*- C++ -*-===//
//
// Simple passthrough kernel for Phoenix NPU testing
// Copies input to output with vectorized operations
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Simple C++ implementation for AIE2
extern "C" {

void passthrough_kernel(uint8_t* restrict in, uint8_t* restrict out, int32_t size) {
    // Simple memcpy - AIE compiler will vectorize this
    for (int i = 0; i < size; i++) {
        out[i] = in[i];
    }
}

}  // extern "C"
