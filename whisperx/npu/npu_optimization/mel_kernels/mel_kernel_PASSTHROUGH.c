// SIMPLEST POSSIBLE TEST: Just copy first 80 bytes of input to output
// This will verify the data path works at all

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void mel_kernel_simple(int8_t *input, int8_t *output) {
    // Just copy first 80 bytes
    for (int i = 0; i < 80; i++) {
        output[i] = input[i];
    }
}

#ifdef __cplusplus
}
#endif
