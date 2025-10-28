// Simplest possible test: main() that writes a pattern
// Testing if main() gets called at all

#include <stdint.h>

// Buffer addresses from MLIR
#define OUTPUT_BUFFER ((int8_t*)0x0400)  // 1024 decimal, matches MLIR

int main() {
    // Write test pattern
    volatile int8_t *out = OUTPUT_BUFFER;
    for (int i = 0; i < 80; i++) {
        out[i] = (int8_t)(i + 100);  // Offset by 100 to make it obvious
    }

    return 0;
}
