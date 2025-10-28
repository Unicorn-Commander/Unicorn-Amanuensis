// Simple test kernel - just write constant values to verify buffer access

#include <stdint.h>

// Memory-mapped buffer addresses from MLIR
#define INPUT_BUFFER_ADDR  0x1000  // 4096 in hex
#define OUTPUT_BUFFER_ADDR 0x0400  // 1024 in hex

int main() {
    // Map output buffer
    int8_t* output = (int8_t*)OUTPUT_BUFFER_ADDR;
    
    // Write test pattern to verify buffer access works
    for (int i = 0; i < 80; i++) {
        output[i] = (int8_t)(i);  // Write 0, 1, 2, 3, ..., 79
    }
    
    return 0;
}
