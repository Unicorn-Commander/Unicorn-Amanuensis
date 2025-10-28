// MEL Kernel with built-in infinite loop and lock management
// This allows using empty MLIR core with elf_file attribute
//
// Buffer addresses from MLIR (mel_int8_complete.mlir):
// Input:  address = 4096 (0x1000), size = 800 bytes
// Output: address = 1024 (0x0400), size = 80 bytes
//
// Lock IDs from MLIR:
// Input producer lock: tile(0,2) lock 0, init=2
// Input consumer lock: tile(0,2) lock 1, init=0
// Output producer lock: tile(0,2) lock 2, init=2
// Output consumer lock: tile(0,2) lock 3, init=0

#include <stdint.h>

// Buffer addresses (from MLIR aie.buffer declarations)
#define INPUT_BUFFER_0  ((int8_t*)0x1000)  // 4096 decimal
#define OUTPUT_BUFFER_0 ((int8_t*)0x0400)  // 1024 decimal

// Lock management (AIE intrinsics)
// These would be replaced with actual AIE lock acquire/release intrinsics
extern void acquire_lock(uint32_t lock_id, int32_t value);
extern void release_lock(uint32_t lock_id, int32_t value);

// Lock IDs
#define INPUT_CONS_LOCK  1
#define INPUT_PROD_LOCK  0
#define OUTPUT_CONS_LOCK 3
#define OUTPUT_PROD_LOCK 2

// Simple mel computation (for testing - just copies pattern)
static void compute_mel(int8_t *input, int8_t *output) {
    // For now, just write a test pattern
    for (int i = 0; i < 80; i++) {
        output[i] = (int8_t)i;
    }
}

// Main entry point - infinite loop with lock synchronization
int main() {
    // Infinite loop - core stays active waiting for DMA data
    while (1) {
        // Acquire input (wait for DMA to fill buffer)
        acquire_lock(INPUT_CONS_LOCK, 1);

        // Acquire output space (wait for previous output to be sent)
        acquire_lock(OUTPUT_PROD_LOCK, 1);

        // Do the computation
        compute_mel(INPUT_BUFFER_0, OUTPUT_BUFFER_0);

        // Release input (signal we're done with it)
        release_lock(INPUT_PROD_LOCK, 1);

        // Release output (signal it's ready to send)
        release_lock(OUTPUT_CONS_LOCK, 1);
    }

    return 0;  // Never reached
}
