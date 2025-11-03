/**
 * INT8 64x64 Matrix Multiplication Kernel for Whisper Encoder
 * Optimized for AIE2 vector operations
 *
 * Computes: C = A @ B
 * Where:
 *   A: [64 x 64] int8 matrix (activation) = 4096 bytes
 *   B: [64 x 64] int8 matrix (weights) = 4096 bytes
 *   C: [64 x 64] int8 matrix (output) = 4096 bytes
 *
 * Memory usage:
 *   Input buffer: 8192 bytes (A + B packed)
 *   Output buffer: 4096 bytes (C)
 *   Accumulator: 16384 bytes (64x64 int32)
 *   Total: ~28 KB (88% of 32 KB tile memory - near limit but safe)
 */

#include <stdint.h>

// Simple memset replacement for AIE2 (no stdlib)
static inline void zero_memory_int32(int32_t* ptr, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        ptr[i] = 0;
    }
}

/**
 * 64x64 matmul with packed input buffer
 * Takes a single 8192-byte buffer containing both A and B matrices
 *
 * This is the largest practical tile size for AIE2:
 * - Uses 88% of 32KB tile memory
 * - 16x fewer kernel invocations vs 16x16
 * - Best throughput for large matrices
 *
 * Buffer layout:
 *   packed_input[0:4095]    = Matrix A (64x64 int8)
 *   packed_input[4096:8191] = Matrix B (64x64 int8)
 */
void matmul_int8_64x64_packed(
    const int8_t* packed_input,  // [8192] = A (4096 bytes) + B (4096 bytes)
    int8_t* C                    // [4096] = Output matrix (64x64 int8)
) {
    // Unpack A and B from input buffer
    const int8_t* A = packed_input;         // First 4096 bytes
    const int8_t* B = packed_input + 4096;  // Next 4096 bytes

    // Accumulator buffer (64x64 int32 = 16384 bytes)
    int32_t acc[4096];

    // Zero accumulator
    zero_memory_int32(acc, 4096);

    // Matrix multiply: C = A @ B
    // Simple nested loops for AIE2 compiler compatibility
    for (uint32_t m = 0; m < 64; m++) {
        for (uint32_t n = 0; n < 64; n++) {
            int32_t sum = 0;

            // Inner product
            for (uint32_t k = 0; k < 64; k++) {
                int32_t a_val = (int32_t)A[m * 64 + k];
                int32_t b_val = (int32_t)B[k * 64 + n];
                sum += a_val * b_val;
            }

            acc[m * 64 + n] = sum;
        }
    }

    // Requantize to INT8 (shift by 7 = divide by 128)
    for (uint32_t i = 0; i < 4096; i++) {
        int32_t val = acc[i] >> 7;

        // Clamp to INT8 range
        if (val > 127) val = 127;
        if (val < -128) val = -128;

        C[i] = (int8_t)val;
    }
}
