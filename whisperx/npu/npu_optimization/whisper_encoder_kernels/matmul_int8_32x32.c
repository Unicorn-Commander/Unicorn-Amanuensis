/**
 * INT8 32x32 Matrix Multiplication Kernel for Whisper Encoder
 * Optimized for AIE2 vector operations
 *
 * Computes: C = A @ B
 * Where:
 *   A: [32 x 32] int8 matrix (activation) = 1024 bytes
 *   B: [32 x 32] int8 matrix (weights) = 1024 bytes
 *   C: [32 x 32] int8 matrix (output) = 1024 bytes
 *
 * Memory usage:
 *   Input buffer: 2048 bytes (A + B packed)
 *   Output buffer: 1024 bytes (C)
 *   Accumulator: 4096 bytes (32x32 int32)
 *   Total: ~7 KB (well within 32 KB tile memory)
 */

#include <stdint.h>

// Simple memset replacement for AIE2 (no stdlib)
static inline void zero_memory_int32(int32_t* ptr, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        ptr[i] = 0;
    }
}

/**
 * 32x32 matmul with packed input buffer
 * Takes a single 2048-byte buffer containing both A and B matrices
 *
 * This matches the pattern used by the 16x16 kernel where multiple
 * inputs are packed into a single buffer for efficient DMA transfer
 *
 * Buffer layout:
 *   packed_input[0:1023]    = Matrix A (32x32 int8)
 *   packed_input[1024:2047] = Matrix B (32x32 int8)
 */
void matmul_int8_32x32_packed(
    const int8_t* packed_input,  // [2048] = A (1024 bytes) + B (1024 bytes)
    int8_t* C                    // [1024] = Output matrix (32x32 int8)
) {
    // Unpack A and B from input buffer
    const int8_t* A = packed_input;         // First 1024 bytes
    const int8_t* B = packed_input + 1024;  // Next 1024 bytes

    // Accumulator buffer (32x32 int32 = 4096 bytes)
    int32_t acc[1024];

    // Zero accumulator
    zero_memory_int32(acc, 1024);

    // Matrix multiply: C = A @ B
    // Process in 32x32 blocks for better cache locality
    for (uint32_t m = 0; m < 32; m++) {
        for (uint32_t n = 0; n < 32; n++) {
            int32_t sum = 0;

            // Inner product: dot(A[m, :], B[:, n])
            for (uint32_t k = 0; k < 32; k++) {
                // A is row-major: A[m, k] = A[m * 32 + k]
                // B is row-major: B[k, n] = B[k * 32 + n]
                int32_t a_val = (int32_t)A[m * 32 + k];
                int32_t b_val = (int32_t)B[k * 32 + n];
                sum += a_val * b_val;
            }

            acc[m * 32 + n] = sum;
        }
    }

    // Requantize to INT8 (shift by 7 = divide by 128)
    // This is the quantization scale for INT8 matmul
    for (uint32_t i = 0; i < 1024; i++) {
        int32_t val = acc[i] >> 7;

        // Clamp to INT8 range
        if (val > 127) val = 127;
        if (val < -128) val = -128;

        C[i] = (int8_t)val;
    }
}
