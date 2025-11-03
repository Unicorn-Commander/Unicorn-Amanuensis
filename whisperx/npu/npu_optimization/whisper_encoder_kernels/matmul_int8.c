/**
 * INT8 Matrix Multiplication Kernel for Whisper Encoder
 * Optimized for AIE2 vector operations
 *
 * Computes: C = A @ B
 * Where:
 *   A: [M x K] int8 matrix (activation)
 *   B: [K x N] int8 matrix (weights)
 *   C: [M x N] int32 matrix (output, scaled)
 *
 * For Whisper base encoder:
 *   Hidden dim = 512
 *   Typical tile: 64x64 matmul
 */

#include <stdint.h>

// Simple memset replacement for AIE2 (no stdlib)
static inline void zero_memory_int32(int32_t* ptr, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        ptr[i] = 0;
    }
}

/**
 * Tiled INT8 Matrix Multiply
 * Tile size: 64x64 for optimal AIE2 memory usage
 *
 * Input buffer layout (flat arrays):
 *   A: [M*K] int8 values
 *   B: [K*N] int8 values
 *   C: [M*N] int32 values (output accumulator)
 */
void matmul_int8_64x64(
    const int8_t* A,     // Input: [64 x 64] activation tile
    const int8_t* B,     // Input: [64 x 64] weight tile
    int32_t* C,          // Output: [64 x 64] result tile
    uint32_t M,          // Rows of A (typically 64)
    uint32_t K,          // Cols of A / Rows of B (typically 64)
    uint32_t N           // Cols of B (typically 64)
) {
    // Zero output accumulator
    zero_memory_int32(C, M * N);

    // AIE2 optimized: 32-element vector multiply-accumulate
    // Process 32 elements per inner loop iteration
    const uint32_t VECTOR_LEN = 32;

    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            int32_t acc = 0;

            // Inner product: vectorized in chunks of 32
            for (uint32_t k = 0; k < K; k += VECTOR_LEN) {
                uint32_t remaining = K - k;
                uint32_t chunk_size = (remaining < VECTOR_LEN) ? remaining : VECTOR_LEN;

                // Vectorized multiply-accumulate
                // AIE2 compiler will optimize this to VMAC instructions
                for (uint32_t v = 0; v < chunk_size; v++) {
                    int32_t a_val = A[m * K + (k + v)];
                    int32_t b_val = B[(k + v) * N + n];
                    acc += a_val * b_val;
                }
            }

            C[m * N + n] = acc;
        }
    }
}

/**
 * Simplified 32x32 matmul for smaller tiles
 * Used for testing and smaller layers
 */
void matmul_int8_32x32(
    const int8_t* A,
    const int8_t* B,
    int32_t* C
) {
    zero_memory_int32(C, 32 * 32);

    for (uint32_t m = 0; m < 32; m++) {
        for (uint32_t n = 0; n < 32; n++) {
            int32_t acc = 0;
            for (uint32_t k = 0; k < 32; k++) {
                acc += (int32_t)A[m * 32 + k] * (int32_t)B[k * 32 + n];
            }
            C[m * 32 + n] = acc;
        }
    }
}

/**
 * Requantize INT32 accumulator back to INT8
 * Apply scale + zero-point quantization
 *
 * Output = clamp((input >> shift), -128, 127)
 */
void requantize_int32_to_int8(
    const int32_t* input,
    int8_t* output,
    uint32_t size,
    uint32_t shift  // Right shift amount (divide by 2^shift)
) {
    for (uint32_t i = 0; i < size; i++) {
        int32_t val = input[i] >> shift;

        // Clamp to INT8 range
        if (val > 127) val = 127;
        if (val < -128) val = -128;

        output[i] = (int8_t)val;
    }
}

/**
 * Simple 16x16 matmul for AIE2 memory constraints
 * AIE2 cores have limited local memory (~32KB)
 */
void matmul_int8_16x16(
    const int8_t* A,     // [16 x 16] = 256 bytes
    const int8_t* B,     // [16 x 16] = 256 bytes
    int8_t* C            // [16 x 16] = 256 bytes output
) {
    // Small accumulator buffer (16x16 int32 = 1024 bytes)
    int32_t acc[256];

    // Zero accumulator
    for (uint32_t i = 0; i < 256; i++) {
        acc[i] = 0;
    }

    // Matrix multiply
    for (uint32_t m = 0; m < 16; m++) {
        for (uint32_t n = 0; n < 16; n++) {
            int32_t sum = 0;
            for (uint32_t k = 0; k < 16; k++) {
                sum += (int32_t)A[m * 16 + k] * (int32_t)B[k * 16 + n];
            }
            acc[m * 16 + n] = sum;
        }
    }

    // Requantize to INT8 (shift by 7 = divide by 128)
    for (uint32_t i = 0; i < 256; i++) {
        int32_t val = acc[i] >> 7;
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        C[i] = (int8_t)val;
    }
}

/**
 * 16x16 matmul with packed input buffer (FIXED VERSION)
 * Takes a single 512-byte buffer containing both A and B matrices
 *
 * This matches the pattern used by other kernels (attention, layernorm, GELU)
 * where multiple inputs are packed into a single buffer
 *
 * Buffer layout:
 *   packed_input[0:255]   = Matrix A (16x16 int8)
 *   packed_input[256:511] = Matrix B (16x16 int8)
 */
void matmul_int8_16x16_packed(
    const int8_t* packed_input,  // [512] = A (256 bytes) + B (256 bytes)
    int8_t* C                    // [256] = Output matrix (16x16 int8)
) {
    // Unpack A and B from input buffer
    const int8_t* A = packed_input;        // First 256 bytes
    const int8_t* B = packed_input + 256;  // Next 256 bytes

    // Small accumulator buffer (16x16 int32 = 1024 bytes)
    int32_t acc[256];

    // Zero accumulator
    for (uint32_t i = 0; i < 256; i++) {
        acc[i] = 0;
    }

    // Matrix multiply: C = A @ B
    for (uint32_t m = 0; m < 16; m++) {
        for (uint32_t n = 0; n < 16; n++) {
            int32_t sum = 0;
            for (uint32_t k = 0; k < 16; k++) {
                // A is row-major: A[m, k] = A[m * 16 + k]
                // B is row-major: B[k, n] = B[k * 16 + n]
                int32_t a_val = (int32_t)A[m * 16 + k];
                int32_t b_val = (int32_t)B[k * 16 + n];
                sum += a_val * b_val;
            }
            acc[m * 16 + n] = sum;
        }
    }

    // Requantize to INT8 (shift by 7 = divide by 128)
    // This is the quantization scale for INT8 matmul
    for (uint32_t i = 0; i < 256; i++) {
        int32_t val = acc[i] >> 7;

        // Clamp to INT8 range
        if (val > 127) val = 127;
        if (val < -128) val = -128;

        C[i] = (int8_t)val;
    }
}
