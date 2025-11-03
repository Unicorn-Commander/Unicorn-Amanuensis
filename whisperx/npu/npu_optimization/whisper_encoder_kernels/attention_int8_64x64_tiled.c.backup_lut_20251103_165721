/**
 * INT8 Attention Mechanism for Whisper Encoder - TILED 64x64
 * Optimized for AIE2 vector operations with memory constraints
 *
 * Implements: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 * TILING STRATEGY:
 * - Process 64x64 as 4 tiles of 32x32
 * - Each tile: 32x32 = 1024 bytes (fits in L1 cache)
 * - Accumulator: 32x32 int32 = 4KB (fits in memory)
 * - Total memory per tile: ~8KB (well within 32KB limit)
 *
 * This kernel processes ONE attention head on 64x64 tiles with internal tiling
 */

#include <stdint.h>
#include <string.h>

/**
 * Fast integer square root (for scaling factor sqrt(d_k))
 * Used for attention scaling: 1/sqrt(64) ≈ 1/8
 */
static inline uint16_t isqrt(uint32_t n) {
    if (n == 0) return 0;
    uint32_t x = n;
    uint32_t y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return (uint16_t)x;
}

/**
 * Improved softmax approximation for INT8 - for up to 64 elements
 * Uses better exponential approximation and proper scaling
 *
 * Input: [N] int8 values (attention scores)
 * Output: [N] int8 values (probabilities summing to ~127)
 */
void softmax_int8_64(const int8_t* input, int8_t* output, uint32_t N) {
    // Find max value for numerical stability
    int8_t max_val = input[0];
    for (uint32_t i = 1; i < N; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Compute exp(x - max) using improved approximation
    int32_t sum = 0;
    int32_t exp_vals[64];  // Temporary storage

    for (uint32_t i = 0; i < N; i++) {
        int32_t x = input[i] - max_val;  // Shift for stability (x <= 0)

        // Clamp to prevent overflow
        if (x < -127) x = -127;
        if (x > 0) x = 0;

        // Improved exp approximation using 4-term Taylor series:
        // exp(x/8) ≈ 1 + x/8 + x²/128 + x³/3072 + x⁴/98304
        // Then raise to 8th power for exp(x)
        // Scaled to fixed-point: multiply by 256 for precision

        // For x in [-127, 0], we use: exp(x) ≈ 256 * (1 + x/8)^8
        // Simplified binomial approximation:
        int32_t x_scaled = (x << 5);  // x * 32 for better precision

        if (x <= -80) {
            // For very negative values, exp is nearly zero
            exp_vals[i] = 1;
        } else if (x <= -40) {
            // Medium negative: exp(x) ≈ 256 / (1 - x/8)^4
            int32_t denom = 256 + ((-x) << 2);  // 256 - x*4
            exp_vals[i] = (65536) / denom;  // Scale by 256
        } else {
            // Closer to 0: Use Taylor series
            // exp(x/32) ≈ 1 + x/32 + (x/32)²/2
            int32_t x_32 = x;  // x is already in range [-40, 0]
            int32_t x2 = (x_32 * x_32) >> 5;  // x²/32
            int32_t result = 256 + (x_32 << 3) + (x2 >> 1);  // 256 * (1 + x/32 + x²/2048)

            if (result < 1) result = 1;
            exp_vals[i] = result;
        }

        sum += exp_vals[i];
    }

    // Normalize: output[i] = exp_vals[i] * 127 / sum
    // Use rounding division for better accuracy
    for (uint32_t i = 0; i < N; i++) {
        if (sum > 0) {
            int32_t normalized = ((exp_vals[i] * 127) + (sum >> 1)) / sum;  // Rounding division
            if (normalized > 127) normalized = 127;
            if (normalized < 0) normalized = 0;
            output[i] = (int8_t)normalized;
        } else {
            // Uniform distribution fallback
            output[i] = 127 / N;
        }
    }
}

/**
 * Process one 32x32 tile of attention
 * This is the core compute unit that fits in memory
 */
static void attention_tile_32x32(
    const int8_t* Q_tile,         // [32 x 64] query tile
    const int8_t* K,              // [64 x 64] full key matrix
    const int8_t* V,              // [64 x 64] full value matrix
    int8_t* output_tile,          // [32 x 64] output tile
    uint32_t q_start,             // Starting row in Q (0 or 32)
    uint32_t scale_shift
) {
    // Step 1: Compute attention scores: Q_tile @ K^T
    // Result: [32 x 64] scores
    int8_t scores[32 * 64];  // 2KB

    for (uint32_t i = 0; i < 32; i++) {
        for (uint32_t j = 0; j < 64; j++) {
            int32_t score = 0;

            // Dot product: Q_tile[i, :] @ K[j, :]
            for (uint32_t k = 0; k < 64; k++) {
                score += (int32_t)Q_tile[i * 64 + k] * (int32_t)K[j * 64 + k];
            }

            // Scale by 1/sqrt(d_k)
            score >>= scale_shift;

            // Clamp to INT8 range
            if (score > 127) score = 127;
            if (score < -128) score = -128;

            scores[i * 64 + j] = (int8_t)score;
        }
    }

    // Step 2: Apply softmax row-wise
    int8_t attention_weights[32 * 64];  // 2KB
    for (uint32_t i = 0; i < 32; i++) {
        softmax_int8_64(&scores[i * 64], &attention_weights[i * 64], 64);
    }

    // Step 3: Weighted sum: attention_weights @ V
    for (uint32_t i = 0; i < 32; i++) {
        for (uint32_t j = 0; j < 64; j++) {
            int32_t weighted_sum = 0;

            // Weighted sum: sum(attention_weights[i, k] * V[k, j])
            for (uint32_t k = 0; k < 64; k++) {
                weighted_sum += (int32_t)attention_weights[i * 64 + k] * (int32_t)V[k * 64 + j];
            }

            // Improved requantization to INT8 with proper scaling
            // attention_weights are in range [0, 127] (from softmax)
            // V values are in range [-128, 127]
            // Product sum can be up to 64 * 127 * 127 = 1,032,256
            //
            // To map back to [-128, 127], divide by (127 * 64 / 64) = 127
            // Using bit shift: divide by 128 (2^7) is close enough
            // But add rounding for accuracy:
            int32_t sign = (weighted_sum >= 0) ? 1 : -1;
            int32_t rounded = (weighted_sum + sign * 64) >> 7;  // Divide by 128 with rounding

            // Clamp to INT8 range
            if (rounded > 127) rounded = 127;
            if (rounded < -128) rounded = -128;

            output_tile[i * 64 + j] = (int8_t)rounded;
        }
    }
}

/**
 * Scaled dot-product attention for 64x64 tiles - TILED VERSION
 *
 * QKV_combined layout:
 *   Bytes 0-4095: Q matrix [64 x 64]
 *   Bytes 4096-8191: K matrix [64 x 64]
 *   Bytes 8192-12287: V matrix [64 x 64]
 *
 * MEMORY OPTIMIZATION: Process as 2 tiles of 32x64 each
 * Peak memory: 32x64 scores + 32x64 weights + 32x64 accum = 6KB
 */
void attention_64x64(
    const int8_t* QKV_combined,  // [12288] combined Q+K+V buffer
    int8_t* output,              // [64 x 64] output matrix
    uint32_t scale_shift         // Right shift for Q@K^T (divide by sqrt(d_k))
) {
    // Unpack Q, K, V from combined buffer
    const int8_t* Q = &QKV_combined[0];       // Bytes 0-4095
    const int8_t* K = &QKV_combined[4096];    // Bytes 4096-8191
    const int8_t* V = &QKV_combined[8192];    // Bytes 8192-12287

    // Process first 32 rows of Q (rows 0-31)
    attention_tile_32x32(&Q[0], K, V, &output[0], 0, scale_shift);

    // Process last 32 rows of Q (rows 32-63)
    attention_tile_32x32(&Q[32 * 64], K, V, &output[32 * 64], 32, scale_shift);
}

/**
 * Simplified attention for testing: just Q @ K^T (no softmax)
 * Useful for validating the pipeline before adding complexity
 */
void attention_scores_only_64x64(
    const int8_t* Q,      // [64 x 64]
    const int8_t* K,      // [64 x 64]
    int8_t* scores,       // [64 x 64] output
    uint32_t scale_shift  // Scaling factor
) {
    for (uint32_t i = 0; i < 64; i++) {
        for (uint32_t j = 0; j < 64; j++) {
            int32_t score = 0;
            for (uint32_t k = 0; k < 64; k++) {
                score += (int32_t)Q[i * 64 + k] * (int32_t)K[j * 64 + k];
            }
            score >>= scale_shift;
            if (score > 127) score = 127;
            if (score < -128) score = -128;
            scores[i * 64 + j] = (int8_t)score;
        }
    }
}

/**
 * Multi-head attention (processes 2 heads in parallel) - TILED VERSION
 * Each head: 64x64 matrices
 */
void multi_head_attention_2heads(
    const int8_t* QKV_combined_2heads,  // [2 × 12288] for 2 heads
    int8_t* output,                      // [2 × 64 × 64] outputs for 2 heads
    uint32_t scale_shift                 // Scaling factor
) {
    // Process head 0
    attention_64x64(&QKV_combined_2heads[0], &output[0], scale_shift);

    // Process head 1
    attention_64x64(&QKV_combined_2heads[12288], &output[4096], scale_shift);
}
