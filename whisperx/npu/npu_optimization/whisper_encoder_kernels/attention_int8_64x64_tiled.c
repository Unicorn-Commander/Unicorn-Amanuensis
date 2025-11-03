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
 *
 * LOOKUP TABLE SOFTMAX (Week 2 Day 3):
 * - Uses pre-computed exp() lookup table for exact values
 * - 128 entries covering INT8 range [-127, 0]
 * - Only 512 bytes memory overhead
 * - Expected correlation: 0.7-0.9 (vs 0.123 with polynomial approximation)
 */

#include <stdint.h>
#include "exp_lut_int8.h"

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
 * Lookup Table Softmax for INT32 scores - for up to 64 elements
 * Uses pre-computed exponential lookup table for exact exp() values
 *
 * Input: [N] int32 values (attention scores - NOT clamped to INT8!)
 * Output: [N] int8 values (probabilities summing to ~127)
 *
 * CRITICAL FIX: Takes INT32 scores to preserve dynamic range before softmax
 * Expected correlation improvement: 0.123 → 0.7-0.9
 *
 * AIE2 Constraints: No 64-bit division, so we scale carefully to stay in 32-bit
 */
void softmax_int32_to_int8(const int32_t* input, int8_t* output, uint32_t N) {
    // Step 1: Find max value for numerical stability
    // NOW operates on INT32 range (±32K) instead of INT8 (±127)
    int32_t max_val = input[0];
    for (uint32_t i = 1; i < N; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Step 2: Compute exp(x - max) using lookup table
    // Map INT32 scores (±32K) to INT8 LUT range (0 to -127)
    // LUT values are scaled by EXP_LUT_SCALE (1048576 = 2^20)
    // For 64 elements with max exp = 1048576, sum can be up to 67M (fits in 32-bit)
    uint32_t sum = 0;  // 32-bit sufficient for 64 elements
    int32_t exp_vals[64];  // Temporary storage for exp values

    for (uint32_t i = 0; i < N; i++) {
        // Shift for stability (x_shifted <= 0)
        int32_t x_shifted = input[i] - max_val;

        // CRITICAL FIX: Scale INT32 range to INT8 LUT range
        // Use bit shift (divide by 256) to map ±32K → ±127 range for LUT lookup
        int32_t x_scaled = x_shifted >> 8;  // Divide by 256 using shift

        // Clamp to valid LUT range [-127, 0]
        if (x_scaled < -127) x_scaled = -127;
        if (x_scaled > 0) x_scaled = 0;

        // Lookup exp(x_scaled) from pre-computed table
        // EXP_LUT_INT8[-x_scaled] gives us exp(x_scaled) * EXP_LUT_SCALE
        exp_vals[i] = EXP_LUT_INT8[-x_scaled];

        sum += (uint32_t)exp_vals[i];
    }

    // Step 3: Normalize to sum to ~127 (INT8 max positive value)
    // output[i] = (exp_vals[i] * 127) / sum
    //
    // To avoid overflow in exp_vals[i] * 127:
    // Max exp_vals[i] = 1048576, so exp_vals[i] * 127 = 133M (fits in 32-bit signed)
    //
    // AIE2 FIX: Use 32-bit arithmetic only (64-bit division not supported)
    for (uint32_t i = 0; i < N; i++) {
        if (sum > 0) {
            // Compute: (exp_vals[i] * 127) / sum
            // Scale down before multiply to prevent overflow
            // exp_vals[i] / (sum/127) ≈ (exp_vals[i] * 127) / sum
            uint32_t scaled_exp = (uint32_t)exp_vals[i] >> 10;  // Divide by 1024
            uint32_t scaled_sum = sum >> 10;  // Divide by 1024

            if (scaled_sum == 0) scaled_sum = 1;  // Prevent division by zero

            uint32_t normalized = (scaled_exp * 127) / scaled_sum;

            // Clamp to INT8 range [0, 127]
            if (normalized > 127) normalized = 127;

            output[i] = (int8_t)normalized;
        } else {
            // Fallback: uniform distribution (should never happen)
            output[i] = (int8_t)(127 / N);
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
    // Step 1 & 2: Compute scores and apply softmax ROW-BY-ROW
    // CRITICAL FIX: Process row-by-row to avoid large INT32 array on stack
    // This preserves INT32 precision for softmax while fitting in AIE2 memory constraints
    int8_t attention_weights[32 * 64];  // 2KB output buffer

    for (uint32_t i = 0; i < 32; i++) {
        // CRITICAL FIX: Small INT32 buffer per row (256 bytes per row)
        // This keeps full precision before softmax, but fits in memory
        int32_t scores_row[64];  // 256 bytes per row (64 * 4 bytes)

        // Compute Q@K^T for this row
        for (uint32_t j = 0; j < 64; j++) {
            int32_t score = 0;

            // Dot product: Q_tile[i, :] @ K[j, :]
            for (uint32_t k = 0; k < 64; k++) {
                score += (int32_t)Q_tile[i * 64 + k] * (int32_t)K[j * 64 + k];
            }

            // Scale by 1/sqrt(d_k)
            score >>= scale_shift;

            // CRITICAL FIX: NO clamping to INT8 here!
            // Keep full INT32 precision for softmax
            // Scores typically range ±32K, need this for proper attention distribution
            scores_row[j] = score;  // Store as INT32
        }

        // Apply softmax to this row immediately
        // CRITICAL FIX: Use INT32-to-INT8 softmax with LUT
        // This preserves score distribution through softmax computation
        softmax_int32_to_int8(scores_row, &attention_weights[i * 64], 64);
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
