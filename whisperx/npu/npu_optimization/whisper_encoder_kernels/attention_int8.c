/**
 * INT8 Attention Mechanism for Whisper Encoder
 * Optimized for AIE2 vector operations
 *
 * Implements: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 * For Whisper base:
 *   Sequence length: 1500 (30 seconds @ 16kHz with 10ms hop)
 *   Hidden dim: 512
 *   Heads: 8
 *   Head dim: 64 (512 / 8)
 *
 * This kernel processes ONE attention head on small tiles (16x16)
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
 * Softmax approximation for INT8
 * Uses lookup table for exp() and integer arithmetic
 *
 * Input: [N] int8 values (attention scores)
 * Output: [N] int8 values (probabilities summing to ~127)
 */
void softmax_int8_16(const int8_t* input, int8_t* output, uint32_t N) {
    // Find max value for numerical stability
    int8_t max_val = input[0];
    for (uint32_t i = 1; i < N; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Compute exp(x - max) using approximation
    // For INT8, we use: exp(x) ≈ 1 + x + x²/2 (Taylor series, 2 terms)
    int32_t sum = 0;
    int32_t exp_vals[16];  // Temporary storage (max 16 elements for memory)

    for (uint32_t i = 0; i < N; i++) {
        int32_t x = input[i] - max_val;  // Shift for stability

        // Clamp to prevent overflow
        if (x < -64) x = -64;
        if (x > 0) x = 0;  // exp(x - max) where x <= max, so always <= 0

        // Approximate exp: 64 * (1 + x/64 + x²/8192)
        // For x in [-64, 0], this gives values in [~0, 64]
        int32_t exp_val = 64 + x + (x * x) / 128;
        if (exp_val < 0) exp_val = 0;

        exp_vals[i] = exp_val;
        sum += exp_val;
    }

    // Normalize: output[i] = exp_vals[i] * 127 / sum
    for (uint32_t i = 0; i < N; i++) {
        if (sum > 0) {
            int32_t normalized = (exp_vals[i] * 127) / sum;
            if (normalized > 127) normalized = 127;
            output[i] = (int8_t)normalized;
        } else {
            output[i] = 0;
        }
    }
}

/**
 * Scaled dot-product attention for 16x16 tiles - COMBINED QKV VERSION
 *
 * Due to Phoenix NPU DMA channel limits (2 per ShimNOC tile),
 * we take a single combined buffer containing Q, K, V
 *
 * QKV_combined layout:
 *   Bytes 0-255: Q matrix [16 x 16]
 *   Bytes 256-511: K matrix [16 x 16]
 *   Bytes 512-767: V matrix [16 x 16]
 */
void attention_16x16(
    const int8_t* QKV_combined,  // [768] combined Q+K+V buffer
    int8_t* output,              // [16 x 16] output matrix
    uint32_t scale_shift         // Right shift for Q@K^T (divide by sqrt(d_k))
) {
    // Unpack Q, K, V from combined buffer
    const int8_t* Q = &QKV_combined[0];    // Bytes 0-255
    const int8_t* K = &QKV_combined[256];  // Bytes 256-511
    const int8_t* V = &QKV_combined[512];  // Bytes 512-767
    // Step 1: Compute attention scores: Q @ K^T
    // Result: [16 x 16] scores
    int8_t scores[256];  // 16x16 = 256 elements

    for (uint32_t i = 0; i < 16; i++) {
        for (uint32_t j = 0; j < 16; j++) {
            int32_t score = 0;

            // Dot product: Q[i, :] @ K[j, :]
            for (uint32_t k = 0; k < 16; k++) {
                score += (int32_t)Q[i * 16 + k] * (int32_t)K[j * 16 + k];
            }

            // Scale by 1/sqrt(d_k)
            score >>= scale_shift;

            // Clamp to INT8 range
            if (score > 127) score = 127;
            if (score < -128) score = -128;

            scores[i * 16 + j] = (int8_t)score;
        }
    }

    // Step 2: Apply softmax row-wise (each query attends to all keys)
    int8_t attention_weights[256];
    for (uint32_t i = 0; i < 16; i++) {
        softmax_int8_16(&scores[i * 16], &attention_weights[i * 16], 16);
    }

    // Step 3: Weighted sum: attention_weights @ V
    for (uint32_t i = 0; i < 16; i++) {
        for (uint32_t j = 0; j < 16; j++) {
            int32_t weighted_sum = 0;

            // Weighted sum: sum(attention_weights[i, k] * V[k, j])
            for (uint32_t k = 0; k < 16; k++) {
                weighted_sum += (int32_t)attention_weights[i * 16 + k] * (int32_t)V[k * 16 + j];
            }

            // Requantize to INT8 (divide by 127 since weights sum to ~127)
            weighted_sum >>= 7;  // Divide by 128 (close to 127)

            if (weighted_sum > 127) weighted_sum = 127;
            if (weighted_sum < -128) weighted_sum = -128;

            output[i * 16 + j] = (int8_t)weighted_sum;
        }
    }
}

/**
 * Simplified attention for testing: just Q @ K^T (no softmax)
 * Useful for validating the pipeline before adding complexity
 */
void attention_scores_only_16x16(
    const int8_t* Q,      // [16 x 16]
    const int8_t* K,      // [16 x 16]
    int8_t* scores,       // [16 x 16] output
    uint32_t scale_shift  // Scaling factor
) {
    for (uint32_t i = 0; i < 16; i++) {
        for (uint32_t j = 0; j < 16; j++) {
            int32_t score = 0;
            for (uint32_t k = 0; k < 16; k++) {
                score += (int32_t)Q[i * 16 + k] * (int32_t)K[j * 16 + k];
            }
            score >>= scale_shift;
            if (score > 127) score = 127;
            if (score < -128) score = -128;
            scores[i * 16 + j] = (int8_t)score;
        }
    }
}

/**
 * Multi-head attention (processes 2 heads in parallel) - COMBINED VERSION
 * Each head: 16x16 matrices
 *
 * QKV_combined_2heads layout:
 *   Head 0: Bytes 0-767 (Q:0-255, K:256-511, V:512-767)
 *   Head 1: Bytes 768-1535 (Q:768-1023, K:1024-1279, V:1280-1535)
 */
void multi_head_attention_2heads(
    const int8_t* QKV_combined_2heads,  // [2 × 768] for 2 heads
    int8_t* output,                      // [2 × 16 × 16] outputs for 2 heads
    uint32_t scale_shift                 // Scaling factor
) {
    // Process head 0 (bytes 0-767)
    attention_16x16(&QKV_combined_2heads[0], &output[0], scale_shift);

    // Process head 1 (bytes 768-1535)
    attention_16x16(&QKV_combined_2heads[768], &output[256], scale_shift);
}
