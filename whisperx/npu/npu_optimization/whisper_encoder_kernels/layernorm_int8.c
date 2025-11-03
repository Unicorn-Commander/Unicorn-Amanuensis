/**
 * INT8 Layer Normalization Kernel for Whisper Encoder
 * Optimized for AIE2 vector operations on AMD Phoenix NPU
 *
 * Implements: LayerNorm(x) = gamma * (x - mean) / sqrt(var + epsilon) + beta
 *
 * For Whisper base encoder:
 *   Hidden dim = 512
 *   12 encoder blocks, each with 2 LayerNorm layers = 24 total
 *   This is a critical operation for encoder performance
 *
 * Starting with 256 dimensions for testing (fits easily in AIE2 memory)
 * Can scale to 512 after validation
 */

#include <stdint.h>
#include <string.h>

/**
 * Fast integer square root using Newton-Raphson method
 * Used for computing 1/sqrt(variance + epsilon)
 *
 * For INT8 fixed-point arithmetic with Q7 format:
 *   Input: variance in range [0, 2^30] (accumulated squares)
 *   Output: sqrt(n) in same scale
 */
static inline uint32_t isqrt(uint32_t n) {
    if (n == 0) return 0;
    if (n == 1) return 1;

    // Newton-Raphson: x_{n+1} = (x_n + n/x_n) / 2
    uint32_t x = n;
    uint32_t y = (x + 1) / 2;

    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }

    return x;
}

/**
 * Compute reciprocal square root: 1/sqrt(x)
 * Using fixed-point Q7 format to match INT8 scale
 *
 * Algorithm:
 *   1. Compute sqrt(x) using isqrt
 *   2. Compute 1/sqrt(x) scaled appropriately for INT8
 *
 * For INT8 Q7 format, we scale by 128 (2^7)
 */
static inline uint32_t fixed_point_rsqrt(uint32_t x) {
    if (x == 0) return 128;  // Return 1.0 in Q7 format

    uint32_t sqrt_x = isqrt(x);
    if (sqrt_x == 0) return 128;

    // Compute 1/sqrt(x) in Q7 format (2^7 = 128)
    // Scale to maintain precision in INT8 range
    uint32_t rsqrt = (128 * 256) / sqrt_x;  // Multiply by 256 for intermediate precision

    return rsqrt;
}

/**
 * Layer Normalization for 256-dimensional vectors - COMBINED BUFFER VERSION
 * Uses INT8 arithmetic with fixed-point intermediate values
 *
 * Due to Phoenix NPU DMA channel limits (2 per ShimNOC tile),
 * we take a single combined buffer containing input, gamma, and beta
 *
 * Combined buffer layout:
 *   Bytes 0-255:   Input features (256 int8)
 *   Bytes 256-511: Gamma parameters (256 int8)
 *   Bytes 512-767: Beta parameters (256 int8)
 *
 * Algorithm:
 *   1. Compute mean (accumulate in int32, divide by N)
 *   2. Compute variance (accumulate squared differences)
 *   3. Compute 1/sqrt(var + epsilon) using fixed-point rsqrt
 *   4. Normalize: (x - mean) * rsqrt
 *   5. Scale and shift: gamma * normalized + beta
 *   6. Clamp to INT8 range [-128, 127]
 *
 * Memory layout:
 *   input_combined: [768] int8 values (input + gamma + beta)
 *   output:         [256] int8 values (normalized output)
 */
void layernorm_int8_256(
    const int8_t* input_combined,  // [768] combined buffer
    int8_t* output                 // [256] normalized output
) {
    // Unpack combined buffer
    const int8_t* input = &input_combined[0];    // Bytes 0-255
    const int8_t* gamma = &input_combined[256];  // Bytes 256-511
    const int8_t* beta = &input_combined[512];   // Bytes 512-767
    const uint32_t N = 256;
    const uint32_t EPSILON_FIXED = 1;  // Small epsilon for numerical stability

    // Step 1: Compute mean
    // Accumulate in int32 to avoid overflow
    int32_t sum = 0;
    for (uint32_t i = 0; i < N; i++) {
        sum += (int32_t)input[i];
    }

    // Mean in Q7 format (divide by N)
    int32_t mean = sum / (int32_t)N;

    // Step 2: Compute variance
    // var = E[(x - mean)^2]
    int64_t var_sum = 0;
    for (uint32_t i = 0; i < N; i++) {
        int32_t diff = (int32_t)input[i] - mean;
        var_sum += (int64_t)(diff * diff);
    }

    // Variance
    uint32_t variance = (uint32_t)(var_sum / (int64_t)N);

    // Step 3: Compute 1/sqrt(variance + epsilon)
    uint32_t std_inv = fixed_point_rsqrt(variance + EPSILON_FIXED);

    // Step 4: Normalize and scale
    // output[i] = gamma[i] * ((input[i] - mean) * std_inv) + beta[i]
    for (uint32_t i = 0; i < N; i++) {
        // Normalize: (x - mean) * std_inv
        int32_t centered = (int32_t)input[i] - mean;

        // Apply inverse std deviation (fixed-point multiply)
        // std_inv has scale factor 256, so shift by 8
        int32_t normalized = (centered * (int32_t)std_inv) >> 8;

        // Scale with gamma (both in Q7 format)
        // gamma is Q7, so shift by 7
        int32_t scaled = ((int32_t)gamma[i] * normalized) >> 7;

        // Add beta (bias term)
        int32_t result = scaled + (int32_t)beta[i];

        // Clamp to INT8 range [-128, 127]
        if (result > 127) result = 127;
        if (result < -128) result = -128;

        output[i] = (int8_t)result;
    }
}

/**
 * Layer Normalization for 512-dimensional vectors (Whisper base full size)
 * Same algorithm as 256-dim version but with larger N
 *
 * Note: 512 * 512 bytes = 262KB for all buffers (input + output + gamma + beta)
 * This fits in AIE2 tile memory (~32KB per buffer)
 */
void layernorm_int8_512(
    const int8_t* input,      // [512] input features
    int8_t* output,           // [512] normalized output
    const int8_t* gamma,      // [512] scale parameters
    const int8_t* beta        // [512] shift parameters
) {
    const uint32_t N = 512;
    const uint32_t EPSILON_FIXED = 128;  // epsilon = 1e-5 in Q15 format

    // Step 1: Compute mean
    int32_t sum = 0;
    for (uint32_t i = 0; i < N; i++) {
        sum += (int32_t)input[i];
    }
    int32_t mean = sum / (int32_t)N;

    // Step 2: Compute variance
    int64_t var_sum = 0;
    for (uint32_t i = 0; i < N; i++) {
        int32_t diff = (int32_t)input[i] - mean;
        var_sum += (int64_t)(diff * diff);
    }
    uint32_t variance = (uint32_t)(var_sum / (int64_t)N);

    // Step 3: Compute 1/sqrt(variance + epsilon)
    uint32_t std_inv = fixed_point_rsqrt(variance + EPSILON_FIXED);

    // Step 4: Normalize and scale
    for (uint32_t i = 0; i < N; i++) {
        int32_t centered = (int32_t)input[i] - mean;
        int32_t normalized = (centered * (int32_t)std_inv) >> 15;
        int32_t scaled = ((int32_t)gamma[i] * normalized) >> 7;
        int32_t result = scaled + (int32_t)beta[i];

        // Clamp to INT8 range
        if (result > 127) result = 127;
        if (result < -128) result = -128;

        output[i] = (int8_t)result;
    }
}

/**
 * Simplified layer norm for testing: compute only mean and variance
 * Useful for validating the statistics computation before full normalization
 *
 * Returns:
 *   mean and variance as int32 values (for debugging)
 */
void layernorm_stats_only_256(
    const int8_t* input,
    int32_t* mean_out,
    int32_t* var_out
) {
    const uint32_t N = 256;

    // Compute mean
    int32_t sum = 0;
    for (uint32_t i = 0; i < N; i++) {
        sum += (int32_t)input[i];
    }
    int32_t mean = sum / (int32_t)N;
    *mean_out = mean;

    // Compute variance
    int64_t var_sum = 0;
    for (uint32_t i = 0; i < N; i++) {
        int32_t diff = (int32_t)input[i] - mean;
        var_sum += (int64_t)(diff * diff);
    }
    *var_out = (int32_t)(var_sum / (int64_t)N);
}
