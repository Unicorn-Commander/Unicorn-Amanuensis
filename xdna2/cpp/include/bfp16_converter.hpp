#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace whisper_xdna2 {
namespace bfp16 {

/**
 * BFP16 Configuration Constants
 *
 * BFP16 (Block Floating Point 16) format stores 8x8 blocks where:
 * - Each block shares a common 8-bit exponent
 * - Each value has an 8-bit mantissa
 * - Total: 64 mantissas + 8 exponents (one per row) = 72 bytes per 8x8 block
 * - Storage ratio: 9 bytes per 8 elements = 1.125 bytes per element
 */
struct BFP16Config {
    static constexpr size_t BLOCK_SIZE = 8;           // 8x8 blocks
    static constexpr size_t BYTES_PER_MANTISSA = 1;   // 8-bit mantissa
    static constexpr size_t BYTES_PER_EXPONENT = 1;   // 8-bit exponent
    static constexpr size_t BYTES_PER_ROW = 9;        // 8 mantissas + 1 exponent
    static constexpr size_t BYTES_PER_BLOCK = 72;     // 8 rows × 9 bytes
    static constexpr int EXPONENT_BIAS = 127;         // FP32 exponent bias
    static constexpr float STORAGE_RATIO = 1.125f;    // 9/8 bytes per element
};

/**
 * Convert FP32 matrix to BFP16 format
 *
 * Converts a floating-point matrix to BFP16 block format where each 8x8 block
 * shares a common exponent. The output is stored in row-major order with
 * 9 bytes per row (8 mantissas + 1 exponent).
 *
 * Memory layout per 8x8 block (72 bytes):
 *   [row0: 8 mantissas | 1 exponent]  // 9 bytes
 *   [row1: 8 mantissas | 1 exponent]  // 9 bytes
 *   ...
 *   [row7: 8 mantissas | 1 exponent]  // 9 bytes
 *
 * @param input Input matrix (FP32), dimensions must be multiples of 8
 * @param output Output matrix (uint8), resized to (rows, cols * 1.125)
 * @throws std::invalid_argument if input dimensions are not multiples of 8
 */
void fp32_to_bfp16(
    const Eigen::MatrixXf& input,
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& output
);

/**
 * Convert BFP16 matrix to FP32 format
 *
 * Converts BFP16 block format back to floating-point. This is the inverse
 * operation of fp32_to_bfp16().
 *
 * @param input Input matrix (uint8) in BFP16 format
 * @param output Output matrix (FP32), resized to (rows, cols)
 * @param rows Number of rows in original FP32 matrix
 * @param cols Number of cols in original FP32 matrix
 * @throws std::invalid_argument if dimensions are invalid
 */
void bfp16_to_fp32(
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
    Eigen::MatrixXf& output,
    size_t rows,
    size_t cols
);

/**
 * Shuffle BFP16 matrix for NPU layout
 *
 * Rearranges BFP16 data to optimize for NPU DMA access patterns. The shuffle
 * operation reorders 8x8 subtiles to be contiguous in memory.
 *
 * Reference: scalarShuffleMatrixForBfp16ebs8() in mm_bfp.cc (lines 30-66)
 *
 * @param input Input matrix (uint8) in BFP16 row-major format
 * @param output Output matrix (uint8) in BFP16 shuffled format
 * @param rows Number of rows
 * @param cols_bytes Number of columns in bytes (already in BFP16 format)
 * @throws std::invalid_argument if dimensions are invalid
 */
void shuffle_for_npu(
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& output,
    size_t rows,
    size_t cols_bytes
);

/**
 * Unshuffle BFP16 matrix from NPU layout
 *
 * Restores row-major BFP16 layout from NPU-shuffled format. This is the
 * inverse operation of shuffle_for_npu().
 *
 * @param input Input matrix (uint8) in BFP16 shuffled format
 * @param output Output matrix (uint8) in BFP16 row-major format
 * @param rows Number of rows
 * @param cols_bytes Number of columns in bytes
 * @throws std::invalid_argument if dimensions are invalid
 */
void unshuffle_from_npu(
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& output,
    size_t rows,
    size_t cols_bytes
);

/**
 * Helper: Calculate BFP16 buffer size in bytes
 *
 * Given FP32 matrix dimensions, calculate the required BFP16 buffer size.
 *
 * @param rows FP32 matrix rows
 * @param cols FP32 matrix cols
 * @return Size in bytes for BFP16 buffer
 */
inline size_t calculate_bfp16_size(size_t rows, size_t cols) {
    size_t blocks_per_row = (cols + BFP16Config::BLOCK_SIZE - 1) /
                            BFP16Config::BLOCK_SIZE;
    size_t bytes_per_row = blocks_per_row * BFP16Config::BYTES_PER_ROW;
    return rows * bytes_per_row;
}

/**
 * Helper: Calculate BFP16 columns in bytes
 *
 * Given FP32 columns, calculate the BFP16 byte width.
 *
 * @param cols_fp32 FP32 matrix columns
 * @return BFP16 byte width
 */
inline size_t calculate_bfp16_cols(size_t cols_fp32) {
    size_t blocks = (cols_fp32 + BFP16Config::BLOCK_SIZE - 1) /
                    BFP16Config::BLOCK_SIZE;
    return blocks * BFP16Config::BYTES_PER_ROW;
}

/**
 * Helper: Extract 8-bit exponent from FP32 value
 *
 * @param value FP32 value
 * @return 8-bit exponent (biased by 127)
 */
inline uint8_t extract_exponent(float value) {
    if (value == 0.0f) return 0;

    int exp;
    std::frexp(std::abs(value), &exp);

    // FP32 exponent with bias
    int biased_exp = exp - 1 + BFP16Config::EXPONENT_BIAS;

    // Clamp to [0, 255]
    return static_cast<uint8_t>(std::max(0, std::min(255, biased_exp)));
}

/**
 * Helper: Find shared exponent for 8x8 block
 *
 * Finds the maximum exponent in the block to use as shared exponent.
 * All values in the block will be scaled relative to this exponent.
 *
 * @param block 8x8 FP32 block
 * @return 8-bit shared exponent
 */
inline uint8_t find_block_exponent(const Eigen::Block<const Eigen::MatrixXf, 8, 8>& block) {
    float max_abs = block.array().abs().maxCoeff();
    if (max_abs == 0.0f) return 0;

    int exp;
    std::frexp(max_abs, &exp);

    // FP32 exponent with bias
    int biased_exp = exp - 1 + BFP16Config::EXPONENT_BIAS;

    // Clamp to [0, 255]
    return static_cast<uint8_t>(std::max(0, std::min(255, biased_exp)));
}

/**
 * Helper: Quantize FP32 value to 8-bit signed mantissa
 *
 * Quantizes a single FP32 value to 8-bit signed mantissa using the
 * block's shared exponent.
 *
 * @param value FP32 value to quantize
 * @param block_exponent Shared exponent for the block
 * @return 8-bit signed mantissa (stored as uint8)
 */
inline uint8_t quantize_mantissa(float value, uint8_t block_exponent) {
    if (value == 0.0f || std::isnan(value)) return 0;
    if (std::isinf(value)) return value > 0 ? 127 : 128;

    // Convert block exponent to scale factor
    int block_exp_unbiased = static_cast<int>(block_exponent) - BFP16Config::EXPONENT_BIAS;
    float block_scale = std::ldexp(1.0f, -(block_exp_unbiased + 1));

    // Scale value by block scale to get mantissa in [-1.0, 1.0] range
    float scaled = value * block_scale;

    // Quantize to 8-bit signed range [-128, 127]
    // Note: We use 255 steps to maximize range usage
    int mantissa_int = static_cast<int>(std::round(scaled * 127.0f));
    mantissa_int = std::max(-128, std::min(127, mantissa_int));

    // Convert to uint8 (two's complement)
    return static_cast<uint8_t>(mantissa_int & 0xFF);
}

} // namespace bfp16
} // namespace whisper_xdna2
