#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace whisper_xdna2 {

/**
 * BFP16 Configuration Constants
 *
 * BFP16 (Block Floating Point 16) format:
 * - Block size: 8x8 elements (64 values per block)
 * - Encoding: Shared 8-bit exponent per block
 * - Storage: 8 bits per mantissa + 1 byte exponent per 8 values
 * - Total: 9 bytes per 8 values = 1.125 bytes per value
 */
struct BFP16Config {
    static constexpr size_t BLOCK_SIZE = 8;           // 8 values share 1 exponent
    static constexpr size_t BYTES_PER_MANTISSA = 1;   // 8-bit mantissa
    static constexpr size_t BYTES_PER_EXPONENT = 1;   // 8-bit shared exponent
    static constexpr size_t BYTES_PER_BLOCK = 9;      // 8 mantissas + 1 exponent
    static constexpr int EXPONENT_BIAS = 127;         // For uint8 encoding
    static constexpr float STORAGE_MULTIPLIER = 1.125f; // 9/8 ratio
};

/**
 * BFP16Quantizer - Handles BFP16 conversion for NPU operations
 *
 * Replaces INT8 quantization with BFP16 block floating point format.
 *
 * Key differences from INT8:
 * - No per-tensor scale needed (scales embedded in exponents)
 * - Block-based quantization (8x8 blocks)
 * - Requires shuffle/unshuffle for NPU layout
 * - Higher accuracy (~99% vs 64.6% for INT8)
 * - Slightly more memory (1.125x vs 1.0x)
 *
 * Usage:
 *   BFP16Quantizer quantizer;
 *
 *   // Convert weights to BFP16 (with shuffle)
 *   quantizer.convert_to_bfp16(weight_fp32, weight_bfp16);
 *
 *   // Convert activations to BFP16 (with shuffle)
 *   quantizer.prepare_for_npu(input_fp32, input_bfp16_shuffled);
 *
 *   // Run NPU matmul (BFP16 @ BFP16 -> BFP16)
 *   npu_matmul(input_bfp16_shuffled, weight_bfp16, output_bfp16_shuffled);
 *
 *   // Convert output back to FP32 (with unshuffle)
 *   quantizer.read_from_npu(output_bfp16_shuffled, output_fp32, M, N);
 */
class BFP16Quantizer {
public:
    /**
     * Convert FP32 tensor to BFP16 format (no shuffle)
     *
     * Converts FP32 matrix to BFP16 format with block-based exponents.
     * Output size: rows × (cols * 1.125) bytes
     *
     * @param input Input tensor (FP32)
     * @param output Output tensor (BFP16 as uint8) - will be resized
     */
    static void convert_to_bfp16(
        const Eigen::MatrixXf& input,
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& output
    );

    /**
     * Convert BFP16 tensor to FP32 format (no unshuffle)
     *
     * Converts BFP16 matrix back to FP32 format.
     *
     * @param input Input tensor (BFP16 as uint8)
     * @param output Output tensor (FP32) - will be resized
     * @param rows Original FP32 rows
     * @param cols Original FP32 cols
     */
    static void convert_from_bfp16(
        const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
        Eigen::MatrixXf& output,
        size_t rows,
        size_t cols
    );

    /**
     * Shuffle BFP16 matrix for NPU layout
     *
     * Rearranges BFP16 data so each 8x8 subtile is contiguous in memory.
     * This enables efficient DMA transfers to AIE cores.
     *
     * Reference: scalarShuffleMatrixForBfp16ebs8() in mm_bfp.cc
     *
     * @param input Input matrix (BFP16, row-major)
     * @param output Output matrix (BFP16, shuffled) - will be resized
     * @param rows Number of rows
     * @param cols_bytes Number of columns in bytes (already in BFP16 format)
     */
    static void shuffle_bfp16(
        const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& output,
        size_t rows,
        size_t cols_bytes
    );

    /**
     * Unshuffle BFP16 matrix from NPU layout
     *
     * Reverses the shuffle operation to restore row-major layout.
     *
     * @param input Input matrix (BFP16, shuffled)
     * @param output Output matrix (BFP16, row-major) - will be resized
     * @param rows Number of rows
     * @param cols_bytes Number of columns in bytes
     */
    static void unshuffle_bfp16(
        const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& output,
        size_t rows,
        size_t cols_bytes
    );

    /**
     * Prepare tensor for NPU: Convert FP32 → BFP16 and shuffle
     *
     * All-in-one function for input preparation:
     * 1. Convert FP32 → BFP16
     * 2. Shuffle for NPU layout
     *
     * @param input Input tensor (FP32)
     * @param output Output tensor (BFP16, shuffled) - will be resized
     */
    static void prepare_for_npu(
        const Eigen::MatrixXf& input,
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& output
    );

    /**
     * Read tensor from NPU: Unshuffle and convert BFP16 → FP32
     *
     * All-in-one function for output processing:
     * 1. Unshuffle from NPU layout
     * 2. Convert BFP16 → FP32
     *
     * @param input Input tensor (BFP16, shuffled)
     * @param output Output tensor (FP32) - will be resized
     * @param rows Original FP32 rows
     * @param cols Original FP32 cols
     */
    static void read_from_npu(
        const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
        Eigen::MatrixXf& output,
        size_t rows,
        size_t cols
    );

    /**
     * Calculate BFP16 buffer size for given FP32 dimensions
     *
     * @param rows FP32 rows
     * @param cols FP32 cols
     * @return Size in bytes for BFP16 buffer
     */
    static inline size_t calculate_bfp16_size(size_t rows, size_t cols) {
        size_t blocks_per_row = (cols + BFP16Config::BLOCK_SIZE - 1) / BFP16Config::BLOCK_SIZE;
        size_t bytes_per_row = blocks_per_row * BFP16Config::BYTES_PER_BLOCK;
        return rows * bytes_per_row;
    }

    /**
     * Calculate BFP16 columns (in bytes) for given FP32 columns
     */
    static inline size_t calculate_bfp16_cols(size_t cols_fp32) {
        size_t blocks = (cols_fp32 + BFP16Config::BLOCK_SIZE - 1) / BFP16Config::BLOCK_SIZE;
        return blocks * BFP16Config::BYTES_PER_BLOCK;
    }

private:
    /**
     * Find shared exponent for 8-value block
     *
     * @param block_data Pointer to 8 FP32 values
     * @return 8-bit shared exponent
     */
    static uint8_t find_block_exponent(const float* block_data);

    /**
     * Quantize single FP32 value to 8-bit mantissa with shared exponent
     *
     * @param value FP32 value to quantize
     * @param block_exponent Shared exponent for the block
     * @return 8-bit mantissa
     */
    static uint8_t quantize_to_8bit_mantissa(float value, uint8_t block_exponent);

    /**
     * Dequantize single 8-bit mantissa to FP32 with shared exponent
     *
     * @param mantissa 8-bit mantissa
     * @param block_exponent Shared exponent for the block
     * @return FP32 value
     */
    static float dequantize_from_8bit_mantissa(uint8_t mantissa, uint8_t block_exponent);
};

} // namespace whisper_xdna2
