#include "bfp16_converter.hpp"
#include <cstring>
#include <iostream>

namespace whisper_xdna2 {
namespace bfp16 {

void fp32_to_bfp16(
    const Eigen::MatrixXf& input,
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& output
) {
    const int rows = input.rows();
    const int cols = input.cols();

    // Validate dimensions (must be multiples of 8)
    if (rows % BFP16Config::BLOCK_SIZE != 0) {
        throw std::invalid_argument(
            "Input rows (" + std::to_string(rows) +
            ") must be a multiple of " + std::to_string(BFP16Config::BLOCK_SIZE)
        );
    }
    if (cols % BFP16Config::BLOCK_SIZE != 0) {
        throw std::invalid_argument(
            "Input cols (" + std::to_string(cols) +
            ") must be a multiple of " + std::to_string(BFP16Config::BLOCK_SIZE)
        );
    }

    // Calculate output dimensions
    const int blocks_per_row = cols / BFP16Config::BLOCK_SIZE;
    const int output_cols = blocks_per_row * BFP16Config::BYTES_PER_ROW;

    // Allocate output buffer
    output.resize(rows, output_cols);
    output.setZero();

    // Process 8x8 blocks
    for (int block_row = 0; block_row < rows; block_row += BFP16Config::BLOCK_SIZE) {
        for (int block_col = 0; block_col < cols; block_col += BFP16Config::BLOCK_SIZE) {

            // Extract 8x8 block
            auto block = input.block<BFP16Config::BLOCK_SIZE, BFP16Config::BLOCK_SIZE>(
                block_row, block_col
            );

            // Find shared exponent for this block
            uint8_t shared_exp = find_block_exponent(block);

            // Calculate output offset for this block
            int block_idx = block_col / BFP16Config::BLOCK_SIZE;
            int out_col_start = block_idx * BFP16Config::BYTES_PER_ROW;

            // Quantize each row of the block
            for (int row = 0; row < BFP16Config::BLOCK_SIZE; row++) {
                int out_row = block_row + row;

                // Quantize 8 values in this row
                for (int col = 0; col < BFP16Config::BLOCK_SIZE; col++) {
                    float value = block(row, col);
                    uint8_t mantissa = quantize_mantissa(value, shared_exp);
                    output(out_row, out_col_start + col) = mantissa;
                }

                // Store shared exponent at end of row
                output(out_row, out_col_start + BFP16Config::BLOCK_SIZE) = shared_exp;
            }
        }
    }
}

void bfp16_to_fp32(
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
    Eigen::MatrixXf& output,
    size_t rows,
    size_t cols
) {
    // Validate dimensions
    if (rows % BFP16Config::BLOCK_SIZE != 0) {
        throw std::invalid_argument(
            "Rows (" + std::to_string(rows) +
            ") must be a multiple of " + std::to_string(BFP16Config::BLOCK_SIZE)
        );
    }
    if (cols % BFP16Config::BLOCK_SIZE != 0) {
        throw std::invalid_argument(
            "Cols (" + std::to_string(cols) +
            ") must be a multiple of " + std::to_string(BFP16Config::BLOCK_SIZE)
        );
    }

    // Calculate expected input dimensions
    const int blocks_per_row = cols / BFP16Config::BLOCK_SIZE;
    const int expected_input_cols = blocks_per_row * BFP16Config::BYTES_PER_ROW;

    if (input.rows() != static_cast<int>(rows)) {
        throw std::invalid_argument(
            "Input rows (" + std::to_string(input.rows()) +
            ") does not match expected (" + std::to_string(rows) + ")"
        );
    }
    if (input.cols() != expected_input_cols) {
        throw std::invalid_argument(
            "Input cols (" + std::to_string(input.cols()) +
            ") does not match expected (" + std::to_string(expected_input_cols) + ")"
        );
    }

    // Allocate output
    output.resize(rows, cols);
    output.setZero();

    // Process 8x8 blocks
    for (size_t block_row = 0; block_row < rows; block_row += BFP16Config::BLOCK_SIZE) {
        for (size_t block_col = 0; block_col < cols; block_col += BFP16Config::BLOCK_SIZE) {

            int block_idx = block_col / BFP16Config::BLOCK_SIZE;
            int in_col_start = block_idx * BFP16Config::BYTES_PER_ROW;

            // Process each row of the block
            for (size_t row = 0; row < BFP16Config::BLOCK_SIZE; row++) {
                size_t in_row = block_row + row;

                // Read shared exponent for this row
                uint8_t shared_exp = input(in_row, in_col_start + BFP16Config::BLOCK_SIZE);

                // Dequantize 8 values in this row
                for (size_t col = 0; col < BFP16Config::BLOCK_SIZE; col++) {
                    uint8_t mantissa = input(in_row, in_col_start + col);

                    // Convert mantissa to signed int8
                    int8_t mantissa_signed = static_cast<int8_t>(mantissa);

                    // Handle zero mantissa
                    if (mantissa_signed == 0) {
                        output(block_row + row, block_col + col) = 0.0f;
                        continue;
                    }

                    // Convert shared exponent back to scale factor
                    int block_exp_unbiased = static_cast<int>(shared_exp) - BFP16Config::EXPONENT_BIAS;
                    float block_scale = std::ldexp(1.0f, block_exp_unbiased + 1);

                    // Dequantize: mantissa_signed is in [-128, 127], normalized to [-1, 1]
                    float mantissa_f = static_cast<float>(mantissa_signed) / 127.0f;

                    // Reconstruct FP32 value
                    float value = mantissa_f * block_scale;

                    output(block_row + row, block_col + col) = value;
                }
            }
        }
    }
}

void shuffle_for_npu(
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& output,
    size_t rows,
    size_t cols_bytes
) {
    // Validate dimensions
    if (input.rows() != static_cast<int>(rows)) {
        throw std::invalid_argument("Input rows mismatch");
    }
    if (input.cols() != static_cast<int>(cols_bytes)) {
        throw std::invalid_argument("Input cols mismatch");
    }

    // Allocate output (same size as input)
    output.resize(rows, cols_bytes);
    output.setZero();

    // Shuffle parameters (from mm_bfp.cc lines 30-66)
    const size_t subtile_width = BFP16Config::BLOCK_SIZE * BFP16Config::STORAGE_RATIO;  // 9 bytes
    const size_t subtile_height = BFP16Config::BLOCK_SIZE;  // 8 rows

    size_t tile_counting_index = 0;

    // Iterate over 8x9 subtiles
    for (size_t subtile_start_y = 0; subtile_start_y < rows; subtile_start_y += subtile_height) {
        for (size_t subtile_start_x = 0; subtile_start_x < cols_bytes; subtile_start_x += subtile_width) {

            // Process each element in the subtile
            for (size_t i = 0; i < subtile_height; i++) {
                for (size_t j = 0; j < subtile_width; j++) {
                    size_t input_y = subtile_start_y + i;
                    size_t input_x = subtile_start_x + j;

                    // Bounds check
                    if (input_y >= rows || input_x >= cols_bytes) continue;

                    // Calculate shuffled output position
                    size_t output_x = tile_counting_index % cols_bytes;
                    size_t output_y = tile_counting_index / cols_bytes;

                    // Bounds check for output
                    if (output_y >= rows || output_x >= cols_bytes) continue;

                    // Copy byte
                    output(output_y, output_x) = input(input_y, input_x);

                    tile_counting_index++;
                }
            }
        }
    }
}

void unshuffle_from_npu(
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& output,
    size_t rows,
    size_t cols_bytes
) {
    // Validate dimensions
    if (input.rows() != static_cast<int>(rows)) {
        throw std::invalid_argument("Input rows mismatch");
    }
    if (input.cols() != static_cast<int>(cols_bytes)) {
        throw std::invalid_argument("Input cols mismatch");
    }

    // Allocate output (same size as input)
    output.resize(rows, cols_bytes);
    output.setZero();

    // Unshuffle parameters (same as shuffle but reversed)
    const size_t subtile_width = BFP16Config::BLOCK_SIZE * BFP16Config::STORAGE_RATIO;  // 9 bytes
    const size_t subtile_height = BFP16Config::BLOCK_SIZE;  // 8 rows

    size_t tile_counting_index = 0;

    // Iterate over 8x9 subtiles
    for (size_t subtile_start_y = 0; subtile_start_y < rows; subtile_start_y += subtile_height) {
        for (size_t subtile_start_x = 0; subtile_start_x < cols_bytes; subtile_start_x += subtile_width) {

            // Process each element in the subtile
            for (size_t i = 0; i < subtile_height; i++) {
                for (size_t j = 0; j < subtile_width; j++) {
                    size_t output_y = subtile_start_y + i;
                    size_t output_x = subtile_start_x + j;

                    // Bounds check
                    if (output_y >= rows || output_x >= cols_bytes) continue;

                    // Calculate shuffled input position
                    size_t input_x = tile_counting_index % cols_bytes;
                    size_t input_y = tile_counting_index / cols_bytes;

                    // Bounds check for input
                    if (input_y >= rows || input_x >= cols_bytes) continue;

                    // Copy byte (reversed from shuffle)
                    output(output_y, output_x) = input(input_y, input_x);

                    tile_counting_index++;
                }
            }
        }
    }
}

} // namespace bfp16
} // namespace whisper_xdna2
