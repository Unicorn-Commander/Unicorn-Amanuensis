#include "bfp16_quantization.hpp"
#include "bfp16_converter.hpp"
#include <cstring>
#include <stdexcept>

namespace whisper_xdna2 {

// ============================================================================
// FP32 <-> BFP16 Conversion (using Phase 1 converter)
// ============================================================================

uint8_t BFP16Quantizer::find_block_exponent(const float* block_data) {
    // Find maximum absolute value in 8-value block
    float max_abs = 0.0f;
    for (int i = 0; i < BFP16Config::BLOCK_SIZE; i++) {
        max_abs = std::max(max_abs, std::abs(block_data[i]));
    }

    if (max_abs == 0.0f) {
        return 0; // All zeros, exponent = 0
    }

    // Extract exponent from FP32 using frexp (more reliable than bit manipulation)
    int exp;
    std::frexp(max_abs, &exp);

    // FP32 exponent with bias (-1 adjustment for mantissa range [0.5, 1.0))
    int biased_exp = exp - 1 + BFP16Config::EXPONENT_BIAS;

    // Clamp to [0, 255]
    return static_cast<uint8_t>(std::max(0, std::min(255, biased_exp)));
}

uint8_t BFP16Quantizer::quantize_to_8bit_mantissa(float value, uint8_t block_exponent) {
    // Handle special cases
    if (value == 0.0f || std::isnan(value)) return 0;
    if (std::isinf(value)) return value > 0 ? 127 : 128;

    // Convert block exponent to scale factor
    int block_exp_unbiased = static_cast<int>(block_exponent) - BFP16Config::EXPONENT_BIAS;
    float block_scale = std::ldexp(1.0f, -(block_exp_unbiased + 1));

    // Scale value by block scale to get mantissa in [-1.0, 1.0] range
    float scaled = value * block_scale;

    // Quantize to 8-bit signed range [-128, 127]
    // We use 127 for maximum range usage (symmetric quantization)
    int mantissa_int = static_cast<int>(std::round(scaled * 127.0f));
    mantissa_int = std::max(-128, std::min(127, mantissa_int));

    // Convert to uint8 (two's complement representation)
    return static_cast<uint8_t>(mantissa_int & 0xFF);
}

float BFP16Quantizer::dequantize_from_8bit_mantissa(uint8_t mantissa, uint8_t block_exponent) {
    // Handle zero mantissa
    if (mantissa == 0) {
        return 0.0f;
    }

    // Convert mantissa to signed int8 (two's complement)
    int8_t mantissa_signed = static_cast<int8_t>(mantissa);

    // Convert shared exponent back to scale factor
    int block_exp_unbiased = static_cast<int>(block_exponent) - BFP16Config::EXPONENT_BIAS;
    float block_scale = std::ldexp(1.0f, block_exp_unbiased + 1);

    // Dequantize: mantissa_signed is in [-128, 127], normalized to [-1, 1]
    float mantissa_f = static_cast<float>(mantissa_signed) / 127.0f;

    // Reconstruct FP32 value
    float value = mantissa_f * block_scale;

    return value;
}

void BFP16Quantizer::convert_to_bfp16(
    const Eigen::MatrixXf& input,
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& output
) {
    // Use the proven Phase 1 converter implementation
    bfp16::fp32_to_bfp16(input, output);
}

void BFP16Quantizer::convert_from_bfp16(
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
    Eigen::MatrixXf& output,
    size_t rows,
    size_t cols
) {
    // Use the proven Phase 1 converter implementation
    bfp16::bfp16_to_fp32(input, output, rows, cols);
}

// ============================================================================
// Shuffle / Unshuffle Operations
// ============================================================================

void BFP16Quantizer::shuffle_bfp16(
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& output,
    size_t rows,
    size_t cols_bytes
) {
    // Use the proven Phase 1 shuffle implementation
    bfp16::shuffle_for_npu(input, output, rows, cols_bytes);
}

void BFP16Quantizer::unshuffle_bfp16(
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& output,
    size_t rows,
    size_t cols_bytes
) {
    // Use the proven Phase 1 unshuffle implementation
    bfp16::unshuffle_from_npu(input, output, rows, cols_bytes);
}

// ============================================================================
// High-Level API
// ============================================================================

void BFP16Quantizer::prepare_for_npu(
    const Eigen::MatrixXf& input,
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& output
) {
    // All-in-one: Convert FP32 → BFP16 and shuffle

    // Step 1: Convert to BFP16
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> bfp16_temp;
    convert_to_bfp16(input, bfp16_temp);

    // Step 2: Shuffle for NPU layout
    shuffle_bfp16(bfp16_temp, output, input.rows(), bfp16_temp.cols());
}

void BFP16Quantizer::read_from_npu(
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
    Eigen::MatrixXf& output,
    size_t rows,
    size_t cols
) {
    // All-in-one: Unshuffle and convert BFP16 → FP32

    // Step 1: Unshuffle from NPU layout
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> bfp16_unshuffled;
    unshuffle_bfp16(input, bfp16_unshuffled, rows, input.cols());

    // Step 2: Convert to FP32
    convert_from_bfp16(bfp16_unshuffled, output, rows, cols);
}

} // namespace whisper_xdna2
