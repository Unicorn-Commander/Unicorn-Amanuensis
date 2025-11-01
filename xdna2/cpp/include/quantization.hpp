#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace whisper_xdna2 {

/**
 * Quantization configuration constants
 */
struct QuantizationConfig {
    static constexpr int8_t QUANT_MIN = -127;
    static constexpr int8_t QUANT_MAX = 127;
    static constexpr int8_t ZERO_POINT = 0;
    static constexpr float MIN_SCALE = 1e-10f;
};

/**
 * Quantizer - Handles INT8 symmetric quantization for NPU operations
 *
 * Implements symmetric per-tensor quantization:
 * - FP32 → INT8: quantized = round(tensor / scale).clip(-127, 127)
 * - INT32 → FP32: dequantized = tensor * scale_A * scale_B
 */
class Quantizer {
public:
    /**
     * Compute quantization scale for a tensor
     *
     * Scale is computed as: max(|min|, |max|) / 127
     *
     * @param tensor Input tensor (FP32)
     * @return Quantization scale
     */
    static float compute_scale(const Eigen::MatrixXf& tensor);

    /**
     * Quantize FP32 tensor to INT8
     *
     * @param input Input tensor (FP32)
     * @param output Output tensor (INT8) - preallocated
     * @param scale Quantization scale (output)
     */
    static void quantize_tensor(
        const Eigen::MatrixXf& input,
        Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>& output,
        float& scale
    );

    /**
     * Quantize FP32 tensor to INT8 with given scale
     *
     * @param input Input tensor (FP32)
     * @param output Output tensor (INT8) - preallocated
     * @param scale Quantization scale (input)
     */
    static void quantize_tensor_with_scale(
        const Eigen::MatrixXf& input,
        Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>& output,
        float scale
    );

    /**
     * Dequantize INT32 matmul output to FP32
     *
     * The NPU computes: C = A_int8 @ B_int8 (result is INT32)
     * To get FP32: C_fp32 = C_int32 * scale_A * scale_B
     *
     * @param input Matmul output from NPU (INT32)
     * @param output Dequantized output (FP32) - preallocated
     * @param input_scale Scale for input matrix
     * @param weight_scale Scale for weight matrix
     */
    static void dequantize_matmul_output(
        const Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>& input,
        Eigen::MatrixXf& output,
        float input_scale,
        float weight_scale
    );

    /**
     * Dequantize INT8 tensor to FP32
     *
     * @param input Quantized tensor (INT8)
     * @param output Dequantized output (FP32) - preallocated
     * @param scale Quantization scale
     */
    static void dequantize_tensor(
        const Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
        Eigen::MatrixXf& output,
        float scale
    );
};

/**
 * Fast quantization helpers (inline for performance)
 */
namespace quantization_helpers {
    /**
     * Fast round and clip for quantization
     */
    inline int8_t quantize_value(float value, float inv_scale) {
        int32_t rounded = static_cast<int32_t>(std::round(value * inv_scale));
        return static_cast<int8_t>(std::clamp(rounded,
            static_cast<int32_t>(QuantizationConfig::QUANT_MIN),
            static_cast<int32_t>(QuantizationConfig::QUANT_MAX)));
    }

    /**
     * Fast dequantize value
     */
    inline float dequantize_value(int8_t value, float scale) {
        return static_cast<float>(value) * scale;
    }

    /**
     * Fast dequantize matmul output value
     */
    inline float dequantize_matmul_value(int32_t value, float combined_scale) {
        return static_cast<float>(value) * combined_scale;
    }
}

} // namespace whisper_xdna2
