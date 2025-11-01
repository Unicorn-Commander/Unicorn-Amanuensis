#include "quantization.hpp"

namespace whisper_xdna2 {

float Quantizer::compute_scale(const Eigen::MatrixXf& tensor) {
    float max_val = tensor.cwiseAbs().maxCoeff();
    return std::max(max_val / 127.0f, QuantizationConfig::MIN_SCALE);
}

void Quantizer::quantize_tensor(
    const Eigen::MatrixXf& input,
    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>& output,
    float& scale
) {
    scale = compute_scale(input);
    output.resize(input.rows(), input.cols());

    float inv_scale = 1.0f / scale;
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            output(i, j) = quantization_helpers::quantize_value(input(i, j), inv_scale);
        }
    }
}

void Quantizer::quantize_tensor_with_scale(
    const Eigen::MatrixXf& input,
    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>& output,
    float scale
) {
    output.resize(input.rows(), input.cols());

    float inv_scale = 1.0f / scale;
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            output(i, j) = quantization_helpers::quantize_value(input(i, j), inv_scale);
        }
    }
}

void Quantizer::dequantize_matmul_output(
    const Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>& input,
    Eigen::MatrixXf& output,
    float input_scale,
    float weight_scale
) {
    float combined_scale = input_scale * weight_scale;
    output.resize(input.rows(), input.cols());

    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            output(i, j) = quantization_helpers::dequantize_matmul_value(input(i, j), combined_scale);
        }
    }
}

void Quantizer::dequantize_tensor(
    const Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>& input,
    Eigen::MatrixXf& output,
    float scale
) {
    output.resize(input.rows(), input.cols());

    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            output(i, j) = quantization_helpers::dequantize_value(input(i, j), scale);
        }
    }
}

} // namespace whisper_xdna2
