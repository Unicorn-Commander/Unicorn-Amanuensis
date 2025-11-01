#include "ffn.hpp"

namespace whisper_xdna2 {

void FeedForward::gelu(Eigen::MatrixXf& x) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_pi = std::sqrt(2.0f / M_PI);
    x = 0.5f * x.array() * (1.0f + (sqrt_2_pi * (x.array() + 0.044715f * x.array().pow(3))).tanh());
}

void FeedForward::gelu(const Eigen::MatrixXf& input, Eigen::MatrixXf& output) {
    const float sqrt_2_pi = std::sqrt(2.0f / M_PI);
    output = 0.5f * input.array() * (1.0f + (sqrt_2_pi * (input.array() + 0.044715f * input.array().pow(3))).tanh());
}

void FeedForward::layer_norm(
    Eigen::MatrixXf& x,
    const Eigen::VectorXf& weight,
    const Eigen::VectorXf& bias,
    float eps
) {
    // LayerNorm(x) = (x - mean) / sqrt(variance + eps) * weight + bias
    // Normalize across feature dimension (columns)

    const int seq_len = x.rows();
    const int hidden_dim = x.cols();

    for (int i = 0; i < seq_len; ++i) {
        // Compute mean across features
        float mean = x.row(i).mean();

        // Subtract mean
        x.row(i).array() -= mean;

        // Compute variance
        float variance = x.row(i).array().square().mean();

        // Normalize: divide by sqrt(variance + eps)
        float inv_std = 1.0f / std::sqrt(variance + eps);
        x.row(i).array() *= inv_std;

        // Apply learned scale and bias
        x.row(i).array() *= weight.array();
        x.row(i).array() += bias.array();
    }
}

void FeedForward::layer_norm(
    const Eigen::MatrixXf& input,
    Eigen::MatrixXf& output,
    const Eigen::VectorXf& weight,
    const Eigen::VectorXf& bias,
    float eps
) {
    // Copy input to output, then apply in-place layer norm
    output = input;
    layer_norm(output, weight, bias, eps);
}

void FeedForward::add_residual(Eigen::MatrixXf& input, const Eigen::MatrixXf& residual) {
    input += residual;
}

} // namespace whisper_xdna2
