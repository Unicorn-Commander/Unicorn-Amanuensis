#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace whisper_xdna2 {

/**
 * FeedForward - Feed-forward network operations
 *
 * Implements:
 * - GELU activation (fast approximation)
 * - Layer normalization
 * - Residual connections
 */
class FeedForward {
public:
    /**
     * GELU activation (fast tanh approximation)
     *
     * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
     *
     * This is faster than the exact erf-based implementation and provides
     * nearly identical results for neural network applications.
     *
     * @param x Input/output matrix (modified in-place)
     */
    static void gelu(Eigen::MatrixXf& x);

    /**
     * GELU activation (non-modifying version)
     *
     * @param input Input matrix
     * @param output Output matrix (preallocated)
     */
    static void gelu(const Eigen::MatrixXf& input, Eigen::MatrixXf& output);

    /**
     * Layer normalization
     *
     * LayerNorm(x) = (x - mean) / sqrt(variance + eps) * weight + bias
     *
     * Normalizes across the last dimension (features).
     *
     * @param x Input/output matrix (seq_len, hidden_dim) - modified in-place
     * @param weight Scale parameters (hidden_dim,)
     * @param bias Shift parameters (hidden_dim,)
     * @param eps Small constant for numerical stability
     */
    static void layer_norm(
        Eigen::MatrixXf& x,
        const Eigen::VectorXf& weight,
        const Eigen::VectorXf& bias,
        float eps = 1e-5f
    );

    /**
     * Layer normalization (non-modifying version)
     *
     * @param input Input matrix (seq_len, hidden_dim)
     * @param output Output matrix (seq_len, hidden_dim) - preallocated
     * @param weight Scale parameters (hidden_dim,)
     * @param bias Shift parameters (hidden_dim,)
     * @param eps Small constant for numerical stability
     */
    static void layer_norm(
        const Eigen::MatrixXf& input,
        Eigen::MatrixXf& output,
        const Eigen::VectorXf& weight,
        const Eigen::VectorXf& bias,
        float eps = 1e-5f
    );

    /**
     * Add residual connection
     *
     * output = input + residual
     *
     * @param input Input matrix (modified in-place)
     * @param residual Residual to add
     */
    static void add_residual(Eigen::MatrixXf& input, const Eigen::MatrixXf& residual);
};

/**
 * Fast activation helpers (inline for performance)
 */
namespace activation_helpers {
    /**
     * Fast GELU for a single value
     */
    inline float gelu_value(float x) {
        // Constants for fast tanh approximation
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;  // sqrt(2/π)
        constexpr float coeff = 0.044715f;

        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        return 0.5f * x * (1.0f + std::tanh(inner));
    }

    /**
     * Fast tanh approximation (if needed for even more speed)
     * Currently we use std::tanh which is quite fast on modern CPUs
     */
    inline float fast_tanh(float x) {
        // Clamp to avoid overflow
        if (x >= 4.0f) return 1.0f;
        if (x <= -4.0f) return -1.0f;

        // Padé approximation: tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2)
        float x2 = x * x;
        return x * (27.0f + x2) / (27.0f + 9.0f * x2);
    }
}

} // namespace whisper_xdna2
