#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace whisper_xdna2 {

/**
 * MultiHeadAttention - Multi-head self-attention mechanism
 *
 * Implements scaled dot-product attention with multiple heads:
 * 1. Split Q, K, V into multiple heads
 * 2. For each head: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
 * 3. Concatenate heads
 */
class MultiHeadAttention {
public:
    /**
     * Constructor
     *
     * @param n_heads Number of attention heads
     * @param head_dim Dimension of each head
     */
    MultiHeadAttention(size_t n_heads, size_t head_dim);

    /**
     * Forward pass of multi-head attention
     *
     * @param Q Query matrix (seq_len, n_state)
     * @param K Key matrix (seq_len, n_state)
     * @param V Value matrix (seq_len, n_state)
     * @param output Output matrix (seq_len, n_state) - preallocated
     */
    void forward(
        const Eigen::MatrixXf& Q,
        const Eigen::MatrixXf& K,
        const Eigen::MatrixXf& V,
        Eigen::MatrixXf& output
    );

    /**
     * Scaled dot-product attention for a single head
     *
     * @param Q_head Query for this head (seq_len, head_dim)
     * @param K_head Key for this head (seq_len, head_dim)
     * @param V_head Value for this head (seq_len, head_dim)
     * @param output Head output (seq_len, head_dim) - preallocated
     */
    void attention_head(
        const Eigen::MatrixXf& Q_head,
        const Eigen::MatrixXf& K_head,
        const Eigen::MatrixXf& V_head,
        Eigen::MatrixXf& output
    );

    /**
     * Compute attention scores (QK^T / sqrt(d_k))
     *
     * @param Q Query matrix (seq_len, head_dim)
     * @param K Key matrix (seq_len, head_dim)
     * @param scores Output scores (seq_len, seq_len) - preallocated
     */
    void compute_attention_scores(
        const Eigen::MatrixXf& Q,
        const Eigen::MatrixXf& K,
        Eigen::MatrixXf& scores
    );

    /**
     * Apply softmax along last dimension
     *
     * @param scores Attention scores (seq_len, seq_len) - modified in-place
     */
    static void apply_softmax(Eigen::MatrixXf& scores);

    /**
     * Apply softmax to a single row
     *
     * @param row Row to apply softmax to (modified in-place)
     */
    static void apply_softmax_row(Eigen::VectorXf& row);

private:
    size_t n_heads_;
    size_t head_dim_;
    float scale_;  // 1 / sqrt(head_dim)

    // Working buffers (reused across calls to avoid allocations)
    std::vector<Eigen::MatrixXf> head_outputs_;
    Eigen::MatrixXf scores_;
    Eigen::MatrixXf attn_weights_;
};

/**
 * Attention helper functions
 */
namespace attention_helpers {
    /**
     * Reshape tensor for multi-head attention
     *
     * Reshape from (seq_len, n_state) to (n_heads, seq_len, head_dim)
     *
     * @param input Input matrix (seq_len, n_state)
     * @param n_heads Number of heads
     * @param head_dim Dimension per head
     * @return Vector of head matrices, each (seq_len, head_dim)
     */
    inline std::vector<Eigen::MatrixXf> split_heads(
        const Eigen::MatrixXf& input,
        size_t n_heads,
        size_t head_dim
    ) {
        const size_t seq_len = input.rows();
        std::vector<Eigen::MatrixXf> heads;
        heads.reserve(n_heads);

        // Split along the hidden dimension
        for (size_t h = 0; h < n_heads; ++h) {
            Eigen::MatrixXf head(seq_len, head_dim);
            // Copy columns for this head
            head = input.middleCols(h * head_dim, head_dim);
            heads.push_back(head);
        }

        return heads;
    }

    /**
     * Concatenate heads back to original shape
     *
     * Concatenate from (n_heads, seq_len, head_dim) to (seq_len, n_state)
     *
     * @param heads Vector of head matrices, each (seq_len, head_dim)
     * @param output Output matrix (seq_len, n_state) - preallocated
     */
    inline void concatenate_heads(
        const std::vector<Eigen::MatrixXf>& heads,
        Eigen::MatrixXf& output
    ) {
        const size_t seq_len = heads[0].rows();
        const size_t head_dim = heads[0].cols();
        const size_t n_heads = heads.size();

        // Resize output if needed
        if (output.rows() != seq_len || output.cols() != n_heads * head_dim) {
            output.resize(seq_len, n_heads * head_dim);
        }

        // Concatenate heads along the hidden dimension
        for (size_t h = 0; h < n_heads; ++h) {
            output.middleCols(h * head_dim, head_dim) = heads[h];
        }
    }

    /**
     * Fast softmax for numerical stability
     *
     * softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
     */
    inline void stable_softmax(Eigen::VectorXf& x) {
        // Find max for numerical stability
        float max_val = x.maxCoeff();

        // Subtract max and exponentiate
        x = (x.array() - max_val).exp();

        // Normalize
        float sum = x.sum();
        x /= sum;
    }
}

} // namespace whisper_xdna2
