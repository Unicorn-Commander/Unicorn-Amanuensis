#include "attention.hpp"

namespace whisper_xdna2 {

MultiHeadAttention::MultiHeadAttention(size_t n_heads, size_t head_dim)
    : n_heads_(n_heads)
    , head_dim_(head_dim)
    , scale_(1.0f / std::sqrt(static_cast<float>(head_dim)))
{
    // Pre-allocate head output buffers
    head_outputs_.resize(n_heads_);
}

void MultiHeadAttention::forward(
    const Eigen::MatrixXf& Q,
    const Eigen::MatrixXf& K,
    const Eigen::MatrixXf& V,
    Eigen::MatrixXf& output
) {
    const int seq_len = Q.rows();
    const int n_state = Q.cols();

    // Process each head
    for (size_t h = 0; h < n_heads_; ++h) {
        // Extract head slice (seq_len, head_dim)
        int head_start = h * head_dim_;

        Eigen::MatrixXf Q_head = Q.block(0, head_start, seq_len, head_dim_);
        Eigen::MatrixXf K_head = K.block(0, head_start, seq_len, head_dim_);
        Eigen::MatrixXf V_head = V.block(0, head_start, seq_len, head_dim_);

        // Allocate head output if needed
        if (head_outputs_[h].rows() != seq_len || head_outputs_[h].cols() != head_dim_) {
            head_outputs_[h].resize(seq_len, head_dim_);
        }

        // Compute attention for this head
        attention_head(Q_head, K_head, V_head, head_outputs_[h]);
    }

    // Concatenate head outputs
    for (size_t h = 0; h < n_heads_; ++h) {
        int head_start = h * head_dim_;
        output.block(0, head_start, seq_len, head_dim_) = head_outputs_[h];
    }
}

void MultiHeadAttention::attention_head(
    const Eigen::MatrixXf& Q_head,
    const Eigen::MatrixXf& K_head,
    const Eigen::MatrixXf& V_head,
    Eigen::MatrixXf& output
) {
    const int seq_len = Q_head.rows();

    // Allocate scores matrix if needed
    if (scores_.rows() != seq_len || scores_.cols() != seq_len) {
        scores_.resize(seq_len, seq_len);
    }

    // Compute attention scores
    compute_attention_scores(Q_head, K_head, scores_);

    // Apply softmax
    apply_softmax(scores_);

    // Multiply by V: output = softmax(scores) * V
    output = scores_ * V_head;
}

void MultiHeadAttention::compute_attention_scores(
    const Eigen::MatrixXf& Q,
    const Eigen::MatrixXf& K,
    Eigen::MatrixXf& scores
) {
    // scores = Q * K^T / sqrt(d_k)
    scores = (Q * K.transpose()) * scale_;
}

void MultiHeadAttention::apply_softmax(Eigen::MatrixXf& scores) {
    // Apply softmax row-by-row
    for (int i = 0; i < scores.rows(); ++i) {
        Eigen::VectorXf row = scores.row(i);
        apply_softmax_row(row);
        scores.row(i) = row;
    }
}

void MultiHeadAttention::apply_softmax_row(Eigen::VectorXf& row) {
    // Numerically stable softmax: subtract max first
    float max_val = row.maxCoeff();
    row.array() -= max_val;
    row = row.array().exp();
    float sum = row.sum();
    row /= sum;
}

} // namespace whisper_xdna2
