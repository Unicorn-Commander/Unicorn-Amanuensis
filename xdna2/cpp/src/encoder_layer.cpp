#include "encoder_layer.hpp"

namespace whisper_xdna2 {

EncoderLayer::EncoderLayer(
    size_t layer_idx,
    size_t n_heads,
    size_t n_state,
    size_t ffn_dim
)
    : layer_idx_(layer_idx)
    , n_heads_(n_heads)
    , n_state_(n_state)
    , ffn_dim_(ffn_dim)
    , head_dim_(n_state / n_heads)
    , attention_(std::make_unique<MultiHeadAttention>(n_heads, n_state / n_heads))
    , npu_callback_fn_(nullptr)
    , npu_user_data_(nullptr)
{
}

void EncoderLayer::load_weights(
    const Eigen::MatrixXf& q_weight,
    const Eigen::MatrixXf& k_weight,
    const Eigen::MatrixXf& v_weight,
    const Eigen::MatrixXf& out_weight,
    const Eigen::VectorXf& q_bias,
    const Eigen::VectorXf& k_bias,
    const Eigen::VectorXf& v_bias,
    const Eigen::VectorXf& out_bias,
    const Eigen::MatrixXf& fc1_weight,
    const Eigen::MatrixXf& fc2_weight,
    const Eigen::VectorXf& fc1_bias,
    const Eigen::VectorXf& fc2_bias,
    const Eigen::VectorXf& attn_ln_weight,
    const Eigen::VectorXf& attn_ln_bias,
    const Eigen::VectorXf& ffn_ln_weight,
    const Eigen::VectorXf& ffn_ln_bias
) {
    // Convert weights to BFP16 format
    BFP16Quantizer bfp16_quantizer;

    bfp16_quantizer.prepare_for_npu(q_weight, q_weight_bfp16_);
    bfp16_quantizer.prepare_for_npu(k_weight, k_weight_bfp16_);
    bfp16_quantizer.prepare_for_npu(v_weight, v_weight_bfp16_);
    bfp16_quantizer.prepare_for_npu(out_weight, out_weight_bfp16_);
    bfp16_quantizer.prepare_for_npu(fc1_weight, fc1_weight_bfp16_);
    bfp16_quantizer.prepare_for_npu(fc2_weight, fc2_weight_bfp16_);

    // Store biases (FP32)
    q_bias_ = q_bias;
    k_bias_ = k_bias;
    v_bias_ = v_bias;
    out_bias_ = out_bias;
    fc1_bias_ = fc1_bias;
    fc2_bias_ = fc2_bias;

    // Store layer norm parameters (FP32)
    attn_ln_weight_ = attn_ln_weight;
    attn_ln_bias_ = attn_ln_bias;
    ffn_ln_weight_ = ffn_ln_weight;
    ffn_ln_bias_ = ffn_ln_bias;
}

void EncoderLayer::set_npu_matmul(NPUMatmulFunction matmul_fn) {
    npu_matmul_fn_ = matmul_fn;
}

void EncoderLayer::set_npu_callback(void* callback, void* user_data) {
    npu_callback_fn_ = callback;
    npu_user_data_ = user_data;
}

void EncoderLayer::forward(
    const Eigen::MatrixXf& input,
    Eigen::MatrixXf& output
) {
    const int seq_len = input.rows();

    // Allocate working buffers if needed
    if (attn_output_.rows() != seq_len || attn_output_.cols() != n_state_) {
        attn_output_.resize(seq_len, n_state_);
        fc2_output_.resize(seq_len, n_state_);
    }

    // 1. Attention block: x = x + Attention(LayerNorm(x))
    run_attention(input, attn_output_);
    output = input + attn_output_;

    // 2. FFN block: x = x + FFN(LayerNorm(x))
    run_ffn(output, fc2_output_);
    output += fc2_output_;
}

void EncoderLayer::run_attention(
    const Eigen::MatrixXf& input,
    Eigen::MatrixXf& output
) {
    const int seq_len = input.rows();

    // Allocate buffers if needed
    if (ln_output_.rows() != seq_len || ln_output_.cols() != n_state_) {
        ln_output_.resize(seq_len, n_state_);
        Q_.resize(seq_len, n_state_);
        K_.resize(seq_len, n_state_);
        V_.resize(seq_len, n_state_);
    }

    // Layer norm
    FeedForward::layer_norm(input, ln_output_, attn_ln_weight_, attn_ln_bias_);

    // Q/K/V projections (NPU matmuls)
    run_npu_linear(ln_output_, q_weight_bfp16_, q_bias_, Q_);
    run_npu_linear(ln_output_, k_weight_bfp16_, k_bias_, K_);
    run_npu_linear(ln_output_, v_weight_bfp16_, v_bias_, V_);

    // Multi-head attention (CPU)
    Eigen::MatrixXf attn_heads(seq_len, n_state_);
    attention_->forward(Q_, K_, V_, attn_heads);

    // Output projection (NPU matmul)
    run_npu_linear(attn_heads, out_weight_bfp16_, out_bias_, output);
}

void EncoderLayer::run_ffn(
    const Eigen::MatrixXf& input,
    Eigen::MatrixXf& output
) {
    const int seq_len = input.rows();

    // Allocate buffers if needed
    if (ln_output_.rows() != seq_len || ln_output_.cols() != n_state_) {
        ln_output_.resize(seq_len, n_state_);
    }
    if (fc1_output_.rows() != seq_len || fc1_output_.cols() != ffn_dim_) {
        fc1_output_.resize(seq_len, ffn_dim_);
    }

    // Layer norm
    FeedForward::layer_norm(input, ln_output_, ffn_ln_weight_, ffn_ln_bias_);

    // FC1: (seq_len, n_state) @ (n_state, ffn_dim) -> (seq_len, ffn_dim)
    run_npu_linear(ln_output_, fc1_weight_bfp16_, fc1_bias_, fc1_output_);

    // GELU activation (CPU)
    FeedForward::gelu(fc1_output_);

    // FC2: (seq_len, ffn_dim) @ (ffn_dim, n_state) -> (seq_len, n_state)
    run_npu_linear(fc1_output_, fc2_weight_bfp16_, fc2_bias_, output);
}

void EncoderLayer::run_npu_linear(
    const Eigen::MatrixXf& input,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& weight_bfp16,
    const Eigen::VectorXf& bias,
    Eigen::MatrixXf& output
) {
    const size_t M = input.rows();
    const size_t K = input.cols();
    const size_t N = weight_bfp16.rows();

    // Create BFP16 quantizer
    BFP16Quantizer bfp16_quantizer;

    // Prepare input for NPU (convert to BFP16 + shuffle)
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> input_bfp16_shuffled;
    bfp16_quantizer.prepare_for_npu(input, input_bfp16_shuffled);

    // Allocate output buffer (BFP16 format, shuffled)
    // BFP16 requires 1.125× size (9 bytes per 8 values)
    const size_t output_cols_bfp16 = ((N + 7) / 8) * 9;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> output_bfp16_shuffled(M, output_cols_bfp16);

    // Call NPU callback
    if (npu_callback_fn_) {
        typedef int (*NPUCallback)(void*, const uint8_t*, const uint8_t*, uint8_t*, size_t, size_t, size_t);
        auto callback = reinterpret_cast<NPUCallback>(npu_callback_fn_);

        int result = callback(
            npu_user_data_,
            input_bfp16_shuffled.data(),
            const_cast<uint8_t*>(weight_bfp16.data()),
            output_bfp16_shuffled.data(),
            M, K, N
        );

        if (result != 0) {
            throw std::runtime_error("NPU callback failed");
        }
    } else {
        throw std::runtime_error("NPU callback not set");
    }

    // Convert NPU output back to FP32
    bfp16_quantizer.read_from_npu(output_bfp16_shuffled, output, M, N);

    // Add bias
    for (size_t i = 0; i < M; ++i) {
        output.row(i) += bias;
    }
}

} // namespace whisper_xdna2
