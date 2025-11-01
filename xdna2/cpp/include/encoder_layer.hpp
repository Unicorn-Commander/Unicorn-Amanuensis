#pragma once

#include "attention.hpp"
#include "ffn.hpp"
#include "quantization.hpp"
#include "bfp16_quantization.hpp"
#include <Eigen/Dense>
#include <memory>
#include <functional>

namespace whisper_xdna2 {

// Forward declaration
class WhisperXDNA2Runtime;

/**
 * NPU matmul function type
 *
 * Signature: (A_int8, B_int8, M, K, N) -> C_int32
 */
using NPUMatmulFunction = std::function<void(
    const Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>&,  // A (M, K)
    const Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>&,  // B (K, N)
    Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>&        // C (M, N)
)>;

/**
 * EncoderLayer - Complete Whisper encoder transformer layer
 *
 * Architecture:
 * 1. Multi-head self-attention with residual
 *    x = x + Attention(LayerNorm(x))
 * 2. Feed-forward network with residual
 *    x = x + FFN(LayerNorm(x))
 *
 * Heavy matmuls (Q/K/V projections, output projection, FC1, FC2) are
 * executed on NPU using BFP16 quantization. Attention scores, softmax,
 * layer norm, and GELU are computed on CPU.
 */
class EncoderLayer {
public:
    /**
     * Constructor
     *
     * @param layer_idx Layer index (0-5 for Whisper Base)
     * @param n_heads Number of attention heads (8 for Whisper Base)
     * @param n_state Hidden dimension (512 for Whisper Base)
     * @param ffn_dim FFN intermediate dimension (2048 for Whisper Base)
     */
    EncoderLayer(
        size_t layer_idx,
        size_t n_heads,
        size_t n_state,
        size_t ffn_dim
    );

    /**
     * Load weights for this layer
     *
     * @param q_weight Query projection weight (n_state, n_state)
     * @param k_weight Key projection weight (n_state, n_state)
     * @param v_weight Value projection weight (n_state, n_state)
     * @param out_weight Output projection weight (n_state, n_state)
     * @param q_bias Query bias (n_state,)
     * @param k_bias Key bias (n_state,)
     * @param v_bias Value bias (n_state,)
     * @param out_bias Output bias (n_state,)
     * @param fc1_weight First FFN weight (ffn_dim, n_state)
     * @param fc2_weight Second FFN weight (n_state, ffn_dim)
     * @param fc1_bias First FFN bias (ffn_dim,)
     * @param fc2_bias Second FFN bias (n_state,)
     * @param attn_ln_weight Attention layer norm weight (n_state,)
     * @param attn_ln_bias Attention layer norm bias (n_state,)
     * @param ffn_ln_weight FFN layer norm weight (n_state,)
     * @param ffn_ln_bias FFN layer norm bias (n_state,)
     */
    void load_weights(
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
    );

    /**
     * Set NPU matmul function
     *
     * This function will be called to execute matmuls on the NPU.
     *
     * @param matmul_fn NPU matmul function
     */
    void set_npu_matmul(NPUMatmulFunction matmul_fn);

    /**
     * Set NPU callback (C-style function pointer for Python)
     *
     * @param callback NPU matmul callback function
     * @param user_data User data pointer (e.g., Python runtime object)
     */
    void set_npu_callback(void* callback, void* user_data);

    /**
     * Forward pass
     *
     * Computes complete encoder layer:
     * 1. x = x + Attention(LayerNorm(x))
     * 2. x = x + FFN(LayerNorm(x))
     *
     * @param input Input (seq_len, n_state)
     * @param output Output (seq_len, n_state) - preallocated
     */
    void forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output);

    /**
     * Run attention block
     *
     * @param input Input (seq_len, n_state)
     * @param output Output (seq_len, n_state) - preallocated
     */
    void run_attention(const Eigen::MatrixXf& input, Eigen::MatrixXf& output);

    /**
     * Run FFN block
     *
     * @param input Input (seq_len, n_state)
     * @param output Output (seq_len, n_state) - preallocated
     */
    void run_ffn(const Eigen::MatrixXf& input, Eigen::MatrixXf& output);

private:
    size_t layer_idx_;
    size_t n_heads_;
    size_t n_state_;
    size_t ffn_dim_;
    size_t head_dim_;

    // Multi-head attention
    std::unique_ptr<MultiHeadAttention> attention_;

    // NPU matmul function (C++ std::function)
    NPUMatmulFunction npu_matmul_fn_;

    // NPU callback (C-style function pointer for Python)
    void* npu_callback_fn_;
    void* npu_user_data_;

    // BFP16 weights (no scales needed - embedded in block exponents)
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> q_weight_bfp16_;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> k_weight_bfp16_;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> v_weight_bfp16_;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> out_weight_bfp16_;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> fc1_weight_bfp16_;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> fc2_weight_bfp16_;

    // Biases (FP32)
    Eigen::VectorXf q_bias_;
    Eigen::VectorXf k_bias_;
    Eigen::VectorXf v_bias_;
    Eigen::VectorXf out_bias_;
    Eigen::VectorXf fc1_bias_;
    Eigen::VectorXf fc2_bias_;

    // Layer norm parameters (FP32)
    Eigen::VectorXf attn_ln_weight_;
    Eigen::VectorXf attn_ln_bias_;
    Eigen::VectorXf ffn_ln_weight_;
    Eigen::VectorXf ffn_ln_bias_;

    // Working buffers (reused across forward passes)
    Eigen::MatrixXf ln_output_;
    Eigen::MatrixXf Q_, K_, V_;
    Eigen::MatrixXf attn_output_;
    Eigen::MatrixXf fc1_output_;
    Eigen::MatrixXf fc2_output_;

    // BFP16 buffers (for NPU operations)
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> input_bfp16_shuffled_;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> output_bfp16_shuffled_;

    /**
     * Run NPU matmul with BFP16 quantization
     *
     * @param input Input matrix (FP32)
     * @param weight_bfp16 Weight matrix (BFP16, pre-shuffled)
     * @param bias Bias (FP32)
     * @param output Output matrix (FP32) - preallocated
     */
    void run_npu_linear(
        const Eigen::MatrixXf& input,
        const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& weight_bfp16,
        const Eigen::VectorXf& bias,
        Eigen::MatrixXf& output
    );
};

} // namespace whisper_xdna2
