#include "encoder_c_api.h"
#include "encoder_layer.hpp"
#include "npu_callback.h"
#include <cstring>
#include <exception>
#include <iostream>

using namespace whisper_xdna2;

// Version
static const char* VERSION = "1.0.0";

extern "C" {

EncoderLayerHandle encoder_layer_create(
    size_t layer_idx,
    size_t n_heads,
    size_t n_state,
    size_t ffn_dim
) {
    try {
        auto* layer = new EncoderLayer(layer_idx, n_heads, n_state, ffn_dim);
        return static_cast<void*>(layer);
    } catch (const std::exception& e) {
        std::cerr << "Error creating encoder layer: " << e.what() << std::endl;
        return nullptr;
    }
}

void encoder_layer_destroy(EncoderLayerHandle handle) {
    if (handle) {
        auto* layer = static_cast<EncoderLayer*>(handle);
        delete layer;
    }
}

int encoder_layer_load_weights(
    EncoderLayerHandle handle,
    const float* q_weight,
    const float* k_weight,
    const float* v_weight,
    const float* out_weight,
    const float* q_bias,
    const float* k_bias,
    const float* v_bias,
    const float* out_bias,
    const float* fc1_weight,
    const float* fc2_weight,
    const float* fc1_bias,
    const float* fc2_bias,
    const float* attn_ln_weight,
    const float* attn_ln_bias,
    const float* ffn_ln_weight,
    const float* ffn_ln_bias,
    size_t n_state,
    size_t ffn_dim
) {
    if (!handle) return -1;

    try {
        auto* layer = static_cast<EncoderLayer*>(handle);

        // Convert C arrays to Eigen matrices
        // Weights are row-major from Python, Eigen defaults to column-major
        Eigen::MatrixXf q_w = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            q_weight, n_state, n_state
        );
        Eigen::MatrixXf k_w = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            k_weight, n_state, n_state
        );
        Eigen::MatrixXf v_w = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            v_weight, n_state, n_state
        );
        Eigen::MatrixXf out_w = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            out_weight, n_state, n_state
        );
        Eigen::MatrixXf fc1_w = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            fc1_weight, ffn_dim, n_state
        );
        Eigen::MatrixXf fc2_w = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            fc2_weight, n_state, ffn_dim
        );

        // Biases are 1D vectors
        Eigen::VectorXf q_b = Eigen::Map<const Eigen::VectorXf>(q_bias, n_state);
        Eigen::VectorXf k_b = Eigen::Map<const Eigen::VectorXf>(k_bias, n_state);
        Eigen::VectorXf v_b = Eigen::Map<const Eigen::VectorXf>(v_bias, n_state);
        Eigen::VectorXf out_b = Eigen::Map<const Eigen::VectorXf>(out_bias, n_state);
        Eigen::VectorXf fc1_b = Eigen::Map<const Eigen::VectorXf>(fc1_bias, ffn_dim);
        Eigen::VectorXf fc2_b = Eigen::Map<const Eigen::VectorXf>(fc2_bias, n_state);
        Eigen::VectorXf attn_ln_w = Eigen::Map<const Eigen::VectorXf>(attn_ln_weight, n_state);
        Eigen::VectorXf attn_ln_b = Eigen::Map<const Eigen::VectorXf>(attn_ln_bias, n_state);
        Eigen::VectorXf ffn_ln_w = Eigen::Map<const Eigen::VectorXf>(ffn_ln_weight, n_state);
        Eigen::VectorXf ffn_ln_b = Eigen::Map<const Eigen::VectorXf>(ffn_ln_bias, n_state);

        // Load weights into layer
        layer->load_weights(
            q_w, k_w, v_w, out_w,
            q_b, k_b, v_b, out_b,
            fc1_w, fc2_w,
            fc1_b, fc2_b,
            attn_ln_w, attn_ln_b,
            ffn_ln_w, ffn_ln_b
        );

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
        return -1;
    }
}

int encoder_layer_forward(
    EncoderLayerHandle handle,
    const float* input,
    float* output,
    size_t seq_len,
    size_t n_state
) {
    if (!handle || !input || !output) return -1;

    try {
        auto* layer = static_cast<EncoderLayer*>(handle);

        // Map input as row-major matrix (seq_len, n_state)
        Eigen::MatrixXf input_mat = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            input, seq_len, n_state
        );

        // Allocate output matrix
        Eigen::MatrixXf output_mat(seq_len, n_state);

        // Run forward pass
        layer->forward(input_mat, output_mat);

        // Copy output back (row-major)
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            output, seq_len, n_state
        ) = output_mat;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error in forward pass: " << e.what() << std::endl;
        return -1;
    }
}

int encoder_layer_set_npu_callback(
    EncoderLayerHandle handle,
    NPUMatmulCallback callback,
    void* user_data
) {
    if (!handle) return -1;

    try {
        auto* layer = static_cast<EncoderLayer*>(handle);
        layer->set_npu_callback(reinterpret_cast<void*>(callback), user_data);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error setting NPU callback: " << e.what() << std::endl;
        return -1;
    }
}

const char* encoder_get_version(void) {
    return VERSION;
}

int encoder_check_config(void) {
    // Basic sanity checks
    if (sizeof(float) != 4) return 0;
    if (sizeof(int8_t) != 1) return 0;
    if (sizeof(int32_t) != 4) return 0;
    return 1;
}

} // extern "C"
