/**
 * C API for Whisper Encoder - Python ctypes integration
 *
 * Simple C-style API to expose C++ encoder to Python
 */

#ifndef WHISPER_ENCODER_C_API_H
#define WHISPER_ENCODER_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

// Opaque handle to encoder layer
typedef void* EncoderLayerHandle;

/**
 * Create a new encoder layer
 *
 * @param layer_idx Layer index (0-5)
 * @param n_heads Number of attention heads (8 for Whisper Base)
 * @param n_state Hidden dimension (512 for Whisper Base)
 * @param ffn_dim FFN dimension (2048 for Whisper Base)
 * @return Encoder layer handle, NULL on failure
 */
EncoderLayerHandle encoder_layer_create(
    size_t layer_idx,
    size_t n_heads,
    size_t n_state,
    size_t ffn_dim
);

/**
 * Destroy encoder layer and free memory
 *
 * @param handle Encoder layer handle
 */
void encoder_layer_destroy(EncoderLayerHandle handle);

/**
 * Load weights into encoder layer
 *
 * All weights are FP32 arrays. They will be quantized to INT8 internally.
 *
 * @param handle Encoder layer handle
 * @param q_weight Query weight (n_state * n_state)
 * @param k_weight Key weight (n_state * n_state)
 * @param v_weight Value weight (n_state * n_state)
 * @param out_weight Output weight (n_state * n_state)
 * @param q_bias Query bias (n_state)
 * @param k_bias Key bias (n_state)
 * @param v_bias Value bias (n_state)
 * @param out_bias Output bias (n_state)
 * @param fc1_weight FC1 weight (ffn_dim * n_state)
 * @param fc2_weight FC2 weight (n_state * ffn_dim)
 * @param fc1_bias FC1 bias (ffn_dim)
 * @param fc2_bias FC2 bias (n_state)
 * @param attn_ln_weight Attention LayerNorm weight (n_state)
 * @param attn_ln_bias Attention LayerNorm bias (n_state)
 * @param ffn_ln_weight FFN LayerNorm weight (n_state)
 * @param ffn_ln_bias FFN LayerNorm bias (n_state)
 * @return 0 on success, -1 on failure
 */
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
);

/**
 * Run encoder layer forward pass
 *
 * @param handle Encoder layer handle
 * @param input Input array (seq_len * n_state), FP32
 * @param output Output array (seq_len * n_state), FP32 (preallocated)
 * @param seq_len Sequence length
 * @param n_state Hidden dimension
 * @return 0 on success, -1 on failure
 */
int encoder_layer_forward(
    EncoderLayerHandle handle,
    const float* input,
    float* output,
    size_t seq_len,
    size_t n_state
);

/**
 * Get library version string
 *
 * @return Version string (e.g., "1.0.0")
 */
const char* encoder_get_version(void);

/**
 * Check if library is compiled with correct configuration
 *
 * @return 1 if OK, 0 if configuration mismatch
 */
int encoder_check_config(void);

#ifdef __cplusplus
}
#endif

#endif // WHISPER_ENCODER_C_API_H
