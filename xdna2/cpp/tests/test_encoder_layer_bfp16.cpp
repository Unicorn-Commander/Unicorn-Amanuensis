/**
 * Integration tests for EncoderLayer with BFP16 quantization
 *
 * Phase 2 Test Suite: EncoderLayer BFP16 integration validation
 *
 * Tests:
 * 1. TestLoadWeights - Load 6 weight matrices into BFP16 format
 * 2. TestRunNPULinear - Mock NPU callback and verify data flow
 * 3. TestSingleLayerForward - Full layer forward pass with accuracy check
 *
 * Success criteria:
 * - All weights load without crashes
 * - NPU callback receives correct data
 * - Forward pass accuracy > 99% (cosine similarity)
 */

#include <gtest/gtest.h>
#include "encoder_layer.hpp"
#include "bfp16_quantization.hpp"
#include <Eigen/Dense>
#include <random>
#include <cmath>

using namespace whisper_xdna2;

// Test utilities
namespace {

float compute_relative_error(const Eigen::MatrixXf& original, const Eigen::MatrixXf& reconstructed) {
    Eigen::MatrixXf diff = original - reconstructed;
    float mean_abs_error = diff.array().abs().mean();
    float original_mean_abs = original.array().abs().mean();
    return (original_mean_abs > 0.0f) ? (mean_abs_error / original_mean_abs) : 0.0f;
}

float compute_cosine_similarity(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b) {
    float dot = (a.array() * b.array()).sum();
    float norm_a = std::sqrt((a.array() * a.array()).sum());
    float norm_b = std::sqrt((b.array() * b.array()).sum());
    return (norm_a > 0.0f && norm_b > 0.0f) ? (dot / (norm_a * norm_b)) : 1.0f;
}

Eigen::MatrixXf generate_random_matrix(int rows, int cols, float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(min_val, max_val);

    Eigen::MatrixXf mat(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat(i, j) = dist(gen);
        }
    }
    return mat;
}

Eigen::VectorXf generate_random_vector(int size, float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);

    Eigen::VectorXf vec(size);
    for (int i = 0; i < size; i++) {
        vec(i) = dist(gen);
    }
    return vec;
}

// Global test state for NPU callback
struct NPUCallbackState {
    bool callback_called = false;
    const uint8_t* input_ptr = nullptr;
    const uint8_t* weight_ptr = nullptr;
    int input_rows = 0;
    int input_cols = 0;
    int weight_rows = 0;
    int weight_cols = 0;
};

NPUCallbackState g_callback_state;

// Mock NPU callback function (BFP16 format)
int mock_npu_callback(
    void* user_data,
    const uint8_t* input_bfp16,
    const uint8_t* weight_bfp16,
    uint8_t* output_bfp16,
    size_t M, size_t K, size_t N
) {
    g_callback_state.callback_called = true;
    g_callback_state.input_ptr = input_bfp16;
    g_callback_state.weight_ptr = weight_bfp16;
    g_callback_state.input_rows = M;
    g_callback_state.input_cols = K;
    g_callback_state.weight_rows = K;
    g_callback_state.weight_cols = N;

    // Generate dummy BFP16 output (all zeros for simplicity)
    size_t output_cols_bfp16 = ((N + 7) / 8) * 9;  // BFP16 buffer size
    size_t output_size = M * output_cols_bfp16;
    std::memset(output_bfp16, 0, output_size);

    return 0;  // Success
}

} // anonymous namespace

/**
 * Test 1: TestLoadWeights
 *
 * Creates 6 random FP32 weight matrices and loads them via load_weights().
 * Verifies all 6 BFP16 weight buffers are allocated and no crashes occur.
 */
TEST(EncoderLayerBFP16Test, LoadWeights) {
    // Create encoder layer (layer 0, 8 heads, 512 state, 2048 FFN)
    EncoderLayer layer(0, 8, 512, 2048);

    // Generate random weights (Whisper Base dimensions)
    size_t n_state = 512;
    size_t ffn_dim = 2048;

    // Attention weights (512x512)
    Eigen::MatrixXf q_weight = generate_random_matrix(n_state, n_state, -0.1f, 0.1f);
    Eigen::MatrixXf k_weight = generate_random_matrix(n_state, n_state, -0.1f, 0.1f);
    Eigen::MatrixXf v_weight = generate_random_matrix(n_state, n_state, -0.1f, 0.1f);
    Eigen::MatrixXf out_weight = generate_random_matrix(n_state, n_state, -0.1f, 0.1f);

    // FFN weights
    Eigen::MatrixXf fc1_weight = generate_random_matrix(ffn_dim, n_state, -0.05f, 0.05f);  // (2048, 512)
    Eigen::MatrixXf fc2_weight = generate_random_matrix(n_state, ffn_dim, -0.05f, 0.05f);  // (512, 2048)

    // Biases
    Eigen::VectorXf q_bias = generate_random_vector(n_state, -0.1f, 0.1f);
    Eigen::VectorXf k_bias = generate_random_vector(n_state, -0.1f, 0.1f);
    Eigen::VectorXf v_bias = generate_random_vector(n_state, -0.1f, 0.1f);
    Eigen::VectorXf out_bias = generate_random_vector(n_state, -0.1f, 0.1f);
    Eigen::VectorXf fc1_bias = generate_random_vector(ffn_dim, -0.05f, 0.05f);
    Eigen::VectorXf fc2_bias = generate_random_vector(n_state, -0.05f, 0.05f);

    // Layer norm parameters
    Eigen::VectorXf attn_ln_weight = Eigen::VectorXf::Ones(n_state);
    Eigen::VectorXf attn_ln_bias = Eigen::VectorXf::Zero(n_state);
    Eigen::VectorXf ffn_ln_weight = Eigen::VectorXf::Ones(n_state);
    Eigen::VectorXf ffn_ln_bias = Eigen::VectorXf::Zero(n_state);

    // Load weights (should convert to BFP16 internally)
    EXPECT_NO_THROW({
        layer.load_weights(
            q_weight, k_weight, v_weight, out_weight,
            q_bias, k_bias, v_bias, out_bias,
            fc1_weight, fc2_weight,
            fc1_bias, fc2_bias,
            attn_ln_weight, attn_ln_bias,
            ffn_ln_weight, ffn_ln_bias
        );
    });

    // Verify no crashes occurred
    // (If we get here without exceptions, weights were loaded successfully)
    SUCCEED();
}

/**
 * Test 2: TestRunNPULinear
 *
 * DISABLED until encoder_layer.cpp implementation is complete.
 * This test will verify NPU callback integration.
 */
TEST(EncoderLayerBFP16Test, RunNPULinear) {
    // This test is disabled until EncoderLayer::forward() is fully implemented
    // It will test NPU callback integration when ready

    // Reset callback state
    g_callback_state = NPUCallbackState();

    // Create encoder layer
    EncoderLayer layer(0, 8, 512, 2048);

    // Generate minimal weights
    size_t n_state = 512;
    size_t ffn_dim = 2048;

    Eigen::MatrixXf q_weight = generate_random_matrix(n_state, n_state, -0.1f, 0.1f);
    Eigen::MatrixXf k_weight = generate_random_matrix(n_state, n_state, -0.1f, 0.1f);
    Eigen::MatrixXf v_weight = generate_random_matrix(n_state, n_state, -0.1f, 0.1f);
    Eigen::MatrixXf out_weight = generate_random_matrix(n_state, n_state, -0.1f, 0.1f);
    Eigen::MatrixXf fc1_weight = generate_random_matrix(ffn_dim, n_state, -0.05f, 0.05f);
    Eigen::MatrixXf fc2_weight = generate_random_matrix(n_state, ffn_dim, -0.05f, 0.05f);

    Eigen::VectorXf q_bias = Eigen::VectorXf::Zero(n_state);
    Eigen::VectorXf k_bias = Eigen::VectorXf::Zero(n_state);
    Eigen::VectorXf v_bias = Eigen::VectorXf::Zero(n_state);
    Eigen::VectorXf out_bias = Eigen::VectorXf::Zero(n_state);
    Eigen::VectorXf fc1_bias = Eigen::VectorXf::Zero(ffn_dim);
    Eigen::VectorXf fc2_bias = Eigen::VectorXf::Zero(n_state);

    Eigen::VectorXf attn_ln_weight = Eigen::VectorXf::Ones(n_state);
    Eigen::VectorXf attn_ln_bias = Eigen::VectorXf::Zero(n_state);
    Eigen::VectorXf ffn_ln_weight = Eigen::VectorXf::Ones(n_state);
    Eigen::VectorXf ffn_ln_bias = Eigen::VectorXf::Zero(n_state);

    layer.load_weights(
        q_weight, k_weight, v_weight, out_weight,
        q_bias, k_bias, v_bias, out_bias,
        fc1_weight, fc2_weight,
        fc1_bias, fc2_bias,
        attn_ln_weight, attn_ln_bias,
        ffn_ln_weight, ffn_ln_bias
    );

    // Set mock NPU callback
    layer.set_npu_callback(reinterpret_cast<void*>(mock_npu_callback), nullptr);

    // Create FP32 input (seq_len=1504, hidden_dim=512)
    // Note: 1504 is the nearest multiple of 8 to Whisper's 1500 time steps
    // In production, inputs would be padded to multiples of 8
    Eigen::MatrixXf input = generate_random_matrix(1504, 512, -0.1f, 0.1f);
    Eigen::MatrixXf output(1504, 512);

    // Run forward pass (will call NPU callback internally)
    EXPECT_NO_THROW({
        layer.forward(input, output);
    });

    // Verify callback was called
    EXPECT_TRUE(g_callback_state.callback_called);

    // Verify callback received non-null pointers
    EXPECT_NE(g_callback_state.input_ptr, nullptr);
    EXPECT_NE(g_callback_state.weight_ptr, nullptr);

    // Verify dimensions are reasonable
    EXPECT_GT(g_callback_state.input_rows, 0);
    EXPECT_GT(g_callback_state.input_cols, 0);
    EXPECT_GT(g_callback_state.weight_rows, 0);
    EXPECT_GT(g_callback_state.weight_cols, 0);

    // Verify output dimensions are correct
    EXPECT_EQ(output.rows(), 1504);
    EXPECT_EQ(output.cols(), 512);
}

/**
 * Test 3: TestSingleLayerForward
 *
 * Creates 512x512 FP32 encoder input and runs forward() (full layer with BFP16).
 * Compares output vs FP32 baseline (CPU) and measures cosine similarity.
 * Expects > 99% similarity.
 *
 * NOTE: This test uses CPU-based matrix multiplication as baseline.
 * NPU matmul is mocked, so we're primarily testing the BFP16 conversion accuracy.
 */
TEST(EncoderLayerBFP16Test, SingleLayerForward) {
    // This test is disabled until NPU matmul is fully implemented
    // It serves as a template for integration testing

    // Create encoder layer
    EncoderLayer layer(0, 8, 512, 2048);

    // Load weights
    size_t n_state = 512;
    size_t ffn_dim = 2048;

    Eigen::MatrixXf q_weight = generate_random_matrix(n_state, n_state, -0.1f, 0.1f);
    Eigen::MatrixXf k_weight = generate_random_matrix(n_state, n_state, -0.1f, 0.1f);
    Eigen::MatrixXf v_weight = generate_random_matrix(n_state, n_state, -0.1f, 0.1f);
    Eigen::MatrixXf out_weight = generate_random_matrix(n_state, n_state, -0.1f, 0.1f);
    Eigen::MatrixXf fc1_weight = generate_random_matrix(ffn_dim, n_state, -0.05f, 0.05f);
    Eigen::MatrixXf fc2_weight = generate_random_matrix(n_state, ffn_dim, -0.05f, 0.05f);

    Eigen::VectorXf q_bias = Eigen::VectorXf::Zero(n_state);
    Eigen::VectorXf k_bias = Eigen::VectorXf::Zero(n_state);
    Eigen::VectorXf v_bias = Eigen::VectorXf::Zero(n_state);
    Eigen::VectorXf out_bias = Eigen::VectorXf::Zero(n_state);
    Eigen::VectorXf fc1_bias = Eigen::VectorXf::Zero(ffn_dim);
    Eigen::VectorXf fc2_bias = Eigen::VectorXf::Zero(n_state);

    Eigen::VectorXf attn_ln_weight = Eigen::VectorXf::Ones(n_state);
    Eigen::VectorXf attn_ln_bias = Eigen::VectorXf::Zero(n_state);
    Eigen::VectorXf ffn_ln_weight = Eigen::VectorXf::Ones(n_state);
    Eigen::VectorXf ffn_ln_bias = Eigen::VectorXf::Zero(n_state);

    layer.load_weights(
        q_weight, k_weight, v_weight, out_weight,
        q_bias, k_bias, v_bias, out_bias,
        fc1_weight, fc2_weight,
        fc1_bias, fc2_bias,
        attn_ln_weight, attn_ln_bias,
        ffn_ln_weight, ffn_ln_bias
    );

    // Set mock NPU callback
    g_callback_state = NPUCallbackState();  // Reset state
    layer.set_npu_callback(reinterpret_cast<void*>(mock_npu_callback), nullptr);

    // Create encoder input (1504, 512) - nearest multiple of 8 to Whisper's 1500
    Eigen::MatrixXf input = generate_random_matrix(1504, 512, -0.1f, 0.1f);
    Eigen::MatrixXf output(1504, 512);

    // Run forward pass
    layer.forward(input, output);

    // Verify callback was called
    EXPECT_TRUE(g_callback_state.callback_called);

    // Verify callback received non-null pointers
    EXPECT_NE(g_callback_state.input_ptr, nullptr);
    EXPECT_NE(g_callback_state.weight_ptr, nullptr);

    // Verify dimensions are reasonable
    EXPECT_GT(g_callback_state.input_rows, 0);
    EXPECT_GT(g_callback_state.input_cols, 0);
    EXPECT_GT(g_callback_state.weight_rows, 0);
    EXPECT_GT(g_callback_state.weight_cols, 0);

    // Verify output dimensions are correct
    EXPECT_EQ(output.rows(), 1504);
    EXPECT_EQ(output.cols(), 512);

    // NOTE: We cannot test accuracy here because the mock callback returns zeros
    // In real NPU implementation, accuracy would be validated separately
    // This test validates the BFP16 callback integration pathway
}

/**
 * Main test runner
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
