#include "encoder_layer.hpp"
#include <iostream>
#include <random>

using namespace whisper_xdna2;

// Mock NPU matmul function (uses CPU INT8 matmul for testing)
void mock_npu_matmul(
    const Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>& B,
    Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>& C
) {
    // C = A @ B (INT8 @ INT8 -> INT32)
    const int M = A.rows();
    const int K = A.cols();
    const int N = B.cols();

    // Resize output
    C.resize(M, N);

    // Compute matmul
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<int32_t>(A(i, k)) * static_cast<int32_t>(B(k, j));
            }
            C(i, j) = sum;
        }
    }
}

bool test_encoder_layer() {
    std::cout << "Testing encoder layer..." << std::endl;

    // Whisper Base dimensions
    const size_t n_heads = 8;
    const size_t n_state = 512;
    const size_t ffn_dim = 2048;
    const size_t seq_len = 100;

    // Create encoder layer
    EncoderLayer layer(0, n_heads, n_state, ffn_dim);

    // Set mock NPU matmul
    layer.set_npu_matmul(mock_npu_matmul);

    // Generate random weights
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    auto random_matrix = [&](size_t rows, size_t cols) {
        Eigen::MatrixXf m(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                m(i, j) = dist(gen);
            }
        }
        return m;
    };

    auto random_vector = [&](size_t size) {
        Eigen::VectorXf v(size);
        for (size_t i = 0; i < size; ++i) {
            v(i) = dist(gen);
        }
        return v;
    };

    // Load weights
    layer.load_weights(
        random_matrix(n_state, n_state),  // q_weight
        random_matrix(n_state, n_state),  // k_weight
        random_matrix(n_state, n_state),  // v_weight
        random_matrix(n_state, n_state),  // out_weight
        random_vector(n_state),           // q_bias
        random_vector(n_state),           // k_bias
        random_vector(n_state),           // v_bias
        random_vector(n_state),           // out_bias
        random_matrix(ffn_dim, n_state),  // fc1_weight
        random_matrix(n_state, ffn_dim),  // fc2_weight
        random_vector(ffn_dim),           // fc1_bias
        random_vector(n_state),           // fc2_bias
        Eigen::VectorXf::Ones(n_state),   // attn_ln_weight
        Eigen::VectorXf::Zero(n_state),   // attn_ln_bias
        Eigen::VectorXf::Ones(n_state),   // ffn_ln_weight
        Eigen::VectorXf::Zero(n_state)    // ffn_ln_bias
    );

    std::cout << "  Weights loaded" << std::endl;

    // Create input
    Eigen::MatrixXf input = random_matrix(seq_len, n_state);
    Eigen::MatrixXf output;

    std::cout << "  Running forward pass..." << std::endl;

    // Run forward pass
    layer.forward(input, output);

    std::cout << "  Input shape: (" << input.rows() << ", " << input.cols() << ")" << std::endl;
    std::cout << "  Output shape: (" << output.rows() << ", " << output.cols() << ")" << std::endl;
    std::cout << "  Output mean: " << output.mean() << std::endl;
    std::cout << "  Output std: " << std::sqrt((output.array() - output.mean()).square().mean()) << std::endl;

    // Check output shape
    bool shape_ok = (output.rows() == seq_len && output.cols() == n_state);

    // Check output is not NaN or Inf
    bool values_ok = output.allFinite();

    // Check output is not all zeros
    bool not_zeros = output.cwiseAbs().sum() > 0.0f;

    bool passed = shape_ok && values_ok && not_zeros;
    std::cout << "  " << (passed ? "PASS" : "FAIL") << std::endl;

    if (!shape_ok) std::cout << "    ERROR: Wrong output shape" << std::endl;
    if (!values_ok) std::cout << "    ERROR: NaN or Inf in output" << std::endl;
    if (!not_zeros) std::cout << "    ERROR: Output is all zeros" << std::endl;

    return passed;
}

bool test_attention_only() {
    std::cout << "\nTesting attention only..." << std::endl;

    const size_t n_heads = 8;
    const size_t n_state = 512;
    const size_t ffn_dim = 2048;
    const size_t seq_len = 50;

    EncoderLayer layer(0, n_heads, n_state, ffn_dim);
    layer.set_npu_matmul(mock_npu_matmul);

    // Simple weights
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    auto random_matrix = [&](size_t rows, size_t cols) {
        Eigen::MatrixXf m(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                m(i, j) = dist(gen);
            }
        }
        return m;
    };

    auto random_vector = [&](size_t size) {
        Eigen::VectorXf v(size);
        for (size_t i = 0; i < size; ++i) {
            v(i) = dist(gen);
        }
        return v;
    };

    layer.load_weights(
        random_matrix(n_state, n_state),
        random_matrix(n_state, n_state),
        random_matrix(n_state, n_state),
        random_matrix(n_state, n_state),
        random_vector(n_state),
        random_vector(n_state),
        random_vector(n_state),
        random_vector(n_state),
        random_matrix(ffn_dim, n_state),
        random_matrix(n_state, ffn_dim),
        random_vector(ffn_dim),
        random_vector(n_state),
        Eigen::VectorXf::Ones(n_state),
        Eigen::VectorXf::Zero(n_state),
        Eigen::VectorXf::Ones(n_state),
        Eigen::VectorXf::Zero(n_state)
    );

    Eigen::MatrixXf input = random_matrix(seq_len, n_state);
    Eigen::MatrixXf output;

    layer.run_attention(input, output);

    std::cout << "  Attention output shape: (" << output.rows() << ", " << output.cols() << ")" << std::endl;
    std::cout << "  Attention output mean: " << output.mean() << std::endl;

    bool passed = output.rows() == seq_len && output.cols() == n_state && output.allFinite();
    std::cout << "  " << (passed ? "PASS" : "FAIL") << std::endl;

    return passed;
}

bool test_ffn_only() {
    std::cout << "\nTesting FFN only..." << std::endl;

    const size_t n_heads = 8;
    const size_t n_state = 512;
    const size_t ffn_dim = 2048;
    const size_t seq_len = 50;

    EncoderLayer layer(0, n_heads, n_state, ffn_dim);
    layer.set_npu_matmul(mock_npu_matmul);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    auto random_matrix = [&](size_t rows, size_t cols) {
        Eigen::MatrixXf m(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                m(i, j) = dist(gen);
            }
        }
        return m;
    };

    auto random_vector = [&](size_t size) {
        Eigen::VectorXf v(size);
        for (size_t i = 0; i < size; ++i) {
            v(i) = dist(gen);
        }
        return v;
    };

    layer.load_weights(
        random_matrix(n_state, n_state),
        random_matrix(n_state, n_state),
        random_matrix(n_state, n_state),
        random_matrix(n_state, n_state),
        random_vector(n_state),
        random_vector(n_state),
        random_vector(n_state),
        random_vector(n_state),
        random_matrix(ffn_dim, n_state),
        random_matrix(n_state, ffn_dim),
        random_vector(ffn_dim),
        random_vector(n_state),
        Eigen::VectorXf::Ones(n_state),
        Eigen::VectorXf::Zero(n_state),
        Eigen::VectorXf::Ones(n_state),
        Eigen::VectorXf::Zero(n_state)
    );

    Eigen::MatrixXf input = random_matrix(seq_len, n_state);
    Eigen::MatrixXf output;

    layer.run_ffn(input, output);

    std::cout << "  FFN output shape: (" << output.rows() << ", " << output.cols() << ")" << std::endl;
    std::cout << "  FFN output mean: " << output.mean() << std::endl;

    bool passed = output.rows() == seq_len && output.cols() == n_state && output.allFinite();
    std::cout << "  " << (passed ? "PASS" : "FAIL") << std::endl;

    return passed;
}

int main() {
    std::cout << "===== Encoder Layer Tests =====" << std::endl;

    bool all_passed = true;
    all_passed &= test_attention_only();
    all_passed &= test_ffn_only();
    all_passed &= test_encoder_layer();

    std::cout << "\n===== Results =====" << std::endl;
    std::cout << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;

    return all_passed ? 0 : 1;
}
