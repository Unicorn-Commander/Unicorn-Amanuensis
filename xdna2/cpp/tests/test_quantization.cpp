#include "quantization.hpp"
#include <iostream>
#include <cmath>

using namespace whisper_xdna2;

bool test_quantize_dequantize() {
    std::cout << "Testing quantize/dequantize..." << std::endl;

    // Create test matrix
    Eigen::MatrixXf input(100, 200);
    input.setRandom();  // Random values in [-1, 1]
    input *= 10.0f;     // Scale to [-10, 10]

    // Quantize
    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> quantized;
    float scale;
    Quantizer::quantize_tensor(input, quantized, scale);

    std::cout << "  Input range: [" << input.minCoeff() << ", " << input.maxCoeff() << "]" << std::endl;
    std::cout << "  Scale: " << scale << std::endl;
    std::cout << "  Quantized range: [" << static_cast<int>(quantized.minCoeff())
              << ", " << static_cast<int>(quantized.maxCoeff()) << "]" << std::endl;

    // Dequantize
    Eigen::MatrixXf output;
    Quantizer::dequantize_tensor(quantized, output, scale);

    // Check error
    float max_error = (input - output).cwiseAbs().maxCoeff();
    float mean_error = (input - output).cwiseAbs().mean();

    std::cout << "  Mean error: " << mean_error << std::endl;
    std::cout << "  Max error: " << max_error << std::endl;

    // Error should be less than scale (quantization error)
    bool passed = mean_error < scale;
    std::cout << "  " << (passed ? "PASS" : "FAIL") << std::endl;

    return passed;
}

bool test_matmul_quantization() {
    std::cout << "\nTesting matmul quantization..." << std::endl;

    // Create test matrices
    Eigen::MatrixXf A(64, 128);
    Eigen::MatrixXf B(128, 256);
    A.setRandom();
    B.setRandom();

    // FP32 reference
    Eigen::MatrixXf C_ref = A * B;

    // Quantize inputs
    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> A_int8, B_int8;
    float scale_A, scale_B;
    Quantizer::quantize_tensor(A, A_int8, scale_A);
    Quantizer::quantize_tensor(B, B_int8, scale_B);

    // Simulate NPU matmul (on CPU)
    Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> C_int32(64, 256);
    for (int i = 0; i < A_int8.rows(); ++i) {
        for (int j = 0; j < B_int8.cols(); ++j) {
            int32_t sum = 0;
            for (int k = 0; k < A_int8.cols(); ++k) {
                sum += static_cast<int32_t>(A_int8(i, k)) * static_cast<int32_t>(B_int8(k, j));
            }
            C_int32(i, j) = sum;
        }
    }

    // Dequantize output
    Eigen::MatrixXf C_quant;
    Quantizer::dequantize_matmul_output(C_int32, C_quant, scale_A, scale_B);

    // Check error
    float mean_error = (C_ref - C_quant).cwiseAbs().mean();
    float max_error = (C_ref - C_quant).cwiseAbs().maxCoeff();
    float relative_error = mean_error / C_ref.cwiseAbs().mean();

    std::cout << "  Scale A: " << scale_A << ", Scale B: " << scale_B << std::endl;
    std::cout << "  Mean error: " << mean_error << std::endl;
    std::cout << "  Max error: " << max_error << std::endl;
    std::cout << "  Relative error: " << (relative_error * 100.0f) << "%" << std::endl;

    // Relative error should be < 2% for INT8 quantization
    bool passed = relative_error < 0.02f;
    std::cout << "  " << (passed ? "PASS" : "FAIL") << std::endl;

    return passed;
}

int main() {
    std::cout << "===== Quantization Tests =====" << std::endl;

    bool all_passed = true;
    all_passed &= test_quantize_dequantize();
    all_passed &= test_matmul_quantization();

    std::cout << "\n===== Results =====" << std::endl;
    std::cout << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;

    return all_passed ? 0 : 1;
}
