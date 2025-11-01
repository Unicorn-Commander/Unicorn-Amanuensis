#include "whisper_xdna2_runtime.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <chrono>

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " [options]\n"
              << "\n"
              << "Options:\n"
              << "  --model <size>    Model size (default: base)\n"
              << "  --4tile           Use 4-tile kernels (default: 32-tile)\n"
              << "  --test-matmul     Run matmul test\n"
              << "  --test-encoder    Run encoder test\n"
              << "  --help            Show this help message\n"
              << "\n";
}

bool test_matmul(whisper_xdna2::WhisperXDNA2Runtime& runtime) {
    std::cout << "\n=== Testing Matrix Multiplication ===\n";

    const size_t M = 512;
    const size_t K = 512;
    const size_t N = 512;

    std::cout << "Matrix dimensions: " << M << "x" << K << " @ " << K << "x" << N << "\n";

    // Allocate and initialize test data
    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(K * N);
    std::vector<int32_t> C(M * N);

    // Fill with simple test pattern
    for (size_t i = 0; i < A.size(); i++) {
        A[i] = (i % 10) - 5;  // Range: -5 to 4
    }
    for (size_t i = 0; i < B.size(); i++) {
        B[i] = (i % 8) - 4;   // Range: -4 to 3
    }

    // Run matmul on NPU
    std::cout << "Running matmul on NPU...\n";

    auto start = std::chrono::high_resolution_clock::now();

    try {
        runtime.run_matmul(A.data(), B.data(), C.data(), M, K, N);
    } catch (const std::exception& e) {
        std::cerr << "Matmul failed: " << e.what() << "\n";
        return false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Calculate performance
    double ops = 2.0 * M * K * N;  // Multiply-add operations
    double gflops = ops / (elapsed_ms / 1000.0) / 1e9;

    std::cout << "Matmul completed in " << elapsed_ms << " ms\n";
    std::cout << "Performance: " << gflops << " GFLOPS\n";

    // Verify first few elements are non-zero
    bool has_nonzero = false;
    for (size_t i = 0; i < std::min(size_t(100), C.size()); i++) {
        if (C[i] != 0) {
            has_nonzero = true;
            break;
        }
    }

    if (!has_nonzero) {
        std::cerr << "Warning: Output appears to be all zeros\n";
        return false;
    }

    std::cout << "Sample outputs (first 10):\n";
    for (size_t i = 0; i < std::min(size_t(10), C.size()); i++) {
        std::cout << "  C[" << i << "] = " << C[i] << "\n";
    }

    std::cout << "\n✓ Matmul test PASSED\n";
    return true;
}

bool test_encoder(whisper_xdna2::WhisperXDNA2Runtime& runtime) {
    std::cout << "\n=== Testing Encoder ===\n";

    auto dims = runtime.get_model_dims();
    std::cout << "Model dimensions:\n"
              << "  n_mels: " << dims.n_mels << "\n"
              << "  n_state: " << dims.n_state << "\n"
              << "  n_head: " << dims.n_head << "\n"
              << "  n_layer: " << dims.n_layer << "\n";

    const size_t seq_len = 100;  // Short sequence for testing
    const size_t n_state = dims.n_state;

    std::cout << "\nTest sequence length: " << seq_len << "\n";

    // Allocate input and output
    std::vector<float> input(seq_len * n_state);
    std::vector<float> output(seq_len * n_state);

    // Initialize with random-ish data
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = (float)(i % 100) / 50.0f - 1.0f;  // Range: -1 to 1
    }

    std::cout << "Running encoder...\n";

    auto start = std::chrono::high_resolution_clock::now();

    try {
        runtime.run_encoder(input.data(), output.data(), seq_len);
    } catch (const std::exception& e) {
        std::cerr << "Encoder failed: " << e.what() << "\n";
        return false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Encoder completed in " << elapsed_ms << " ms\n";

    // Get performance stats
    auto stats = runtime.get_perf_stats();
    std::cout << "\nPerformance statistics:\n"
              << "  Total inference: " << stats.total_inference_ms << " ms\n"
              << "  Matmul time: " << stats.matmul_ms << " ms\n"
              << "  CPU ops time: " << stats.cpu_ops_ms << " ms\n"
              << "  Num matmuls: " << stats.num_matmuls << "\n"
              << "  Avg GFLOPS: " << stats.avg_gflops << "\n";

    std::cout << "\nSample outputs (first 10):\n";
    for (size_t i = 0; i < std::min(size_t(10), output.size()); i++) {
        std::cout << "  output[" << i << "] = " << output[i] << "\n";
    }

    std::cout << "\n✓ Encoder test PASSED\n";
    return true;
}

int main(int argc, char* argv[]) {
    std::string model_size = "base";
    bool use_4tile = false;
    bool test_matmul_flag = false;
    bool test_encoder_flag = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_size = argv[++i];
        } else if (arg == "--4tile") {
            use_4tile = true;
        } else if (arg == "--test-matmul") {
            test_matmul_flag = true;
        } else if (arg == "--test-encoder") {
            test_encoder_flag = true;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // If no test specified, run both
    if (!test_matmul_flag && !test_encoder_flag) {
        test_matmul_flag = true;
        test_encoder_flag = true;
    }

    std::cout << "========================================\n";
    std::cout << "Whisper XDNA2 C++ Runtime Test\n";
    std::cout << "========================================\n";
    std::cout << "Model size: " << model_size << "\n";
    std::cout << "Tile mode: " << (use_4tile ? "4-tile" : "32-tile") << "\n";
    std::cout << "========================================\n\n";

    try {
        // Create runtime
        std::cout << "Creating runtime...\n";
        whisper_xdna2::WhisperXDNA2Runtime runtime(model_size, use_4tile);

        // Initialize
        std::cout << "Initializing NPU...\n";
        runtime.initialize();

        if (!runtime.is_initialized()) {
            std::cerr << "Failed to initialize runtime\n";
            return 1;
        }

        std::cout << "✓ Runtime initialized successfully\n";

        bool all_passed = true;

        // Run tests
        if (test_matmul_flag) {
            if (!test_matmul(runtime)) {
                all_passed = false;
            }
        }

        if (test_encoder_flag) {
            if (!test_encoder(runtime)) {
                all_passed = false;
            }
        }

        // Summary
        std::cout << "\n========================================\n";
        if (all_passed) {
            std::cout << "✓ All tests PASSED\n";
        } else {
            std::cout << "✗ Some tests FAILED\n";
        }
        std::cout << "========================================\n";

        return all_passed ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << "\n";
        return 1;
    }
}
