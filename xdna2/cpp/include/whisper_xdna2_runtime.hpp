#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

// Forward declaration - PyObject is defined in Python.h
// Include Python.h in .cpp files before this header
struct _object;
typedef struct _object PyObject;

namespace whisper_xdna2 {

/**
 * WhisperXDNA2Runtime - Core C++ runtime for Whisper encoder on XDNA2 NPU
 *
 * This runtime provides high-performance C++ inference for Whisper encoder
 * with NPU acceleration via XRT Python bindings. Target: 17-28x realtime.
 *
 * Architecture:
 * - Direct kernel invocation via XRT Python API
 * - Zero-copy buffer management where possible
 * - RAII resource management
 * - Thread-safe operations
 *
 * Performance Goals:
 * - 3-5x speedup over Python runtime (5.59x realtime -> 17-28x realtime)
 * - Eliminate 50-60% Python overhead
 * - Maintain NPU utilization >90%
 */
class WhisperXDNA2Runtime {
public:
    /**
     * Construct runtime with model configuration
     *
     * @param model_size Whisper model size ("base", "small", "medium", "large")
     * @param use_4tile Use 4-tile kernels (true) or 32-tile kernels (false)
     */
    WhisperXDNA2Runtime(const std::string& model_size = "base", bool use_4tile = false);

    /**
     * Destructor - cleans up all NPU resources
     */
    ~WhisperXDNA2Runtime();

    // Disable copy (use move semantics only)
    WhisperXDNA2Runtime(const WhisperXDNA2Runtime&) = delete;
    WhisperXDNA2Runtime& operator=(const WhisperXDNA2Runtime&) = delete;

    // Move semantics
    WhisperXDNA2Runtime(WhisperXDNA2Runtime&&) noexcept;
    WhisperXDNA2Runtime& operator=(WhisperXDNA2Runtime&&) noexcept;

    /**
     * Initialize NPU device and load kernels
     *
     * @throws std::runtime_error if initialization fails
     */
    void initialize();

    /**
     * Check if runtime is initialized
     */
    bool is_initialized() const { return initialized_; }

    /**
     * Load encoder weights from file or memory
     *
     * @param weights_path Path to quantized weights file
     * @throws std::runtime_error if loading fails
     */
    void load_encoder_weights(const std::string& weights_path);

    /**
     * Run encoder inference on NPU
     *
     * @param input Input features (FP32) - flattened array
     * @param output Output features (FP32) - flattened array (must be pre-allocated)
     * @param seq_len Sequence length (number of time steps)
     * @throws std::runtime_error if inference fails
     */
    void run_encoder(const float* input, float* output, size_t seq_len);

    /**
     * Run single matrix multiplication on NPU
     *
     * This is the core primitive operation. All encoder layers use this.
     *
     * @param A Input matrix A (int8) - MxK in row-major
     * @param B Input matrix B (int8) - KxN in row-major
     * @param C Output matrix C (int32) - MxN in row-major (must be pre-allocated)
     * @param M Number of rows in A and C
     * @param K Number of columns in A, rows in B
     * @param N Number of columns in B and C
     * @throws std::runtime_error if matmul fails
     */
    void run_matmul(const int8_t* A, const int8_t* B, int32_t* C,
                    size_t M, size_t K, size_t N);

    /**
     * Get model dimensions for the current model size
     */
    struct ModelDims {
        size_t n_mels;      // Mel bins (80 for all Whisper models)
        size_t n_ctx;       // Context length (1500)
        size_t n_state;     // Hidden dimension (512 for base)
        size_t n_head;      // Attention heads (8 for base)
        size_t n_layer;     // Encoder layers (6 for base)
    };

    ModelDims get_model_dims() const;

    /**
     * Get performance statistics
     */
    struct PerfStats {
        double total_inference_ms;
        double matmul_ms;
        double cpu_ops_ms;
        size_t num_matmuls;
        double avg_gflops;
    };

    PerfStats get_perf_stats() const { return perf_stats_; }
    void reset_perf_stats();

private:
    // Python interpreter and module handles
    PyObject* pyxrt_module_;
    PyObject* device_obj_;

    // Kernel instances (dictionary of kernel name -> kernel object)
    std::unordered_map<std::string, PyObject*> kernel_apps_;

    // Configuration
    std::string model_size_;
    bool use_4tile_;
    bool initialized_;

    // Model dimensions
    ModelDims model_dims_;

    // Performance tracking
    mutable PerfStats perf_stats_;

    // Helper methods
    void init_python();
    void cleanup_python();
    void load_kernels();
    void load_kernel(const std::string& name, size_t M, size_t K, size_t N,
                     const std::string& xclbin_path, const std::string& insts_path);

    // Select best kernel for given dimensions
    std::string select_kernel(size_t M, size_t K, size_t N) const;
};

} // namespace whisper_xdna2
