#include <Python.h>  // Must come first before other headers
#include "whisper_xdna2_runtime.hpp"
#include "buffer_manager.hpp"
#include "kernel_loader.hpp"
#include <iostream>
#include <stdexcept>

namespace whisper_xdna2 {

WhisperXDNA2Runtime::WhisperXDNA2Runtime(const std::string& model_size, bool use_4tile)
    : pyxrt_module_(nullptr)
    , device_obj_(nullptr)
    , model_size_(model_size)
    , use_4tile_(use_4tile)
    , initialized_(false)
{
    // Initialize model dimensions based on model size
    if (model_size_ == "base") {
        model_dims_.n_mels = 80;
        model_dims_.n_ctx = 1500;
        model_dims_.n_state = 512;
        model_dims_.n_head = 8;
        model_dims_.n_layer = 6;
    } else {
        throw std::runtime_error("Only 'base' model size supported");
    }

    // Reset performance stats
    reset_perf_stats();
}

WhisperXDNA2Runtime::~WhisperXDNA2Runtime() {
    cleanup_python();
}

WhisperXDNA2Runtime::WhisperXDNA2Runtime(WhisperXDNA2Runtime&& other) noexcept
    : pyxrt_module_(other.pyxrt_module_)
    , device_obj_(other.device_obj_)
    , kernel_apps_(std::move(other.kernel_apps_))
    , model_size_(std::move(other.model_size_))
    , use_4tile_(other.use_4tile_)
    , initialized_(other.initialized_)
    , model_dims_(other.model_dims_)
    , perf_stats_(other.perf_stats_)
{
    other.pyxrt_module_ = nullptr;
    other.device_obj_ = nullptr;
    other.initialized_ = false;
}

WhisperXDNA2Runtime& WhisperXDNA2Runtime::operator=(WhisperXDNA2Runtime&& other) noexcept {
    if (this != &other) {
        cleanup_python();

        pyxrt_module_ = other.pyxrt_module_;
        device_obj_ = other.device_obj_;
        kernel_apps_ = std::move(other.kernel_apps_);
        model_size_ = std::move(other.model_size_);
        use_4tile_ = other.use_4tile_;
        initialized_ = other.initialized_;
        model_dims_ = other.model_dims_;
        perf_stats_ = other.perf_stats_;

        other.pyxrt_module_ = nullptr;
        other.device_obj_ = nullptr;
        other.initialized_ = false;
    }
    return *this;
}

void WhisperXDNA2Runtime::init_python() {
    // Initialize Python interpreter if not already initialized
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    // Import pyxrt module
    pyxrt_module_ = PyImport_ImportModule("pyxrt");
    if (!pyxrt_module_) {
        PyErr_Print();
        throw std::runtime_error("Failed to import pyxrt module");
    }

    // Create device object (device 0)
    PyObject* device_class = PyObject_GetAttrString(pyxrt_module_, "device");
    if (!device_class) {
        PyErr_Print();
        throw std::runtime_error("Failed to get xrt.device class");
    }

    device_obj_ = PyObject_CallFunction(device_class, "i", 0);
    Py_DECREF(device_class);
    if (!device_obj_) {
        PyErr_Print();
        throw std::runtime_error("Failed to create xrt.device(0)");
    }
}

void WhisperXDNA2Runtime::cleanup_python() {
    // Clean up kernel applications
    for (auto& kv : kernel_apps_) {
        if (kv.second) {
            Py_DECREF(kv.second);
        }
    }
    kernel_apps_.clear();

    // Clean up device
    if (device_obj_) {
        Py_DECREF(device_obj_);
        device_obj_ = nullptr;
    }

    // Clean up module
    if (pyxrt_module_) {
        Py_DECREF(pyxrt_module_);
        pyxrt_module_ = nullptr;
    }
}

void WhisperXDNA2Runtime::initialize() {
    if (initialized_) {
        return;
    }

    std::cout << "Initializing Whisper XDNA2 Runtime (model=" << model_size_
              << ", 4tile=" << use_4tile_ << ")" << std::endl;

    init_python();
    // Kernels will be loaded on-demand
    initialized_ = true;

    std::cout << "Runtime initialized" << std::endl;
}

void WhisperXDNA2Runtime::load_encoder_weights(const std::string& weights_path) {
    if (!initialized_) {
        throw std::runtime_error("Runtime not initialized");
    }

    std::cout << "Loading encoder weights from: " << weights_path << std::endl;
    // TODO: Implement weight loading
    std::cout << "Weights loaded (stub)" << std::endl;
}

void WhisperXDNA2Runtime::run_encoder(const float* input, float* output, size_t seq_len) {
    if (!initialized_) {
        throw std::runtime_error("Runtime not initialized");
    }

    // TODO: Implement encoder forward pass using NPU kernels
    std::cout << "Running encoder (seq_len=" << seq_len << ")" << std::endl;

    // For now, just copy input to output as a stub
    size_t total_elements = seq_len * model_dims_.n_state;
    for (size_t i = 0; i < total_elements; i++) {
        output[i] = input[i];
    }
}

void WhisperXDNA2Runtime::run_matmul(const int8_t* A, const int8_t* B, int32_t* C,
                                     size_t M, size_t K, size_t N) {
    if (!initialized_) {
        throw std::runtime_error("Runtime not initialized");
    }

    // TODO: Select appropriate kernel and execute
    std::string kernel_name = select_kernel(M, K, N);
    std::cout << "Running matmul " << M << "x" << K << "x" << N
              << " using kernel: " << kernel_name << std::endl;

    // Stub: CPU matmul for now
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            int32_t sum = 0;
            for (size_t k = 0; k < K; k++) {
                sum += static_cast<int32_t>(A[i*K + k]) * static_cast<int32_t>(B[k*N + j]);
            }
            C[i*N + j] = sum;
        }
    }
}

WhisperXDNA2Runtime::ModelDims WhisperXDNA2Runtime::get_model_dims() const {
    return model_dims_;
}

void WhisperXDNA2Runtime::reset_perf_stats() {
    perf_stats_.total_inference_ms = 0.0;
    perf_stats_.matmul_ms = 0.0;
    perf_stats_.cpu_ops_ms = 0.0;
    perf_stats_.num_matmuls = 0;
    perf_stats_.avg_gflops = 0.0;
}

void WhisperXDNA2Runtime::load_kernels() {
    // TODO: Load standard kernels for the model
}

void WhisperXDNA2Runtime::load_kernel(const std::string& name, size_t M, size_t K, size_t N,
                                      const std::string& xclbin_path, const std::string& insts_path) {
    // TODO: Load individual kernel using KernelLoader pattern
}

std::string WhisperXDNA2Runtime::select_kernel(size_t M, size_t K, size_t N) const {
    // Simple kernel selection: prefer exact match
    std::string exact = std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);
    if (kernel_apps_.find(exact) != kernel_apps_.end()) {
        return exact;
    }

    // Fall back to 512x512x512 for chunking
    std::string fallback = "512x512x512";
    if (kernel_apps_.find(fallback) != kernel_apps_.end()) {
        return fallback;
    }

    throw std::runtime_error("No suitable kernel found for " + exact);
}

} // namespace whisper_xdna2
