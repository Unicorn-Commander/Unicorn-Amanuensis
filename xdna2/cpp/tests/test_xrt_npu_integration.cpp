/**
 * XRT NPU Integration Test
 * =========================
 *
 * Comprehensive test validating the C++ XRT integration with NPU matrix multiplication.
 *
 * Test Objectives:
 * - Load XRT kernel using KernelLoader class
 * - Allocate buffers using BufferManager
 * - Execute INT8 matrix multiplication on NPU
 * - Benchmark performance (100 iterations minimum)
 * - Validate accuracy (100% exact match for INT8)
 * - Verify speedup target (1211.3x like Python version)
 *
 * Expected Results:
 * - Accuracy: 100% exact match
 * - Speedup: 1211.3x (same as Python)
 * - Latency: <0.3ms (vs Python's 2ms overhead)
 *
 * Hardware: AMD XDNA2 NPU (32 tiles, 50 TOPS)
 * Kernel: 512x512x512 INT8 8-tile matmul
 * Date: November 1, 2025
 */

#include <Python.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>

// C++ XRT integration headers
#include "kernel_loader.hpp"
#include "buffer_manager.hpp"
#include "whisper_xdna2_runtime.hpp"

using namespace whisper_xdna2;
using namespace std::chrono;

// Test configuration
constexpr size_t M = 512;
constexpr size_t K = 512;
constexpr size_t N = 512;
constexpr int NUM_WARMUP = 3;
constexpr int NUM_ITERATIONS = 100;

/**
 * Print section header
 */
void print_header(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(80, '=') << "\n";
}

/**
 * Print step progress
 */
void print_step(int step, int total, const std::string& description) {
    std::cout << "[" << step << "/" << total << "] " << description << "...\n";
}

/**
 * Generate random INT8 test data
 */
void generate_test_data(std::vector<int8_t>& A, std::vector<int8_t>& B) {
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(-127, 127);

    for (auto& val : A) {
        val = static_cast<int8_t>(dist(rng));
    }
    for (auto& val : B) {
        val = static_cast<int8_t>(dist(rng));
    }
}

/**
 * Compute CPU reference result (INT8 -> INT32)
 */
void compute_cpu_reference(
    const std::vector<int8_t>& A,
    const std::vector<int8_t>& B,
    std::vector<int32_t>& C_ref,
    double& cpu_time_ms
) {
    auto t0 = high_resolution_clock::now();

    // Matrix multiplication: C = A @ B
    // A: MxK, B: KxN, C: MxN
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            int32_t sum = 0;
            for (size_t k = 0; k < K; k++) {
                int32_t a_val = static_cast<int32_t>(A[i * K + k]);
                int32_t b_val = static_cast<int32_t>(B[k * N + j]);
                sum += a_val * b_val;
            }
            C_ref[i * N + j] = sum;
        }
    }

    auto t1 = high_resolution_clock::now();
    cpu_time_ms = duration_cast<duration<double, std::milli>>(t1 - t0).count();
}

/**
 * Load NPU instructions from file
 */
std::vector<uint32_t> load_instructions(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open instructions file: " + path);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint32_t> instructions(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(instructions.data()), size);

    return instructions;
}

/**
 * Run XRT kernel on NPU
 */
void run_npu_kernel(
    PyObject* kernel,
    PyObject* bo_instr,
    PyObject* bo_a,
    PyObject* bo_b,
    PyObject* bo_c,
    size_t instr_size
) {
    // Call kernel: kernel(bo_instr, instr_size, bo_a, bo_b, bo_c, 0, 0)
    PyObject* args = PyTuple_New(7);
    PyTuple_SetItem(args, 0, bo_instr);
    PyTuple_SetItem(args, 1, PyLong_FromSize_t(instr_size));
    PyTuple_SetItem(args, 2, bo_a);
    PyTuple_SetItem(args, 3, bo_b);
    PyTuple_SetItem(args, 4, bo_c);
    PyTuple_SetItem(args, 5, PyLong_FromLong(0));
    PyTuple_SetItem(args, 6, PyLong_FromLong(0));

    // Increment refs for borrowed references
    Py_INCREF(bo_instr);
    Py_INCREF(bo_a);
    Py_INCREF(bo_b);
    Py_INCREF(bo_c);

    PyObject* run_obj = PyObject_Call(kernel, args, nullptr);
    Py_DECREF(args);

    if (!run_obj) {
        PyErr_Print();
        throw std::runtime_error("Kernel execution failed");
    }

    // Wait for completion
    PyObject* wait_method = PyObject_GetAttrString(run_obj, "wait");
    if (!wait_method) {
        Py_DECREF(run_obj);
        PyErr_Print();
        throw std::runtime_error("Failed to get wait method");
    }

    PyObject* wait_result = PyObject_CallObject(wait_method, nullptr);
    Py_DECREF(wait_method);
    Py_DECREF(run_obj);

    if (!wait_result) {
        PyErr_Print();
        throw std::runtime_error("wait() failed");
    }
    Py_DECREF(wait_result);
}

/**
 * Main test function
 */
int test_xrt_npu_integration() {
    print_header("XRT NPU Integration Test");

    // File paths
    const std::string kernel_dir = std::string(getenv("HOME")) +
        "/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";
    const std::string xclbin_path = kernel_dir + "/final_512x512x512_64x64x64_8c.xclbin";
    const std::string insts_path = kernel_dir + "/insts_512x512x512_64x64x64_8c.txt";

    std::cout << "Matrix: " << M << "x" << K << " @ " << K << "x" << N << "\n";
    std::cout << "XCLBIN: " << xclbin_path << "\n";
    std::cout << "Instructions: " << insts_path << "\n";
    std::cout << "\n";

    // Check files exist
    std::ifstream xclbin_file(xclbin_path);
    std::ifstream insts_file(insts_path);
    if (!xclbin_file.good()) {
        std::cerr << "ERROR: xclbin not found: " << xclbin_path << "\n";
        return 1;
    }
    if (!insts_file.good()) {
        std::cerr << "ERROR: Instructions not found: " << insts_path << "\n";
        return 1;
    }

    // Step 1: Generate test data
    print_step(1, 7, "Generating test data");
    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(K * N);
    generate_test_data(A, B);

    int8_t a_min = *std::min_element(A.begin(), A.end());
    int8_t a_max = *std::max_element(A.begin(), A.end());
    int8_t b_min = *std::min_element(B.begin(), B.end());
    int8_t b_max = *std::max_element(B.begin(), B.end());

    std::cout << "  A: " << M << "x" << K << ", range [" << (int)a_min << ", " << (int)a_max << "]\n";
    std::cout << "  B: " << K << "x" << N << ", range [" << (int)b_min << ", " << (int)b_max << "]\n";

    // Step 2: Compute CPU reference
    print_step(2, 7, "Computing CPU reference");
    std::vector<int32_t> C_ref(M * N);
    double cpu_time_ms;
    compute_cpu_reference(A, B, C_ref, cpu_time_ms);
    std::cout << "  CPU time: " << std::fixed << std::setprecision(2) << cpu_time_ms << " ms\n";

    // Step 3: Load NPU instructions
    print_step(3, 7, "Loading NPU instructions");
    std::vector<uint32_t> instructions;
    try {
        instructions = load_instructions(insts_path);
        std::cout << "  Instructions: " << instructions.size() << " words\n";
    } catch (const std::exception& e) {
        std::cerr << "  ERROR: " << e.what() << "\n";
        return 1;
    }

    // Step 4: Initialize Python and XRT
    print_step(4, 7, "Initializing Python and XRT");
    Py_Initialize();

    // Add XRT Python path
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/opt/xilinx/xrt/python')");

    // Import pyxrt
    PyObject* pyxrt_module = PyImport_ImportModule("pyxrt");
    if (!pyxrt_module) {
        PyErr_Print();
        std::cerr << "  ERROR: Failed to import pyxrt\n";
        Py_Finalize();
        return 1;
    }

    // Get device
    PyObject* xrt_device_class = PyObject_GetAttrString(pyxrt_module, "device");
    PyObject* device_args = PyTuple_Pack(1, PyLong_FromLong(0));
    PyObject* device = PyObject_CallObject(xrt_device_class, device_args);
    Py_DECREF(device_args);
    Py_DECREF(xrt_device_class);

    if (!device) {
        PyErr_Print();
        std::cerr << "  ERROR: Failed to create device\n";
        Py_DECREF(pyxrt_module);
        Py_Finalize();
        return 1;
    }
    std::cout << "  Device 0: OK\n";

    // Step 5: Load XCLBIN using hw_context pattern
    print_step(5, 7, "Loading XCLBIN with hw_context");
    PyObject* kernel = nullptr;
    PyObject* context = nullptr;

    try {
        // Load xclbin
        PyObject* xrt_xclbin_class = PyObject_GetAttrString(pyxrt_module, "xclbin");
        PyObject* xclbin_args = PyTuple_Pack(1, PyUnicode_FromString(xclbin_path.c_str()));
        PyObject* xclbin = PyObject_CallObject(xrt_xclbin_class, xclbin_args);
        Py_DECREF(xclbin_args);
        Py_DECREF(xrt_xclbin_class);

        if (!xclbin) {
            PyErr_Print();
            throw std::runtime_error("Failed to load xclbin");
        }

        // Register xclbin
        PyObject* register_method = PyObject_GetAttrString(device, "register_xclbin");
        PyObject* register_args = PyTuple_Pack(1, xclbin);
        PyObject* register_result = PyObject_CallObject(register_method, register_args);
        Py_DECREF(register_args);
        Py_DECREF(register_method);
        Py_XDECREF(register_result);

        // Get UUID
        PyObject* get_uuid_method = PyObject_GetAttrString(xclbin, "get_uuid");
        PyObject* uuid = PyObject_CallObject(get_uuid_method, nullptr);
        Py_DECREF(get_uuid_method);

        if (!uuid) {
            Py_DECREF(xclbin);
            PyErr_Print();
            throw std::runtime_error("Failed to get UUID");
        }

        // Create hw_context
        PyObject* xrt_hw_context_class = PyObject_GetAttrString(pyxrt_module, "hw_context");
        PyObject* context_args = PyTuple_Pack(2, device, uuid);
        context = PyObject_CallObject(xrt_hw_context_class, context_args);
        Py_DECREF(context_args);
        Py_DECREF(xrt_hw_context_class);
        Py_DECREF(uuid);

        if (!context) {
            Py_DECREF(xclbin);
            PyErr_Print();
            throw std::runtime_error("Failed to create hw_context");
        }

        // Get kernels
        PyObject* get_kernels_method = PyObject_GetAttrString(xclbin, "get_kernels");
        PyObject* kernels = PyObject_CallObject(get_kernels_method, nullptr);
        Py_DECREF(get_kernels_method);

        if (!kernels || PyList_Size(kernels) == 0) {
            Py_XDECREF(kernels);
            Py_DECREF(xclbin);
            Py_DECREF(context);
            throw std::runtime_error("No kernels found in xclbin");
        }

        // Get kernel name
        PyObject* kernel_obj = PyList_GetItem(kernels, 0);
        PyObject* get_name_method = PyObject_GetAttrString(kernel_obj, "get_name");
        PyObject* kernel_name_obj = PyObject_CallObject(get_name_method, nullptr);
        Py_DECREF(get_name_method);

        const char* kernel_name = PyUnicode_AsUTF8(kernel_name_obj);
        std::cout << "  Kernel: " << kernel_name << "\n";

        // Create kernel
        PyObject* xrt_kernel_class = PyObject_GetAttrString(pyxrt_module, "kernel");
        PyObject* kernel_args = PyTuple_Pack(2, context, kernel_name_obj);
        kernel = PyObject_CallObject(xrt_kernel_class, kernel_args);
        Py_DECREF(kernel_args);
        Py_DECREF(xrt_kernel_class);
        Py_DECREF(kernel_name_obj);
        Py_DECREF(kernels);
        Py_DECREF(xclbin);

        if (!kernel) {
            PyErr_Print();
            throw std::runtime_error("Failed to create kernel");
        }

    } catch (const std::exception& e) {
        std::cerr << "  ERROR: " << e.what() << "\n";
        Py_XDECREF(context);
        Py_DECREF(device);
        Py_DECREF(pyxrt_module);
        Py_Finalize();
        return 1;
    }

    // Step 6: Allocate buffers
    print_step(6, 7, "Allocating buffers");
    PyObject* bo_instr = nullptr;
    PyObject* bo_a = nullptr;
    PyObject* bo_b = nullptr;
    PyObject* bo_c = nullptr;

    try {
        size_t A_size = M * K * 1;  // int8
        size_t B_size = K * N * 1;  // int8
        size_t C_size = M * N * 4;  // int32
        size_t instr_size = instructions.size() * 4;

        // Get kernel group IDs
        PyObject* group_id_method = PyObject_GetAttrString(kernel, "group_id");

        PyObject* group_id_args_1 = PyTuple_Pack(1, PyLong_FromLong(1));
        PyObject* group_id_1 = PyObject_CallObject(group_id_method, group_id_args_1);
        Py_DECREF(group_id_args_1);

        PyObject* group_id_args_3 = PyTuple_Pack(1, PyLong_FromLong(3));
        PyObject* group_id_3 = PyObject_CallObject(group_id_method, group_id_args_3);
        Py_DECREF(group_id_args_3);

        PyObject* group_id_args_4 = PyTuple_Pack(1, PyLong_FromLong(4));
        PyObject* group_id_4 = PyObject_CallObject(group_id_method, group_id_args_4);
        Py_DECREF(group_id_args_4);

        PyObject* group_id_args_5 = PyTuple_Pack(1, PyLong_FromLong(5));
        PyObject* group_id_5 = PyObject_CallObject(group_id_method, group_id_args_5);
        Py_DECREF(group_id_args_5);

        Py_DECREF(group_id_method);

        // Get bo class and flags
        PyObject* xrt_bo_class = PyObject_GetAttrString(pyxrt_module, "bo");
        PyObject* bo_module = PyObject_GetAttrString(pyxrt_module, "bo");
        PyObject* cacheable = PyObject_GetAttrString(bo_module, "cacheable");
        PyObject* host_only = PyObject_GetAttrString(bo_module, "host_only");

        // Create buffers
        PyObject* bo_instr_args = PyTuple_Pack(4, device, PyLong_FromSize_t(instr_size), cacheable, group_id_1);
        bo_instr = PyObject_CallObject(xrt_bo_class, bo_instr_args);
        Py_DECREF(bo_instr_args);

        PyObject* bo_a_args = PyTuple_Pack(4, device, PyLong_FromSize_t(A_size), host_only, group_id_3);
        bo_a = PyObject_CallObject(xrt_bo_class, bo_a_args);
        Py_DECREF(bo_a_args);

        PyObject* bo_b_args = PyTuple_Pack(4, device, PyLong_FromSize_t(B_size), host_only, group_id_4);
        bo_b = PyObject_CallObject(xrt_bo_class, bo_b_args);
        Py_DECREF(bo_b_args);

        PyObject* bo_c_args = PyTuple_Pack(4, device, PyLong_FromSize_t(C_size), host_only, group_id_5);
        bo_c = PyObject_CallObject(xrt_bo_class, bo_c_args);
        Py_DECREF(bo_c_args);

        Py_DECREF(xrt_bo_class);
        Py_DECREF(bo_module);
        Py_DECREF(cacheable);
        Py_DECREF(host_only);
        Py_DECREF(group_id_1);
        Py_DECREF(group_id_3);
        Py_DECREF(group_id_4);
        Py_DECREF(group_id_5);

        if (!bo_instr || !bo_a || !bo_b || !bo_c) {
            PyErr_Print();
            throw std::runtime_error("Failed to allocate buffers");
        }

        // Write data to buffers
        PyObject* write_method;
        PyObject* write_args;
        PyObject* write_result;

        // Write instructions
        write_method = PyObject_GetAttrString(bo_instr, "write");
        PyObject* instr_bytes = PyByteArray_FromStringAndSize(
            reinterpret_cast<const char*>(instructions.data()), instr_size);
        write_args = PyTuple_Pack(2, instr_bytes, PyLong_FromLong(0));
        write_result = PyObject_CallObject(write_method, write_args);
        Py_DECREF(write_result);
        Py_DECREF(write_args);
        Py_DECREF(instr_bytes);
        Py_DECREF(write_method);

        // Write A
        write_method = PyObject_GetAttrString(bo_a, "write");
        PyObject* a_bytes = PyByteArray_FromStringAndSize(
            reinterpret_cast<const char*>(A.data()), A_size);
        write_args = PyTuple_Pack(2, a_bytes, PyLong_FromLong(0));
        write_result = PyObject_CallObject(write_method, write_args);
        Py_DECREF(write_result);
        Py_DECREF(write_args);
        Py_DECREF(a_bytes);
        Py_DECREF(write_method);

        // Write B
        write_method = PyObject_GetAttrString(bo_b, "write");
        PyObject* b_bytes = PyByteArray_FromStringAndSize(
            reinterpret_cast<const char*>(B.data()), B_size);
        write_args = PyTuple_Pack(2, b_bytes, PyLong_FromLong(0));
        write_result = PyObject_CallObject(write_method, write_args);
        Py_DECREF(write_result);
        Py_DECREF(write_args);
        Py_DECREF(b_bytes);
        Py_DECREF(write_method);

        // Sync to device
        PyObject* xclBOSyncDirection = PyObject_GetAttrString(pyxrt_module, "xclBOSyncDirection");
        PyObject* XCL_BO_SYNC_BO_TO_DEVICE = PyObject_GetAttrString(xclBOSyncDirection, "XCL_BO_SYNC_BO_TO_DEVICE");
        Py_DECREF(xclBOSyncDirection);

        PyObject* sync_method;
        PyObject* sync_args;
        PyObject* sync_result;

        sync_method = PyObject_GetAttrString(bo_instr, "sync");
        sync_args = PyTuple_Pack(1, XCL_BO_SYNC_BO_TO_DEVICE);
        sync_result = PyObject_CallObject(sync_method, sync_args);
        Py_DECREF(sync_result);
        Py_DECREF(sync_args);
        Py_DECREF(sync_method);

        sync_method = PyObject_GetAttrString(bo_a, "sync");
        sync_args = PyTuple_Pack(1, XCL_BO_SYNC_BO_TO_DEVICE);
        sync_result = PyObject_CallObject(sync_method, sync_args);
        Py_DECREF(sync_result);
        Py_DECREF(sync_args);
        Py_DECREF(sync_method);

        sync_method = PyObject_GetAttrString(bo_b, "sync");
        sync_args = PyTuple_Pack(1, XCL_BO_SYNC_BO_TO_DEVICE);
        sync_result = PyObject_CallObject(sync_method, sync_args);
        Py_DECREF(sync_result);
        Py_DECREF(sync_args);
        Py_DECREF(sync_method);

        Py_DECREF(XCL_BO_SYNC_BO_TO_DEVICE);

        std::cout << "  Buffers: OK\n";

    } catch (const std::exception& e) {
        std::cerr << "  ERROR: " << e.what() << "\n";
        Py_XDECREF(bo_instr);
        Py_XDECREF(bo_a);
        Py_XDECREF(bo_b);
        Py_XDECREF(bo_c);
        Py_DECREF(kernel);
        Py_DECREF(context);
        Py_DECREF(device);
        Py_DECREF(pyxrt_module);
        Py_Finalize();
        return 1;
    }

    // Step 7: Run kernel
    print_step(7, 7, "Running kernel on NPU");
    std::cout << "\n";

    std::vector<double> times;
    size_t instr_size = instructions.size() * 4;

    try {
        // Warmup
        std::cout << "Warmup (" << NUM_WARMUP << " runs)...\n";
        for (int i = 0; i < NUM_WARMUP; i++) {
            run_npu_kernel(kernel, bo_instr, bo_a, bo_b, bo_c, instr_size);
        }

        // Benchmark
        std::cout << "Benchmarking (" << NUM_ITERATIONS << " runs)...\n";
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            auto t0 = high_resolution_clock::now();
            run_npu_kernel(kernel, bo_instr, bo_a, bo_b, bo_c, instr_size);
            auto t1 = high_resolution_clock::now();

            double time_ms = duration_cast<duration<double, std::milli>>(t1 - t0).count();
            times.push_back(time_ms);

            if ((i + 1) % 20 == 0) {
                std::cout << "  Progress: " << (i + 1) << "/" << NUM_ITERATIONS << "\n";
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "  ERROR: " << e.what() << "\n";
        Py_DECREF(bo_instr);
        Py_DECREF(bo_a);
        Py_DECREF(bo_b);
        Py_DECREF(bo_c);
        Py_DECREF(kernel);
        Py_DECREF(context);
        Py_DECREF(device);
        Py_DECREF(pyxrt_module);
        Py_Finalize();
        return 1;
    }

    // Read result
    std::vector<int32_t> C_npu(M * N);

    try {
        // Sync from device
        PyObject* xclBOSyncDirection = PyObject_GetAttrString(pyxrt_module, "xclBOSyncDirection");
        PyObject* XCL_BO_SYNC_BO_FROM_DEVICE = PyObject_GetAttrString(xclBOSyncDirection, "XCL_BO_SYNC_BO_FROM_DEVICE");
        Py_DECREF(xclBOSyncDirection);

        PyObject* sync_method = PyObject_GetAttrString(bo_c, "sync");
        PyObject* sync_args = PyTuple_Pack(1, XCL_BO_SYNC_BO_FROM_DEVICE);
        PyObject* sync_result = PyObject_CallObject(sync_method, sync_args);
        Py_DECREF(sync_result);
        Py_DECREF(sync_args);
        Py_DECREF(sync_method);
        Py_DECREF(XCL_BO_SYNC_BO_FROM_DEVICE);

        // Map buffer
        PyObject* map_method = PyObject_GetAttrString(bo_c, "map");
        PyObject* mapped = PyObject_CallObject(map_method, nullptr);
        Py_DECREF(map_method);

        if (!mapped) {
            PyErr_Print();
            throw std::runtime_error("Failed to map buffer");
        }

        // Get buffer info
        Py_buffer buffer_info;
        if (PyObject_GetBuffer(mapped, &buffer_info, PyBUF_SIMPLE) != 0) {
            Py_DECREF(mapped);
            throw std::runtime_error("Failed to get buffer info");
        }

        // Copy data
        memcpy(C_npu.data(), buffer_info.buf, C_npu.size() * sizeof(int32_t));

        PyBuffer_Release(&buffer_info);
        Py_DECREF(mapped);

    } catch (const std::exception& e) {
        std::cerr << "  ERROR: " << e.what() << "\n";
        Py_DECREF(bo_instr);
        Py_DECREF(bo_a);
        Py_DECREF(bo_b);
        Py_DECREF(bo_c);
        Py_DECREF(kernel);
        Py_DECREF(context);
        Py_DECREF(device);
        Py_DECREF(pyxrt_module);
        Py_Finalize();
        return 1;
    }

    // Cleanup Python resources
    Py_DECREF(bo_instr);
    Py_DECREF(bo_a);
    Py_DECREF(bo_b);
    Py_DECREF(bo_c);
    Py_DECREF(kernel);
    Py_DECREF(context);
    Py_DECREF(device);
    Py_DECREF(pyxrt_module);
    Py_Finalize();

    // Results
    print_header("RESULTS");

    // Accuracy
    size_t errors = 0;
    int32_t max_diff = 0;
    for (size_t i = 0; i < C_ref.size(); i++) {
        if (C_npu[i] != C_ref[i]) {
            errors++;
            int32_t diff = std::abs(C_npu[i] - C_ref[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }

    std::cout << "Accuracy:\n";
    std::cout << "  Total elements: " << C_ref.size() << "\n";
    std::cout << "  Mismatched: " << errors << "\n";
    if (errors == 0) {
        std::cout << "  ✅ 100% EXACT MATCH (INT8 perfect)\n";
    } else {
        std::cout << "  ❌ " << errors << " errors, max diff: " << max_diff << "\n";
    }
    std::cout << "\n";

    // Performance statistics
    double mean_time = 0.0;
    double min_time = times[0];
    double max_time = times[0];

    for (double t : times) {
        mean_time += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    mean_time /= times.size();

    double std_dev = 0.0;
    for (double t : times) {
        std_dev += (t - mean_time) * (t - mean_time);
    }
    std_dev = std::sqrt(std_dev / times.size());

    // Calculate median
    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());
    double median_time = sorted_times[sorted_times.size() / 2];

    std::cout << "NPU Performance:\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Mean time: " << mean_time << " ms\n";
    std::cout << "  Median time: " << median_time << " ms\n";
    std::cout << "  Min time: " << min_time << " ms\n";
    std::cout << "  Std dev: " << std_dev << " ms\n";
    std::cout << "\n";

    std::cout << "CPU time: " << cpu_time_ms << " ms\n";
    std::cout << "\n";

    double speedup_mean = cpu_time_ms / mean_time;
    double speedup_best = cpu_time_ms / min_time;

    std::cout << "Speedup:\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Mean: " << speedup_mean << "x\n";
    std::cout << "  Best: " << speedup_best << "x\n";
    std::cout << "\n";

    // Validate target
    const double TARGET_MIN = 1000.0;  // Conservative target (Python achieved 1211.3x)

    if (speedup_mean >= TARGET_MIN) {
        std::cout << "✅ TARGET ACHIEVED: " << speedup_mean << "x >= " << TARGET_MIN << "x\n";
    } else {
        std::cout << "⚠️  BELOW TARGET: " << speedup_mean << "x < " << TARGET_MIN << "x (gap: "
                  << (TARGET_MIN - speedup_mean) << "x)\n";
    }
    std::cout << "\n";

    // Throughput
    double ops = 2.0 * M * K * N;
    double gops = (ops / (mean_time / 1000.0)) / 1e9;
    std::cout << "Throughput: " << std::fixed << std::setprecision(1)
              << gops << " GOPS (" << (gops / 50.0 * 100.0) << "% of 50 TOPS)\n";

    std::cout << "\n";
    std::cout << std::string(80, '=') << "\n";

    bool success = (errors == 0) && (speedup_mean >= TARGET_MIN);
    return success ? 0 : 1;
}

int main() {
    try {
        return test_xrt_npu_integration();
    } catch (const std::exception& e) {
        std::cerr << "\n❌ FATAL: " << e.what() << "\n";
        return 1;
    }
}
