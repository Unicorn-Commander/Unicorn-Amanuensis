#pragma once

/**
 * Native XRT C API Wrapper
 *
 * C API for Python ctypes/pybind11 FFI integration.
 * Wraps xrt_native.hpp for use from Python.
 *
 * Usage from Python:
 *   lib = ctypes.cdll.LoadLibrary("libxrt_native.so")
 *   handle = lib.xrt_native_create("base", False)
 *   lib.xrt_native_initialize(handle, xclbin_path)
 *   lib.xrt_native_run_matmul(handle, A, B, C, M, K, N)
 *
 * Author: CC-1L Native XRT Team
 * Date: November 1, 2025
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle to XRTNative instance
 */
typedef void* XRTNativeHandle;

/**
 * Create native XRT runtime instance
 *
 * @param model_size Model size ("base", "small", etc.)
 * @param use_4tile Use 4-tile kernels (1) or 32-tile (0)
 * @return Handle to runtime instance (NULL on failure)
 */
XRTNativeHandle xrt_native_create(const char* model_size, bool use_4tile);

/**
 * Destroy native XRT runtime instance
 *
 * @param handle Runtime handle from xrt_native_create()
 */
void xrt_native_destroy(XRTNativeHandle handle);

/**
 * Initialize XRT device and load xclbin
 *
 * @param handle Runtime handle
 * @param xclbin_path Path to .xclbin file
 * @return 0 on success, -1 on failure
 */
int xrt_native_initialize(XRTNativeHandle handle, const char* xclbin_path);

/**
 * Check if initialized
 *
 * @param handle Runtime handle
 * @return 1 if initialized, 0 otherwise
 */
int xrt_native_is_initialized(XRTNativeHandle handle);

/**
 * Create buffer on device
 *
 * @param handle Runtime handle
 * @param size Buffer size in bytes
 * @param flags Buffer flags (XCL_BO_FLAGS_CACHEABLE=1, XRT_BO_FLAGS_HOST_ONLY=0)
 * @param group_id Memory bank group ID
 * @return Buffer ID (0 on failure)
 */
size_t xrt_native_create_buffer(XRTNativeHandle handle, size_t size,
                                 uint32_t flags, int group_id);

/**
 * Write data to buffer
 *
 * @param handle Runtime handle
 * @param buffer_id Buffer ID
 * @param data Source data pointer
 * @param size Size in bytes
 * @return 0 on success, -1 on failure
 */
int xrt_native_write_buffer(XRTNativeHandle handle, size_t buffer_id,
                             const void* data, size_t size);

/**
 * Read data from buffer
 *
 * @param handle Runtime handle
 * @param buffer_id Buffer ID
 * @param data Destination data pointer
 * @param size Size in bytes
 * @return 0 on success, -1 on failure
 */
int xrt_native_read_buffer(XRTNativeHandle handle, size_t buffer_id,
                            void* data, size_t size);

/**
 * Sync buffer to/from device
 *
 * @param handle Runtime handle
 * @param buffer_id Buffer ID
 * @param to_device 1=host→device, 0=device→host
 * @return 0 on success, -1 on failure
 */
int xrt_native_sync_buffer(XRTNativeHandle handle, size_t buffer_id, bool to_device);

/**
 * Load instructions from file
 *
 * @param handle Runtime handle
 * @param insts_path Path to instructions .txt file
 * @return Instruction buffer ID (0 on failure)
 */
size_t xrt_native_load_instructions(XRTNativeHandle handle, const char* insts_path);

/**
 * Execute kernel on NPU
 *
 * @param handle Runtime handle
 * @param bo_instr Instruction buffer ID
 * @param instr_size Instruction size in bytes
 * @param bo_a Input buffer A ID
 * @param bo_b Input buffer B ID
 * @param bo_c Output buffer C ID
 * @return 0 on success, -1 on failure
 */
int xrt_native_run_kernel(XRTNativeHandle handle, size_t bo_instr, size_t instr_size,
                          size_t bo_a, size_t bo_b, size_t bo_c);

/**
 * Run matrix multiplication on NPU
 *
 * @param handle Runtime handle
 * @param A Input matrix A (int8) - MxK
 * @param B Input matrix B (int8) - KxN
 * @param C Output matrix C (int32) - MxN
 * @param M Rows in A and C
 * @param K Cols in A, rows in B
 * @param N Cols in B and C
 * @return 0 on success, -1 on failure
 */
int xrt_native_run_matmul(XRTNativeHandle handle,
                          const int8_t* A, const int8_t* B, int32_t* C,
                          size_t M, size_t K, size_t N);

/**
 * Get kernel group ID
 *
 * @param handle Runtime handle
 * @param arg_index Kernel argument index (1=instr, 3=A, 4=B, 5=C)
 * @return Group ID (-1 on failure)
 */
int xrt_native_get_group_id(XRTNativeHandle handle, int arg_index);

/**
 * Release buffer
 *
 * @param handle Runtime handle
 * @param buffer_id Buffer ID
 */
void xrt_native_release_buffer(XRTNativeHandle handle, size_t buffer_id);

/**
 * Get model dimensions
 */
struct XRTNativeModelDims {
    size_t n_mels;
    size_t n_ctx;
    size_t n_state;
    size_t n_head;
    size_t n_layer;
};

/**
 * Get model dimensions
 *
 * @param handle Runtime handle
 * @param dims Output dimensions structure
 * @return 0 on success, -1 on failure
 */
int xrt_native_get_model_dims(XRTNativeHandle handle, struct XRTNativeModelDims* dims);

/**
 * Get performance statistics
 */
struct XRTNativePerfStats {
    double total_kernel_ms;
    double avg_kernel_ms;
    size_t num_kernel_calls;
    double min_kernel_ms;
    double max_kernel_ms;
};

/**
 * Get performance statistics
 *
 * @param handle Runtime handle
 * @param stats Output statistics structure
 * @return 0 on success, -1 on failure
 */
int xrt_native_get_perf_stats(XRTNativeHandle handle, struct XRTNativePerfStats* stats);

/**
 * Reset performance statistics
 *
 * @param handle Runtime handle
 */
void xrt_native_reset_perf_stats(XRTNativeHandle handle);

/**
 * Get version string
 *
 * @return Version string (e.g., "1.0.0-native-xrt")
 */
const char* xrt_native_get_version();

#ifdef __cplusplus
}
#endif
