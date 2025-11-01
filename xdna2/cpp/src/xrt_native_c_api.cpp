/**
 * Native XRT C API Implementation
 *
 * C wrapper around xrt_native.hpp for Python FFI integration.
 *
 * Author: CC-1L Native XRT Team
 * Date: November 1, 2025
 */

#include "xrt_native_c_api.h"
#include "xrt_native.hpp"
#include <cstring>
#include <iostream>

using namespace whisper_xdna2::native;

// Version string
static const char* VERSION = "1.0.0-native-xrt";

extern "C" {

XRTNativeHandle xrt_native_create(const char* model_size, bool use_4tile) {
    try {
        auto* runtime = new XRTNative(model_size, use_4tile);
        return static_cast<XRTNativeHandle>(runtime);
    } catch (const std::exception& e) {
        std::cerr << "[xrt_native_c_api] Create failed: " << e.what() << std::endl;
        return nullptr;
    }
}

void xrt_native_destroy(XRTNativeHandle handle) {
    if (handle) {
        auto* runtime = static_cast<XRTNative*>(handle);
        delete runtime;
    }
}

int xrt_native_initialize(XRTNativeHandle handle, const char* xclbin_path) {
    if (!handle || !xclbin_path) {
        return -1;
    }

    try {
        auto* runtime = static_cast<XRTNative*>(handle);
        runtime->initialize(xclbin_path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[xrt_native_c_api] Initialize failed: " << e.what() << std::endl;
        return -1;
    }
}

int xrt_native_is_initialized(XRTNativeHandle handle) {
    if (!handle) {
        return 0;
    }

    auto* runtime = static_cast<XRTNative*>(handle);
    return runtime->is_initialized() ? 1 : 0;
}

size_t xrt_native_create_buffer(XRTNativeHandle handle, size_t size,
                                 uint32_t flags, int group_id) {
    if (!handle) {
        return 0;
    }

    try {
        auto* runtime = static_cast<XRTNative*>(handle);
        return runtime->create_buffer(size, flags, group_id);
    } catch (const std::exception& e) {
        std::cerr << "[xrt_native_c_api] Create buffer failed: " << e.what() << std::endl;
        return 0;
    }
}

int xrt_native_write_buffer(XRTNativeHandle handle, size_t buffer_id,
                             const void* data, size_t size) {
    if (!handle || !data) {
        return -1;
    }

    try {
        auto* runtime = static_cast<XRTNative*>(handle);
        runtime->write_buffer(buffer_id, data, size);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[xrt_native_c_api] Write buffer failed: " << e.what() << std::endl;
        return -1;
    }
}

int xrt_native_read_buffer(XRTNativeHandle handle, size_t buffer_id,
                            void* data, size_t size) {
    if (!handle || !data) {
        return -1;
    }

    try {
        auto* runtime = static_cast<XRTNative*>(handle);
        runtime->read_buffer(buffer_id, data, size);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[xrt_native_c_api] Read buffer failed: " << e.what() << std::endl;
        return -1;
    }
}

int xrt_native_sync_buffer(XRTNativeHandle handle, size_t buffer_id, bool to_device) {
    if (!handle) {
        return -1;
    }

    try {
        auto* runtime = static_cast<XRTNative*>(handle);
        runtime->sync_buffer(buffer_id, to_device);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[xrt_native_c_api] Sync buffer failed: " << e.what() << std::endl;
        return -1;
    }
}

size_t xrt_native_load_instructions(XRTNativeHandle handle, const char* insts_path) {
    if (!handle || !insts_path) {
        return 0;
    }

    try {
        auto* runtime = static_cast<XRTNative*>(handle);
        return runtime->load_instructions(insts_path);
    } catch (const std::exception& e) {
        std::cerr << "[xrt_native_c_api] Load instructions failed: " << e.what() << std::endl;
        return 0;
    }
}

int xrt_native_run_kernel(XRTNativeHandle handle, size_t bo_instr, size_t instr_size,
                          size_t bo_a, size_t bo_b, size_t bo_c) {
    if (!handle) {
        return -1;
    }

    try {
        auto* runtime = static_cast<XRTNative*>(handle);
        runtime->run_kernel(bo_instr, instr_size, bo_a, bo_b, bo_c);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[xrt_native_c_api] Run kernel failed: " << e.what() << std::endl;
        return -1;
    }
}

int xrt_native_run_matmul(XRTNativeHandle handle,
                          const int8_t* A, const int8_t* B, int32_t* C,
                          size_t M, size_t K, size_t N) {
    if (!handle || !A || !B || !C) {
        return -1;
    }

    try {
        auto* runtime = static_cast<XRTNative*>(handle);
        runtime->run_matmul(A, B, C, M, K, N);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[xrt_native_c_api] Run matmul failed: " << e.what() << std::endl;
        return -1;
    }
}

int xrt_native_get_group_id(XRTNativeHandle handle, int arg_index) {
    if (!handle) {
        return -1;
    }

    try {
        auto* runtime = static_cast<XRTNative*>(handle);
        return runtime->get_group_id(arg_index);
    } catch (const std::exception& e) {
        std::cerr << "[xrt_native_c_api] Get group ID failed: " << e.what() << std::endl;
        return -1;
    }
}

void xrt_native_release_buffer(XRTNativeHandle handle, size_t buffer_id) {
    if (!handle) {
        return;
    }

    try {
        auto* runtime = static_cast<XRTNative*>(handle);
        runtime->release_buffer(buffer_id);
    } catch (const std::exception& e) {
        std::cerr << "[xrt_native_c_api] Release buffer failed: " << e.what() << std::endl;
    }
}

int xrt_native_get_model_dims(XRTNativeHandle handle, struct XRTNativeModelDims* dims) {
    if (!handle || !dims) {
        return -1;
    }

    try {
        auto* runtime = static_cast<XRTNative*>(handle);
        auto model_dims = runtime->get_model_dims();
        dims->n_mels = model_dims.n_mels;
        dims->n_ctx = model_dims.n_ctx;
        dims->n_state = model_dims.n_state;
        dims->n_head = model_dims.n_head;
        dims->n_layer = model_dims.n_layer;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[xrt_native_c_api] Get model dims failed: " << e.what() << std::endl;
        return -1;
    }
}

int xrt_native_get_perf_stats(XRTNativeHandle handle, struct XRTNativePerfStats* stats) {
    if (!handle || !stats) {
        return -1;
    }

    try {
        auto* runtime = static_cast<XRTNative*>(handle);
        auto perf_stats = runtime->get_perf_stats();
        stats->total_kernel_ms = perf_stats.total_kernel_ms;
        stats->avg_kernel_ms = perf_stats.avg_kernel_ms;
        stats->num_kernel_calls = perf_stats.num_kernel_calls;
        stats->min_kernel_ms = perf_stats.min_kernel_ms;
        stats->max_kernel_ms = perf_stats.max_kernel_ms;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[xrt_native_c_api] Get perf stats failed: " << e.what() << std::endl;
        return -1;
    }
}

void xrt_native_reset_perf_stats(XRTNativeHandle handle) {
    if (!handle) {
        return;
    }

    try {
        auto* runtime = static_cast<XRTNative*>(handle);
        runtime->reset_perf_stats();
    } catch (const std::exception& e) {
        std::cerr << "[xrt_native_c_api] Reset perf stats failed: " << e.what() << std::endl;
    }
}

const char* xrt_native_get_version() {
    return VERSION;
}

} // extern "C"
