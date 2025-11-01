#include <Python.h>  // Must come first before kernel_loader.hpp
#include "kernel_loader.hpp"
#include <iostream>
#include <stdexcept>
#include <fstream>

namespace whisper_xdna2 {

KernelLoader::KernelLoader(PyObject* device_obj)
    : device_obj_(device_obj)
    , aie_utils_module_(nullptr)
{
    if (!device_obj_) {
        throw std::runtime_error("device_obj cannot be null");
    }
    Py_INCREF(device_obj_);
    init_python();
}

KernelLoader::~KernelLoader() {
    cleanup_python();
    if (device_obj_) {
        Py_DECREF(device_obj_);
        device_obj_ = nullptr;
    }
}

void KernelLoader::init_python() {
    // Import pyxrt module
    PyObject* pyxrt_module = PyImport_ImportModule("pyxrt");
    if (!pyxrt_module) {
        PyErr_Print();
        throw std::runtime_error("Failed to import pyxrt module");
    }
    Py_DECREF(pyxrt_module);

    // Import numpy
    PyObject* numpy_module = PyImport_ImportModule("numpy");
    if (!numpy_module) {
        PyErr_Print();
        throw std::runtime_error("Failed to import numpy module");
    }
    Py_DECREF(numpy_module);
}

void KernelLoader::cleanup_python() {
    // Clean up all kernels
    for (auto& info : kernels_) {
        // Release Python buffer objects
        for (int i = 0; i < 3; i++) {
            if (info.buffers[i]) {
                Py_DECREF(info.buffers[i]);
                info.buffers[i] = nullptr;
            }
        }
        // Release kernel object
        if (info.app) {
            Py_DECREF(info.app);
            info.app = nullptr;
        }
    }
    kernels_.clear();

    if (aie_utils_module_) {
        Py_DECREF(aie_utils_module_);
        aie_utils_module_ = nullptr;
    }
}

KernelInfo KernelLoader::load_kernel(
    const std::string& name,
    const std::string& xclbin_path,
    const std::string& insts_path,
    size_t M, size_t K, size_t N
) {
    // Check if files exist
    if (!file_exists(xclbin_path)) {
        throw std::runtime_error("XCLBIN not found: " + xclbin_path);
    }
    if (!file_exists(insts_path)) {
        throw std::runtime_error("Instructions not found: " + insts_path);
    }

    // CORRECT XRT API PATTERN for MLIR-AIE kernels
    // Following the pattern from XRT_API_FIX_GUIDE.md and test_int8_8tile_simple.py

    // 1. Load xclbin object
    PyObject* xrt_module = PyImport_ImportModule("pyxrt");
    if (!xrt_module) {
        PyErr_Print();
        throw std::runtime_error("Failed to import pyxrt");
    }

    PyObject* xclbin_class = PyObject_GetAttrString(xrt_module, "xclbin");
    if (!xclbin_class) {
        Py_DECREF(xrt_module);
        PyErr_Print();
        throw std::runtime_error("Failed to get xrt.xclbin class");
    }

    PyObject* xclbin_obj = PyObject_CallFunction(xclbin_class, "s", xclbin_path.c_str());
    Py_DECREF(xclbin_class);
    if (!xclbin_obj) {
        Py_DECREF(xrt_module);
        PyErr_Print();
        throw std::runtime_error("Failed to load xclbin: " + xclbin_path);
    }

    // 2. Register xclbin with device
    PyObject* register_result = PyObject_CallMethod(device_obj_, "register_xclbin", "O", xclbin_obj);
    if (!register_result) {
        Py_DECREF(xclbin_obj);
        Py_DECREF(xrt_module);
        PyErr_Print();
        throw std::runtime_error("Failed to register xclbin");
    }
    Py_DECREF(register_result);

    // 3. Get UUID from xclbin
    PyObject* uuid_obj = PyObject_CallMethod(xclbin_obj, "get_uuid", nullptr);
    if (!uuid_obj) {
        Py_DECREF(xclbin_obj);
        Py_DECREF(xrt_module);
        PyErr_Print();
        throw std::runtime_error("Failed to get xclbin UUID");
    }

    // 4. Create hw_context (CRITICAL - this is the key difference!)
    PyObject* hw_context_class = PyObject_GetAttrString(xrt_module, "hw_context");
    if (!hw_context_class) {
        Py_DECREF(uuid_obj);
        Py_DECREF(xclbin_obj);
        Py_DECREF(xrt_module);
        PyErr_Print();
        throw std::runtime_error("Failed to get xrt.hw_context class");
    }

    PyObject* context_obj = PyObject_CallFunction(hw_context_class, "OO", device_obj_, uuid_obj);
    Py_DECREF(hw_context_class);
    Py_DECREF(uuid_obj);
    if (!context_obj) {
        Py_DECREF(xclbin_obj);
        Py_DECREF(xrt_module);
        PyErr_Print();
        throw std::runtime_error("Failed to create hw_context");
    }

    // 5. Get kernel name from xclbin
    PyObject* kernels_list = PyObject_CallMethod(xclbin_obj, "get_kernels", nullptr);
    Py_DECREF(xclbin_obj);
    if (!kernels_list) {
        Py_DECREF(context_obj);
        Py_DECREF(xrt_module);
        PyErr_Print();
        throw std::runtime_error("Failed to get kernels from xclbin");
    }

    PyObject* first_kernel = PyList_GetItem(kernels_list, 0);
    if (!first_kernel) {
        Py_DECREF(kernels_list);
        Py_DECREF(context_obj);
        Py_DECREF(xrt_module);
        PyErr_Print();
        throw std::runtime_error("No kernels found in xclbin");
    }

    PyObject* kernel_name_obj = PyObject_CallMethod(first_kernel, "get_name", nullptr);
    Py_DECREF(kernels_list);
    if (!kernel_name_obj) {
        Py_DECREF(context_obj);
        Py_DECREF(xrt_module);
        PyErr_Print();
        throw std::runtime_error("Failed to get kernel name");
    }

    const char* kernel_name_cstr = PyUnicode_AsUTF8(kernel_name_obj);
    std::string kernel_name(kernel_name_cstr);
    Py_DECREF(kernel_name_obj);

    // 6. Create kernel from context (not device!)
    PyObject* kernel_class = PyObject_GetAttrString(xrt_module, "kernel");
    if (!kernel_class) {
        Py_DECREF(context_obj);
        Py_DECREF(xrt_module);
        PyErr_Print();
        throw std::runtime_error("Failed to get xrt.kernel class");
    }

    PyObject* kernel_obj = PyObject_CallFunction(kernel_class, "Os", context_obj, kernel_name.c_str());
    Py_DECREF(kernel_class);
    Py_DECREF(context_obj);
    Py_DECREF(xrt_module);
    if (!kernel_obj) {
        PyErr_Print();
        throw std::runtime_error("Failed to create kernel: " + kernel_name);
    }

    // Create KernelInfo
    KernelInfo info;
    info.name = name;
    info.xclbin_path = xclbin_path;
    info.insts_path = insts_path;
    info.M = M;
    info.K = K;
    info.N = N;
    info.app = kernel_obj;  // Store kernel object (not AIE_Application for direct XRT)

    // Buffers will be allocated on-demand
    info.buffers[0] = nullptr;
    info.buffers[1] = nullptr;
    info.buffers[2] = nullptr;

    // Store kernel
    kernels_.push_back(info);

    std::cout << "Loaded kernel '" << name << "' (" << M << "x" << K << "x" << N << ") from " << xclbin_path << std::endl;

    return info;
}

std::vector<KernelInfo> KernelLoader::load_standard_kernels(
    bool use_4tile,
    const std::string& kernel_dir
) {
    std::vector<KernelInfo> loaded;

    // Standard kernel configurations
    struct KernelConfig {
        size_t M, K, N;
        std::string suffix;
    };

    std::vector<KernelConfig> configs = {
        {512, 512, 512, "512x512x512"},
        {1024, 1024, 1024, "1024x1024x1024"},
        {256, 256, 256, "256x256x256"},
    };

    std::string tile_suffix = use_4tile ? "4c" : "32c";

    for (const auto& cfg : configs) {
        std::string xclbin_name = "final_" + cfg.suffix + "_64x64x64_" + tile_suffix + ".xclbin";
        std::string insts_name = "insts_" + cfg.suffix + "_64x64x64_" + tile_suffix + ".txt";

        std::string xclbin_path = kernel_dir + "/" + xclbin_name;
        std::string insts_path = kernel_dir + "/" + insts_name;

        if (file_exists(xclbin_path) && file_exists(insts_path)) {
            try {
                KernelInfo info = load_kernel(cfg.suffix, xclbin_path, insts_path, cfg.M, cfg.K, cfg.N);
                loaded.push_back(info);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to load kernel " << cfg.suffix << ": " << e.what() << std::endl;
            }
        }
    }

    return loaded;
}

const KernelInfo* KernelLoader::get_kernel(const std::string& name) const {
    for (const auto& info : kernels_) {
        if (info.name == name) {
            return &info;
        }
    }
    return nullptr;
}

bool KernelLoader::has_kernel(const std::string& name) const {
    return get_kernel(name) != nullptr;
}

std::vector<std::string> KernelLoader::get_kernel_names() const {
    std::vector<std::string> names;
    for (const auto& info : kernels_) {
        names.push_back(info.name);
    }
    return names;
}

std::string KernelLoader::select_kernel(size_t M, size_t K, size_t N) const {
    // Try exact match first
    for (const auto& info : kernels_) {
        if (info.M == M && info.K == K && info.N == N) {
            return info.name;
        }
    }

    // Fall back to chunking with 512x512x512 kernel
    for (const auto& info : kernels_) {
        if (info.M == 512 && info.K == 512 && info.N == 512) {
            return info.name;
        }
    }

    throw std::runtime_error("No suitable kernel found for dimensions " +
                           std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N));
}

bool KernelLoader::file_exists(const std::string& path) const {
    std::ifstream f(path);
    return f.good();
}

} // namespace whisper_xdna2
