#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

// Forward declaration - PyObject is defined in Python.h
// Include Python.h in .cpp files before this header
struct _object;
typedef struct _object PyObject;

namespace whisper_xdna2 {

/**
 * KernelInfo - Information about a loaded kernel
 */
struct KernelInfo {
    std::string name;          // Kernel name (e.g., "512x512x512")
    std::string xclbin_path;   // Path to xclbin file
    std::string insts_path;    // Path to instructions file
    size_t M, K, N;            // Matrix dimensions
    PyObject* app;             // AIE_Application object
    PyObject* buffers[3];      // Input A, Input B, Output C buffers

    KernelInfo() : M(0), K(0), N(0), app(nullptr) {
        buffers[0] = buffers[1] = buffers[2] = nullptr;
    }
};

/**
 * KernelLoader - Load and manage NPU kernels
 *
 * Handles loading of XCLBIN files and creation of AIE applications.
 * Manages multiple kernel variants (4-tile, 32-tile, different dimensions).
 *
 * Kernel Selection Strategy:
 * - Prefer exact dimension match
 * - Fall back to chunking with 512x512x512 kernel
 * - Support both 4-tile (stable) and 32-tile (100% NPU utilization)
 */
class KernelLoader {
public:
    /**
     * Construct kernel loader
     *
     * @param device_obj Python pyxrt.device object
     */
    explicit KernelLoader(PyObject* device_obj);

    /**
     * Destructor - cleans up all kernels
     */
    ~KernelLoader();

    // Disable copy
    KernelLoader(const KernelLoader&) = delete;
    KernelLoader& operator=(const KernelLoader&) = delete;

    /**
     * Load a single kernel from xclbin and instructions
     *
     * @param name Kernel identifier (e.g., "512x512x512")
     * @param xclbin_path Path to .xclbin file
     * @param insts_path Path to .bin instructions file
     * @param M Matrix dimension M (rows of A)
     * @param K Matrix dimension K (cols of A, rows of B)
     * @param N Matrix dimension N (cols of B)
     * @return KernelInfo for the loaded kernel
     * @throws std::runtime_error if loading fails
     */
    KernelInfo load_kernel(
        const std::string& name,
        const std::string& xclbin_path,
        const std::string& insts_path,
        size_t M, size_t K, size_t N
    );

    /**
     * Load all standard kernels for a tile configuration
     *
     * @param use_4tile True for 4-tile kernels, false for 32-tile
     * @param kernel_dir Directory containing kernel files
     * @return Vector of loaded KernelInfo
     */
    std::vector<KernelInfo> load_standard_kernels(
        bool use_4tile,
        const std::string& kernel_dir
    );

    /**
     * Get kernel by name
     *
     * @param name Kernel name
     * @return Pointer to KernelInfo, or nullptr if not found
     */
    const KernelInfo* get_kernel(const std::string& name) const;

    /**
     * Check if kernel is loaded
     */
    bool has_kernel(const std::string& name) const;

    /**
     * Get all loaded kernel names
     */
    std::vector<std::string> get_kernel_names() const;

    /**
     * Select best kernel for given dimensions
     *
     * Tries to find exact match first, then falls back to chunking strategy.
     *
     * @param M, K, N Matrix dimensions
     * @return Name of best kernel to use
     * @throws std::runtime_error if no suitable kernel found
     */
    std::string select_kernel(size_t M, size_t K, size_t N) const;

private:
    PyObject* device_obj_;     // Reference to device object
    std::vector<KernelInfo> kernels_;

    // Python module for AIE utilities
    PyObject* aie_utils_module_;

    // Helper to initialize Python environment
    void init_python();
    void cleanup_python();

    // Helper to check if file exists
    bool file_exists(const std::string& path) const;
};

} // namespace whisper_xdna2
