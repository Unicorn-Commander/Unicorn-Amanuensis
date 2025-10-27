// NPU XCLBIN Loader - C++ XRT Direct Implementation
// Tests loading XCLBIN on AMD Phoenix NPU using native XRT API
//
// Compile:
//   g++ -o npu_loader npu_loader.cpp \
//       -I/opt/xilinx/xrt/include \
//       -L/opt/xilinx/xrt/lib \
//       -lxrt_coreutil \
//       -std=c++17
//
// Usage:
//   ./npu_loader passthrough_with_pdi.xclbin

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <exception>
#include <chrono>

class NPULoader {
private:
    xrt::device device;
    xrt::uuid uuid;
    std::string xclbin_path;

public:
    NPULoader(unsigned int device_index = 0) : device(device_index) {
        std::cout << "[*] Initialized XRT device " << device_index << std::endl;

        // Get device info
        auto device_name = device.get_info<xrt::info::device::name>();
        std::cout << "[*] Device name: " << device_name << std::endl;
    }

    bool load_xclbin(const std::string& xclbin_file) {
        xclbin_path = xclbin_file;

        try {
            std::cout << "[*] Loading XCLBIN: " << xclbin_file << std::endl;

            // Check if file exists
            std::ifstream file(xclbin_file);
            if (!file.good()) {
                std::cerr << "[!] ERROR: XCLBIN file not found: " << xclbin_file << std::endl;
                return false;
            }

            // Get file size
            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();
            std::cout << "[*] XCLBIN size: " << file_size << " bytes" << std::endl;
            file.close();

            // Load XCLBIN using XRT API
            auto start = std::chrono::high_resolution_clock::now();

            uuid = device.load_xclbin(xclbin_file);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::cout << "[✓] SUCCESS! XCLBIN loaded on NPU" << std::endl;
            std::cout << "[*] UUID: " << uuid.to_string() << std::endl;
            std::cout << "[*] Load time: " << duration.count() << " ms" << std::endl;

            return true;

        } catch (const std::exception& e) {
            std::cerr << "[!] ERROR loading XCLBIN: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_kernel_access() {
        try {
            std::cout << "[*] Testing kernel access..." << std::endl;

            // Try to access the DPU kernel
            xrt::kernel kernel(device, uuid, "DPU:passthrough");

            std::cout << "[✓] Kernel 'DPU:passthrough' accessible!" << std::endl;

            // Get kernel info
            auto kernel_name = kernel.get_name();
            std::cout << "[*] Kernel name: " << kernel_name << std::endl;

            return true;

        } catch (const std::exception& e) {
            std::cerr << "[!] Cannot access kernel: " << e.what() << std::endl;
            std::cerr << "[i] This is expected if kernel definition is incomplete" << std::endl;
            return false;
        }
    }

    void print_device_info() {
        std::cout << "\n[*] Device Information:" << std::endl;

        try {
            auto bdf = device.get_info<xrt::info::device::bdf>();
            std::cout << "    BDF: " << bdf << std::endl;
        } catch (...) {}

        try {
            auto max_clock = device.get_info<xrt::info::device::max_clock_frequency_mhz>();
            std::cout << "    Max clock: " << max_clock << " MHz" << std::endl;
        } catch (...) {}

        std::cout << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "==================================================" << std::endl;
    std::cout << "  NPU XCLBIN Loader - C++ XRT Implementation" << std::endl;
    std::cout << "  Magic Unicorn Unconventional Technology & Stuff" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << std::endl;

    // Default XCLBIN file
    std::string xclbin_file = "passthrough_with_pdi.xclbin";

    // Override with command-line argument if provided
    if (argc > 1) {
        xclbin_file = argv[1];
    }

    try {
        // Initialize NPU loader
        NPULoader loader(0);  // Device 0 = /dev/accel/accel0

        // Print device info
        loader.print_device_info();

        // Load XCLBIN
        if (!loader.load_xclbin(xclbin_file)) {
            std::cerr << "[!] XCLBIN load FAILED" << std::endl;
            return 1;
        }

        // Test kernel access (may fail if kernel incomplete, that's OK)
        loader.test_kernel_access();

        std::cout << "\n[✓] All tests complete!" << std::endl;
        std::cout << "[*] XCLBIN successfully loaded on AMD Phoenix NPU" << std::endl;
        std::cout << "\n==================================================" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[!] FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
}
