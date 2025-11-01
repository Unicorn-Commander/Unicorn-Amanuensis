#include "whisper_xdna2_runtime.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "TESTING: Runtime Initialization" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    try {
        // Test 1: Initialize runtime
        std::cout << "\nTest 1: Initialize runtime..." << std::endl;
        whisper_xdna2::WhisperXDNA2Runtime runtime("base", false);
        runtime.initialize();
        assert(runtime.is_initialized());
        std::cout << "[PASS] Runtime initialized successfully" << std::endl;
        
        // Test 2: Check initialization is idempotent
        std::cout << "\nTest 2: Check initialization is idempotent..." << std::endl;
        runtime.initialize();
        assert(runtime.is_initialized());
        std::cout << "[PASS] Multiple initialization calls handled correctly" << std::endl;
        
        // Test 3: Cleanup
        std::cout << "\nTest 3: Cleanup..." << std::endl;
        runtime.cleanup();
        std::cout << "[PASS] Cleanup successful" << std::endl;
        
        std::cout << "\n===========================================" << std::endl;
        std::cout << "ALL TESTS PASSED" << std::endl;
        std::cout << "===========================================" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Test failed: " << e.what() << std::endl;
        return 1;
    }
}
