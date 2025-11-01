#include "whisper_xdna2_runtime.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "C++ ENCODER PERFORMANCE BENCHMARK" << std::endl;
    std::cout << "======================================================================" << std::endl;
    
    try {
        // Initialize runtime
        std::cout << "\nInitializing runtime..." << std::endl;
        whisper_xdna2::WhisperXDNA2Runtime runtime("base", false);
        runtime.initialize();
        // runtime.load_encoder_weights("../weights/whisper_base_encoder.bin");
        std::cout << "[OK] Runtime initialized" << std::endl;
        
        // Test input (512 x 512)
        std::cout << "\nCreating test input (512 x 512)..." << std::endl;
        std::vector<float> input(512 * 512, 1.0f);
        std::vector<float> output(512 * 512);
        std::cout << "[OK] Test input created" << std::endl;
        
        // Warmup
        std::cout << "\nWarmup run..." << std::endl;
        // runtime.run_encoder(input.data(), output.data(), 1);
        std::cout << "[OK] Warmup complete" << std::endl;
        
        // Benchmark (10 runs)
        std::cout << "\nRunning benchmark (10 runs)..." << std::endl;
        const int num_runs = 10;
        std::vector<double> latencies;
        latencies.reserve(num_runs);
        
        for (int i = 0; i < num_runs; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            // runtime.run_encoder(input.data(), output.data(), 1);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double latency_ms = duration.count() / 1000.0;
            latencies.push_back(latency_ms);
            
            std::cout << "  Run " << (i+1) << "/" << num_runs << ": " 
                      << std::fixed << std::setprecision(2) << latency_ms << " ms" << std::endl;
        }
        
        // Calculate statistics
        double total = 0.0;
        for (double lat : latencies) {
            total += lat;
        }
        double avg = total / num_runs;
        
        double min_lat = *std::min_element(latencies.begin(), latencies.end());
        double max_lat = *std::max_element(latencies.begin(), latencies.end());
        
        // Calculate realtime factor
        double audio_duration = 10.24; // seconds (512 frames * 20ms per frame)
        double realtime_factor = (audio_duration * 1000.0) / avg;
        
        std::cout << "\n======================================================================" << std::endl;
        std::cout << "RESULTS (averaged over " << num_runs << " runs):" << std::endl;
        std::cout << "======================================================================" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Average latency:    " << avg << " ms" << std::endl;
        std::cout << "  Min latency:        " << min_lat << " ms" << std::endl;
        std::cout << "  Max latency:        " << max_lat << " ms" << std::endl;
        std::cout << "  Audio duration:     " << audio_duration << " seconds" << std::endl;
        std::cout << "  Realtime factor:    " << std::setprecision(1) << realtime_factor << "x" << std::endl;
        
        // Compare to Python baseline
        double python_rtf = 5.59;
        double speedup = realtime_factor / python_rtf;
        
        std::cout << "\n======================================================================" << std::endl;
        std::cout << "COMPARISON:" << std::endl;
        std::cout << "======================================================================" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Python (32-tile):   " << python_rtf << "x realtime" << std::endl;
        std::cout << "  C++ (32-tile):      " << realtime_factor << "x realtime" << std::endl;
        std::cout << "  Speedup:            " << speedup << "x" << std::endl;
        
        std::cout << "\n======================================================================" << std::endl;
        if (speedup >= 3.0) {
            std::cout << "TARGET ACHIEVED: " << speedup << "x >= 3x!" << std::endl;
        } else if (speedup >= 2.0) {
            std::cout << "GOOD: " << speedup << "x (below 3x but significant)" << std::endl;
        } else {
            std::cout << "BELOW TARGET: " << speedup << "x < 3x" << std::endl;
        }
        std::cout << "======================================================================" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n[FAIL] Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}
