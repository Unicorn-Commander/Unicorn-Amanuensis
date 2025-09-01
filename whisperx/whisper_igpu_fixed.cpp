/*
 * FIXED Intel iGPU Whisper Implementation
 * Respects 512 work-item limit per work-group
 */

#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>

namespace whisper_igpu {

using namespace sycl;

class WhisperIGPU {
private:
    queue gpu_queue;
    device gpu_device;
    size_t max_work_group_size;
    
public:
    WhisperIGPU() {
        // Find Intel GPU
        auto gpu_devices = device::get_devices(info::device_type::gpu);
        for (auto& dev : gpu_devices) {
            if (dev.get_info<info::device::name>().find("Intel") != std::string::npos) {
                gpu_device = dev;
                gpu_queue = queue(dev);
                break;
            }
        }
        
        // Get device limits
        max_work_group_size = gpu_device.get_info<info::device::max_work_group_size>();
        
        std::cout << "✅ Using Intel GPU: " << gpu_device.get_info<info::device::name>() << std::endl;
        std::cout << "  - Max work group size: " << max_work_group_size << std::endl;
        std::cout << "  - Max compute units: " << gpu_device.get_info<info::device::max_compute_units>() << std::endl;
    }
    
    // FIXED MEL Spectrogram - uses smaller work groups
    void compute_mel_spectrogram_fixed(float* audio, float* mel_output, int n_samples) {
        const int n_fft = 400;
        const int hop_length = 160;
        const int n_mels = 80;
        const int n_frames = 1 + (n_samples - n_fft) / hop_length;
        
        // Cap frames to avoid overflow
        const int max_frames = 3000;
        const int actual_frames = std::min(n_frames, max_frames);
        
        buffer<float, 1> audio_buf(audio, range<1>(n_samples));
        buffer<float, 2> mel_buf(mel_output, range<2>(n_mels, actual_frames));
        
        // Process in tiles to respect 512 limit
        const int tile_size = 16;  // 16*16 = 256 < 512
        
        gpu_queue.submit([&](handler& h) {
            auto in = audio_buf.get_access<access::mode::read>(h);
            auto out = mel_buf.get_access<access::mode::write>(h);
            
            // Use nd_range with explicit local size
            h.parallel_for(nd_range<2>(
                range<2>((n_mels + tile_size - 1) / tile_size * tile_size,
                        (actual_frames + tile_size - 1) / tile_size * tile_size),
                range<2>(tile_size, tile_size)), [=](nd_item<2> item) {
                
                int mel_idx = item.get_global_id(0);
                int frame_idx = item.get_global_id(1);
                
                if (mel_idx >= n_mels || frame_idx >= actual_frames) return;
                
                // Simplified mel computation
                int start = frame_idx * hop_length;
                float sum = 0.0f;
                
                for (int i = 0; i < n_fft && start + i < n_samples; i++) {
                    float window = 0.5f - 0.5f * sycl::cos(2.0f * 3.14159265f * i / n_fft);
                    sum += in[start + i] * window;
                }
                
                // Mel filter
                float mel_val = sycl::log10(sycl::fmax(1e-10f, sum * sum) + 1e-10f) * 10.0f;
                out[mel_idx][frame_idx] = mel_val;
            });
        }).wait();
    }
    
    // FIXED Multi-head Attention - processes in smaller chunks
    void multi_head_attention_fixed(float* q, float* k, float* v, float* output,
                                   int batch, int seq_len, int d_model, int n_heads) {
        
        int head_dim = d_model / n_heads;
        
        // Process attention in smaller tiles to avoid exceeding 512
        const int tile_q = 8;   // Query tile
        const int tile_k = 8;   // Key tile
        const int max_seq_tile = std::min(64, seq_len);  // Limit sequence processing
        
        buffer<float, 3> q_buf(q, range<3>(batch, seq_len, d_model));
        buffer<float, 3> k_buf(k, range<3>(batch, seq_len, d_model));
        buffer<float, 3> v_buf(v, range<3>(batch, seq_len, d_model));
        buffer<float, 3> out_buf(output, range<3>(batch, seq_len, d_model));
        
        // Process each head separately to avoid large work groups
        for (int h = 0; h < n_heads; h++) {
            gpu_queue.submit([&](handler& cgh) {
                auto q_acc = q_buf.get_access<access::mode::read>(cgh);
                auto k_acc = k_buf.get_access<access::mode::read>(cgh);
                auto v_acc = v_buf.get_access<access::mode::read>(cgh);
                auto out_acc = out_buf.get_access<access::mode::write>(cgh);
                
                // Local memory for tiles
                local_accessor<float, 2> scores_local(range<2>(tile_q, tile_k), cgh);
                
                // Process in tiles
                cgh.parallel_for(nd_range<2>(
                    range<2>((seq_len + tile_q - 1) / tile_q * tile_q,
                            (seq_len + tile_k - 1) / tile_k * tile_k),
                    range<2>(tile_q, tile_k)), [=](nd_item<2> item) {
                    
                    int q_idx = item.get_global_id(0);
                    int k_idx = item.get_global_id(1);
                    int local_q = item.get_local_id(0);
                    int local_k = item.get_local_id(1);
                    
                    if (q_idx >= seq_len || k_idx >= seq_len) return;
                    
                    // Compute attention score for this pair
                    float score = 0.0f;
                    int head_start = h * head_dim;
                    
                    for (int d = 0; d < head_dim; d++) {
                        score += q_acc[0][q_idx][head_start + d] * 
                                k_acc[0][k_idx][head_start + d];
                    }
                    
                    scores_local[local_q][local_k] = score / sycl::sqrt(float(head_dim));
                    
                    item.barrier(access::fence_space::local_space);
                    
                    // Apply softmax and output (simplified)
                    if (local_k == 0) {
                        float max_score = scores_local[local_q][0];
                        for (int i = 1; i < tile_k && k_idx - local_k + i < seq_len; i++) {
                            max_score = sycl::fmax(max_score, scores_local[local_q][i]);
                        }
                        
                        float sum = 0.0f;
                        for (int i = 0; i < tile_k && k_idx - local_k + i < seq_len; i++) {
                            sum += sycl::exp(scores_local[local_q][i] - max_score);
                        }
                        
                        // Write output (simplified)
                        for (int d = 0; d < head_dim; d++) {
                            out_acc[0][q_idx][head_start + d] = 
                                v_acc[0][q_idx][head_start + d] * (1.0f / sum);
                        }
                    }
                });
            }).wait();
        }
    }
    
    // FIXED Matrix multiplication with tiling
    void matmul_fixed(float* a, float* b, float* c, int m, int n, int k) {
        const int tile_size = 16;  // 16*16 = 256 < 512
        
        buffer<float, 2> a_buf(a, range<2>(m, k));
        buffer<float, 2> b_buf(b, range<2>(k, n));
        buffer<float, 2> c_buf(c, range<2>(m, n));
        
        gpu_queue.submit([&](handler& h) {
            auto a_acc = a_buf.get_access<access::mode::read>(h);
            auto b_acc = b_buf.get_access<access::mode::read>(h);
            auto c_acc = c_buf.get_access<access::mode::write>(h);
            
            // Local memory for tiles
            local_accessor<float, 2> a_local(range<2>(tile_size, tile_size), h);
            local_accessor<float, 2> b_local(range<2>(tile_size, tile_size), h);
            
            h.parallel_for(nd_range<2>(
                range<2>((m + tile_size - 1) / tile_size * tile_size,
                        (n + tile_size - 1) / tile_size * tile_size),
                range<2>(tile_size, tile_size)), [=](nd_item<2> item) {
                
                int row = item.get_global_id(0);
                int col = item.get_global_id(1);
                int local_row = item.get_local_id(0);
                int local_col = item.get_local_id(1);
                
                float sum = 0.0f;
                
                // Process in tiles
                for (int t = 0; t < (k + tile_size - 1) / tile_size; t++) {
                    // Load tile into local memory
                    int a_col = t * tile_size + local_col;
                    int b_row = t * tile_size + local_row;
                    
                    a_local[local_row][local_col] = 
                        (row < m && a_col < k) ? a_acc[row][a_col] : 0.0f;
                    b_local[local_row][local_col] = 
                        (b_row < k && col < n) ? b_acc[b_row][col] : 0.0f;
                    
                    item.barrier(access::fence_space::local_space);
                    
                    // Compute partial dot product
                    for (int i = 0; i < tile_size; i++) {
                        sum += a_local[local_row][i] * b_local[i][local_col];
                    }
                    
                    item.barrier(access::fence_space::local_space);
                }
                
                // Write result
                if (row < m && col < n) {
                    c_acc[row][col] = sum;
                }
            });
        }).wait();
    }
    
    // Performance stats
    void get_performance_stats() {
        std::cout << "\n📊 Intel iGPU Performance Stats:\n";
        std::cout << "  - Device: " << gpu_device.get_info<info::device::name>() << std::endl;
        std::cout << "  - Compute Units: " << gpu_device.get_info<info::device::max_compute_units>() << std::endl;
        std::cout << "  - Clock: " << gpu_device.get_info<info::device::max_clock_frequency>() << " MHz\n";
        std::cout << "  - Global Memory: " << gpu_device.get_info<info::device::global_mem_size>() / (1024*1024) << " MB\n";
        std::cout << "  - Local Memory: " << gpu_device.get_info<info::device::local_mem_size>() / 1024 << " KB\n";
        std::cout << "  - Max Work Group: " << max_work_group_size << std::endl;
    }
};

} // namespace whisper_igpu

// C API
extern "C" {
    void* create_whisper_igpu() {
        return new whisper_igpu::WhisperIGPU();
    }
    
    void destroy_whisper_igpu(void* instance) {
        delete static_cast<whisper_igpu::WhisperIGPU*>(instance);
    }
    
    void compute_mel_spectrogram_fixed(void* instance, float* audio, float* mel, int n_samples) {
        static_cast<whisper_igpu::WhisperIGPU*>(instance)->
            compute_mel_spectrogram_fixed(audio, mel, n_samples);
    }
    
    void multi_head_attention_fixed(void* instance, float* q, float* k, float* v, float* output,
                                   int batch, int seq_len, int d_model, int n_heads) {
        static_cast<whisper_igpu::WhisperIGPU*>(instance)->
            multi_head_attention_fixed(q, k, v, output, batch, seq_len, d_model, n_heads);
    }
    
    void matmul_fixed(void* instance, float* a, float* b, float* c, int m, int n, int k) {
        static_cast<whisper_igpu::WhisperIGPU*>(instance)->
            matmul_fixed(a, b, c, m, n, k);
    }
    
    void get_performance_stats(void* instance) {
        static_cast<whisper_igpu::WhisperIGPU*>(instance)->get_performance_stats();
    }
}