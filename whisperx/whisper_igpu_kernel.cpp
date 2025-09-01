/*
 * Custom Intel iGPU Kernels for Whisper
 * Direct hardware execution using Level Zero and SYCL
 */

#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>

namespace whisper_igpu {

using namespace sycl;

class WhisperIGPU {
private:
    queue gpu_queue;
    
public:
    WhisperIGPU() {
        // Initialize SYCL queue for Intel GPU
        auto gpu_devices = device::get_devices(info::device_type::gpu);
        
        for (auto& dev : gpu_devices) {
            if (dev.get_info<info::device::vendor>().find("Intel") != std::string::npos) {
                gpu_queue = queue(dev);
                std::cout << "✅ Using Intel GPU: " 
                          << dev.get_info<info::device::name>() << std::endl;
                break;
            }
        }
    }
    
    // Custom kernel for mel spectrogram computation on iGPU
    void compute_mel_spectrogram(float* audio, float* mel_output, 
                                 int n_samples, int n_mel) {
        buffer<float, 1> audio_buf(audio, range<1>(n_samples));
        buffer<float, 2> mel_buf(mel_output, range<2>(n_mel, n_samples/512));
        
        gpu_queue.submit([&](handler& h) {
            auto audio_acc = audio_buf.get_access<access::mode::read>(h);
            auto mel_acc = mel_buf.get_access<access::mode::write>(h);
            
            h.parallel_for(range<2>(n_mel, n_samples/512), [=](id<2> idx) {
                int mel_idx = idx[0];
                int time_idx = idx[1];
                
                // FFT and mel filterbank on GPU
                float sum = 0.0f;
                int start = time_idx * 512;
                
                for (int i = 0; i < 512; i++) {
                    if (start + i < n_samples) {
                        // Simplified mel computation (use float literals)
                        float angle = 2.0f * 3.14159265f * i * mel_idx / 512.0f;
                        sum += audio_acc[start + i] * sycl::cos(angle);
                    }
                }
                
                mel_acc[idx] = sycl::log(sycl::max(sum * sum, 1e-10f));
            });
        }).wait();
    }
    
    // Custom kernel for attention mechanism on iGPU
    void compute_attention(float* query, float* key, float* value,
                          float* output, int seq_len, int d_model) {
        const int heads = 8;
        const int d_head = d_model / heads;
        
        buffer<float, 2> q_buf(query, range<2>(seq_len, d_model));
        buffer<float, 2> k_buf(key, range<2>(seq_len, d_model));
        buffer<float, 2> v_buf(value, range<2>(seq_len, d_model));
        buffer<float, 2> out_buf(output, range<2>(seq_len, d_model));
        
        gpu_queue.submit([&](handler& h) {
            auto q = q_buf.get_access<access::mode::read>(h);
            auto k = k_buf.get_access<access::mode::read>(h);
            auto v = v_buf.get_access<access::mode::read>(h);
            auto out = out_buf.get_access<access::mode::write>(h);
            
            // Local memory for attention scores
            local_accessor<float, 1> local_scores(range<1>(seq_len), h);
            
            h.parallel_for(nd_range<2>(range<2>(seq_len, heads),
                                       range<2>(1, heads)), 
                          [=](nd_item<2> item) {
                int pos = item.get_global_id(0);
                int head = item.get_global_id(1);
                
                // Compute attention scores for this position and head
                for (int i = 0; i < seq_len; i++) {
                    float score = 0.0f;
                    for (int j = 0; j < d_head; j++) {
                        int q_idx = head * d_head + j;
                        int k_idx = head * d_head + j;
                        score += q[pos][q_idx] * k[i][k_idx];
                    }
                    local_scores[i] = score / sycl::sqrt(float(d_head));
                }
                
                // Softmax
                float max_score = local_scores[0];
                for (int i = 1; i < seq_len; i++) {
                    max_score = sycl::max(max_score, local_scores[i]);
                }
                
                float sum = 0.0f;
                for (int i = 0; i < seq_len; i++) {
                    local_scores[i] = sycl::exp(local_scores[i] - max_score);
                    sum += local_scores[i];
                }
                
                for (int i = 0; i < seq_len; i++) {
                    local_scores[i] /= sum;
                }
                
                // Apply attention to values
                for (int j = 0; j < d_head; j++) {
                    float result = 0.0f;
                    for (int i = 0; i < seq_len; i++) {
                        result += local_scores[i] * v[i][head * d_head + j];
                    }
                    out[pos][head * d_head + j] = result;
                }
            });
        }).wait();
    }
    
    // Custom kernel for layer normalization on iGPU
    void layer_norm(float* input, float* output, int batch, int dim) {
        buffer<float, 2> in_buf(input, range<2>(batch, dim));
        buffer<float, 2> out_buf(output, range<2>(batch, dim));
        
        gpu_queue.submit([&](handler& h) {
            auto in = in_buf.get_access<access::mode::read>(h);
            auto out = out_buf.get_access<access::mode::write>(h);
            
            h.parallel_for(range<1>(batch), [=](id<1> idx) {
                int b = idx[0];
                
                // Compute mean
                float mean = 0.0f;
                for (int i = 0; i < dim; i++) {
                    mean += in[b][i];
                }
                mean /= dim;
                
                // Compute variance
                float var = 0.0f;
                for (int i = 0; i < dim; i++) {
                    float diff = in[b][i] - mean;
                    var += diff * diff;
                }
                var = sycl::sqrt(var / dim + 1e-5f);
                
                // Normalize
                for (int i = 0; i < dim; i++) {
                    out[b][i] = (in[b][i] - mean) / var;
                }
            });
        }).wait();
    }
    
    // Custom kernel for matrix multiplication on iGPU
    void matmul(float* a, float* b, float* c, int m, int n, int k) {
        buffer<float, 2> a_buf(a, range<2>(m, k));
        buffer<float, 2> b_buf(b, range<2>(k, n));
        buffer<float, 2> c_buf(c, range<2>(m, n));
        
        gpu_queue.submit([&](handler& h) {
            auto a_acc = a_buf.get_access<access::mode::read>(h);
            auto b_acc = b_buf.get_access<access::mode::read>(h);
            auto c_acc = c_buf.get_access<access::mode::write>(h);
            
            // Use local memory for tiling
            constexpr int tile_size = 16;
            local_accessor<float, 2> a_local(range<2>(tile_size, tile_size), h);
            local_accessor<float, 2> b_local(range<2>(tile_size, tile_size), h);
            
            h.parallel_for(nd_range<2>(range<2>(m, n), range<2>(tile_size, tile_size)),
                          [=](nd_item<2> item) {
                int global_row = item.get_global_id(0);
                int global_col = item.get_global_id(1);
                int local_row = item.get_local_id(0);
                int local_col = item.get_local_id(1);
                
                float sum = 0.0f;
                
                for (int tile = 0; tile < k; tile += tile_size) {
                    // Load tiles into local memory
                    if (tile + local_col < k && global_row < m) {
                        a_local[local_row][local_col] = a_acc[global_row][tile + local_col];
                    } else {
                        a_local[local_row][local_col] = 0.0f;
                    }
                    
                    if (tile + local_row < k && global_col < n) {
                        b_local[local_row][local_col] = b_acc[tile + local_row][global_col];
                    } else {
                        b_local[local_row][local_col] = 0.0f;
                    }
                    
                    item.barrier();
                    
                    // Compute partial sum
                    for (int i = 0; i < tile_size; i++) {
                        sum += a_local[local_row][i] * b_local[i][local_col];
                    }
                    
                    item.barrier();
                }
                
                if (global_row < m && global_col < n) {
                    c_acc[global_row][global_col] = sum;
                }
            });
        }).wait();
    }
    
    // Get GPU memory info
    void get_memory_info() {
        auto device = gpu_queue.get_device();
        std::cout << "GPU Global Memory: " 
                  << device.get_info<info::device::global_mem_size>() / (1024*1024) 
                  << " MB" << std::endl;
        std::cout << "GPU Local Memory: "
                  << device.get_info<info::device::local_mem_size>() / 1024
                  << " KB" << std::endl;
        std::cout << "Max Work Group Size: "
                  << device.get_info<info::device::max_work_group_size>() 
                  << std::endl;
    }
};

} // namespace whisper_igpu

// C API for Python bindings
extern "C" {
    void* create_whisper_igpu() {
        return new whisper_igpu::WhisperIGPU();
    }
    
    void destroy_whisper_igpu(void* instance) {
        delete static_cast<whisper_igpu::WhisperIGPU*>(instance);
    }
    
    void compute_mel_spectrogram(void* instance, float* audio, float* mel_output,
                                 int n_samples, int n_mel) {
        static_cast<whisper_igpu::WhisperIGPU*>(instance)->
            compute_mel_spectrogram(audio, mel_output, n_samples, n_mel);
    }
    
    void compute_attention(void* instance, float* query, float* key, float* value,
                          float* output, int seq_len, int d_model) {
        static_cast<whisper_igpu::WhisperIGPU*>(instance)->
            compute_attention(query, key, value, output, seq_len, d_model);
    }
    
    void matmul(void* instance, float* a, float* b, float* c, int m, int n, int k) {
        static_cast<whisper_igpu::WhisperIGPU*>(instance)->
            matmul(a, b, c, m, n, k);
    }
}