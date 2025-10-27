// Simple Mel Spectrogram Kernel for AIE2 - Phase 2.1
// Goal: Prove NPU can process audio and produce magnitude spectrum output
//
// Phase 2.1 Scope (Week 1):
// - Basic FFT implementation (no mel filtering yet)
// - No INT8 quantization (use FP16/FP32)
// - Process one frame at a time
// - Output: magnitude spectrum (not yet mel-filtered)

#include <stdint.h>

// Manual memory operations for AIE2 (no standard library)
static inline void zero_memory(void* ptr, uint32_t size) {
    uint8_t* p = (uint8_t*)ptr;
    for (uint32_t i = 0; i < size; i++) {
        p[i] = 0;
    }
}

static inline void copy_memory(void* dest, const void* src, uint32_t size) {
    uint8_t* d = (uint8_t*)dest;
    const uint8_t* s = (const uint8_t*)src;
    for (uint32_t i = 0; i < size; i++) {
        d[i] = s[i];
    }
}

// Window size for Whisper: 25ms at 16kHz = 400 samples
// FFT size: 512 (zero-padded)
#define WINDOW_SIZE 400
#define FFT_SIZE 512
#define HOP_SIZE 160  // 10ms hop

// Complex number structure (from fft_radix2.c)
typedef struct {
    float real;
    float imag;
} complex_t;

// External FFT function (from fft_radix2.c)
extern void fft_radix2_512(int16_t* input, complex_t* output);
extern void compute_magnitude(complex_t* fft_output, int32_t* magnitude, uint32_t size);

// Hann window coefficients (precomputed for 400 samples)
// Hann(n) = 0.5 * (1 - cos(2*π*n/(N-1)))
// For now, we'll use a simple stub - real implementation would have all 400 values
static const float hann_window[WINDOW_SIZE] = {
    // First few values as placeholder
    0.0000f, 0.0001f, 0.0003f, 0.0006f, 0.0010f,
    // ... rest would be precomputed
    // This is just a placeholder for Phase 2.1
};

// Apply Hann window to audio frame
void apply_hann_window(int16_t* input, int16_t* output, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        // Apply window: output = input * window
        // Scale down to prevent overflow
        float windowed = (float)input[i] * hann_window[i];
        output[i] = (int16_t)windowed;
    }
}

// Main kernel entry point
// This function will be called by the AIE core
// Input: raw audio samples (int16, 16kHz)
// Output: magnitude spectrum (int32, 256 frequency bins)
void mel_simple_kernel(int16_t* restrict audio_in,
                      int32_t* restrict spectrum_out,
                      uint32_t num_frames) {

    // Local buffers in AIE tile memory (64KB available)
    int16_t windowed_frame[FFT_SIZE];
    complex_t fft_output[FFT_SIZE];
    int32_t magnitude[FFT_SIZE / 2];  // Only need first half (Nyquist)

    // Process each frame
    for (uint32_t frame = 0; frame < num_frames; frame++) {
        // Calculate input offset (hop size between frames)
        int16_t* frame_input = audio_in + (frame * HOP_SIZE);

        // Step 1: Apply Hann window to reduce spectral leakage
        apply_hann_window(frame_input, windowed_frame, WINDOW_SIZE);

        // Step 2: Zero-pad to FFT size (400 → 512)
        zero_memory(windowed_frame + WINDOW_SIZE,
                   (FFT_SIZE - WINDOW_SIZE) * sizeof(int16_t));

        // Step 3: Compute 512-point FFT
        fft_radix2_512(windowed_frame, fft_output);

        // Step 4: Compute magnitude spectrum
        // Only first 256 bins are unique (rest are conjugate symmetric)
        compute_magnitude(fft_output, magnitude, FFT_SIZE / 2);

        // Step 5: Write output for this frame
        // Output offset: each frame produces 256 magnitude values
        int32_t* frame_output = spectrum_out + (frame * (FFT_SIZE / 2));
        copy_memory(frame_output, magnitude, (FFT_SIZE / 2) * sizeof(int32_t));
    }
}

// AIE core main entry point
// This is called when the core starts execution
int main() {
    // In a real implementation, this would be an infinite loop
    // waiting for DMA signals and processing data
    //
    // For Phase 2.1, the MLIR DMA configuration handles:
    // - Moving audio data from host to AIE tile
    // - Calling mel_simple_kernel()
    // - Moving results back to host
    //
    // The core just needs to return successfully

    return 0;
}
