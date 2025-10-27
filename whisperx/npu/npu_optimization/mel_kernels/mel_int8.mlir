// Phase 2.3: MLIR wrapper for INT8-optimized mel spectrogram kernel
// Uses AIE2 SIMD vectorization for maximum performance
//
// Input:  Audio frames [num_frames, 400] INT16
// Output: Mel features [num_frames, 80] INT8 (Q7 format)

module {
  // External function declaration for the INT8 optimized kernel
  func.func private @mel_spectrogram_int8_kernel(memref<?x400xi16>, memref<?x80xi8>, i32) -> ()

  aie.device(npu1) {
    // Define compute tile (using tile 0,2 as in previous phases)
    %tile_0_2 = aie.tile(0, 2)

    // Buffers for INT8 mel spectrogram computation
    // Input: 400 samples per frame (INT16)
    // Output: 80 mel bins per frame (INT8 Q7)
    %buff_audio = aie.buffer(%tile_0_2) {sym_name = "audio_buffer_int8"} : memref<400xi16>
    %buff_mel = aie.buffer(%tile_0_2) {sym_name = "mel_buffer_int8"} : memref<80xi8>

    // Core with mel_int8_optimized.o ELF
    // This kernel includes:
    // - Audio quantization (INT16 → INT8)
    // - Hann window (Q7 × Q7)
    // - 512-point FFT with block scaling
    // - Magnitude spectrum with log LUT
    // - Mel filterbank (80 bins, vectorized)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {elf_file = "mel_int8_optimized.o"}
  }
}
