// Phase 2.2: MLIR wrapper for mel_fft_basic kernel
// Computes real mel spectrograms on AMD Phoenix NPU
//
// Input:  Audio frames [num_frames, 400] INT16
// Output: Mel features [num_frames, 80] INT8

module {
  // External function declaration for the C kernel
  func.func private @mel_spectrogram_kernel(memref<?x400xi16>, memref<?x80xi8>, i32) -> ()

  aie.device(npu1) {
    // Define compute tile (using tile 0,2 as in Phase 2.1)
    %tile_0_2 = aie.tile(0, 2)

    // Buffers for mel spectrogram computation
    // Input: 400 samples per frame (Whisper uses 400-sample windows)
    // Output: 80 mel bins per frame
    %buff_audio = aie.buffer(%tile_0_2) {sym_name = "audio_buffer"} : memref<400xi16>
    %buff_mel = aie.buffer(%tile_0_2) {sym_name = "mel_buffer"} : memref<80xi8>

    // Core with mel_fft_basic.o ELF
    %core_0_2 = aie.core(%tile_0_2) {
      // Core body is empty when elf_file is specified
      // The ELF file contains the mel_spectrogram_kernel implementation
      aie.end
    } {elf_file = "mel_fft_basic.o"}
  }
}
