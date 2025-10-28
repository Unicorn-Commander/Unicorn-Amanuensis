// Minimal MEL INT8 kernel without C++ ELF - for testing pipeline only
//
// This version has an empty core to test the XCLBIN generation pipeline
// Once this works, we'll add the actual C++ implementation

module {
  aie.device(npu1) {
    // Define compute tile
    %tile_0_2 = aie.tile(0, 2)

    // Buffers for mel spectrogram computation
    %buff_audio = aie.buffer(%tile_0_2) {sym_name = "audio_buffer_int8"} : memref<400xi16>
    %buff_mel = aie.buffer(%tile_0_2) {sym_name = "mel_buffer_int8"} : memref<80xi8>

    // Core with compiled ELF file
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {elf_file = "mel_kernel_empty.o"}
  }
}
