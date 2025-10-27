// MLIR-AIE2 Minimal Wrapper for Phase 2.1 Proof-of-Concept
// Simplified version to get first XCLBIN compiling
// Target: AMD Phoenix NPU (npu1, 4×6 tile array)

module {
  // Declare external C kernel function
  func.func private @mel_simple_kernel(memref<512xi16>, memref<256xi32>, i32) -> ()

  aie.device(npu1) {
    // Tile definitions
    %tile_0_2 = aie.tile(0, 2)  // Compute tile

    // Buffers in tile local memory (64KB available)
    %buff_in = aie.buffer(%tile_0_2) {sym_name = "input_buffer"} : memref<512xi16>
    %buff_out = aie.buffer(%tile_0_2) {sym_name = "output_buffer"} : memref<256xi32>

    // Locks for synchronization
    %lock_in = aie.lock(%tile_0_2, 0) {init = 1 : i32}
    %lock_out = aie.lock(%tile_0_2, 1) {init = 0 : i32}

    // Core definition - references the compiled ELF file
    // When elf_file is specified, core body must only contain aie.end
    // The kernel execution logic is in the compiled C code
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {elf_file = "mel_simple.o"}
  }
}
