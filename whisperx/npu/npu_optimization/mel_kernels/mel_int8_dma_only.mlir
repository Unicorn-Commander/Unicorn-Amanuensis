//===- mel_int8_dma_only.mlir ----------------------------------*- MLIR -*-===//
//
// Minimal MEL INT8 kernel with ONLY runtime DMA sequences (no ObjectFIFO)
// This allows the core body to be empty as required for elf_file attribute
//
//===----------------------------------------------------------------------===//

module @mel_int8_npu {
    aie.device(npu1) {
        // Declare tiles
        %tile00 = aie.tile(0, 0)
        %tile02 = aie.tile(0, 2)

        // Empty core with ELF file reference
        %core02 = aie.core(%tile02) {
            aie.end
        } { elf_file = "mel_kernel_empty.o" }

        // Shim DMA allocations for runtime sequence
        aie.shim_dma_allocation @audioIn_shim (MM2S, 0, 0)
        aie.shim_dma_allocation @melOut_shim (S2MM, 0, 0)

        // Runtime sequence for host-NPU data movement
        // Input: 400 INT16 samples = 800 bytes = 200 32-bit words
        // Output: 80 INT8 values = 80 bytes = 20 32-bit words
        aiex.runtime_sequence(%in : memref<200xi32>, %arg1 : memref<1xi32>, %out : memref<20xi32>) {
            %c0 = arith.constant 0 : i64
            %c1 = arith.constant 1 : i64
            %c200 = arith.constant 200 : i64  // Input size in 32-bit words (800 bytes / 4)
            %c20 = arith.constant 20 : i64    // Output size in 32-bit words (80 bytes / 4)

            // DMA memcpy for input audio data (MM2S - host to NPU)
            aiex.npu.dma_memcpy_nd (%in[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c200][%c0, %c0, %c0, %c1]) { metadata = @audioIn_shim, id = 1 : i64 } : memref<200xi32>

            // DMA memcpy for output mel features (S2MM - NPU to host)
            aiex.npu.dma_memcpy_nd (%out[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c20][%c0, %c0, %c0, %c1]) { metadata = @melOut_shim, id = 0 : i64 } : memref<20xi32>

            // Synchronization - wait for DMA completion
            aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
        }
    }
}
