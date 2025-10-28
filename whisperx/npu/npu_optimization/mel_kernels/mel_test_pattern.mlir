//===- mel_test_pattern.mlir - Test pattern for simple kernel ---*- MLIR -*-===//
//
// Minimal test - empty core with elf_file, using working infrastructure
//
//===----------------------------------------------------------------------===//

module @mel_test {
    aie.device(npu1) {
        %tile00 = aie.tile(0, 0)
        %tile02 = aie.tile(0, 2)

        // Empty core with elf_file (pattern that we need to test)
        %core02 = aie.core(%tile02) {
            aie.end
        } { elf_file = "mel_kernel_simple.o" }

        // Shim DMA allocations
        aie.shim_dma_allocation @audioIn_shim (MM2S, 0, 0)
        aie.shim_dma_allocation @melOut_shim (S2MM, 0, 0)

        // Memory DMA infrastructure (simplified)
        %mem02 = aie.mem(%tile02) {
            %dma_in = aie.dma_start(S2MM, 0, ^bb_in, ^bb_out)
        ^bb_in:
            aie.end
        ^bb_out:
            aie.end
        }

        // Runtime sequence (800 bytes in, 80 bytes out)
        aiex.runtime_sequence(%in : memref<200xi32>, %arg1 : memref<1xi32>, %out : memref<20xi32>) {
            %c0 = arith.constant 0 : i64
            %c1 = arith.constant 1 : i64
            %c200 = arith.constant 200 : i64
            %c20 = arith.constant 20 : i64

            aiex.npu.dma_memcpy_nd (%in[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c200][%c0, %c0, %c0, %c1])
                { metadata = @audioIn_shim, id = 1 : i64 } : memref<200xi32>

            aiex.npu.dma_memcpy_nd (%out[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c20][%c0, %c0, %c0, %c1])
                { metadata = @melOut_shim, id = 0 : i64 } : memref<20xi32>

            aiex.npu.dma_wait {symbol = @melOut_shim}
        }
    }
}
