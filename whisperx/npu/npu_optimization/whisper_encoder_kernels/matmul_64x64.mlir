//===- matmul_64x64.mlir ------------------------------------------------*- MLIR -*-===//
//
// INT8 64x64 Matrix Multiply for Whisper Encoder
// Maximum practical tile size for AIE2 (88% of 32KB memory)
//
// Input: 8192 bytes = Matrix A (4096 bytes) + Matrix B (4096 bytes)
// Output: 4096 bytes = Matrix C (64x64 int8)
//
// Expected performance improvement over smaller tiles:
// - 16x fewer kernel invocations vs 16x16
// - Higher latency per operation (~0.8-1.0ms) due to memory pressure
// - 6-8x overall speedup for large matrices (best case)
//
//===----------------------------------------------------------------------===//

module @matmul_npu_64x64 {
    aie.device(npu1) {
        // Declare external matmul kernel function (compiled from matmul_int8_64x64.c)
        // Takes packed input: first 4096 bytes = A, next 4096 bytes = B
        func.func private @matmul_int8_64x64_packed(memref<8192xi8>, memref<4096xi8>)

        // Declare tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile (DMA)
        %tile02 = aie.tile(0, 2)  // Compute tile

        // Input ObjectFIFO: Packed A+B matrices (8192 bytes)
        aie.objectfifo @of_input(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>

        // Output ObjectFIFO: C matrix (64x64 int8 = 4096 bytes)
        aie.objectfifo @of_output(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>

        // Core logic - infinite loop with event-driven execution
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index  // Infinite loop

            // Infinite loop - core stays active
            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire input buffer (packed A+B)
                %subviewInput = aie.objectfifo.acquire @of_input(Consume, 1) : !aie.objectfifosubview<memref<8192xi8>>
                %elemInput = aie.objectfifo.subview.access %subviewInput[0] : !aie.objectfifosubview<memref<8192xi8>> -> memref<8192xi8>

                // Acquire output buffer
                %subviewOutput = aie.objectfifo.acquire @of_output(Produce, 1) : !aie.objectfifosubview<memref<4096xi8>>
                %elemOutput = aie.objectfifo.subview.access %subviewOutput[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>

                // Call matmul kernel: unpacks A and B from input buffer internally
                func.call @matmul_int8_64x64_packed(%elemInput, %elemOutput) : (memref<8192xi8>, memref<4096xi8>) -> ()

                // Release ObjectFIFOs
                aie.objectfifo.release @of_input(Consume, 1)
                aie.objectfifo.release @of_output(Produce, 1)
            }

            aie.end
        } {link_with="matmul_64x64.o"}

        // Runtime sequence
        aiex.runtime_sequence(%input : memref<8192xi8>, %output : memref<4096xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c4096_i64 = arith.constant 4096 : i64
            %c8192_i64 = arith.constant 8192 : i64

            // DMA transfer: Packed input (host -> NPU)
            aiex.npu.dma_memcpy_nd(%input[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                        [%c1_i64, %c1_i64, %c1_i64, %c8192_i64]
                                        [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input,
                id = 1 : i64
            } : memref<8192xi8>

            // DMA transfer: Output (NPU -> host)
            aiex.npu.dma_memcpy_nd(%output[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                        [%c1_i64, %c1_i64, %c1_i64, %c4096_i64]
                                        [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output,
                id = 0 : i64
            } : memref<4096xi8>

            // Wait for output DMA completion
            aiex.npu.dma_wait {symbol = @of_output}
        }
    }
}
