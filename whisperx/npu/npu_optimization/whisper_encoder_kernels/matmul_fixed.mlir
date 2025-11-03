//===- matmul_fixed.mlir ------------------------------------------------*- MLIR -*-===//
//
// INT8 Matrix Multiply for Whisper Encoder - FIXED VERSION
// Takes packed input buffer (A+B combined) to match Python test pattern
//
// Input: 512 bytes = Matrix A (256 bytes) + Matrix B (256 bytes)
// Output: 256 bytes = Matrix C (16x16 int8)
//
// This matches the pattern used by attention, layernorm, and GELU kernels
//
//===----------------------------------------------------------------------===//

module @matmul_npu {
    aie.device(npu1) {
        // Declare external matmul kernel function (compiled from matmul_int8.c)
        // Takes packed input: first 256 bytes = A, next 256 bytes = B
        func.func private @matmul_int8_16x16_packed(memref<512xi8>, memref<256xi8>)

        // Declare tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile (DMA)
        %tile02 = aie.tile(0, 2)  // Compute tile

        // Input ObjectFIFO: Packed A+B matrices (512 bytes)
        aie.objectfifo @of_input(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

        // Output ObjectFIFO: C matrix (16x16 int8 = 256 bytes)
        aie.objectfifo @of_output(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<256xi8>>

        // Core logic - infinite loop with event-driven execution
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index  // Infinite loop

            // Infinite loop - core stays active
            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire input buffer (packed A+B)
                %subviewInput = aie.objectfifo.acquire @of_input(Consume, 1) : !aie.objectfifosubview<memref<512xi8>>
                %elemInput = aie.objectfifo.subview.access %subviewInput[0] : !aie.objectfifosubview<memref<512xi8>> -> memref<512xi8>

                // Acquire output buffer
                %subviewOutput = aie.objectfifo.acquire @of_output(Produce, 1) : !aie.objectfifosubview<memref<256xi8>>
                %elemOutput = aie.objectfifo.subview.access %subviewOutput[0] : !aie.objectfifosubview<memref<256xi8>> -> memref<256xi8>

                // Call matmul kernel: unpacks A and B from input buffer internally
                func.call @matmul_int8_16x16_packed(%elemInput, %elemOutput) : (memref<512xi8>, memref<256xi8>) -> ()

                // Release ObjectFIFOs
                aie.objectfifo.release @of_input(Consume, 1)
                aie.objectfifo.release @of_output(Produce, 1)
            }

            aie.end
        } {link_with="matmul_fixed.o"}

        // Runtime sequence
        aiex.runtime_sequence(%input : memref<512xi8>, %output : memref<256xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c256_i64 = arith.constant 256 : i64
            %c512_i64 = arith.constant 512 : i64

            // DMA transfer: Packed input (host -> NPU)
            aiex.npu.dma_memcpy_nd(%input[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                        [%c1_i64, %c1_i64, %c1_i64, %c512_i64]
                                        [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input,
                id = 1 : i64
            } : memref<512xi8>

            // DMA transfer: Output (NPU -> host)
            aiex.npu.dma_memcpy_nd(%output[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                        [%c1_i64, %c1_i64, %c1_i64, %c256_i64]
                                        [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output,
                id = 0 : i64
            } : memref<256xi8>

            // Wait for output DMA completion
            aiex.npu.dma_wait {symbol = @of_output}
        }
    }
}
