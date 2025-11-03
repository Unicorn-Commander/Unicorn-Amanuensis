//===- matmul_32x32.mlir ------------------------------------------------*- MLIR -*-===//
//
// INT8 32x32 Matrix Multiply for Whisper Encoder
// Scaled up from 16x16 for better performance (fewer kernel invocations)
//
// Input: 2048 bytes = Matrix A (1024 bytes) + Matrix B (1024 bytes)
// Output: 1024 bytes = Matrix C (32x32 int8)
//
// Expected performance improvement over 16x16:
// - 4x fewer kernel invocations for same total size
// - Similar latency per operation (~0.5ms)
// - 3-4x overall speedup
//
//===----------------------------------------------------------------------===//

module @matmul_npu_32x32 {
    aie.device(npu1) {
        // Declare external matmul kernel function (compiled from matmul_int8_32x32.c)
        // Takes packed input: first 1024 bytes = A, next 1024 bytes = B
        func.func private @matmul_int8_32x32_packed(memref<2048xi8>, memref<1024xi8>)

        // Declare tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile (DMA)
        %tile02 = aie.tile(0, 2)  // Compute tile

        // Input ObjectFIFO: Packed A+B matrices (2048 bytes)
        aie.objectfifo @of_input(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // Output ObjectFIFO: C matrix (32x32 int8 = 1024 bytes)
        aie.objectfifo @of_output(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>

        // Core logic - infinite loop with event-driven execution
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index  // Infinite loop

            // Infinite loop - core stays active
            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire input buffer (packed A+B)
                %subviewInput = aie.objectfifo.acquire @of_input(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemInput = aie.objectfifo.subview.access %subviewInput[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                // Acquire output buffer
                %subviewOutput = aie.objectfifo.acquire @of_output(Produce, 1) : !aie.objectfifosubview<memref<1024xi8>>
                %elemOutput = aie.objectfifo.subview.access %subviewOutput[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>

                // Call matmul kernel: unpacks A and B from input buffer internally
                func.call @matmul_int8_32x32_packed(%elemInput, %elemOutput) : (memref<2048xi8>, memref<1024xi8>) -> ()

                // Release ObjectFIFOs
                aie.objectfifo.release @of_input(Consume, 1)
                aie.objectfifo.release @of_output(Produce, 1)
            }

            aie.end
        } {link_with="matmul_32x32.o"}

        // Runtime sequence
        aiex.runtime_sequence(%input : memref<2048xi8>, %output : memref<1024xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c1024_i64 = arith.constant 1024 : i64
            %c2048_i64 = arith.constant 2048 : i64

            // DMA transfer: Packed input (host -> NPU)
            aiex.npu.dma_memcpy_nd(%input[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                        [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                        [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input,
                id = 1 : i64
            } : memref<2048xi8>

            // DMA transfer: Output (NPU -> host)
            aiex.npu.dma_memcpy_nd(%output[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                        [%c1_i64, %c1_i64, %c1_i64, %c1024_i64]
                                        [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output,
                id = 0 : i64
            } : memref<1024xi8>

            // Wait for output DMA completion
            aiex.npu.dma_wait {symbol = @of_output}
        }
    }
}
