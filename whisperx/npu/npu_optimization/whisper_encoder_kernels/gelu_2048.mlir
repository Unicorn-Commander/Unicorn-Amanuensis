//===- gelu_2048.mlir -------------------------------------------*- MLIR -*-===//
//
// INT8 GELU Activation for Whisper FFN Intermediate Layer (2048 elements)
// Based on working attention and matmul patterns
//
// GELU(x) = x * Φ(x) using lookup table approach
// Ultra-fast: 1 cycle per element
//
// For Whisper base FFN:
//   Linear(512, 2048) -> GELU(2048) -> Linear(2048, 512)
//
// Input: [2048] int8 values
// Output: [2048] int8 values
//
//===----------------------------------------------------------------------===//

module @gelu_2048_npu {
    aie.device(npu1) {
        // Declare external GELU kernel function for 2048 elements
        func.func private @gelu_int8_2048(memref<2048xi8>, memref<2048xi8>, i32)

        // Declare tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile (DMA)
        %tile02 = aie.tile(0, 2)  // Compute tile

        // Input ObjectFIFO: 2048 int8 elements = 2048 bytes
        aie.objectfifo @of_input(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // Output ObjectFIFO: 2048 int8 elements = 2048 bytes
        aie.objectfifo @of_output(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // Core logic - infinite loop with event-driven execution
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index  // Infinite loop
            %c_N = arith.constant 2048 : i32  // Number of elements (FFN intermediate)

            // Infinite loop - core stays active
            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire input buffer
                %subviewIn = aie.objectfifo.acquire @of_input(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                // Acquire output buffer
                %subviewOut = aie.objectfifo.acquire @of_output(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                // Call GELU kernel: GELU(input) -> output
                // Processes all 2048 elements in FFN intermediate layer
                func.call @gelu_int8_2048(%elemIn, %elemOut, %c_N)
                    : (memref<2048xi8>, memref<2048xi8>, i32) -> ()

                // Release ObjectFIFOs
                aie.objectfifo.release @of_input(Consume, 1)
                aie.objectfifo.release @of_output(Produce, 1)
            }

            aie.end
        } {link_with="gelu_combined.o"}

        // Runtime sequence
        aiex.runtime_sequence(%input : memref<2048xi8>, %output : memref<2048xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c2048_i64 = arith.constant 2048 : i64

            // DMA transfer: Input buffer (host -> NPU)
            aiex.npu.dma_memcpy_nd(%input[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input,
                id = 1 : i64
            } : memref<2048xi8>

            // DMA transfer: Output buffer (NPU -> host)
            aiex.npu.dma_memcpy_nd(%output[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output,
                id = 0 : i64
            } : memref<2048xi8>

            // Wait for output DMA completion
            aiex.npu.dma_wait {symbol = @of_output}
        }
    }
}
