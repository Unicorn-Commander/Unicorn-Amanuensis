//===- gelu_simple.mlir -----------------------------------------*- MLIR -*-===//
//
// INT8 GELU Activation for Whisper Encoder
// Based on working attention and matmul patterns
//
// GELU(x) = x * Φ(x) using lookup table approach
// Ultra-fast: 1 cycle per element
//
// Input: [512] int8 values (typical Whisper hidden dim)
// Output: [512] int8 values
//
// For Whisper FFN layer: Can also handle 2048 elements
//
//===----------------------------------------------------------------------===//

module @gelu_npu {
    aie.device(npu1) {
        // Declare external GELU kernel function
        // Takes 512-element INT8 buffer, applies GELU via LUT
        func.func private @gelu_int8_512(memref<512xi8>, memref<512xi8>, i32)

        // Declare tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile (DMA)
        %tile02 = aie.tile(0, 2)  // Compute tile

        // Input ObjectFIFO: 512 int8 elements = 512 bytes
        aie.objectfifo @of_input(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

        // Output ObjectFIFO: 512 int8 elements = 512 bytes
        aie.objectfifo @of_output(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

        // Core logic - infinite loop with event-driven execution
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index  // Infinite loop
            %c_N = arith.constant 512 : i32  // Number of elements

            // Infinite loop - core stays active
            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire input buffer
                %subviewIn = aie.objectfifo.acquire @of_input(Consume, 1) : !aie.objectfifosubview<memref<512xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<512xi8>> -> memref<512xi8>

                // Acquire output buffer
                %subviewOut = aie.objectfifo.acquire @of_output(Produce, 1) : !aie.objectfifosubview<memref<512xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<512xi8>> -> memref<512xi8>

                // Call GELU kernel: GELU(input) -> output
                // Ultra-fast LUT-based implementation
                func.call @gelu_int8_512(%elemIn, %elemOut, %c_N)
                    : (memref<512xi8>, memref<512xi8>, i32) -> ()

                // Release ObjectFIFOs
                aie.objectfifo.release @of_input(Consume, 1)
                aie.objectfifo.release @of_output(Produce, 1)
            }

            aie.end
        } {link_with="gelu_combined.o"}

        // Runtime sequence
        aiex.runtime_sequence(%input : memref<512xi8>, %output : memref<512xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c512_i64 = arith.constant 512 : i64

            // DMA transfer: Input buffer (host -> NPU)
            aiex.npu.dma_memcpy_nd(%input[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c512_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input,
                id = 1 : i64
            } : memref<512xi8>

            // DMA transfer: Output buffer (NPU -> host)
            aiex.npu.dma_memcpy_nd(%output[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c512_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output,
                id = 0 : i64
            } : memref<512xi8>

            // Wait for output DMA completion
            aiex.npu.dma_wait {symbol = @of_output}
        }
    }
}
