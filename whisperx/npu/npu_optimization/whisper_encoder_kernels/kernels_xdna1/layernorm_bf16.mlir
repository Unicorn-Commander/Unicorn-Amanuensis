//===- layernorm_bf16.mlir --------------------------------------*- MLIR -*-===//
//
// BF16 Layer Normalization for Whisper Encoder - XDNA1 Phoenix NPU
// Essential operation for transformer encoder/decoder layers
//
// LayerNorm(x) = (x - mean) / sqrt(var + eps)
//
// Input: [1024] bfloat16 values
// Output: [1024] bfloat16 values
//
// Buffer size: 1024 elements × 2 bytes/bfloat16 = 2048 bytes
//
//===----------------------------------------------------------------------===//

module @layernorm_bf16_npu {
    aie.device(npu1) {
        // Declare external LayerNorm kernel function (simplified version)
        func.func private @layernorm_simple_bf16(memref<2048xi8>, memref<2048xi8>)

        // Declare tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile (DMA)
        %tile02 = aie.tile(0, 2)  // Compute tile

        // Input ObjectFIFO: 1024 bfloat16 elements = 2048 bytes
        aie.objectfifo @of_input(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // Output ObjectFIFO: 1024 bfloat16 elements = 2048 bytes
        aie.objectfifo @of_output(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // Core logic - infinite loop with event-driven execution
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire input buffer
                %subviewIn = aie.objectfifo.acquire @of_input(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                // Acquire output buffer
                %subviewOut = aie.objectfifo.acquire @of_output(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                // Call LayerNorm kernel
                func.call @layernorm_simple_bf16(%elemIn, %elemOut)
                    : (memref<2048xi8>, memref<2048xi8>) -> ()

                // Release ObjectFIFOs
                aie.objectfifo.release @of_input(Consume, 1)
                aie.objectfifo.release @of_output(Produce, 1)
            }

            aie.end
        } {link_with="layernorm_bf16_xdna1.o"}

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
