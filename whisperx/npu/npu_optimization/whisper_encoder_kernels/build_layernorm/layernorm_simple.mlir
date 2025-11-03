//===- layernorm_simple.mlir ----------------------------------------*- MLIR -*-===//
//
// INT8 Layer Normalization for Whisper Encoder - 256-dimensional vectors
// Based on working attention_simple.mlir and matmul_simple.mlir patterns
//
// Computes: LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
//
// Input: Combined buffer [input + gamma + beta] = 768 bytes (256*3)
//        - Bytes 0-255:   Input features (256 int8)
//        - Bytes 256-511: Gamma parameters (256 int8)
//        - Bytes 512-767: Beta parameters (256 int8)
// Output: Normalized features (256 int8) = 256 bytes
//
// Note: Phoenix NPU ShimNOC has only 2 DMA channels (input + output)
// So we combine input, gamma, and beta into a single input buffer
//
//===----------------------------------------------------------------------===//

module @layernorm_npu {
    aie.device(npu1) {
        // Declare external layer normalization kernel function
        // Takes combined input buffer (768 bytes = 256*3)
        func.func private @layernorm_int8_256(memref<768xi8>, memref<256xi8>)

        // Declare tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile (DMA)
        %tile02 = aie.tile(0, 2)  // Compute tile

        // Input ObjectFIFO: Combined input+gamma+beta (3 × 256 = 768 bytes)
        aie.objectfifo @of_input_combined(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<768xi8>>

        // Output ObjectFIFO: Normalized output (256 int8 = 256 bytes)
        aie.objectfifo @of_output(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<256xi8>>

        // Core logic - infinite loop with event-driven execution
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index  // Infinite loop

            // Infinite loop - core stays active
            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire combined input buffer (input + gamma + beta)
                %subviewInput = aie.objectfifo.acquire @of_input_combined(Consume, 1) : !aie.objectfifosubview<memref<768xi8>>
                %elemInput = aie.objectfifo.subview.access %subviewInput[0] : !aie.objectfifosubview<memref<768xi8>> -> memref<768xi8>

                // Acquire output buffer
                %subviewOutput = aie.objectfifo.acquire @of_output(Produce, 1) : !aie.objectfifosubview<memref<256xi8>>
                %elemOutput = aie.objectfifo.subview.access %subviewOutput[0] : !aie.objectfifosubview<memref<256xi8>> -> memref<256xi8>

                // Call layer norm kernel
                // Kernel will unpack: input (bytes 0-255), gamma (256-511), beta (512-767)
                func.call @layernorm_int8_256(%elemInput, %elemOutput)
                    : (memref<768xi8>, memref<256xi8>) -> ()

                // Release ObjectFIFOs
                aie.objectfifo.release @of_input_combined(Consume, 1)
                aie.objectfifo.release @of_output(Produce, 1)
            }

            aie.end
        } {link_with="layernorm_combined.o"}

        // Runtime sequence
        aiex.runtime_sequence(%input_combined : memref<768xi8>, %output : memref<256xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c256_i64 = arith.constant 256 : i64
            %c768_i64 = arith.constant 768 : i64

            // DMA transfer: Combined input buffer (host -> NPU)
            // Host must pack: input (bytes 0-255), gamma (256-511), beta (512-767)
            aiex.npu.dma_memcpy_nd(%input_combined[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                                   [%c1_i64, %c1_i64, %c1_i64, %c768_i64]
                                                   [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input_combined,
                id = 1 : i64
            } : memref<768xi8>

            // DMA transfer: Normalized output (NPU -> host)
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
