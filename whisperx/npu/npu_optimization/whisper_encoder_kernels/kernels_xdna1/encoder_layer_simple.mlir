//===- encoder_layer_simple.mlir --------------------------------*- MLIR -*-===//
//
// Simplified Whisper Encoder Layer - XDNA1 Phoenix NPU
// Chains multiple kernels to demonstrate encoder layer pattern
//
// Simplified Pipeline:
//   Input -> LayerNorm -> Softmax -> GELU -> Output
//
// Note: MatMul is excluded from this first version because it requires
// different buffer sizes (8192 bytes for 64x64 matrices vs 2048 bytes).
// A separate version with proper MatMul integration will follow.
//
// This tests the kernel chaining pattern with 3 sequential operations.
//
// Tile Assignment:
// - Tile (0,0): ShimNOC for DMA input/output
// - Tile (0,2): LayerNorm kernel
// - Tile (0,3): Softmax kernel
// - Tile (0,4): GELU kernel
//
// Buffer Sizes (consistent 2048 bytes throughout):
// - LayerNorm: 1024 bfloat16 = 2048 bytes (in/out)
// - Softmax: 1024 bfloat16 = 2048 bytes (in/out)
// - GELU: 1024 bfloat16 = 2048 bytes (in/out)
//
// Expected Combined Performance: ~3.8 ms
// - LayerNorm: 0.8 ms
// - Softmax: 1.5 ms
// - GELU: 1.5 ms
//
// Data Flow:
//   Host -> [DMA] -> LayerNorm -> [ObjectFIFO] -> Softmax -> [ObjectFIFO] -> GELU -> [DMA] -> Host
//
//===----------------------------------------------------------------------===//

module @encoder_layer_simple_npu {
    aie.device(npu1) {
        // =================================================================
        // External Kernel Function Declarations
        // =================================================================

        // LayerNorm kernel: (input, output) -> void
        func.func private @layernorm_simple_bf16(memref<2048xi8>, memref<2048xi8>)

        // Softmax kernel: (input, output) -> void
        func.func private @softmax_bf16(memref<2048xi8>, memref<2048xi8>)

        // GELU kernel: (input, output) -> void
        func.func private @gelu_bf16(memref<2048xi8>, memref<2048xi8>)

        // =================================================================
        // Tile Declarations
        // =================================================================

        %tile00 = aie.tile(0, 0)  // ShimNOC tile (DMA)
        %tile02 = aie.tile(0, 2)  // LayerNorm compute tile
        %tile03 = aie.tile(0, 3)  // Softmax compute tile
        %tile04 = aie.tile(0, 4)  // GELU compute tile

        // =================================================================
        // ObjectFIFOs for Data Movement (kernel chaining)
        // =================================================================

        // --- Stage 1: Input to LayerNorm ---
        // Host DMA sends input to LayerNorm tile
        aie.objectfifo @of_input_to_ln(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // --- Stage 2: LayerNorm to Softmax ---
        // LayerNorm output feeds directly to Softmax
        aie.objectfifo @of_ln_to_softmax(%tile02, {%tile03}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // --- Stage 3: Softmax to GELU ---
        // Softmax output feeds to GELU (simulating FFN activation)
        aie.objectfifo @of_softmax_to_gelu(%tile03, {%tile04}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // --- Stage 4: GELU to Output ---
        // GELU output back to host
        aie.objectfifo @of_gelu_to_output(%tile04, {%tile00}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // =================================================================
        // Core 0: LayerNorm (Tile 0,2)
        // First stage: normalize input features
        // =================================================================

        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire input from host
                %subviewIn = aie.objectfifo.acquire @of_input_to_ln(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                // Acquire output buffer to send to Softmax
                %subviewOut = aie.objectfifo.acquire @of_ln_to_softmax(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                // Execute LayerNorm
                func.call @layernorm_simple_bf16(%elemIn, %elemOut)
                    : (memref<2048xi8>, memref<2048xi8>) -> ()

                // Release buffers
                aie.objectfifo.release @of_input_to_ln(Consume, 1)
                aie.objectfifo.release @of_ln_to_softmax(Produce, 1)
            }

            aie.end
        } {link_with="layernorm_bf16_xdna1.o"}

        // =================================================================
        // Core 1: Softmax (Tile 0,3)
        // Second stage: compute attention scores (or normalize features)
        // =================================================================

        %core03 = aie.core(%tile03) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire input from LayerNorm
                %subviewIn = aie.objectfifo.acquire @of_ln_to_softmax(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                // Acquire output buffer to send to GELU
                %subviewOut = aie.objectfifo.acquire @of_softmax_to_gelu(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                // Execute Softmax
                func.call @softmax_bf16(%elemIn, %elemOut)
                    : (memref<2048xi8>, memref<2048xi8>) -> ()

                // Release buffers
                aie.objectfifo.release @of_ln_to_softmax(Consume, 1)
                aie.objectfifo.release @of_softmax_to_gelu(Produce, 1)
            }

            aie.end
        } {link_with="softmax_bf16_xdna1.o"}

        // =================================================================
        // Core 2: GELU (Tile 0,4)
        // Third stage: apply GELU activation (FFN layer)
        // =================================================================

        %core04 = aie.core(%tile04) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire input from Softmax
                %subviewIn = aie.objectfifo.acquire @of_softmax_to_gelu(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                // Acquire output buffer to send to host
                %subviewOut = aie.objectfifo.acquire @of_gelu_to_output(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                // Execute GELU
                func.call @gelu_bf16(%elemIn, %elemOut)
                    : (memref<2048xi8>, memref<2048xi8>) -> ()

                // Release buffers
                aie.objectfifo.release @of_softmax_to_gelu(Consume, 1)
                aie.objectfifo.release @of_gelu_to_output(Produce, 1)
            }

            aie.end
        } {link_with="gelu_simple_xdna1.o"}

        // =================================================================
        // Runtime Sequence - DMA Configuration
        // =================================================================

        aiex.runtime_sequence(
            %input : memref<2048xi8>,           // Input to LayerNorm
            %output : memref<2048xi8>           // Final output from GELU
        ) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c2048_i64 = arith.constant 2048 : i64

            // --- Input DMA Transfer (Host -> NPU) ---

            // Transfer input to LayerNorm tile
            aiex.npu.dma_memcpy_nd(%input[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input_to_ln,
                id = 1 : i64
            } : memref<2048xi8>

            // --- Output DMA Transfer (NPU -> Host) ---

            // Receive final output from GELU tile
            aiex.npu.dma_memcpy_nd(%output[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_gelu_to_output,
                id = 0 : i64
            } : memref<2048xi8>

            // Wait for final output DMA completion
            aiex.npu.dma_wait {symbol = @of_gelu_to_output}
        }
    }
}
