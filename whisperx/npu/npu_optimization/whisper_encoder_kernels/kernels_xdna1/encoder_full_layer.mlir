//===- encoder_full_layer.mlir -----------------------------------*- MLIR -*-===//
//
// Whisper Full Encoder Layer - XDNA1 Phoenix NPU
// Processes entire encoder layer in single kernel invocation
//
// Architecture:
// - Uses all 4 columns for parallel processing
// - Streams data through tiles in pipeline fashion
// - Processes 64-element vectors to fit tile memory
//
// Operations per layer:
// 1. LayerNorm (pre-attention)
// 2. Q, K, V projections (3x matmul 512x512)
// 3. Attention: Q@K^T, softmax, @V
// 4. Output projection (matmul 512x512)
// 5. Residual add
// 6. LayerNorm (pre-FFN)
// 7. FFN FC1 (matmul 512x2048)
// 8. GELU activation
// 9. FFN FC2 (matmul 2048x512)
// 10. Residual add
//
// Memory Strategy:
// - Tile each (1500×512) operation into (64×64) chunks
// - Stream through pipeline: 1500/64 = 24 chunks per sequence position
// - Each tile processes one operation type
//
// Tile Assignment:
// Column 0: Input DMA, LayerNorm, Q projection
// Column 1: K projection, V projection, QK matmul
// Column 2: Softmax, Attention@V, O projection
// Column 3: FFN FC1, GELU, FFN FC2, Output DMA
//
// Expected Performance:
// - Target: 10-20ms per layer
// - 6 layers = 60-120ms
// - 30s audio = 250-500x realtime
//
//===----------------------------------------------------------------------===//

module @whisper_encoder_full_layer {
    aie.device(npu1) {
        // =================================================================
        // External Kernel Function Declarations
        // =================================================================

        // Element-wise operations (process 2048 bytes = 1024 BF16)
        func.func private @layernorm_bf16(memref<2048xi8>, memref<2048xi8>)
        func.func private @softmax_bf16(memref<2048xi8>, memref<2048xi8>)
        func.func private @gelu_bf16(memref<2048xi8>, memref<2048xi8>)

        // Matrix multiply 64x64 (processes 8192 bytes each)
        func.func private @matmul_bf16_64x64(memref<8192xi8>, memref<8192xi8>, memref<8192xi8>)

        // Vector add for residual (2048 bytes)
        func.func private @vector_add_bf16(memref<2048xi8>, memref<2048xi8>, memref<2048xi8>)

        // =================================================================
        // Tile Declarations - Using 2 columns for initial version
        // =================================================================

        // Column 0: DMA and initial processing
        %tile00 = aie.tile(0, 0)  // ShimNOC - DMA controller
        %tile02 = aie.tile(0, 2)  // LayerNorm 1
        %tile03 = aie.tile(0, 3)  // Q projection (tiled matmul)
        %tile04 = aie.tile(0, 4)  // K projection

        // Column 1: Attention and output
        %tile10 = aie.tile(1, 0)  // ShimNOC - DMA controller 2
        %tile12 = aie.tile(1, 2)  // V projection
        %tile13 = aie.tile(1, 3)  // Attention computation
        %tile14 = aie.tile(1, 4)  // Output projection

        // Column 2: FFN path
        %tile22 = aie.tile(2, 2)  // LayerNorm 2
        %tile23 = aie.tile(2, 3)  // FFN FC1
        %tile24 = aie.tile(2, 4)  // GELU

        // Column 3: FFN output
        %tile32 = aie.tile(3, 2)  // FFN FC2
        %tile33 = aie.tile(3, 3)  // Residual add 2

        // =================================================================
        // ObjectFIFOs for Data Movement
        // =================================================================

        // Buffer size: 2048 bytes for element-wise, 8192 for matmul

        // --- Input path ---
        aie.objectfifo @of_input(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // --- LayerNorm 1 to QKV projections ---
        aie.objectfifo @of_ln1_to_q(%tile02, {%tile03}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
        aie.objectfifo @of_ln1_to_k(%tile02, {%tile04}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
        aie.objectfifo @of_ln1_to_v(%tile02, {%tile12}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // --- Q, K, V to attention ---
        aie.objectfifo @of_q(%tile03, {%tile13}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
        aie.objectfifo @of_k(%tile04, {%tile13}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
        aie.objectfifo @of_v(%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // --- Attention to output projection ---
        aie.objectfifo @of_attn(%tile13, {%tile14}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // --- Output projection to residual ---
        aie.objectfifo @of_o_proj(%tile14, {%tile22}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // --- Residual 1 (need original input) - use double buffer ---
        aie.objectfifo @of_residual1(%tile02, {%tile22}, 4 : i32) : !aie.objectfifo<memref<2048xi8>>

        // --- LayerNorm 2 to FFN ---
        aie.objectfifo @of_ln2(%tile22, {%tile23}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // --- FFN path ---
        aie.objectfifo @of_fc1(%tile23, {%tile24}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
        aie.objectfifo @of_gelu(%tile24, {%tile32}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
        aie.objectfifo @of_fc2(%tile32, {%tile33}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // --- Final output ---
        aie.objectfifo @of_output(%tile33, {%tile00}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // =================================================================
        // Core Implementations - Simplified for Initial Version
        // =================================================================

        // Core: LayerNorm 1 (Tile 0,2)
        // Also broadcasts to residual path
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                // Get input
                %subviewIn = aie.objectfifo.acquire @of_input(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                // Get output buffers
                %subviewOutQ = aie.objectfifo.acquire @of_ln1_to_q(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemOutQ = aie.objectfifo.subview.access %subviewOutQ[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                // Apply LayerNorm
                func.call @layernorm_bf16(%elemIn, %elemOutQ) : (memref<2048xi8>, memref<2048xi8>) -> ()

                // Release buffers
                aie.objectfifo.release @of_input(Consume, 1)
                aie.objectfifo.release @of_ln1_to_q(Produce, 1)
            }
            aie.end
        } {link_with="layernorm_bf16_xdna1.o"}

        // Core: Attention with softmax (Tile 1,3)
        // Simplified: takes Q, K, V and produces attention output
        %core13 = aie.core(%tile13) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                // Get Q, K, V inputs
                %subviewQ = aie.objectfifo.acquire @of_q(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemQ = aie.objectfifo.subview.access %subviewQ[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                %subviewK = aie.objectfifo.acquire @of_k(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemK = aie.objectfifo.subview.access %subviewK[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                %subviewV = aie.objectfifo.acquire @of_v(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemV = aie.objectfifo.subview.access %subviewV[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                // Get output
                %subviewOut = aie.objectfifo.acquire @of_attn(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                // Compute attention (simplified - needs proper implementation)
                // In real version: scores = Q@K^T/sqrt(d), probs = softmax(scores), out = probs@V
                // For now, just pass through V
                func.call @softmax_bf16(%elemV, %elemOut) : (memref<2048xi8>, memref<2048xi8>) -> ()

                // Release all
                aie.objectfifo.release @of_q(Consume, 1)
                aie.objectfifo.release @of_k(Consume, 1)
                aie.objectfifo.release @of_v(Consume, 1)
                aie.objectfifo.release @of_attn(Produce, 1)
            }
            aie.end
        } {link_with="softmax_bf16_xdna1.o"}

        // Core: GELU activation (Tile 2,4)
        %core24 = aie.core(%tile24) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                %subviewIn = aie.objectfifo.acquire @of_fc1(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                %subviewOut = aie.objectfifo.acquire @of_gelu(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                func.call @gelu_bf16(%elemIn, %elemOut) : (memref<2048xi8>, memref<2048xi8>) -> ()

                aie.objectfifo.release @of_fc1(Consume, 1)
                aie.objectfifo.release @of_gelu(Produce, 1)
            }
            aie.end
        } {link_with="gelu_simple_xdna1.o"}

        // =================================================================
        // Runtime Sequence - Simplified single-pass for testing
        // =================================================================

        aiex.runtime_sequence(
            %input : memref<2048xi8>,
            %output : memref<2048xi8>
        ) {
            %c0 = arith.constant 0 : i64
            %c1 = arith.constant 1 : i64
            %c2048 = arith.constant 2048 : i64

            // Input DMA
            aiex.npu.dma_memcpy_nd(%input[%c0, %c0, %c0, %c0]
                                          [%c1, %c1, %c1, %c2048]
                                          [%c0, %c0, %c0, %c1]) {
                metadata = @of_input,
                id = 1 : i64
            } : memref<2048xi8>

            // Output DMA
            aiex.npu.dma_memcpy_nd(%output[%c0, %c0, %c0, %c0]
                                          [%c1, %c1, %c1, %c2048]
                                          [%c0, %c0, %c0, %c1]) {
                metadata = @of_output,
                id = 0 : i64
            } : memref<2048xi8>

            aiex.npu.dma_wait {symbol = @of_output}
        }
    }
}

//===----------------------------------------------------------------------===//
// NOTES:
//
// This is a simplified version for initial testing. Full implementation needs:
//
// 1. All cores implemented (currently only 3 shown)
// 2. Proper tiled matmul for projections (64x64 tiles)
// 3. Weight loading buffers for projections
// 4. Attention score computation with Q@K^T
// 5. Multi-chunk streaming for full 1500-length sequences
// 6. Residual connection handling
// 7. LayerNorm scale/bias parameters
//
// The simplified version tests:
// - Multi-tile pipeline
// - ObjectFIFO connectivity
// - Basic operation chaining
//
// Performance expectation for this simplified version: ~5ms
// Full implementation target: 10-20ms per layer
//===----------------------------------------------------------------------===//
