//===- encoder_streaming_layer.mlir ------------------------------*- MLIR -*-===//
//
// Whisper Encoder Layer with Streaming Architecture
// Processes sequence in chunks of 512 elements (embedding dimension)
//
// Strategy:
// - Process 1500 sequence positions sequentially
// - Each position: 512 elements (1024 bytes bf16)
// - Use 4-column parallelization for operations
// - Internal streaming with ObjectFIFOs
//
// Expected Performance: 10-20ms per full sequence (1500×512)
//
//===----------------------------------------------------------------------===//

module @whisper_encoder_streaming {
    aie.device(npu1) {
        // =================================================================
        // External Kernel Function Declarations
        // =================================================================

        // Streaming kernels (512 elements = 1024 bytes)
        func.func private @layernorm_streaming_bf16(memref<1024xi8>, memref<1024xi8>)
        func.func private @gelu_ffn_bf16(memref<4096xi8>, memref<4096xi8>)

        // Sequence-level softmax (1500 elements = 3000 bytes)
        func.func private @softmax_streaming_bf16(memref<3000xi8>, memref<3000xi8>)

        // 64×64 tiled matmul (3× 8192 bytes)
        func.func private @matmul_64x64_bf16(memref<8192xi8>, memref<8192xi8>, memref<8192xi8>)

        // =================================================================
        // Tile Declarations - 4 columns for parallel processing
        // =================================================================

        // Column 0: Input DMA and LayerNorm 1
        %tile00 = aie.tile(0, 0)  // ShimNOC - DMA
        %tile01 = aie.tile(0, 1)  // MemTile - L2 buffer
        %tile02 = aie.tile(0, 2)  // LayerNorm 1

        // Column 1: Q/K/V projections
        %tile10 = aie.tile(1, 0)  // ShimNOC
        %tile12 = aie.tile(1, 2)  // Q projection (matmul)
        %tile13 = aie.tile(1, 3)  // K projection

        // Column 2: Attention computation
        %tile22 = aie.tile(2, 2)  // V projection
        %tile23 = aie.tile(2, 3)  // Attention scores (Q×K^T)
        %tile24 = aie.tile(2, 4)  // Softmax

        // Column 3: FFN path
        %tile32 = aie.tile(3, 2)  // LayerNorm 2
        %tile33 = aie.tile(3, 3)  // FFN FC1 + GELU
        %tile34 = aie.tile(3, 4)  // FFN FC2

        // =================================================================
        // ObjectFIFOs for Streaming Data Movement
        // =================================================================

        // Input stream: 512 bf16 elements per chunk (1024 bytes)
        aie.objectfifo @of_input(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>

        // LayerNorm 1 output → Q/K/V projections (broadcast)
        aie.objectfifo @of_ln1_out(%tile02, {%tile12, %tile13, %tile22}, 4 : i32) : !aie.objectfifo<memref<1024xi8>>

        // Q, K, V streams to attention
        aie.objectfifo @of_q(%tile12, {%tile23}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>
        aie.objectfifo @of_k(%tile13, {%tile23}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>
        aie.objectfifo @of_v(%tile22, {%tile24}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>

        // Attention scores to softmax (full sequence: 1500 elements = 3000 bytes)
        aie.objectfifo @of_scores(%tile23, {%tile24}, 2 : i32) : !aie.objectfifo<memref<3000xi8>>

        // Softmax output → weighted V
        aie.objectfifo @of_attn_weights(%tile24, {%tile32}, 2 : i32) : !aie.objectfifo<memref<3000xi8>>

        // LayerNorm 2 → FFN
        aie.objectfifo @of_ln2_out(%tile32, {%tile33}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>

        // FFN intermediate (2048 elements = 4096 bytes)
        aie.objectfifo @of_ffn_intermediate(%tile33, {%tile34}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>

        // Final output
        aie.objectfifo @of_output(%tile34, {%tile00}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>

        // =================================================================
        // Core Implementations - Streaming Processing
        // =================================================================

        // Core: LayerNorm 1 (Tile 0,2)
        // Processes 512 elements at a time
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c1500 = arith.constant 1500 : index  // Full sequence length

            scf.for %iter = %c0 to %c1500 step %c1 {
                // Get input chunk (512 elements)
                %subviewIn = aie.objectfifo.acquire @of_input(Consume, 1) : !aie.objectfifosubview<memref<1024xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>

                // Get output buffer
                %subviewOut = aie.objectfifo.acquire @of_ln1_out(Produce, 1) : !aie.objectfifosubview<memref<1024xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>

                // Apply LayerNorm on 512 elements
                func.call @layernorm_streaming_bf16(%elemIn, %elemOut) : (memref<1024xi8>, memref<1024xi8>) -> ()

                // Release buffers
                aie.objectfifo.release @of_input(Consume, 1)
                aie.objectfifo.release @of_ln1_out(Produce, 1)
            }
            aie.end
        } {link_with="layernorm_streaming_bf16.o"}

        // Core: LayerNorm 2 (Tile 3,2)
        %core32 = aie.core(%tile32) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c1500 = arith.constant 1500 : index

            scf.for %iter = %c0 to %c1500 step %c1 {
                %subviewIn = aie.objectfifo.acquire @of_attn_weights(Consume, 1) : !aie.objectfifosubview<memref<3000xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<3000xi8>> -> memref<3000xi8>

                %subviewOut = aie.objectfifo.acquire @of_ln2_out(Produce, 1) : !aie.objectfifosubview<memref<1024xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>

                // For now, simplified - just pass through
                // TODO: Implement proper attention weighting
                func.call @layernorm_streaming_bf16(%elemIn, %elemOut) : (memref<3000xi8>, memref<1024xi8>) -> ()

                aie.objectfifo.release @of_attn_weights(Consume, 1)
                aie.objectfifo.release @of_ln2_out(Produce, 1)
            }
            aie.end
        } {link_with="layernorm_streaming_bf16.o"}

        // Core: FFN with GELU (Tile 3,3)
        // FC1: 512 → 2048, GELU, FC2: 2048 → 512
        %core33 = aie.core(%tile33) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c1500 = arith.constant 1500 : index

            scf.for %iter = %c0 to %c1500 step %c1 {
                // Get input (512 elements)
                %subviewIn = aie.objectfifo.acquire @of_ln2_out(Consume, 1) : !aie.objectfifosubview<memref<1024xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>

                // Get intermediate buffer (2048 elements)
                %subviewIntermediate = aie.objectfifo.acquire @of_ffn_intermediate(Produce, 1) : !aie.objectfifosubview<memref<4096xi8>>
                %elemIntermediate = aie.objectfifo.subview.access %subviewIntermediate[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>

                // TODO: FC1 matmul (simplified for now - just GELU)
                func.call @gelu_ffn_bf16(%elemIn, %elemIntermediate) : (memref<1024xi8>, memref<4096xi8>) -> ()

                aie.objectfifo.release @of_ln2_out(Consume, 1)
                aie.objectfifo.release @of_ffn_intermediate(Produce, 1)
            }
            aie.end
        } {link_with="gelu_ffn_bf16.o"}

        // =================================================================
        // Runtime Sequence - Stream 1500 chunks of 512 elements
        // =================================================================

        aiex.runtime_sequence(
            %input : memref<1536000xi8>,   // 1500×512×2 bytes
            %output : memref<1536000xi8>
        ) {
            %c0 = arith.constant 0 : i64
            %c1 = arith.constant 1 : i64
            %c1024 = arith.constant 1024 : i64  // Chunk size (512 bf16)
            %c1500 = arith.constant 1500 : i64  // Number of chunks

            // Stream input in chunks
            aiex.npu.dma_memcpy_nd(%input[%c0, %c0, %c0, %c0]
                                          [%c1, %c1, %c1500, %c1024]
                                          [%c0, %c0, %c1024, %c1]) {
                metadata = @of_input,
                id = 1 : i64
            } : memref<1536000xi8>

            // Stream output
            aiex.npu.dma_memcpy_nd(%output[%c0, %c0, %c0, %c0]
                                          [%c1, %c1, %c1500, %c1024]
                                          [%c0, %c0, %c1024, %c1]) {
                metadata = @of_output,
                id = 0 : i64
            } : memref<1536000xi8>

            aiex.npu.dma_wait {symbol = @of_output}
        }
    }
}

//===----------------------------------------------------------------------===//
// NOTES:
//
// This is a streaming architecture that processes 1500×512 input in chunks.
// Each chunk (512 elements) flows through the pipeline:
//
// Input → LayerNorm1 → [Q/K/V projections] → Attention → LayerNorm2 → FFN → Output
//
// Simplified for initial testing:
// - Attention computation is placeholder (will add Q×K^T, softmax, ×V)
// - Matmul ops use simplified kernels (need tiled 64×64 implementation)
// - Residual connections not yet implemented
//
// Expected performance: 10-20ms for full 1500×512 sequence
// Memory efficient: Only 1024-4096 bytes per tile at a time
//===----------------------------------------------------------------------===//
