//===- attention_64x64.mlir ------------------------------------*- MLIR -*-===//
//
// INT8 Attention Mechanism for Whisper Encoder - SCALED TO 64x64 tiles
// Based on working mel_with_loop and matmul_simple patterns
//
// Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
//
// Input: Combined Q,K,V buffer [3 x 64x64] = 12288 bytes
// Output: Attention output [64x64] = 4096 bytes
//
// Note: Phoenix NPU ShimNOC has only 2 DMA channels (input + output)
// So we combine Q, K, V into a single input buffer
//
// MEMORY CONSTRAINT: AIE2 cores have ~32KB
// 64x64 int32 accumulator = 16KB (should fit)
//
//===----------------------------------------------------------------------===//

module @attention_npu_64x64 {
    aie.device(npu1) {
        // Declare external attention kernel function
        // Now takes combined QKV buffer (12288 bytes) for 64x64 matrices
        func.func private @attention_64x64(memref<12288xi8>, memref<4096xi8>, i32)

        // Declare tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile (DMA)
        %tile02 = aie.tile(0, 2)  // Compute tile

        // Input ObjectFIFO: Combined Q+K+V (3 × 64x64 = 12288 bytes)
        aie.objectfifo @of_QKV(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<12288xi8>>

        // Output ObjectFIFO: Attention output (64x64 int8 = 4096 bytes)
        aie.objectfifo @of_out(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>

        // Core logic - infinite loop with event-driven execution
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index  // Infinite loop
            %c_scale = arith.constant 4 : i32  // Scale shift: sqrt(64) = 8, so shift by 3 bits

            // Infinite loop - core stays active
            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire combined QKV input buffer
                %subviewQKV = aie.objectfifo.acquire @of_QKV(Consume, 1) : !aie.objectfifosubview<memref<12288xi8>>
                %elemQKV = aie.objectfifo.subview.access %subviewQKV[0] : !aie.objectfifosubview<memref<12288xi8>> -> memref<12288xi8>

                // Acquire output buffer
                %subviewOut = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<4096xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>

                // Call attention kernel: Attention(QKV_combined) -> output
                // Kernel will unpack Q (bytes 0-4095), K (4096-8191), V (8192-12287)
                func.call @attention_64x64(%elemQKV, %elemOut, %c_scale)
                    : (memref<12288xi8>, memref<4096xi8>, i32) -> ()

                // Release ObjectFIFOs
                aie.objectfifo.release @of_QKV(Consume, 1)
                aie.objectfifo.release @of_out(Produce, 1)
            }

            aie.end
        } {link_with="attention_combined_64x64.o"}

        // Runtime sequence
        aiex.runtime_sequence(%QKV_combined : memref<12288xi8>, %out : memref<4096xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c4096_i64 = arith.constant 4096 : i64
            %c12288_i64 = arith.constant 12288 : i64

            // DMA transfer: Combined Q+K+V buffer (host -> NPU)
            // Host must pack: Q (bytes 0-4095), K (4096-8191), V (8192-12287)
            aiex.npu.dma_memcpy_nd(%QKV_combined[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                                [%c1_i64, %c1_i64, %c1_i64, %c12288_i64]
                                                [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_QKV,
                id = 1 : i64
            } : memref<12288xi8>

            // DMA transfer: Attention output (NPU -> host)
            aiex.npu.dma_memcpy_nd(%out[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                       [%c1_i64, %c1_i64, %c1_i64, %c4096_i64]
                                       [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_out,
                id = 0 : i64
            } : memref<4096xi8>

            // Wait for output DMA completion
            aiex.npu.dma_wait {symbol = @of_out}
        }
    }
}
