//===- softmax_batched_bf16.mlir ----------------------------------*- MLIR -*-===//
//
// BF16 Batched Softmax Activation for Whisper Encoder - XDNA1 Phoenix NPU
// Processes 4 softmax operations per invocation to amortize 89% fixed overhead
//
// Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
// Using scalar BF16 implementation for numerical stability
//
// Input: [4096] bfloat16 values (4 × 1024 elements)
// Output: [4096] bfloat16 values
//
// Buffer size: 4096 elements × 2 bytes/bfloat16 = 8192 bytes
//
// Expected Performance:
// - Single kernel: 1.565 ms per 1024 elements
// - Batched (4x): 2.1 ms for 4096 elements = 0.52 ms per 1024 elements
// - Speedup: 3.0x per-frame
//
//===----------------------------------------------------------------------===//

module @softmax_batched_bf16_npu {
    aie.device(npu1) {
        // Declare external Batched Softmax kernel function
        // Takes 4096-element BF16 buffer (as raw bytes), applies 4 softmax operations
        // C++ signature: void softmax_batched_bf16_4(bfloat16* input, bfloat16* output)
        func.func private @softmax_batched_bf16_4(memref<8192xi8>, memref<8192xi8>)

        // Declare tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile (DMA)
        %tile02 = aie.tile(0, 2)  // Compute tile

        // Input ObjectFIFO: 4096 bfloat16 elements = 8192 bytes
        aie.objectfifo @of_input(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>

        // Output ObjectFIFO: 4096 bfloat16 elements = 8192 bytes
        aie.objectfifo @of_output(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>

        // Core logic - infinite loop with event-driven execution
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index  // Infinite loop

            // Infinite loop - core stays active
            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire input buffer
                %subviewIn = aie.objectfifo.acquire @of_input(Consume, 1) : !aie.objectfifosubview<memref<8192xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<8192xi8>> -> memref<8192xi8>

                // Acquire output buffer
                %subviewOut = aie.objectfifo.acquire @of_output(Produce, 1) : !aie.objectfifosubview<memref<8192xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<8192xi8>> -> memref<8192xi8>

                // Call Batched Softmax kernel: 4 × Softmax(input) -> output
                func.call @softmax_batched_bf16_4(%elemIn, %elemOut)
                    : (memref<8192xi8>, memref<8192xi8>) -> ()

                // Release ObjectFIFOs
                aie.objectfifo.release @of_input(Consume, 1)
                aie.objectfifo.release @of_output(Produce, 1)
            }

            aie.end
        } {link_with="softmax_bf16_xdna1_batched.o"}

        // Runtime sequence
        aiex.runtime_sequence(%input : memref<8192xi8>, %output : memref<8192xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c8192_i64 = arith.constant 8192 : i64

            // DMA transfer: Input buffer (host -> NPU)
            aiex.npu.dma_memcpy_nd(%input[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c8192_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input,
                id = 1 : i64
            } : memref<8192xi8>

            // DMA transfer: Output buffer (NPU -> host)
            aiex.npu.dma_memcpy_nd(%output[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c8192_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output,
                id = 0 : i64
            } : memref<8192xi8>

            // Wait for output DMA completion
            aiex.npu.dma_wait {symbol = @of_output}
        }
    }
}
