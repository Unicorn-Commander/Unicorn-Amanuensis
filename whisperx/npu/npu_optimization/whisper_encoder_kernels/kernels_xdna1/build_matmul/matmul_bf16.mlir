//===- matmul_bf16.mlir -----------------------------------------*- MLIR -*-===//
//
// BF16 Matrix Multiplication for Whisper Encoder - XDNA1 Phoenix NPU
// Tile-based matmul for attention and FFN layers
//
// C = A * B
// Using BF16 inputs/outputs with FP32 accumulation
//
// Fixed size: 64×64 matrices
// Input A: 64×64 bfloat16 = 8192 bytes
// Input B: 64×64 bfloat16 = 8192 bytes
// Output C: 64×64 bfloat16 = 8192 bytes
//
//===----------------------------------------------------------------------===//

module @matmul_bf16_npu {
    aie.device(npu1) {
        // Declare external MatMul kernel function
        // Takes three 64×64 BF16 matrices (as raw bytes)
        // C = A * B
        func.func private @matmul_bf16_64x64(memref<8192xi8>, memref<8192xi8>, memref<8192xi8>)

        // Declare tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile (DMA)
        %tile02 = aie.tile(0, 2)  // Compute tile

        // Input A ObjectFIFO: 64×64 bfloat16 elements = 8192 bytes
        aie.objectfifo @of_input_A(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>

        // Input B ObjectFIFO: 64×64 bfloat16 elements = 8192 bytes
        aie.objectfifo @of_input_B(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>

        // Output C ObjectFIFO: 64×64 bfloat16 elements = 8192 bytes
        aie.objectfifo @of_output(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>

        // Core logic - infinite loop with event-driven execution
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index  // Infinite loop

            // Infinite loop - core stays active
            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire input A buffer
                %subviewA = aie.objectfifo.acquire @of_input_A(Consume, 1) : !aie.objectfifosubview<memref<8192xi8>>
                %elemA = aie.objectfifo.subview.access %subviewA[0] : !aie.objectfifosubview<memref<8192xi8>> -> memref<8192xi8>

                // Acquire input B buffer
                %subviewB = aie.objectfifo.acquire @of_input_B(Consume, 1) : !aie.objectfifosubview<memref<8192xi8>>
                %elemB = aie.objectfifo.subview.access %subviewB[0] : !aie.objectfifosubview<memref<8192xi8>> -> memref<8192xi8>

                // Acquire output buffer
                %subviewOut = aie.objectfifo.acquire @of_output(Produce, 1) : !aie.objectfifosubview<memref<8192xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<8192xi8>> -> memref<8192xi8>

                // Call MatMul kernel: C = A * B
                func.call @matmul_bf16_64x64(%elemA, %elemB, %elemOut)
                    : (memref<8192xi8>, memref<8192xi8>, memref<8192xi8>) -> ()

                // Release ObjectFIFOs
                aie.objectfifo.release @of_input_A(Consume, 1)
                aie.objectfifo.release @of_input_B(Consume, 1)
                aie.objectfifo.release @of_output(Produce, 1)
            }

            aie.end
        } {link_with="matmul_bf16_xdna1.o"}

        // Runtime sequence - DMA transfers for 3 buffers
        aiex.runtime_sequence(%inputA : memref<8192xi8>, %inputB : memref<8192xi8>, %output : memref<8192xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c8192_i64 = arith.constant 8192 : i64

            // DMA transfer: Input A buffer (host -> NPU)
            aiex.npu.dma_memcpy_nd(%inputA[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c8192_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input_A,
                id = 1 : i64
            } : memref<8192xi8>

            // DMA transfer: Input B buffer (host -> NPU)
            aiex.npu.dma_memcpy_nd(%inputB[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c8192_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input_B,
                id = 2 : i64
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
