//===- matmul_simple.mlir ---------------------------------------*- MLIR -*-===//
//
// INT8 Matrix Multiply for Whisper Encoder - Simple 16x16 tile
// Based on working mel_with_loop pattern from mel_kernels
//
// Input: 16x16 int8 matrix A (256 bytes)
// Input: 16x16 int8 matrix B (256 bytes)
// Output: 16x16 int8 matrix C (256 bytes)
//
//===----------------------------------------------------------------------===//

module @matmul_npu {
    aie.device(npu1) {
        // Declare external matmul kernel function (compiled from matmul_int8.c)
        func.func private @matmul_int8_16x16(memref<256xi8>, memref<256xi8>, memref<256xi8>)

        // Declare tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile (DMA)
        %tile02 = aie.tile(0, 2)  // Compute tile

        // Input ObjectFIFO: A matrix (16x16 int8 = 256 bytes)
        aie.objectfifo @of_matA(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<256xi8>>

        // Input ObjectFIFO: B matrix (16x16 int8 = 256 bytes)
        aie.objectfifo @of_matB(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<256xi8>>

        // Output ObjectFIFO: C matrix (16x16 int8 = 256 bytes)
        aie.objectfifo @of_matC(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<256xi8>>

        // Core logic - infinite loop with event-driven execution
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index  // Infinite loop

            // Infinite loop - core stays active
            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire input matrices from ObjectFIFOs
                %subviewA = aie.objectfifo.acquire @of_matA(Consume, 1) : !aie.objectfifosubview<memref<256xi8>>
                %elemA = aie.objectfifo.subview.access %subviewA[0] : !aie.objectfifosubview<memref<256xi8>> -> memref<256xi8>

                %subviewB = aie.objectfifo.acquire @of_matB(Consume, 1) : !aie.objectfifosubview<memref<256xi8>>
                %elemB = aie.objectfifo.subview.access %subviewB[0] : !aie.objectfifosubview<memref<256xi8>> -> memref<256xi8>

                // Acquire output buffer
                %subviewC = aie.objectfifo.acquire @of_matC(Produce, 1) : !aie.objectfifosubview<memref<256xi8>>
                %elemC = aie.objectfifo.subview.access %subviewC[0] : !aie.objectfifosubview<memref<256xi8>> -> memref<256xi8>

                // Call matmul kernel: A @ B -> C (INT8 output, quantized internally)
                func.call @matmul_int8_16x16(%elemA, %elemB, %elemC) : (memref<256xi8>, memref<256xi8>, memref<256xi8>) -> ()

                // Release ObjectFIFOs
                aie.objectfifo.release @of_matA(Consume, 1)
                aie.objectfifo.release @of_matB(Consume, 1)
                aie.objectfifo.release @of_matC(Produce, 1)
            }

            aie.end
        } {link_with="matmul_combined.o"}

        // Runtime sequence
        aiex.runtime_sequence(%matA : memref<256xi8>, %matB : memref<256xi8>, %matC : memref<256xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c256_i64 = arith.constant 256 : i64

            // DMA transfer: Matrix A (host -> NPU)
            aiex.npu.dma_memcpy_nd(%matA[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                        [%c1_i64, %c1_i64, %c1_i64, %c256_i64]
                                        [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_matA,
                id = 1 : i64
            } : memref<256xi8>

            // DMA transfer: Matrix B (host -> NPU)
            aiex.npu.dma_memcpy_nd(%matB[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                        [%c1_i64, %c1_i64, %c1_i64, %c256_i64]
                                        [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_matB,
                id = 2 : i64
            } : memref<256xi8>

            // DMA transfer: Matrix C (NPU -> host)
            aiex.npu.dma_memcpy_nd(%matC[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                        [%c1_i64, %c1_i64, %c1_i64, %c256_i64]
                                        [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_matC,
                id = 0 : i64
            } : memref<256xi8>

            // Wait for output DMA completion
            aiex.npu.dma_wait {symbol = @of_matC}
        }
    }
}
