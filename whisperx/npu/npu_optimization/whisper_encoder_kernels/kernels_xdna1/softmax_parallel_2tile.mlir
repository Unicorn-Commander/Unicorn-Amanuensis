//===- softmax_parallel_2tile.mlir --------------------------------*- MLIR -*-===//
//
// BF16 Parallel Softmax - 2 Tiles Processing Simultaneously
// XDNA1 Phoenix NPU - Uses 2 compute tiles in parallel (within DMA limits)
//
// Architecture:
// - Tile (0,0): ShimNOC for DMA input/output
// - Tile (0,2): Compute tile 1 - processes frame 0
// - Tile (0,3): Compute tile 2 - processes frame 1
//
// Input: [2048] bfloat16 values (2 × 1024 elements)
// Output: [2048] bfloat16 values
//
// Expected Performance:
// - Sequential (2x single): 3.1 ms for 2 operations
// - Parallel (this): ~1.6 ms for 2 operations
// - Speedup: ~2x per batch
//
//===----------------------------------------------------------------------===//

module @softmax_parallel_2tile_npu {
    aie.device(npu1) {
        // Declare external softmax kernel function
        func.func private @softmax_bf16(memref<2048xi8>, memref<2048xi8>)

        // Declare tiles - using single column with 2 compute rows
        %tile00 = aie.tile(0, 0)  // ShimNOC tile (DMA)
        %tile02 = aie.tile(0, 2)  // Compute tile 1
        %tile03 = aie.tile(0, 3)  // Compute tile 2

        // Input ObjectFIFOs - one for each compute tile (2048 bytes each)
        aie.objectfifo @of_input0(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
        aie.objectfifo @of_input1(%tile00, {%tile03}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // Output ObjectFIFOs - one from each compute tile
        aie.objectfifo @of_output0(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
        aie.objectfifo @of_output1(%tile03, {%tile00}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // Core 0 - processes frame 0
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                %subviewIn = aie.objectfifo.acquire @of_input0(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                %subviewOut = aie.objectfifo.acquire @of_output0(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                func.call @softmax_bf16(%elemIn, %elemOut) : (memref<2048xi8>, memref<2048xi8>) -> ()

                aie.objectfifo.release @of_input0(Consume, 1)
                aie.objectfifo.release @of_output0(Produce, 1)
            }
            aie.end
        } {link_with="softmax_bf16_xdna1.o"}

        // Core 1 - processes frame 1
        %core03 = aie.core(%tile03) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                %subviewIn = aie.objectfifo.acquire @of_input1(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                %subviewOut = aie.objectfifo.acquire @of_output1(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                func.call @softmax_bf16(%elemIn, %elemOut) : (memref<2048xi8>, memref<2048xi8>) -> ()

                aie.objectfifo.release @of_input1(Consume, 1)
                aie.objectfifo.release @of_output1(Produce, 1)
            }
            aie.end
        } {link_with="softmax_bf16_xdna1.o"}

        // Runtime sequence - transfers 2 frames in parallel
        aiex.runtime_sequence(%in0 : memref<2048xi8>, %in1 : memref<2048xi8>,
                              %out0 : memref<2048xi8>, %out1 : memref<2048xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c2048_i64 = arith.constant 2048 : i64

            // Input DMAs - both frames
            aiex.npu.dma_memcpy_nd(%in0[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                   [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                   [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input0, id = 0 : i64
            } : memref<2048xi8>

            aiex.npu.dma_memcpy_nd(%in1[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                   [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                   [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input1, id = 1 : i64
            } : memref<2048xi8>

            // Output DMAs - both frames
            aiex.npu.dma_memcpy_nd(%out0[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                   [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                   [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output0, id = 2 : i64
            } : memref<2048xi8>

            aiex.npu.dma_memcpy_nd(%out1[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                   [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                   [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output1, id = 3 : i64
            } : memref<2048xi8>

            // Wait for all outputs
            aiex.npu.dma_wait {symbol = @of_output0}
            aiex.npu.dma_wait {symbol = @of_output1}
        }
    }
}
