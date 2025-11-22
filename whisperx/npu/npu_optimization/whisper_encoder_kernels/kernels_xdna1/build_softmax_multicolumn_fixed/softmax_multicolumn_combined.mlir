//===- softmax_multicolumn_combined.mlir -----------------------*- MLIR -*-===//
//
// BF16 Multi-Column Parallel Softmax - 4 Tiles with Combined Buffers
// XDNA1 Phoenix NPU - Uses 2 columns x 2 tiles = 4 compute tiles
//
// Fixed Version: Uses combined input/output buffers to work around XRT
// 5-argument limitation. Instead of 8 separate buffers (which causes segfault),
// we use 2 combined buffers (8192 bytes each) with DMA offsets.
//
// Architecture:
// - Tile (0,0): ShimNOC for column 0 DMA (frames 0,1)
// - Tile (1,0): ShimNOC for column 1 DMA (frames 2,3)
// - Tile (0,2): Compute tile 0 - processes frame 0
// - Tile (0,3): Compute tile 1 - processes frame 1
// - Tile (1,2): Compute tile 2 - processes frame 2
// - Tile (1,3): Compute tile 3 - processes frame 3
//
// Input: Combined buffer [8192] bytes (4 x 2048 bytes = 4 x 1024 bf16 elements)
// Output: Combined buffer [8192] bytes
//
// Expected Performance:
// - Sequential (4x single): 6.3 ms
// - Parallel (4 tiles): ~1.6-2.0 ms (4x speedup)
//
//===----------------------------------------------------------------------===//

module @softmax_multicolumn_combined_npu {
    aie.device(npu1) {
        // Declare external softmax kernel function
        func.func private @softmax_bf16(memref<2048xi8>, memref<2048xi8>)

        // Column 0 tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile for column 0
        %tile02 = aie.tile(0, 2)  // Compute tile 0
        %tile03 = aie.tile(0, 3)  // Compute tile 1

        // Column 1 tiles
        %tile10 = aie.tile(1, 0)  // ShimNOC tile for column 1
        %tile12 = aie.tile(1, 2)  // Compute tile 2
        %tile13 = aie.tile(1, 3)  // Compute tile 3

        // Column 0 ObjectFIFOs (frames 0, 1)
        aie.objectfifo @of_input0(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
        aie.objectfifo @of_input1(%tile00, {%tile03}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
        aie.objectfifo @of_output0(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
        aie.objectfifo @of_output1(%tile03, {%tile00}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // Column 1 ObjectFIFOs (frames 2, 3)
        aie.objectfifo @of_input2(%tile10, {%tile12}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
        aie.objectfifo @of_input3(%tile10, {%tile13}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
        aie.objectfifo @of_output2(%tile12, {%tile10}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
        aie.objectfifo @of_output3(%tile13, {%tile10}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

        // Core 0 (column 0, row 2) - processes frame 0
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

        // Core 1 (column 0, row 3) - processes frame 1
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

        // Core 2 (column 1, row 2) - processes frame 2
        %core12 = aie.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                %subviewIn = aie.objectfifo.acquire @of_input2(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                %subviewOut = aie.objectfifo.acquire @of_output2(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                func.call @softmax_bf16(%elemIn, %elemOut) : (memref<2048xi8>, memref<2048xi8>) -> ()

                aie.objectfifo.release @of_input2(Consume, 1)
                aie.objectfifo.release @of_output2(Produce, 1)
            }
            aie.end
        } {link_with="softmax_bf16_xdna1.o"}

        // Core 3 (column 1, row 3) - processes frame 3
        %core13 = aie.core(%tile13) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                %subviewIn = aie.objectfifo.acquire @of_input3(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                %subviewOut = aie.objectfifo.acquire @of_output3(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

                func.call @softmax_bf16(%elemIn, %elemOut) : (memref<2048xi8>, memref<2048xi8>) -> ()

                aie.objectfifo.release @of_input3(Consume, 1)
                aie.objectfifo.release @of_output3(Produce, 1)
            }
            aie.end
        } {link_with="softmax_bf16_xdna1.o"}

        // Runtime sequence - COMBINED BUFFERS with DMA offsets
        // Uses only 2 data arguments to work within XRT 5-argument limit
        aiex.runtime_sequence(%in_combined : memref<8192xi8>,
                              %out_combined : memref<8192xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c2048_i64 = arith.constant 2048 : i64
            %c4096_i64 = arith.constant 4096 : i64
            %c6144_i64 = arith.constant 6144 : i64

            // Column 0 Input DMAs with offsets into combined buffer
            // Frame 0: offset 0
            aiex.npu.dma_memcpy_nd(%in_combined[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                   [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                   [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input0, id = 0 : i64
            } : memref<8192xi8>

            // Frame 1: offset 2048
            aiex.npu.dma_memcpy_nd(%in_combined[%c0_i64, %c0_i64, %c0_i64, %c2048_i64]
                                   [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                   [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input1, id = 1 : i64
            } : memref<8192xi8>

            // Column 1 Input DMAs with offsets into combined buffer
            // Frame 2: offset 4096
            aiex.npu.dma_memcpy_nd(%in_combined[%c0_i64, %c0_i64, %c0_i64, %c4096_i64]
                                   [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                   [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input2, id = 2 : i64
            } : memref<8192xi8>

            // Frame 3: offset 6144
            aiex.npu.dma_memcpy_nd(%in_combined[%c0_i64, %c0_i64, %c0_i64, %c6144_i64]
                                   [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                   [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input3, id = 3 : i64
            } : memref<8192xi8>

            // Column 0 Output DMAs with offsets into combined buffer
            // Output frame 0: offset 0
            aiex.npu.dma_memcpy_nd(%out_combined[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                   [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                   [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output0, id = 4 : i64
            } : memref<8192xi8>

            // Output frame 1: offset 2048
            aiex.npu.dma_memcpy_nd(%out_combined[%c0_i64, %c0_i64, %c0_i64, %c2048_i64]
                                   [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                   [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output1, id = 5 : i64
            } : memref<8192xi8>

            // Column 1 Output DMAs with offsets into combined buffer
            // Output frame 2: offset 4096
            aiex.npu.dma_memcpy_nd(%out_combined[%c0_i64, %c0_i64, %c0_i64, %c4096_i64]
                                   [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                   [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output2, id = 6 : i64
            } : memref<8192xi8>

            // Output frame 3: offset 6144
            aiex.npu.dma_memcpy_nd(%out_combined[%c0_i64, %c0_i64, %c0_i64, %c6144_i64]
                                   [%c1_i64, %c1_i64, %c1_i64, %c2048_i64]
                                   [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output3, id = 7 : i64
            } : memref<8192xi8>

            // Wait for all outputs
            aiex.npu.dma_wait {symbol = @of_output0}
            aiex.npu.dma_wait {symbol = @of_output1}
            aiex.npu.dma_wait {symbol = @of_output2}
            aiex.npu.dma_wait {symbol = @of_output3}
        }
    }
}
