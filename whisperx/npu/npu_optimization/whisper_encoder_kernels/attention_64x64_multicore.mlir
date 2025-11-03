//===- attention_64x64_multicore.mlir -----------------------------------*- MLIR -*-===//
//
// Multi-Core Attention Mechanism for Phoenix NPU
// Uses all 4 columns for 4× throughput improvement
//
// Phoenix NPU Architecture:
//   Row 2: [Core 0,2] [Core 1,2] [Core 2,2] [Core 3,2]  ← 4 compute tiles
//   Row 1: [Mem 0,1]  [Mem 1,1]  [Mem 2,1]  [Mem 3,1]   ← Memory tiles
//   Row 0: [Shim 0,0] [Shim 1,0] [Shim 2,0] [Shim 3,0]  ← DMA/NOC
//
// Strategy: Process 4 tiles in parallel (one per column)
//
// Input: 4 × (Q+K+V) = 4 × 12288 bytes = 49152 bytes
// Output: 4 × (64×64) = 4 × 4096 bytes = 16384 bytes
//
// Performance: 4× throughput (process 4 tiles in time of 1)
//
//===----------------------------------------------------------------------===//

module @attention_multicore {
    aie.device(npu1) {
        // Declare external attention kernel (same C code, different tiles)
        func.func private @attention_kernel_64x64_int8(
            memref<12288xi8>,  // Q+K+V input
            memref<4096xi8>    // Attention output
        )

        // ================================================================
        // COLUMN 0: Process Tile 0
        // ================================================================
        %tile00 = aie.tile(0, 0)  // Shim (DMA)
        %tile02 = aie.tile(0, 2)  // Compute

        // Input ObjectFIFO for tile 0
        aie.objectfifo @of_input_0(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<12288xi8>>

        // Output ObjectFIFO for tile 0
        aie.objectfifo @of_output_0(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>

        // Core 0 logic
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                %subviewIn = aie.objectfifo.acquire @of_input_0(Consume, 1) : !aie.objectfifosubview<memref<12288xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<12288xi8>> -> memref<12288xi8>

                %subviewOut = aie.objectfifo.acquire @of_output_0(Produce, 1) : !aie.objectfifosubview<memref<4096xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>

                func.call @attention_kernel_64x64_int8(%elemIn, %elemOut) : (memref<12288xi8>, memref<4096xi8>) -> ()

                aie.objectfifo.release @of_input_0(Consume, 1)
                aie.objectfifo.release @of_output_0(Produce, 1)
            }

            aie.end
        } {link_with="attention_int8_64x64_tiled.o"}

        // ================================================================
        // COLUMN 1: Process Tile 1
        // ================================================================
        %tile10 = aie.tile(1, 0)  // Shim (DMA)
        %tile12 = aie.tile(1, 2)  // Compute

        aie.objectfifo @of_input_1(%tile10, {%tile12}, 2 : i32) : !aie.objectfifo<memref<12288xi8>>
        aie.objectfifo @of_output_1(%tile12, {%tile10}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>

        %core12 = aie.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                %subviewIn = aie.objectfifo.acquire @of_input_1(Consume, 1) : !aie.objectfifosubview<memref<12288xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<12288xi8>> -> memref<12288xi8>

                %subviewOut = aie.objectfifo.acquire @of_output_1(Produce, 1) : !aie.objectfifosubview<memref<4096xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>

                func.call @attention_kernel_64x64_int8(%elemIn, %elemOut) : (memref<12288xi8>, memref<4096xi8>) -> ()

                aie.objectfifo.release @of_input_1(Consume, 1)
                aie.objectfifo.release @of_output_1(Produce, 1)
            }

            aie.end
        } {link_with="attention_int8_64x64_tiled.o"}

        // ================================================================
        // COLUMN 2: Process Tile 2
        // ================================================================
        %tile20 = aie.tile(2, 0)  // Shim (DMA)
        %tile22 = aie.tile(2, 2)  // Compute

        aie.objectfifo @of_input_2(%tile20, {%tile22}, 2 : i32) : !aie.objectfifo<memref<12288xi8>>
        aie.objectfifo @of_output_2(%tile22, {%tile20}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>

        %core22 = aie.core(%tile22) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                %subviewIn = aie.objectfifo.acquire @of_input_2(Consume, 1) : !aie.objectfifosubview<memref<12288xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<12288xi8>> -> memref<12288xi8>

                %subviewOut = aie.objectfifo.acquire @of_output_2(Produce, 1) : !aie.objectfifosubview<memref<4096xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>

                func.call @attention_kernel_64x64_int8(%elemIn, %elemOut) : (memref<12288xi8>, memref<4096xi8>) -> ()

                aie.objectfifo.release @of_input_2(Consume, 1)
                aie.objectfifo.release @of_output_2(Produce, 1)
            }

            aie.end
        } {link_with="attention_int8_64x64_tiled.o"}

        // ================================================================
        // COLUMN 3: Process Tile 3
        // ================================================================
        %tile30 = aie.tile(3, 0)  // Shim (DMA)
        %tile32 = aie.tile(3, 2)  // Compute

        aie.objectfifo @of_input_3(%tile30, {%tile32}, 2 : i32) : !aie.objectfifo<memref<12288xi8>>
        aie.objectfifo @of_output_3(%tile32, {%tile30}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>

        %core32 = aie.core(%tile32) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                %subviewIn = aie.objectfifo.acquire @of_input_3(Consume, 1) : !aie.objectfifosubview<memref<12288xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<12288xi8>> -> memref<12288xi8>

                %subviewOut = aie.objectfifo.acquire @of_output_3(Produce, 1) : !aie.objectfifosubview<memref<4096xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>

                func.call @attention_kernel_64x64_int8(%elemIn, %elemOut) : (memref<12288xi8>, memref<4096xi8>) -> ()

                aie.objectfifo.release @of_input_3(Consume, 1)
                aie.objectfifo.release @of_output_3(Produce, 1)
            }

            aie.end
        } {link_with="attention_int8_64x64_tiled.o"}

        // ================================================================
        // Runtime Sequence: Parallel DMA for all 4 columns
        // ================================================================
        aiex.runtime_sequence(
            %input_0 : memref<12288xi8>, %input_1 : memref<12288xi8>,
            %input_2 : memref<12288xi8>, %input_3 : memref<12288xi8>,
            %output_0 : memref<4096xi8>, %output_1 : memref<4096xi8>,
            %output_2 : memref<4096xi8>, %output_3 : memref<4096xi8>
        ) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c4096_i64 = arith.constant 4096 : i64
            %c12288_i64 = arith.constant 12288 : i64

            // DMA transfers for Column 0 (tile 0)
            aiex.npu.dma_memcpy_nd(%input_0[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                         [%c1_i64, %c1_i64, %c1_i64, %c12288_i64]
                                         [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input_0, id = 1 : i64
            } : memref<12288xi8>

            aiex.npu.dma_memcpy_nd(%output_0[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c4096_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output_0, id = 0 : i64
            } : memref<4096xi8>

            // DMA transfers for Column 1 (tile 1)
            aiex.npu.dma_memcpy_nd(%input_1[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                         [%c1_i64, %c1_i64, %c1_i64, %c12288_i64]
                                         [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input_1, id = 3 : i64
            } : memref<12288xi8>

            aiex.npu.dma_memcpy_nd(%output_1[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c4096_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output_1, id = 2 : i64
            } : memref<4096xi8>

            // DMA transfers for Column 2 (tile 2)
            aiex.npu.dma_memcpy_nd(%input_2[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                         [%c1_i64, %c1_i64, %c1_i64, %c12288_i64]
                                         [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input_2, id = 5 : i64
            } : memref<12288xi8>

            aiex.npu.dma_memcpy_nd(%output_2[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c4096_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output_2, id = 4 : i64
            } : memref<4096xi8>

            // DMA transfers for Column 3 (tile 3)
            aiex.npu.dma_memcpy_nd(%input_3[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                         [%c1_i64, %c1_i64, %c1_i64, %c12288_i64]
                                         [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input_3, id = 7 : i64
            } : memref<12288xi8>

            aiex.npu.dma_memcpy_nd(%output_3[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c4096_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output_3, id = 6 : i64
            } : memref<4096xi8>

            // Wait for all output DMAs to complete
            aiex.npu.dma_wait {symbol = @of_output_0}
            aiex.npu.dma_wait {symbol = @of_output_1}
            aiex.npu.dma_wait {symbol = @of_output_2}
            aiex.npu.dma_wait {symbol = @of_output_3}
        }
    }
}
