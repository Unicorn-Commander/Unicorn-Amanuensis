//===- mel_int8_with_dma.mlir ----------------------------------*- MLIR -*-===//
//
// MEL spectrogram INT8 kernel with DMA sequences for NPU
// Based on working vision_passthrough example
//
//===----------------------------------------------------------------------===//

module @mel_int8_npu {
    aie.device(npu1) {
        // Declare tiles: ShimNOC at (0,0) and Compute at (0,2)
        %tile00 = aie.tile(0, 0)
        %tile02 = aie.tile(0, 2)

        // ObjectFIFO for input audio data (400 INT16 samples = 800 bytes)
        // Using ui8 for byte-level addressing, but conceptually INT16 audio
        aie.objectfifo @audioIn(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<800xui8>>

        // ObjectFIFO for output mel features (80 INT8 mel bins)
        aie.objectfifo @melOut(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<80xui8>>

        // Define the algorithm for the compute tile
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %iterations = arith.constant 1 : index

            scf.for %iter = %c0 to %iterations step %c1 {
                // Acquire input objectfifo
                %subviewIn = aie.objectfifo.acquire @audioIn(Consume, 1) : !aie.objectfifosubview<memref<800xui8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<800xui8>> -> memref<800xui8>

                // Acquire output objectfifo
                %subviewOut = aie.objectfifo.acquire @melOut(Produce, 1) : !aie.objectfifosubview<memref<80xui8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<80xui8>> -> memref<80xui8>

                // TODO: Add mel spectrogram computation here
                // For now, this is an empty passthrough to test DMA sequences

                // Release objectfifos
                aie.objectfifo.release @audioIn(Consume, 1)
                aie.objectfifo.release @melOut(Produce, 1)
            }
            aie.end
        }

        // Runtime sequence for host-NPU data movement
        // Input: 400 INT16 samples = 800 bytes = 200 32-bit words
        // Output: 80 INT8 values = 80 bytes = 20 32-bit words
        aiex.runtime_sequence(%in : memref<200xi32>, %arg1 : memref<1xi32>, %out : memref<20xi32>) {
            %c0 = arith.constant 0 : i64
            %c1 = arith.constant 1 : i64
            %c200 = arith.constant 200 : i64  // Input size in 32-bit words (800 bytes / 4)
            %c20 = arith.constant 20 : i64    // Output size in 32-bit words (80 bytes / 4)

            // DMA memcpy for input audio data
            // Format: [offset][length][stride] in 32-bit words
            aiex.npu.dma_memcpy_nd (%in[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c200][%c0, %c0, %c0, %c1]) { metadata = @audioIn, id = 1 : i64 } : memref<200xi32>

            // DMA memcpy for output mel features
            aiex.npu.dma_memcpy_nd (%out[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c20][%c0, %c0, %c0, %c1]) { metadata = @melOut, id = 0 : i64 } : memref<20xi32>

            // Synchronization - wait for DMA completion
            aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
        }
    }
}
