// Simple Test MLIR-AIE2 Kernel for Phoenix NPU
// Based on working vision_passthrough example
// Purpose: Test compilation pipeline

module @simple_test {
  aie.device(npu1) {
    // Declare external kernel function
    func.func private @simpleKernel(%in: memref<512xui8>, %out: memref<512xui8>, %size: i32) -> ()

    // Shim tile for DMA (row 0)
    %tile00 = aie.tile(0, 0)

    // Compute tile (row 2)
    %tile02 = aie.tile(0, 2)

    // ObjectFIFOs with 512-byte buffers
    aie.objectfifo @inOF(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<512xui8>>
    aie.objectfifo @outOF(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<512xui8>>

    // Core computation
    %core02 = aie.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : i32

      // Process 4 iterations of 512 bytes each
      scf.for %iter = %c0 to %c4 step %c1 {
        // Acquire input
        %subviewIn = aie.objectfifo.acquire @inOF(Consume, 1) : !aie.objectfifosubview<memref<512xui8>>
        %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<512xui8>> -> memref<512xui8>

        // Acquire output
        %subviewOut = aie.objectfifo.acquire @outOF(Produce, 1) : !aie.objectfifosubview<memref<512xui8>>
        %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<512xui8>> -> memref<512xui8>

        // Call kernel
        func.call @simpleKernel(%elemIn, %elemOut, %c512) : (memref<512xui8>, memref<512xui8>, i32) -> ()

        // Release buffers
        aie.objectfifo.release @inOF(Consume, 1)
        aie.objectfifo.release @outOF(Produce, 1)
      }
      aie.end
    }

    // Runtime sequence: 4 x 512 bytes = 2048 bytes total
    // As 32-bit words: 2048/4 = 512 words
    // Per iteration: 512/4 = 128 words
    aiex.runtime_sequence(%in : memref<512xi32>, %dummy : memref<1xi32>, %out : memref<512xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c4 = arith.constant 4 : i64
      %c128 = arith.constant 128 : i64  // 512 bytes / 4 bytes per word

      // Input DMA: 4 iterations x 128 words (512 bytes) each
      aiex.npu.dma_memcpy_nd (0, 0, %in[%c0, %c0, %c0, %c0][%c1, %c1, %c4, %c128][%c0, %c0, %c128, %c1])
        { metadata = @inOF, id = 1 : i64 } : memref<512xi32>

      // Output DMA: 4 iterations x 128 words (512 bytes) each
      aiex.npu.dma_memcpy_nd (0, 0, %out[%c0, %c0, %c0, %c0][%c1, %c1, %c4, %c128][%c0, %c0, %c128, %c1])
        { metadata = @outOF, id = 0 : i64 } : memref<512xi32>

      // Synchronization
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
