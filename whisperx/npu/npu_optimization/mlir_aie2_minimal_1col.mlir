// Corrected Minimal MLIR-AIE2 Kernel for WhisperX NPU
// Target: AMD Phoenix NPU (npu)
// Purpose: Simple vector multiply-add for testing compilation

module @whisperx_minimal_corrected {
  aie.device(npu1_1col) {
    // Declare external kernel function
    func.func private @vector_multiply(%in: memref<1024xi32>, %out: memref<1024xi32>, %size: i32) -> ()

    // Define shim tile (row 0) for DMA
    %tile00 = aie.tile(0, 0)

    // Define compute tile (row 2 has memory in AIE2)
    %tile02 = aie.tile(0, 2)

    // ObjectFIFOs for data movement between shim and compute tiles
    // Depth of 2 enables ping-pong buffering
    aie.objectfifo @inOF(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @outOF(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>

    // Core computation on tile02
    %core02 = aie.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c1024 = arith.constant 1024 : i32

      // Process 4 iterations
      scf.for %iter = %c0 to %c4 step %c1 {
        // Acquire input objectfifo
        %subviewIn = aie.objectfifo.acquire @inOF(Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
        %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

        // Acquire output objectfifo
        %subviewOut = aie.objectfifo.acquire @outOF(Produce, 1) : !aie.objectfifosubview<memref<1024xi32>>
        %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

        // Call the kernel function
        func.call @vector_multiply(%elemIn, %elemOut, %c1024) : (memref<1024xi32>, memref<1024xi32>, i32) -> ()

        // Release objectfifos
        aie.objectfifo.release @inOF(Consume, 1)
        aie.objectfifo.release @outOF(Produce, 1)
      }

      aie.end
    }

    // Runtime sequence for data movement
    aiex.runtime_sequence(%in : memref<4096xi32>, %dummy : memref<1xi32>, %out : memref<4096xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c4 = arith.constant 4 : i64
      %c1024 = arith.constant 1024 : i64

      // DMA memcpy for input: 4 iterations of 1024 elements
      // Format: dma_memcpy_nd (offset, length, stride)
      aiex.npu.dma_memcpy_nd (0, 0, %in[%c0, %c0, %c0, %c0][%c1, %c1, %c4, %c1024][%c0, %c0, %c1024, %c1])
        { metadata = @inOF, id = 1 : i64 } : memref<4096xi32>

      // DMA memcpy for output: 4 iterations of 1024 elements
      aiex.npu.dma_memcpy_nd (0, 0, %out[%c0, %c0, %c0, %c0][%c1, %c1, %c4, %c1024][%c0, %c0, %c1024, %c1])
        { metadata = @outOF, id = 0 : i64 } : memref<4096xi32>

      // Synchronization - wait for completion
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
