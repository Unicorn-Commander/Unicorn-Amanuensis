// Complete passthrough kernel for Phoenix NPU (npu1_1col)
// Based on Xilinx mlir-aie working examples
// Device: AMD Ryzen AI Phoenix NPU

module @passthrough_complete {
  aie.device(npu1) {
    // Tile declarations
    %tile_0_0 = aie.tile(0, 0)  // Shim tile - DMA only
    %tile_0_2 = aie.tile(0, 2)  // Compute tile

    // External kernel function
    func.func private @passthrough_kernel(
      memref<1024xui8>,
      memref<1024xui8>,
      i32
    )

    // ObjectFIFOs for data movement (depth=2 for ping-pong)
    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xui8>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xui8>>

    // Compute core program
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : i32

      // Single iteration for testing
      scf.for %iter = %c0 to %c1 step %c1 {
        // Acquire input
        %subview_in = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<1024xui8>>
        %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>

        // Acquire output
        %subview_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
        %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>

        // Call kernel
        func.call @passthrough_kernel(%elem_in, %elem_out, %c1024) : (memref<1024xui8>, memref<1024xui8>, i32) -> ()

        // Release buffers
        aie.objectfifo.release @of_in(Consume, 1)
        aie.objectfifo.release @of_out(Produce, 1)
      }

      aie.end
    }

    // Runtime sequence for host-to-NPU data movement
    aiex.runtime_sequence @sequence(%arg_in: memref<1024xi32>, %arg_out: memref<1024xi32>) {
      // Input DMA: transfer 256 i32s (1024 bytes) from host to tile
      aiex.npu.dma_memcpy_nd(%arg_in[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @of_in} : memref<1024xi32>

      // Output DMA: transfer 256 i32s (1024 bytes) from tile to host
      aiex.npu.dma_memcpy_nd(%arg_out[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @of_out} : memref<1024xi32>

      // Wait for completion
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
