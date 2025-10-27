module @passthrough_test {
  aie.device(npu1_1col) {
    // Declare memory tile (0,1) and compute tile (0,2)
    %tile_0_0 = aie.tile(0, 0)  // Shim tile for DMA
    %tile_0_2 = aie.tile(0, 2)  // Compute tile

    // External kernel function declaration
    func.func private @passthrough_kernel(
      memref<1024xui8>,
      memref<1024xui8>,
      i32
    )

    // ObjectFIFOs for data movement
    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xui8>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xui8>>

    // Core program on compute tile
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1024 : i32

      // Single iteration for testing
      scf.for %iter = %c0 to %c1 step %c1 {
        // Acquire input buffer
        %subview_in = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<1024xui8>>
        %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>

        // Acquire output buffer
        %subview_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
        %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>

        // Call kernel
        func.call @passthrough_kernel(%elem_in, %elem_out, %c1_i32) : (memref<1024xui8>, memref<1024xui8>, i32) -> ()

        // Release buffers
        aie.objectfifo.release @of_in(Consume, 1)
        aie.objectfifo.release @of_out(Produce, 1)
      }

      aie.end
    }
  }
}
