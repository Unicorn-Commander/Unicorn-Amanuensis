module {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)

    // Input ObjectFIFO: 12288 bytes (Q+K+V for 64×64)
    aie.objectfifo @of_input(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<12288xi8>>

    // Output ObjectFIFO: 4096 bytes (64×64 result)
    aie.objectfifo @of_output(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>

    // Declare attention kernel
    func.func private @attention_64x64(memref<12288xi8>, memref<4096xi8>, i32)

    // Core with infinite loop (like mel)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index

      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c1_2 = arith.constant 1 : index

        scf.for %arg1 = %c0_0 to %c1_1 step %c1_2 {
          // Acquire input
          %0 = aie.objectfifo.acquire @of_input(Consume, 1) : !aie.objectfifosubview<memref<12288xi8>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<12288xi8>> -> memref<12288xi8>

          // Acquire output
          %2 = aie.objectfifo.acquire @of_output(Produce, 1) : !aie.objectfifosubview<memref<4096xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>

          // Call kernel
          %c3_i32 = arith.constant 3 : i32
          func.call @attention_64x64(%1, %3, %c3_i32) : (memref<12288xi8>, memref<4096xi8>, i32) -> ()

          // Release
          aie.objectfifo.release @of_input(Consume, 1)
          aie.objectfifo.release @of_output(Produce, 1)
        }
      }
      aie.end
    } {link_with = "attention_int8_64x64_tiled.o", stack_size = 16384 : i32}

    // Runtime sequence with NEW NPU DMA API (like mel)
    aiex.runtime_sequence(%in : memref<12288xi8>, %out : memref<4096xi8>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c12288_i64 = arith.constant 12288 : i64
      %c4096_i64 = arith.constant 4096 : i64

      // DMA from host to NPU (input)
      aiex.npu.dma_memcpy_nd(%in[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                               [%c1_i64, %c1_i64, %c1_i64, %c12288_i64]
                               [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
          metadata = @of_input,
          id = 1 : i64
      } : memref<12288xi8>

      // DMA from NPU to host (output)
      aiex.npu.dma_memcpy_nd(%out[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                [%c1_i64, %c1_i64, %c1_i64, %c4096_i64]
                                [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
          metadata = @of_output,
          id = 0 : i64
      } : memref<4096xi8>

      // Wait for output DMA completion
      aiex.npu.dma_wait {symbol = @of_output}
    }
  }
}
