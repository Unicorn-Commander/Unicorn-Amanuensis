module {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    aie.objectfifo @of_input_0(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<12288xi8>> 
    aie.objectfifo @of_input_1(%shim_noc_tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<12288xi8>> 
    aie.objectfifo @of_input_2(%shim_noc_tile_2_0, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<12288xi8>> 
    aie.objectfifo @of_input_3(%shim_noc_tile_3_0, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<12288xi8>> 
    aie.objectfifo @of_output_0(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<4096xi8>> 
    aie.objectfifo @of_output_1(%tile_1_2, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<4096xi8>> 
    aie.objectfifo @of_output_2(%tile_2_2, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<4096xi8>> 
    aie.objectfifo @of_output_3(%tile_3_2, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<4096xi8>> 
    func.func private @attention_64x64(memref<12288xi8>, memref<4096xi8>, i32)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c1_2 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c1_1 step %c1_2 {
          %0 = aie.objectfifo.acquire @of_input_0(Consume, 1) : !aie.objectfifosubview<memref<12288xi8>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<12288xi8>> -> memref<12288xi8>
          %2 = aie.objectfifo.acquire @of_output_0(Produce, 1) : !aie.objectfifosubview<memref<4096xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
          %c3_i32 = arith.constant 3 : i32
          func.call @attention_64x64(%1, %3, %c3_i32) : (memref<12288xi8>, memref<4096xi8>, i32) -> ()
          aie.objectfifo.release @of_input_0(Consume, 1)
          aie.objectfifo.release @of_output_0(Produce, 1)
        }
      }
      aie.end
    } {link_with = "attention_int8_64x64_tiled.o", stack_size = 16384 : i32}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c1_2 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c1_1 step %c1_2 {
          %0 = aie.objectfifo.acquire @of_input_1(Consume, 1) : !aie.objectfifosubview<memref<12288xi8>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<12288xi8>> -> memref<12288xi8>
          %2 = aie.objectfifo.acquire @of_output_1(Produce, 1) : !aie.objectfifosubview<memref<4096xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
          %c3_i32 = arith.constant 3 : i32
          func.call @attention_64x64(%1, %3, %c3_i32) : (memref<12288xi8>, memref<4096xi8>, i32) -> ()
          aie.objectfifo.release @of_input_1(Consume, 1)
          aie.objectfifo.release @of_output_1(Produce, 1)
        }
      }
      aie.end
    } {link_with = "attention_int8_64x64_tiled.o", stack_size = 16384 : i32}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c1_2 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c1_1 step %c1_2 {
          %0 = aie.objectfifo.acquire @of_input_2(Consume, 1) : !aie.objectfifosubview<memref<12288xi8>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<12288xi8>> -> memref<12288xi8>
          %2 = aie.objectfifo.acquire @of_output_2(Produce, 1) : !aie.objectfifosubview<memref<4096xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
          %c3_i32 = arith.constant 3 : i32
          func.call @attention_64x64(%1, %3, %c3_i32) : (memref<12288xi8>, memref<4096xi8>, i32) -> ()
          aie.objectfifo.release @of_input_2(Consume, 1)
          aie.objectfifo.release @of_output_2(Produce, 1)
        }
      }
      aie.end
    } {link_with = "attention_int8_64x64_tiled.o", stack_size = 16384 : i32}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c1_2 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c1_1 step %c1_2 {
          %0 = aie.objectfifo.acquire @of_input_3(Consume, 1) : !aie.objectfifosubview<memref<12288xi8>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<12288xi8>> -> memref<12288xi8>
          %2 = aie.objectfifo.acquire @of_output_3(Produce, 1) : !aie.objectfifosubview<memref<4096xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
          %c3_i32 = arith.constant 3 : i32
          func.call @attention_64x64(%1, %3, %c3_i32) : (memref<12288xi8>, memref<4096xi8>, i32) -> ()
          aie.objectfifo.release @of_input_3(Consume, 1)
          aie.objectfifo.release @of_output_3(Produce, 1)
        }
      }
      aie.end
    } {link_with = "attention_int8_64x64_tiled.o", stack_size = 16384 : i32}
    aiex.runtime_sequence(%arg0: memref<12288xi8>, %arg1: memref<12288xi8>, %arg2: memref<12288xi8>, %arg3: memref<12288xi8>, %arg4: memref<4096xi8>, %arg5: memref<4096xi8>, %arg6: memref<4096xi8>, %arg7: memref<4096xi8>) {
      // Configure all input DMA tasks (parallel execution)
      %0 = aiex.dma_configure_task_for @of_input_0 {
        aie.dma_bd(%arg0 : memref<12288xi8>, 0, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 12288, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %2 = aiex.dma_configure_task_for @of_input_1 {
        aie.dma_bd(%arg1 : memref<12288xi8>, 0, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 12288, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %4 = aiex.dma_configure_task_for @of_input_2 {
        aie.dma_bd(%arg2 : memref<12288xi8>, 0, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 12288, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %6 = aiex.dma_configure_task_for @of_input_3 {
        aie.dma_bd(%arg3 : memref<12288xi8>, 0, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 12288, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }

      // Configure all output DMA tasks (parallel execution)
      %1 = aiex.dma_configure_task_for @of_output_0 {
        aie.dma_bd(%arg4 : memref<4096xi8>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      %3 = aiex.dma_configure_task_for @of_output_1 {
        aie.dma_bd(%arg5 : memref<4096xi8>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      %5 = aiex.dma_configure_task_for @of_output_2 {
        aie.dma_bd(%arg6 : memref<4096xi8>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      %7 = aiex.dma_configure_task_for @of_output_3 {
        aie.dma_bd(%arg7 : memref<4096xi8>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}

      // Start all input DMAs in parallel (no waits between them!)
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%2)
      aiex.dma_start_task(%4)
      aiex.dma_start_task(%6)

      // Start all output DMAs in parallel (no waits between them!)
      aiex.dma_start_task(%1)
      aiex.dma_start_task(%3)
      aiex.dma_start_task(%5)
      aiex.dma_start_task(%7)

      // Wait for all outputs to complete (cores run in parallel)
      aiex.dma_await_task(%1)
      aiex.dma_await_task(%3)
      aiex.dma_await_task(%5)
      aiex.dma_await_task(%7)

      // Free all input tasks
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%6)
    }
  }
}
