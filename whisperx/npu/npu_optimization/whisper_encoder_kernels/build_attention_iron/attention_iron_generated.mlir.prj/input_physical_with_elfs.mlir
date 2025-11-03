module {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_2_2 = aie.tile(2, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_3_2 = aie.tile(3, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %of_output_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "of_output_3_cons_prod_lock_0"}
    %of_output_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "of_output_3_cons_cons_lock_0"}
    %of_output_3_buff_0 = aie.buffer(%tile_3_2) {address = 40960 : i32, sym_name = "of_output_3_buff_0"} : memref<4096xi8> 
    %of_output_3_buff_1 = aie.buffer(%tile_3_2) {address = 45056 : i32, sym_name = "of_output_3_buff_1"} : memref<4096xi8> 
    %of_output_3_prod_lock_0 = aie.lock(%tile_3_2, 2) {init = 2 : i32, sym_name = "of_output_3_prod_lock_0"}
    %of_output_3_cons_lock_0 = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "of_output_3_cons_lock_0"}
    %of_output_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "of_output_2_cons_prod_lock_0"}
    %of_output_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "of_output_2_cons_cons_lock_0"}
    %of_output_2_buff_0 = aie.buffer(%tile_2_2) {address = 40960 : i32, sym_name = "of_output_2_buff_0"} : memref<4096xi8> 
    %of_output_2_buff_1 = aie.buffer(%tile_2_2) {address = 45056 : i32, sym_name = "of_output_2_buff_1"} : memref<4096xi8> 
    %of_output_2_prod_lock_0 = aie.lock(%tile_2_2, 2) {init = 2 : i32, sym_name = "of_output_2_prod_lock_0"}
    %of_output_2_cons_lock_0 = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "of_output_2_cons_lock_0"}
    %of_output_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "of_output_1_cons_prod_lock_0"}
    %of_output_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "of_output_1_cons_cons_lock_0"}
    %of_output_1_buff_0 = aie.buffer(%tile_1_2) {address = 40960 : i32, sym_name = "of_output_1_buff_0"} : memref<4096xi8> 
    %of_output_1_buff_1 = aie.buffer(%tile_1_2) {address = 45056 : i32, sym_name = "of_output_1_buff_1"} : memref<4096xi8> 
    %of_output_1_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "of_output_1_prod_lock_0"}
    %of_output_1_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "of_output_1_cons_lock_0"}
    %of_output_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "of_output_0_cons_prod_lock_0"}
    %of_output_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "of_output_0_cons_cons_lock_0"}
    %of_output_0_buff_0 = aie.buffer(%tile_0_2) {address = 40960 : i32, sym_name = "of_output_0_buff_0"} : memref<4096xi8> 
    %of_output_0_buff_1 = aie.buffer(%tile_0_2) {address = 45056 : i32, sym_name = "of_output_0_buff_1"} : memref<4096xi8> 
    %of_output_0_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "of_output_0_prod_lock_0"}
    %of_output_0_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "of_output_0_cons_lock_0"}
    %of_input_3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 16384 : i32, sym_name = "of_input_3_cons_buff_0"} : memref<12288xi8> 
    %of_input_3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 28672 : i32, sym_name = "of_input_3_cons_buff_1"} : memref<12288xi8> 
    %of_input_3_cons_prod_lock_0 = aie.lock(%tile_3_2, 0) {init = 2 : i32, sym_name = "of_input_3_cons_prod_lock_0"}
    %of_input_3_cons_cons_lock_0 = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "of_input_3_cons_cons_lock_0"}
    %of_input_3_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "of_input_3_prod_lock_0"}
    %of_input_3_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "of_input_3_cons_lock_0"}
    %of_input_2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 16384 : i32, sym_name = "of_input_2_cons_buff_0"} : memref<12288xi8> 
    %of_input_2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 28672 : i32, sym_name = "of_input_2_cons_buff_1"} : memref<12288xi8> 
    %of_input_2_cons_prod_lock_0 = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "of_input_2_cons_prod_lock_0"}
    %of_input_2_cons_cons_lock_0 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "of_input_2_cons_cons_lock_0"}
    %of_input_2_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "of_input_2_prod_lock_0"}
    %of_input_2_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "of_input_2_cons_lock_0"}
    %of_input_1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 16384 : i32, sym_name = "of_input_1_cons_buff_0"} : memref<12288xi8> 
    %of_input_1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 28672 : i32, sym_name = "of_input_1_cons_buff_1"} : memref<12288xi8> 
    %of_input_1_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "of_input_1_cons_prod_lock_0"}
    %of_input_1_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "of_input_1_cons_cons_lock_0"}
    %of_input_1_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "of_input_1_prod_lock_0"}
    %of_input_1_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "of_input_1_cons_lock_0"}
    %of_input_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 16384 : i32, sym_name = "of_input_0_cons_buff_0"} : memref<12288xi8> 
    %of_input_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 28672 : i32, sym_name = "of_input_0_cons_buff_1"} : memref<12288xi8> 
    %of_input_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "of_input_0_cons_prod_lock_0"}
    %of_input_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "of_input_0_cons_cons_lock_0"}
    %of_input_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "of_input_0_prod_lock_0"}
    %of_input_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "of_input_0_cons_lock_0"}
    func.func private @attention_64x64(memref<12288xi8>, memref<4096xi8>, i32)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {elf_file = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/build_attention_iron/attention_iron_generated.mlir.prj/main_core_0_2.elf", link_with = "attention_int8_64x64_tiled.o", stack_size = 16384 : i32}
    %core_1_2 = aie.core(%tile_1_2) {
      aie.end
    } {elf_file = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/build_attention_iron/attention_iron_generated.mlir.prj/main_core_1_2.elf", link_with = "attention_int8_64x64_tiled.o", stack_size = 16384 : i32}
    %core_2_2 = aie.core(%tile_2_2) {
      aie.end
    } {elf_file = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/build_attention_iron/attention_iron_generated.mlir.prj/main_core_2_2.elf", link_with = "attention_int8_64x64_tiled.o", stack_size = 16384 : i32}
    %core_3_2 = aie.core(%tile_3_2) {
      aie.end
    } {elf_file = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/build_attention_iron/attention_iron_generated.mlir.prj/main_core_3_2.elf", link_with = "attention_int8_64x64_tiled.o", stack_size = 16384 : i32}
    aiex.runtime_sequence(%arg0: memref<12288xi8>, %arg1: memref<12288xi8>, %arg2: memref<12288xi8>, %arg3: memref<12288xi8>, %arg4: memref<4096xi8>, %arg5: memref<4096xi8>, %arg6: memref<4096xi8>, %arg7: memref<4096xi8>) {
      %0 = aiex.dma_configure_task_for @of_input_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<12288xi8>, 0, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 12288, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %1 = aiex.dma_configure_task_for @of_input_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<12288xi8>, 0, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 12288, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %2 = aiex.dma_configure_task_for @of_input_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<12288xi8>, 0, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 12288, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %3 = aiex.dma_configure_task_for @of_input_3_shim_alloc {
        aie.dma_bd(%arg3 : memref<12288xi8>, 0, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 12288, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %4 = aiex.dma_configure_task_for @of_output_0_shim_alloc {
        aie.dma_bd(%arg4 : memref<4096xi8>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      %5 = aiex.dma_configure_task_for @of_output_1_shim_alloc {
        aie.dma_bd(%arg5 : memref<4096xi8>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      %6 = aiex.dma_configure_task_for @of_output_2_shim_alloc {
        aie.dma_bd(%arg6 : memref<4096xi8>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      %7 = aiex.dma_configure_task_for @of_output_3_shim_alloc {
        aie.dma_bd(%arg7 : memref<4096xi8>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_start_task(%2)
      aiex.dma_start_task(%3)
      aiex.dma_start_task(%4)
      aiex.dma_start_task(%5)
      aiex.dma_start_task(%6)
      aiex.dma_start_task(%7)
      aiex.dma_await_task(%4)
      aiex.dma_await_task(%5)
      aiex.dma_await_task(%6)
      aiex.dma_await_task(%7)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
    }
    aie.shim_dma_allocation @of_input_0_shim_alloc(MM2S, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_input_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_0_cons_buff_0 : memref<12288xi8>, 0, 12288) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_input_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_input_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_0_cons_buff_1 : memref<12288xi8>, 0, 12288) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_input_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_output_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output_0_buff_0 : memref<4096xi8>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%of_output_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_output_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output_0_buff_1 : memref<4096xi8>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%of_output_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @of_input_1_shim_alloc(MM2S, 0, 1)
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_input_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_1_cons_buff_0 : memref<12288xi8>, 0, 12288) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_input_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_input_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_1_cons_buff_1 : memref<12288xi8>, 0, 12288) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_input_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_output_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output_1_buff_0 : memref<4096xi8>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%of_output_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_output_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output_1_buff_1 : memref<4096xi8>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%of_output_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @of_input_2_shim_alloc(MM2S, 0, 2)
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_input_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_2_cons_buff_0 : memref<12288xi8>, 0, 12288) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_input_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_input_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_2_cons_buff_1 : memref<12288xi8>, 0, 12288) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_input_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_output_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output_2_buff_0 : memref<4096xi8>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%of_output_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_output_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output_2_buff_1 : memref<4096xi8>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%of_output_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @of_input_3_shim_alloc(MM2S, 0, 3)
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_input_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_3_cons_buff_0 : memref<12288xi8>, 0, 12288) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_input_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_input_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_3_cons_buff_1 : memref<12288xi8>, 0, 12288) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_input_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_output_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output_3_buff_0 : memref<4096xi8>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%of_output_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_output_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output_3_buff_1 : memref<4096xi8>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%of_output_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @of_output_0_shim_alloc(S2MM, 0, 0)
    aie.shim_dma_allocation @of_output_1_shim_alloc(S2MM, 0, 1)
    aie.shim_dma_allocation @of_output_2_shim_alloc(S2MM, 0, 2)
    aie.shim_dma_allocation @of_output_3_shim_alloc(S2MM, 0, 3)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 0, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<North : 2, South : 2>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, South : 2>
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_2_0 = aie.shim_mux(%shim_noc_tile_2_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %mem_tile_2_1 = aie.tile(2, 1)
    %switchbox_2_1 = aie.switchbox(%mem_tile_2_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<North : 2, South : 2>
    }
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, South : 2>
    }
    %switchbox_3_0 = aie.switchbox(%shim_noc_tile_3_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<North : 0, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_3_0 = aie.shim_mux(%shim_noc_tile_3_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %mem_tile_3_1 = aie.tile(3, 1)
    %switchbox_3_1 = aie.switchbox(%mem_tile_3_1) {
      aie.connect<South : 0, North : 0>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
    aie.wire(%switchbox_1_0 : East, %switchbox_2_0 : West)
    aie.wire(%shim_mux_2_0 : North, %switchbox_2_0 : South)
    aie.wire(%shim_noc_tile_2_0 : DMA, %shim_mux_2_0 : DMA)
    aie.wire(%switchbox_1_1 : East, %switchbox_2_1 : West)
    aie.wire(%mem_tile_2_1 : Core, %switchbox_2_1 : Core)
    aie.wire(%mem_tile_2_1 : DMA, %switchbox_2_1 : DMA)
    aie.wire(%switchbox_2_0 : North, %switchbox_2_1 : South)
    aie.wire(%switchbox_1_2 : East, %switchbox_2_2 : West)
    aie.wire(%tile_2_2 : Core, %switchbox_2_2 : Core)
    aie.wire(%tile_2_2 : DMA, %switchbox_2_2 : DMA)
    aie.wire(%switchbox_2_1 : North, %switchbox_2_2 : South)
    aie.wire(%switchbox_2_0 : East, %switchbox_3_0 : West)
    aie.wire(%shim_mux_3_0 : North, %switchbox_3_0 : South)
    aie.wire(%shim_noc_tile_3_0 : DMA, %shim_mux_3_0 : DMA)
    aie.wire(%switchbox_2_1 : East, %switchbox_3_1 : West)
    aie.wire(%mem_tile_3_1 : Core, %switchbox_3_1 : Core)
    aie.wire(%mem_tile_3_1 : DMA, %switchbox_3_1 : DMA)
    aie.wire(%switchbox_3_0 : North, %switchbox_3_1 : South)
    aie.wire(%switchbox_2_2 : East, %switchbox_3_2 : West)
    aie.wire(%tile_3_2 : Core, %switchbox_3_2 : Core)
    aie.wire(%tile_3_2 : DMA, %switchbox_3_2 : DMA)
    aie.wire(%switchbox_3_1 : North, %switchbox_3_2 : South)
  }
}
