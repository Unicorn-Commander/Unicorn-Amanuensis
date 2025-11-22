module @softmax_multicolumn_combined_npu {
  aie.device(npu1) {
    %c6144_i64 = arith.constant 6144 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    func.func private @softmax_bf16(memref<2048xi8>, memref<2048xi8>)
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %of_output3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 6) {init = 0 : i32, sym_name = "of_output3_cons_prod_lock_0"}
    %of_output3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 7) {init = 0 : i32, sym_name = "of_output3_cons_cons_lock_0"}
    %of_output3_buff_0 = aie.buffer(%tile_1_3) {address = 1024 : i32, sym_name = "of_output3_buff_0"} : memref<2048xi8> 
    %of_output3_buff_1 = aie.buffer(%tile_1_3) {address = 3072 : i32, sym_name = "of_output3_buff_1"} : memref<2048xi8> 
    %of_output3_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 2 : i32, sym_name = "of_output3_prod_lock_0"}
    %of_output3_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "of_output3_cons_lock_0"}
    %of_output2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 4) {init = 0 : i32, sym_name = "of_output2_cons_prod_lock_0"}
    %of_output2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 5) {init = 0 : i32, sym_name = "of_output2_cons_cons_lock_0"}
    %of_output2_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, sym_name = "of_output2_buff_0"} : memref<2048xi8> 
    %of_output2_buff_1 = aie.buffer(%tile_1_2) {address = 3072 : i32, sym_name = "of_output2_buff_1"} : memref<2048xi8> 
    %of_output2_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "of_output2_prod_lock_0"}
    %of_output2_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "of_output2_cons_lock_0"}
    %of_input3_cons_buff_0 = aie.buffer(%tile_1_3) {address = 5120 : i32, sym_name = "of_input3_cons_buff_0"} : memref<2048xi8> 
    %of_input3_cons_buff_1 = aie.buffer(%tile_1_3) {address = 7168 : i32, sym_name = "of_input3_cons_buff_1"} : memref<2048xi8> 
    %of_input3_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "of_input3_cons_prod_lock_0"}
    %of_input3_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "of_input3_cons_cons_lock_0"}
    %of_input3_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "of_input3_prod_lock_0"}
    %of_input3_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "of_input3_cons_lock_0"}
    %of_input2_cons_buff_0 = aie.buffer(%tile_1_2) {address = 5120 : i32, sym_name = "of_input2_cons_buff_0"} : memref<2048xi8> 
    %of_input2_cons_buff_1 = aie.buffer(%tile_1_2) {address = 7168 : i32, sym_name = "of_input2_cons_buff_1"} : memref<2048xi8> 
    %of_input2_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "of_input2_cons_prod_lock_0"}
    %of_input2_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "of_input2_cons_cons_lock_0"}
    %of_input2_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "of_input2_prod_lock_0"}
    %of_input2_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "of_input2_cons_lock_0"}
    %of_output1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 6) {init = 0 : i32, sym_name = "of_output1_cons_prod_lock_0"}
    %of_output1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 7) {init = 0 : i32, sym_name = "of_output1_cons_cons_lock_0"}
    %of_output1_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, sym_name = "of_output1_buff_0"} : memref<2048xi8> 
    %of_output1_buff_1 = aie.buffer(%tile_0_3) {address = 3072 : i32, sym_name = "of_output1_buff_1"} : memref<2048xi8> 
    %of_output1_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "of_output1_prod_lock_0"}
    %of_output1_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "of_output1_cons_lock_0"}
    %of_output0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "of_output0_cons_prod_lock_0"}
    %of_output0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "of_output0_cons_cons_lock_0"}
    %of_output0_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "of_output0_buff_0"} : memref<2048xi8> 
    %of_output0_buff_1 = aie.buffer(%tile_0_2) {address = 3072 : i32, sym_name = "of_output0_buff_1"} : memref<2048xi8> 
    %of_output0_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "of_output0_prod_lock_0"}
    %of_output0_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "of_output0_cons_lock_0"}
    %of_input1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 5120 : i32, sym_name = "of_input1_cons_buff_0"} : memref<2048xi8> 
    %of_input1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 7168 : i32, sym_name = "of_input1_cons_buff_1"} : memref<2048xi8> 
    %of_input1_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "of_input1_cons_prod_lock_0"}
    %of_input1_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "of_input1_cons_cons_lock_0"}
    %of_input1_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "of_input1_prod_lock_0"}
    %of_input1_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "of_input1_cons_lock_0"}
    %of_input0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 5120 : i32, sym_name = "of_input0_cons_buff_0"} : memref<2048xi8> 
    %of_input0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 7168 : i32, sym_name = "of_input0_cons_buff_1"} : memref<2048xi8> 
    %of_input0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "of_input0_cons_prod_lock_0"}
    %of_input0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "of_input0_cons_cons_lock_0"}
    %of_input0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "of_input0_prod_lock_0"}
    %of_input0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "of_input0_cons_lock_0"}
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {elf_file = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1/build_softmax_multicolumn_fixed/softmax_multicolumn_combined.mlir.prj/main_core_0_2.elf", link_with = "softmax_bf16_xdna1.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      aie.end
    } {elf_file = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1/build_softmax_multicolumn_fixed/softmax_multicolumn_combined.mlir.prj/main_core_0_3.elf", link_with = "softmax_bf16_xdna1.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      aie.end
    } {elf_file = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1/build_softmax_multicolumn_fixed/softmax_multicolumn_combined.mlir.prj/main_core_1_2.elf", link_with = "softmax_bf16_xdna1.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      aie.end
    } {elf_file = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1/build_softmax_multicolumn_fixed/softmax_multicolumn_combined.mlir.prj/main_core_1_3.elf", link_with = "softmax_bf16_xdna1.o"}
    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[512, 0, 0, 0, -2147483648, 33554432, 0, 33554432]>
    aiex.runtime_sequence(%arg0: memref<8192xi8>, %arg1: memref<8192xi8>) {
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119316 : ui32, value = 0 : ui32}
      %1 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%1) {address = 118816 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 0 : i32, arg_plus = 2048 : i32}
      aiex.npu.write32 {address = 119324 : ui32, value = 1 : ui32}
      %2 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%2) {address = 33673280 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 33673284 : ui32, arg_idx = 0 : i32, arg_plus = 4096 : i32}
      aiex.npu.write32 {address = 33673748 : ui32, value = 2 : ui32}
      %3 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%3) {address = 33673312 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 33673316 : ui32, arg_idx = 0 : i32, arg_plus = 6144 : i32}
      aiex.npu.write32 {address = 33673756 : ui32, value = 3 : ui32}
      %4 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%4) {address = 118912 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118916 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 119296 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, value = 2147483652 : ui32}
      %5 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%5) {address = 118944 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118948 : ui32, arg_idx = 1 : i32, arg_plus = 2048 : i32}
      aiex.npu.maskwrite32 {address = 119304 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119308 : ui32, value = 2147483653 : ui32}
      %6 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%6) {address = 33673408 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 33673412 : ui32, arg_idx = 1 : i32, arg_plus = 4096 : i32}
      aiex.npu.maskwrite32 {address = 33673728 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 33673732 : ui32, value = 2147483654 : ui32}
      %7 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%7) {address = 33673440 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 33673444 : ui32, arg_idx = 1 : i32, arg_plus = 6144 : i32}
      aiex.npu.maskwrite32 {address = 33673736 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 33673740 : ui32, value = 2147483655 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 1 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    aie.shim_dma_allocation @of_input0_shim_alloc(MM2S, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_input0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input0_cons_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_input0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_input0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input0_cons_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_input0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_output0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output0_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%of_output0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_output0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output0_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%of_output0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @of_input1_shim_alloc(MM2S, 1, 0)
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_input1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input1_cons_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_input1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_input1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input1_cons_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_input1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_output1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output1_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%of_output1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_output1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output1_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%of_output1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @of_output0_shim_alloc(S2MM, 0, 0)
    aie.shim_dma_allocation @of_output1_shim_alloc(S2MM, 1, 0)
    aie.shim_dma_allocation @of_input2_shim_alloc(MM2S, 0, 1)
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_input2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input2_cons_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_input2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_input2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input2_cons_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_input2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_output2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output2_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%of_output2_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_output2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output2_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%of_output2_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @of_input3_shim_alloc(MM2S, 1, 1)
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_input3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input3_cons_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_input3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_input3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input3_cons_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_input3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_output3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output3_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%of_output3_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_output3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output3_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%of_output3_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @of_output2_shim_alloc(S2MM, 0, 1)
    aie.shim_dma_allocation @of_output3_shim_alloc(S2MM, 1, 1)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, North : 2>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 2, North : 2>
      aie.connect<North : 0, South : 0>
      aie.connect<North : 1, South : 1>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 2, North : 0>
      aie.connect<DMA : 0, South : 0>
      aie.connect<North : 1, South : 1>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<DMA : 0, South : 1>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, North : 5>
      aie.connect<North : 2, South : 2>
      aie.connect<North : 0, South : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 5, North : 5>
      aie.connect<North : 2, South : 2>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 5, North : 5>
      aie.connect<DMA : 0, South : 2>
      aie.connect<North : 1, South : 0>
    }
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<South : 5, DMA : 0>
      aie.connect<DMA : 0, South : 1>
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
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
    aie.wire(%switchbox_0_3 : East, %switchbox_1_3 : West)
    aie.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
    aie.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
    aie.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
  }
}
