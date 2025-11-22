module @encoder_layer_simple_npu {
  aie.device(npu1) {
    %c2048_i64 = arith.constant 2048 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    func.func private @layernorm_simple_bf16(memref<2048xi8>, memref<2048xi8>)
    func.func private @softmax_bf16(memref<2048xi8>, memref<2048xi8>)
    func.func private @gelu_bf16(memref<2048xi8>, memref<2048xi8>)
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %of_gelu_to_output_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "of_gelu_to_output_cons_prod_lock_0"}
    %of_gelu_to_output_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "of_gelu_to_output_cons_cons_lock_0"}
    %of_gelu_to_output_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, sym_name = "of_gelu_to_output_buff_0"} : memref<2048xi8> 
    %of_gelu_to_output_buff_1 = aie.buffer(%tile_0_4) {address = 3072 : i32, sym_name = "of_gelu_to_output_buff_1"} : memref<2048xi8> 
    %of_gelu_to_output_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "of_gelu_to_output_prod_lock_0"}
    %of_gelu_to_output_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "of_gelu_to_output_cons_lock_0"}
    %of_softmax_to_gelu_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, sym_name = "of_softmax_to_gelu_buff_0"} : memref<2048xi8> 
    %of_softmax_to_gelu_buff_1 = aie.buffer(%tile_0_3) {address = 3072 : i32, sym_name = "of_softmax_to_gelu_buff_1"} : memref<2048xi8> 
    %of_softmax_to_gelu_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "of_softmax_to_gelu_prod_lock_0"}
    %of_softmax_to_gelu_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "of_softmax_to_gelu_cons_lock_0"}
    %of_ln_to_softmax_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "of_ln_to_softmax_buff_0"} : memref<2048xi8> 
    %of_ln_to_softmax_buff_1 = aie.buffer(%tile_0_2) {address = 3072 : i32, sym_name = "of_ln_to_softmax_buff_1"} : memref<2048xi8> 
    %of_ln_to_softmax_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "of_ln_to_softmax_prod_lock_0"}
    %of_ln_to_softmax_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "of_ln_to_softmax_cons_lock_0"}
    %of_input_to_ln_cons_buff_0 = aie.buffer(%tile_0_2) {address = 5120 : i32, sym_name = "of_input_to_ln_cons_buff_0"} : memref<2048xi8> 
    %of_input_to_ln_cons_buff_1 = aie.buffer(%tile_0_2) {address = 7168 : i32, sym_name = "of_input_to_ln_cons_buff_1"} : memref<2048xi8> 
    %of_input_to_ln_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "of_input_to_ln_cons_prod_lock_0"}
    %of_input_to_ln_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "of_input_to_ln_cons_cons_lock_0"}
    %of_input_to_ln_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "of_input_to_ln_prod_lock_0"}
    %of_input_to_ln_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "of_input_to_ln_cons_lock_0"}
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {elf_file = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1/build_encoder_simple/encoder_layer_simple.mlir.prj/main_core_0_2.elf", link_with = "layernorm_bf16_xdna1.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      aie.end
    } {elf_file = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1/build_encoder_simple/encoder_layer_simple.mlir.prj/main_core_0_3.elf", link_with = "softmax_bf16_xdna1.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      aie.end
    } {elf_file = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1/build_encoder_simple/encoder_layer_simple.mlir.prj/main_core_0_4.elf", link_with = "gelu_simple_xdna1.o"}
    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[512, 0, 0, 0, -2147483648, 33554432, 0, 33554432]>
    aiex.runtime_sequence(%arg0: memref<2048xi8>, %arg1: memref<2048xi8>) {
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 118816 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119316 : ui32, value = 1 : ui32}
      %1 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%1) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 119296 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    aie.shim_dma_allocation @of_input_to_ln_shim_alloc(MM2S, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_input_to_ln_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_to_ln_cons_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_input_to_ln_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_input_to_ln_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_to_ln_cons_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_input_to_ln_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_gelu_to_output_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_gelu_to_output_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_gelu_to_output_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_gelu_to_output_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_gelu_to_output_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_gelu_to_output_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    aie.shim_dma_allocation @of_gelu_to_output_shim_alloc(S2MM, 0, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 3, South : 2>
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
      aie.connect<North : 3, South : 3>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<North : 1, South : 3>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<North : 1, South : 1>
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
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
    aie.wire(%tile_0_4 : Core, %switchbox_0_4 : Core)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
  }
}
