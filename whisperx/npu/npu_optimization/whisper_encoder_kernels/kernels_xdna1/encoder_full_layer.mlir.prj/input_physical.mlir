module @whisper_encoder_full_layer {
  aie.device(npu1) {
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %switchbox_3_0 = aie.switchbox(%shim_noc_tile_3_0) {
    }
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %c2048_i64 = arith.constant 2048 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    func.func private @layernorm_bf16(memref<2048xi8>, memref<2048xi8>)
    func.func private @softmax_bf16(memref<2048xi8>, memref<2048xi8>)
    func.func private @gelu_bf16(memref<2048xi8>, memref<2048xi8>)
    func.func private @matmul_bf16_64x64(memref<8192xi8>, memref<8192xi8>, memref<8192xi8>)
    func.func private @vector_add_bf16(memref<2048xi8>, memref<2048xi8>, memref<2048xi8>)
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_2_2 = aie.tile(2, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_2_3 = aie.tile(2, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_2_4 = aie.tile(2, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_3_2 = aie.tile(3, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_3_3 = aie.tile(3, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %of_output_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "of_output_cons_prod_lock_0"}
    %of_output_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "of_output_cons_cons_lock_0"}
    %of_output_buff_0 = aie.buffer(%tile_3_3) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "of_output_buff_0"} : memref<2048xi8> 
    %of_output_buff_1 = aie.buffer(%tile_3_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "of_output_buff_1"} : memref<2048xi8> 
    %of_output_prod_lock_0 = aie.lock(%tile_3_3, 0) {init = 2 : i32, sym_name = "of_output_prod_lock_0"}
    %of_output_cons_lock_0 = aie.lock(%tile_3_3, 1) {init = 0 : i32, sym_name = "of_output_cons_lock_0"}
    %of_fc2_buff_0 = aie.buffer(%tile_3_2) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "of_fc2_buff_0"} : memref<2048xi8> 
    %of_fc2_buff_1 = aie.buffer(%tile_3_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "of_fc2_buff_1"} : memref<2048xi8> 
    %of_fc2_prod_lock_0 = aie.lock(%tile_3_2, 2) {init = 2 : i32, sym_name = "of_fc2_prod_lock_0"}
    %of_fc2_cons_lock_0 = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "of_fc2_cons_lock_0"}
    %of_gelu_cons_buff_0 = aie.buffer(%tile_3_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "of_gelu_cons_buff_0"} : memref<2048xi8> 
    %of_gelu_cons_buff_1 = aie.buffer(%tile_3_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "of_gelu_cons_buff_1"} : memref<2048xi8> 
    %of_gelu_cons_prod_lock_0 = aie.lock(%tile_3_2, 0) {init = 2 : i32, sym_name = "of_gelu_cons_prod_lock_0"}
    %of_gelu_cons_cons_lock_0 = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "of_gelu_cons_cons_lock_0"}
    %of_gelu_buff_0 = aie.buffer(%tile_2_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "of_gelu_buff_0"} : memref<2048xi8> 
    %of_gelu_buff_1 = aie.buffer(%tile_2_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "of_gelu_buff_1"} : memref<2048xi8> 
    %of_gelu_prod_lock_0 = aie.lock(%tile_2_4, 0) {init = 2 : i32, sym_name = "of_gelu_prod_lock_0"}
    %of_gelu_cons_lock_0 = aie.lock(%tile_2_4, 1) {init = 0 : i32, sym_name = "of_gelu_cons_lock_0"}
    %of_fc1_buff_0 = aie.buffer(%tile_2_3) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "of_fc1_buff_0"} : memref<2048xi8> 
    %of_fc1_buff_1 = aie.buffer(%tile_2_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "of_fc1_buff_1"} : memref<2048xi8> 
    %of_fc1_prod_lock_0 = aie.lock(%tile_2_3, 0) {init = 2 : i32, sym_name = "of_fc1_prod_lock_0"}
    %of_fc1_cons_lock_0 = aie.lock(%tile_2_3, 1) {init = 0 : i32, sym_name = "of_fc1_cons_lock_0"}
    %of_ln2_buff_0 = aie.buffer(%tile_2_2) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "of_ln2_buff_0"} : memref<2048xi8> 
    %of_ln2_buff_1 = aie.buffer(%tile_2_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "of_ln2_buff_1"} : memref<2048xi8> 
    %of_ln2_prod_lock_0 = aie.lock(%tile_2_2, 4) {init = 2 : i32, sym_name = "of_ln2_prod_lock_0"}
    %of_ln2_cons_lock_0 = aie.lock(%tile_2_2, 5) {init = 0 : i32, sym_name = "of_ln2_cons_lock_0"}
    %of_residual1_cons_buff_0 = aie.buffer(%tile_2_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "of_residual1_cons_buff_0"} : memref<2048xi8> 
    %of_residual1_cons_buff_1 = aie.buffer(%tile_2_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "of_residual1_cons_buff_1"} : memref<2048xi8> 
    %of_residual1_cons_buff_2 = aie.buffer(%tile_2_2) {address = 2048 : i32, mem_bank = 0 : i32, sym_name = "of_residual1_cons_buff_2"} : memref<2048xi8> 
    %of_residual1_cons_buff_3 = aie.buffer(%tile_2_2) {address = 18432 : i32, mem_bank = 1 : i32, sym_name = "of_residual1_cons_buff_3"} : memref<2048xi8> 
    %of_residual1_cons_prod_lock_0 = aie.lock(%tile_2_2, 2) {init = 4 : i32, sym_name = "of_residual1_cons_prod_lock_0"}
    %of_residual1_cons_cons_lock_0 = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "of_residual1_cons_cons_lock_0"}
    %of_residual1_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "of_residual1_buff_0"} : memref<2048xi8> 
    %of_residual1_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "of_residual1_buff_1"} : memref<2048xi8> 
    %of_residual1_buff_2 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "of_residual1_buff_2"} : memref<2048xi8> 
    %of_residual1_buff_3 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "of_residual1_buff_3"} : memref<2048xi8> 
    %of_residual1_prod_lock_0 = aie.lock(%tile_0_2, 8) {init = 4 : i32, sym_name = "of_residual1_prod_lock_0"}
    %of_residual1_cons_lock_0 = aie.lock(%tile_0_2, 9) {init = 0 : i32, sym_name = "of_residual1_cons_lock_0"}
    %of_o_proj_cons_buff_0 = aie.buffer(%tile_2_2) {address = 34816 : i32, mem_bank = 2 : i32, sym_name = "of_o_proj_cons_buff_0"} : memref<2048xi8> 
    %of_o_proj_cons_buff_1 = aie.buffer(%tile_2_2) {address = 51200 : i32, mem_bank = 3 : i32, sym_name = "of_o_proj_cons_buff_1"} : memref<2048xi8> 
    %of_o_proj_cons_prod_lock_0 = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "of_o_proj_cons_prod_lock_0"}
    %of_o_proj_cons_cons_lock_0 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "of_o_proj_cons_cons_lock_0"}
    %of_o_proj_buff_0 = aie.buffer(%tile_1_4) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "of_o_proj_buff_0"} : memref<2048xi8> 
    %of_o_proj_buff_1 = aie.buffer(%tile_1_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "of_o_proj_buff_1"} : memref<2048xi8> 
    %of_o_proj_prod_lock_0 = aie.lock(%tile_1_4, 0) {init = 2 : i32, sym_name = "of_o_proj_prod_lock_0"}
    %of_o_proj_cons_lock_0 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "of_o_proj_cons_lock_0"}
    %of_attn_buff_0 = aie.buffer(%tile_1_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "of_attn_buff_0"} : memref<2048xi8> 
    %of_attn_buff_1 = aie.buffer(%tile_1_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "of_attn_buff_1"} : memref<2048xi8> 
    %of_attn_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 2 : i32, sym_name = "of_attn_prod_lock_0"}
    %of_attn_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "of_attn_cons_lock_0"}
    %of_v_buff_0 = aie.buffer(%tile_1_2) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "of_v_buff_0"} : memref<2048xi8> 
    %of_v_buff_1 = aie.buffer(%tile_1_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "of_v_buff_1"} : memref<2048xi8> 
    %of_v_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "of_v_prod_lock_0"}
    %of_v_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "of_v_cons_lock_0"}
    %of_k_cons_buff_0 = aie.buffer(%tile_1_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "of_k_cons_buff_0"} : memref<2048xi8> 
    %of_k_cons_buff_1 = aie.buffer(%tile_1_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "of_k_cons_buff_1"} : memref<2048xi8> 
    %of_k_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "of_k_cons_prod_lock_0"}
    %of_k_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "of_k_cons_cons_lock_0"}
    %of_k_buff_0 = aie.buffer(%tile_0_4) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "of_k_buff_0"} : memref<2048xi8> 
    %of_k_buff_1 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "of_k_buff_1"} : memref<2048xi8> 
    %of_k_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "of_k_prod_lock_0"}
    %of_k_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "of_k_cons_lock_0"}
    %of_q_buff_0 = aie.buffer(%tile_0_3) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "of_q_buff_0"} : memref<2048xi8> 
    %of_q_buff_1 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "of_q_buff_1"} : memref<2048xi8> 
    %of_q_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "of_q_prod_lock_0"}
    %of_q_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "of_q_cons_lock_0"}
    %of_ln1_to_v_buff_0 = aie.buffer(%tile_0_2) {address = 3072 : i32, mem_bank = 0 : i32, sym_name = "of_ln1_to_v_buff_0"} : memref<2048xi8> 
    %of_ln1_to_v_buff_1 = aie.buffer(%tile_0_2) {address = 18432 : i32, mem_bank = 1 : i32, sym_name = "of_ln1_to_v_buff_1"} : memref<2048xi8> 
    %of_ln1_to_v_prod_lock_0 = aie.lock(%tile_0_2, 6) {init = 2 : i32, sym_name = "of_ln1_to_v_prod_lock_0"}
    %of_ln1_to_v_cons_lock_0 = aie.lock(%tile_0_2, 7) {init = 0 : i32, sym_name = "of_ln1_to_v_cons_lock_0"}
    %of_ln1_to_k_cons_buff_0 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "of_ln1_to_k_cons_buff_0"} : memref<2048xi8> 
    %of_ln1_to_k_cons_buff_1 = aie.buffer(%tile_0_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "of_ln1_to_k_cons_buff_1"} : memref<2048xi8> 
    %of_ln1_to_k_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "of_ln1_to_k_cons_prod_lock_0"}
    %of_ln1_to_k_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "of_ln1_to_k_cons_cons_lock_0"}
    %of_ln1_to_k_buff_0 = aie.buffer(%tile_0_2) {address = 34816 : i32, mem_bank = 2 : i32, sym_name = "of_ln1_to_k_buff_0"} : memref<2048xi8> 
    %of_ln1_to_k_buff_1 = aie.buffer(%tile_0_2) {address = 51200 : i32, mem_bank = 3 : i32, sym_name = "of_ln1_to_k_buff_1"} : memref<2048xi8> 
    %of_ln1_to_k_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "of_ln1_to_k_prod_lock_0"}
    %of_ln1_to_k_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "of_ln1_to_k_cons_lock_0"}
    %of_ln1_to_q_buff_0 = aie.buffer(%tile_0_2) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "of_ln1_to_q_buff_0"} : memref<2048xi8> 
    %of_ln1_to_q_buff_1 = aie.buffer(%tile_0_2) {address = 20480 : i32, mem_bank = 1 : i32, sym_name = "of_ln1_to_q_buff_1"} : memref<2048xi8> 
    %of_ln1_to_q_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "of_ln1_to_q_prod_lock_0"}
    %of_ln1_to_q_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "of_ln1_to_q_cons_lock_0"}
    %of_input_cons_buff_0 = aie.buffer(%tile_0_2) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "of_input_cons_buff_0"} : memref<2048xi8> 
    %of_input_cons_buff_1 = aie.buffer(%tile_0_2) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "of_input_cons_buff_1"} : memref<2048xi8> 
    %of_input_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "of_input_cons_prod_lock_0"}
    %of_input_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "of_input_cons_cons_lock_0"}
    %of_input_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "of_input_prod_lock_0"}
    %of_input_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "of_input_cons_lock_0"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c4294967294 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_ln1_to_q_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @layernorm_bf16(%of_input_cons_buff_0, %of_ln1_to_q_buff_0) : (memref<2048xi8>, memref<2048xi8>) -> ()
      aie.use_lock(%of_input_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_ln1_to_q_cons_lock_0, Release, 1)
      aie.use_lock(%of_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_ln1_to_q_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @layernorm_bf16(%of_input_cons_buff_1, %of_ln1_to_q_buff_1) : (memref<2048xi8>, memref<2048xi8>) -> ()
      aie.use_lock(%of_input_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_ln1_to_q_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%of_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_ln1_to_q_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @layernorm_bf16(%of_input_cons_buff_0, %of_ln1_to_q_buff_0) : (memref<2048xi8>, memref<2048xi8>) -> ()
      aie.use_lock(%of_input_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_ln1_to_q_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "layernorm_bf16_xdna1.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c4294967294 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_q_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_k_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_v_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_attn_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @softmax_bf16(%of_v_buff_0, %of_attn_buff_0) : (memref<2048xi8>, memref<2048xi8>) -> ()
      aie.use_lock(%of_q_prod_lock_0, Release, 1)
      aie.use_lock(%of_k_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_v_prod_lock_0, Release, 1)
      aie.use_lock(%of_attn_cons_lock_0, Release, 1)
      aie.use_lock(%of_q_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_k_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_v_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_attn_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @softmax_bf16(%of_v_buff_1, %of_attn_buff_1) : (memref<2048xi8>, memref<2048xi8>) -> ()
      aie.use_lock(%of_q_prod_lock_0, Release, 1)
      aie.use_lock(%of_k_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_v_prod_lock_0, Release, 1)
      aie.use_lock(%of_attn_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%of_q_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_k_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_v_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_attn_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @softmax_bf16(%of_v_buff_0, %of_attn_buff_0) : (memref<2048xi8>, memref<2048xi8>) -> ()
      aie.use_lock(%of_q_prod_lock_0, Release, 1)
      aie.use_lock(%of_k_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_v_prod_lock_0, Release, 1)
      aie.use_lock(%of_attn_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "softmax_bf16_xdna1.o"}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c4294967294 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_fc1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_gelu_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @gelu_bf16(%of_fc1_buff_0, %of_gelu_buff_0) : (memref<2048xi8>, memref<2048xi8>) -> ()
      aie.use_lock(%of_fc1_prod_lock_0, Release, 1)
      aie.use_lock(%of_gelu_cons_lock_0, Release, 1)
      aie.use_lock(%of_fc1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_gelu_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @gelu_bf16(%of_fc1_buff_1, %of_gelu_buff_1) : (memref<2048xi8>, memref<2048xi8>) -> ()
      aie.use_lock(%of_fc1_prod_lock_0, Release, 1)
      aie.use_lock(%of_gelu_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%of_fc1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_gelu_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @gelu_bf16(%of_fc1_buff_0, %of_gelu_buff_0) : (memref<2048xi8>, memref<2048xi8>) -> ()
      aie.use_lock(%of_fc1_prod_lock_0, Release, 1)
      aie.use_lock(%of_gelu_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "gelu_simple_xdna1.o"}
    aiex.runtime_sequence(%arg0: memref<2048xi8>, %arg1: memref<2048xi8>) {
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c2048_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, metadata = @of_input_shim_alloc} : memref<2048xi8>
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c2048_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @of_output_shim_alloc} : memref<2048xi8>
      aiex.npu.dma_wait {symbol = @of_output_shim_alloc}
    }
    aie.shim_dma_allocation @of_input_shim_alloc(MM2S, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_input_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_cons_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_input_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_input_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_cons_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_input_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_ln1_to_k_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_ln1_to_k_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%of_ln1_to_k_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_ln1_to_k_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_ln1_to_k_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%of_ln1_to_k_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb11)
    ^bb7:  // 2 preds: ^bb6, ^bb10
      aie.use_lock(%of_residual1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_residual1_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%of_residual1_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%of_residual1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_residual1_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%of_residual1_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb9:  // pred: ^bb8
      aie.use_lock(%of_residual1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_residual1_buff_2 : memref<2048xi8>, 0, 2048) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%of_residual1_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%of_residual1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_residual1_buff_3 : memref<2048xi8>, 0, 2048) {bd_id = 7 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%of_residual1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb11:  // pred: ^bb6
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_ln1_to_k_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_ln1_to_k_cons_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_ln1_to_k_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_ln1_to_k_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_ln1_to_k_cons_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_ln1_to_k_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_k_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_k_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%of_k_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_k_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_k_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%of_k_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_k_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_k_cons_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_k_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_k_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_k_cons_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_k_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_o_proj_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_o_proj_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_o_proj_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_o_proj_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_o_proj_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_o_proj_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_o_proj_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_o_proj_cons_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_o_proj_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_o_proj_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_o_proj_cons_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_o_proj_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb8)
    ^bb4:  // 2 preds: ^bb3, ^bb7
      aie.use_lock(%of_residual1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_residual1_cons_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%of_residual1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_residual1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_residual1_cons_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%of_residual1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%of_residual1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_residual1_cons_buff_2 : memref<2048xi8>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%of_residual1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%of_residual1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_residual1_cons_buff_3 : memref<2048xi8>, 0, 2048) {bd_id = 5 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%of_residual1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb8:  // pred: ^bb3
      aie.end
    }
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_gelu_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_gelu_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_gelu_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_gelu_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_gelu_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_gelu_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_gelu_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_gelu_cons_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_gelu_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_gelu_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_gelu_cons_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_gelu_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_output_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_output_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_output_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output_buff_1 : memref<2048xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_output_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    aie.shim_dma_allocation @of_output_shim_alloc(S2MM, 0, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<East : 1, South : 2>
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
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, East : 0>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 3, North : 0>
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<North : 1, DMA : 0>
      aie.connect<North : 3, South : 3>
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<West : 0, South : 1>
      aie.connect<DMA : 0, South : 3>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<North : 3, East : 3>
      aie.connect<West : 0, East : 1>
    }
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<West : 3, DMA : 0>
      aie.connect<West : 1, DMA : 1>
      aie.connect<North : 0, East : 3>
      aie.connect<East : 1, South : 1>
    }
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<North : 1, South : 0>
    }
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      aie.connect<DMA : 0, South : 1>
    }
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<West : 3, DMA : 0>
      aie.connect<North : 1, West : 1>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<East : 3, West : 1>
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      aie.connect<North : 1, West : 3>
    }
    %mem_tile_2_1 = aie.tile(2, 1)
    %switchbox_2_1 = aie.switchbox(%mem_tile_2_1) {
      aie.connect<North : 1, South : 1>
    }
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
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
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_0_3 : East, %switchbox_1_3 : West)
    aie.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
    aie.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
    aie.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
    aie.wire(%switchbox_0_4 : East, %switchbox_1_4 : West)
    aie.wire(%tile_1_4 : Core, %switchbox_1_4 : Core)
    aie.wire(%tile_1_4 : DMA, %switchbox_1_4 : DMA)
    aie.wire(%switchbox_1_3 : North, %switchbox_1_4 : South)
    aie.wire(%switchbox_1_0 : East, %switchbox_2_0 : West)
    aie.wire(%mem_tile_2_1 : Core, %switchbox_2_1 : Core)
    aie.wire(%mem_tile_2_1 : DMA, %switchbox_2_1 : DMA)
    aie.wire(%switchbox_2_0 : North, %switchbox_2_1 : South)
    aie.wire(%switchbox_1_2 : East, %switchbox_2_2 : West)
    aie.wire(%tile_2_2 : Core, %switchbox_2_2 : Core)
    aie.wire(%tile_2_2 : DMA, %switchbox_2_2 : DMA)
    aie.wire(%switchbox_2_1 : North, %switchbox_2_2 : South)
    aie.wire(%switchbox_1_3 : East, %switchbox_2_3 : West)
    aie.wire(%tile_2_3 : Core, %switchbox_2_3 : Core)
    aie.wire(%tile_2_3 : DMA, %switchbox_2_3 : DMA)
    aie.wire(%switchbox_2_2 : North, %switchbox_2_3 : South)
    aie.wire(%switchbox_1_4 : East, %switchbox_2_4 : West)
    aie.wire(%tile_2_4 : Core, %switchbox_2_4 : Core)
    aie.wire(%tile_2_4 : DMA, %switchbox_2_4 : DMA)
    aie.wire(%switchbox_2_3 : North, %switchbox_2_4 : South)
    aie.wire(%switchbox_2_0 : East, %switchbox_3_0 : West)
    aie.wire(%switchbox_2_2 : East, %switchbox_3_2 : West)
    aie.wire(%tile_3_2 : Core, %switchbox_3_2 : Core)
    aie.wire(%tile_3_2 : DMA, %switchbox_3_2 : DMA)
    aie.wire(%switchbox_2_3 : East, %switchbox_3_3 : West)
    aie.wire(%tile_3_3 : Core, %switchbox_3_3 : Core)
    aie.wire(%tile_3_3 : DMA, %switchbox_3_3 : DMA)
    aie.wire(%switchbox_3_2 : North, %switchbox_3_3 : South)
  }
}

