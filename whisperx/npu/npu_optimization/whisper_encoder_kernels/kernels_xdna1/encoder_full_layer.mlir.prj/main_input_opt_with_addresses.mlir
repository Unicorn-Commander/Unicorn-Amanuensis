module @whisper_encoder_full_layer attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @of_input_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_input_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_ln1_to_q_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_ln1_to_q_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_ln1_to_k_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_ln1_to_k_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_ln1_to_k_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_ln1_to_k_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_ln1_to_v_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_ln1_to_v_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_q_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_q_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_k_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_k_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_k_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_k_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_v_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_v_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_attn_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_attn_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_o_proj_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_o_proj_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_o_proj_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_o_proj_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_residual1_buff_3() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_residual1_buff_2() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_residual1_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_residual1_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_residual1_cons_buff_3() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_residual1_cons_buff_2() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_residual1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_residual1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_ln2_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_ln2_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_fc1_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_fc1_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_gelu_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_gelu_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_gelu_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_gelu_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_fc2_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_fc2_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_output_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @of_output_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.event(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @layernorm_bf16(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @softmax_bf16(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @gelu_bf16(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @matmul_bf16_64x64(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @vector_add_bf16(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @core_2_4() {
    %0 = llvm.mlir.addressof @of_fc1_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @of_gelu_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @of_fc1_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @of_gelu_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(2 : index) : i64
    %5 = llvm.mlir.constant(4294967294 : index) : i64
    %6 = llvm.mlir.constant(49 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(48 : i32) : i32
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.mlir.constant(-1 : i32) : i32
    %11 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%11 : i64)
  ^bb1(%12: i64):  // 2 preds: ^bb0, ^bb2
    %13 = llvm.icmp "slt" %12, %5 : i64
    llvm.cond_br %13, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%9, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %10) : (i32, i32) -> ()
    %14 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %15 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    llvm.call @gelu_bf16(%15, %14) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%7, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%9, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %10) : (i32, i32) -> ()
    %16 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %17 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    llvm.call @gelu_bf16(%17, %16) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%7, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %9) : (i32, i32) -> ()
    %18 = llvm.add %12, %4 : i64
    llvm.br ^bb1(%18 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%9, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %10) : (i32, i32) -> ()
    %19 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %20 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    llvm.call @gelu_bf16(%20, %19) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%7, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %9) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_1_3() {
    %0 = llvm.mlir.addressof @of_v_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @of_attn_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @of_v_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @of_attn_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(2 : index) : i64
    %5 = llvm.mlir.constant(4294967294 : index) : i64
    %6 = llvm.mlir.constant(51 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(48 : i32) : i32
    %9 = llvm.mlir.constant(16 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(49 : i32) : i32
    %12 = llvm.mlir.constant(17 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(-1 : i32) : i32
    %15 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%15 : i64)
  ^bb1(%16: i64):  // 2 preds: ^bb0, ^bb2
    %17 = llvm.icmp "slt" %16, %5 : i64
    llvm.cond_br %17, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%10, %14) : (i32, i32) -> ()
    %18 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %19 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    llvm.call @softmax_bf16(%19, %18) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%9, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%10, %14) : (i32, i32) -> ()
    %20 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %21 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    llvm.call @softmax_bf16(%21, %20) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%9, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %13) : (i32, i32) -> ()
    %22 = llvm.add %16, %4 : i64
    llvm.br ^bb1(%22 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%10, %14) : (i32, i32) -> ()
    %23 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %24 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    llvm.call @softmax_bf16(%24, %23) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%9, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %13) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @of_input_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @of_ln1_to_q_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @of_input_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @of_ln1_to_q_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(2 : index) : i64
    %5 = llvm.mlir.constant(4294967294 : index) : i64
    %6 = llvm.mlir.constant(51 : i32) : i32
    %7 = llvm.mlir.constant(48 : i32) : i32
    %8 = llvm.mlir.constant(50 : i32) : i32
    %9 = llvm.mlir.constant(49 : i32) : i32
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.mlir.constant(-1 : i32) : i32
    %12 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%12 : i64)
  ^bb1(%13: i64):  // 2 preds: ^bb0, ^bb2
    %14 = llvm.icmp "slt" %13, %5 : i64
    llvm.cond_br %14, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%9, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %11) : (i32, i32) -> ()
    %15 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %16 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    llvm.call @layernorm_bf16(%16, %15) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%7, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%9, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %11) : (i32, i32) -> ()
    %17 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %18 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    llvm.call @layernorm_bf16(%18, %17) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%7, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %10) : (i32, i32) -> ()
    %19 = llvm.add %13, %4 : i64
    llvm.br ^bb1(%19 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%9, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %11) : (i32, i32) -> ()
    %20 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %21 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    llvm.call @layernorm_bf16(%21, %20) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%7, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %10) : (i32, i32) -> ()
    llvm.return
  }
}

