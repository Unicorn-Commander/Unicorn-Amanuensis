module @mel_npu_batch100 attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @of_in_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<800 x i8>
  llvm.mlir.global external @of_in_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<800 x i8>
  llvm.mlir.global external @of_out_buff_1() {addr_space = 0 : i32} : !llvm.array<80 x i8>
  llvm.mlir.global external @of_out_buff_0() {addr_space = 0 : i32} : !llvm.array<80 x i8>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.event(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @mel_kernel_simple(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @of_in_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @of_out_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @of_in_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @of_out_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(100 : index) : i64
    %5 = llvm.mlir.constant(4294967295 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(51 : i32) : i32
    %8 = llvm.mlir.constant(48 : i32) : i32
    %9 = llvm.mlir.constant(50 : i32) : i32
    %10 = llvm.mlir.constant(49 : i32) : i32
    %11 = llvm.mlir.constant(1 : i32) : i32
    %12 = llvm.mlir.constant(-1 : i32) : i32
    %13 = llvm.mlir.constant(2 : index) : i64
    %14 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%14 : i64)
  ^bb1(%15: i64):  // 2 preds: ^bb0, ^bb4
    %16 = llvm.icmp "slt" %15, %5 : i64
    llvm.cond_br %16, ^bb2(%14 : i64), ^bb5
  ^bb2(%17: i64):  // 2 preds: ^bb1, ^bb3
    %18 = llvm.icmp "slt" %17, %4 : i64
    llvm.cond_br %18, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2.acquire(%10, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%9, %12) : (i32, i32) -> ()
    %19 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<80 x i8>
    %20 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<800 x i8>
    llvm.call @mel_kernel_simple(%20, %19) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%8, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%10, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%9, %12) : (i32, i32) -> ()
    %21 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<80 x i8>
    %22 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<800 x i8>
    llvm.call @mel_kernel_simple(%22, %21) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%8, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %11) : (i32, i32) -> ()
    %23 = llvm.add %17, %13 : i64
    llvm.br ^bb2(%23 : i64)
  ^bb4:  // pred: ^bb2
    %24 = llvm.add %15, %6 : i64
    llvm.br ^bb1(%24 : i64)
  ^bb5:  // pred: ^bb1
    llvm.return
  }
}

