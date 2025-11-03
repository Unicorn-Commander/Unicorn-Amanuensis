module @mel_npu_batch30 attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @of_in_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<24000 x i8>
  llvm.mlir.global external @of_out_buff_0() {addr_space = 0 : i32} : !llvm.array<2400 x i8>
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
    %0 = llvm.mlir.addressof @of_out_buff_0 : !llvm.ptr
    %1 = llvm.mlir.addressof @of_in_cons_buff_0 : !llvm.ptr
    %2 = llvm.mlir.constant(30 : index) : i64
    %3 = llvm.mlir.constant(4294967295 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(51 : i32) : i32
    %6 = llvm.mlir.constant(48 : i32) : i32
    %7 = llvm.mlir.constant(50 : i32) : i32
    %8 = llvm.mlir.constant(49 : i32) : i32
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.mlir.constant(-1 : i32) : i32
    %11 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%11 : i64)
  ^bb1(%12: i64):  // 2 preds: ^bb0, ^bb5
    %13 = llvm.icmp "slt" %12, %3 : i64
    llvm.cond_br %13, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%8, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%7, %10) : (i32, i32) -> ()
    llvm.br ^bb3(%11 : i64)
  ^bb3(%14: i64):  // 2 preds: ^bb2, ^bb4
    %15 = llvm.icmp "slt" %14, %2 : i64
    llvm.cond_br %15, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %16 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<24000 x i8>
    %17 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2400 x i8>
    llvm.call @mel_kernel_simple(%16, %17) : (!llvm.ptr, !llvm.ptr) -> ()
    %18 = llvm.add %14, %4 : i64
    llvm.br ^bb3(%18 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.release(%6, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%5, %9) : (i32, i32) -> ()
    %19 = llvm.add %12, %4 : i64
    llvm.br ^bb1(%19 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}

