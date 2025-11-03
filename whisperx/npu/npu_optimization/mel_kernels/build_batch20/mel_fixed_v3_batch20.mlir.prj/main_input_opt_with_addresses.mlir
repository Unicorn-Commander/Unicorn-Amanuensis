module @mel_npu_batch20 attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @of_in_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<16000 x i8>
  llvm.mlir.global external @of_in_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<16000 x i8>
  llvm.mlir.global external @of_out_buff_1() {addr_space = 0 : i32} : !llvm.array<1600 x i8>
  llvm.mlir.global external @of_out_buff_0() {addr_space = 0 : i32} : !llvm.array<1600 x i8>
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
    %0 = llvm.mlir.addressof @of_out_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @of_in_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @of_out_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @of_in_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(2 : index) : i64
    %5 = llvm.mlir.constant(4294967294 : index) : i64
    %6 = llvm.mlir.constant(20 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(51 : i32) : i32
    %9 = llvm.mlir.constant(48 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(49 : i32) : i32
    %12 = llvm.mlir.constant(1 : i32) : i32
    %13 = llvm.mlir.constant(-1 : i32) : i32
    %14 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%14 : i64)
  ^bb1(%15: i64):  // 2 preds: ^bb0, ^bb8
    %16 = llvm.icmp "slt" %15, %5 : i64
    llvm.cond_br %16, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%11, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%10, %13) : (i32, i32) -> ()
    llvm.br ^bb3(%14 : i64)
  ^bb3(%17: i64):  // 2 preds: ^bb2, ^bb4
    %18 = llvm.icmp "slt" %17, %6 : i64
    llvm.cond_br %18, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %19 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16000 x i8>
    %20 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1600 x i8>
    llvm.call @mel_kernel_simple(%19, %20) : (!llvm.ptr, !llvm.ptr) -> ()
    %21 = llvm.add %17, %7 : i64
    llvm.br ^bb3(%21 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.release(%9, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%10, %13) : (i32, i32) -> ()
    llvm.br ^bb6(%14 : i64)
  ^bb6(%22: i64):  // 2 preds: ^bb5, ^bb7
    %23 = llvm.icmp "slt" %22, %6 : i64
    llvm.cond_br %23, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %24 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16000 x i8>
    %25 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1600 x i8>
    llvm.call @mel_kernel_simple(%24, %25) : (!llvm.ptr, !llvm.ptr) -> ()
    %26 = llvm.add %22, %7 : i64
    llvm.br ^bb6(%26 : i64)
  ^bb8:  // pred: ^bb6
    llvm.call @llvm.aie2.release(%9, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %12) : (i32, i32) -> ()
    %27 = llvm.add %15, %4 : i64
    llvm.br ^bb1(%27 : i64)
  ^bb9:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%11, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%10, %13) : (i32, i32) -> ()
    llvm.br ^bb10(%14 : i64)
  ^bb10(%28: i64):  // 2 preds: ^bb9, ^bb11
    %29 = llvm.icmp "slt" %28, %6 : i64
    llvm.cond_br %29, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %30 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16000 x i8>
    %31 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1600 x i8>
    llvm.call @mel_kernel_simple(%30, %31) : (!llvm.ptr, !llvm.ptr) -> ()
    %32 = llvm.add %28, %7 : i64
    llvm.br ^bb10(%32 : i64)
  ^bb12:  // pred: ^bb10
    llvm.call @llvm.aie2.release(%9, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %12) : (i32, i32) -> ()
    llvm.return
  }
}

