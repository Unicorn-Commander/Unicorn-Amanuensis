module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @of_input_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<12288 x i8>
  llvm.mlir.global external @of_input_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<12288 x i8>
  llvm.mlir.global external @of_output_buff_1() {addr_space = 0 : i32} : !llvm.array<4096 x i8>
  llvm.mlir.global external @of_output_buff_0() {addr_space = 0 : i32} : !llvm.array<4096 x i8>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.event(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @attention_64x64(!llvm.ptr, !llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @of_input_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @of_output_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @of_input_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @of_output_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(2 : index) : i64
    %5 = llvm.mlir.constant(9223372036854775806 : index) : i64
    %6 = llvm.mlir.constant(51 : i32) : i32
    %7 = llvm.mlir.constant(48 : i32) : i32
    %8 = llvm.mlir.constant(50 : i32) : i32
    %9 = llvm.mlir.constant(49 : i32) : i32
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.mlir.constant(-1 : i32) : i32
    %12 = llvm.mlir.constant(3 : i32) : i32
    %13 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%13 : i64)
  ^bb1(%14: i64):  // 2 preds: ^bb0, ^bb2
    %15 = llvm.icmp "slt" %14, %5 : i64
    llvm.cond_br %15, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%9, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %11) : (i32, i32) -> ()
    %16 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4096 x i8>
    %17 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<12288 x i8>
    llvm.call @attention_64x64(%17, %16, %12) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%9, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %11) : (i32, i32) -> ()
    %18 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4096 x i8>
    %19 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<12288 x i8>
    llvm.call @attention_64x64(%19, %18, %12) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %10) : (i32, i32) -> ()
    %20 = llvm.add %14, %4 : i64
    llvm.br ^bb1(%20 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%9, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %11) : (i32, i32) -> ()
    %21 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4096 x i8>
    %22 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<12288 x i8>
    llvm.call @attention_64x64(%22, %21, %12) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %10) : (i32, i32) -> ()
    llvm.return
  }
}

