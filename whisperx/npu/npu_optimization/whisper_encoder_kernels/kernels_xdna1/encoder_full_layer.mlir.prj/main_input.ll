; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@of_input_cons_buff_1 = external global [2048 x i8]
@of_input_cons_buff_0 = external global [2048 x i8]
@of_ln1_to_q_buff_1 = external global [2048 x i8]
@of_ln1_to_q_buff_0 = external global [2048 x i8]
@of_ln1_to_k_buff_1 = external global [2048 x i8]
@of_ln1_to_k_buff_0 = external global [2048 x i8]
@of_ln1_to_k_cons_buff_1 = external global [2048 x i8]
@of_ln1_to_k_cons_buff_0 = external global [2048 x i8]
@of_ln1_to_v_buff_1 = external global [2048 x i8]
@of_ln1_to_v_buff_0 = external global [2048 x i8]
@of_q_buff_1 = external global [2048 x i8]
@of_q_buff_0 = external global [2048 x i8]
@of_k_buff_1 = external global [2048 x i8]
@of_k_buff_0 = external global [2048 x i8]
@of_k_cons_buff_1 = external global [2048 x i8]
@of_k_cons_buff_0 = external global [2048 x i8]
@of_v_buff_1 = external global [2048 x i8]
@of_v_buff_0 = external global [2048 x i8]
@of_attn_buff_1 = external global [2048 x i8]
@of_attn_buff_0 = external global [2048 x i8]
@of_o_proj_buff_1 = external global [2048 x i8]
@of_o_proj_buff_0 = external global [2048 x i8]
@of_o_proj_cons_buff_1 = external global [2048 x i8]
@of_o_proj_cons_buff_0 = external global [2048 x i8]
@of_residual1_buff_3 = external global [2048 x i8]
@of_residual1_buff_2 = external global [2048 x i8]
@of_residual1_buff_1 = external global [2048 x i8]
@of_residual1_buff_0 = external global [2048 x i8]
@of_residual1_cons_buff_3 = external global [2048 x i8]
@of_residual1_cons_buff_2 = external global [2048 x i8]
@of_residual1_cons_buff_1 = external global [2048 x i8]
@of_residual1_cons_buff_0 = external global [2048 x i8]
@of_ln2_buff_1 = external global [2048 x i8]
@of_ln2_buff_0 = external global [2048 x i8]
@of_fc1_buff_1 = external global [2048 x i8]
@of_fc1_buff_0 = external global [2048 x i8]
@of_gelu_buff_1 = external global [2048 x i8]
@of_gelu_buff_0 = external global [2048 x i8]
@of_gelu_cons_buff_1 = external global [2048 x i8]
@of_gelu_cons_buff_0 = external global [2048 x i8]
@of_fc2_buff_1 = external global [2048 x i8]
@of_fc2_buff_0 = external global [2048 x i8]
@of_output_buff_1 = external global [2048 x i8]
@of_output_buff_0 = external global [2048 x i8]

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie2.event(i32)

; Unknown intrinsic
declare void @llvm.aie2.put.ms(i32, i32)

; Unknown intrinsic
declare { i32, i32 } @llvm.aie2.get.ss()

; Unknown intrinsic
declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

; Unknown intrinsic
declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

; Unknown intrinsic
declare void @llvm.aie2.acquire(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2.release(i32, i32)

declare void @layernorm_bf16(ptr, ptr)

declare void @softmax_bf16(ptr, ptr)

declare void @gelu_bf16(ptr, ptr)

declare void @matmul_bf16_64x64(ptr, ptr, ptr)

declare void @vector_add_bf16(ptr, ptr, ptr)

define void @core_2_4() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %5, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967294
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 1, i32 -1)
  call void @llvm.aie2.acquire(i32 48, i32 -1)
  call void @gelu_bf16(ptr @of_fc1_buff_0, ptr @of_gelu_buff_0)
  call void @llvm.aie2.release(i32 0, i32 1)
  call void @llvm.aie2.release(i32 49, i32 1)
  call void @llvm.aie2.acquire(i32 1, i32 -1)
  call void @llvm.aie2.acquire(i32 48, i32 -1)
  call void @gelu_bf16(ptr @of_fc1_buff_1, ptr @of_gelu_buff_1)
  call void @llvm.aie2.release(i32 0, i32 1)
  call void @llvm.aie2.release(i32 49, i32 1)
  %5 = add i64 %2, 2
  br label %1

6:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 1, i32 -1)
  call void @llvm.aie2.acquire(i32 48, i32 -1)
  call void @gelu_bf16(ptr @of_fc1_buff_0, ptr @of_gelu_buff_0)
  call void @llvm.aie2.release(i32 0, i32 1)
  call void @llvm.aie2.release(i32 49, i32 1)
  ret void
}

define void @core_1_3() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %5, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967294
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 17, i32 -1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 1, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @softmax_bf16(ptr @of_v_buff_0, ptr @of_attn_buff_0)
  call void @llvm.aie2.release(i32 16, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 0, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 17, i32 -1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 1, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @softmax_bf16(ptr @of_v_buff_1, ptr @of_attn_buff_1)
  call void @llvm.aie2.release(i32 16, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 0, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %5 = add i64 %2, 2
  br label %1

6:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 17, i32 -1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 1, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @softmax_bf16(ptr @of_v_buff_0, ptr @of_attn_buff_0)
  call void @llvm.aie2.release(i32 16, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 0, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

define void @core_0_2() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %5, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967294
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @layernorm_bf16(ptr @of_input_cons_buff_0, ptr @of_ln1_to_q_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @layernorm_bf16(ptr @of_input_cons_buff_1, ptr @of_ln1_to_q_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %5 = add i64 %2, 2
  br label %1

6:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @layernorm_bf16(ptr @of_input_cons_buff_0, ptr @of_ln1_to_q_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
