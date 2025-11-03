; ModuleID = '/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/fresh_attention_iron.mlir.prj/main_input.llpeanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2"

@of_input_0_cons_buff_1 = external global [12288 x i8]
@of_input_0_cons_buff_0 = external global [12288 x i8]
@of_input_1_cons_buff_1 = external global [12288 x i8]
@of_input_1_cons_buff_0 = external global [12288 x i8]
@of_input_2_cons_buff_1 = external global [12288 x i8]
@of_input_2_cons_buff_0 = external global [12288 x i8]
@of_input_3_cons_buff_1 = external global [12288 x i8]
@of_input_3_cons_buff_0 = external global [12288 x i8]
@of_output_0_buff_1 = external global [4096 x i8]
@of_output_0_buff_0 = external global [4096 x i8]
@of_output_1_buff_1 = external global [4096 x i8]
@of_output_1_buff_0 = external global [4096 x i8]
@of_output_2_buff_1 = external global [4096 x i8]
@of_output_2_buff_0 = external global [4096 x i8]
@of_output_3_buff_1 = external global [4096 x i8]
@of_output_3_buff_0 = external global [4096 x i8]

; Function Attrs: nounwind
declare void @llvm.aie2.acquire(i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.aie2.release(i32, i32) #0

declare void @attention_64x64(ptr, ptr, i32) local_unnamed_addr

define void @core_3_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @attention_64x64(ptr nonnull @of_input_3_cons_buff_0, ptr nonnull @of_output_3_buff_0, i32 3)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @attention_64x64(ptr nonnull @of_input_3_cons_buff_1, ptr nonnull @of_output_3_buff_1, i32 3)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @attention_64x64(ptr nonnull @of_input_3_cons_buff_0, ptr nonnull @of_output_3_buff_0, i32 3)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

define void @core_2_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @attention_64x64(ptr nonnull @of_input_2_cons_buff_0, ptr nonnull @of_output_2_buff_0, i32 3)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @attention_64x64(ptr nonnull @of_input_2_cons_buff_1, ptr nonnull @of_output_2_buff_1, i32 3)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @attention_64x64(ptr nonnull @of_input_2_cons_buff_0, ptr nonnull @of_output_2_buff_0, i32 3)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

define void @core_1_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @attention_64x64(ptr nonnull @of_input_1_cons_buff_0, ptr nonnull @of_output_1_buff_0, i32 3)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @attention_64x64(ptr nonnull @of_input_1_cons_buff_1, ptr nonnull @of_output_1_buff_1, i32 3)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @attention_64x64(ptr nonnull @of_input_1_cons_buff_0, ptr nonnull @of_output_1_buff_0, i32 3)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

define void @core_0_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @attention_64x64(ptr nonnull @of_input_0_cons_buff_0, ptr nonnull @of_output_0_buff_0, i32 3)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @attention_64x64(ptr nonnull @of_input_0_cons_buff_1, ptr nonnull @of_output_0_buff_1, i32 3)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @attention_64x64(ptr nonnull @of_input_0_cons_buff_0, ptr nonnull @of_output_0_buff_0, i32 3)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

attributes #0 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
