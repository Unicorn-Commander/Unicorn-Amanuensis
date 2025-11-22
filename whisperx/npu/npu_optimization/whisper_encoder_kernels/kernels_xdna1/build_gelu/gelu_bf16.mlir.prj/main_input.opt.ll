; ModuleID = '/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1/build_gelu/gelu_bf16.mlir.prj/main_input.llpeanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2"

@of_input_cons_buff_1 = external global [2048 x i8]
@of_input_cons_buff_0 = external global [2048 x i8]
@of_output_buff_1 = external global [2048 x i8]
@of_output_buff_0 = external global [2048 x i8]

; Function Attrs: nounwind
declare void @llvm.aie2.acquire(i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.aie2.release(i32, i32) #0

declare void @gelu_bf16(ptr, ptr) local_unnamed_addr

define void @core_0_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %5, %0
  %2 = phi i64 [ 0, %0 ], [ %6, %5 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @gelu_bf16(ptr nonnull @of_input_cons_buff_0, ptr nonnull @of_output_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @gelu_bf16(ptr nonnull @of_input_cons_buff_1, ptr nonnull @of_output_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @gelu_bf16(ptr nonnull @of_input_cons_buff_0, ptr nonnull @of_output_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @gelu_bf16(ptr nonnull @of_input_cons_buff_1, ptr nonnull @of_output_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %3 = or disjoint i64 %2, 4
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @gelu_bf16(ptr nonnull @of_input_cons_buff_0, ptr nonnull @of_output_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @gelu_bf16(ptr nonnull @of_input_cons_buff_1, ptr nonnull @of_output_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %4 = icmp ult i64 %3, 4294967292
  br i1 %4, label %5, label %7

5:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @gelu_bf16(ptr nonnull @of_input_cons_buff_0, ptr nonnull @of_output_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @gelu_bf16(ptr nonnull @of_input_cons_buff_1, ptr nonnull @of_output_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %6 = add nuw nsw i64 %2, 8
  br label %1

7:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @gelu_bf16(ptr nonnull @of_input_cons_buff_0, ptr nonnull @of_output_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

attributes #0 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
