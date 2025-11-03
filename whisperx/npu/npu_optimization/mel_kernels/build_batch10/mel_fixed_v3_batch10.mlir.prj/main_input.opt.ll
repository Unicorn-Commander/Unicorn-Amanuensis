; ModuleID = '/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/build_batch10/mel_fixed_v3_batch10.mlir.prj/main_input.llpeanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2"

@of_in_cons_buff_1 = external global [8000 x i8]
@of_in_cons_buff_0 = external global [8000 x i8]
@of_out_buff_1 = external global [800 x i8]
@of_out_buff_0 = external global [800 x i8]

; Function Attrs: nounwind
declare void @llvm.aie2.acquire(i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.aie2.release(i32, i32) #0

declare void @mel_kernel_simple(ptr, ptr) local_unnamed_addr

define void @core_0_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ 0, %0 ], [ %5, %4 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %3 = icmp ult i64 %2, 4294967292
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_1, ptr nonnull @of_out_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %5 = add nuw nsw i64 %2, 4
  br label %1

6:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @mel_kernel_simple(ptr nonnull @of_in_cons_buff_0, ptr nonnull @of_out_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

attributes #0 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
