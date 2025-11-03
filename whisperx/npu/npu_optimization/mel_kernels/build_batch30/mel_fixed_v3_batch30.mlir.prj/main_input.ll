; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@of_in_cons_buff_0 = external global [24000 x i8]
@of_out_buff_0 = external global [2400 x i8]

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

declare void @mel_kernel_simple(ptr, ptr)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %10, %0
  %2 = phi i64 [ %11, %10 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %12

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %9, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 30
  br i1 %7, label %8, label %10

8:                                                ; preds = %5
  call void @mel_kernel_simple(ptr @of_in_cons_buff_0, ptr @of_out_buff_0)
  %9 = add i64 %6, 1
  br label %5

10:                                               ; preds = %5
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %11 = add i64 %2, 1
  br label %1

12:                                               ; preds = %1
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
