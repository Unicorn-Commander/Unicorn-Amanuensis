; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@of_in_cons_buff_1 = external global [8000 x i8]
@of_in_cons_buff_0 = external global [8000 x i8]
@of_out_buff_1 = external global [800 x i8]
@of_out_buff_0 = external global [800 x i8]

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

1:                                                ; preds = %16, %0
  %2 = phi i64 [ %17, %16 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967294
  br i1 %3, label %4, label %18

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %9, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 10
  br i1 %7, label %8, label %10

8:                                                ; preds = %5
  call void @mel_kernel_simple(ptr @of_in_cons_buff_0, ptr @of_out_buff_0)
  %9 = add i64 %6, 1
  br label %5

10:                                               ; preds = %5
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  br label %11

11:                                               ; preds = %14, %10
  %12 = phi i64 [ %15, %14 ], [ 0, %10 ]
  %13 = icmp slt i64 %12, 10
  br i1 %13, label %14, label %16

14:                                               ; preds = %11
  call void @mel_kernel_simple(ptr @of_in_cons_buff_1, ptr @of_out_buff_1)
  %15 = add i64 %12, 1
  br label %11

16:                                               ; preds = %11
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %17 = add i64 %2, 2
  br label %1

18:                                               ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  br label %19

19:                                               ; preds = %22, %18
  %20 = phi i64 [ %23, %22 ], [ 0, %18 ]
  %21 = icmp slt i64 %20, 10
  br i1 %21, label %22, label %24

22:                                               ; preds = %19
  call void @mel_kernel_simple(ptr @of_in_cons_buff_0, ptr @of_out_buff_0)
  %23 = add i64 %20, 1
  br label %19

24:                                               ; preds = %19
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
