module @mel_simple_single attributes {llvm.target_triple = "aie2"} {
  memref.global "public" @out_buffer : memref<80xi8>
  memref.global "public" @in_buffer : memref<800xi8>
  func.func private @debug_i32(i32)
  func.func private @llvm.aie2.event(i32)
  func.func private @llvm.aie2.put.ms(i32, i32)
  func.func private @llvm.aie2.get.ss() -> (i32, i32)
  func.func private @llvm.aie2.mcd.write.vec(vector<16xi32>, i32)
  func.func private @llvm.aie2.scd.read.vec(i32) -> vector<16xi32>
  func.func private @llvm.aie2.acquire(i32, i32)
  func.func private @llvm.aie2.release(i32, i32)
  func.func private @mel_kernel_simple(memref<800xi8>, memref<80xi8>)
  func.func @core_0_2() {
    %c48 = arith.constant 48 : index
    %c49 = arith.constant 49 : index
    %c50 = arith.constant 50 : index
    %c51 = arith.constant 51 : index
    %0 = arith.index_cast %c49 : index to i32
    %c-1_i32 = arith.constant -1 : i32
    call @llvm.aie2.acquire(%0, %c-1_i32) : (i32, i32) -> ()
    %1 = arith.index_cast %c50 : index to i32
    %c-1_i32_0 = arith.constant -1 : i32
    call @llvm.aie2.acquire(%1, %c-1_i32_0) : (i32, i32) -> ()
    %2 = memref.get_global @in_buffer : memref<800xi8>
    %assume_align = memref.assume_alignment %2, 32 : memref<800xi8>
    %3 = memref.get_global @out_buffer : memref<80xi8>
    %assume_align_1 = memref.assume_alignment %3, 32 : memref<80xi8>
    call @mel_kernel_simple(%2, %3) : (memref<800xi8>, memref<80xi8>) -> ()
    %4 = arith.index_cast %c48 : index to i32
    %c1_i32 = arith.constant 1 : i32
    call @llvm.aie2.release(%4, %c1_i32) : (i32, i32) -> ()
    %5 = arith.index_cast %c51 : index to i32
    %c1_i32_2 = arith.constant 1 : i32
    call @llvm.aie2.release(%5, %c1_i32_2) : (i32, i32) -> ()
    return
  }
}

