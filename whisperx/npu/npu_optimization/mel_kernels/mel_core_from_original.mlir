module @mel_npu_with_loop attributes {llvm.target_triple = "aie2"} {
  func.func private @debug_i32(i32)
  func.func private @llvm.aie2.event(i32)
  func.func private @llvm.aie2.put.ms(i32, i32)
  func.func private @llvm.aie2.get.ss() -> (i32, i32)
  func.func private @llvm.aie2.mcd.write.vec(vector<16xi32>, i32)
  func.func private @llvm.aie2.scd.read.vec(i32) -> vector<16xi32>
  func.func private @llvm.aie2.acquire(i32, i32)
  func.func private @llvm.aie2.release(i32, i32)
  func.func private @mel_kernel_simple(memref<800xi8>, memref<80xi8>)
}

