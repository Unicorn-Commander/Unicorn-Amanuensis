module @passthrough_complete {
  aie.device(npu1) {
    memref.global "public" @of_out_cons : memref<1024xui8>
    memref.global "public" @of_out : memref<1024xui8>
    memref.global "public" @of_in_cons : memref<1024xui8>
    memref.global "public" @of_in : memref<1024xui8>
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %of_out_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "of_out_cons_prod_lock_0"}
    %of_out_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "of_out_cons_cons_lock_0"}
    %of_out_buff_0 = aie.buffer(%tile_0_2) {sym_name = "of_out_buff_0"} : memref<1024xui8> 
    %of_out_buff_1 = aie.buffer(%tile_0_2) {sym_name = "of_out_buff_1"} : memref<1024xui8> 
    %of_out_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "of_out_prod_lock_0"}
    %of_out_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "of_out_cons_lock_0"}
    %of_in_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "of_in_cons_buff_0"} : memref<1024xui8> 
    %of_in_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "of_in_cons_buff_1"} : memref<1024xui8> 
    %of_in_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "of_in_cons_prod_lock_0"}
    %of_in_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "of_in_cons_cons_lock_0"}
    %of_in_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "of_in_prod_lock_0"}
    %of_in_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "of_in_cons_lock_0"}
    func.func private @passthrough_kernel(memref<1024xui8>, memref<1024xui8>, i32)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024_i32 = arith.constant 1024 : i32
      aie.use_lock(%of_in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_out_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @passthrough_kernel(%of_in_cons_buff_0, %of_out_buff_0, %c1024_i32) : (memref<1024xui8>, memref<1024xui8>, i32) -> ()
      aie.use_lock(%of_in_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_out_cons_lock_0, Release, 1)
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @of_in} : memref<1024xi32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @of_out} : memref<1024xi32>
      aiex.npu.dma_wait {symbol = @of_out}
    }
    aie.shim_dma_allocation @of_in(MM2S, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_in_cons_buff_0 : memref<1024xui8>, 0, 1024)
      aie.use_lock(%of_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_in_cons_buff_1 : memref<1024xui8>, 0, 1024)
      aie.use_lock(%of_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_out_buff_0 : memref<1024xui8>, 0, 1024)
      aie.use_lock(%of_out_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_out_buff_1 : memref<1024xui8>, 0, 1024)
      aie.use_lock(%of_out_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @of_out(S2MM, 0, 0)
  }
}

