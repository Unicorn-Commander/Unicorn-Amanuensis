module @mel_simple_single {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    func.func private @mel_kernel_simple(memref<800xi8>, memref<80xi8>)
    %in_prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "in_prod_lock"}
    %in_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in_cons_lock"}
    %out_prod_lock = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "out_prod_lock"}
    %out_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out_cons_lock"}
    %in_buffer = aie.buffer(%tile_0_2) {address = 4096 : i32, sym_name = "in_buffer"} : memref<800xi8> 
    %out_buffer = aie.buffer(%tile_0_2) {address = 8192 : i32, sym_name = "out_buffer"} : memref<80xi8> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c48 = arith.constant 48 : index
      %c49 = arith.constant 49 : index
      %c50 = arith.constant 50 : index
      %c51 = arith.constant 51 : index
      aie.use_lock(%c49, AcquireGreaterEqual, 1)
      aie.use_lock(%c50, AcquireGreaterEqual, 1)
      func.call @mel_kernel_simple(%in_buffer, %out_buffer) : (memref<800xi8>, memref<80xi8>) -> ()
      aie.use_lock(%c48, Release, 1)
      aie.use_lock(%c51, Release, 1)
      aie.end
    } {link_with = "mel_kernel_simple.o"}
    aie.shim_dma_allocation @in_shim(MM2S, 0, 0)
    aie.shim_dma_allocation @out_shim(S2MM, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%in_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_buffer : memref<800xi8>, 0, 800)
      aie.use_lock(%in_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%out_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buffer : memref<80xi8>, 0, 80)
      aie.use_lock(%out_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<800xi8>, %arg1: memref<80xi8>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c800_i64 = arith.constant 800 : i64
      %c80_i64 = arith.constant 80 : i64
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c800_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, metadata = @in_shim} : memref<800xi8>
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c80_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @out_shim} : memref<80xi8>
      aiex.npu.dma_wait {symbol = @out_shim}
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<North : 0, South : 2>
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 0, North : 0>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
  }
}

