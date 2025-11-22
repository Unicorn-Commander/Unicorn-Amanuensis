module @test_layernorm_512 {
  aie.device(npu1) {
    func.func private @layernorm_512_simple(memref<1024xi8>, memref<1024xi8>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %of_output_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "of_output_cons_prod_lock_0"}
    %of_output_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "of_output_cons_cons_lock_0"}
    %of_output_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "of_output_buff_0"} : memref<1024xi8> 
    %of_output_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "of_output_buff_1"} : memref<1024xi8> 
    %of_output_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "of_output_prod_lock_0"}
    %of_output_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "of_output_cons_lock_0"}
    %of_input_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "of_input_cons_buff_0"} : memref<1024xi8> 
    %of_input_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "of_input_cons_buff_1"} : memref<1024xi8> 
    %of_input_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "of_input_cons_prod_lock_0"}
    %of_input_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "of_input_cons_cons_lock_0"}
    %of_input_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "of_input_prod_lock_0"}
    %of_input_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "of_input_cons_lock_0"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c4294967294 step %c2 {
        aie.use_lock(%of_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
        aie.use_lock(%of_output_prod_lock_0, AcquireGreaterEqual, 1)
        func.call @layernorm_512_simple(%of_input_cons_buff_0, %of_output_buff_0) : (memref<1024xi8>, memref<1024xi8>) -> ()
        aie.use_lock(%of_input_cons_prod_lock_0, Release, 1)
        aie.use_lock(%of_output_cons_lock_0, Release, 1)
        aie.use_lock(%of_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
        aie.use_lock(%of_output_prod_lock_0, AcquireGreaterEqual, 1)
        func.call @layernorm_512_simple(%of_input_cons_buff_1, %of_output_buff_1) : (memref<1024xi8>, memref<1024xi8>) -> ()
        aie.use_lock(%of_input_cons_prod_lock_0, Release, 1)
        aie.use_lock(%of_output_cons_lock_0, Release, 1)
      }
      aie.use_lock(%of_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_output_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @layernorm_512_simple(%of_input_cons_buff_0, %of_output_buff_0) : (memref<1024xi8>, memref<1024xi8>) -> ()
      aie.use_lock(%of_input_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_output_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "layernorm_512_simple.o"}
    aiex.runtime_sequence(%arg0: memref<1024xi8>, %arg1: memref<1024xi8>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c1024_i64 = arith.constant 1024 : i64
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c1024_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, metadata = @of_input_shim_alloc} : memref<1024xi8>
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c1024_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @of_output_shim_alloc} : memref<1024xi8>
      aiex.npu.dma_wait {symbol = @of_output_shim_alloc}
    }
    aie.shim_dma_allocation @of_input_shim_alloc(MM2S, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_input_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_cons_buff_0 : memref<1024xi8>, 0, 1024)
      aie.use_lock(%of_input_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_input_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_cons_buff_1 : memref<1024xi8>, 0, 1024)
      aie.use_lock(%of_input_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_output_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output_buff_0 : memref<1024xi8>, 0, 1024)
      aie.use_lock(%of_output_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_output_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output_buff_1 : memref<1024xi8>, 0, 1024)
      aie.use_lock(%of_output_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @of_output_shim_alloc(S2MM, 0, 0)
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 0, South : 2>
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
  }
}

