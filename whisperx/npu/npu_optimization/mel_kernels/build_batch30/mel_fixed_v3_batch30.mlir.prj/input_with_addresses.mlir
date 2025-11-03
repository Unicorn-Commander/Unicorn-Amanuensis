module @mel_npu_batch30 {
  aie.device(npu1) {
    %c2400_i64 = arith.constant 2400 : i64
    %c24000_i64 = arith.constant 24000 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    func.func private @mel_kernel_simple(memref<800xi8>, memref<80xi8>)
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %of_out_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "of_out_cons_prod_lock_0"}
    %of_out_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "of_out_cons_cons_lock_0"}
    %of_out_buff_0 = aie.buffer(%tile_0_2) {address = 25024 : i32, sym_name = "of_out_buff_0"} : memref<2400xi8> 
    %of_out_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "of_out_prod_lock_0"}
    %of_out_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "of_out_cons_lock_0"}
    %of_in_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "of_in_cons_buff_0"} : memref<24000xi8> 
    %of_in_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "of_in_cons_prod_lock_0"}
    %of_in_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "of_in_cons_cons_lock_0"}
    %of_in_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "of_in_prod_lock_0"}
    %of_in_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "of_in_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c30 = arith.constant 30 : index
      %c800 = arith.constant 800 : index
      %c80 = arith.constant 80 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_out_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c30 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %4 = arith.muli %2, %c800 : index
      %5 = arith.muli %2, %c80 : index
      %subview = memref.subview %of_in_cons_buff_0[%4] [800] [1] : memref<24000xi8> to memref<800xi8, strided<[1], offset: ?>>
      %subview_0 = memref.subview %of_out_buff_0[%5] [80] [1] : memref<2400xi8> to memref<80xi8, strided<[1], offset: ?>>
      %cast = memref.cast %subview : memref<800xi8, strided<[1], offset: ?>> to memref<800xi8>
      %cast_1 = memref.cast %subview_0 : memref<80xi8, strided<[1], offset: ?>> to memref<80xi8>
      func.call @mel_kernel_simple(%cast, %cast_1) : (memref<800xi8>, memref<80xi8>) -> ()
      %6 = arith.addi %2, %c1 : index
      cf.br ^bb3(%6 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%of_in_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_out_cons_lock_0, Release, 1)
      %7 = arith.addi %0, %c1 : index
      cf.br ^bb1(%7 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_with = "mel_fixed_combined.o"}
    aiex.runtime_sequence(%arg0: memref<24000xi8>, %arg1: memref<2400xi8>) {
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c24000_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, metadata = @of_in_shim_alloc} : memref<24000xi8>
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c2400_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @of_out_shim_alloc} : memref<2400xi8>
      aiex.npu.dma_wait {symbol = @of_out_shim_alloc}
    }
    aie.shim_dma_allocation @of_in_shim_alloc(MM2S, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%of_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_in_cons_buff_0 : memref<24000xi8>, 0, 24000) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%of_out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_out_buff_0 : memref<2400xi8>, 0, 2400) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_out_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    aie.shim_dma_allocation @of_out_shim_alloc(S2MM, 0, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
