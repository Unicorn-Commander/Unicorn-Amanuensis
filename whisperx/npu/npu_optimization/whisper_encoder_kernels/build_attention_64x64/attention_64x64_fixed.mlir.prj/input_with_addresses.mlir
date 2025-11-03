module @attention_npu_64x64 {
  aie.device(npu1) {
    %c12288_i64 = arith.constant 12288 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    func.func private @attention_64x64(memref<12288xi8>, memref<4096xi8>, i32)
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %of_out_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "of_out_cons_prod_lock_0"}
    %of_out_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "of_out_cons_cons_lock_0"}
    %of_out_buff_0 = aie.buffer(%tile_0_2) {address = 25600 : i32, sym_name = "of_out_buff_0"} : memref<4096xi8> 
    %of_out_buff_1 = aie.buffer(%tile_0_2) {address = 29696 : i32, sym_name = "of_out_buff_1"} : memref<4096xi8> 
    %of_out_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "of_out_prod_lock_0"}
    %of_out_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "of_out_cons_lock_0"}
    %of_QKV_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "of_QKV_cons_buff_0"} : memref<12288xi8> 
    %of_QKV_cons_buff_1 = aie.buffer(%tile_0_2) {address = 13312 : i32, sym_name = "of_QKV_cons_buff_1"} : memref<12288xi8> 
    %of_QKV_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "of_QKV_cons_prod_lock_0"}
    %of_QKV_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "of_QKV_cons_cons_lock_0"}
    %of_QKV_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "of_QKV_prod_lock_0"}
    %of_QKV_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "of_QKV_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c4_i32 = arith.constant 4 : i32
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c4294967294 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_QKV_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_out_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @attention_64x64(%of_QKV_cons_buff_0, %of_out_buff_0, %c4_i32) : (memref<12288xi8>, memref<4096xi8>, i32) -> ()
      aie.use_lock(%of_QKV_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_out_cons_lock_0, Release, 1)
      aie.use_lock(%of_QKV_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_out_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @attention_64x64(%of_QKV_cons_buff_1, %of_out_buff_1, %c4_i32) : (memref<12288xi8>, memref<4096xi8>, i32) -> ()
      aie.use_lock(%of_QKV_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_out_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%of_QKV_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_out_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @attention_64x64(%of_QKV_cons_buff_0, %of_out_buff_0, %c4_i32) : (memref<12288xi8>, memref<4096xi8>, i32) -> ()
      aie.use_lock(%of_QKV_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_out_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "attention_combined_64x64.o"}
    aiex.runtime_sequence(%arg0: memref<12288xi8>, %arg1: memref<4096xi8>) {
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c12288_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, metadata = @of_QKV_shim_alloc} : memref<12288xi8>
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c4096_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @of_out_shim_alloc} : memref<4096xi8>
      aiex.npu.dma_wait {symbol = @of_out_shim_alloc}
    }
    aie.shim_dma_allocation @of_QKV_shim_alloc(MM2S, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_QKV_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_QKV_cons_buff_0 : memref<12288xi8>, 0, 12288) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_QKV_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_QKV_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_QKV_cons_buff_1 : memref<12288xi8>, 0, 12288) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_QKV_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_out_buff_0 : memref<4096xi8>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%of_out_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_out_buff_1 : memref<4096xi8>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%of_out_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @of_out_shim_alloc(S2MM, 0, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
