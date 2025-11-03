module @mel_npu_batch100 {
  aie.device(npu1) {
    %c80_i64 = arith.constant 80 : i64
    %c800_i64 = arith.constant 800 : i64
    %c100_i64 = arith.constant 100 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    func.func private @mel_kernel_simple(memref<800xi8>, memref<80xi8>)
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %of_out_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "of_out_cons_prod_lock_0"}
    %of_out_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "of_out_cons_cons_lock_0"}
    %of_out_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "of_out_buff_0"} : memref<80xi8> 
    %of_out_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "of_out_buff_1"} : memref<80xi8> 
    %of_out_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "of_out_prod_lock_0"}
    %of_out_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "of_out_cons_lock_0"}
    %of_in_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "of_in_cons_buff_0"} : memref<800xi8> 
    %of_in_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "of_in_cons_buff_1"} : memref<800xi8> 
    %of_in_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "of_in_cons_prod_lock_0"}
    %of_in_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "of_in_cons_cons_lock_0"}
    %of_in_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "of_in_prod_lock_0"}
    %of_in_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "of_in_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c100 = arith.constant 100 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c100 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%of_in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_out_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @mel_kernel_simple(%of_in_cons_buff_0, %of_out_buff_0) : (memref<800xi8>, memref<80xi8>) -> ()
      aie.use_lock(%of_in_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_out_cons_lock_0, Release, 1)
      aie.use_lock(%of_in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_out_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @mel_kernel_simple(%of_in_cons_buff_1, %of_out_buff_1) : (memref<800xi8>, memref<80xi8>) -> ()
      aie.use_lock(%of_in_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_out_cons_lock_0, Release, 1)
      %4 = arith.addi %2, %c2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_with = "mel_fixed_combined.o"}
    aiex.runtime_sequence(%arg0: memref<80000xi8>, %arg1: memref<8000xi8>) {
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c100_i64, %c800_i64][%c0_i64, %c0_i64, %c800_i64, %c1_i64]) {id = 1 : i64, metadata = @of_in_shim_alloc} : memref<80000xi8>
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c100_i64, %c80_i64][%c0_i64, %c0_i64, %c80_i64, %c1_i64]) {id = 0 : i64, metadata = @of_out_shim_alloc} : memref<8000xi8>
      aiex.npu.dma_wait {symbol = @of_out_shim_alloc}
    }
    aie.shim_dma_allocation @of_in_shim_alloc(MM2S, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_in_cons_buff_0 : memref<800xi8>, 0, 800) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_in_cons_buff_1 : memref<800xi8>, 0, 800) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_out_buff_0 : memref<80xi8>, 0, 80) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%of_out_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_out_buff_1 : memref<80xi8>, 0, 80) {bd_id = 3 : i32, next_bd_id = 2 : i32}
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
