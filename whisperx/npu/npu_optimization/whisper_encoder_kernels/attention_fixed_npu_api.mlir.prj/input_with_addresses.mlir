module {
  aie.device(npu1) {
    %c4096_i64 = arith.constant 4096 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %of_output_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "of_output_cons_prod_lock_0"}
    %of_output_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "of_output_cons_cons_lock_0"}
    %of_output_buff_0 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "of_output_buff_0"} : memref<4096xi8> 
    %of_output_buff_1 = aie.buffer(%tile_0_2) {address = 28672 : i32, mem_bank = 1 : i32, sym_name = "of_output_buff_1"} : memref<4096xi8> 
    %of_output_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "of_output_prod_lock_0"}
    %of_output_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "of_output_cons_lock_0"}
    %of_input_cons_buff_0 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "of_input_cons_buff_0"} : memref<12288xi8> 
    %of_input_cons_buff_1 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "of_input_cons_buff_1"} : memref<12288xi8> 
    %of_input_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "of_input_cons_prod_lock_0"}
    %of_input_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "of_input_cons_cons_lock_0"}
    %of_input_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "of_input_prod_lock_0"}
    %of_input_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "of_input_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    func.func private @attention_64x64(memref<12288xi8>, memref<4096xi8>, i32)
    %core_0_2 = aie.core(%tile_0_2) {
      %c3_i32 = arith.constant 3 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_output_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @attention_64x64(%of_input_cons_buff_0, %of_output_buff_0, %c3_i32) : (memref<12288xi8>, memref<4096xi8>, i32) -> ()
      aie.use_lock(%of_input_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_output_cons_lock_0, Release, 1)
      aie.use_lock(%of_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_output_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @attention_64x64(%of_input_cons_buff_1, %of_output_buff_1, %c3_i32) : (memref<12288xi8>, memref<4096xi8>, i32) -> ()
      aie.use_lock(%of_input_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_output_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%of_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_output_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @attention_64x64(%of_input_cons_buff_0, %of_output_buff_0, %c3_i32) : (memref<12288xi8>, memref<4096xi8>, i32) -> ()
      aie.use_lock(%of_input_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_output_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "attention_int8_64x64_tiled.o", stack_size = 16384 : i32}
    aiex.runtime_sequence(%arg0: memref<12288xi8>, %arg1: memref<4096xi8>) {
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c12288_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, metadata = @of_input_shim_alloc} : memref<12288xi8>
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c4096_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @of_output_shim_alloc} : memref<4096xi8>
      aiex.npu.dma_wait {symbol = @of_output_shim_alloc}
    }
    aie.shim_dma_allocation @of_input_shim_alloc(MM2S, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_input_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_cons_buff_0 : memref<12288xi8>, 0, 12288) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_input_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_input_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_input_cons_buff_1 : memref<12288xi8>, 0, 12288) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_input_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_output_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output_buff_0 : memref<4096xi8>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%of_output_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_output_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_output_buff_1 : memref<4096xi8>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%of_output_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @of_output_shim_alloc(S2MM, 0, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
