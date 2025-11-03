module @mel_npu_batch10 {
  aie.device(npu1) {
    %c800_i64 = arith.constant 800 : i64
    %c8000_i64 = arith.constant 8000 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    func.func private @mel_kernel_simple(memref<800xi8>, memref<80xi8>)
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %of_out_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "of_out_cons_prod_lock_0"}
    %of_out_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "of_out_cons_cons_lock_0"}
    %of_out_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "of_out_buff_0"} : memref<800xi8> 
    %of_out_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "of_out_buff_1"} : memref<800xi8> 
    %of_out_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "of_out_prod_lock_0"}
    %of_out_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "of_out_cons_lock_0"}
    %of_in_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "of_in_cons_buff_0"} : memref<8000xi8> 
    %of_in_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "of_in_cons_buff_1"} : memref<8000xi8> 
    %of_in_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "of_in_cons_prod_lock_0"}
    %of_in_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "of_in_cons_cons_lock_0"}
    %of_in_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "of_in_prod_lock_0"}
    %of_in_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "of_in_cons_lock_0"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c10 = arith.constant 10 : index
      %c800 = arith.constant 800 : index
      %c80 = arith.constant 80 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c4294967294 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_out_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c10 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %4 = arith.muli %2, %c800 : index
      %5 = arith.muli %2, %c80 : index
      %subview = memref.subview %of_in_cons_buff_0[%4] [800] [1] : memref<8000xi8> to memref<800xi8, strided<[1], offset: ?>>
      %subview_0 = memref.subview %of_out_buff_0[%5] [80] [1] : memref<800xi8> to memref<80xi8, strided<[1], offset: ?>>
      %cast = memref.cast %subview : memref<800xi8, strided<[1], offset: ?>> to memref<800xi8>
      %cast_1 = memref.cast %subview_0 : memref<80xi8, strided<[1], offset: ?>> to memref<80xi8>
      func.call @mel_kernel_simple(%cast, %cast_1) : (memref<800xi8>, memref<80xi8>) -> ()
      %6 = arith.addi %2, %c1 : index
      cf.br ^bb3(%6 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%of_in_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_out_cons_lock_0, Release, 1)
      aie.use_lock(%of_in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_out_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb6(%c0 : index)
    ^bb6(%7: index):  // 2 preds: ^bb5, ^bb7
      %8 = arith.cmpi slt, %7, %c10 : index
      cf.cond_br %8, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      %9 = arith.muli %7, %c800 : index
      %10 = arith.muli %7, %c80 : index
      %subview_2 = memref.subview %of_in_cons_buff_1[%9] [800] [1] : memref<8000xi8> to memref<800xi8, strided<[1], offset: ?>>
      %subview_3 = memref.subview %of_out_buff_1[%10] [80] [1] : memref<800xi8> to memref<80xi8, strided<[1], offset: ?>>
      %cast_4 = memref.cast %subview_2 : memref<800xi8, strided<[1], offset: ?>> to memref<800xi8>
      %cast_5 = memref.cast %subview_3 : memref<80xi8, strided<[1], offset: ?>> to memref<80xi8>
      func.call @mel_kernel_simple(%cast_4, %cast_5) : (memref<800xi8>, memref<80xi8>) -> ()
      %11 = arith.addi %7, %c1 : index
      cf.br ^bb6(%11 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%of_in_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_out_cons_lock_0, Release, 1)
      %12 = arith.addi %0, %c2 : index
      cf.br ^bb1(%12 : index)
    ^bb9:  // pred: ^bb1
      aie.use_lock(%of_in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%of_out_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb10(%c0 : index)
    ^bb10(%13: index):  // 2 preds: ^bb9, ^bb11
      %14 = arith.cmpi slt, %13, %c10 : index
      cf.cond_br %14, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      %15 = arith.muli %13, %c800 : index
      %16 = arith.muli %13, %c80 : index
      %subview_6 = memref.subview %of_in_cons_buff_0[%15] [800] [1] : memref<8000xi8> to memref<800xi8, strided<[1], offset: ?>>
      %subview_7 = memref.subview %of_out_buff_0[%16] [80] [1] : memref<800xi8> to memref<80xi8, strided<[1], offset: ?>>
      %cast_8 = memref.cast %subview_6 : memref<800xi8, strided<[1], offset: ?>> to memref<800xi8>
      %cast_9 = memref.cast %subview_7 : memref<80xi8, strided<[1], offset: ?>> to memref<80xi8>
      func.call @mel_kernel_simple(%cast_8, %cast_9) : (memref<800xi8>, memref<80xi8>) -> ()
      %17 = arith.addi %13, %c1 : index
      cf.br ^bb10(%17 : index)
    ^bb12:  // pred: ^bb10
      aie.use_lock(%of_in_cons_prod_lock_0, Release, 1)
      aie.use_lock(%of_out_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "mel_fixed_combined.o"}
    aiex.runtime_sequence(%arg0: memref<8000xi8>, %arg1: memref<800xi8>) {
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c8000_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, metadata = @of_in_shim_alloc} : memref<8000xi8>
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c800_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @of_out_shim_alloc} : memref<800xi8>
      aiex.npu.dma_wait {symbol = @of_out_shim_alloc}
    }
    aie.shim_dma_allocation @of_in_shim_alloc(MM2S, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_in_cons_buff_0 : memref<8000xi8>, 0, 8000) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%of_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_in_cons_buff_1 : memref<8000xi8>, 0, 8000) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%of_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of_out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_out_buff_0 : memref<800xi8>, 0, 800) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%of_out_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of_out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%of_out_buff_1 : memref<800xi8>, 0, 800) {bd_id = 3 : i32, next_bd_id = 2 : i32}
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
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 0, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
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

