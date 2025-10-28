//===- mel_simple_single_call.mlir -----------------------------*- MLIR -*-===//
//
// Simple MEL kernel - single call pattern (no loop in core)
// This pattern works with manual aie-opt compilation
//
//===----------------------------------------------------------------------===//

module @mel_simple_single {
  aie.device(npu1) {
    // Tiles
    %tile_0_0 = aie.tile(0, 0)  // Shim tile for host I/O
    %tile_0_2 = aie.tile(0, 2)  // Compute tile

    // External kernel function (compiled separately)
    func.func private @mel_kernel_simple(%arg0: memref<800xi8>, %arg1: memref<80xi8>)

    // Locks for input buffers
    %in_prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "in_prod_lock"}
    %in_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in_cons_lock"}

    // Locks for output buffers
    %out_prod_lock = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "out_prod_lock"}
    %out_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out_cons_lock"}

    // Buffers on compute tile
    %in_buffer = aie.buffer(%tile_0_2) {address = 4096 : i32, sym_name = "in_buffer"} : memref<800xi8>
    %out_buffer = aie.buffer(%tile_0_2) {address = 8192 : i32, sym_name = "out_buffer"} : memref<80xi8>

    // Simple core - single call, no loop
    %core = aie.core(%tile_0_2) {
      aie.use_lock(%in_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
      func.call @mel_kernel_simple(%in_buffer, %out_buffer) : (memref<800xi8>, memref<80xi8>) -> ()
      aie.use_lock(%in_prod_lock, Release, 1)
      aie.use_lock(%out_cons_lock, Release, 1)
      aie.end
    } {link_with = "mel_kernel_simple.o"}

    // Shim DMA allocations
    aie.shim_dma_allocation @in_shim (MM2S, 0, 0)
    aie.shim_dma_allocation @out_shim (S2MM, 0, 0)

    // Memory tile DMA
    %mem = aie.mem(%tile_0_2) {
      %dma_in = aie.dma_start(S2MM, 0, ^input, ^output)
    ^input:
      aie.use_lock(%in_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_buffer : memref<800xi8>, 0, 800)
      aie.use_lock(%in_cons_lock, Release, 1)
      aie.next_bd ^input
    ^output:
      %dma_out = aie.dma_start(MM2S, 0, ^output_bd, ^end)
    ^output_bd:
      aie.use_lock(%out_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buffer : memref<80xi8>, 0, 80)
      aie.use_lock(%out_prod_lock, Release, 1)
      aie.next_bd ^output_bd
    ^end:
      aie.end
    }

    // Runtime sequence
    aiex.runtime_sequence(%in_data: memref<800xi8>, %out_data: memref<80xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c800 = arith.constant 800 : i64
      %c80 = arith.constant 80 : i64

      // Send input data to NPU
      aiex.npu.dma_memcpy_nd(%in_data[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c800][%c0, %c0, %c0, %c1])
        {metadata = @in_shim, id = 1 : i64} : memref<800xi8>

      // Receive output data from NPU
      aiex.npu.dma_memcpy_nd(%out_data[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c80][%c0, %c0, %c0, %c1])
        {metadata = @out_shim, id = 0 : i64} : memref<80xi8>

      // Wait for completion
      aiex.npu.dma_wait {symbol = @out_shim}
    }

    // Switchbox routing
    %switchbox_0_0 = aie.switchbox(%tile_0_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<North : 0, South : 2>
    }

    %shim_mux = aie.shim_mux(%tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }

    %tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%tile_0_1) {
      aie.connect<South : 0, North : 0>
      aie.connect<North : 0, South : 0>
    }

    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
  }
}
