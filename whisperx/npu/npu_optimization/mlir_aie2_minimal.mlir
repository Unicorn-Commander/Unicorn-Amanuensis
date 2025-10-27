// Minimal MLIR-AIE2 Kernel for WhisperX NPU
// Target: AMD Phoenix NPU (npu1_4col)
// Purpose: Simple vector multiply-add for attention computation

module @whisperx_minimal {
  aie.device(npu1_4col) {
    // Define compute tiles (row 2-5 have memory in AIE2)
    %tile02 = aie.tile(0, 2)
    %tile12 = aie.tile(1, 2)
    %tile22 = aie.tile(2, 2)
    %tile32 = aie.tile(3, 2)

    // Define shim tiles for DMA (row 0)
    %tile00 = aie.tile(0, 0)
    %tile10 = aie.tile(1, 0)

    // Buffers in compute tiles (tiles at row >= 2 have local memory)
    %buf0 = aie.buffer(%tile02) {sym_name = "buf0"} : memref<1024xi32>
    %buf1 = aie.buffer(%tile12) {sym_name = "buf1"} : memref<1024xi32>
    %buf2 = aie.buffer(%tile22) {sym_name = "buf2"} : memref<1024xi32>

    // Locks for synchronization
    %lock0 = aie.lock(%tile02, 0) {init = 1 : i32}
    %lock1 = aie.lock(%tile12, 0) {init = 0 : i32}
    %lock2 = aie.lock(%tile22, 0) {init = 0 : i32}

    // Core computation on tile02
    %core02 = aie.core(%tile02) {
      // Acquire lock
      aie.use_lock(%lock0, "Acquire", 1)

      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index

      // Simple loop to initialize buffer
      scf.for %i = %c0 to %c1024 step %c1 {
        %val = arith.constant 42 : i32
        memref.store %val, %buf0[%i] : memref<1024xi32>
      }

      // Release lock
      aie.use_lock(%lock0, "Release", 0)
      aie.end
    }

    // Core computation on tile12 - vector operations
    %core12 = aie.core(%tile12) {
      aie.use_lock(%lock1, "Acquire", 1)

      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c16 = arith.constant 16 : index

      // Vectorized computation
      scf.for %i = %c0 to %c1024 step %c16 {
        %vec = vector.load %buf1[%i] : memref<1024xi32>, vector<16xi32>
        %two = arith.constant 2 : i32
        %splat = vector.splat %two : vector<16xi32>
        %result = arith.muli %vec, %splat : vector<16xi32>
        vector.store %result, %buf1[%i] : memref<1024xi32>, vector<16xi32>
      }

      aie.use_lock(%lock1, "Release", 0)
      aie.end
    }

    // DMA configuration for shim tile
    %mem00 = aie.mem(%tile00) {
      %dma = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.end_bd ^end
    ^end:
      aie.end
    }

    // Flow between tiles
    aie.flow(%tile02, Core : 0, %tile12, Core : 0)
    aie.flow(%tile12, Core : 0, %tile22, Core : 0)
  }
}
