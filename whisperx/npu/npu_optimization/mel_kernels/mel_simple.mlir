// MLIR-AIE2 Wrapper for Simple Mel Spectrogram Kernel (Phase 2.1)
// Based on passthrough_step3.mlir working template
// Target: AMD Phoenix NPU (npu1, 4×6 tile array)

module {
  aie.device(npu1) {
    // Memory tiles and buffers
    // Using ObjectFIFO for modern data movement pattern

    // Tile definitions
    %tile_0_0 = aie.tile(0, 0)  // ShimNOC tile for DMA
    %tile_0_2 = aie.tile(0, 2)  // Compute tile for FFT processing

    // Buffer sizes for Phase 2.1
    // Input: 512 int16 samples (1024 bytes) + overhead = 2KB buffer
    // Output: 256 int32 values (1024 bytes) + overhead = 2KB buffer
    %buff_in_size = arith.constant 2048 : i32
    %buff_out_size = arith.constant 2048 : i32

    // Locks for synchronization
    %lock_in_prod = aie.lock(%tile_0_2, 0) {init = 1 : i32}   // Producer lock for input
    %lock_in_cons = aie.lock(%tile_0_2, 1) {init = 0 : i32}   // Consumer lock for input
    %lock_out_prod = aie.lock(%tile_0_2, 2) {init = 1 : i32}  // Producer lock for output
    %lock_out_cons = aie.lock(%tile_0_2, 3) {init = 0 : i32}  // Consumer lock for output

    // Buffers in tile local memory (64KB available)
    %buff_in_0 = aie.buffer(%tile_0_2) {sym_name = "input_buffer_0"} : memref<512xi16>
    %buff_in_1 = aie.buffer(%tile_0_2) {sym_name = "input_buffer_1"} : memref<512xi16>
    %buff_out_0 = aie.buffer(%tile_0_2) {sym_name = "output_buffer_0"} : memref<256xi32>
    %buff_out_1 = aie.buffer(%tile_0_2) {sym_name = "output_buffer_1"} : memref<256xi32>

    // ObjectFIFO for input data flow (host → compute tile)
    aie.objectfifo @in0(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<512xi16>>

    // ObjectFIFO for output data flow (compute tile → host)
    aie.objectfifo @out0(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    // Core definition - references compiled C kernel
    %core_0_2 = aie.core(%tile_0_2) {
      // Reference to compiled mel kernel object file
      // This will be compiled with Peano from mel_simple.c + fft_radix2.c
      %kernel_ref = aie.external_func "mel_simple_kernel"
        (memref<512xi16>, memref<256xi32>, i32) -> ()

      // Processing loop
      // For Phase 2.1, we process frames as they arrive
      aie.use_lock(%lock_in_cons, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_out_prod, AcquireGreaterEqual, 1)

      // Get input buffer
      %in_obj = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<512xi16>>
      %in_buff = aie.objectfifo.subview.access %in_obj[0] : !aie.objectfifosubview<memref<512xi16>> -> memref<512xi16>

      // Get output buffer
      %out_obj = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<256xi32>>
      %out_buff = aie.objectfifo.subview.access %out_obj[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>

      // Call kernel: mel_simple_kernel(input, output, num_frames=1)
      %c1 = arith.constant 1 : i32
      func.call @mel_simple_kernel(%in_buff, %out_buff, %c1) : (memref<512xi16>, memref<256xi32>, i32) -> ()

      // Release buffers
      aie.objectfifo.release @in0(Consume, 1)
      aie.objectfifo.release @out0(Produce, 1)

      aie.use_lock(%lock_in_prod, Release, 1)
      aie.use_lock(%lock_out_cons, Release, 1)

      aie.end
    } {elf_file = "mel_simple.o"}

    // Runtime sequence for DMA operations
    // This is called from host to execute the kernel
    func.func @sequence(%in: memref<512xi16>, %out: memref<256xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c512 = arith.constant 512 : i64
      %c256 = arith.constant 256 : i64

      // Shim DMA configuration for input (MM2S - Memory to Stream)
      aiex.npu.dma_memcpy_nd(0, 0, %in[%c0, %c0, %c0, %c0]
                                   [%c1, %c1, %c1, %c512]
                                   [%c0, %c0, %c0])
        {id = 0 : i64, metadata = @in0} : memref<512xi16>

      // Shim DMA configuration for output (S2MM - Stream to Memory)
      aiex.npu.dma_memcpy_nd(0, 0, %out[%c0, %c0, %c0, %c0]
                                    [%c1, %c1, %c1, %c256]
                                    [%c0, %c0, %c0])
        {id = 1 : i64, metadata = @out0} : memref<256xi32>

      // Synchronization - wait for completion
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, direction = 0 : i32, row = 0 : i32}

      return
    }

    // ShimDMA configuration
    %shim_dma = aie.shimDMA(%tile_0_0) {
      // MM2S channel 0 - input data
      %lock_mm2s = aie.lock(%tile_0_0, 0) {init = 1 : i32}
      aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lock_mm2s, AcquireGreaterEqual, 1)
      aie.dma_bd(%buff_in_0 : memref<512xi16>, 0, 512)
      aie.use_lock(%lock_mm2s, Release, 0)
      aie.next_bd ^bd0

      // S2MM channel 0 - output data
      %lock_s2mm = aie.lock(%tile_0_0, 1) {init = 0 : i32}
      aie.dma_start(S2MM, 0, ^bd1, ^end)
    ^bd1:
      aie.use_lock(%lock_s2mm, AcquireGreaterEqual, 1)
      aie.dma_bd(%buff_out_0 : memref<256xi32>, 0, 256)
      aie.use_lock(%lock_s2mm, Release, 0)
      aie.next_bd ^bd1

    ^end:
      aie.end
    }

    // Memory tile DMA (if needed for buffering)
    // For Phase 2.1, we use direct communication
    // This can be expanded in later phases

  } // end device
} // end module
