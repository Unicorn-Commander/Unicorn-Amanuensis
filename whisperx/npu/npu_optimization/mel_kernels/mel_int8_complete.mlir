//===- mel_int8_complete.mlir ----------------------------------*- MLIR -*-===//
//
// Complete MEL INT8 kernel with proper aie.mem DMA infrastructure
// Based on working passthrough_step3.mlir structure
//
//===----------------------------------------------------------------------===//

module @mel_int8_npu {
    aie.device(npu1) {
        // Declare tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile for host communication
        %tile02 = aie.tile(0, 2)  // Compute tile for processing

        // Output ObjectFIFO locks and buffers
        %of_out_cons_prod_lock = aie.lock(%tile00, 2) {init = 0 : i32, sym_name = "of_out_cons_prod_lock"}
        %of_out_cons_cons_lock = aie.lock(%tile00, 3) {init = 0 : i32, sym_name = "of_out_cons_cons_lock"}

        // Output buffers on compute tile (80 INT8 mel features)
        %of_out_buff_0 = aie.buffer(%tile02) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "of_out_buff_0"} : memref<80xi8>
        %of_out_buff_1 = aie.buffer(%tile02) {address = 2048 : i32, mem_bank = 1 : i32, sym_name = "of_out_buff_1"} : memref<80xi8>

        %of_out_prod_lock = aie.lock(%tile02, 2) {init = 2 : i32, sym_name = "of_out_prod_lock"}
        %of_out_cons_lock = aie.lock(%tile02, 3) {init = 0 : i32, sym_name = "of_out_cons_lock"}

        // Input buffers on compute tile (400 INT16 audio samples = 800 bytes)
        %of_in_cons_buff_0 = aie.buffer(%tile02) {address = 4096 : i32, mem_bank = 2 : i32, sym_name = "of_in_cons_buff_0"} : memref<800xi8>
        %of_in_cons_buff_1 = aie.buffer(%tile02) {address = 8192 : i32, mem_bank = 3 : i32, sym_name = "of_in_cons_buff_1"} : memref<800xi8>

        %of_in_cons_prod_lock = aie.lock(%tile02, 0) {init = 2 : i32, sym_name = "of_in_cons_prod_lock"}
        %of_in_cons_cons_lock = aie.lock(%tile02, 1) {init = 0 : i32, sym_name = "of_in_cons_cons_lock"}

        %of_in_prod_lock = aie.lock(%tile00, 0) {init = 0 : i32, sym_name = "of_in_prod_lock"}
        %of_in_cons_lock = aie.lock(%tile00, 1) {init = 0 : i32, sym_name = "of_in_cons_lock"}

        // Core with ELF file reference (test kernel with main())
        // When elf_file is specified, core body must be empty - kernel executes via DMA
        %core02 = aie.core(%tile02) {
            aie.end
        } { elf_file = "mel_kernel_test_main.o" }

        // Shim DMA allocations for runtime sequence
        aie.shim_dma_allocation @audioIn_shim (MM2S, 0, 0)
        aie.shim_dma_allocation @melOut_shim (S2MM, 0, 0)

        // Memory tile DMA infrastructure - THIS IS THE KEY MISSING PIECE!
        %mem02 = aie.mem(%tile02) {
            // S2MM: Stream-to-Memory-Mapped (input from shim to compute tile)
            %dma_in = aie.dma_start(S2MM, 0, ^bb_in1, ^bb_out0)
        ^bb_in1:
            aie.use_lock(%of_in_cons_prod_lock, AcquireGreaterEqual, 1)
            aie.dma_bd(%of_in_cons_buff_0 : memref<800xi8>, 0, 800) {bd_id = 0 : i32, next_bd_id = 1 : i32}
            aie.use_lock(%of_in_cons_cons_lock, Release, 1)
            aie.next_bd ^bb_in2
        ^bb_in2:
            aie.use_lock(%of_in_cons_prod_lock, AcquireGreaterEqual, 1)
            aie.dma_bd(%of_in_cons_buff_1 : memref<800xi8>, 0, 800) {bd_id = 1 : i32, next_bd_id = 0 : i32}
            aie.use_lock(%of_in_cons_cons_lock, Release, 1)
            aie.next_bd ^bb_in1
        ^bb_out0:
            // MM2S: Memory-Mapped-to-Stream (output from compute tile to shim)
            %dma_out = aie.dma_start(MM2S, 0, ^bb_out1, ^bb_end)
        ^bb_out1:
            aie.use_lock(%of_out_cons_lock, AcquireGreaterEqual, 1)
            aie.dma_bd(%of_out_buff_0 : memref<80xi8>, 0, 80) {bd_id = 2 : i32, next_bd_id = 3 : i32}
            aie.use_lock(%of_out_prod_lock, Release, 1)
            aie.next_bd ^bb_out2
        ^bb_out2:
            aie.use_lock(%of_out_cons_lock, AcquireGreaterEqual, 1)
            aie.dma_bd(%of_out_buff_1 : memref<80xi8>, 0, 80) {bd_id = 3 : i32, next_bd_id = 2 : i32}
            aie.use_lock(%of_out_prod_lock, Release, 1)
            aie.next_bd ^bb_out1
        ^bb_end:
            aie.end
        }

        // Runtime sequence for host-NPU data movement
        // Input: 400 INT16 samples = 800 bytes = 200 32-bit words
        // Output: 80 INT8 values = 80 bytes = 20 32-bit words
        aiex.runtime_sequence(%in : memref<200xi32>, %arg1 : memref<1xi32>, %out : memref<20xi32>) {
            %c0 = arith.constant 0 : i64
            %c1 = arith.constant 1 : i64
            %c200 = arith.constant 200 : i64  // Input size in 32-bit words (800 bytes / 4)
            %c20 = arith.constant 20 : i64    // Output size in 32-bit words (80 bytes / 4)

            // DMA memcpy for input audio data (MM2S - host to NPU)
            aiex.npu.dma_memcpy_nd (%in[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c200][%c0, %c0, %c0, %c1]) { metadata = @audioIn_shim, id = 1 : i64 } : memref<200xi32>

            // DMA memcpy for output mel features (S2MM - NPU to host)
            aiex.npu.dma_memcpy_nd (%out[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c20][%c0, %c0, %c0, %c1]) { metadata = @melOut_shim, id = 0 : i64 } : memref<20xi32>

            // Synchronization - wait for DMA completion
            aiex.npu.dma_wait {symbol = @melOut_shim}
        }

        // Switchbox configuration for routing data through tile interconnect
        %mem_tile_01 = aie.tile(0, 1)  // Memory tile between shim and compute

        %switchbox_00 = aie.switchbox(%tile00) {
            aie.connect<South : 3, North : 1>
            aie.connect<North : 0, South : 2>
        }

        %shim_mux_00 = aie.shim_mux(%tile00) {
            aie.connect<DMA : 0, North : 3>
            aie.connect<North : 2, DMA : 0>
        }

        %switchbox_01 = aie.switchbox(%mem_tile_01) {
            aie.connect<South : 1, North : 1>
            aie.connect<North : 0, South : 0>
        }

        %switchbox_02 = aie.switchbox(%tile02) {
            aie.connect<South : 1, DMA : 0>
            aie.connect<DMA : 0, South : 0>
        }

        // Wire connections between tiles
        aie.wire(%shim_mux_00 : North, %switchbox_00 : South)
        aie.wire(%tile00 : DMA, %shim_mux_00 : DMA)
        aie.wire(%mem_tile_01 : Core, %switchbox_01 : Core)
        aie.wire(%mem_tile_01 : DMA, %switchbox_01 : DMA)
        aie.wire(%switchbox_00 : North, %switchbox_01 : South)
        aie.wire(%tile02 : Core, %switchbox_02 : Core)
        aie.wire(%tile02 : DMA, %switchbox_02 : DMA)
        aie.wire(%switchbox_01 : North, %switchbox_02 : South)
    }
}
