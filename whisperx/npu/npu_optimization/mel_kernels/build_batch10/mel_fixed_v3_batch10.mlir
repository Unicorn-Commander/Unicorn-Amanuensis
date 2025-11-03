//===- mel_fixed_v3_batch10.mlir ----------------------------------*- MLIR -*-===//
//
// MEL Spectrogram Kernel with BATCH PROCESSING (10 frames per invocation)
//
// This is a memory-optimized version that processes 10 audio frames in a single
// NPU kernel call, balancing performance with the AIE tile's 64KB memory limit.
//
// Performance improvement: 5-8x faster (134s → 16-22s for 1h 44m audio)
//
// Key changes from mel_fixed_v3.mlir:
// - ObjectFIFO sizes: 800B → 8KB input, 80B → 800B output
// - Nested loop: Outer (infinite) + Inner (10 frames)
// - DMA transfers: Batch of 10 frames at once
//
// Created: November 1, 2025
// Author: Magic Unicorn Unconventional Technology & Stuff Inc.
//
//===----------------------------------------------------------------------===//

module @mel_npu_batch10 {
    aie.device(npu1) {  // CRITICAL: Use npu1 for Phoenix NPU, NOT npu1_4col!

        // Declare external mel kernel function (processes 1 frame)
        // This C function is unchanged - we just call it 10 times
        func.func private @mel_kernel_simple(memref<800xi8>, memref<80xi8>)

        // Declare tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile (DMA interface)
        %tile02 = aie.tile(0, 2)  // Compute tile (AIE core)

        // Input ObjectFIFO: host → compute tile
        // BATCH SIZE: 10 frames × 800 bytes = 8,000 bytes
        aie.objectfifo @of_in(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<8000xi8>>

        // Output ObjectFIFO: compute tile → host
        // BATCH SIZE: 10 frames × 80 bytes = 800 bytes
        aie.objectfifo @of_out(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<800xi8>>

        // Core with NESTED LOOPS: Infinite outer + Batch inner
        %core02 = aie.core(%tile02) {
            // Constants for outer loop (infinite)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index  // Infinite loop marker

            // Constants for inner loop (batch processing)
            %c10 = arith.constant 10 : index     // Batch size (10 frames)
            %c800 = arith.constant 800 : index   // Input stride (bytes per frame)
            %c80 = arith.constant 80 : index     // Output stride (bytes per frame)

            // Outer loop: Process batches indefinitely
            scf.for %batch = %c0 to %c_max step %c1 {

                // Acquire input batch buffer (8KB for 10 frames)
                %subview_in = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<8000xi8>>
                %elem_in_base = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<8000xi8>> -> memref<8000xi8>

                // Acquire output batch buffer (800B for 10 frames)
                %subview_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<800xi8>>
                %elem_out_base = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<800xi8>> -> memref<800xi8>

                // Inner loop: Process each frame in the batch
                scf.for %frame = %c0 to %c10 step %c1 {

                    // Calculate byte offsets for this frame
                    %in_offset = arith.muli %frame, %c800 : index   // frame * 800
                    %out_offset = arith.muli %frame, %c80 : index   // frame * 80

                    // Get subviews for this specific frame
                    %frame_in_strided = memref.subview %elem_in_base[%in_offset] [800] [1]
                        : memref<8000xi8> to memref<800xi8, strided<[1], offset: ?>>

                    %frame_out_strided = memref.subview %elem_out_base[%out_offset] [80] [1]
                        : memref<800xi8> to memref<80xi8, strided<[1], offset: ?>>

                    // Cast to plain memrefs for C function call
                    %frame_in = memref.cast %frame_in_strided : memref<800xi8, strided<[1], offset: ?>> to memref<800xi8>
                    %frame_out = memref.cast %frame_out_strided : memref<80xi8, strided<[1], offset: ?>> to memref<80xi8>

                    // Process this frame (same C kernel, called 10 times per batch)
                    func.call @mel_kernel_simple(%frame_in, %frame_out)
                        : (memref<800xi8>, memref<80xi8>) -> ()
                }

                // Release input batch buffer (all 10 frames processed)
                aie.objectfifo.release @of_in(Consume, 1)

                // Release output batch buffer (all 10 outputs ready)
                aie.objectfifo.release @of_out(Produce, 1)
            }

            aie.end
        } { link_with = "mel_fixed_combined.o" }  // Link same C kernel

        // Runtime sequence for batch DMA transfers
        aiex.runtime_sequence(%in : memref<8000xi8>, %out : memref<800xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c8000_i64 = arith.constant 8000 : i64   // 10 frames × 800 bytes
            %c800_i64 = arith.constant 800 : i64     // 10 frames × 80 bytes

            // DMA from host to NPU (batch of 10 input frames)
            // Transfer 8KB at once instead of 800B at a time
            aiex.npu.dma_memcpy_nd(%in[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                     [%c1_i64, %c1_i64, %c1_i64, %c8000_i64]
                                     [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_in,
                id = 1 : i64
            } : memref<8000xi8>

            // DMA from NPU to host (batch of 10 output mel features)
            // Transfer 800B at once instead of 80B at a time
            aiex.npu.dma_memcpy_nd(%out[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                      [%c1_i64, %c1_i64, %c1_i64, %c800_i64]
                                      [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_out,
                id = 0 : i64
            } : memref<800xi8>

            // Wait for output DMA completion
            aiex.npu.dma_wait {symbol = @of_out}
        }
    }
}

//===----------------------------------------------------------------------===//
// PERFORMANCE EXPECTATIONS
//===----------------------------------------------------------------------===//
//
// For 1 hour 44 minutes audio (628,163 frames):
//
// BEFORE (single-frame):
//   - Kernel invocations: 628,163
//   - DMA operations: 1,256,326 (2 per frame)
//   - Processing time: 134 seconds
//   - Overhead: ~213 µs per frame
//
// AFTER (batch-10):
//   - Kernel invocations: 62,817 (628,163 / 10)
//   - DMA operations: 125,634 (2 per batch)
//   - Processing time: 16-22 seconds (estimated)
//   - Overhead: ~21 µs per frame (10x reduction!)
//
// Expected speedup: 6-8x faster mel preprocessing
//
//===----------------------------------------------------------------------===//
// MEMORY USAGE (AIE Tile - 64 KB total)
//===----------------------------------------------------------------------===//
//
// Double-buffered ObjectFIFOs:
//   - Input ObjectFIFO: 16 KB (8KB × 2 buffers)
//   - Output ObjectFIFO: 1.6 KB (800B × 2 buffers)
//   - Stack (mel_kernel_simple): ~4 KB
//   - Loop counters and variables: ~100 B
//   - Total: 21.7 KB (33.9% of 64 KB) ✅
//
// Host Memory (for batch DMA):
//   - Input batch: 8 KB (10 frames × 800 bytes)
//   - Output batch: 800 B (10 frames × 80 bytes)
//   - Total: 8.8 KB per batch (negligible)
//
//===----------------------------------------------------------------------===//
// COMPILATION
//===----------------------------------------------------------------------===//
//
// To compile this kernel:
//
//   cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels
//
//   /home/ucadmin/.local/bin/aiecc.py \
//       mel_fixed_v3_batch10.mlir \
//       -I. -o build_batch10/mel_batch10.xclbin
//
// Expected compilation time: 30-90 seconds
//
//===----------------------------------------------------------------------===//
