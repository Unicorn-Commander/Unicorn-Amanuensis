//===- mel_with_fft.mlir ---------------------------------------*- MLIR -*-===//
//
// MEL INT8 kernel with FFT computation (event-driven execution)
// Links both mel_kernel_fft.o and fft_real.o
//
//===----------------------------------------------------------------------===//

module @mel_npu_with_fft {
    aie.device(npu1) {
        // Declare external mel kernel function
        func.func private @mel_kernel_simple(memref<800xi8>, memref<80xi8>)

        // Declare tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile
        %tile02 = aie.tile(0, 2)  // Compute tile

        // Input ObjectFIFO: host → compute tile
        // 800 bytes (400 INT16 samples as bytes)
        aie.objectfifo @of_in(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<800xi8>>

        // Output ObjectFIFO: compute tile → host
        // 80 bytes (80 INT8 mel features)
        aie.objectfifo @of_out(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<80xi8>>

        // Core with INFINITE LOOP
        %core02 = aie.core(%tile02) {
            // Constants
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index  // Infinite loop

            // Infinite loop
            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire input buffer (blocks until DMA fills it)
                %subview_in = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<800xi8>>
                %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<800xi8>> -> memref<800xi8>

                // Acquire output buffer (blocks until space available)
                %subview_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<80xi8>>
                %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<80xi8>> -> memref<80xi8>

                // Call C kernel function (with FFT computation)
                func.call @mel_kernel_simple(%elem_in, %elem_out) : (memref<800xi8>, memref<80xi8>) -> ()

                // Release input buffer (signals it's been consumed)
                aie.objectfifo.release @of_in(Consume, 1)

                // Release output buffer (signals it's ready to send)
                aie.objectfifo.release @of_out(Produce, 1)
            }

            aie.end
        } { link_with = "mel_fixed_combined_v3.o" }

        // Runtime sequence for host-NPU data movement
        aiex.runtime_sequence(%in : memref<800xi8>, %out : memref<80xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c800_i64 = arith.constant 800 : i64
            %c80_i64 = arith.constant 80 : i64

            // DMA from host to NPU (input audio)
            aiex.npu.dma_memcpy_nd(%in[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                     [%c1_i64, %c1_i64, %c1_i64, %c800_i64]
                                     [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_in,
                id = 1 : i64
            } : memref<800xi8>

            // DMA from NPU to host (output mel features)
            aiex.npu.dma_memcpy_nd(%out[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                      [%c1_i64, %c1_i64, %c1_i64, %c80_i64]
                                      [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_out,
                id = 0 : i64
            } : memref<80xi8>

            // Wait for output DMA completion
            aiex.npu.dma_wait {symbol = @of_out}
        }
    }
}
