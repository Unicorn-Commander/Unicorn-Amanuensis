#!/usr/bin/env python3
"""
MEL Kernel AIE Design using Python IRON API
Based on working matrix_transpose example from MLIR-AIE
"""

import sys
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_

# Whisper mel spectrogram dimensions
# Input: 400 INT16 audio samples = 800 bytes
# Output: 80 INT8 mel features = 80 bytes

input_size = 800   # bytes (400 INT16 samples)
output_size = 80   # bytes (80 INT8 mel features)

def mel_design():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1)
        def device_body():
            # Type declarations
            input_ty = np.ndarray[(input_size,), np.dtype[np.int8]]   # 800 bytes
            output_ty = np.ndarray[(output_size,), np.dtype[np.int8]]  # 80 bytes

            # Declare external C kernel function
            mel_func = external_func(
                "mel_kernel_simple",
                inputs=[input_ty, output_ty]
            )

            # Tile declarations
            shim_tile = tile(0, 0)      # Shim NOC tile for host communication
            compute_tile = tile(0, 2)   # Compute tile for processing

            # Object FIFOs for data movement
            # Depth=2 for double buffering (one filling while one processing)
            fifo_in = object_fifo("fifo_in", shim_tile, compute_tile, 2, input_ty)
            fifo_out = object_fifo("fifo_out", compute_tile, shim_tile, 2, output_ty)

            # Core with infinite loop (THIS IS THE KEY!)
            @core(compute_tile, "mel_kernel_simple.o")
            def core_body():
                # Infinite loop: 0xFFFFFFFF iterations
                for _ in range_(0, 0xFFFFFFFF):
                    # Acquire input buffer (blocks until DMA fills it)
                    elem_in = fifo_in.acquire(ObjectFifoPort.Consume, 1)

                    # Acquire output buffer (blocks until space available)
                    elem_out = fifo_out.acquire(ObjectFifoPort.Produce, 1)

                    # Call C kernel function (pure computation)
                    mel_func(elem_in, elem_out)

                    # Release input buffer (signals DMA it's consumed)
                    fifo_in.release(ObjectFifoPort.Consume, 1)

                    # Release output buffer (signals DMA it's ready)
                    fifo_out.release(ObjectFifoPort.Produce, 1)

            # Runtime sequence for host-NPU data movement
            @runtime_sequence(input_ty, output_ty)
            def sequence(input_buffer, output_buffer):
                # DMA from host to NPU (input audio samples)
                npu_dma_memcpy_nd(
                    metadata=fifo_in,
                    bd_id=1,
                    mem=input_buffer,
                    offsets=[0, 0, 0, 0],
                    sizes=[1, 1, 1, input_size],   # 1D transfer of 800 bytes
                    strides=[0, 0, 0, 1],
                )

                # DMA from NPU to host (output mel features)
                npu_dma_memcpy_nd(
                    metadata=fifo_out,
                    bd_id=0,
                    mem=output_buffer,
                    offsets=[0, 0, 0, 0],
                    sizes=[1, 1, 1, output_size],  # 1D transfer of 80 bytes
                    strides=[0, 0, 0, 1],
                )

                # Wait for output DMA to complete
                dma_wait(fifo_out)

    # Print generated MLIR
    print(ctx.module)

if __name__ == "__main__":
    mel_design()
