#!/usr/bin/env python3
"""
INT8 Attention Kernel for AMD Phoenix NPU (XDNA1) - IRON API

This demonstrates the IRON (MLIR-AIE Python API) approach for XDNA1 devices.
Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Device: Phoenix NPU (XDNA1) - NPU1Col1 (4 columns × 6 rows = 24 tiles)
Tile Size: 64×64 (matches existing C++ kernel: attention_int8_64x64_tiled.c)

Key XDNA1 vs XDNA2 Differences:
- Device: NPU1Col1 (4 columns) vs NPU2Col1 (8 columns)
- Memory: ~32KB per tile (both)
- Tile count: 24 compute cores vs 56+ compute cores

Based on XDNA2 patterns from:
- /home/ucadmin/mlir-aie-source/programming_examples/ml/conv2d/conv2d.py
- /home/ucadmin/mlir-aie-source/programming_examples/ml/resnet/layers_conv2_x/resnet.py
"""

import numpy as np
import sys

from aie.iron import (
    GlobalBuffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1  # Phoenix NPU = 4 columns
from aie.iron.controlflow import range_

def attention_64x64_xdna1(dev, batch_size: int = 1):
    """
    INT8 Attention kernel for 64×64 matrices on Phoenix NPU

    Args:
        dev: NPU1Col1 device instance
        batch_size: Number of attention heads to process (default: 1)

    Memory layout:
        Q: 64×64 int8 = 4096 bytes
        K: 64×64 int8 = 4096 bytes
        V: 64×64 int8 = 4096 bytes
        Output: 64×64 int8 = 4096 bytes
        Total: 16384 bytes (~16KB, fits in 32KB tile memory)
    """

    # Dimensions (64×64 attention block)
    seq_len = 64
    d_k = 64

    # Buffer sizes
    qkv_size = seq_len * d_k  # 4096 bytes per matrix
    combined_qkv_size = 3 * qkv_size  # 12288 bytes (Q+K+V packed)
    output_size = seq_len * d_k  # 4096 bytes

    # Type definitions
    # Combined QKV input (12288 bytes = 3 × 64×64)
    combined_qkv_ty = np.ndarray[(combined_qkv_size,), np.dtype[np.int8]]

    # Individual Q/K/V buffers (4096 bytes each)
    qkv_buffer_ty = np.ndarray[(qkv_size,), np.dtype[np.int8]]

    # Output buffer (4096 bytes)
    output_ty = np.ndarray[(output_size,), np.dtype[np.int8]]

    # C++ kernel declaration (links to attention_int8_64x64_tiled.c)
    # Takes combined QKV buffer and produces attention output
    attention_kernel = Kernel(
        "attention_64x64",  # Kernel function name
        "attention_int8_64x64_tiled.o",  # Compiled object file
        [
            combined_qkv_ty,  # Input: Combined Q+K+V buffer
            output_ty,        # Output: Attention result
            np.int32,         # scale: Scaling factor (sqrt(d_k))
        ],
    )

    # ObjectFIFO pattern (modern MLIR-AIE approach)
    # Input: Combined Q+K+V from host (L3) to NPU tile (L2)
    of_qkv_L3L2 = ObjectFifo(combined_qkv_ty, name="qkv_L3L2", depth=2)

    # Output: Attention result from NPU tile (L2) to host (L3)
    of_out_L2L3 = ObjectFifo(output_ty, name="out_L2L3", depth=2)

    # Runtime parameters (for dynamic scaling)
    rtp = GlobalBuffer(
        np.ndarray[(16,), np.dtype[np.int32]],
        name="rtp_attention",
        use_write_rtp=True,
    )

    # Barrier for synchronization
    rtp_barrier = WorkerRuntimeBarrier()

    # Core task: Process attention computation
    def attention_core_fn(of_qkv, of_out, my_rtp, kernel, barrier):
        """
        Core function that runs on NPU tile

        XDNA1 Note: Uses single tile (0,2) - Phoenix has 4 columns
        For multi-tile: distribute across (0,2), (1,2), (2,2), (3,2)
        """
        # Wait for runtime parameters to be set
        barrier.wait_for_value(1)

        # Get scaling factor (sqrt(d_k) = sqrt(64) = 8, so scale = 3 bit shift)
        scale = my_rtp[0]

        # Process batch (or single attention head)
        for _ in range_(batch_size):
            # Acquire input QKV buffer
            elem_qkv = of_qkv.acquire(1)

            # Acquire output buffer
            elem_out = of_out.acquire(1)

            # Call C++ attention kernel
            # Kernel computes: softmax(Q @ K^T / scale) @ V
            kernel(elem_qkv, elem_out, scale)

            # Release buffers
            of_qkv.release(1)
            of_out.release(1)

    # Create worker (maps to NPU compute tile)
    worker = Worker(
        attention_core_fn,
        [
            of_qkv_L3L2.cons(),   # Consumer of QKV input
            of_out_L2L3.prod(),   # Producer of output
            rtp,                   # Runtime parameters
            attention_kernel,      # C++ kernel reference
            rtp_barrier,           # Synchronization barrier
        ],
        stack_size=0x800,  # 2KB stack (conservative for XDNA1)
    )

    # Runtime sequence (host side)
    rt = Runtime()
    with rt.sequence(combined_qkv_ty, output_ty) as (QKV, OUT):
        # Set runtime parameters
        def set_rtps(my_rtp):
            my_rtp[0] = 3  # scale = 3 (shift by 3 bits = divide by 8 = sqrt(64))

        rt.inline_ops(set_rtps, [rtp])

        # Signal barrier
        rt.set_barrier(rtp_barrier, 1)

        # Start worker
        rt.start(worker)

        # DMA transfers
        rt.fill(of_qkv_L3L2.prod(), QKV)   # Host → NPU (input)
        rt.drain(of_out_L2L3.cons(), OUT, wait=True)  # NPU → Host (output)

    # Generate MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    """
    Usage:
        python3 attention_xdna1_iron.py npu > attention_64x64.mlir

    Then compile with aie-opt and aie-translate to generate XCLBIN.
    """

    # Parse arguments
    try:
        device_name = str(sys.argv[1]) if len(sys.argv) > 1 else "npu"
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1

        if device_name == "npu":
            # XDNA1: Phoenix NPU (4 columns)
            dev = NPU1Col1()
        else:
            raise ValueError(f"[ERROR] Device '{device_name}' not supported. Use 'npu' for XDNA1.")

        if batch_size < 1 or batch_size > 16:
            raise ValueError("Batch size must be between 1 and 16")

    except (IndexError, ValueError) as e:
        print(f"Usage: {sys.argv[0]} <device> [batch_size]", file=sys.stderr)
        print(f"  device: 'npu' (XDNA1 Phoenix)", file=sys.stderr)
        print(f"  batch_size: 1-16 (default: 1)", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate MLIR
    module = attention_64x64_xdna1(dev, batch_size)
    print(module)

    # Print info
    print("# Generated IRON API attention kernel for XDNA1 (Phoenix NPU)", file=sys.stderr)
    print(f"# Device: NPU1Col1 (4 columns)", file=sys.stderr)
    print(f"# Tile size: 64×64", file=sys.stderr)
    print(f"# Batch size: {batch_size}", file=sys.stderr)
    print(f"# Memory: ~16KB per tile", file=sys.stderr)
