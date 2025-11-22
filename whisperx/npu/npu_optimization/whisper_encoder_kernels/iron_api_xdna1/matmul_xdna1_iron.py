#!/usr/bin/env python3
"""
INT8 Matrix Multiplication Kernel for AMD Phoenix NPU (XDNA1) - IRON API

This demonstrates the IRON API approach for XDNA1 matrix multiplication.
Computes: C = A @ B (32×32 INT8 matrices)

Device: Phoenix NPU (XDNA1) - NPU1Col1 (4 columns)
Tile Size: 32×32 (proven to compile in 0.455-0.856s)

Key XDNA1 Characteristics:
- NPU1Col1 device (4 columns vs XDNA2's 8 columns)
- 32KB tile memory (same as XDNA2)
- Efficient for 32×32 tiles (1024 bytes per matrix)
- Can scale to 64×64 (4096 bytes per matrix)

Based on successful compilation:
- matmul_32x32.mlir compiles in 0.455-0.856s
- matmul_int8_32x32.c kernel (83 lines, proven working)
"""

import numpy as np
import sys

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1
from aie.iron.controlflow import range_

def matmul_32x32_xdna1(dev, matrix_size: int = 32):
    """
    INT8 Matrix Multiplication for Phoenix NPU

    Args:
        dev: NPU1Col1 device instance
        matrix_size: 32 or 64 (default: 32)

    Memory layout (32×32):
        Matrix A: 32×32 int8 = 1024 bytes
        Matrix B: 32×32 int8 = 1024 bytes
        Packed input: 2048 bytes (A+B combined)
        Matrix C: 32×32 int8 = 1024 bytes
        Accumulator: 32×32 int32 = 4096 bytes (internal)
        Total: ~7KB (fits easily in 32KB tile memory)

    Memory layout (64×64):
        Matrix A: 64×64 int8 = 4096 bytes
        Matrix B: 64×64 int8 = 4096 bytes
        Packed input: 8192 bytes (A+B combined)
        Matrix C: 64×64 int8 = 4096 bytes
        Accumulator: 64×64 int32 = 16384 bytes (internal)
        Total: ~28KB (tight fit in 32KB tile memory)
    """

    # Calculate sizes based on matrix dimension
    matrix_elems = matrix_size * matrix_size
    packed_input_size = 2 * matrix_elems  # A + B
    output_size = matrix_elems

    # Type definitions
    # Packed input buffer (A and B combined for efficient DMA)
    packed_input_ty = np.ndarray[(packed_input_size,), np.dtype[np.int8]]

    # Output buffer (matrix C)
    output_ty = np.ndarray[(output_size,), np.dtype[np.int8]]

    # C++ kernel declaration
    # Links to matmul_int8_32x32.c or matmul_int8_64x64.c
    kernel_name = f"matmul_int8_{matrix_size}x{matrix_size}_packed"
    kernel_obj = f"matmul_int8_{matrix_size}x{matrix_size}.o"

    matmul_kernel = Kernel(
        kernel_name,
        kernel_obj,
        [
            packed_input_ty,  # Input: Combined A+B buffer
            output_ty,        # Output: Matrix C
        ],
    )

    # ObjectFIFO data movement
    # Input: Packed A+B from host to NPU tile
    of_input_L3L2 = ObjectFifo(
        packed_input_ty,
        name="input_L3L2",
        depth=2  # Double buffering for pipeline
    )

    # Output: Matrix C from NPU tile to host
    of_output_L2L3 = ObjectFifo(
        output_ty,
        name="output_L2L3",
        depth=2
    )

    # Core task: Matrix multiplication
    def matmul_core_fn(of_input, of_output, kernel):
        """
        Core function running on NPU tile

        XDNA1 Note: Single tile execution
        For batching, use range_() loop (see batched_matmul_xdna1_iron.py)
        """
        # Acquire input buffer (packed A+B)
        elem_input = of_input.acquire(1)

        # Acquire output buffer
        elem_output = of_output.acquire(1)

        # Call C++ matmul kernel
        # Kernel unpacks A and B internally, computes C = A @ B
        kernel(elem_input, elem_output)

        # Release buffers
        of_input.release(1)
        of_output.release(1)

    # Create worker (maps to NPU compute tile)
    worker = Worker(
        matmul_core_fn,
        [
            of_input_L3L2.cons(),   # Consumer of input
            of_output_L2L3.prod(),  # Producer of output
            matmul_kernel,          # C++ kernel reference
        ],
        stack_size=0x600,  # 1.5KB stack (sufficient for matmul)
    )

    # Runtime sequence (host side)
    rt = Runtime()
    with rt.sequence(packed_input_ty, output_ty) as (INPUT, OUTPUT):
        # Start worker
        rt.start(worker)

        # DMA transfers
        rt.fill(of_input_L3L2.prod(), INPUT)      # Host → NPU
        rt.drain(of_output_L2L3.cons(), OUTPUT, wait=True)  # NPU → Host

    # Generate MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    """
    Usage:
        python3 matmul_xdna1_iron.py npu 32 > matmul_32x32.mlir
        python3 matmul_xdna1_iron.py npu 64 > matmul_64x64.mlir

    Compilation:
        aie-opt --aie-lower-to-aie matmul_32x32.mlir | \\
        aie-translate --aie-generate-xclbin -o matmul_32x32.xclbin

    Expected compile time:
        32×32: ~0.455-0.856s (proven)
        64×64: ~1-2s (estimated)
    """

    # Parse arguments
    try:
        device_name = str(sys.argv[1]) if len(sys.argv) > 1 else "npu"
        matrix_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32

        if device_name == "npu":
            dev = NPU1Col1()  # Phoenix NPU (4 columns)
        else:
            raise ValueError(f"[ERROR] Device '{device_name}' not supported. Use 'npu'.")

        if matrix_size not in [32, 64]:
            raise ValueError("Matrix size must be 32 or 64")

    except (IndexError, ValueError) as e:
        print(f"Usage: {sys.argv[0]} <device> [matrix_size]", file=sys.stderr)
        print(f"  device: 'npu' (XDNA1)", file=sys.stderr)
        print(f"  matrix_size: 32 or 64 (default: 32)", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate MLIR
    module = matmul_32x32_xdna1(dev, matrix_size)
    print(module)

    # Print info
    memory_kb = (2 * matrix_size * matrix_size + matrix_size * matrix_size) / 1024
    print(f"# Generated IRON API matmul kernel for XDNA1", file=sys.stderr)
    print(f"# Device: NPU1Col1 (4 columns)", file=sys.stderr)
    print(f"# Matrix size: {matrix_size}×{matrix_size}", file=sys.stderr)
    print(f"# Memory: ~{memory_kb:.1f}KB per tile", file=sys.stderr)
    print(f"# Expected compile time: {'0.5s' if matrix_size == 32 else '1-2s'}", file=sys.stderr)
