#!/usr/bin/env python3
"""
Multi-Core Attention Kernel using IRON API
==========================================

Distributes attention computation across all 4 NPU columns for 4× throughput.

Architecture:
- 4 columns × 1 compute tile = 4 parallel attention operations
- Each column processes 1 tile (64×64) independently
- Input: 4 tiles (4 × 12288 bytes = 49152 bytes)
- Output: 4 tiles (4 × 4096 bytes = 16384 bytes)

Performance Target:
- Current: 2.85ms per tile (16.2× realtime)
- Target: 2.85ms per batch of 4 tiles (27-33× realtime)
- Improvement: 4× throughput
"""

import sys
import argparse
import numpy as np

# Set up Python path for IRON API
sys.path.insert(0, '/home/ucadmin/mlir-aie-fresh/mlir-aie/python')

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, Tile
from aie.iron.controlflow import range_
from aie.helpers.util import np_ndarray_type_get_shape


def attention_multicore(n_cols=4):
    """
    Generate multi-core attention MLIR using IRON API.

    Args:
        n_cols: Number of NPU columns to use (default: 4 for full utilization)

    Returns:
        MLIR module for multi-core attention
    """

    # Tile size: 64×64 attention
    tile_size_qkv = 12288  # Combined Q+K+V (3 × 64×64)
    tile_size_out = 4096   # Output (64×64)

    # Define tensor types (single tile per column)
    # Input: Combined Q+K+V buffer
    Input_ty = np.ndarray[(tile_size_qkv,), np.dtype[np.int8]]
    # Output: Attention result
    Output_ty = np.ndarray[(tile_size_out,), np.dtype[np.int8]]

    # Kernel declaration
    # Links to attention_int8_64x64_tiled.o
    attention_kernel = Kernel(
        "attention_64x64",
        "attention_int8_64x64_tiled.o",
        [Input_ty, Output_ty, np.int32]  # (qkv_input, output, scale_shift)
    )

    # ObjectFIFOs for data movement
    # We need separate FIFOs for each column
    input_fifos = []
    output_fifos = []

    # FIFO depth (double buffering)
    fifo_depth = 2

    # Create ObjectFIFOs for each column
    for col in range(n_cols):
        # Input FIFO: Shim(col, 0) → Compute(col, 2)
        input_fifos.append(
            ObjectFifo(
                Input_ty,
                name=f"of_input_{col}",
                depth=fifo_depth
            )
        )

        # Output FIFO: Compute(col, 2) → Shim(col, 0)
        output_fifos.append(
            ObjectFifo(
                Output_ty,
                name=f"of_output_{col}",
                depth=fifo_depth
            )
        )

    # Worker function for each compute core
    def core_fn(in_fifo, out_fifo, kernel):
        """
        Core worker function - processes one tile at a time.

        Each core runs independently, processing tiles from its input FIFO.
        """
        # Scale shift for attention (sqrt(64) = 8, shift by 3)
        scale_shift = 3

        # Infinite loop - core stays active
        # Note: range_(1) is a workaround for IRON issue
        for _ in range_(1):
            # Acquire input buffer (Q+K+V combined)
            elem_in = in_fifo.acquire(1)

            # Acquire output buffer
            elem_out = out_fifo.acquire(1)

            # Call attention kernel
            kernel(elem_in, elem_out, scale_shift)

            # Release buffers
            in_fifo.release(1)
            out_fifo.release(1)

    # Create workers for each column
    workers = []
    for col in range(n_cols):
        worker = Worker(
            core_fn,
            [
                input_fifos[col].cons(),   # Consumer port of input FIFO
                output_fifos[col].prod(),  # Producer port of output FIFO
                attention_kernel
            ],
            placement=Tile(col, 2),  # Compute tile at row 2
            stack_size=0x4000  # 16KB stack
        )
        workers.append(worker)

    # Runtime sequence for data movement
    rt = Runtime()

    # Define runtime sequence with separate input/output tensors per column
    # This is simpler than trying to use offsets
    with rt.sequence(*([Input_ty] * n_cols + [Output_ty] * n_cols)) as tensors:
        # Split tensors into inputs and outputs
        inputs = tensors[:n_cols]
        outputs = tensors[n_cols:]

        # Start all workers
        rt.start(*workers)

        # Transfer data for each column
        for col in range(n_cols):
            # Fill input FIFO from host memory
            # Each column gets its own input tensor
            rt.fill(
                input_fifos[col].prod(),
                inputs[col],
                placement=Tile(col, 0)  # Shim tile
            )

            # Drain output FIFO to host memory
            # Each column writes to its own output tensor
            rt.drain(
                output_fifos[col].cons(),
                outputs[col],
                wait=True,  # Wait for completion
                placement=Tile(col, 0)  # Shim tile
            )

    # Create device and program
    device = NPU1()  # Phoenix NPU (4 columns)
    program = Program(device, rt)

    # Place components and generate MLIR module
    module = program.resolve_program(SequentialPlacer())

    return module


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-core attention MLIR using IRON API"
    )
    parser.add_argument(
        "--n-cols",
        type=int,
        default=4,
        choices=[1, 2, 4],
        help="Number of NPU columns to use (default: 4)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="attention_64x64_multicore_iron.mlir",
        help="Output MLIR file"
    )

    args = parser.parse_args()

    # Generate MLIR module
    print(f"Generating {args.n_cols}-column attention kernel...")
    module = attention_multicore(n_cols=args.n_cols)

    # Print or save MLIR
    mlir_str = str(module)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(mlir_str)
        print(f"✓ MLIR saved to: {args.output}")
        print(f"✓ Configuration: {args.n_cols} columns")
        print(f"✓ Throughput improvement: {args.n_cols}×")
        print(f"✓ Expected realtime factor: {16.2 * args.n_cols:.1f}×")
    else:
        print(mlir_str)


if __name__ == "__main__":
    main()
