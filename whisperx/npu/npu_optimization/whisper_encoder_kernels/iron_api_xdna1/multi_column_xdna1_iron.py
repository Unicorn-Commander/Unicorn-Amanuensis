#!/usr/bin/env python3
"""
Multi-Column Distribution for AMD Phoenix NPU (XDNA1) - IRON API

Demonstrates how to distribute Whisper encoder operations across Phoenix's 4 columns.

Phoenix NPU Architecture (XDNA1):
    NPU1Col1 device = 4 columns × 6 rows = 24 compute tiles
    Column 0: Tiles (0,2), (0,3), (0,4), (0,5) - 4 compute cores
    Column 1: Tiles (1,2), (1,3), (1,4), (1,5) - 4 compute cores
    Column 2: Tiles (2,2), (2,3), (2,4), (2,5) - 4 compute cores
    Column 3: Tiles (3,2), (3,3), (3,4), (3,5) - 4 compute cores

Strategy: Distribute encoder components across columns
    Column 0: Attention mechanism (Q@K^T, softmax)
    Column 1: Attention value multiplication (softmax @ V)
    Column 2: FFN Layer 1 (first linear + GELU)
    Column 3: FFN Layer 2 + LayerNorm

Benefits:
    - 4x parallelism (all columns active simultaneously)
    - Reduced memory pressure (distributed across tiles)
    - Pipeline efficiency (data flows column-to-column)

Based on ResNet example:
    - Explicit tile placement with Tile(col, row)
    - Data sharing via ObjectFIFO forwarding
    - Snake pattern for efficient memory access
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
from aie.iron.device import NPU1Col1, Tile
from aie.iron.controlflow import range_

def whisper_encoder_multi_column_xdna1(dev, seq_len: int = 64, d_model: int = 64):
    """
    Whisper encoder block distributed across 4 Phoenix NPU columns

    Args:
        dev: NPU1Col1 device instance
        seq_len: Sequence length (default: 64)
        d_model: Model dimension (default: 64)

    Architecture:
        Input (seq_len × d_model) →
            [Column 0] Attention QK^T + Softmax →
            [Column 1] Attention output (softmax @ V) →
            [Column 2] FFN Layer 1 + GELU →
            [Column 3] FFN Layer 2 + LayerNorm →
        Output (seq_len × d_model)
    """

    # Buffer sizes (64×64 example)
    hidden_size = seq_len * d_model  # 4096 bytes
    qkv_size = 3 * hidden_size       # 12288 bytes (Q+K+V)
    ffn_hidden = 4 * d_model         # 256 (FFN expansion factor = 4)

    # Type definitions
    hidden_ty = np.ndarray[(hidden_size,), np.dtype[np.int8]]
    qkv_ty = np.ndarray[(qkv_size,), np.dtype[np.int8]]
    ffn_hidden_ty = np.ndarray[(seq_len * ffn_hidden,), np.dtype[np.int8]]

    # Kernel declarations
    attention_qk_kernel = Kernel(
        "attention_qk_softmax",
        "attention_qk.o",
        [qkv_ty, hidden_ty, np.int32]  # QKV → attention weights
    )

    attention_v_kernel = Kernel(
        "attention_value_multiply",
        "attention_v.o",
        [hidden_ty, qkv_ty, hidden_ty]  # weights + V → output
    )

    ffn1_kernel = Kernel(
        "ffn_layer1_gelu",
        "ffn1.o",
        [hidden_ty, ffn_hidden_ty]  # hidden → FFN hidden
    )

    ffn2_kernel = Kernel(
        "ffn_layer2_layernorm",
        "ffn2.o",
        [ffn_hidden_ty, hidden_ty]  # FFN hidden → output
    )

    # ObjectFIFOs for data flow
    # Input: Host → Column 0
    of_input = ObjectFifo(qkv_ty, name="input_L3L2")

    # Column 0 → Column 1: Attention weights
    of_attn_weights = ObjectFifo(
        hidden_ty,
        name="attn_weights_col0_col1",
        depth=2
    )

    # Column 1 → Column 2: Attention output
    of_attn_out = ObjectFifo(
        hidden_ty,
        name="attn_out_col1_col2",
        depth=2
    )

    # Column 2 → Column 3: FFN intermediate
    of_ffn_intermediate = ObjectFifo(
        ffn_hidden_ty,
        name="ffn_intermediate_col2_col3",
        depth=2
    )

    # Column 3 → Host: Final output
    of_output = ObjectFifo(
        hidden_ty,
        name="output_L2L3"
    )

    # Explicit tile placement (ResNet pattern)
    # Use specific tiles for deterministic placement
    col0_tile = Tile(0, 2)  # Column 0, Row 2
    col1_tile = Tile(1, 2)  # Column 1, Row 2
    col2_tile = Tile(2, 2)  # Column 2, Row 2
    col3_tile = Tile(3, 2)  # Column 3, Row 2

    # Worker 1: Attention QK + Softmax (Column 0)
    def attn_qk_fn(of_in, of_out, kernel):
        """Compute attention weights: softmax(Q @ K^T / sqrt(d))"""
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)

        scale = 8  # sqrt(64) = 8
        kernel(elem_in, elem_out, scale)

        of_in.release(1)
        of_out.release(1)

    worker_col0 = Worker(
        attn_qk_fn,
        [of_input.cons(), of_attn_weights.prod(), attention_qk_kernel],
        placement=col0_tile,  # Explicit placement
        stack_size=0x600
    )

    # Worker 2: Attention Value Multiply (Column 1)
    def attn_v_fn(of_weights, of_qkv, of_out, kernel):
        """Multiply attention weights by V: attn_weights @ V"""
        elem_weights = of_weights.acquire(1)
        elem_qkv = of_qkv.acquire(1)  # Need V from QKV
        elem_out = of_out.acquire(1)

        kernel(elem_weights, elem_qkv, elem_out)

        of_weights.release(1)
        of_qkv.release(1)
        of_out.release(1)

    # Forward QKV to column 1 (need V for multiplication)
    of_input_fwd = of_input.cons().forward(name="qkv_fwd_col1")

    worker_col1 = Worker(
        attn_v_fn,
        [
            of_attn_weights.cons(),
            of_input_fwd.cons(),
            of_attn_out.prod(),
            attention_v_kernel
        ],
        placement=col1_tile,
        stack_size=0x600
    )

    # Worker 3: FFN Layer 1 + GELU (Column 2)
    def ffn1_fn(of_in, of_out, kernel):
        """FFN expansion + GELU activation"""
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)

        kernel(elem_in, elem_out)

        of_in.release(1)
        of_out.release(1)

    worker_col2 = Worker(
        ffn1_fn,
        [of_attn_out.cons(), of_ffn_intermediate.prod(), ffn1_kernel],
        placement=col2_tile,
        stack_size=0x600
    )

    # Worker 4: FFN Layer 2 + LayerNorm (Column 3)
    def ffn2_fn(of_in, of_out, kernel):
        """FFN projection + LayerNorm"""
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)

        kernel(elem_in, elem_out)

        of_in.release(1)
        of_out.release(1)

    worker_col3 = Worker(
        ffn2_fn,
        [of_ffn_intermediate.cons(), of_output.prod(), ffn2_kernel],
        placement=col3_tile,
        stack_size=0x600
    )

    # Runtime sequence
    rt = Runtime()
    with rt.sequence(qkv_ty, hidden_ty) as (INPUT, OUTPUT):
        # Start all workers (parallel execution across columns)
        rt.start(worker_col0, worker_col1, worker_col2, worker_col3)

        # DMA transfers
        rt.fill(of_input.prod(), INPUT, placement=Tile(0, 0))  # Shim tile for col 0
        rt.drain(of_output.cons(), OUTPUT, placement=Tile(3, 0), wait=True)  # Shim tile for col 3

    return Program(dev, rt).resolve_program(SequentialPlacer())


def whisper_encoder_snake_pattern(dev):
    """
    Snake pattern distribution (ResNet-inspired)

    Column flow: 0 → 1 → 2 → 3
    Row flow:    2 → 3 → 4 → 5 (within column)

    Pattern:
        Col 0: (0,2) → (0,3) → (0,4) → (0,5)  ⤵
        Col 1: (1,5) → (1,4) → (1,3) → (1,2)  ⤵  (reversed!)
        Col 2: (2,2) → (2,3) → (2,4) → (2,5)  ⤵
        Col 3: (3,5) → (3,4) → (3,3) → (3,2)  ⤵  (reversed!)

    Benefits:
        - Shared memory access between adjacent tiles
        - Reduced wire length for data transfers
        - Better cache locality
    """

    # Define snake tile pattern
    tiles = [
        # Column 0 (ascending rows)
        [Tile(0, 2), Tile(0, 3), Tile(0, 4), Tile(0, 5)],
        # Column 1 (descending rows)
        [Tile(1, 5), Tile(1, 4), Tile(1, 3), Tile(1, 2)],
        # Column 2 (ascending rows)
        [Tile(2, 2), Tile(2, 3), Tile(2, 4), Tile(2, 5)],
        # Column 3 (descending rows)
        [Tile(3, 5), Tile(3, 4), Tile(3, 3), Tile(3, 2)],
    ]

    # This pattern enables:
    # - 16 parallel workers (4 columns × 4 rows)
    # - Efficient memory sharing
    # - Minimal wire congestion

    print("# Snake pattern tile layout:", file=sys.stderr)
    for col_idx, col_tiles in enumerate(tiles):
        print(f"# Column {col_idx}: {[f'({t.col},{t.row})' for t in col_tiles]}", file=sys.stderr)

    return tiles


if __name__ == "__main__":
    """
    Usage:
        python3 multi_column_xdna1_iron.py npu > encoder_multi_column.mlir

    Architecture:
        4 columns working in parallel
        Each column handles different encoder operation
        Data flows: Col0 → Col1 → Col2 → Col3

    Expected benefits:
        - 4x parallelism vs single column
        - Better memory distribution
        - Reduced execution time
    """

    try:
        device_name = str(sys.argv[1]) if len(sys.argv) > 1 else "npu"
        seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 64
        d_model = int(sys.argv[3]) if len(sys.argv) > 3 else 64

        if device_name != "npu":
            raise ValueError("Only 'npu' device supported (XDNA1)")

        dev = NPU1Col1()

        # Generate multi-column encoder
        module = whisper_encoder_multi_column_xdna1(dev, seq_len, d_model)
        print(module)

        # Print snake pattern info
        snake_tiles = whisper_encoder_snake_pattern(dev)

        print(f"# Multi-column distribution for Phoenix NPU", file=sys.stderr)
        print(f"# Device: NPU1Col1 (4 columns × 6 rows)", file=sys.stderr)
        print(f"# Strategy: Attention→FFN pipeline across columns", file=sys.stderr)
        print(f"# Parallelism: 4x (all columns active)", file=sys.stderr)

    except (IndexError, ValueError) as e:
        print(f"Usage: {sys.argv[0]} npu [seq_len] [d_model]", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
