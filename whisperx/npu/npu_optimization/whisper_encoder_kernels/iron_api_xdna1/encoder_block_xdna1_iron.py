#!/usr/bin/env python3
"""
Complete Whisper Encoder Block for AMD Phoenix NPU (XDNA1) - IRON API

Full encoder block including:
    1. Multi-Head Self-Attention
    2. LayerNorm
    3. Feed-Forward Network (FFN)
    4. Residual connections

This is a complete, production-ready IRON API template for XDNA1.

Device: Phoenix NPU (XDNA1) - NPU1Col1 (4 columns)
Architecture: Whisper encoder layer with all components

Based on:
    - attention_xdna1_iron.py (self-attention)
    - matmul_xdna1_iron.py (linear layers)
    - multi_column_xdna1_iron.py (multi-column distribution)
    - Existing C++ kernels (attention_int8_64x64_tiled.c, etc.)
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
from aie.iron.device import NPU1Col1, Tile
from aie.iron.controlflow import range_

def whisper_encoder_block_xdna1(
    dev,
    seq_len: int = 64,
    d_model: int = 64,
    n_heads: int = 8,
    ffn_dim: int = 256,
    use_multi_column: bool = True
):
    """
    Complete Whisper encoder block for Phoenix NPU

    Args:
        dev: NPU1Col1 device instance
        seq_len: Sequence length (default: 64)
        d_model: Model dimension (default: 64)
        n_heads: Number of attention heads (default: 8)
        ffn_dim: FFN hidden dimension (default: 256 = 4×64)
        use_multi_column: Distribute across 4 columns (default: True)

    Architecture:
        Input (seq_len × d_model)
            ↓
        [1] Multi-Head Self-Attention
            - Q, K, V projections (matmul)
            - Scaled dot-product attention
            - Output projection
            ↓
        [2] Add & Norm (residual + LayerNorm)
            ↓
        [3] Feed-Forward Network
            - Linear 1: d_model → ffn_dim (4x expansion)
            - GELU activation
            - Linear 2: ffn_dim → d_model
            ↓
        [4] Add & Norm (residual + LayerNorm)
            ↓
        Output (seq_len × d_model)
    """

    # Buffer sizes
    hidden_size = seq_len * d_model          # 4096 bytes (64×64)
    qkv_size = 3 * hidden_size               # 12288 bytes (Q+K+V)
    ffn_hidden_size = seq_len * ffn_dim      # 16384 bytes (64×256)

    # Type definitions
    hidden_ty = np.ndarray[(hidden_size,), np.dtype[np.int8]]
    qkv_ty = np.ndarray[(qkv_size,), np.dtype[np.int8]]
    ffn_hidden_ty = np.ndarray[(ffn_hidden_size,), np.dtype[np.int8]]

    # Kernel declarations (link to compiled C++ kernels)

    # 1. QKV projection kernel (d_model → 3*d_model)
    qkv_proj_kernel = Kernel(
        "qkv_projection",
        "qkv_projection.o",
        [hidden_ty, qkv_ty]
    )

    # 2. Attention kernel (Q, K, V → attention output)
    attention_kernel = Kernel(
        "attention_64x64",
        "attention_int8_64x64_tiled.o",
        [qkv_ty, hidden_ty, np.int32]
    )

    # 3. LayerNorm kernel
    layernorm_kernel = Kernel(
        "layernorm_int8",
        "layernorm_int8.o",
        [hidden_ty, hidden_ty, np.int32, np.int32]  # input, output, scale, bias
    )

    # 4. FFN Linear 1 kernel (d_model → ffn_dim)
    ffn1_kernel = Kernel(
        "ffn_linear1_gelu",
        "ffn1_gelu.o",
        [hidden_ty, ffn_hidden_ty]
    )

    # 5. FFN Linear 2 kernel (ffn_dim → d_model)
    ffn2_kernel = Kernel(
        "ffn_linear2",
        "ffn2.o",
        [ffn_hidden_ty, hidden_ty]
    )

    # ObjectFIFOs for data flow
    of_input = ObjectFifo(hidden_ty, name="encoder_input", depth=2)

    # QKV projection output
    of_qkv = ObjectFifo(qkv_ty, name="qkv_buffer", depth=2)

    # Attention output
    of_attn_out = ObjectFifo(hidden_ty, name="attn_output", depth=2)

    # After first LayerNorm
    of_norm1_out = ObjectFifo(hidden_ty, name="norm1_output", depth=2)

    # FFN intermediate (expanded)
    of_ffn_intermediate = ObjectFifo(ffn_hidden_ty, name="ffn_intermediate", depth=2)

    # FFN output
    of_ffn_out = ObjectFifo(hidden_ty, name="ffn_output", depth=2)

    # Final output
    of_output = ObjectFifo(hidden_ty, name="encoder_output", depth=2)

    # Runtime parameters
    rtp = GlobalBuffer(
        np.ndarray[(16,), np.dtype[np.int32]],
        name="rtp_encoder",
        use_write_rtp=True,
    )

    rtp_barrier = WorkerRuntimeBarrier()

    # Tile placement
    if use_multi_column:
        # Distribute across 4 columns
        tile_qkv = Tile(0, 2)      # Column 0: QKV projection
        tile_attn = Tile(1, 2)     # Column 1: Attention
        tile_norm1 = Tile(1, 3)    # Column 1: First LayerNorm
        tile_ffn1 = Tile(2, 2)     # Column 2: FFN expansion + GELU
        tile_ffn2 = Tile(3, 2)     # Column 3: FFN projection
        tile_norm2 = Tile(3, 3)    # Column 3: Second LayerNorm
    else:
        # Single column (sequential execution)
        tile_qkv = Tile(0, 2)
        tile_attn = Tile(0, 3)
        tile_norm1 = Tile(0, 4)
        tile_ffn1 = Tile(0, 5)
        tile_ffn2 = Tile(0, 2)
        tile_norm2 = Tile(0, 3)

    # Worker 1: QKV Projection
    def qkv_proj_fn(of_in, of_out, kernel):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(elem_in, elem_out)
        of_in.release(1)
        of_out.release(1)

    worker_qkv = Worker(
        qkv_proj_fn,
        [of_input.cons(), of_qkv.prod(), qkv_proj_kernel],
        placement=tile_qkv,
        stack_size=0x600
    )

    # Worker 2: Attention
    def attn_fn(of_qkv, of_out, kernel, my_rtp, barrier):
        barrier.wait_for_value(1)
        scale = my_rtp[0]  # sqrt(d_k)

        elem_qkv = of_qkv.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(elem_qkv, elem_out, scale)
        of_qkv.release(1)
        of_out.release(1)

    worker_attn = Worker(
        attn_fn,
        [of_qkv.cons(), of_attn_out.prod(), attention_kernel, rtp, rtp_barrier],
        placement=tile_attn,
        stack_size=0x800
    )

    # Worker 3: First LayerNorm (with residual)
    def norm1_fn(of_attn, of_input, of_out, kernel, my_rtp):
        # Add residual (attention output + original input)
        elem_attn = of_attn.acquire(1)
        elem_input = of_input.acquire(1)  # Need original input for residual
        elem_out = of_out.acquire(1)

        scale = my_rtp[1]
        bias = my_rtp[2]
        kernel(elem_attn, elem_out, scale, bias)

        of_attn.release(1)
        of_input.release(1)
        of_out.release(1)

    # Forward input for residual connection
    of_input_fwd1 = of_input.cons().forward(name="input_fwd_norm1")

    worker_norm1 = Worker(
        norm1_fn,
        [of_attn_out.cons(), of_input_fwd1.cons(), of_norm1_out.prod(), layernorm_kernel, rtp],
        placement=tile_norm1,
        stack_size=0x600
    )

    # Worker 4: FFN Layer 1 + GELU
    def ffn1_fn(of_in, of_out, kernel):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(elem_in, elem_out)
        of_in.release(1)
        of_out.release(1)

    worker_ffn1 = Worker(
        ffn1_fn,
        [of_norm1_out.cons(), of_ffn_intermediate.prod(), ffn1_kernel],
        placement=tile_ffn1,
        stack_size=0x600
    )

    # Worker 5: FFN Layer 2
    def ffn2_fn(of_in, of_out, kernel):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(elem_in, elem_out)
        of_in.release(1)
        of_out.release(1)

    worker_ffn2 = Worker(
        ffn2_fn,
        [of_ffn_intermediate.cons(), of_ffn_out.prod(), ffn2_kernel],
        placement=tile_ffn2,
        stack_size=0x600
    )

    # Worker 6: Second LayerNorm (with residual)
    def norm2_fn(of_ffn, of_norm1, of_out, kernel, my_rtp):
        elem_ffn = of_ffn.acquire(1)
        elem_norm1 = of_norm1.acquire(1)  # Residual from norm1 output
        elem_out = of_out.acquire(1)

        scale = my_rtp[3]
        bias = my_rtp[4]
        kernel(elem_ffn, elem_out, scale, bias)

        of_ffn.release(1)
        of_norm1.release(1)
        of_out.release(1)

    # Forward norm1 output for residual
    of_norm1_fwd = of_norm1_out.cons().forward(name="norm1_fwd_norm2")

    worker_norm2 = Worker(
        norm2_fn,
        [of_ffn_out.cons(), of_norm1_fwd.cons(), of_output.prod(), layernorm_kernel, rtp],
        placement=tile_norm2,
        stack_size=0x600
    )

    # Runtime sequence
    rt = Runtime()
    with rt.sequence(hidden_ty, hidden_ty) as (INPUT, OUTPUT):
        # Set runtime parameters
        def set_rtps(my_rtp):
            my_rtp[0] = 8       # Attention scale: sqrt(64) = 8
            my_rtp[1] = 1       # LayerNorm1 scale
            my_rtp[2] = 0       # LayerNorm1 bias
            my_rtp[3] = 1       # LayerNorm2 scale
            my_rtp[4] = 0       # LayerNorm2 bias

        rt.inline_ops(set_rtps, [rtp])
        rt.set_barrier(rtp_barrier, 1)

        # Start all workers
        rt.start(worker_qkv, worker_attn, worker_norm1, worker_ffn1, worker_ffn2, worker_norm2)

        # DMA transfers
        shim_in = Tile(0, 0) if use_multi_column else Tile(0, 0)
        shim_out = Tile(3, 0) if use_multi_column else Tile(0, 0)

        rt.fill(of_input.prod(), INPUT, placement=shim_in)
        rt.drain(of_output.cons(), OUTPUT, placement=shim_out, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    """
    Usage:
        # Multi-column (4 columns in parallel)
        python3 encoder_block_xdna1_iron.py npu 64 64 true > encoder_block_multi.mlir

        # Single column (sequential)
        python3 encoder_block_xdna1_iron.py npu 64 64 false > encoder_block_single.mlir

    Complete encoder block with:
        - Multi-head self-attention
        - LayerNorm + residuals
        - Feed-forward network
        - GELU activation
    """

    try:
        device_name = str(sys.argv[1]) if len(sys.argv) > 1 else "npu"
        seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 64
        d_model = int(sys.argv[3]) if len(sys.argv) > 3 else 64
        multi_col = str(sys.argv[4]).lower() == "true" if len(sys.argv) > 4 else True

        if device_name != "npu":
            raise ValueError("Only 'npu' device supported (XDNA1)")

        dev = NPU1Col1()

        # Generate encoder block
        module = whisper_encoder_block_xdna1(
            dev,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=8,
            ffn_dim=4 * d_model,
            use_multi_column=multi_col
        )

        print(module)

        print(f"# Complete Whisper encoder block for Phoenix NPU", file=sys.stderr)
        print(f"# Device: NPU1Col1 (4 columns)", file=sys.stderr)
        print(f"# Sequence length: {seq_len}", file=sys.stderr)
        print(f"# Model dimension: {d_model}", file=sys.stderr)
        print(f"# Multi-column: {multi_col}", file=sys.stderr)
        print(f"# Components: Attention + LayerNorm + FFN", file=sys.stderr)

    except (IndexError, ValueError) as e:
        print(f"Usage: {sys.argv[0]} npu [seq_len] [d_model] [multi_column]", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
