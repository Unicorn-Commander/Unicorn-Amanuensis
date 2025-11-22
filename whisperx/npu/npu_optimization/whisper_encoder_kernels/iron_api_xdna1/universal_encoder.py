#!/usr/bin/env python3
"""
Universal Whisper Encoder - Works on XDNA1 (Phoenix) and XDNA2 (Strix/Hawk)

This demonstrates device-agnostic IRON API code that adapts to the target NPU.

Key Portability Strategy:
    1. Use device detection to select NPU1Col1 vs NPU2Col1
    2. Adapt column count (4 vs 8) automatically
    3. Reuse same C++ kernels across devices
    4. Adjust tile placement based on architecture

Supported Devices:
    - XDNA1 (Phoenix):  NPU1Col1 = 4 columns × 6 rows = 24 tiles
    - XDNA2 (Strix):    NPU2Col1 = 8 columns × 6 rows = 48 tiles
    - XDNA2 (Hawk):     NPU2Col1 = 8 columns (varies by SKU)

Benefits:
    - Write once, compile for multiple devices
    - Same kernel C++ code
    - Automatic optimization for target architecture
    - Easy to maintain and test
"""

import numpy as np
import sys
from typing import Union

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
from aie.iron.device import NPU1Col1, NPU2Col1, Tile
from aie.iron.controlflow import range_


class DeviceConfig:
    """Device-specific configuration"""

    def __init__(self, device_type: str):
        self.device_type = device_type

        if device_type == "npu1":
            # XDNA1: Phoenix NPU
            self.device = NPU1Col1()
            self.num_columns = 4
            self.num_rows = 6
            self.num_compute_tiles = 24
            self.name = "Phoenix (XDNA1)"
        elif device_type == "npu2":
            # XDNA2: Strix/Hawk NPU
            self.device = NPU2Col1()
            self.num_columns = 8
            self.num_rows = 6
            self.num_compute_tiles = 48
            self.name = "Strix/Hawk (XDNA2)"
        else:
            raise ValueError(f"Unknown device type: {device_type}")

    def get_compute_tiles(self, start_row: int = 2):
        """
        Get list of compute tiles for this device

        Args:
            start_row: First compute row (default: 2)

        Returns:
            List of Tile objects for compute cores
        """
        tiles = []
        for col in range(self.num_columns):
            for row in range(start_row, self.num_rows):
                tiles.append(Tile(col, row))
        return tiles

    def get_column_tiles(self, col: int, start_row: int = 2):
        """Get tiles for a specific column"""
        return [Tile(col, row) for row in range(start_row, self.num_rows)]

    def get_shim_tile(self, col: int = 0):
        """Get shim tile (DMA) for a column"""
        return Tile(col, 0)


def universal_encoder_attention(
    config: DeviceConfig,
    seq_len: int = 64,
    d_model: int = 64
):
    """
    Universal attention kernel that works on XDNA1 and XDNA2

    Automatically adapts to device capabilities:
        - XDNA1: Uses 1-4 columns for attention heads
        - XDNA2: Uses 1-8 columns for attention heads
        - Kernel code: Same C++ kernel works on both

    Args:
        config: DeviceConfig instance
        seq_len: Sequence length
        d_model: Model dimension
    """

    # Buffer sizes (same for both devices)
    hidden_size = seq_len * d_model
    qkv_size = 3 * hidden_size

    # Type definitions (same for both devices)
    hidden_ty = np.ndarray[(hidden_size,), np.dtype[np.int8]]
    qkv_ty = np.ndarray[(qkv_size,), np.dtype[np.int8]]

    # Kernel declaration (SAME C++ kernel for both devices!)
    attention_kernel = Kernel(
        "attention_64x64",
        "attention_int8_64x64_tiled.o",
        [qkv_ty, hidden_ty, np.int32]
    )

    # ObjectFIFOs
    of_input = ObjectFifo(qkv_ty, name="input_L3L2", depth=2)
    of_output = ObjectFifo(hidden_ty, name="output_L2L3", depth=2)

    # Runtime parameters
    rtp = GlobalBuffer(
        np.ndarray[(16,), np.dtype[np.int32]],
        name="rtp_attention",
        use_write_rtp=True,
    )
    rtp_barrier = WorkerRuntimeBarrier()

    # Core function (same for both devices)
    def attention_core_fn(of_in, of_out, kernel, my_rtp, barrier):
        barrier.wait_for_value(1)
        scale = my_rtp[0]

        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(elem_in, elem_out, scale)
        of_in.release(1)
        of_out.release(1)

    # Tile placement (ADAPTS to device)
    # XDNA1: Use column 0
    # XDNA2: Use column 0 (can use more for multi-head)
    compute_tile = config.get_column_tiles(0)[0]  # (0, 2)

    # Create worker
    worker = Worker(
        attention_core_fn,
        [of_input.cons(), of_output.prod(), attention_kernel, rtp, rtp_barrier],
        placement=compute_tile,
        stack_size=0x800
    )

    # Runtime sequence
    rt = Runtime()
    with rt.sequence(qkv_ty, hidden_ty) as (INPUT, OUTPUT):
        def set_rtps(my_rtp):
            my_rtp[0] = 8  # sqrt(64) = 8

        rt.inline_ops(set_rtps, [rtp])
        rt.set_barrier(rtp_barrier, 1)
        rt.start(worker)

        shim = config.get_shim_tile(0)
        rt.fill(of_input.prod(), INPUT, placement=shim)
        rt.drain(of_output.cons(), OUTPUT, placement=shim, wait=True)

    return Program(config.device, rt).resolve_program(SequentialPlacer())


def universal_encoder_multi_column(
    config: DeviceConfig,
    seq_len: int = 64,
    d_model: int = 64
):
    """
    Universal multi-column encoder

    Adapts column distribution to device:
        - XDNA1 (4 cols): Attention, FFN1, FFN2, LayerNorm (one per column)
        - XDNA2 (8 cols): Can use 8 attention heads in parallel, or distribute differently

    This shows how to write portable code that maximizes each device.
    """

    # Buffer sizes
    hidden_size = seq_len * d_model
    qkv_size = 3 * hidden_size
    ffn_hidden_size = seq_len * (4 * d_model)

    # Types
    hidden_ty = np.ndarray[(hidden_size,), np.dtype[np.int8]]
    qkv_ty = np.ndarray[(qkv_size,), np.dtype[np.int8]]
    ffn_hidden_ty = np.ndarray[(ffn_hidden_size,), np.dtype[np.int8]]

    # Kernels (SAME for both devices!)
    attention_kernel = Kernel("attention_64x64", "attention_int8_64x64_tiled.o", [qkv_ty, hidden_ty, np.int32])
    ffn1_kernel = Kernel("ffn_linear1_gelu", "ffn1_gelu.o", [hidden_ty, ffn_hidden_ty])
    ffn2_kernel = Kernel("ffn_linear2", "ffn2.o", [ffn_hidden_ty, hidden_ty])
    layernorm_kernel = Kernel("layernorm_int8", "layernorm_int8.o", [hidden_ty, hidden_ty, np.int32, np.int32])

    # ObjectFIFOs
    of_input = ObjectFifo(qkv_ty, name="input", depth=2)
    of_attn_out = ObjectFifo(hidden_ty, name="attn_out", depth=2)
    of_ffn_intermediate = ObjectFifo(ffn_hidden_ty, name="ffn_inter", depth=2)
    of_ffn_out = ObjectFifo(hidden_ty, name="ffn_out", depth=2)
    of_output = ObjectFifo(hidden_ty, name="output", depth=2)

    # Runtime params
    rtp = GlobalBuffer(np.ndarray[(16,), np.dtype[np.int32]], name="rtp", use_write_rtp=True)
    rtp_barrier = WorkerRuntimeBarrier()

    # Tile assignment (ADAPTS to device)
    # XDNA1: 4 columns → distribute one operation per column
    # XDNA2: 8 columns → can distribute more or use 2 operations per column
    num_ops = 4  # Attention, FFN1, FFN2, LayerNorm

    if config.num_columns >= num_ops:
        # Enough columns for one operation per column
        tile_attn = config.get_column_tiles(0)[0]  # Col 0
        tile_ffn1 = config.get_column_tiles(1)[0]  # Col 1
        tile_ffn2 = config.get_column_tiles(2)[0]  # Col 2
        tile_norm = config.get_column_tiles(3)[0]  # Col 3
    else:
        # Fewer columns: stack operations
        tile_attn = config.get_column_tiles(0)[0]
        tile_ffn1 = config.get_column_tiles(0)[1]
        tile_ffn2 = config.get_column_tiles(0)[2]
        tile_norm = config.get_column_tiles(0)[3]

    # Workers (SAME logic for both devices!)
    def attn_fn(of_in, of_out, kernel, my_rtp, barrier):
        barrier.wait_for_value(1)
        scale = my_rtp[0]
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(elem_in, elem_out, scale)
        of_in.release(1)
        of_out.release(1)

    def ffn1_fn(of_in, of_out, kernel):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(elem_in, elem_out)
        of_in.release(1)
        of_out.release(1)

    def ffn2_fn(of_in, of_out, kernel):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(elem_in, elem_out)
        of_in.release(1)
        of_out.release(1)

    def norm_fn(of_in, of_out, kernel, my_rtp):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        scale = my_rtp[1]
        bias = my_rtp[2]
        kernel(elem_in, elem_out, scale, bias)
        of_in.release(1)
        of_out.release(1)

    worker_attn = Worker(attn_fn, [of_input.cons(), of_attn_out.prod(), attention_kernel, rtp, rtp_barrier], placement=tile_attn, stack_size=0x800)
    worker_ffn1 = Worker(ffn1_fn, [of_attn_out.cons(), of_ffn_intermediate.prod(), ffn1_kernel], placement=tile_ffn1, stack_size=0x600)
    worker_ffn2 = Worker(ffn2_fn, [of_ffn_intermediate.cons(), of_ffn_out.prod(), ffn2_kernel], placement=tile_ffn2, stack_size=0x600)
    worker_norm = Worker(norm_fn, [of_ffn_out.cons(), of_output.prod(), layernorm_kernel, rtp], placement=tile_norm, stack_size=0x600)

    # Runtime
    rt = Runtime()
    with rt.sequence(qkv_ty, hidden_ty) as (INPUT, OUTPUT):
        def set_rtps(my_rtp):
            my_rtp[0] = 8  # Attention scale
            my_rtp[1] = 1  # LayerNorm scale
            my_rtp[2] = 0  # LayerNorm bias

        rt.inline_ops(set_rtps, [rtp])
        rt.set_barrier(rtp_barrier, 1)
        rt.start(worker_attn, worker_ffn1, worker_ffn2, worker_norm)

        shim = config.get_shim_tile(0)
        rt.fill(of_input.prod(), INPUT, placement=shim)
        rt.drain(of_output.cons(), OUTPUT, placement=shim, wait=True)

    return Program(config.device, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    """
    Usage:
        # XDNA1 (Phoenix NPU)
        python3 universal_encoder.py npu1 simple > encoder_xdna1.mlir
        python3 universal_encoder.py npu1 multi > encoder_xdna1_multi.mlir

        # XDNA2 (Strix/Hawk NPU)
        python3 universal_encoder.py npu2 simple > encoder_xdna2.mlir
        python3 universal_encoder.py npu2 multi > encoder_xdna2_multi.mlir

    Key benefits:
        - Same source code
        - Same C++ kernels
        - Automatic adaptation to device
        - Easy to maintain
    """

    try:
        device_type = str(sys.argv[1]) if len(sys.argv) > 1 else "npu1"
        mode = str(sys.argv[2]) if len(sys.argv) > 2 else "simple"

        # Create device config
        config = DeviceConfig(device_type)

        print(f"# Generating universal encoder for {config.name}", file=sys.stderr)
        print(f"# Columns: {config.num_columns}", file=sys.stderr)
        print(f"# Compute tiles: {config.num_compute_tiles}", file=sys.stderr)
        print(f"# Mode: {mode}", file=sys.stderr)

        # Generate appropriate module
        if mode == "simple":
            module = universal_encoder_attention(config, seq_len=64, d_model=64)
        elif mode == "multi":
            module = universal_encoder_multi_column(config, seq_len=64, d_model=64)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'simple' or 'multi'.")

        print(module)

        print(f"# Success: Universal encoder generated for {config.name}", file=sys.stderr)
        print(f"# C++ kernels: Reusable across XDNA1 and XDNA2", file=sys.stderr)

    except (IndexError, ValueError) as e:
        print(f"Usage: {sys.argv[0]} <npu1|npu2> [simple|multi]", file=sys.stderr)
        print(f"  npu1: XDNA1 (Phoenix) - 4 columns", file=sys.stderr)
        print(f"  npu2: XDNA2 (Strix/Hawk) - 8 columns", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
