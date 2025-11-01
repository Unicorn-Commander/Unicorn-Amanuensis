#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
# Adapted for Whisper Encoder BFP16 kernels by Magic Unicorn Tech
#
# This script generates MLIR-AIE2 code for BFP16 matrix multiplication kernels
# optimized for Whisper encoder dimensions on XDNA2 NPU.
#
# Whisper Encoder Dimensions:
#   - M=512, K=512, N=512   (attention Q/K/V/out projections)
#   - M=512, K=512, N=2048  (FFN fc1 expansion)
#   - M=512, K=2048, N=512  (FFN fc2 reduction)
#
# Key Changes from Original:
#   - Default to --dev npu2 (XDNA2)
#   - Default to --dtype_in bf16 --dtype_out bf16
#   - Default to --emulate-bf16-mmul-with-bfp16 true (8x8x8 tiles)
#   - Whisper-appropriate default dimensions
#   - BFP16 kernel: mm_bfp.cc

import argparse
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, str_to_dtype
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D


microkernel_mac_dim_map = {
    "npu": {
        "bf16": (4, 8, 4),
        "i8": (4, 8, 8),
        "i16": (4, 4, 4),
    },
    "npu2": {
        "bf16": {
            # emulate_bf16_mmul_with_bfp16
            True: (8, 8, 8),    # BFP16 mode: 8x8x8 tiles
            False: (4, 8, 8),   # Native BF16 mode: 4x8x8 tiles
        },
        "i8": (8, 8, 8),
        "i16": (4, 4, 8),
    },
}


def main():
    argparser = argparse.ArgumentParser(
        prog="Whisper BFP16 Matrix Multiplication MLIR Generator",
        description="Generates MLIR-AIE2 code for BFP16 matrix multiplication optimized for Whisper encoder dimensions",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu2",
                          help="Target device (default: npu2 for XDNA2)")
    argparser.add_argument("-M", type=int, default=512,
                          help="Matrix A rows (default: 512 for Whisper)")
    argparser.add_argument("-K", type=int, default=512,
                          help="Matrix A cols / B rows (default: 512 for Whisper)")
    argparser.add_argument("-N", type=int, default=512,
                          help="Matrix B cols (default: 512 for Whisper attention)")
    argparser.add_argument("-m", type=int, default=64,
                          help="Tile size for M dimension (default: 64)")
    argparser.add_argument("-k", type=int, default=64,
                          help="Tile size for K dimension (default: 64)")
    argparser.add_argument("-n", type=int, default=64,
                          help="Tile size for N dimension (default: 64)")
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="bf16",
        help="Input data type (default: bf16 for BFP16 mode)"
    )
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="bf16",
        help="Output data type (default: bf16 for BFP16 mode)"
    )
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0,
                          help="B matrix column-major layout (default: 0 = row-major)")
    argparser.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=True,
                          help="Use BFP16 emulation for BF16 matmul (default: True)")
    argparser.add_argument("--trace_size", type=int, default=0,
                          help="Trace buffer size (0 = disabled)")
    argparser.add_argument(
        "--generate-taps",
        action="store_true",
        help="Generate TensorAccessPatterns for visualization (instead of MLIR)"
    )
    args = argparser.parse_args()

    # Validation: BFP16 mode requires bf16 input/output
    if args.emulate_bf16_mmul_with_bfp16 and (args.dtype_in != "bf16" or args.dtype_out != "bf16"):
        raise ValueError(
            "BFP16 emulation (--emulate-bf16-mmul-with-bfp16 true) requires "
            "--dtype_in bf16 and --dtype_out bf16"
        )

    maybe_module = my_matmul(
        args.dev,
        args.M,
        args.K,
        args.N,
        args.m,
        args.k,
        args.n,
        args.dtype_in,
        args.dtype_out,
        args.b_col_maj,
        args.emulate_bf16_mmul_with_bfp16,
        args.trace_size,
        args.generate_taps,
    )
    if args.generate_taps:
        # maybe_module is actually taps
        return maybe_module
    else:
        # print mlir
        print(maybe_module)


# Need ceildiv to capture partial tiling patterns
def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    emulate_bf16_mmul_with_bfp16,
    trace_size,
    generate_taps=False,
):

    assert M % m == 0, f"M ({M}) must be divisible by m ({m})"
    assert K % k == 0, f"K ({K}) must be divisible by k ({k})"
    assert N % n == 0, f"N ({N}) must be divisible by n ({n})"

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map[dev][dtype_in_str]
    if dev == "npu2" and dtype_in_str == "bf16":
        r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]
    else:
        r, s, t = mac_dims

    assert m % r == 0, f"m ({m}) must be divisible by r ({r})"
    assert k % s == 0, f"k ({k}) must be divisible by s ({s})"
    assert n % t == 0, f"n ({n}) must be divisible by t ({t})"

    vectorized = True
    enable_tracing = True if trace_size > 0 else False

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    # These will hold TensorAccessPattern objects that represent the runtime
    # npu_dma_memcpy_nd operations of this design. They are only used if generate_taps is true
    A_taps = []
    B_taps = []
    C_taps = []

    # Define tensor types
    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    b_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    # AIE Core Function declarations
    # For BFP16, we use matmul_vectorized_bfp16 from mm_bfp.cc
    if emulate_bf16_mmul_with_bfp16:
        kernel_name = "matmul_vectorized_bfp16"
        zero_kernel_name = "zero_kernel"
    else:
        func_type = "" if vectorized else "scalar_"
        zero_kernel_name = f"zero_{func_type}{dtype_out_str}"
        kernel_name = f"matmul_{dtype_in_str}_{dtype_out_str}"

    zero_kernel = Kernel(
        zero_kernel_name, f"mm_{m}x{k}x{n}.o", [c_ty]
    )
    matmul_kernel = Kernel(
        kernel_name,
        f"mm_{m}x{k}x{n}.o",
        [a_ty, b_ty, c_ty],
    )

    # AIE-array data movement with object fifos
    # Input A
    inA = ObjectFifo(a_ty, name="inA")
    a_dims = None
    if vectorized:
        a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    # Input B
    inB = ObjectFifo(b_ty, name="inB")
    b_dims = None
    if vectorized:
        if b_col_maj:
            b_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    # Output C
    memC = ObjectFifo(c_ty, name="memC")
    c_dims = None
    if vectorized:
        c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    # Task each core will run
    def core_fn(of_a, of_b, of_c, zero, matmul):
        for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547
            elem_out = of_c.acquire(1)
            zero(elem_out)

            # issue #1547
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    # Create worker from task
    worker = Worker(
        core_fn,
        [memA.cons(), memB.cons(), memC.prod(), zero_kernel, matmul_kernel],
        stack_size=0xD00,
    )

    # only do 4 tile rows at a time before synchronizing, so we can reuse BDs
    rows_per_block = 4

    # Define tensor access patterns for inputs/outputs
    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n
    )
    # There is only one access pattern for B - it tiles the entire matrix in (k x n) tiles.
    if b_col_maj:
        b_tap = TensorTiler2D.group_tiler((N, K), (n, k), (N_div_n, K_div_k))[0]
    else:
        b_tap = TensorTiler2D.group_tiler(
            (K, N), (k, n), (K_div_k, N_div_n), tile_group_col_major=True
        )[0]

    C_tiles = TensorTiler2D.group_tiler((M, N), (m, n), (rows_per_block // 2, N_div_n))
    c_index = 0

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.enable_trace(trace_size, workers=[worker])
        rt.start(worker)

        tgs = []
        for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
            # we only sync on half the BDs before reusing them, so the other half can concurrently keep running
            # that's what this loop is for. We can track of this in the task groups for syncing.
            for pingpong in [0, 1]:

                row_base = (
                    tile_row_block * rows_per_block + pingpong * rows_per_block // 2
                )
                num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                if num_tile_rows <= 0:
                    # At the very last iteration, we may not need a 'pong' iteration
                    break
                tgs.append(rt.task_group())
                for tile_row in range(num_tile_rows):
                    # -- A --
                    tile_offset = (row_base + tile_row) % len(A_tiles)
                    rt.fill(inA.prod(), A, tap=A_tiles[tile_offset], task_group=tgs[-1])
                    A_taps.append(A_tiles[tile_offset])

                    # -- B --
                    rt.fill(inB.prod(), B, tap=b_tap, task_group=tgs[-1])
                    B_taps.append(b_tap)

                # -- C --
                rt.drain(
                    outC.cons(), C, tap=C_tiles[c_index], task_group=tgs[-1], wait=True
                )
                C_taps.append(C_tiles[c_index])
                c_index += 1

                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]

        rt.finish_task_group(tgs[-1])
        del tgs[-1]

    if generate_taps:
        # If generate taps is true, return a representation of tensor access patterns
        # representing all the npu_dma_memcpy_nd runtime sequence operations per input/ouput tensor.
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )

    # Create the program from the device type and runtime
    if dev == "npu":
        dev_ty = NPU1()
    else:
        dev_ty = NPU2()
    my_program = Program(dev_ty, rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())
    return module


if __name__ == "__main__":
    main()
