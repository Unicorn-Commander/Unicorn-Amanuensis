#!/usr/bin/env python3
"""
Test Multi-Core Attention Kernel (IRON-generated)
================================================

Tests the 4-column parallel attention implementation.

Usage:
    # After XCLBIN compilation succeeds:
    python3 test_attention_multicore_iron.py

Expected Results:
    - 4× throughput improvement
    - ~2.85ms per batch of 4 tiles
    - 27-33× realtime factor
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add XRT Python bindings to path
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt

# Configuration
TILE_SIZE = 64
QKV_SIZE = TILE_SIZE * TILE_SIZE  # 4096 elements per matrix
INPUT_SIZE_PER_TILE = QKV_SIZE * 3  # Q + K + V = 12288 bytes
OUTPUT_SIZE_PER_TILE = QKV_SIZE  # 4096 bytes
N_COLUMNS = 4  # Number of NPU columns to use
BATCH_SIZE = N_COLUMNS  # Process 4 tiles in parallel

XCLBIN_PATH = "build_attention_iron/attention_multicore.xclbin"


def generate_test_data(n_tiles=4):
    """
    Generate random test data for attention.

    Args:
        n_tiles: Number of tiles to generate (default: 4 for full batch)

    Returns:
        tuple: (qkv_inputs, expected_shapes)
            qkv_inputs: List of n_tiles input arrays (12288 bytes each)
            expected_shapes: Expected output shapes
    """
    print(f"Generating {n_tiles} tiles of test data...")

    qkv_inputs = []

    for i in range(n_tiles):
        # Create Q, K, V matrices (64×64 int8)
        # Use deterministic seed for reproducibility
        np.random.seed(42 + i)

        Q = np.random.randint(-64, 64, size=(TILE_SIZE, TILE_SIZE), dtype=np.int8)
        K = np.random.randint(-64, 64, size=(TILE_SIZE, TILE_SIZE), dtype=np.int8)
        V = np.random.randint(-64, 64, size=(TILE_SIZE, TILE_SIZE), dtype=np.int8)

        # Combine into single input buffer
        qkv_combined = np.concatenate([
            Q.flatten(),
            K.flatten(),
            V.flatten()
        ])

        assert qkv_combined.shape == (INPUT_SIZE_PER_TILE,)
        assert qkv_combined.dtype == np.int8

        qkv_inputs.append(qkv_combined)

    print(f"✓ Generated {n_tiles} tiles")
    print(f"  Input size per tile: {INPUT_SIZE_PER_TILE} bytes")
    print(f"  Output size per tile: {OUTPUT_SIZE_PER_TILE} bytes")

    return qkv_inputs


def load_xclbin(device_idx=0):
    """
    Load XCLBIN onto NPU device using register_xclbin API.

    Args:
        device_idx: Device index (default: 0)

    Returns:
        tuple: (device, hw_ctx, kernel)
    """
    print(f"\nLoading XCLBIN from: {XCLBIN_PATH}")

    if not Path(XCLBIN_PATH).exists():
        print(f"✗ Error: XCLBIN not found at {XCLBIN_PATH}")
        print("  Please compile the kernel first:")
        print("  ./compile_attention_iron.sh")
        return None, None, None

    try:
        # Use register_xclbin API (correct method for Phoenix NPU)
        device = xrt.device(device_idx)
        xclbin_obj = xrt.xclbin(XCLBIN_PATH)
        uuid = xclbin_obj.get_uuid()
        device.register_xclbin(xclbin_obj)

        print(f"✓ XCLBIN registered successfully")
        print(f"  Device: {device_idx}")
        print(f"  UUID: {uuid}")

        # Create hardware context
        hw_ctx = xrt.hw_context(device, uuid)
        print(f"✓ Hardware context created")

        # Get kernel
        kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
        print(f"✓ Kernel found: MLIR_AIE")

        return device, hw_ctx, kernel

    except Exception as e:
        print(f"✗ Error loading XCLBIN: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def allocate_buffers(device, n_tiles=4):
    """
    Allocate device buffers for multi-tile processing.

    Args:
        device: XRT device handle
        n_tiles: Number of tiles in batch

    Returns:
        tuple: (input_buffers, output_buffers)
    """
    print(f"\nAllocating buffers for {n_tiles} tiles...")

    input_buffers = []
    output_buffers = []

    for i in range(n_tiles):
        # Input buffer: Q+K+V combined (12288 bytes)
        in_bo = xrt.bo(device, INPUT_SIZE_PER_TILE, xrt.bo.flags.host_only, 0)
        input_buffers.append(in_bo)

        # Output buffer: Attention result (4096 bytes)
        out_bo = xrt.bo(device, OUTPUT_SIZE_PER_TILE, xrt.bo.flags.host_only, 0)
        output_buffers.append(out_bo)

    total_input = INPUT_SIZE_PER_TILE * n_tiles / 1024
    total_output = OUTPUT_SIZE_PER_TILE * n_tiles / 1024

    print(f"✓ Allocated {n_tiles} input buffers ({total_input:.1f} KB total)")
    print(f"✓ Allocated {n_tiles} output buffers ({total_output:.1f} KB total)")

    return input_buffers, output_buffers


def run_multicore_attention(device, kernel, qkv_inputs):
    """
    Run multi-core attention on NPU using MLIR_AIE kernel with instructions.

    Args:
        device: XRT device handle
        kernel: XRT kernel handle
        qkv_inputs: List of input arrays (n_tiles × 12288 bytes)

    Returns:
        tuple: (outputs, elapsed_time)
    """
    n_tiles = len(qkv_inputs)
    print(f"\nRunning multi-core attention on {n_tiles} tiles...")

    # Load instruction sequence
    insts_path = "build_attention_iron/insts.bin"
    if not Path(insts_path).exists():
        raise FileNotFoundError(f"Instructions not found at {insts_path}")

    with open(insts_path, "rb") as f:
        insts = f.read()
    n_insts = len(insts)
    print(f"Loaded {n_insts} bytes of instructions")

    # Allocate instruction buffer
    # Kernel signature: opcode, instr_bo, instr_size, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7
    # group_id: 0=cacheable for instructions
    instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, 0)
    instr_bo.write(insts, 0)
    instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

    # Allocate input buffers (arg0-3: memref<12288xi8>)
    # group_id: 1=host_only for data transfer
    input_buffers = []
    for i in range(n_tiles):
        in_bo = xrt.bo(device, INPUT_SIZE_PER_TILE, xrt.bo.flags.host_only, 1)
        in_bo.write(qkv_inputs[i], 0)
        in_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, INPUT_SIZE_PER_TILE, 0)
        input_buffers.append(in_bo)

    # Allocate output buffers (arg4-7: memref<4096xi8>)
    # group_id: 1=host_only for data transfer
    output_buffers = []
    for i in range(n_tiles):
        out_bo = xrt.bo(device, OUTPUT_SIZE_PER_TILE, xrt.bo.flags.host_only, 1)
        output_buffers.append(out_bo)

    print(f"Allocated buffers for {n_tiles} tiles")

    # Run kernel with all buffers
    print(f"Executing multi-core kernel...")
    start_time = time.time()

    opcode = 3  # Standard opcode for NPU kernels
    run = kernel(opcode, instr_bo, n_insts,
                 input_buffers[0], input_buffers[1], input_buffers[2], input_buffers[3],
                 output_buffers[0], output_buffers[1], output_buffers[2], output_buffers[3])

    state = run.wait(5000)  # 5 second timeout
    if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        raise RuntimeError(f"Kernel execution failed with state {state}")

    elapsed_time = time.time() - start_time

    # Copy results back
    print("Copying results from device...")
    outputs = []
    for out_bo in output_buffers:
        out_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, OUTPUT_SIZE_PER_TILE, 0)
        result = np.frombuffer(out_bo.read(OUTPUT_SIZE_PER_TILE, 0), dtype=np.int8)
        outputs.append(result.reshape(TILE_SIZE, TILE_SIZE))

    print(f"✓ Execution complete in {elapsed_time*1000:.2f}ms")

    return outputs, elapsed_time


def benchmark_throughput(device, kernel, n_batches=10):
    """
    Benchmark multi-core throughput.

    Args:
        device: XRT device handle
        kernel: XRT kernel handle
        n_batches: Number of batches to run

    Returns:
        dict: Performance metrics
    """
    print(f"\n{'='*60}")
    print(f"THROUGHPUT BENCHMARK")
    print(f"{'='*60}")

    # Generate test data
    qkv_inputs = generate_test_data(n_tiles=BATCH_SIZE)

    # Warm-up run
    print("\nWarm-up run...")
    _, _ = run_multicore_attention(device, kernel, qkv_inputs)

    # Benchmark runs
    print(f"\nRunning {n_batches} benchmark batches...")
    times = []

    for batch_idx in range(n_batches):
        _, elapsed = run_multicore_attention(device, kernel, qkv_inputs)
        times.append(elapsed)
        print(f"  Batch {batch_idx+1}/{n_batches}: {elapsed*1000:.2f}ms")

    # Calculate statistics
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    # Calculate throughput metrics
    time_per_batch_ms = avg_time * 1000
    time_per_tile_ms = time_per_batch_ms / BATCH_SIZE
    tiles_per_second = 1000 / time_per_tile_ms

    # For 80ms audio chunks at 16kHz = 1280 samples
    # Each tile represents ~17.5ms of audio (1500 frames / 64 = 23.4 tiles per chunk)
    audio_chunk_duration_ms = 80  # ms
    tiles_per_chunk = 23.4  # approximate
    time_to_process_chunk_ms = tiles_per_chunk * time_per_tile_ms
    realtime_factor = audio_chunk_duration_ms / time_to_process_chunk_ms

    # Calculate improvement vs single-core
    single_core_time_per_tile_ms = 2.85  # measured baseline
    speedup = single_core_time_per_tile_ms / time_per_tile_ms
    single_core_rtf = 16.2
    multicore_rtf = single_core_rtf * speedup

    print(f"\n{'='*60}")
    print(f"PERFORMANCE RESULTS")
    print(f"{'='*60}")
    print(f"Batches processed: {n_batches}")
    print(f"Tiles per batch: {BATCH_SIZE}")
    print(f"Total tiles: {n_batches * BATCH_SIZE}")
    print()
    print(f"Time per batch: {time_per_batch_ms:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"  Min: {min_time*1000:.2f}ms")
    print(f"  Max: {max_time*1000:.2f}ms")
    print()
    print(f"Time per tile (effective): {time_per_tile_ms:.2f}ms")
    print(f"Tiles per second: {tiles_per_second:.1f}")
    print()
    print(f"{'='*60}")
    print(f"THROUGHPUT IMPROVEMENT")
    print(f"{'='*60}")
    print(f"Single-core baseline: {single_core_time_per_tile_ms:.2f}ms per tile")
    print(f"Multi-core (4 columns): {time_per_tile_ms:.2f}ms per tile")
    print(f"Speedup: {speedup:.2f}×")
    print()
    print(f"Single-core RTF: {single_core_rtf:.1f}×")
    print(f"Multi-core RTF: {multicore_rtf:.1f}×")
    print(f"RTF improvement: {multicore_rtf - single_core_rtf:.1f}× realtime")
    print()

    if speedup >= 3.5:
        print("✅ SUCCESS: Achieved target 4× throughput improvement!")
    elif speedup >= 2.5:
        print("⚠ PARTIAL: Good improvement but below 4× target")
    else:
        print("❌ ISSUE: Significant overhead detected")

    print(f"{'='*60}\n")

    return {
        "time_per_batch_ms": time_per_batch_ms,
        "time_per_tile_ms": time_per_tile_ms,
        "tiles_per_second": tiles_per_second,
        "speedup": speedup,
        "multicore_rtf": multicore_rtf,
        "times": times
    }


def main():
    """
    Main test function.
    """
    print("="*60)
    print("MULTI-CORE ATTENTION KERNEL TEST (IRON)")
    print("="*60)
    print(f"Configuration:")
    print(f"  NPU Columns: {N_COLUMNS}")
    print(f"  Batch Size: {BATCH_SIZE} tiles")
    print(f"  Tile Size: {TILE_SIZE}×{TILE_SIZE}")
    print(f"  Input per tile: {INPUT_SIZE_PER_TILE} bytes (Q+K+V)")
    print(f"  Output per tile: {OUTPUT_SIZE_PER_TILE} bytes")
    print("="*60)

    # Load XCLBIN
    device, hw_ctx, kernel = load_xclbin()
    if device is None:
        print("\n✗ Test aborted: Cannot load XCLBIN")
        print("  Please compile kernel first:")
        print("  ./compile_attention_iron.sh")
        return 1

    # Run benchmark
    try:
        results = benchmark_throughput(device, kernel, n_batches=10)

        print("\nTest completed successfully!")
        print(f"Multi-core RTF: {results['multicore_rtf']:.1f}×")
        print(f"Speedup: {results['speedup']:.2f}×")

        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
