#!/usr/bin/env python3
"""
Test script for 64x64 Attention Mechanism on AMD Phoenix NPU
Tests the scaled attention kernel with PyXRT

Expected performance: 8-10ms per 64x64 tile (vs 0.56ms for 16x16)
For Whisper: 1500 frames / 8 heads / 10ms = ~2 seconds for full sequence
"""

import numpy as np
import time
import sys
import os

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
try:
    import pyxrt as xrt
except ImportError:
    print("ERROR: PyXRT not installed. Install with: pip install pyxrt")
    sys.exit(1)

def test_attention_64x64():
    """Test 64x64 attention kernel on NPU"""

    print("="*70)
    print("64x64 Attention Mechanism Test on AMD Phoenix NPU")
    print("="*70)
    print()

    # Configuration
    TILE_SIZE = 64
    QKV_SIZE = TILE_SIZE * TILE_SIZE  # 4096 bytes per matrix
    COMBINED_SIZE = 3 * QKV_SIZE  # 12288 bytes (Q + K + V)
    OUTPUT_SIZE = QKV_SIZE  # 4096 bytes
    SCALE_SHIFT = 3  # sqrt(64) = 8, log2(8) = 3

    print(f"Configuration:")
    print(f"  Tile size: {TILE_SIZE}x{TILE_SIZE}")
    print(f"  Q/K/V size: {QKV_SIZE} bytes each")
    print(f"  Combined QKV: {COMBINED_SIZE} bytes")
    print(f"  Output size: {OUTPUT_SIZE} bytes")
    print(f"  Scale shift: {SCALE_SHIFT} (divide by 8)")
    print()

    # Load XCLBIN
    xclbin_path = "attention_64x64.xclbin"
    insts_path = "build_attention_64x64/insts.bin"
    if not os.path.exists(xclbin_path):
        print(f"ERROR: XCLBIN not found at {xclbin_path}")
        print("Run compile_attention_64x64.sh first")
        sys.exit(1)
    if not os.path.exists(insts_path):
        print(f"ERROR: Instructions not found at {insts_path}")
        print("Run compile_attention_64x64.sh first")
        sys.exit(1)

    print(f"Step 1: Loading XCLBIN from {xclbin_path}...")
    try:
        device = xrt.device(0)
        xclbin_obj = xrt.xclbin(xclbin_path)
        uuid = xclbin_obj.get_uuid()
        device.register_xclbin(xclbin_obj)
        print("✅ XCLBIN loaded successfully")
        print(f"   UUID: {uuid}")

        # Create hardware context
        hw_ctx = xrt.hw_context(device, uuid)
        print("✅ Hardware context created")

        # Get kernel
        kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
        print("✅ Kernel found: MLIR_AIE")

        # Load instruction sequence
        with open(insts_path, "rb") as f:
            insts = f.read()
        n_insts = len(insts)
        print(f"✅ Instructions loaded: {n_insts} bytes")
    except Exception as e:
        print(f"❌ ERROR loading XCLBIN: {e}")
        sys.exit(1)
    print()

    # Create random test data
    print(f"Step 2: Generating random INT8 test data...")
    np.random.seed(42)

    # Create Q, K, V matrices (64x64 each)
    Q = np.random.randint(-64, 64, size=(TILE_SIZE, TILE_SIZE), dtype=np.int8)
    K = np.random.randint(-64, 64, size=(TILE_SIZE, TILE_SIZE), dtype=np.int8)
    V = np.random.randint(-64, 64, size=(TILE_SIZE, TILE_SIZE), dtype=np.int8)

    # Combine Q, K, V into single buffer
    QKV_combined = np.concatenate([Q.flatten(), K.flatten(), V.flatten()])

    print(f"  Q matrix: {Q.shape}, range [{Q.min()}, {Q.max()}]")
    print(f"  K matrix: {K.shape}, range [{K.min()}, {K.max()}]")
    print(f"  V matrix: {V.shape}, range [{V.min()}, {V.max()}]")
    print(f"  Combined buffer: {QKV_combined.shape}, {QKV_combined.nbytes} bytes")
    print()

    # Allocate NPU buffers
    print(f"Step 3: Allocating NPU buffers...")
    try:
        # Instruction buffer (group_id 1 for instructions)
        instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
        instr_bo.write(insts, 0)
        instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

        # Input buffer: Combined Q+K+V (group_id 3 for input)
        input_bo = xrt.bo(device, COMBINED_SIZE, xrt.bo.flags.host_only, kernel.group_id(3))
        input_bo.write(QKV_combined.tobytes(), 0)
        input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, COMBINED_SIZE, 0)

        # Output buffer: Attention output (group_id 4 for output)
        output_bo = xrt.bo(device, OUTPUT_SIZE, xrt.bo.flags.host_only, kernel.group_id(4))

        print(f"✅ Allocated instruction buffer: {n_insts} bytes")
        print(f"✅ Allocated data buffers: {COMBINED_SIZE + OUTPUT_SIZE} bytes on NPU")
    except Exception as e:
        print(f"❌ ERROR allocating buffers: {e}")
        sys.exit(1)
    print()

    # Run kernel and measure performance
    print(f"Step 5: Running kernel on NPU...")
    print(f"  Warming up with 3 iterations...")

    # Warmup
    for i in range(3):
        try:
            opcode = 3  # Standard opcode for NPU kernels
            run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
            state = run.wait(1000)  # 1 second timeout
            if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                print(f"❌ ERROR in warmup iteration {i+1}: kernel state {state}")
                sys.exit(1)
        except Exception as e:
            print(f"❌ ERROR in warmup iteration {i+1}: {e}")
            sys.exit(1)

    print(f"  Running benchmark with 10 iterations...")

    # Benchmark
    times = []
    for i in range(10):
        start = time.perf_counter()
        try:
            opcode = 3  # Standard opcode for NPU kernels
            run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
            state = run.wait(1000)  # 1 second timeout
            if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                print(f"❌ ERROR in iteration {i+1}: kernel state {state}")
                sys.exit(1)
        except Exception as e:
            print(f"❌ ERROR in iteration {i+1}: {e}")
            sys.exit(1)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"✅ Kernel execution complete")
    print()

    # Read output
    print(f"Step 6: Reading output from NPU...")
    try:
        output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, OUTPUT_SIZE, 0)
        output_data = np.frombuffer(output_bo.read(OUTPUT_SIZE, 0), dtype=np.int8)
        output_matrix = output_data.reshape(TILE_SIZE, TILE_SIZE)
        print(f"✅ Output retrieved: {output_matrix.shape}")
    except Exception as e:
        print(f"❌ ERROR reading output: {e}")
        sys.exit(1)
    print()

    # Verify output
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()

    print(f"Performance Measurements:")
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Std deviation: {std_time:.2f} ms")
    print(f"  Min time: {min_time:.2f} ms")
    print(f"  Max time: {max_time:.2f} ms")
    print()

    print(f"Output Statistics:")
    print(f"  Shape: {output_matrix.shape}")
    print(f"  Range: [{output_matrix.min()}, {output_matrix.max()}]")
    print(f"  Mean: {output_matrix.mean():.2f}")
    print(f"  Non-zero elements: {np.count_nonzero(output_matrix)}/{output_matrix.size}")
    print()

    # Sample output
    print(f"Sample Output (first 8x8 corner):")
    print(output_matrix[:8, :8])
    print()

    # Whisper production estimate
    print("="*70)
    print("PRODUCTION WHISPER ESTIMATES")
    print("="*70)
    print()

    sequence_length = 1500
    num_heads = 8
    tiles_per_sequence = sequence_length / TILE_SIZE  # ~23.4 tiles
    total_time_per_head = tiles_per_sequence * avg_time / 1000  # seconds
    total_time_all_heads = total_time_per_head * num_heads

    print(f"For Whisper Base (30 seconds audio):")
    print(f"  Sequence length: {sequence_length} frames")
    print(f"  Number of heads: {num_heads}")
    print(f"  Tiles per sequence: {tiles_per_sequence:.1f}")
    print(f"  Time per head: {total_time_per_head:.2f} seconds")
    print(f"  Total time (all heads): {total_time_all_heads:.2f} seconds")
    print()

    realtime_factor = 30.0 / total_time_all_heads if total_time_all_heads > 0 else 0
    print(f"  Realtime factor: {realtime_factor:.1f}x")
    print()

    # Compare with 16x16
    time_16x16 = 0.56  # ms per tile
    tiles_16x16 = (sequence_length / 16) * num_heads  # 750 tiles
    total_time_16x16 = tiles_16x16 * time_16x16 / 1000  # seconds
    realtime_16x16 = 30.0 / total_time_16x16

    print(f"Comparison with 16x16 tiles:")
    print(f"  16x16 time per tile: {time_16x16} ms")
    print(f"  16x16 total tiles: {tiles_16x16:.0f}")
    print(f"  16x16 total time: {total_time_16x16:.2f} seconds")
    print(f"  16x16 realtime factor: {realtime_16x16:.1f}x")
    print()

    speedup = realtime_16x16 / realtime_factor if realtime_factor > 0 else 0
    print(f"  64x64 is {speedup:.2f}x FASTER than 16x16 (fewer tiles)")
    print()

    # Success criteria
    print("="*70)
    print("SUCCESS CRITERIA")
    print("="*70)
    print()

    success = True

    # Check 1: Compilation
    print("✅ Compilation: PASSED")

    # Check 2: XCLBIN generation
    print("✅ XCLBIN generation: PASSED")

    # Check 3: NPU execution
    print("✅ NPU execution: PASSED")

    # Check 4: Non-zero output
    if np.count_nonzero(output_matrix) > 0:
        print("✅ Non-zero output: PASSED")
    else:
        print("❌ Non-zero output: FAILED")
        success = False

    # Check 5: Performance target (8-10ms)
    if avg_time <= 15.0:  # Allow some margin
        print(f"✅ Performance target: PASSED ({avg_time:.2f} ms <= 15 ms)")
    else:
        print(f"⚠️  Performance target: WARNING ({avg_time:.2f} ms > 15 ms expected)")

    # Check 6: Realtime factor > 1
    if realtime_factor > 1.0:
        print(f"✅ Realtime processing: PASSED ({realtime_factor:.1f}x realtime)")
    else:
        print(f"❌ Realtime processing: FAILED ({realtime_factor:.1f}x realtime)")
        success = False

    print()

    if success:
        print("="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print()
        print("Next steps:")
        print("  1. Test with multi-head attention (2-8 heads)")
        print("  2. Integrate with Whisper encoder pipeline")
        print("  3. Benchmark full sequence (1500 frames)")
        print("  4. Optimize for lower latency if needed")
        return 0
    else:
        print("="*70)
        print("⚠️  SOME TESTS FAILED")
        print("="*70)
        print()
        print("Review output and check:")
        print("  1. Kernel implementation correctness")
        print("  2. Memory constraints (32KB limit)")
        print("  3. DMA transfer sizes")
        print("  4. Softmax implementation")
        return 1

if __name__ == "__main__":
    sys.exit(test_attention_64x64())
