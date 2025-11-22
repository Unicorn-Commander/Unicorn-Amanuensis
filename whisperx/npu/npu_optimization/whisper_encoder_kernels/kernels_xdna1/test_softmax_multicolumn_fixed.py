#!/usr/bin/env python3
"""
Test Fixed Multi-Column Softmax with Combined Buffers
AMD Phoenix NPU - XDNA1

This test validates the 4-tile multi-column softmax kernel that uses combined
input/output buffers to work around the XRT 5-argument limitation.

Architecture:
  - Column 0: ShimNOC (0,0), Compute (0,2), (0,3)
  - Column 1: ShimNOC (1,0), Compute (1,2), (1,3)
  - Combined input buffer: 8192 bytes (4 x 2048)
  - Combined output buffer: 8192 bytes (4 x 2048)
"""

import numpy as np
import pyxrt as xrt
import struct
import time
import sys
import os

def bf16_to_float(bf16_bytes):
    """Convert BF16 bytes to float32"""
    result = np.zeros(len(bf16_bytes) // 2, dtype=np.float32)
    for i in range(len(result)):
        upper = struct.unpack('H', bf16_bytes[i*2:(i+1)*2])[0]
        result[i] = struct.unpack('f', struct.pack('I', upper << 16))[0]
    return result

def float_to_bf16(floats):
    """Convert float32 to BF16 bytes"""
    result = bytearray(len(floats) * 2)
    for i, val in enumerate(floats):
        bits = struct.unpack('I', struct.pack('f', val))[0]
        upper = (bits >> 16) & 0xFFFF
        struct.pack_into('H', result, i*2, upper)
    return bytes(result)

def softmax_ref(x):
    """Reference softmax implementation"""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

def main():
    print("=" * 70)
    print("Fixed Multi-Column Softmax BF16 - 4 Tiles - Combined Buffers")
    print("AMD Phoenix NPU - XDNA1")
    print("=" * 70)
    print()

    num_elements = 1024
    num_frames = 4
    num_iterations = 10
    buffer_size_per_frame = num_elements * 2  # 2048 bytes per frame
    combined_buffer_size = num_frames * buffer_size_per_frame  # 8192 bytes

    print(f"Test Configuration:")
    print(f"  Elements per frame: {num_elements}")
    print(f"  Parallel frames: {num_frames}")
    print(f"  Total elements: {num_elements * num_frames}")
    print(f"  Buffer size per frame: {buffer_size_per_frame} bytes")
    print(f"  Combined buffer size: {combined_buffer_size} bytes")
    print(f"  Iterations: {num_iterations}")
    print()
    print(f"Architecture (Multi-Column):")
    print(f"  Column 0: ShimNOC (0,0), Compute (0,2), (0,3)")
    print(f"  Column 1: ShimNOC (1,0), Compute (1,2), (1,3)")
    print()

    # Determine the base directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xclbin_path = os.path.join(script_dir, "build_softmax_multicolumn_fixed/softmax_multicolumn_combined.xclbin")
    insts_path = os.path.join(script_dir, "build_softmax_multicolumn_fixed/insts.bin")

    print(f"XCLBIN: {xclbin_path}")
    print(f"Instructions: {insts_path}")
    print()

    try:
        # Step 1: Load XCLBIN
        print("Step 1: Loading XCLBIN...")
        device = xrt.device(0)
        xclbin_obj = xrt.xclbin(xclbin_path)
        uuid = xclbin_obj.get_uuid()
        device.register_xclbin(xclbin_obj)
        print("  XCLBIN loaded successfully")
        print()

        # Step 2: Create hardware context
        print("Step 2: Creating hardware context...")
        hw_ctx = xrt.hw_context(device, uuid)
        print("  Hardware context created")
        print()

        # Step 3: Get kernel
        print("Step 3: Getting kernel...")
        kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
        print("  Kernel found: MLIR_AIE")
        print()

        # Step 4: Analyze kernel arguments
        print("Step 4: Analyzing kernel arguments...")
        print("  Available group_ids:")
        for i in range(8):
            try:
                gid = kernel.group_id(i)
                print(f"    group_id({i}) = {gid}")
            except:
                break
        print()

        # Step 5: Load instruction sequence
        print("Step 5: Loading instruction sequence...")
        with open(insts_path, "rb") as f:
            insts = f.read()
        print(f"  Instructions loaded: {len(insts)} bytes")
        print()

        # Step 6: Allocate buffers
        print("Step 6: Allocating buffers...")

        # Instruction buffer
        bo_instr = xrt.bo(device, len(insts), xrt.bo.flags.cacheable, kernel.group_id(1))
        bo_instr.write(insts, 0)
        bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Combined input/output buffers - ONLY 2 data arguments
        bo_input = xrt.bo(device, combined_buffer_size, xrt.bo.flags.host_only, kernel.group_id(3))
        bo_output = xrt.bo(device, combined_buffer_size, xrt.bo.flags.host_only, kernel.group_id(4))

        print(f"  Instruction buffer: {len(insts)} bytes (group 1)")
        print(f"  Input buffer: {combined_buffer_size} bytes (group 3)")
        print(f"  Output buffer: {combined_buffer_size} bytes (group 4)")
        print()

        # Step 7: Prepare test data
        print("Step 7: Preparing test data...")

        # Generate 4 frames of random test data
        input_combined = bytearray(combined_buffer_size)
        expected = []

        np.random.seed(42)  # Reproducible results
        for i in range(num_frames):
            # Generate random input for each frame
            data = np.random.randn(num_elements).astype(np.float32)
            bf16_data = float_to_bf16(data)

            # Pack into combined buffer at appropriate offset
            offset = i * buffer_size_per_frame
            input_combined[offset:offset + buffer_size_per_frame] = bf16_data

            # Calculate expected output
            expected.append(softmax_ref(data))

            print(f"    Frame {i}: offset={offset}, max={np.max(data):.4f}, min={np.min(data):.4f}")

        print()

        # Step 8: Execute kernel
        print("Step 8: Executing kernel (warmup + timing runs)...")

        # Warmup run
        bo_input.write(bytes(input_combined), 0)
        bo_input.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        opcode = 3
        run = kernel(opcode, bo_instr, len(insts), bo_input, bo_output)
        run.wait()

        bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        print("  Warmup run completed")

        # Timing runs
        times = []
        for iteration in range(num_iterations):
            # Sync input
            bo_input.write(bytes(input_combined), 0)
            bo_input.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            # Execute and time
            start = time.perf_counter()
            run = kernel(opcode, bo_instr, len(insts), bo_input, bo_output)
            run.wait()
            end = time.perf_counter()

            times.append(end - start)

            # Sync output
            bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

        print(f"  Completed {num_iterations} timing runs")
        print()

        # Step 9: Read and validate results
        print("Step 9: Reading and validating results...")

        output_bytes = bo_output.read(combined_buffer_size, 0).tobytes()

        max_errors = []
        mean_errors = []

        for i in range(num_frames):
            # Extract frame from combined output
            offset = i * buffer_size_per_frame
            frame_bytes = output_bytes[offset:offset + buffer_size_per_frame]
            actual = bf16_to_float(frame_bytes)

            # Calculate error vs expected
            abs_error = np.abs(actual - expected[i])
            max_err = np.max(abs_error)
            mean_err = np.mean(abs_error)

            max_errors.append(max_err)
            mean_errors.append(mean_err)

            # Check if results sum to 1 (softmax property)
            sum_actual = np.sum(actual)

            print(f"    Frame {i}: max_error={max_err:.6f}, mean_error={mean_err:.6f}, sum={sum_actual:.6f}")

        print()

        # Step 10: Performance analysis
        print("Step 10: Performance Analysis")
        print("=" * 70)

        avg_time = np.mean(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        std_time = np.std(times) * 1000

        print(f"Timing Results:")
        print(f"  Average time (4 frames): {avg_time:.3f} ms")
        print(f"  Min time: {min_time:.3f} ms")
        print(f"  Max time: {max_time:.3f} ms")
        print(f"  Std deviation: {std_time:.3f} ms")
        print(f"  Per-frame time: {avg_time/num_frames:.3f} ms")
        print()

        # Performance comparison
        single_time = 1.565  # ms from single kernel test
        two_tile_time = 1.548  # ms for 2 frames parallel (0.774 ms/frame)

        print("Performance Comparison:")
        print(f"  Sequential (4x single): {single_time * num_frames:.3f} ms")
        print(f"  2-tile (scaled to 4 frames): {two_tile_time * 2:.3f} ms")
        print(f"  4-tile combined: {avg_time:.3f} ms")
        print()

        speedup_vs_sequential = (single_time * num_frames) / avg_time
        speedup_vs_2tile = (two_tile_time * 2) / avg_time

        print(f"Speedup Analysis:")
        print(f"  Speedup vs sequential: {speedup_vs_sequential:.2f}x")
        print(f"  Speedup vs 2-tile scaled: {speedup_vs_2tile:.2f}x")
        print()

        # Accuracy summary
        overall_max_error = max(max_errors)
        overall_mean_error = np.mean(mean_errors)

        print("Accuracy Summary:")
        print(f"  Overall max error: {overall_max_error:.6f}")
        print(f"  Overall mean error: {overall_mean_error:.6f}")

        # Determine pass/fail
        accuracy_pass = overall_max_error < 0.01  # Reasonable threshold for BF16
        performance_pass = avg_time < 3.0  # Should be well under 3ms for 4 frames

        print()
        print("=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        print(f"  Accuracy: {'PASS' if accuracy_pass else 'FAIL'} (max error {overall_max_error:.6f} < 0.01)")
        print(f"  Performance: {'PASS' if performance_pass else 'FAIL'} ({avg_time:.3f} ms < 3.0 ms)")
        print()

        if accuracy_pass and performance_pass:
            print("OVERALL: SUCCESS - 4-tile multi-column kernel working correctly!")
            print()
            print("The combined buffer approach successfully works around the")
            print("XRT 5-argument limitation while enabling 4-tile parallelism.")
            return 0
        else:
            print("OVERALL: ISSUES DETECTED - see details above")
            return 1

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
