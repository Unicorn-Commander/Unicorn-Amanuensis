#!/usr/bin/env python3
"""
Test Batched Softmax BF16 Kernel on AMD Phoenix NPU
Processes 4 softmax operations per invocation to achieve 3x speedup
"""

import numpy as np
import pyxrt as xrt
import struct
import time

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
    print("Batched Softmax BF16 Kernel Test - AMD Phoenix NPU")
    print("=" * 70)
    print()

    # Test parameters
    batch_size = 4
    elements_per_frame = 1024
    num_elements = batch_size * elements_per_frame
    num_iterations = 10

    print(f"Test Configuration:")
    print(f"  Batch size: {batch_size} softmax operations")
    print(f"  Elements per frame: {elements_per_frame}")
    print(f"  Total elements: {num_elements}")
    print(f"  Data Type: BF16 ({num_elements * 2} bytes)")
    print(f"  Iterations: {num_iterations}")
    print()

    # Paths
    xclbin_path = "build_softmax_batched/softmax_batched_bf16.xclbin"
    insts_path = "build_softmax_batched/insts.bin"

    try:
        # Load XCLBIN
        print("Step 1: Loading XCLBIN...")
        device = xrt.device(0)
        xclbin_obj = xrt.xclbin(xclbin_path)
        uuid = xclbin_obj.get_uuid()
        device.register_xclbin(xclbin_obj)
        print("XCLBIN loaded successfully")
        print()

        # Create hardware context
        print("Step 2: Creating hardware context...")
        hw_ctx = xrt.hw_context(device, uuid)
        print("Hardware context created")
        print()

        # Get kernel
        print("Step 3: Getting kernel...")
        kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
        print("Kernel found: MLIR_AIE")
        print()

        # Load instruction sequence
        print("Step 4: Loading instruction sequence...")
        with open(insts_path, "rb") as f:
            insts = f.read()
        print(f"Instructions loaded: {len(insts)} bytes")
        print()

        # Prepare buffers
        print("Step 5: Preparing buffers...")
        buffer_size = num_elements * 2  # 2 bytes per BF16

        # Create test input: 4 different ranges for 4 softmax operations
        input_floats = np.zeros(num_elements, dtype=np.float32)
        for i in range(batch_size):
            start_idx = i * elements_per_frame
            end_idx = (i + 1) * elements_per_frame
            # Different ranges for each frame to test independence
            input_floats[start_idx:end_idx] = np.linspace(
                -5.0 + i, 5.0 + i, elements_per_frame, dtype=np.float32
            )
        input_bf16 = float_to_bf16(input_floats)

        # Allocate XRT buffers
        bo_instr = xrt.bo(device, len(insts), xrt.bo.flags.cacheable, kernel.group_id(1))
        bo_input = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(3))
        bo_output = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(4))

        # Write instructions
        bo_instr.write(insts, 0)
        bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        print("Buffers allocated and instructions written")
        print()

        # Compute reference softmax for each frame
        print("Step 6: Computing reference softmax...")
        expected = np.zeros(num_elements, dtype=np.float32)
        for i in range(batch_size):
            start_idx = i * elements_per_frame
            end_idx = (i + 1) * elements_per_frame
            expected[start_idx:end_idx] = softmax_ref(input_floats[start_idx:end_idx])
        print("Reference computed for all 4 frames")
        print()

        # Run kernel
        print("Step 7: Running batched kernel on NPU...")
        times = []
        opcode = 3  # Standard NPU kernel opcode

        for i in range(num_iterations):
            # Write input
            bo_input.write(input_bf16, 0)
            bo_input.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            # Execute kernel with opcode
            start = time.perf_counter()
            run = kernel(opcode, bo_instr, len(insts), bo_input, bo_output)
            run.wait()
            end = time.perf_counter()
            times.append(end - start)

            # Read output
            bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            output_bytes = bo_output.read(buffer_size, 0).tobytes()
            output_floats = bf16_to_float(output_bytes)

        print("Kernel execution complete")
        print()

        # Performance metrics
        print("Step 8: Performance Measurements:")
        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        per_frame_time = avg_time / batch_size

        print(f"  Total batch time (avg): {avg_time:.3f} ms")
        print(f"  Per-frame time: {per_frame_time:.3f} ms")
        print(f"  Std deviation: {std_time:.3f} ms")
        print(f"  Min time: {min_time:.3f} ms")
        print(f"  Max time: {max_time:.3f} ms")
        print()

        # Compare with single kernel baseline
        single_kernel_time = 1.565  # ms from SUCCESS_REPORT.md
        speedup = single_kernel_time / per_frame_time
        print(f"  Single kernel baseline: {single_kernel_time:.3f} ms")
        print(f"  Batched per-frame: {per_frame_time:.3f} ms")
        print(f"  SPEEDUP: {speedup:.2f}x")
        print()

        # Verify results for each frame
        print("Step 9: Verifying results for each frame...")
        all_passed = True
        for i in range(batch_size):
            start_idx = i * elements_per_frame
            end_idx = (i + 1) * elements_per_frame
            
            frame_output = output_floats[start_idx:end_idx]
            frame_expected = expected[start_idx:end_idx]
            
            output_sum = np.sum(frame_output)
            max_error = np.max(np.abs(frame_output - frame_expected))
            mean_error = np.mean(np.abs(frame_output - frame_expected))

            sum_ok = abs(output_sum - 1.0) < 0.01
            error_ok = max_error < 0.01

            status = "PASS" if (sum_ok and error_ok) else "FAIL"
            if not (sum_ok and error_ok):
                all_passed = False

            print(f"  Frame {i}: sum={output_sum:.4f}, max_err={max_error:.6f}, mean_err={mean_error:.6f} [{status}]")

        print()

        if all_passed:
            print("=" * 70)
            print("BATCHED SOFTMAX KERNEL TEST PASSED!")
            print(f"Speedup: {speedup:.2f}x (target: 3.0x)")
            print("=" * 70)
        else:
            print("=" * 70)
            print("SOME FRAMES FAILED - Check output above")
            print("=" * 70)

        return 0 if all_passed else 1

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
