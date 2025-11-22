#!/usr/bin/env python3
"""
Test LayerNorm BF16 Kernel
AMD Phoenix NPU - XDNA1
"""

import numpy as np
import pyxrt as xrt
import struct
import time
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

def layernorm_ref(x, eps=1e-5):
    """Reference LayerNorm implementation (simplified without gamma/beta)"""
    mean = np.mean(x)
    variance = np.var(x)
    return (x - mean) / np.sqrt(variance + eps)

def main():
    print("=" * 70)
    print("LayerNorm BF16 Kernel Test - AMD Phoenix NPU")
    print("=" * 70)
    print()

    num_elements = 1024
    buffer_size = num_elements * 2  # 2048 bytes
    num_iterations = 10

    print(f"Test Configuration:")
    print(f"  Elements: {num_elements}")
    print(f"  Buffer size: {buffer_size} bytes")
    print(f"  Iterations: {num_iterations}")
    print()

    # Check for kernel files
    build_dir = os.path.dirname(os.path.abspath(__file__))
    xclbin_path = os.path.join(build_dir, "layernorm_bf16.xclbin")
    insts_path = os.path.join(build_dir, "insts.bin")

    if not os.path.exists(xclbin_path):
        print(f"ERROR: XCLBIN not found at {xclbin_path}")
        print("Please compile the LayerNorm kernel first.")
        return 1

    if not os.path.exists(insts_path):
        print(f"ERROR: Instructions not found at {insts_path}")
        return 1

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

        # Prepare test data
        print("Step 5: Preparing test data...")
        # Use standard normal distribution with some scale
        input_floats = np.random.randn(num_elements).astype(np.float32) * 2.0 + 0.5
        input_bf16 = float_to_bf16(input_floats)
        expected = layernorm_ref(input_floats)

        print(f"Input statistics:")
        print(f"  Mean: {np.mean(input_floats):.4f}")
        print(f"  Std:  {np.std(input_floats):.4f}")
        print(f"  Min:  {np.min(input_floats):.4f}")
        print(f"  Max:  {np.max(input_floats):.4f}")
        print()

        print(f"Expected output statistics (after LayerNorm):")
        print(f"  Mean: {np.mean(expected):.6f} (should be ~0)")
        print(f"  Std:  {np.std(expected):.6f} (should be ~1)")
        print()

        # Allocate XRT buffers
        print("Step 6: Allocating XRT buffers...")
        bo_instr = xrt.bo(device, len(insts), xrt.bo.flags.cacheable, kernel.group_id(1))
        bo_in = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(3))
        bo_out = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(4))

        bo_instr.write(insts, 0)
        bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        print("Buffers allocated and instruction buffer synced")
        print()

        # Run kernel
        print("Step 7: Running LayerNorm kernel on NPU...")
        times = []
        opcode = 3

        for iteration in range(num_iterations):
            bo_in.write(input_bf16, 0)
            bo_in.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            start = time.perf_counter()
            run = kernel(opcode, bo_instr, len(insts), bo_in, bo_out)
            run.wait()
            end = time.perf_counter()
            times.append(end - start)

            bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            output_bf16 = bo_out.read(buffer_size, 0).tobytes()
            output_floats = bf16_to_float(output_bf16)

        print("Kernel execution complete")
        print()

        # Performance metrics
        print("Step 8: Performance Measurements:")
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000

        print(f"  Average time: {avg_time:.3f} ms")
        print(f"  Std deviation: {std_time:.3f} ms")
        print(f"  Min time: {min_time:.3f} ms")
        print(f"  Max time: {max_time:.3f} ms")
        print()

        # Throughput
        elements_per_sec = num_elements / (avg_time / 1000)
        print(f"  Throughput: {elements_per_sec/1e6:.2f} M elements/second")
        print()

        # Verify accuracy
        print("Step 9: Verifying accuracy...")

        # Output statistics
        print(f"NPU output statistics:")
        print(f"  Mean: {np.mean(output_floats):.6f} (should be ~0)")
        print(f"  Std:  {np.std(output_floats):.6f} (should be ~1)")
        print()

        # Error analysis
        abs_error = np.abs(output_floats - expected)
        max_error = np.max(abs_error)
        mean_error = np.mean(abs_error)

        # Correlation
        if np.std(output_floats) > 0 and np.std(expected) > 0:
            correlation = np.corrcoef(output_floats, expected)[0, 1]
        else:
            correlation = 0.0

        print(f"Error Analysis:")
        print(f"  Max absolute error: {max_error:.6f}")
        print(f"  Mean absolute error: {mean_error:.6f}")
        print(f"  Correlation: {correlation:.6f}")
        print()

        # Determine pass/fail
        # LayerNorm BF16 might have some precision loss
        if max_error < 0.1 and correlation > 0.99:
            status = "PASS"
        elif max_error < 0.2 and correlation > 0.95:
            status = "MARGINAL PASS"
        else:
            status = "FAIL"

        print(f"Accuracy check: {status}")
        print()

        # Sample values comparison
        print("Sample values (first 10):")
        print(f"{'Index':<8} {'Input':<12} {'NPU Output':<12} {'Expected':<12} {'Error':<10}")
        print("-" * 54)
        for i in range(10):
            err = abs(output_floats[i] - expected[i])
            print(f"{i:<8} {input_floats[i]:<12.4f} {output_floats[i]:<12.4f} {expected[i]:<12.4f} {err:<10.6f}")
        print()

        print("=" * 70)
        if status == "PASS":
            print("LAYERNORM KERNEL TEST PASSED!")
        elif status == "MARGINAL PASS":
            print("LAYERNORM KERNEL TEST MARGINAL PASS (BF16 precision)")
        else:
            print("LAYERNORM KERNEL TEST FAILED!")
        print("=" * 70)

        return 0 if "PASS" in status else 1

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
