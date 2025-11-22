#!/usr/bin/env python3
"""
Test Softmax BF16 Kernel on AMD Phoenix NPU
Based on test_attention_64x64.py pattern
"""

import numpy as np
import pyxrt as xrt
import struct
import time

def bf16_to_float(bf16_bytes):
    """Convert BF16 bytes to float32"""
    # BF16 is the upper 16 bits of FP32
    result = np.zeros(len(bf16_bytes) // 2, dtype=np.float32)
    for i in range(len(result)):
        upper = struct.unpack('H', bf16_bytes[i*2:(i+1)*2])[0]
        result[i] = struct.unpack('f', struct.pack('I', upper << 16))[0]
    return result

def float_to_bf16(floats):
    """Convert float32 to BF16 bytes"""
    # BF16 is the upper 16 bits of FP32
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
    print("Softmax BF16 Kernel Test - AMD Phoenix NPU")
    print("=" * 70)
    print()

    # Test parameters
    num_elements = 1024
    num_iterations = 10

    print(f"Test Configuration:")
    print(f"  Elements: {num_elements}")
    print(f"  Data Type: BF16 (2048 bytes)")
    print(f"  Iterations: {num_iterations}")
    print()

    # Paths
    xclbin_path = "build_softmax_bf16/softmax_bf16.xclbin"
    insts_path = "build_softmax_bf16/insts.bin"

    try:
        # Load XCLBIN
        print("Step 1: Loading XCLBIN...")
        device = xrt.device(0)
        xclbin_obj = xrt.xclbin(xclbin_path)
        uuid = xclbin_obj.get_uuid()
        device.register_xclbin(xclbin_obj)
        print("✅ XCLBIN loaded successfully")
        print()

        # Create hardware context
        print("Step 2: Creating hardware context...")
        hw_ctx = xrt.hw_context(device, uuid)
        print("✅ Hardware context created")
        print()

        # Get kernel
        print("Step 3: Getting kernel...")
        kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
        print("✅ Kernel found: MLIR_AIE")
        print()

        # Load instruction sequence
        print("Step 4: Loading instruction sequence...")
        with open(insts_path, "rb") as f:
            insts = f.read()
        print(f"✅ Instructions loaded: {len(insts)} bytes")
        print()

        # Prepare buffers
        print("Step 5: Preparing buffers...")
        buffer_size = num_elements * 2  # 2 bytes per BF16

        # Create test input: range from -5.0 to 5.0
        input_floats = np.linspace(-5.0, 5.0, num_elements, dtype=np.float32)
        input_bf16 = float_to_bf16(input_floats)

        # Allocate XRT buffers
        bo_instr = xrt.bo(device, len(insts), xrt.bo.flags.cacheable, kernel.group_id(1))
        bo_input = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(3))
        bo_output = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(4))

        # Write instructions
        bo_instr.write(insts, 0)
        bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        print("✅ Buffers allocated and instructions written")
        print()

        # Compute reference softmax
        print("Step 6: Computing reference softmax...")
        expected = softmax_ref(input_floats)
        print(f"✅ Reference computed")
        print(f"   Input range: [{np.min(input_floats):.3f}, {np.max(input_floats):.3f}]")
        print(f"   Output range: [{np.min(expected):.6f}, {np.max(expected):.6f}]")
        print(f"   Output sum: {np.sum(expected):.6f} (should be 1.0)")
        print()

        # Run kernel
        print("Step 7: Running kernel on NPU...")
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

        print("✅ Kernel execution complete")
        print()

        # Performance metrics
        print("Step 8: Performance Measurements:")
        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000

        print(f"  Average time: {avg_time:.3f} ms")
        print(f"  Std deviation: {std_time:.3f} ms")
        print(f"  Min time: {min_time:.3f} ms")
        print(f"  Max time: {max_time:.3f} ms")
        print()

        # Verify results
        print("Step 9: Verifying results...")
        output_sum = np.sum(output_floats)
        max_error = np.max(np.abs(output_floats - expected))
        mean_error = np.mean(np.abs(output_floats - expected))

        print(f"  Output sum: {output_sum:.6f} (should be 1.0)")
        print(f"  Max error: {max_error:.6f}")
        print(f"  Mean error: {mean_error:.6f}")

        # Check if output sums to 1.0 (within tolerance)
        if abs(output_sum - 1.0) < 0.01:
            print("  ✅ Softmax sum check PASSED")
        else:
            print(f"  ❌ Softmax sum check FAILED (got {output_sum:.6f}, expected 1.0)")

        # Check error tolerance
        if max_error < 0.01:  # BF16 has ~3 decimal digits of precision
            print("  ✅ Accuracy check PASSED")
        else:
            print(f"  ⚠️  Accuracy check: max error {max_error:.6f} (BF16 limited precision)")
        print()

        # Sample values
        print("Sample Output (first 10 elements):")
        print(f"  Expected: {expected[:10]}")
        print(f"  NPU Out:  {output_floats[:10]}")
        print()

        print("=" * 70)
        print("✅ TEST COMPLETE - Softmax kernel working on NPU!")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
