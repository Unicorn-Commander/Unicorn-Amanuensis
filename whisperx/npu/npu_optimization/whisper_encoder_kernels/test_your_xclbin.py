#!/usr/bin/env python3
"""
Test YOUR 13KB XCLBIN with the WORKING pyxrt API pattern
Based on the successful test_softmax.py pattern
"""

import numpy as np
import pyxrt as xrt
import struct
import time
import sys

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

def test_xclbin(xclbin_path, insts_path, num_elements=1024):
    """Test an XCLBIN with the working API pattern"""
    print("=" * 70)
    print(f"Testing XCLBIN: {xclbin_path}")
    print("=" * 70)
    print()

    print(f"Configuration:")
    print(f"  XCLBIN: {xclbin_path}")
    print(f"  Instructions: {insts_path}")
    print(f"  Elements: {num_elements}")
    print()

    try:
        # Step 1: Load XCLBIN using the WORKING API
        print("Step 1: Loading XCLBIN...")
        device = xrt.device(0)
        xclbin_obj = xrt.xclbin(xclbin_path)
        uuid = xclbin_obj.get_uuid()
        device.register_xclbin(xclbin_obj)  # This is the KEY - not load_xclbin()!
        print("✅ XCLBIN loaded successfully")
        print()

        # Step 2: Create hardware context
        print("Step 2: Creating hardware context...")
        hw_ctx = xrt.hw_context(device, uuid)
        print("✅ Hardware context created")
        print()

        # Step 3: Get kernel
        print("Step 3: Getting kernel...")
        kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
        print("✅ Kernel found: MLIR_AIE")
        print()

        # Step 4: Load instruction sequence
        print("Step 4: Loading instruction sequence...")
        with open(insts_path, "rb") as f:
            insts = f.read()
        print(f"✅ Instructions loaded: {len(insts)} bytes")
        print()

        # Step 5: Prepare buffers
        print("Step 5: Preparing buffers...")
        buffer_size = num_elements * 2  # 2 bytes per BF16

        # Create test input
        input_floats = np.random.randn(num_elements).astype(np.float32) * 0.1
        input_bf16 = float_to_bf16(input_floats)

        # Allocate XRT buffers using the WORKING pattern
        bo_instr = xrt.bo(device, len(insts), xrt.bo.flags.cacheable, kernel.group_id(1))
        bo_input = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(3))
        bo_output = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(4))

        # Write instructions
        bo_instr.write(insts, 0)
        bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        print("✅ Buffers allocated and instructions written")
        print()

        # Step 6: Run kernel
        print("Step 6: Running kernel on NPU...")
        num_iterations = 5
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

        # Step 7: Performance metrics
        print("Step 7: Performance Measurements:")
        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000

        print(f"  Average time: {avg_time:.3f} ms")
        print(f"  Std deviation: {std_time:.3f} ms")
        print(f"  Min time: {min_time:.3f} ms")
        print()

        # Step 8: Output verification
        print("Step 8: Output check:")
        print(f"  Input range: [{np.min(input_floats):.4f}, {np.max(input_floats):.4f}]")
        print(f"  Output range: [{np.min(output_floats):.4f}, {np.max(output_floats):.4f}]")
        print(f"  Output mean: {np.mean(output_floats):.6f}")
        print(f"  Output std: {np.std(output_floats):.6f}")
        print()

        print("Sample Output (first 10 elements):")
        print(f"  Input:  {input_floats[:10]}")
        print(f"  Output: {output_floats[:10]}")
        print()

        print("=" * 70)
        print("✅ TEST COMPLETE - Your XCLBIN is working on NPU!")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 test_your_xclbin.py <xclbin_path> <insts_path> [num_elements]")
        print()
        print("Example:")
        print("  python3 test_your_xclbin.py mel_fft.xclbin insts.txt 1024")
        sys.exit(1)

    xclbin_path = sys.argv[1]
    insts_path = sys.argv[2]
    num_elements = int(sys.argv[3]) if len(sys.argv) > 3 else 1024

    exit(test_xclbin(xclbin_path, insts_path, num_elements))
