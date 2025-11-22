#!/usr/bin/env python3
"""
Test LayerNorm NPU kernel with XRT runtime.
Tests the compiled layernorm_512_nosqrt.xclbin on NPU hardware.
"""

import os
import sys
import numpy as np
import struct

# XRT Python bindings
try:
    import pyxrt as xrt
    HAS_XRT = True
except ImportError:
    print("WARNING: pyxrt not available, attempting direct XRT import")
    try:
        import xrt
        HAS_XRT = True
    except ImportError:
        print("ERROR: XRT Python bindings not found")
        HAS_XRT = False
        sys.exit(1)

def bf16_to_float32(bf16_bytes):
    """Convert BF16 bytes to float32."""
    # BF16 is stored as uint16, convert to float32 by shifting to upper 16 bits
    bf16_int = struct.unpack('H', bf16_bytes)[0]
    float32_int = bf16_int << 16
    return struct.unpack('f', struct.pack('I', float32_int))[0]

def float32_to_bf16(f):
    """Convert float32 to BF16 bytes."""
    # Get float32 as int, take upper 16 bits
    float32_bytes = struct.pack('f', f)
    float32_int = struct.unpack('I', float32_bytes)[0]
    bf16_int = (float32_int >> 16) & 0xFFFF
    return struct.pack('H', bf16_int)

def layernorm_reference(x, eps=1e-5):
    """Reference LayerNorm implementation in NumPy."""
    mean = np.mean(x)
    var = np.var(x)
    return (x - mean) / np.sqrt(var + eps)

def test_layernorm_npu():
    """Test LayerNorm kernel on NPU."""

    print("=" * 70)
    print("NPU LayerNorm Test - layernorm_512_nosqrt")
    print("=" * 70)

    # Configuration
    XCLBIN_PATH = "build_layernorm_nosqrt/main.xclbin"
    EMBEDDING_DIM = 512
    BUFFER_SIZE = EMBEDDING_DIM * 2  # 2 bytes per BF16

    if not os.path.exists(XCLBIN_PATH):
        print(f"ERROR: XCLBIN not found at {XCLBIN_PATH}")
        return False

    print(f"\n1. Loading XCLBIN: {XCLBIN_PATH}")
    print(f"   File size: {os.path.getsize(XCLBIN_PATH)} bytes")

    # Initialize XRT device
    try:
        device = xrt.device(0)  # Use first NPU device
        print(f"   ✓ NPU device opened: /dev/accel/accel0")
    except Exception as e:
        print(f"   ✗ Failed to open NPU device: {e}")
        return False

    # Load XCLBIN
    try:
        xclbin_uuid = device.load_xclbin(XCLBIN_PATH)
        print(f"   ✓ XCLBIN loaded, UUID: {xclbin_uuid}")
    except Exception as e:
        print(f"   ✗ Failed to load XCLBIN: {e}")
        return False

    # Generate test data
    print(f"\n2. Preparing test data ({EMBEDDING_DIM} elements)")
    np.random.seed(42)
    input_fp32 = np.random.randn(EMBEDDING_DIM).astype(np.float32)

    # Convert to BF16 for NPU
    input_bf16 = b''.join(float32_to_bf16(x) for x in input_fp32)

    print(f"   Input statistics:")
    print(f"     Mean: {input_fp32.mean():.6f}")
    print(f"     Std:  {input_fp32.std():.6f}")
    print(f"     Min:  {input_fp32.min():.6f}")
    print(f"     Max:  {input_fp32.max():.6f}")

    # Allocate buffers
    print(f"\n3. Allocating NPU buffers ({BUFFER_SIZE} bytes each)")
    try:
        input_bo = xrt.bo(device, BUFFER_SIZE, xrt.bo.flags.host_only, 0)
        output_bo = xrt.bo(device, BUFFER_SIZE, xrt.bo.flags.host_only, 0)
        print(f"   ✓ Buffers allocated")
    except Exception as e:
        print(f"   ✗ Failed to allocate buffers: {e}")
        return False

    # Write input data
    print(f"\n4. Writing input data to NPU buffer")
    try:
        input_bo.write(input_bf16, 0)
        input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, BUFFER_SIZE, 0)
        print(f"   ✓ Input data transferred to NPU")
    except Exception as e:
        print(f"   ✗ Failed to write input: {e}")
        return False

    # Get kernel handle
    print(f"\n5. Getting kernel handle")
    try:
        # Try different possible kernel names
        kernel_names = [
            "test_layernorm_512_nosqrt",
            "layernorm_512_nosqrt",
            "MLIR_AIE"
        ]

        kernel = None
        for name in kernel_names:
            try:
                kernel = xrt.kernel(device, xclbin_uuid, name)
                print(f"   ✓ Kernel found: {name}")
                break
            except:
                continue

        if kernel is None:
            print(f"   ✗ No kernel found with names: {kernel_names}")
            return False

    except Exception as e:
        print(f"   ✗ Failed to get kernel: {e}")
        return False

    # Execute kernel
    print(f"\n6. Executing kernel on NPU")
    try:
        run = kernel(input_bo, output_bo)
        state = run.wait(timeout=1000)  # 1 second timeout
        print(f"   ✓ Kernel execution completed: {state}")
    except Exception as e:
        print(f"   ✗ Kernel execution failed: {e}")
        return False

    # Read output
    print(f"\n7. Reading output from NPU")
    try:
        output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, BUFFER_SIZE, 0)
        output_bf16 = output_bo.read(BUFFER_SIZE, 0).tobytes()
        print(f"   ✓ Output data retrieved ({len(output_bf16)} bytes)")
    except Exception as e:
        print(f"   ✗ Failed to read output: {e}")
        return False

    # Convert BF16 back to float32
    output_fp32 = np.array([
        bf16_to_float32(output_bf16[i:i+2])
        for i in range(0, len(output_bf16), 2)
    ])

    # Compute reference
    print(f"\n8. Computing reference (CPU)")
    reference_fp32 = layernorm_reference(input_fp32)

    # Compare results
    print(f"\n9. Validation Results")
    print(f"   NPU output statistics:")
    print(f"     Mean: {output_fp32.mean():.6f}")
    print(f"     Std:  {output_fp32.std():.6f}")
    print(f"     Min:  {output_fp32.min():.6f}")
    print(f"     Max:  {output_fp32.max():.6f}")

    print(f"\n   Reference output statistics:")
    print(f"     Mean: {reference_fp32.mean():.6f}")
    print(f"     Std:  {reference_fp32.std():.6f}")
    print(f"     Min:  {reference_fp32.min():.6f}")
    print(f"     Max:  {reference_fp32.max():.6f}")

    # Compute error metrics
    abs_error = np.abs(output_fp32 - reference_fp32)
    rel_error = abs_error / (np.abs(reference_fp32) + 1e-8)

    print(f"\n   Error metrics:")
    print(f"     Mean absolute error: {abs_error.mean():.6f}")
    print(f"     Max absolute error:  {abs_error.max():.6f}")
    print(f"     Mean relative error: {rel_error.mean():.6f}")
    print(f"     Max relative error:  {rel_error.max():.6f}")

    # Correlation
    correlation = np.corrcoef(output_fp32, reference_fp32)[0, 1]
    print(f"     Correlation:         {correlation:.6f}")

    # Check if results are acceptable
    if correlation > 0.95 and abs_error.mean() < 0.1:
        print(f"\n   ✓ VALIDATION PASSED")
        print(f"     NPU LayerNorm kernel is working correctly!")
        success = True
    else:
        print(f"\n   ✗ VALIDATION FAILED")
        print(f"     Output does not match reference")
        success = False

    # Show sample values
    print(f"\n   Sample values (first 10 elements):")
    print(f"     Index | Input      | NPU Output | Reference  | Error")
    print(f"     " + "-" * 60)
    for i in range(10):
        error = abs(output_fp32[i] - reference_fp32[i])
        print(f"     {i:5d} | {input_fp32[i]:10.6f} | {output_fp32[i]:10.6f} | "
              f"{reference_fp32[i]:10.6f} | {error:.6f}")

    print("\n" + "=" * 70)

    return success

if __name__ == "__main__":
    success = test_layernorm_npu()
    sys.exit(0 if success else 1)
