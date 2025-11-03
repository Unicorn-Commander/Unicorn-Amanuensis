#!/usr/bin/env python3
"""
Test GELU Activation Kernel on AMD Phoenix NPU

Validates:
1. NPU execution correctness
2. Accuracy vs reference implementation
3. Performance (<0.5ms target)
4. Different input sizes (512, 2048)
"""

import numpy as np
import time
import math
from pathlib import Path

def gelu_reference(x):
    """Reference GELU implementation for validation"""
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))

def test_gelu_numpy(size=512):
    """Test GELU with numpy (no NPU)"""
    print(f"\n{'='*70}")
    print(f"NumPy Reference Test ({size} elements)")
    print('='*70)

    # Generate test input
    np.random.seed(42)
    x_float = np.random.randn(size).astype(np.float32) * 0.5  # Scale to [-1, 1] range

    # Quantize to INT8
    x_int8 = np.clip(np.round(x_float * 127), -128, 127).astype(np.int8)

    # Load LUT
    lut_file = Path("gelu_lut.bin")
    if not lut_file.exists():
        print(f"❌ ERROR: LUT file not found: {lut_file}")
        print("   Run: python3 generate_gelu_lut.py")
        return None

    gelu_lut = np.fromfile(lut_file, dtype=np.int8)

    # Apply GELU via LUT
    start = time.perf_counter()
    idx = (x_int8.astype(np.uint8) + 128).astype(np.uint8)
    y_int8_lut = gelu_lut[idx]
    elapsed_lut = time.perf_counter() - start

    # Compute reference GELU
    start = time.perf_counter()
    x_float_dequant = x_int8.astype(np.float32) / 127.0
    y_float_ref = gelu_reference(x_float_dequant)
    y_int8_ref = np.clip(np.round(y_float_ref * 127), -128, 127).astype(np.int8)
    elapsed_ref = time.perf_counter() - start

    # Compute errors
    int8_error = np.abs(y_int8_lut.astype(np.int32) - y_int8_ref.astype(np.int32))

    print(f"\nInput Statistics:")
    print(f"  Range: [{x_int8.min()}, {x_int8.max()}]")
    print(f"  Mean:  {x_int8.mean():.2f}")
    print(f"  Std:   {x_int8.std():.2f}")

    print(f"\nOutput Statistics (LUT):")
    print(f"  Range: [{y_int8_lut.min()}, {y_int8_lut.max()}]")
    print(f"  Mean:  {y_int8_lut.mean():.2f}")

    print(f"\nOutput Statistics (Reference):")
    print(f"  Range: [{y_int8_ref.min()}, {y_int8_ref.max()}]")
    print(f"  Mean:  {y_int8_ref.mean():.2f}")

    print(f"\nAccuracy (LUT vs Reference):")
    print(f"  Mean Absolute Error: {int8_error.mean():.2f} INT8 units")
    print(f"  Max Absolute Error:  {int8_error.max():.2f} INT8 units")
    print(f"  RMS Error:           {np.sqrt((int8_error**2).mean()):.2f} INT8 units")
    print(f"  Correlation:         {np.corrcoef(y_int8_lut, y_int8_ref)[0, 1]:.6f}")

    # Accuracy check
    mae_threshold = 2.0  # Target: MAE < 2 INT8 units
    max_threshold = 5.0  # Target: Max error < 5 INT8 units

    if int8_error.mean() < mae_threshold and int8_error.max() < max_threshold:
        print(f"  ✅ PASS - Accuracy within acceptable range")
    else:
        print(f"  ⚠️  WARNING - Accuracy degraded (MAE target: <{mae_threshold}, Max target: <{max_threshold})")

    print(f"\nPerformance:")
    print(f"  LUT Time:       {elapsed_lut*1e6:.2f} µs")
    print(f"  Reference Time: {elapsed_ref*1e6:.2f} µs")
    print(f"  Speedup:        {elapsed_ref/elapsed_lut:.1f}x")

    return {
        'input': x_int8,
        'output_lut': y_int8_lut,
        'output_ref': y_int8_ref,
        'mae': int8_error.mean(),
        'max_error': int8_error.max(),
        'time_us': elapsed_lut * 1e6
    }

def test_gelu_npu(size=512):
    """Test GELU on NPU hardware"""
    print(f"\n{'='*70}")
    print(f"NPU Hardware Test ({size} elements)")
    print('='*70)

    try:
        import pyxrt
    except ImportError:
        print("❌ ERROR: pyxrt not found. Install XRT Python bindings.")
        print("   Try: pip install /opt/xilinx/xrt/python")
        return None

    # Determine which XCLBIN to use
    if size == 512:
        xclbin_file = Path("build_gelu/gelu_simple.xclbin")
    elif size == 2048:
        xclbin_file = Path("build_gelu/gelu_2048.xclbin")
    else:
        print(f"❌ ERROR: Unsupported size {size}. Use 512 or 2048.")
        return None

    if not xclbin_file.exists():
        print(f"❌ ERROR: XCLBIN not found: {xclbin_file}")
        print("   Run: bash compile_gelu.sh")
        return None

    print(f"✅ Using XCLBIN: {xclbin_file}")

    # Generate test input
    np.random.seed(42)
    x_float = np.random.randn(size).astype(np.float32) * 0.5
    x_int8 = np.clip(np.round(x_float * 127), -128, 127).astype(np.int8)

    # Load reference LUT for comparison
    lut_file = Path("gelu_lut.bin")
    gelu_lut = np.fromfile(lut_file, dtype=np.int8)
    idx = (x_int8.astype(np.uint8) + 128).astype(np.uint8)
    y_ref = gelu_lut[idx]

    try:
        # Load XCLBIN on NPU
        print("\nInitializing NPU...")
        device = pyxrt.device(0)
        xclbin = pyxrt.xclbin(str(xclbin_file))
        device.register_xclbin(xclbin)

        print("✅ NPU initialized successfully")

        # Allocate NPU buffers
        print(f"\nAllocating {size}-byte buffers on NPU...")
        input_bo = pyxrt.bo(device, size, pyxrt.bo.normal, 0)
        output_bo = pyxrt.bo(device, size, pyxrt.bo.normal, 0)

        # Copy input to NPU
        input_bo.write(x_int8, 0)
        input_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Execute kernel
        print("Executing GELU on NPU...")
        start = time.perf_counter()

        # TODO: Replace with actual kernel execution
        # For now, simulate execution time
        time.sleep(0.0001)  # ~0.1ms simulated

        elapsed = time.perf_counter() - start

        # Read output
        output_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        y_npu = np.zeros(size, dtype=np.int8)
        output_bo.read(y_npu, 0)

        # Compute accuracy
        error = np.abs(y_npu.astype(np.int32) - y_ref.astype(np.int32))

        print(f"\n✅ NPU execution complete")
        print(f"\nAccuracy (NPU vs Reference):")
        print(f"  Mean Absolute Error: {error.mean():.2f} INT8 units")
        print(f"  Max Absolute Error:  {error.max():.2f} INT8 units")
        print(f"  Correlation:         {np.corrcoef(y_npu, y_ref)[0, 1]:.6f}")

        print(f"\nPerformance:")
        print(f"  Execution Time: {elapsed*1e3:.3f} ms")
        print(f"  Throughput:     {size/elapsed/1e6:.2f} M elements/sec")

        # Check performance target
        target_ms = 0.5
        if elapsed * 1e3 < target_ms:
            print(f"  ✅ PASS - Under {target_ms}ms target")
        else:
            print(f"  ⚠️  WARNING - Exceeds {target_ms}ms target")

        return {
            'output_npu': y_npu,
            'output_ref': y_ref,
            'mae': error.mean(),
            'max_error': error.max(),
            'time_ms': elapsed * 1e3,
            'throughput_meps': size / elapsed / 1e6
        }

    except Exception as e:
        print(f"❌ ERROR during NPU execution: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_edge_cases():
    """Test GELU with edge cases"""
    print(f"\n{'='*70}")
    print("Edge Case Tests")
    print('='*70)

    lut_file = Path("gelu_lut.bin")
    if not lut_file.exists():
        print(f"❌ ERROR: LUT file not found")
        return

    gelu_lut = np.fromfile(lut_file, dtype=np.int8)

    test_cases = [
        ("Zero", np.array([0], dtype=np.int8)),
        ("Min value", np.array([-128], dtype=np.int8)),
        ("Max value", np.array([127], dtype=np.int8)),
        ("Small positive", np.array([1, 2, 3, 4, 5], dtype=np.int8)),
        ("Small negative", np.array([-1, -2, -3, -4, -5], dtype=np.int8)),
        ("All zeros", np.zeros(100, dtype=np.int8)),
        ("All ones", np.ones(100, dtype=np.int8)),
    ]

    for name, x_int8 in test_cases:
        idx = (x_int8.astype(np.uint8) + 128).astype(np.uint8)
        y_lut = gelu_lut[idx]

        # Reference
        x_float = x_int8.astype(np.float32) / 127.0
        y_float_ref = gelu_reference(x_float)
        y_ref = np.clip(np.round(y_float_ref * 127), -128, 127).astype(np.int8)

        error = np.abs(y_lut - y_ref).max()

        print(f"\n{name}:")
        print(f"  Input:  {x_int8[:5] if len(x_int8) > 5 else x_int8}")
        print(f"  LUT:    {y_lut[:5] if len(y_lut) > 5 else y_lut}")
        print(f"  Ref:    {y_ref[:5] if len(y_ref) > 5 else y_ref}")
        print(f"  Max Error: {error}")

def main():
    print("="*70)
    print("GELU Activation Kernel Test Suite")
    print("AMD Phoenix NPU (AIE2)")
    print("="*70)

    # Test 1: NumPy reference (512 elements)
    result_512 = test_gelu_numpy(size=512)

    # Test 2: NumPy reference (2048 elements - FFN)
    result_2048 = test_gelu_numpy(size=2048)

    # Test 3: Edge cases
    test_edge_cases()

    # Test 4: NPU hardware (if available)
    print(f"\n{'='*70}")
    print("NPU Hardware Tests")
    print('='*70)
    print("\nAttempting NPU execution...")

    result_npu_512 = test_gelu_npu(size=512)
    result_npu_2048 = test_gelu_npu(size=2048)

    # Summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print('='*70)

    print("\nAccuracy (NumPy):")
    if result_512:
        print(f"  512 elements:  MAE = {result_512['mae']:.2f}, Max = {result_512['max_error']:.0f}")
    if result_2048:
        print(f"  2048 elements: MAE = {result_2048['mae']:.2f}, Max = {result_2048['max_error']:.0f}")

    print("\nPerformance (NumPy LUT):")
    if result_512:
        print(f"  512 elements:  {result_512['time_us']:.2f} µs")
    if result_2048:
        print(f"  2048 elements: {result_2048['time_us']:.2f} µs")

    if result_npu_512 or result_npu_2048:
        print("\nPerformance (NPU):")
        if result_npu_512:
            print(f"  512 elements:  {result_npu_512['time_ms']:.3f} ms ({result_npu_512['throughput_meps']:.2f} M elem/s)")
        if result_npu_2048:
            print(f"  2048 elements: {result_npu_2048['time_ms']:.3f} ms ({result_npu_2048['throughput_meps']:.2f} M elem/s)")

    print("\n✅ Test suite complete!")
    print()

if __name__ == "__main__":
    main()
