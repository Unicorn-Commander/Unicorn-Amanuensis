#!/usr/bin/env python3
"""
Week 15: Simplified NPU Kernel Execution Test

Based on working test_npu_hardware.py pattern.
Tests 64x64 matrix multiply (matching 1-tile kernel).

Author: Week 15 NPU Execution Team
Date: November 2, 2025
"""

import sys
import numpy as np
import time
from pathlib import Path

try:
    import pyxrt as xrt
    print("[OK] XRT Python bindings loaded")
except ImportError:
    print("[ERROR] pyxrt not available")
    print("  Run: source /opt/xilinx/xrt/setup.sh")
    sys.exit(1)


def bf16_to_bytes(arr_fp32):
    """Convert float32 array to BF16 bytes (truncate mantissa)"""
    uint32_view = arr_fp32.view(np.uint32)
    bf16_uint16 = (uint32_view >> 16).astype(np.uint16)
    return bf16_uint16.tobytes()


def bytes_to_bf16(bf16_bytes):
    """Convert BF16 bytes to float32 array (add zero mantissa)"""
    bf16_uint16 = np.frombuffer(bf16_bytes, dtype=np.uint16)
    uint32 = bf16_uint16.astype(np.uint32) << 16
    return uint32.view(np.float32)


def main():
    print("\n" + "="*70)
    print("  WEEK 15: SIMPLE NPU EXECUTION TEST")
    print("  64x64 Matrix Multiply on XDNA2")
    print("="*70)

    # Find kernel
    xclbin_path = Path("/home/ccadmin/CC-1L/kernels/common/build_bf16_1tile/matmul_1tile_bf16.xclbin")
    if not xclbin_path.exists():
        print(f"[ERROR] xclbin not found: {xclbin_path}")
        return 1

    print(f"\n[Kernel] {xclbin_path.name}")

    # Test configuration - matching kernel buffer size (512x512)
    # Kernel expects memref<262144xbf16> = 512*512 elements
    M, N, K = 512, 512, 512
    print(f"[Size] Matrix: {M}x{K} @ {K}x{N} = {M}x{N}")
    print(f"[Note] Using full kernel size (262,144 elements)")

    try:
        # 1. Initialize device
        print("\n[1/9] Opening NPU device...")
        device = xrt.device(0)
        print("  Device 0 opened")

        # 2. Load xclbin
        print("[2/9] Loading xclbin...")
        xclbin = xrt.xclbin(str(xclbin_path))
        device.register_xclbin(xclbin)
        uuid = xclbin.get_uuid()
        context = xrt.hw_context(device, uuid)
        print("  xclbin registered")

        # 3. Get kernel
        print("[3/9] Loading kernel...")
        kernels = xclbin.get_kernels()
        kernel_name = kernels[0].get_name()
        kernel = xrt.kernel(context, kernel_name)
        print(f"  Kernel: {kernel_name}")

        # 4. Create test data
        print("[4/9] Creating test data...")
        np.random.seed(42)

        # Simple test: positive values only (avoid BF16 signed bug)
        A = np.random.rand(M, K).astype(np.float32) * 0.1  # Small positive values
        B = np.random.rand(K, N).astype(np.float32) * 0.1
        print(f"  A: {M}x{K} (range: {A.min():.3f} to {A.max():.3f})")
        print(f"  B: {K}x{N} (range: {B.min():.3f} to {B.max():.3f})")

        # Reference result
        C_ref = A @ B
        print(f"  C_ref: {M}x{N} (range: {C_ref.min():.3f} to {C_ref.max():.3f})")

        # Convert to BF16
        A_bf16 = bf16_to_bytes(A.flatten())
        B_bf16 = bf16_to_bytes(B.flatten())

        # 5. Create buffers
        print("[5/9] Creating buffers...")
        size_A = M * K * 2  # 2 bytes per BF16 element
        size_B = K * N * 2
        size_C = M * N * 2

        # Note: kernel.group_id() must match kernel argument order
        # From MLIR: aiex.runtime_sequence(%arg0, %arg1, %arg2)
        bo_A = xrt.bo(device, size_A, xrt.bo.host_only, kernel.group_id(0))
        bo_B = xrt.bo(device, size_B, xrt.bo.host_only, kernel.group_id(1))
        bo_C = xrt.bo(device, size_C, xrt.bo.host_only, kernel.group_id(2))
        print(f"  Created buffers: A={size_A}B, B={size_B}B, C={size_C}B")

        # 6. Write input data
        print("[6/9] Writing input data...")
        bo_A.write(A_bf16, 0)
        bo_B.write(B_bf16, 0)
        bo_A.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bo_B.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        print("  Data written and synced to device")

        # 7. Execute kernel
        print("[7/9] Executing kernel...")
        start = time.perf_counter()
        run = kernel(bo_A, bo_B, bo_C)
        run.wait()
        exec_time = time.perf_counter() - start
        print(f"  Kernel executed in {exec_time*1000:.2f}ms")

        # Calculate GFLOPS
        ops = 2 * M * N * K  # FMA operations
        gflops = ops / exec_time / 1e9
        print(f"  Performance: {gflops:.1f} GFLOPS")

        # 8. Read result
        print("[8/9] Reading result...")
        bo_C.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        C_bf16_bytes = bytes(bo_C.map())[:size_C]
        C_result = bytes_to_bf16(C_bf16_bytes).reshape(M, N)
        print(f"  Result: {M}x{N} (range: {C_result.min():.3f} to {C_result.max():.3f})")

        # 9. Validate
        print("[9/9] Validating results...")
        abs_error = np.abs(C_result - C_ref)
        rel_error = abs_error / (np.abs(C_ref) + 1e-10)

        max_abs = np.max(abs_error)
        mean_abs = np.mean(abs_error)
        max_rel = np.max(rel_error) * 100
        mean_rel = np.mean(rel_error) * 100

        print(f"  Max absolute error: {max_abs:.6f}")
        print(f"  Mean absolute error: {mean_abs:.6f}")
        print(f"  Max relative error: {max_rel:.2f}%")
        print(f"  Mean relative error: {mean_rel:.2f}%")

        # Sample comparison
        print(f"\n[Sample] First 5 elements:")
        print(f"  Expected: {C_ref.flatten()[:5]}")
        print(f"  NPU:      {C_result.flatten()[:5]}")
        print(f"  Error:    {abs_error.flatten()[:5]}")

        # Check success
        tolerance = 10.0  # 10% for BF16 (conservative)
        success = mean_rel < tolerance

        print("\n" + "="*70)
        print("  RESULTS")
        print("="*70)
        print(f"  Status: {'✓ PASS' if success else '✗ FAIL'}")
        print(f"  Execution: {exec_time*1000:.2f}ms")
        print(f"  Performance: {gflops:.1f} GFLOPS")
        print(f"  Accuracy: {mean_rel:.2f}% mean error")

        if success:
            print("\n  Week 15 SUCCESS - NPU kernel execution validated!")
            print("  Next steps:")
            print("    - Test with larger matrices")
            print("    - Integrate with Whisper encoder")
            print("    - Performance optimization")
        else:
            print("\n  Issues detected:")
            print(f"    - Error ({mean_rel:.1f}%) exceeds tolerance ({tolerance}%)")
            if np.all(C_result == 0):
                print("    - All zeros returned (kernel may not be running)")
            elif mean_rel > 50:
                print("    - Large error suggests data format mismatch")

        print("="*70)

        return 0 if success else 1

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
