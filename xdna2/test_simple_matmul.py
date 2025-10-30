#!/usr/bin/env python3
"""
Simple standalone test for 4-tile INT8 matmul kernel.

This is a minimal test based on the proven run_npu_test_4tile_int8.py.
"""

import numpy as np
import time
import sys
import os
from pathlib import Path

# Add XRT bindings
sys.path.insert(0, "/opt/xilinx/xrt/python")

try:
    from aie.utils.xrt import AIE_Application
    print("✅ XRT bindings loaded")
except ImportError as e:
    print(f"❌ Failed to load XRT bindings: {e}")
    print("   Make sure ironenv is activated:")
    print("   source ~/mlir-aie/ironenv/bin/activate")
    sys.exit(1)

def main():
    print("=" * 70)
    print("Simple 4-Tile INT8 Matmul Test")
    print("=" * 70)
    print()

    # Matrix dimensions
    M, K, N = 512, 512, 512
    dtype_in = np.int8
    dtype_out = np.int32

    print(f"Matrix dimensions: ({M} x {K}) @ ({K} x {N})")
    print(f"Input type: {dtype_in.__name__}")
    print(f"Output type: {dtype_out.__name__}")
    print()

    # Generate test data
    print("[1/5] Generating test data...")
    A = np.random.randint(-8, 8, (M, K), dtype=dtype_in)
    B = np.random.randint(-8, 8, (K, N), dtype=dtype_in)

    # CPU reference
    print("[2/5] Computing CPU reference...")
    C_cpu = A.astype(np.int32) @ B.astype(np.int32)

    # Load kernel
    print("[3/5] Loading kernel...")
    kernel_dir = Path(__file__).parent / "kernels" / "common" / "build"
    xclbin_path = kernel_dir / "matmul_4tile_int8.xclbin"
    insts_path = kernel_dir / "insts_4tile_int8.bin"

    if not xclbin_path.exists():
        print(f"   ❌ XCLBin not found: {xclbin_path}")
        return 1
    if not insts_path.exists():
        print(f"   ❌ Instructions not found: {insts_path}")
        return 1

    try:
        app = AIE_Application(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
        print(f"   ✅ Kernel loaded: {xclbin_path.name}")

        # Flatten matrices
        A_flat = A.flatten()
        B_flat = B.flatten()

        # Register buffers
        app.register_buffer(3, dtype_in, (M * K,))
        app.register_buffer(4, dtype_in, (K * N,))
        app.register_buffer(5, dtype_out, (M * N,))
        print("   ✅ Buffers registered")

        # Write inputs
        app.buffers[3].write(A_flat)
        app.buffers[4].write(B_flat)
        print("   ✅ Inputs written")

        # Execute
        print("[4/5] Executing on NPU...")
        app.run()  # Warmup

        start = time.perf_counter()
        app.run()
        elapsed = time.perf_counter() - start

        # Read output
        C_npu = app.buffers[5].read().reshape(M, N)
        print(f"   ✅ Execution complete: {elapsed*1000:.2f}ms")

    except Exception as e:
        print(f"   ❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Verify
    print("[5/5] Verifying results...")
    errors = np.sum(C_npu != C_cpu)

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)

    if errors == 0:
        print("✅ PASS: 100% accuracy!")
    else:
        print(f"❌ FAIL: {errors} errors")
        return 1

    ops = 2 * M * K * N
    gflops = ops / elapsed / 1e9
    print(f"Elapsed: {elapsed*1000:.2f}ms")
    print(f"Performance: {gflops:.1f} GFLOPS")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
