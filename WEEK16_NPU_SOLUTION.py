#!/usr/bin/env python3
"""
Week 16: NPU Execution SOLUTION

ROOT CAUSE: Missing instruction buffer!
The runtime_sequence DMA configuration is compiled into insts.txt
and MUST be loaded separately from the xclbin.

SOLUTION: Use mlir-aie utility functions (setup_aie/execute) which
properly handle both xclbin AND instruction loading.

Author: Week 16 NPU Debugging Team
Date: November 2, 2025
"""

import sys
import numpy as np
import time
from pathlib import Path

# Add mlir-aie utilities to path
sys.path.append(str(Path.home() / "mlir-aie" / "python"))

try:
    from aie.utils.xrt import setup_aie, execute
    print("[OK] mlir-aie utilities loaded")
except ImportError as e:
    print(f"[ERROR] mlir-aie utilities not available: {e}")
    print("  Make sure mlir-aie Python package is installed")
    sys.exit(1)


def bf16_to_float32(bf16_uint16):
    """Convert BF16 (as uint16) to float32"""
    uint32 = bf16_uint16.astype(np.uint32) << 16
    return uint32.view(np.float32)


def float32_to_bf16(fp32):
    """Convert float32 to BF16 (as uint16)"""
    uint32 = fp32.view(np.uint32)
    return (uint32 >> 16).astype(np.uint16)


def main():
    print("\n" + "="*70)
    print("  WEEK 16: NPU EXECUTION SOLUTION")
    print("  64x64 Matrix Multiply with Instruction Loading")
    print("="*70)

    # Kernel files
    kernel_dir = Path("/home/ccadmin/CC-1L/kernels/common/build_bf16_1tile")
    xclbin_path = kernel_dir / "matmul_1tile_bf16.xclbin"
    instr_path = kernel_dir / "insts_1tile_bf16.bin"  # Use .bin (binary format)

    if not xclbin_path.exists():
        print(f"[ERROR] xclbin not found: {xclbin_path}")
        return 1

    if not instr_path.exists():
        print(f"[ERROR] Instructions not found: {instr_path}")
        return 1

    print(f"\n[Files]")
    print(f"  xclbin: {xclbin_path.name}")
    print(f"  instructions: {instr_path.name}")

    # Test configuration - 512x512 to match kernel memref size
    M, N, K = 512, 512, 512
    print(f"\n[Size] Matrix: {M}x{K} @ {K}x{N} = {M}x{N}")
    print(f"  Total elements per matrix: {M*N} (262,144)")

    try:
        # Create test data
        print("\n[1/6] Creating test data...")
        np.random.seed(42)

        # Use small positive values for BF16
        A_fp32 = np.random.rand(M, K).astype(np.float32) * 0.1
        B_fp32 = np.random.rand(K, N).astype(np.float32) * 0.1

        print(f"  A: {M}x{K} (range: {A_fp32.min():.3f} to {A_fp32.max():.3f})")
        print(f"  B: {K}x{N} (range: {B_fp32.min():.3f} to {B_fp32.max():.3f})")

        # Reference result (FP32)
        C_ref = A_fp32 @ B_fp32
        print(f"  C_ref: {M}x{N} (range: {C_ref.min():.3f} to {C_ref.max():.3f})")

        # Convert to BF16
        A_bf16 = float32_to_bf16(A_fp32.flatten())
        B_bf16 = float32_to_bf16(B_fp32.flatten())

        # Setup AIE application (this loads BOTH xclbin AND instructions!)
        print("\n[2/6] Setting up AIE application...")
        print("  This loads xclbin AND instruction sequence...")

        app = setup_aie(
            xclbin_path=str(xclbin_path),
            insts_path=str(instr_path),
            in_0_shape=(M*K,),  # Flat array for input A
            in_0_dtype=np.uint16,  # BF16 as uint16
            in_1_shape=(K*N,),  # Flat array for input B
            in_1_dtype=np.uint16,  # BF16 as uint16
            out_buf_shape=(M*N,),  # Flat array for output C
            out_buf_dtype=np.uint16,  # BF16 as uint16
            kernel_name="MLIR_AIE",
            verbosity=1
        )
        print("  ✓ AIE application setup complete")
        print("    (xclbin loaded, instructions loaded, buffers allocated)")

        # Write input data
        print("\n[3/6] Writing input data to buffers...")
        app.buffers[3].write(A_bf16)
        app.buffers[4].write(B_bf16)
        print("  ✓ Data written and synced to device")

        # Execute kernel
        print("\n[4/6] Executing kernel...")
        start = time.perf_counter()

        app.run()  # This runs: sync_to_device, kernel(), wait()

        exec_time = time.perf_counter() - start
        print(f"  ✓ Kernel executed in {exec_time*1000:.2f}ms")

        # Calculate performance
        ops = 2 * M * N * K  # FMA operations
        gflops = ops / exec_time / 1e9
        print(f"  Performance: {gflops:.1f} GFLOPS")

        # Read result
        print("\n[5/6] Reading result...")
        C_bf16 = app.buffers[5].read()  # This does sync_from_device + read
        C_fp32 = bf16_to_float32(C_bf16).reshape(M, N)
        print(f"  Result: {M}x{N} (range: {C_fp32.min():.6f} to {C_fp32.max():.6f})")

        # Validate
        print("\n[6/6] Validating results...")

        # Check for non-zero
        non_zero = np.count_nonzero(C_fp32)
        print(f"  Non-zero elements: {non_zero}/{M*N} ({non_zero/(M*N)*100:.1f}%)")

        if non_zero == 0:
            print("\n  ✗ FAIL: Still getting all zeros!")
            print("  This suggests a deeper issue with buffer configuration.")
            return 1

        # Calculate errors
        abs_error = np.abs(C_fp32 - C_ref)
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
        print(f"\n  [Sample] First 5 elements:")
        print(f"    Expected: {C_ref.flatten()[:5]}")
        print(f"    NPU:      {C_fp32.flatten()[:5]}")
        print(f"    Error:    {abs_error.flatten()[:5]}")

        # Check success
        tolerance = 10.0  # 10% for BF16
        success = mean_rel < tolerance

        print("\n" + "="*70)
        print("  RESULTS")
        print("="*70)
        print(f"  Status: {'✓ PASS' if success else '✗ FAIL'}")
        print(f"  Execution: {exec_time*1000:.2f}ms")
        print(f"  Performance: {gflops:.1f} GFLOPS")
        print(f"  Accuracy: {mean_rel:.2f}% mean error")

        if success:
            print("\n  🎉 SUCCESS - NPU KERNEL RETURNS CORRECT VALUES!")
            print("\n  Root cause identified:")
            print("    ❌ Week 15 test was missing instruction buffer loading")
            print("    ✓ Instructions (insts.txt) contain DMA configuration")
            print("    ✓ Must be loaded separately from xclbin")
            print("\n  Solution:")
            print("    1. Use mlir-aie utilities: setup_aie() and execute()")
            print("    2. Pass BOTH xclbin_path AND insts_path")
            print("    3. These handle instruction loading automatically")
            print("\n  Next steps:")
            print("    - Update all NPU test scripts to use setup_aie/execute")
            print("    - Test with larger matrices")
            print("    - Integrate with Whisper encoder")
            print("    - Document this pattern for future kernels")
        else:
            print("\n  Issues detected:")
            print(f"    - Error ({mean_rel:.1f}%) exceeds tolerance ({tolerance}%)")
            if mean_rel > 50:
                print("    - Large error suggests data format issue")

        print("="*70)

        # Cleanup
        del app

        return 0 if success else 1

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
