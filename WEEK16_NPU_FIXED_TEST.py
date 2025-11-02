#!/usr/bin/env python3
"""
Week 16: FIXED NPU Kernel Execution Test

ROOT CAUSE IDENTIFIED: Instruction buffer was missing!
The NPU requires a binary instruction sequence (insts.txt) to be loaded
alongside the xclbin. Without this, the kernel runs but does nothing.

This fixed version follows the working mlir-aie test pattern:
1. Load instruction binary from insts_1tile_bf16.txt
2. Create instruction buffer object (cacheable, group_id 1)
3. Pass opcode, instruction buffer, and data buffers to kernel
4. Use proper kernel argument order: opcode, bo_instr, instr_len, bo_A, bo_B, bo_C

Author: Week 16 NPU Debugging Team
Date: November 2, 2025
"""

import sys
import numpy as np
import time
import struct
import os
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


def load_instr_binary(filepath):
    """Load instruction binary file as uint32 array"""
    with open(filepath, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"{len(data)//4}I", data))


def main():
    print("\n" + "="*70)
    print("  WEEK 16: FIXED NPU EXECUTION TEST")
    print("  64x64 Matrix Multiply on XDNA2 (WITH INSTRUCTIONS)")
    print("="*70)

    # Find kernel and instructions
    kernel_dir = Path("/home/ccadmin/CC-1L/kernels/common/build_bf16_1tile")
    xclbin_path = kernel_dir / "matmul_1tile_bf16.xclbin"
    instr_path = kernel_dir / "insts_1tile_bf16.txt"

    if not xclbin_path.exists():
        print(f"[ERROR] xclbin not found: {xclbin_path}")
        return 1

    if not instr_path.exists():
        print(f"[ERROR] Instructions not found: {instr_path}")
        return 1

    print(f"\n[Kernel] {xclbin_path.name}")
    print(f"[Instructions] {instr_path.name}")

    # Test configuration - matching 1-tile kernel (64x64)
    M, N, K = 64, 64, 64
    print(f"[Size] Matrix: {M}x{K} @ {K}x{N} = {M}x{N}")

    try:
        # 1. Load instruction binary
        print("\n[1/10] Loading NPU instructions...")
        instr_v = load_instr_binary(instr_path)
        print(f"  Instruction count: {len(instr_v)}")

        # 2. Initialize device
        print("[2/10] Opening NPU device...")
        device = xrt.device(0)
        print("  Device 0 opened")

        # 3. Load xclbin
        print("[3/10] Loading xclbin...")
        xclbin = xrt.xclbin(str(xclbin_path))
        device.register_xclbin(xclbin)
        uuid = xclbin.get_uuid()
        context = xrt.hw_context(device, uuid)
        print("  xclbin registered")

        # 4. Get kernel
        print("[4/10] Loading kernel...")
        kernels = xclbin.get_kernels()
        kernel_name = kernels[0].get_name()
        kernel = xrt.kernel(context, kernel_name)
        print(f"  Kernel: {kernel_name}")

        # 5. Create test data
        print("[5/10] Creating test data...")
        np.random.seed(42)

        # Simple test: small positive values
        A = np.random.rand(M, K).astype(np.float32) * 0.1
        B = np.random.rand(K, N).astype(np.float32) * 0.1
        print(f"  A: {M}x{K} (range: {A.min():.3f} to {A.max():.3f})")
        print(f"  B: {K}x{N} (range: {B.min():.3f} to {B.max():.3f})")

        # Reference result
        C_ref = A @ B
        print(f"  C_ref: {M}x{N} (range: {C_ref.min():.3f} to {C_ref.max():.3f})")

        # Convert to BF16
        A_bf16 = bf16_to_bytes(A.flatten())
        B_bf16 = bf16_to_bytes(B.flatten())

        # 6. Create buffers (CRITICAL: instruction buffer is group_id(1) and cacheable!)
        print("[6/10] Creating buffers...")
        size_A = M * K * 2  # 2 bytes per BF16 element
        size_B = K * N * 2
        size_C = M * N * 2
        size_instr = len(instr_v) * 4  # 4 bytes per uint32

        # Following working test pattern from mlir-aie
        bo_instr = xrt.bo(device, size_instr, xrt.bo.cacheable, kernel.group_id(1))
        bo_A = xrt.bo(device, size_A, xrt.bo.host_only, kernel.group_id(3))
        bo_B = xrt.bo(device, size_B, xrt.bo.host_only, kernel.group_id(4))
        bo_C = xrt.bo(device, size_C, xrt.bo.host_only, kernel.group_id(5))
        print(f"  Created buffers:")
        print(f"    Instruction: {size_instr}B (group_id 1, cacheable)")
        print(f"    A: {size_A}B (group_id 3, host_only)")
        print(f"    B: {size_B}B (group_id 4, host_only)")
        print(f"    C: {size_C}B (group_id 5, host_only)")

        # 7. Write input data
        print("[7/10] Writing input data...")

        # Write instruction buffer
        buf_instr = np.array(instr_v, dtype=np.uint32)
        bo_instr.write(buf_instr, 0)

        # Write data buffers
        bo_A.write(A_bf16, 0)
        bo_B.write(B_bf16, 0)

        # Sync to device
        bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bo_A.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bo_B.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        print("  Data written and synced to device")

        # 8. Execute kernel (CRITICAL: pass opcode and instruction buffer!)
        print("[8/10] Executing kernel...")
        opcode = 3  # Standard opcode from working tests
        start = time.perf_counter()

        # Kernel call with proper argument order:
        # kernel(opcode, bo_instr, instr_len, bo_A, bo_B, bo_C)
        run = kernel(opcode, bo_instr, len(instr_v), bo_A, bo_B, bo_C)
        state = run.wait()

        exec_time = time.perf_counter() - start

        # Check completion state
        if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            print(f"[WARNING] Kernel state: {state} (expected COMPLETED)")

        print(f"  Kernel executed in {exec_time*1000:.2f}ms (state: {state})")

        # Calculate GFLOPS
        ops = 2 * M * N * K  # FMA operations
        gflops = ops / exec_time / 1e9
        print(f"  Performance: {gflops:.1f} GFLOPS")

        # 9. Read result
        print("[9/10] Reading result...")
        bo_C.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        C_bf16_bytes = bytes(bo_C.map())[:size_C]
        C_result = bytes_to_bf16(C_bf16_bytes).reshape(M, N)
        print(f"  Result: {M}x{N} (range: {C_result.min():.3f} to {C_result.max():.3f})")

        # 10. Validate
        print("[10/10] Validating results...")

        # Check if we got non-zero results
        non_zero_count = np.count_nonzero(C_result)
        print(f"  Non-zero elements: {non_zero_count}/{M*N} ({non_zero_count/(M*N)*100:.1f}%)")

        if non_zero_count == 0:
            print("\n" + "="*70)
            print("  CRITICAL: Still getting all zeros!")
            print("  Possible issues:")
            print("    - Instruction format may be wrong")
            print("    - Kernel may expect different buffer layout")
            print("    - Opcode may be incorrect")
            print("="*70)
            return 1

        # Calculate errors
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
            print("\n  🎉 BREAKTHROUGH - NPU KERNEL RETURNS NON-ZERO VALUES!")
            print("\n  Root cause was: MISSING INSTRUCTION BUFFER")
            print("  Solution:")
            print("    1. Load insts.txt binary file")
            print("    2. Create cacheable instruction buffer (group_id 1)")
            print("    3. Pass opcode + instruction buffer to kernel")
            print("    4. Use proper argument order")
            print("\n  Next steps:")
            print("    - Update all NPU test scripts with instruction loading")
            print("    - Test with larger matrices (512x512)")
            print("    - Integrate with Whisper encoder")
            print("    - Performance optimization")
        else:
            print("\n  Issues detected:")
            if mean_rel > 50:
                print("    - Large error suggests data format mismatch")
            else:
                print(f"    - Error ({mean_rel:.1f}%) exceeds tolerance ({tolerance}%)")
                print("    - May need buffer layout adjustment")

        print("="*70)

        return 0 if success else 1

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
