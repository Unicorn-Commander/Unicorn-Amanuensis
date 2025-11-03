#!/usr/bin/env python3
"""
Clean test: initialize fresh, run one test, check output
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
from pathlib import Path

def run_test(name, A, B):
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")

    # Fresh initialization for each test
    base_path = Path(__file__).parent / "build_matmul_fixed"
    xclbin_path = base_path / "matmul_16x16.xclbin"
    insts_path = base_path / "main_sequence.bin"

    device = xrt.device(0)
    xclbin = xrt.xclbin(str(xclbin_path))
    device.register_xclbin(xclbin)
    uuid = xclbin.get_uuid()
    ctx = xrt.hw_context(device, uuid)
    kernel = xrt.kernel(ctx, "MLIR_AIE")

    with open(insts_path, "rb") as f:
        insts = f.read()
    n_insts = len(insts)

    instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
    input_bo = xrt.bo(device, 512, xrt.bo.flags.host_only, kernel.group_id(3))
    output_bo = xrt.bo(device, 256, xrt.bo.flags.host_only, kernel.group_id(4))

    instr_bo.write(insts, 0)
    instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

    # Send input
    packed_input = np.concatenate([A.flatten(), B.flatten()])
    input_bo.write(packed_input.tobytes(), 0)
    input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 512, 0)

    # Execute
    opcode = 3
    run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
    run.wait(1000)

    # Read output
    output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 256, 0)
    C_npu = np.frombuffer(output_bo.read(256, 0), dtype=np.int8).reshape(16, 16)

    # CPU reference
    C_ref_int32 = A.astype(np.int32) @ B.astype(np.int32)
    C_ref = (C_ref_int32 >> 7).astype(np.int8)

    print(f"\nInput A (first 4x4):")
    print(A[:4, :4])
    print(f"\nInput B (first 4x4):")
    print(B[:4, :4])
    print(f"\nNPU output (first 4x4):")
    print(C_npu[:4, :4])
    print(f"\nExpected (first 4x4):")
    print(C_ref[:4, :4])

    match = np.allclose(C_npu, C_ref, atol=1)
    correlation = np.corrcoef(C_npu.flatten(), C_ref.flatten())[0, 1]

    print(f"\nMatch (atol=1): {match}")
    print(f"Correlation: {correlation:.6f}")
    print(f"Max error: {np.max(np.abs(C_npu - C_ref))}")

    return match, correlation

# Test 1: Random (should work)
print("="*70)
print("RUNNING CLEAN TESTS")
print("="*70)

np.random.seed(42)
A = np.random.randint(-64, 64, (16, 16), dtype=np.int8)
B = np.random.randint(-64, 64, (16, 16), dtype=np.int8)
match1, corr1 = run_test("Random Matrices", A, B)

# Test 2: Identity (fails?)
A = np.eye(16, dtype=np.int8) * 64
B = np.eye(16, dtype=np.int8) * 64
match2, corr2 = run_test("Identity 64*I @ 64*I", A, B)

# Test 3: Small constant
A = np.ones((16, 16), dtype=np.int8) * 8
B = np.ones((16, 16), dtype=np.int8) * 8
match3, corr3 = run_test("8*ones @ 8*ones", A, B)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Random:   Match={match1}, Correlation={corr1:.4f}")
print(f"Identity: Match={match2}, Correlation={corr2:.4f}")
print(f"Constant: Match={match3}, Correlation={corr3:.4f}")
