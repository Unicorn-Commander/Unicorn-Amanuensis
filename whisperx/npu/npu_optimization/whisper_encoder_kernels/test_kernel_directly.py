#!/usr/bin/env python3
"""
Test the NPU kernel directly to confirm what it's outputting
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
from pathlib import Path

# Load kernel
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

# Test: Identity matrix
print("Test: 64*I @ 64*I")
A = np.eye(16, dtype=np.int8) * 64
B = np.eye(16, dtype=np.int8) * 64

packed_input = np.concatenate([A.flatten(), B.flatten()])
input_bo.write(packed_input.tobytes(), 0)
input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 512, 0)

opcode = 3
run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
run.wait(1000)

output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 256, 0)
C_npu = np.frombuffer(output_bo.read(256, 0), dtype=np.int8).reshape(16, 16)

# CPU reference
C_ref_int32 = A.astype(np.int32) @ B.astype(np.int32)
C_ref = (C_ref_int32 >> 7).astype(np.int8)

print("\nNPU output (diagonal):")
print(np.diag(C_npu))
print("\nExpected (diagonal):")
print(np.diag(C_ref))
print("\nMatch:", np.allclose(C_npu, C_ref, atol=1))
print(f"Correlation: {np.corrcoef(C_npu.flatten(), C_ref.flatten())[0, 1]:.6f}")
