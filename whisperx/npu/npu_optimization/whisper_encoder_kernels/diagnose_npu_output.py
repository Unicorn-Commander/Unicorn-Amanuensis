#!/usr/bin/env python3
"""
Diagnostic script to understand NPU matmul output format
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
from pathlib import Path

# Find xclbin
base_path = Path(__file__).parent / "build_matmul_fixed"
xclbin_path = base_path / "matmul_16x16.xclbin"
insts_path = base_path / "main_sequence.bin"

print(f"Loading kernel from: {xclbin_path}")

# Initialize device
device = xrt.device(0)
xclbin = xrt.xclbin(str(xclbin_path))
device.register_xclbin(xclbin)
uuid = xclbin.get_uuid()
ctx = xrt.hw_context(device, uuid)
kernel = xrt.kernel(ctx, "MLIR_AIE")

# Load instructions
with open(insts_path, "rb") as f:
    insts = f.read()
n_insts = len(insts)

# Create buffers
instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
input_bo = xrt.bo(device, 512, xrt.bo.flags.host_only, kernel.group_id(3))
output_bo = xrt.bo(device, 256, xrt.bo.flags.host_only, kernel.group_id(4))

# Write instructions
instr_bo.write(insts, 0)
instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

# Test 1: Identity matrix (simple case)
print("\n" + "="*70)
print("TEST 1: Identity Matrix (64 * I) @ (64 * I)")
print("="*70)

A = np.eye(16, dtype=np.int8) * 64
B = np.eye(16, dtype=np.int8) * 64

print("\nInput A (first 4x4):")
print(A[:4, :4])
print("\nInput B (first 4x4):")
print(B[:4, :4])

# Expected output (CPU reference)
C_ref_int32 = A.astype(np.int32) @ B.astype(np.int32)
print("\nExpected output INT32 (first 4x4):")
print(C_ref_int32[:4, :4])
print(f"Range: [{C_ref_int32.min()}, {C_ref_int32.max()}]")

# After >>7 scaling
C_ref_scaled = C_ref_int32 >> 7
print("\nExpected after >>7 scaling (first 4x4):")
print(C_ref_scaled[:4, :4])
print(f"Range: [{C_ref_scaled.min()}, {C_ref_scaled.max()}]")

# Pack and send to NPU
packed_input = np.concatenate([A.flatten(), B.flatten()])
input_bo.write(packed_input.tobytes(), 0)
input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 512, 0)

# Execute
opcode = 3
run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
run.wait(1000)

# Read output
output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 256, 0)
output_bytes = output_bo.read(256, 0)

# Try different interpretations
print("\n" + "="*70)
print("NPU OUTPUT ANALYSIS")
print("="*70)

# Interpretation 1: INT8
C_npu_int8 = np.frombuffer(output_bytes, dtype=np.int8).reshape(16, 16)
print("\n1. As INT8 (current interpretation):")
print(C_npu_int8[:4, :4])
print(f"Range: [{C_npu_int8.min()}, {C_npu_int8.max()}]")
print(f"Non-zero count: {np.count_nonzero(C_npu_int8)} / 256")

# Interpretation 2: UINT8
C_npu_uint8 = np.frombuffer(output_bytes, dtype=np.uint8).reshape(16, 16)
print("\n2. As UINT8:")
print(C_npu_uint8[:4, :4])
print(f"Range: [{C_npu_uint8.min()}, {C_npu_uint8.max()}]")

# Interpretation 3: INT16 (maybe output is 16-bit?)
C_npu_int16 = np.frombuffer(output_bytes, dtype=np.int16).reshape(16, 8)
print("\n3. As INT16 (16x8 matrix):")
print(C_npu_int16[:4, :4])
print(f"Range: [{C_npu_int16.min()}, {C_npu_int16.max()}]")

# Interpretation 4: INT32 (maybe output is 32-bit accumulator?)
C_npu_int32 = np.frombuffer(output_bytes, dtype=np.int32).reshape(8, 8)
print("\n4. As INT32 (8x8 matrix):")
print(C_npu_int32[:4, :4])
print(f"Range: [{C_npu_int32.min()}, {C_npu_int32.max()}]")

# Check if there's a pattern
print("\n" + "="*70)
print("PATTERN ANALYSIS")
print("="*70)

# Look at raw bytes
print("\nFirst 32 bytes (hex):")
print(" ".join(f"{b:02x}" for b in output_bytes[:32]))

print("\nFirst 32 bytes (decimal):")
print(" ".join(f"{b:3d}" for b in output_bytes[:32]))

# Test 2: Simple known values
print("\n" + "="*70)
print("TEST 2: Simple Matrix [1,2,3...] @ [1,1,1...]")
print("="*70)

A = np.arange(1, 257, dtype=np.int8).reshape(16, 16)
B = np.ones((16, 16), dtype=np.int8)

print("\nInput A (first 4x4):")
print(A[:4, :4])
print("\nInput B (first 4x4):")
print(B[:4, :4])

# Expected
C_ref_int32 = A.astype(np.int32) @ B.astype(np.int32)
print("\nExpected INT32 output (first row):")
print(C_ref_int32[0, :])

# Send to NPU
packed_input = np.concatenate([A.flatten(), B.flatten()])
input_bo.write(packed_input.tobytes(), 0)
input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 512, 0)

run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
run.wait(1000)

output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 256, 0)
output_bytes = output_bo.read(256, 0)

C_npu_int8 = np.frombuffer(output_bytes, dtype=np.int8).reshape(16, 16)
print("\nNPU output (INT8, first row):")
print(C_npu_int8[0, :])

C_npu_int32 = np.frombuffer(output_bytes, dtype=np.int32).reshape(8, 8)
print("\nNPU output (INT32, first row):")
print(C_npu_int32[0, :])

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
