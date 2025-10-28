#!/usr/bin/env python3
"""
Test if main() function executes in AIE kernel
Expected output: [100, 101, 102, ..., 179] if main() runs
              Or: [0, 0, 0, ..., 0] if it doesn't run
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np

print("=" * 70)
print("Testing AIE Kernel main() Execution")
print("=" * 70)

# Load XCLBIN
print("\n[1/5] Loading XCLBIN...")
device = xrt.device(0)
xclbin_obj = xrt.xclbin('build/mel_int8_final.xclbin')
uuid = xclbin_obj.get_uuid()
device.register_xclbin(xclbin_obj)
print("✅ XCLBIN loaded")

# Create hardware context
print("\n[2/5] Creating hardware context...")
hw_ctx = xrt.hw_context(device, uuid)
print("✅ Hardware context created")

# Get kernel
print("\n[3/5] Getting kernel...")
kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
print(f"✅ Kernel: {kernel}")

# Create buffers
print("\n[4/5] Creating buffers...")
# Input: 200 x 32-bit words = 800 bytes
input_bo = xrt.bo(device, 800, xrt.bo.flags.host_only, kernel.group_id(3))
# Output: 20 x 32-bit words = 80 bytes
output_bo = xrt.bo(device, 80, xrt.bo.flags.host_only, kernel.group_id(4))

# Fill input with test data (not important for this test)
input_data = np.zeros(200, dtype=np.int32)
input_bo.write(input_data, 0)
input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 800, 0)
print("✅ Buffers created and synced")

# Execute kernel
print("\n[5/5] Executing kernel...")
# Create instruction buffer (empty for now)
instr_bo = xrt.bo(device, 4096, xrt.bo.flags.cacheable, kernel.group_id(1))
instr_data = np.zeros(1024, dtype=np.uint32)
instr_bo.write(instr_data, 0)
instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 4096, 0)

# Run kernel
run = kernel(0, instr_bo, 0, input_bo, output_bo)
run.wait()
print("✅ Kernel execution complete")

# Read output
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 80, 0)
output_data = np.frombuffer(output_bo.read(80, 0), dtype=np.int8)

print("\nOutput data (first 20 bytes):")
print(output_data[:20])

# Check if main() executed
if np.all(output_data == 0):
    print("\n❌ RESULT: All zeros - main() DID NOT execute")
    print("   The kernel loaded but the C code never ran.")
elif np.all(output_data == np.arange(100, 180, dtype=np.int8)):
    print("\n✅ RESULT: Pattern matches [100-179] - main() EXECUTED!")
    print("   SUCCESS! The C main() function ran on the NPU!")
else:
    print(f"\n⚠️  RESULT: Unexpected pattern")
    print(f"   Expected: [100, 101, 102, ..., 179]")
    print(f"   Got:      {output_data[:10]} ...")
    # Check if it's partially correct
    if output_data[0] == 100:
        print("   ✅ First byte correct - main() is executing!")

print("=" * 70)
