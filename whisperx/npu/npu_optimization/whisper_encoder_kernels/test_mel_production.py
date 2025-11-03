#!/usr/bin/env python3
"""
Test Production Mel Kernel to Compare Buffer Allocation

This tests the WORKING mel kernel with the same test infrastructure
to determine if our test code has issues or if it's an attention-specific problem.
"""

import numpy as np
import pyxrt as xrt
import time

print("=" * 70)
print("TESTING PRODUCTION MEL KERNEL (KNOWN WORKING)")
print("=" * 70)
print()

# File paths - use production v1.0 which is documented as working
xclbin_path = "../mel_kernels/build_fixed_v3/mel_fixed_v3_PRODUCTION_v1.0.xclbin"

print(f"XCLBIN: {xclbin_path}")
print()

# Initialize NPU
print("Initializing NPU...")
device = xrt.device(0)
xclbin = xrt.xclbin(xclbin_path)
device.register_xclbin(xclbin)
uuid = xclbin.get_uuid()
hw_ctx = xrt.hw_context(device, uuid)
kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
print("✅ NPU initialized")
print()

# Note: Mel kernel may have instructions embedded in XCLBIN
print("Note: Mel kernel may have embedded instructions (no separate .bin file)")
print()

# Mel kernel configuration (from production code)
# Input: 800 bytes (audio samples)
# Output: 80 bytes (mel features)
INPUT_SIZE = 800
OUTPUT_SIZE = 80

print(f"Configuration: Mel spectrogram")
print(f"  Input: {INPUT_SIZE} bytes (audio)")
print(f"  Output: {OUTPUT_SIZE} bytes (mel features)")
print()

# Check kernel group IDs
print("Kernel group IDs:")
for i in range(6):
    try:
        gid = kernel.group_id(i)
        print(f"  Arg {i}: group_id = {gid}")
    except:
        break
print()

# Create buffers - try the pattern from production mel code
print("Creating buffers with PRODUCTION pattern...")
print("  Pattern from test_mel_wer_validation.py:")
print("    input_bo = kernel.group_id(3)")
print("    output_bo = kernel.group_id(4)")
print()

input_bo = xrt.bo(device, INPUT_SIZE, xrt.bo.flags.host_only, kernel.group_id(3))
output_bo = xrt.bo(device, OUTPUT_SIZE, xrt.bo.flags.host_only, kernel.group_id(4))
print("✅ Buffers created")
print()

# Write data
print("Preparing test data...")

# Random audio data
np.random.seed(42)
test_data = np.random.randint(-128, 127, INPUT_SIZE, dtype=np.int8)
input_bo.write(test_data.tobytes(), 0)
input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
print("✅ Data synced to NPU")
print()

# Execute kernel (mel may not need instruction buffer)
print("Executing production mel kernel...")
try:
    run = kernel(input_bo, output_bo)
except Exception as e:
    print(f"Failed with 2 args: {e}")
    print("Trying with 3 args...")
    run = kernel(input_bo, output_bo, 3)

start = time.time()
run.wait()
elapsed_ms = (time.time() - start) * 1000

print(f"✅ Execution: {elapsed_ms:.2f}ms")
print()

# Read results
output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
output_data = np.frombuffer(output_bo.read(OUTPUT_SIZE, 0), dtype=np.int8)

# Analyze
nonzero_count = np.count_nonzero(output_data)
nonzero_pct = 100.0 * nonzero_count / output_data.size

print("=" * 70)
print("RESULTS - PRODUCTION MEL KERNEL")
print("=" * 70)
print()
print(f"📊 Performance:")
print(f"   Execution: {elapsed_ms:.2f}ms")
print()
print(f"📊 Output quality:")
print(f"   Non-zero: {nonzero_count}/{output_data.size} ({nonzero_pct:.1f}%)")
print(f"   Range: [{output_data.min()}, {output_data.max()}]")
print(f"   Mean: {output_data.mean():.2f}")
print(f"   Std: {output_data.std():.2f}")
print()
print(f"First 20 values: {output_data[:20]}")
print()

# Verdict
print("=" * 70)
if nonzero_pct > 50:
    print("✅ MEL KERNEL STILL WORKS!")
    print("=" * 70)
    print()
    print("This proves:")
    print("  1. Our test infrastructure is correct")
    print("  2. XRT runtime is working")
    print("  3. NPU is functional")
    print("  4. The issue is specific to attention kernels")
    print()
    print("Conclusion: Attention kernels need different compilation")
    print("            or different MLIR structure than mel kernel")
elif nonzero_pct > 0:
    print("⚠️  MEL kernel produces SOME output")
    print("=" * 70)
    print()
    print("Partial output suggests possible issue with test infrastructure")
else:
    print("❌ MEL KERNEL ALSO RETURNS ZEROS!")
    print("=" * 70)
    print()
    print("This proves:")
    print("  1. Test infrastructure issue (likely)")
    print("  2. OR: System state has changed since mel was tested")
    print("  3. OR: Mel kernel has similar undetected issue")
    print()
    print("Need to test mel in production environment where it worked")

print()
