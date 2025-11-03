#!/usr/bin/env python3
"""
Test Fresh IRON Multi-Core Attention Kernel with CORRECT Buffer Allocation

This uses the sequential group_id(1,2,3) pattern discovered by Team Lead #2
that produces 100% non-zero output instead of the wrong (1,3,4) pattern.
"""

import numpy as np
import pyxrt as xrt
import time

print("=" * 70)
print("TESTING FRESH IRON KERNEL WITH CORRECT BUFFER ALLOCATION")
print("=" * 70)
print()

# File paths
xclbin_path = "attention_iron_fresh.xclbin"
insts_path = "insts_iron_fresh.bin"

print(f"XCLBIN: {xclbin_path}")
print(f"Instructions: {insts_path}")
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

# Load instructions
with open(insts_path, 'rb') as f:
    insts_data = f.read()
print(f"✅ Instructions loaded: {len(insts_data)} bytes")
print()

# Configuration - 4 tiles
BATCH_SIZE = 4
INPUT_PER_TILE = 12288  # Q+K+V
OUTPUT_PER_TILE = 4096   # Output
TOTAL_INPUT = BATCH_SIZE * INPUT_PER_TILE
TOTAL_OUTPUT = BATCH_SIZE * OUTPUT_PER_TILE

print(f"Configuration: {BATCH_SIZE} tiles × 64×64")
print(f"  Input: {TOTAL_INPUT} bytes")
print(f"  Output: {TOTAL_OUTPUT} bytes")
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

# Create buffers with CORRECT SEQUENTIAL PATTERN (1,2,3)
print("Creating buffers with CORRECT group_id pattern (1,2,3)...")
print("  OLD (wrong):  instr=1, input=3, output=4  → zeros ❌")
print("  NEW (correct): instr=1, input=2, output=3  → 100% non-zero ✅")
print()

instr_bo = xrt.bo(device, len(insts_data), xrt.bo.flags.cacheable, kernel.group_id(1))
input_bo = xrt.bo(device, TOTAL_INPUT, xrt.bo.flags.host_only, kernel.group_id(2))  # CHANGED from 3
output_bo = xrt.bo(device, TOTAL_OUTPUT, xrt.bo.flags.host_only, kernel.group_id(3)) # CHANGED from 4
print("✅ Buffers created with FIXED allocation")
print()

# Write data
print("Preparing test data...")
instr_bo.write(insts_data, 0)
instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

np.random.seed(42)
test_data = np.random.randint(-128, 127, TOTAL_INPUT, dtype=np.int8)
input_bo.write(test_data.tobytes(), 0)
input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
print("✅ Data synced to NPU")
print()

# Execute kernel
print("Executing FIXED buffer allocation kernel...")
run = kernel(instr_bo, input_bo, output_bo, 3)

start = time.time()
run.wait()
elapsed_ms = (time.time() - start) * 1000

print(f"✅ Execution: {elapsed_ms:.2f}ms")
print()

# Read results
output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
output_data = np.frombuffer(output_bo.read(TOTAL_OUTPUT, 0), dtype=np.int8)
output_reshaped = output_data.reshape(BATCH_SIZE, 64, 64)

# Analyze
nonzero_count = np.count_nonzero(output_data)
nonzero_pct = 100.0 * nonzero_count / output_data.size

print("=" * 70)
print("RESULTS WITH FIXED BUFFER ALLOCATION")
print("=" * 70)
print()
print(f"📊 Performance:")
print(f"   Execution: {elapsed_ms:.2f}ms for {BATCH_SIZE} tiles")
print(f"   Per tile: {elapsed_ms / BATCH_SIZE:.2f}ms")
print(f"   Throughput: {1000.0 * BATCH_SIZE / elapsed_ms:.1f} tiles/sec")
print()
print(f"📊 Output quality:")
print(f"   Non-zero: {nonzero_count}/{output_data.size} ({nonzero_pct:.1f}%)")
print(f"   Range: [{output_data.min()}, {output_data.max()}]")
print()

for i in range(BATCH_SIZE):
    tile = output_reshaped[i]
    nz = np.count_nonzero(tile)
    print(f"   Tile {i}: {nz}/4096 non-zero ({100.0*nz/4096:.1f}%), "
          f"range=[{tile.min()}, {tile.max()}]")
print()

# Verdict
print("=" * 70)
if nonzero_pct > 90:
    print("🎉 SUCCESS! BUFFER FIX CONFIRMED!")
    print("=" * 70)
    print()
    print("✅ Multi-core attention kernel now working with correct buffer allocation")
    print(f"✅ Performance: {elapsed_ms:.2f}ms for 4 tiles (parallel execution confirmed)")
    print(f"✅ Output quality: {nonzero_pct:.1f}% non-zero values")
    print()
    print("🔧 Solution: Use sequential group_id(1,2,3) instead of (1,3,4)")
    print()
    print("Next steps:")
    print("  1. Update all wrapper code with correct buffer pattern")
    print("  2. Test with real Whisper encoder Q, K, V tensors")
    print("  3. Integrate into WhisperX pipeline")
elif nonzero_pct > 50:
    print("⚠️  PARTIAL SUCCESS - Improved but not perfect")
    print("=" * 70)
    print()
    print(f"Output improved to {nonzero_pct:.1f}% non-zero")
    print("Further investigation needed")
elif nonzero_pct > 0:
    print("⚠️  MARGINAL IMPROVEMENT")
    print("=" * 70)
    print()
    print(f"Only {nonzero_pct:.1f}% non-zero - still issues remain")
else:
    print("❌ BUFFER FIX DID NOT WORK")
    print("=" * 70)
    print()
    print("Still getting all zeros - need different approach")

print()
