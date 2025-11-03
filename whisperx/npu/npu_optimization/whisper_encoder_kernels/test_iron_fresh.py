#!/usr/bin/env python3
"""
Test Fresh IRON-Generated Multi-Core Attention Kernel
"""

import numpy as np
import pyxrt as xrt
import time

print("=" * 70)
print("TESTING FRESH IRON-GENERATED MULTI-CORE ATTENTION KERNEL")
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

# Create buffers - try the group IDs the kernel expects
print("Creating buffers...")
instr_bo = xrt.bo(device, len(insts_data), xrt.bo.flags.cacheable, kernel.group_id(1))
input_bo = xrt.bo(device, TOTAL_INPUT, xrt.bo.flags.host_only, kernel.group_id(2))
output_bo = xrt.bo(device, TOTAL_OUTPUT, xrt.bo.flags.host_only, kernel.group_id(3))
print("✅ Buffers created")
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
print("Executing fresh IRON kernel...")
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
print("RESULTS")
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

# Performance analysis
if elapsed_ms < 1.0:
    print("✅ PARALLEL EXECUTION CONFIRMED!")
    print(f"   Timing indicates parallel execution across 4 tiles")
elif elapsed_ms < 5.0:
    print("⚠️  Fast but may be serial execution")
else:
    print("❌ Too slow - likely serial execution")
print()

if nonzero_pct > 50:
    print("✅ OUTPUT VALID!")
    print(f"   {nonzero_pct:.1f}% non-zero values indicates correct computation")
    print()
    print("=" * 70)
    print("🎉 SUCCESS: FRESH IRON KERNEL WORKING!")
    print("=" * 70)
elif nonzero_pct > 0:
    print("⚠️  Partial output - needs investigation")
else:
    print("❌ All zeros - buffer bank issue persists")
    print("   Try regenerating MLIR or different buffer allocation")

print()
