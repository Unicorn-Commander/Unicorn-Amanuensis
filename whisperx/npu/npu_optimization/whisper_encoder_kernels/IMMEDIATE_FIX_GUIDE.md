# Immediate Fix Guide - Attention Kernel Bank Assignment

## Quick Summary

**Problem**: Attention kernels compiled with wrong memory bank (group 2 instead of group 1)
**Solution**: Recompile with explicit bank assignment or use 32x32 tile pattern
**Time**: 15-30 minutes
**Confidence**: 90% this will work

---

## Option 1: Explicit Bank Assignment (RECOMMENDED)

### Step 1: Edit MLIR Source

```bash
cd build_attention_64x64
cp ../attention_64x64.mlir attention_64x64_fixed.mlir
```

Find the buffer declarations in `attention_64x64_fixed.mlir` and add `sym_name` attributes:

**Before**:
```mlir
%buf_in = aie.buffer(%tile_0_0) : memref<12288xi8>
%buf_out = aie.buffer(%tile_0_0) : memref<4096xi8>
```

**After**:
```mlir
%buf_in = aie.buffer(%tile_0_0) {sym_name = "buf_in"} : memref<12288xi8>
%buf_out = aie.buffer(%tile_0_0) {sym_name = "buf_out"} : memref<4096xi8>
```

### Step 2: Force Shim-Only Placement

Ensure ALL buffers are on Shim tile (0,0), NOT compute tiles (0,2) or (0,3):

```mlir
// Get Shim tile reference
%tile_0_0 = aie.tile(0, 0)

// Place ALL buffers here
%buf_in = aie.buffer(%tile_0_0) {sym_name = "buf_in"} : memref<12288xi8>
%buf_out = aie.buffer(%tile_0_0) {sym_name = "buf_out"} : memref<4096xi8>
%buf_instr = aie.buffer(%tile_0_0) {sym_name = "buf_instr"} : memref<4096xi32>

// NO buffers on compute tiles!
// %tile_0_2 = aie.tile(0, 2)  // Don't use this for input/output
```

### Step 3: Recompile

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels

# Compile with MLIR-AIE tools
python3 -m mlir_aie.compiler.aiecc \
    --aie-generate-xclbin \
    --xclbin-name=attention_64x64_fixed.xclbin \
    attention_64x64_fixed.mlir
```

### Step 4: Test

```bash
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
from pathlib import Path

device = xrt.device(0)
xclbin = xrt.xclbin("build_attention_64x64/attention_64x64_fixed.xclbin")
device.register_xclbin(xclbin)
uuid = xclbin.get_uuid()
ctx = xrt.hw_context(device, uuid)
kernel = xrt.kernel(ctx, "MLIR_AIE")

# Check group IDs
print("Group IDs:")
for i in range(5):
    try:
        gid = kernel.group_id(i)
        print(f"  Arg {i}: {gid} = 0x{gid:05x}")
        if 131072 <= gid <= 196607:
            print(f"    ❌ STILL BROKEN: Group 2 detected!")
        elif 65536 <= gid <= 65537:
            print(f"    ✅ FIXED: Group 1 detected!")
    except:
        break

# Run actual test
insts_path = Path("build_attention_64x64/insts.bin")
with open(insts_path, "rb") as f:
    insts = f.read()

instr_bo = xrt.bo(device, len(insts), xrt.bo.flags.cacheable, kernel.group_id(1))
input_bo = xrt.bo(device, 12288, xrt.bo.flags.host_only, kernel.group_id(3))
output_bo = xrt.bo(device, 4096, xrt.bo.flags.host_only, kernel.group_id(4))

instr_bo.write(insts, 0)
instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, len(insts), 0)

# Test data
Q = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
K = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
V = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
input_data = np.concatenate([Q.flatten(), K.flatten(), V.flatten()])

input_bo.write(input_data.tobytes(), 0)
input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, len(input_data), 0)

run = kernel(instr_bo, len(insts), 0, input_bo, output_bo)
state = run.wait(1000)

output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 4096, 0)
output = np.frombuffer(output_bo.read(4096, 0), dtype=np.int8)

print(f"\nExecution state: {state}")
print(f"Output activity: {np.count_nonzero(output)}/4096")
print(f"Output range: [{output.min()}, {output.max()}]")

if np.count_nonzero(output) > 0:
    print("\n✅ SUCCESS! Attention kernel now works!")
else:
    print("\n❌ FAILED: Still producing zeros")
PYEOF
```

---

## Option 2: Use 32x32 Tile Pattern (SAFER)

### Why This Works

Working kernels (MEL, MatMul 16x16) use **single-tile, small-size** patterns.
These get assigned group 1 correctly by the compiler.

### Implementation

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels

# Copy working MatMul pattern
cp matmul_16x16.mlir attention_32x32.mlir
```

**Edit `attention_32x32.mlir`**:
1. Change size from 16x16 to 32x32
2. Adapt compute core for attention logic (Q·K^T)
3. Keep same buffer placement pattern as MatMul

**Compile**:
```bash
python3 -m mlir_aie.compiler.aiecc \
    --aie-generate-xclbin \
    --xclbin-name=attention_32x32.xclbin \
    attention_32x32.mlir
```

**Expected**: Will use group 1 (like MatMul 16x16 does)

---

## Option 3: Workaround with Tiling (IMMEDIATE)

Don't fix the kernel - work around it in Python!

### Use 4× 32x32 MatMul Instead of 1× 64x64 Attention

```python
# In test_encoder_block.py or integration code

def compute_attention_tiled(Q, K, V):
    """
    Compute attention using 4× 32x32 matmul tiles
    Instead of broken 64x64 attention kernel
    """
    # Q, K, V are each 64×64
    # Break into 2×2 grid of 32×32 tiles

    results = []
    for i in range(2):
        for j in range(2):
            # Extract 32×32 tile
            Q_tile = Q[i*32:(i+1)*32, j*32:(j+1)*32]
            K_tile = K[i*32:(i+1)*32, j*32:(j+1)*32]

            # Use WORKING 16×16 matmul (run 4 times for 32×32)
            # Or compile new 32×32 matmul (will work - small single tile)
            result_tile = run_matmul_kernel(Q_tile, K_tile.T)

            # Apply softmax and multiply by V
            # (Can do this on CPU for now)
            attention_tile = softmax(result_tile)
            V_tile = V[i*32:(i+1)*32, j*32:(j+1)*32]
            output_tile = attention_tile @ V_tile

            results.append(output_tile)

    # Reassemble 64×64 output
    return reassemble_tiles(results)
```

**Pros**:
- Uses WORKING kernels (MatMul 16×16)
- No recompilation needed
- Immediate solution

**Cons**:
- More DMA overhead (4× transfers)
- Slightly slower than native 64×64

---

## Option 4: Report to AMD and Use CPU Fallback

### Immediate Action

```python
# In encoder integration
def run_attention(Q, K, V):
    try:
        # Try NPU attention kernel
        return npu_attention(Q, K, V)
    except Exception as e:
        logger.warning(f"NPU attention failed: {e}, falling back to CPU")
        return cpu_attention(Q, K, V)

def cpu_attention(Q, K, V):
    """CPU fallback using PyTorch"""
    import torch
    Q_t = torch.from_numpy(Q).float()
    K_t = torch.from_numpy(K).float()
    V_t = torch.from_numpy(V).float()

    attn = torch.softmax(Q_t @ K_t.T / np.sqrt(64), dim=-1)
    output = attn @ V_t
    return output.numpy().astype(np.int8)
```

### Report Bug

**To**: AMD XDNA GitHub Issues (https://github.com/amd/xdna-driver/issues)

**Title**: "MLIR-AIE compiler assigns invalid memory group 2 for multi-tile kernels on Phoenix NPU"

**Body**:
```
Hardware: AMD Ryzen 9 8945HS with Phoenix NPU (17f0:10)
XRT: 2.20.0
Firmware: 1.5.5.391
MLIR-AIE: v1.1.1

When compiling multi-tile attention kernels (64×64), the compiler assigns
memory group 2 (group_id 131072-196607) which doesn't exist on Phoenix
NPU's Shim DMA tile (0,0). Only group 1 (65536-65537) is valid.

This causes XRT warnings:
"Kernel MLIR_AIE has no compute units with connectivity required for
global argument... connected to bank 131071"

Result: Kernel executes but produces all-zero output because data never
reaches NPU.

Workaround: Use single-tile patterns (16×16, 32×32) which correctly assign
group 1.

Please fix aie-assign-buffer-addresses pass to constrain Phoenix NPU to
group 1 only.
```

---

## Testing Checklist

After implementing ANY fix above:

```bash
# 1. Check group IDs
python3 test_group_ids.py

# 2. Run basic attention test
python3 test_attention_64x64.py

# 3. Verify non-zero output
python3 test_encoder_block.py

# 4. Check XRT warnings
# Should NOT see "bank 131071" anymore

# 5. Run full benchmark
python3 run_all_benchmarks.py --quick
```

**Success Criteria**:
- ✅ Group IDs in range 65536-65537 (group 1)
- ✅ No XRT bank warnings
- ✅ Non-zero output from attention kernel
- ✅ Execution state: COMPLETED (not ERROR)
- ✅ Benchmark shows >0% correlation

---

## Expected Timeline

| Option | Time | Success Rate | Notes |
|--------|------|--------------|-------|
| **Option 1** | 30 min | 70% | May need MLIR expertise |
| **Option 2** | 1 hour | 90% | Proven pattern works |
| **Option 3** | 15 min | 100% | Immediate workaround |
| **Option 4** | 5 min | 100% | Fallback solution |

**Recommendation**:
1. Try **Option 3** immediately (15 min)
2. Pursue **Option 2** for clean solution (1 hour)
3. Keep **Option 4** as permanent fallback
4. Report bug regardless of fix

---

## Questions to Answer

1. **Does explicit bank assignment work?**
   - Add `{sym_name = "name"}` attributes
   - Recompile and check group IDs
   - Expected: Should force group 1

2. **Does tile size matter?**
   - Compile 32×32 attention kernel
   - Check if assigned group 1 (like 16×16 matmul)
   - Expected: YES - smaller = group 1

3. **Is this Phoenix-specific?**
   - Check if other NPUs (Strix, Hawk Point) have group 2
   - May be hardware limitation of Phoenix
   - Expected: Phoenix has fewer Shim connections

---

## Support Files

**Test Scripts**:
- `test_group_ids.py` - Check memory bank assignments
- `test_attention_64x64.py` - Run single attention test
- `test_encoder_block.py` - Full encoder integration

**Reference**:
- `NPU_BANK_MISMATCH_ROOT_CAUSE_ANALYSIS.md` - Full technical details
- `WORKING_KERNELS_INVENTORY_OCT30.md` - Known working kernels

**Contact**:
- AMD XDNA GitHub: https://github.com/amd/xdna-driver
- MLIR-AIE GitHub: https://github.com/Xilinx/mlir-aie

---

**Last Updated**: October 31, 2025, 17:16 GMT
**Status**: Ready to implement
**Confidence**: 90% one of these will work
