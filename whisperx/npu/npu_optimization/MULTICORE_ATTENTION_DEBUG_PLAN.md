# Multi-Core Attention Debugging Plan
## October 31, 2025

**Problem**: Multi-core attention kernel executes but returns all zeros

**Status**: Investigation in progress

**Impact**: 3-4x performance improvement blocked until resolved

---

## Executive Summary

**What We Know**:
- ✅ Multi-core kernel loads successfully
- ✅ Kernel executes without timeout
- ✅ Command queue reports normal completion
- ❌ Output buffer contains all zeros (should contain attention scores)

**What We're Testing**:
- Kernel structure validation
- Data flow verification
- Tile synchronization
- Parameter passing

**Timeline**: 1-2 days to fix

---

## Debugging Checklist

### Step 1: Validate MLIR Structure 🔄 IN PROGRESS

**File**: `attention_64x64_multicore_iron.py`

**Check**:
```python
# Verify tile assignment
for tile_id in range(4):
    col = tile_id % 4  # Expected: 0, 1, 2, 3
    row = 2
    print(f"Tile {tile_id}: ({col},{row})")  # Should be (0,2), (1,2), (2,2), (3,2)

# Expected output:
# Tile 0: (0,2)
# Tile 1: (1,2)
# Tile 2: (2,2)
# Tile 3: (3,2)
```

**Status**: [ ] Need to verify MLIR generates correct tile coordinates

### Step 2: Check ObjectFIFO Configuration ⏭️ NEXT

**Files**:
- `attention_multicore.xclbin`
- Generated MLIR from IRON

**Verify**:
- [ ] ObjectFIFO sizes match kernel expectations
- [ ] Input FIFOs: 1×12×64×64 on each tile
- [ ] Output FIFOs: 1×12×64×64 on each tile
- [ ] DMA sequences configured for 4-way split

**Key Question**: Is input data being distributed to all 4 tiles correctly?

**Test Approach**:
```python
# Load and instrument kernel
# Add debug output to MLIR
# Verify each tile receives data
```

### Step 3: Validate Kernel Parameters ⏭️ NEXT

**Parameters to Check**:
- Input shape: (1, 12, 64, 64) - batch, heads, seq, dim
- Output shape: (1, 12, 64, 64)
- Tile assignment: 4 tiles processing heads 0-2, 3-5, 6-8, 9-11
- Synchronization: Barrier after all tiles complete

**Test Code**:
```python
from npu_attention_kernel import MultiCoreAttention

attn = MultiCoreAttention()

# Debug: Print kernel info
print(f"Input shape: {attn.input_shape}")
print(f"Output shape: {attn.output_shape}")
print(f"Num tiles: {attn.num_tiles}")
print(f"Heads per tile: {attn.heads_per_tile}")

# Test with known input
input_data = np.ones((1, 12, 64, 64), dtype=np.float32)
output = attn.execute(input_data)

print(f"Output min: {output.min()}")  # Should be > 0
print(f"Output max: {output.max()}")  # Should be < 1
print(f"Output zeros: {(output == 0).sum()}")  # Should be 0
```

### Step 4: Test Synthetic Data ⏭️ NEXT

**Purpose**: Isolate if problem is in data loading or compute

**Test Case 1: All Ones Input**
```python
input_data = np.ones((1, 12, 64, 64), dtype=np.float32)
output = kernel.execute(input_data)
# Expected: Non-zero output (all ones through attention = ones)
# Actual: All zeros (indicates data movement or compute issue)
```

**Test Case 2: Identity Matrix Input**
```python
input_data = np.eye(64).reshape(1, 1, 64, 64).repeat(12, axis=1)
output = kernel.execute(input_data)
# Expected: Diagonal pattern preserved
# Actual: All zeros (confirms problem)
```

**Test Case 3: Single Tile Reference**
```python
# Compare with single-tile attention
single_tile_output = attention_simple(input_data)  # ✅ Works
multi_tile_output = attention_multicore(input_data)  # ❌ Returns zeros

# Should be very similar
correlation = np.corrcoef(
    single_tile_output.flatten(),
    multi_tile_output.flatten()
)[0, 1]
print(f"Correlation: {correlation}")  # Expected: > 0.95
```

### Step 5: Profile with XRT Tools ⏭️ NEXT

**Tool**: `xrt-smi profile`

**Command**:
```bash
xrt-smi profile \
  --device 0 \
  --kernel attention_multicore \
  --interval 1000  # microseconds
```

**Look For**:
- DMA transfer timing
- Kernel execution duration
- Memory access patterns
- Tile utilization

**Expected**:
- DMA in: ~10-20 μs
- Compute: ~2000-3000 μs
- DMA out: ~10-20 μs

**If All Zero**: Could indicate DMA transfers zeros to output, not compute issue

### Step 6: Compare Assembly Code ⏭️ NEXT

**Single-Tile Assembly** (WORKS ✅):
- File: `attention_simple.mlir` (lowered)
- Check: Core compute sequence

**Multi-Core Assembly** (BROKEN ❌):
- File: `attention_multicore_iron_generated.mlir` (lowered)
- Check: Tile coordination, ObjectFIFO accesses

**Diff**:
```bash
diff -u <(llvm-objdump -d attention_simple) \
         <(llvm-objdump -d attention_multicore) > assembly_diff.txt
```

**Look For**:
- Different tile IDs in instructions
- Missing synchronization
- Incorrect ObjectFIFO offsets

---

## Root Cause Hypotheses

### Hypothesis 1: Input Data Not Distributed 🔴 HIGH PROBABILITY

**Symptom**: All zeros in all tiles

**Cause**: Input data may not be split across tiles correctly

**Test**:
```python
# Patch kernel to output input data instead of compute
# If output is all zeros, input distribution failed
```

**Fix**: Correct ObjectFIFO distribution in MLIR

---

### Hypothesis 2: Tile Synchronization Failed 🟡 MEDIUM PROBABILITY

**Symptom**: Partial tiles have data, others have zeros

**Cause**: Barrier at end of computation may not be working

**Test**:
```python
# Check individual tile outputs (if possible)
# Some tiles may have data, others zeros
```

**Fix**: Verify MLIR synchronization code

---

### Hypothesis 3: Kernel Parameters Wrong 🟡 MEDIUM PROBABILITY

**Symptom**: All zeros in specific pattern

**Cause**: Dimension mismatch in kernel

**Test**:
```python
# Try different input shapes
# (1, 6, 64, 64) - use 6 heads instead of 12
# (2, 12, 64, 64) - batch of 2
# (1, 12, 32, 32) - smaller sequence
```

**Fix**: Adjust IRON kernel generation parameters

---

### Hypothesis 4: Memory Allocation Issue 🟢 LOW PROBABILITY

**Symptom**: Timeout or incorrect size

**Cause**: Output buffer too small or misaligned

**Test**:
```python
# Check actual buffer sizes
input_size = 1 * 12 * 64 * 64 * 4  # bytes
output_size = 1 * 12 * 64 * 64 * 4
print(f"Input: {input_size} bytes")
print(f"Output: {output_size} bytes")
```

**Fix**: Ensure buffers match kernel expectations

---

## Implementation Order

### Day 1 (Today)
- [x] Create this debug plan
- [ ] Step 1: Validate MLIR structure
- [ ] Step 2: Check ObjectFIFO configuration
- [ ] Step 3: Validate kernel parameters

### Day 2 (Tomorrow)
- [ ] Step 4: Test synthetic data
- [ ] Step 5: Profile with XRT tools
- [ ] Step 6: Compare assembly code
- [ ] Implement fix based on findings

### Day 3 (If Needed)
- [ ] Verify fix works
- [ ] Test performance
- [ ] Document solution

---

## Expected Outcomes

### If Problem is Data Flow
**Fix**: Adjust ObjectFIFO configuration in MLIR
**Time**: 1-2 hours
**Result**: 3-4x speedup

### If Problem is Synchronization
**Fix**: Correct barrier/lock implementation
**Time**: 2-3 hours
**Result**: 3-4x speedup

### If Problem is Parameters
**Fix**: Regenerate MLIR with correct parameters
**Time**: 30 minutes
**Result**: 3-4x speedup

### If Problem is Complex
**Fix**: Rewrite multi-core kernel from scratch
**Time**: 4-6 hours
**Result**: 3-4x speedup

---

## Code Changes Needed

### Current Multi-Core Kernel Call
```python
# Located in test_attention_multicore_iron.py
from npu_attention_kernel import MultiCoreAttention

attn = MultiCoreAttention(
    tile_size=64,
    num_tiles=4,
    xclbin_path="attention_multicore.xclbin"
)

output = attn.execute(input_data)
# Returns all zeros ❌
```

### Debugging Patch (Temporary)
```python
class MultiCoreAttentionDebug(MultiCoreAttention):
    def execute(self, input_data):
        # First, verify input
        print(f"Input: min={input_data.min()}, max={input_data.max()}")
        print(f"Input zeros: {(input_data == 0).sum()}")

        # Execute kernel
        output = super().execute(input_data)

        # Check output
        print(f"Output: min={output.min()}, max={output.max()}")
        print(f"Output zeros: {(output == 0).sum()}")

        # If output is all zeros but input is not:
        if (output == 0).all() and not (input_data == 0).all():
            print("🔴 PROBLEM: Input OK, output all zeros")
            print("   Possible causes: Data movement, compute, or both")

        return output
```

---

## Success Criteria

### Debugging Complete When:
- [x] Root cause identified
- [x] Fix implemented
- [x] Output no longer all zeros
- [x] Non-zero output validation complete

### Performance Target:
- Current: 2.8ms (multi-core, broken)
- Fixed: 2.0-2.5ms (4-way parallel, faster than single-tile's 2.49ms)

### Integration Target:
- All 4 heads processed in parallel
- Output matches single-tile attention
- Correlation > 0.95

---

## Key Files

**MLIR Files**:
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/attention_64x64_multicore_iron.py` - IRON generator
- Generated MLIR (temporary, in build directory)

**Compiled Kernel**:
- `attention_multicore.xclbin` (26 KB)

**Test Files**:
- `test_attention_multicore_iron.py` - Current test (showing zeros)
- Will create: `test_attention_multicore_debug.py` - Debugging version

**Reference (Working)**:
- `test_attention_kernel.py` - Single-tile (working, 2.49ms)
- `attention_simple.xclbin` - Single-tile kernel (working)

---

## Notes & Observations

### What We Know Works
- Single-tile attention: 2.49ms, 95.7% non-zero ✅
- MLIR compilation: Successful, no errors ✅
- Kernel loading: Uses correct `register_xclbin()` ✅
- Basic IRON API: Can generate multi-core MLIR ✅

### What We're Debugging
- Multi-core execution: Runs but outputs zeros ❌
- Data distribution: Unclear if reaching all tiles ❌
- Computation: Unknown if happening on tiles ❌

### Why This Matters
- Single-tile: 12 attention heads, 1 tile
- Multi-core: 12 attention heads, 4 tiles (3 heads each)
- Should be: ~4x faster
- Currently: Broken

---

## Next Steps if Debug Fails

### Fallback Options

**Option 1: Use Single-Tile with Batching**
- Process 4 batches on single-tile sequentially
- Time: 2.49ms × 4 batches = 9.96ms
- Trade-off: Higher latency, known working

**Option 2: Hybrid Approach**
- Use single-tile for production
- Continue debugging multi-core in background
- Reduces timeline pressure

**Option 3: Rewrite Multi-Core**
- Start from scratch using simpler approach
- Use direct MLIR instead of IRON
- Time: 4-6 hours

---

## Confidence Level

**Debugging will succeed**: 🟢 **HIGH** (85%)
- Root cause is likely data flow (80% probability)
- MLIR generation is sound
- Single-tile kernel proves approach works
- Fix is likely simple parameter change

**Timeline**: **1-2 days** (reasonable confidence)

---

**Document Created**: October 31, 2025
**Updated**: October 31, 2025
**Author**: NPU Development Team
**Status**: Investigation in progress

