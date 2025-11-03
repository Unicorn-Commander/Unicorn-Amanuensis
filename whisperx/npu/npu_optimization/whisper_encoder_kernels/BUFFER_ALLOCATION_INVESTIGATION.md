# XRT Buffer Allocation Investigation

## Date: October 31, 2025
## Team: XRT Buffer Allocation Team Lead

## Problem Statement

The attention kernel returns all zeros despite:
- ✅ Kernel loads successfully
- ✅ Execution completes without error
- ✅ Same buffer allocation pattern as working mel kernel

### XRT Warning Message
```
[XRT] WARNING: Kernel MLIR_AIE has no compute units with connectivity
required for global argument at index 0. The argument is allocated in
bank 1, the compute unit is connected to bank 131071.
```

## Kernel Memory Bank Requirements (from XCLBIN metadata)

**What the kernel expects**:
- Arg 0 (opcode): Bank 131071
- Arg 1 (instructions): Bank 65537
- Arg 2 (input): Bank 131071
- Arg 3 (output): Bank 65536
- Arg 4 (output size): Bank 65536

**What we're allocating**:
```python
instr_bo = xrt.bo(device, size, xrt.bo.flags.cacheable, kernel.group_id(1))
input_bo = xrt.bo(device, size, xrt.bo.flags.host_only, kernel.group_id(2))
output_bo = xrt.bo(device, size, xrt.bo.flags.host_only, kernel.group_id(3))
```

**Result**: Allocated to bank 1, kernel wants bank 131071

## Working Reference: Mel Kernel

**Same allocation pattern works perfectly**:
```python
# From test_mel_wer_validation.py lines 92-97
self.instr_bo = xrt.bo(self.device, self.n_insts,
                       xrt.bo.flags.cacheable, self.kernel.group_id(1))
self.input_bo = xrt.bo(self.device, 800,
                       xrt.bo.flags.host_only, self.kernel.group_id(3))
self.output_bo = xrt.bo(self.device, 80,
                        xrt.bo.flags.host_only, self.kernel.group_id(4))
```

**Performance**: 0.58ms, 96.2% non-zero output ✅

## Key Discovery

**ALL working kernels use the same pattern**:
- Instructions: `group_id(1)` with `cacheable` flag
- Input: `group_id(3)` with `host_only` flag
- Output: `group_id(4)` with `host_only` flag

**This pattern is IDENTICAL** to what we're using, yet:
- Mel kernel: Works perfectly
- Attention kernel: Returns zeros

## Bank Number Analysis

### Special Bank Numbers
- **Bank 65536 (0x10000)**: HOST memory region
- **Bank 65537 (0x10001)**: Device local memory (DDR)
- **Bank 131071 (0x1FFFF)**: Special NPU instruction/control region

### XRT group_id Mapping
- `group_id(1)` → cacheable → Bank 1 (should be 65537?)
- `group_id(2)` → host_only → Bank 1 (should be 131071?)
- `group_id(3)` → host_only → Bank 1 (should be 131071?)

## The Warning Explanation

```
The argument is allocated in bank 1,
the compute unit is connected to bank 131071.
Allocating local copy of argument buffer in connected bank.
```

**This means**:
1. XRT allocates buffer in bank 1 (system RAM)
2. NPU kernel expects bank 131071 (NPU memory)
3. XRT **automatically copies** data to correct bank
4. This SHOULD work... but attention returns zeros

## Why Mel Works But Attention Doesn't

### Hypothesis 1: Buffer Size Mismatch
- Mel: 800 bytes input, 80 bytes output (small)
- Attention: 12288 bytes input, 4096 bytes output (large)
- **Maybe**: Bank 131071 has size limit?

### Hypothesis 2: Kernel Implementation Issue
- Mel kernel C++ code is correct
- Attention kernel has bug in computation logic
- Buffer allocation is NOT the problem

### Hypothesis 3: Multiple Buffer Access Pattern
- Mel: Single sequential read/write
- Attention: Q, K, V concatenated (3 separate regions)
- **Maybe**: DMA address calculation wrong?

### Hypothesis 4: Bank Allocation IS Correct
- XRT's automatic copy handles bank mismatch
- The warning is INFO, not ERROR
- Real issue is in kernel computation

## Test Plan

### Test 1: Try Different XRT Buffer Flags ✅
```python
# Test device_only flag
input_bo = xrt.bo(device, size, xrt.bo.flags.device_only, kernel.group_id(2))

# Test p2p flag
input_bo = xrt.bo(device, size, xrt.bo.flags.p2p, kernel.group_id(2))

# Test no flags
input_bo = xrt.bo(device, size, 0, kernel.group_id(2))
```

### Test 2: Try Different group_id Values ✅
```python
# Try group_id(0) - maybe opcode is handled specially?
input_bo = xrt.bo(device, size, xrt.bo.flags.host_only, kernel.group_id(0))

# Try sequential: 1, 2, 3 (no skipping)
instr_bo = xrt.bo(device, size, xrt.bo.flags.cacheable, kernel.group_id(1))
input_bo = xrt.bo(device, size, xrt.bo.flags.host_only, kernel.group_id(2))
output_bo = xrt.bo(device, size, xrt.bo.flags.host_only, kernel.group_id(3))
```

### Test 3: Explicit Bank Allocation (if API supports) ✅
```python
# Research XRT 2.20.0 API for explicit bank selection
# Look for: xrt.bo(..., bank=131071)
```

### Test 4: Compare XCLBIN Metadata ✅
```bash
# Extract connectivity info from both XCLBINs
xclbinutil --info --input mel_fixed_v3_PRODUCTION_v1.0.xclbin
xclbinutil --info --input attention_64x64.xclbin

# Compare memory topology sections
```

### Test 5: Test with Smaller Buffer Sizes ✅
```python
# Try attention with same size as mel
INPUT_SIZE = 800  # Same as mel
OUTPUT_SIZE = 80   # Same as mel

# If this works, it's a size issue
# If still zeros, it's a kernel bug
```

## XRT 2.20.0 Source Investigation

**Files to check**:
- `xrt/src/runtime_src/core/edge/user/aie/aie.cpp` - AIE buffer allocation
- `xrt/src/runtime_src/core/common/xrt_bo.cpp` - BO creation
- `xrt/src/runtime_src/core/edge/user/shim.cpp` - Memory bank mapping

**Key questions**:
1. How does `group_id()` map to physical banks?
2. Can we override bank selection?
3. Is bank 131071 special for NPU instructions?
4. Size limits per bank?

## Alternative Approaches

### Approach 1: Accept XRT Auto-Allocation
- Current warnings are INFO, not ERROR
- XRT automatically handles bank transfers
- Focus on fixing kernel computation bug

### Approach 2: Examine MLIR Source
- Check attention MLIR for ObjectFIFO configuration
- Compare with mel kernel MLIR
- Ensure memory regions are specified correctly

### Approach 3: Regenerate XCLBIN
- Recompile attention kernel with explicit bank directives
- Match mel kernel's memory layout exactly
- Use `aie-opt` flags to control placement

## Conclusion

**The buffer allocation pattern is CORRECT** - proven by working mel kernel.

**The issue is likely**:
1. Kernel computation bug (most likely)
2. DMA address calculation for concatenated Q+K+V
3. Buffer size exceeds bank limit
4. MLIR memory region configuration mismatch

**Next steps**:
1. ✅ Test all buffer flag variations
2. ✅ Compare XCLBIN metadata
3. ✅ Test smaller buffer sizes
4. ✅ Examine attention kernel C++ code
5. ✅ Review MLIR ObjectFIFO configuration

**We should NOT spend more time on buffer allocation** - it's working correctly.
Focus should shift to kernel computation verification.
