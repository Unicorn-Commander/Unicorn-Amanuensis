# NPU Kernel Quick Reference

**For CC-1L Project - XDNA2 NPU Development**

---

## Critical Requirement: Load BOTH Files

❌ **WRONG** (Week 15 mistake):
```python
# Only loading xclbin - WILL RETURN ALL ZEROS!
device.register_xclbin(xclbin)
kernel = xrt.kernel(context, "MLIR_AIE")
run = kernel(bo_A, bo_B, bo_C)  # Executes but does nothing
```

✅ **CORRECT** (Week 16 solution):
```python
from aie.utils.xrt import setup_aie

# Loads BOTH xclbin AND instructions
app = setup_aie(
    xclbin_path="path/to/kernel.xclbin",
    insts_path="path/to/insts.bin",  # REQUIRED!
    in_0_shape=(size,),
    in_0_dtype=np.uint16,
    in_1_shape=(size,),
    in_1_dtype=np.uint16,
    out_buf_shape=(size,),
    out_buf_dtype=np.uint16,
    kernel_name="MLIR_AIE"
)
```

---

## Complete Example: Matrix Multiplication

```python
#!/usr/bin/env python3
import sys
import numpy as np
from pathlib import Path

# Add mlir-aie utilities
sys.path.append(str(Path.home() / "mlir-aie" / "python"))
from aie.utils.xrt import setup_aie

# File paths
kernel_dir = Path("path/to/kernel/build")
xclbin_path = kernel_dir / "kernel.xclbin"
instr_path = kernel_dir / "insts.bin"  # .bin or .txt

# Matrix size
M, N, K = 512, 512, 512

# Test data (BF16 as uint16)
A_fp32 = np.random.rand(M, K).astype(np.float32)
B_fp32 = np.random.rand(K, N).astype(np.float32)

# Convert to BF16
def fp32_to_bf16(fp32):
    return (fp32.view(np.uint32) >> 16).astype(np.uint16)

A_bf16 = fp32_to_bf16(A_fp32.flatten())
B_bf16 = fp32_to_bf16(B_fp32.flatten())

# Setup NPU (loads xclbin + instructions)
app = setup_aie(
    xclbin_path=str(xclbin_path),
    insts_path=str(instr_path),
    in_0_shape=(M*K,),
    in_0_dtype=np.uint16,
    in_1_shape=(K*N,),
    in_1_dtype=np.uint16,
    out_buf_shape=(M*N,),
    out_buf_dtype=np.uint16,
    kernel_name="MLIR_AIE"
)

# Write inputs
app.buffers[3].write(A_bf16)
app.buffers[4].write(B_bf16)

# Execute
app.run()  # Handles sync, kernel call, wait

# Read output
C_bf16 = app.buffers[5].read()

# Convert back to FP32
def bf16_to_fp32(bf16):
    return (bf16.astype(np.uint32) << 16).view(np.float32)

C_result = bf16_to_fp32(C_bf16).reshape(M, N)

print(f"Result: {C_result[:5, :5]}")
```

---

## File Naming Convention

mlir-aie compiler generates TWO files:

1. **Kernel binary** (`.xclbin`)
   - Contains AIE tile compute kernels
   - Loaded via `device.register_xclbin()`

2. **Instruction sequence** (`.bin` or `.txt`)
   - Contains DMA configuration (runtime_sequence)
   - Loaded via `setup_aie(..., insts_path=...)`

Example:
```
build/
├── matmul_1tile_bf16.xclbin       # Kernel
├── insts_1tile_bf16.bin           # Instructions (binary)
└── insts_1tile_bf16.txt           # Instructions (hex text)
```

Use either `.bin` or `.txt` - both work.

---

## Buffer Group IDs

Standard mlir-aie convention:

| Group ID | Purpose | Buffer Type |
|----------|---------|-------------|
| 0 | Reserved/Opcode | - |
| 1 | **Instructions** | `cacheable` |
| 2 | Reserved | - |
| 3 | Input A | `host_only` |
| 4 | Input B | `host_only` |
| 5 | Output C | `host_only` |
| 6+ | Additional (trace, etc.) | varies |

Access via:
```python
app.buffers[3]  # Input A
app.buffers[4]  # Input B
app.buffers[5]  # Output C
```

---

## Common Mistakes

### 1. Missing Instruction File
**Symptom**: Kernel executes quickly (~1ms) but returns all zeros

**Cause**: Not loading `insts.bin`

**Fix**: Use `setup_aie()` with `insts_path` parameter

### 2. Wrong Instruction Format
**Symptom**: `UnicodeDecodeError` when loading instructions

**Cause**: Trying to load `.txt` as text when it's actually binary

**Fix**: Use `.bin` file or check file format:
```python
from aie.utils.xrt import read_insts
insts = read_insts(path)  # Auto-detects .bin or .txt
```

### 3. Manual XRT API Usage
**Symptom**: Complex code, hard to debug, instruction loading not obvious

**Fix**: Use mlir-aie utilities instead:
- ✅ `setup_aie()` - Handles everything
- ✅ `execute()` - Simplified execution
- ✅ `AIE_Buffer` - Automatic sync

### 4. Wrong Buffer Sizes
**Symptom**: Errors about buffer size mismatch

**Fix**: Check MLIR runtime_sequence signature:
```mlir
aiex.runtime_sequence(%arg0: memref<262144xbf16>, ...)
                                     ^^^^^^
                                     512*512 elements
```

---

## BF16 Data Format

### Conversion Functions

```python
import numpy as np

def float32_to_bf16(fp32_array):
    """Convert FP32 to BF16 (stored as uint16)"""
    uint32 = fp32_array.view(np.uint32)
    return (uint32 >> 16).astype(np.uint16)

def bf16_to_float32(bf16_uint16):
    """Convert BF16 (as uint16) back to FP32"""
    uint32 = bf16_uint16.astype(np.uint32) << 16
    return uint32.view(np.float32)
```

### Why uint16?
- BF16 is 16-bit floating point
- Python/NumPy don't have native BF16 type
- Store as uint16, convert for computation
- NPU hardware handles BF16 natively

---

## Validation Checklist

After running kernel, verify:

- [ ] Non-zero output: `np.count_nonzero(result) > 0`
- [ ] Reasonable range: `result.min()` and `result.max()` make sense
- [ ] Accuracy: Compare with reference implementation
- [ ] Performance: Check execution time and GFLOPS

### Expected Accuracy for BF16
- Mean error: 3-5%
- Max error: 10-25%
- Anything worse suggests a problem

---

## Troubleshooting

### All Zeros Output
1. Check instruction file is loaded: `insts_path` parameter
2. Verify instruction file exists and is correct format
3. Check buffer sizes match MLIR runtime_sequence
4. Validate data is written before kernel execution

### Segmentation Fault
1. Check buffer group IDs match kernel expectations
2. Verify buffer sizes are correct
3. Ensure proper sync_to_device before kernel call

### Low Accuracy
1. Verify BF16 conversion is correct
2. Check for signed value bug (use BF16 workaround if needed)
3. Compare with CPU reference at same precision

### Slow Performance
1. Check matrix size matches kernel design
2. Verify using correct multi-tile kernel for large matrices
3. Profile DMA transfer time vs compute time

---

## Performance Expectations

### 512×512 Matrix Multiply (1-tile kernel)
- **Execution time**: 3-4ms
- **Performance**: 60-80 GFLOPS
- **Accuracy**: 95-97% (3-5% error)

### Scaling
- 2-tile kernel: ~2x performance
- 4-tile kernel: ~3-4x performance
- 32-tile kernel: ~20-25x performance

---

## Key References

### Documentation
- MLIR-AIE: `/home/ccadmin/mlir-aie/`
- Week 16 Report: `WEEK16_BREAKTHROUGH_REPORT.md`
- BF16 Bug: `/home/ccadmin/CC-1L/docs/bf16/README.md`

### Code Examples
- Working test: `WEEK16_NPU_SOLUTION.py`
- mlir-aie utils: `~/mlir-aie/python/utils/xrt.py`
- Test examples: `~/mlir-aie/programming_examples/`

### Helper Scripts
- `setup_aie()`: Loads xclbin + instructions
- `execute()`: Simplified kernel execution
- `read_insts()`: Load .bin or .txt instructions

---

## Template for New Kernels

```python
#!/usr/bin/env python3
"""My NPU Kernel Test"""
import sys
import numpy as np
from pathlib import Path

# Add mlir-aie utilities
sys.path.append(str(Path.home() / "mlir-aie" / "python"))
from aie.utils.xrt import setup_aie

def main():
    # File paths
    kernel_dir = Path("build")
    xclbin_path = kernel_dir / "kernel.xclbin"
    instr_path = kernel_dir / "insts.bin"

    # Setup NPU
    app = setup_aie(
        xclbin_path=str(xclbin_path),
        insts_path=str(instr_path),    # DON'T FORGET!
        in_0_shape=(size,),
        in_0_dtype=np.uint16,
        # ... configure buffers ...
        kernel_name="MLIR_AIE",
        verbosity=1
    )

    # Write inputs
    app.buffers[3].write(input_a)
    app.buffers[4].write(input_b)

    # Execute
    app.run()

    # Read output
    output = app.buffers[5].read()

    # Validate
    assert np.count_nonzero(output) > 0, "All zeros!"

    print("✓ Success")

if __name__ == "__main__":
    main()
```

---

**Last Updated**: November 2, 2025
**Status**: Verified working on XDNA2 NPU
**Next Review**: After Week 17 integration testing
