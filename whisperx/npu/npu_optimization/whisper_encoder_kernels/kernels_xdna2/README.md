# XDNA2 (Strix NPU) Kernel Placeholder

**Target Hardware**: AMD Ryzen AI Strix NPU (XDNA2)
**Architecture**: 8 columns × 6 rows = 48 AIE2 tiles
**Performance**: 32 TOPS INT8, 64 TFLOPS BF16
**Status**: 🚧 Future Development

---

## Overview

This directory is reserved for **XDNA2-specific optimized kernels** targeting the next-generation AMD Ryzen AI Strix NPU platform.

### XDNA2 vs XDNA1 Comparison

| Feature | XDNA1 (Phoenix) | XDNA2 (Strix) |
|---------|-----------------|---------------|
| **Columns** | 4 | **8** (2× more) |
| **Total Tiles** | 24 (4×6) | **48** (8×6) (2× more) |
| **INT8 Performance** | 16 TOPS | **32 TOPS** (2× faster) |
| **BF16 Performance** | 32 TFLOPS | **64 TFLOPS** (2× faster) |
| **Memory per Tile** | 32 KB | 32 KB (same) |
| **Total Memory** | 768 KB | **1536 KB** (2× more) |
| **Device Node** | `/dev/accel/accel0` | `/dev/accel/accel0` |
| **Platform Name** | `npu1` | `npu2` (expected) |
| **Target Market** | Laptops (2023-2024) | High-end laptops (2025+) |

---

## Why Separate XDNA2 Kernels?

### Architectural Differences Requiring Separate Code

1. **Column Count**: 8 columns enable different parallelization strategies
   - **XDNA1**: 4-way parallelism across columns
   - **XDNA2**: 8-way parallelism (2× throughput potential)
   - **Impact**: MLIR designs must specify different tile layouts

2. **Platform Specification**: MLIR requires different device targets
   - **XDNA1**: `aie.device(npu1)` with 4-column constraints
   - **XDNA2**: `aie.device(npu2)` with 8-column layout (expected)
   - **Impact**: Cannot share MLIR wrapper files

3. **Optimization Strategies**: More columns = different kernel assignments
   - **XDNA1**: Fit attention + GELU + matmul in 4 columns
   - **XDNA2**: Can dedicate entire columns to single operations
   - **Impact**: Different performance characteristics

4. **Memory Access Patterns**: 2× tiles = different DMA strategies
   - **XDNA1**: Shared memory bandwidth across 4 columns
   - **XDNA2**: More parallel DMA channels available
   - **Impact**: Different buffer allocation

---

## Current Status

### XDNA1 Kernels (Ready)
✅ Located in: `../kernels_xdna1/`
- `softmax_xdna1.cc` - Vectorized BF16 softmax
- `gelu_optimized_xdna1.cc` - Tanh approximation GELU
- `swiglu_xdna1.cc` - Modern activation function
- `softmax_bf16_xdna1.cc` - High-precision softmax

**Status**: Ready for compilation and testing on Phoenix NPU

### XDNA2 Kernels (Planned)
🚧 **Not Yet Implemented**

**Reason for Delay**:
1. XDNA2 hardware not yet widely available
2. Focus on XDNA1 optimization first (proven hardware)
3. Kernel algorithms same, only MLIR wrappers differ

**Timeline**:
- **Q1 2025**: XDNA2 hardware availability
- **Q2 2025**: Begin XDNA2 kernel development
- **Q3 2025**: XDNA2 production deployment

---

## Future XDNA2 Kernel Plan

### Phase 1: Direct Port (4-6 weeks)
Copy XDNA1 kernels with minimal changes:
1. Rename files: `*_xdna1.cc` → `*_xdna2.cc`
2. Update header comments
3. No algorithm changes (C++ code identical)

**Files to Create**:
- `softmax_xdna2.cc`
- `gelu_optimized_xdna2.cc`
- `swiglu_xdna2.cc`
- `softmax_bf16_xdna2.cc`

### Phase 2: MLIR Optimization (6-8 weeks)
Create 8-column MLIR designs:
1. Redesign tile layout for 8 columns
2. Parallelize operations across more tiles
3. Optimize DMA for higher bandwidth

**Example MLIR Differences**:
```mlir
// XDNA1 (4 columns)
aie.device(npu1) {
  %tile02 = aie.tile(0, 2)  // Column 0
  %tile12 = aie.tile(1, 2)  // Column 1
  %tile22 = aie.tile(2, 2)  // Column 2
  %tile32 = aie.tile(3, 2)  // Column 3
}

// XDNA2 (8 columns)
aie.device(npu2) {
  %tile02 = aie.tile(0, 2)  // Column 0
  %tile12 = aie.tile(1, 2)  // Column 1
  // ... up to column 7
  %tile72 = aie.tile(7, 2)  // Column 7
  // 2× more parallelism!
}
```

### Phase 3: Advanced Optimization (8-10 weeks)
Leverage XDNA2-specific features:
1. Wider matrix tiles (128×128 instead of 64×64)
2. More complex pipelines
3. Multi-column softmax (parallel reduction)

---

## Expected Performance Gains

### Theoretical Speedup: 2× (Linear Scaling)
If kernels scale perfectly with columns:
- **XDNA1**: 10-15x realtime (4 columns)
- **XDNA2**: 20-30x realtime (8 columns) - **2× faster**

### Realistic Speedup: 1.5-1.8× (Amdahl's Law)
Due to non-parallelizable overhead:
- Memory bandwidth limits
- DMA synchronization
- Host-NPU transfers

**Best Case**:
- **XDNA1**: 15x realtime
- **XDNA2**: 27x realtime (1.8× faster)

**Conservative**:
- **XDNA1**: 10x realtime
- **XDNA2**: 15x realtime (1.5× faster)

---

## When to Use XDNA2 Kernels

### Use XDNA2 When:
✅ You have Strix hardware (8 columns)
✅ You need maximum throughput (>20x realtime)
✅ You want future-proof deployment

### Use XDNA1 When:
✅ You have Phoenix hardware (4 columns)
✅ 10-15x realtime is sufficient
✅ You want proven, tested code

### Hardware Detection
```python
import xrt

device = xrt.xrt_device(0)
num_columns = detect_npu_columns(device)

if num_columns == 4:
    print("Phoenix NPU (XDNA1) detected - use kernels_xdna1/")
elif num_columns == 8:
    print("Strix NPU (XDNA2) detected - use kernels_xdna2/")
else:
    print(f"Unknown NPU with {num_columns} columns")
```

---

## Development Roadmap

### Milestone 1: XDNA1 Complete (Current)
✅ All XDNA1 kernels copied and documented
⏳ Compilation pending
⏳ Testing on Phoenix NPU

### Milestone 2: XDNA1 Production (Q4 2024)
⏳ All kernels compiled and tested
⏳ Integrated with Whisper pipeline
⏳ Achieving 10-15x realtime

### Milestone 3: XDNA2 Planning (Q1 2025)
- Obtain Strix hardware for testing
- Design 8-column MLIR layouts
- Estimate performance gains

### Milestone 4: XDNA2 Development (Q2 2025)
- Port kernels to XDNA2
- Compile and test
- Optimize for 8 columns

### Milestone 5: XDNA2 Production (Q3 2025)
- Deploy XDNA2 kernels
- Achieve 20-30x realtime
- Performance comparison report

---

## Directory Structure (When Implemented)

```
kernels_xdna2/
├── README.md                    ← This file
├── softmax_xdna2.cc             ← 8-column optimized
├── gelu_optimized_xdna2.cc      ← 8-column optimized
├── swiglu_xdna2.cc              ← 8-column optimized
├── softmax_bf16_xdna2.cc        ← 8-column optimized
├── compile_all_xdna2.sh         ← Compilation script
└── tests/
    ├── test_softmax_xdna2.py
    ├── test_gelu_xdna2.py
    └── benchmark_xdna2.py
```

---

## Key Differences from XDNA1

### 1. Kernel Files (C++)
**Similarity**: 95% identical
- Same algorithms (softmax, GELU, SwiGLU)
- Same AIE2 intrinsics
- Same vector operations

**Differences**: Only header comments
- File names have `_xdna2` suffix
- Comments mention "8 columns"

### 2. MLIR Designs
**Similarity**: 50% identical
- Same kernel calls
- Same data types

**Differences**: Tile layout
- 8 columns instead of 4
- Different DMA routing
- Different buffer allocation

### 3. XCLBIN Files
**Similarity**: 0% (completely different)
- Platform-specific binary format
- Column configuration baked in
- Cannot share XCLBINs between XDNA1/XDNA2

---

## Integration Strategy

### Single-Codebase Approach
```python
# Auto-detect NPU generation
npu_gen = detect_npu_generation()

if npu_gen == "XDNA1":
    from kernels_xdna1 import softmax, gelu
    xclbin_path = "whisper_encoder_xdna1.xclbin"
elif npu_gen == "XDNA2":
    from kernels_xdna2 import softmax, gelu
    xclbin_path = "whisper_encoder_xdna2.xclbin"
else:
    raise RuntimeError(f"Unsupported NPU: {npu_gen}")
```

### Dual-XCLBIN Packaging
Ship both XCLBINs in production:
```
/models/
├── whisper_encoder_xdna1.xclbin  # For Phoenix
├── whisper_encoder_xdna2.xclbin  # For Strix
└── select_xclbin.py              # Auto-selection
```

---

## Conclusion

**Current Priority**: Focus on XDNA1 (Phoenix)
- Hardware available now
- Proven architecture
- Large user base

**Future Work**: XDNA2 when hardware available
- 2× performance potential
- Future-proof investment
- Straightforward port from XDNA1

**Recommendation**: Develop XDNA1 fully, then port to XDNA2 when hardware ships.

---

**Document Version**: 1.0
**Last Updated**: November 17, 2025
**Status**: Placeholder - awaiting XDNA2 hardware availability
**Contact**: Kernel Integration Team Lead
