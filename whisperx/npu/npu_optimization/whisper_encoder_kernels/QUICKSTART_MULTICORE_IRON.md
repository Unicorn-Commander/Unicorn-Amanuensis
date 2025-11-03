# Multi-Core Attention Quick Start Guide

**Goal**: Run 4-column parallel attention for 4× throughput improvement

---

## Prerequisites

✅ **Already Have**:
- AMD Phoenix NPU hardware
- XRT 2.20.0 installed
- MLIR-AIE v1.1.1 installed
- Compiled single-core kernel object

⏳ **Need to Install**:
- AMD AIETools (chess compiler suite)

---

## Quick Start (After AIETools Installation)

### Step 1: Generate Multi-Core MLIR (DONE ✅)

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels

# Already generated: attention_iron_generated.mlir
ls -lh attention_iron_generated.mlir
```

### Step 2: Compile to XCLBIN

```bash
# Set up environment
export AIETOOLS=/path/to/aietools
export PATH=$AIETOOLS/bin:$PATH

# Compile
./compile_attention_iron.sh

# Expected output:
# ✓ XCLBIN generated: build_attention_iron/attention_multicore.xclbin
```

### Step 3: Test Multi-Core Performance

```bash
# Run test script
./test_attention_multicore_iron.py

# Expected results:
# Time per batch (4 tiles): ~2.85ms
# Speedup: 4.0×
# Realtime factor: 52-65×
```

---

## Files Overview

### Implementation
- `attention_64x64_multicore_iron.py` - IRON API generator (218 lines)
- `attention_iron_generated.mlir` - Generated 4-column MLIR (8.9KB)

### Build
- `compile_attention_iron.sh` - Compilation script
- `build_attention_iron/` - Build directory (created automatically)

### Testing
- `test_attention_multicore_iron.py` - Test harness with benchmarking

### Documentation
- `QUICKSTART_MULTICORE_IRON.md` - This file (quick start)
- `IRON_MULTICORE_IMPLEMENTATION.md` - Technical details (13KB)
- `MULTICORE_IRON_SESSION_SUMMARY.md` - Session summary (15KB)

---

## Architecture Summary

```
Phoenix NPU (4 columns × 6 rows):

Row 2: [Compute 0,2] [Compute 1,2] [Compute 2,2] [Compute 3,2]
        ↑ attention   ↑ attention   ↑ attention   ↑ attention
        |             |             |             |
Row 1: [Mem 0,1]     [Mem 1,1]     [Mem 2,1]     [Mem 3,1]
        ↑↓ FIFO      ↑↓ FIFO       ↑↓ FIFO       ↑↓ FIFO
        |             |             |             |
Row 0: [Shim 0,0]    [Shim 1,0]    [Shim 2,0]    [Shim 3,0]
        ↑↓ DMA       ↑↓ DMA        ↑↓ DMA        ↑↓ DMA
        |             |             |             |
       Host Memory (4 tiles in parallel)
```

**Key Features**:
- 4 independent attention computations in parallel
- Each column processes 1 tile (64×64)
- Double buffering for DMA overlap
- Automatic synchronization (IRON-generated)

---

## Performance Expectations

### Current (Single-Core)
```
NPU Utilization: 25% (1 column)
Time per tile: 2.85ms
Realtime factor: 16.2×
```

### Target (Multi-Core)
```
NPU Utilization: 100% (4 columns)
Time per batch of 4: 2.85ms
Effective time per tile: 0.71ms
Realtime factor: 52-65×
Improvement: 4× throughput
```

---

## Troubleshooting

### Issue: "chess-llvm-link not found"

**Solution**: Install AMD AIETools
```bash
# Download from https://www.xilinx.com/support/download.html
export AIETOOLS=/path/to/aietools
export PATH=$AIETOOLS/bin:$PATH
```

### Issue: "XCLBIN load failed"

**Check**:
```bash
# Verify NPU device
ls -l /dev/accel/accel0

# Check XRT
/opt/xilinx/xrt/bin/xrt-smi examine
```

### Issue: "Lower than expected speedup"

**Profile**:
```python
# Run with fewer batches to see individual times
./test_attention_multicore_iron.py --n-batches 3
```

---

## Integration with Whisper

### Current Integration Point
```python
# In encoder block: single-tile processing
result = run_attention_npu(q, k, v)  # 2.85ms per tile
```

### Multi-Core Integration
```python
# Batch 4 tiles together
results = run_attention_multicore_npu([
    (q0, k0, v0),
    (q1, k1, v1),
    (q2, k2, v2),
    (q3, k3, v3)
])  # 2.85ms for all 4 tiles (4× faster)
```

---

## Next Steps After Success

1. **Validate Accuracy**: Compare outputs with single-core
2. **Profile Pipeline**: Identify remaining bottlenecks
3. **Scale to Full Encoder**: Use multi-core for all attention layers
4. **Measure End-to-End**: Full Whisper transcription benchmark

---

## Support

**Documentation**:
- Technical: `IRON_MULTICORE_IMPLEMENTATION.md`
- Session: `MULTICORE_IRON_SESSION_SUMMARY.md`

**IRON Examples**: `/home/ucadmin/mlir-aie-fresh/mlir-aie/programming_examples/`

**AMD Resources**: https://www.xilinx.com/products/design-tools/aie.html

---

**Status**: Ready for compilation after AIETools installation
**Expected Completion**: 4-6 hours after tools available
**Performance Target**: 4× throughput improvement (27-33× realtime)
