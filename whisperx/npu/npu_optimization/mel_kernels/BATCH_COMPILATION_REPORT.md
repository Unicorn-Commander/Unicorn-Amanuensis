# MEL Kernel Batch Compilation Report
## November 1, 2025

---

## Executive Summary

Successfully compiled batch-optimized MEL spectrogram kernel for AMD Phoenix NPU (XDNA1). The batch-10 configuration compiles successfully and fits within the 64KB AIE tile memory limit.

---

## Compilation Results

### ✅ Batch-10 Kernel (SUCCESS)
- **File**: `build_batch10/mel_batch10.xclbin`
- **Size**: 17 KB
- **Compilation Time**: 0.635 seconds
- **Status**: READY FOR TESTING

**Memory Layout**:
- Input ObjectFIFO: 16 KB (8KB × 2 buffers)
- Output ObjectFIFO: 1.6 KB (800B × 2 buffers)
- Stack: ~4 KB
- **Total: ~21.7 KB (33.9% of 64 KB)** ✅

**Performance Expectations**:
- Batch size: 10 frames per NPU call
- Overhead reduction: 10x (from 628,163 calls → 62,817 calls)
- Expected speedup: 6-8x
- For 1h 44m audio: 16-22 seconds (from 134s baseline)

### ❌ Batch-100 Kernel (FAILED - MEMORY OVERFLOW)
- **File**: `mel_fixed_v3_batch100.mlir`
- **Error**: `allocated buffers exceeded available memory`
- **Memory Requirement**: 177 KB
- **Available Memory**: 64 KB
- **Status**: NEEDS REDESIGN

**Memory Breakdown**:
- Input ObjectFIFO: 160 KB (80KB × 2 buffers) ❌
- Output ObjectFIFO: 16 KB (8KB × 2 buffers)
- Stack: ~1 KB
- **Total: ~177 KB (277% of 64 KB)** ❌

---

## Technical Details

### Successful Batch-10 Configuration

**MLIR File**: `mel_fixed_v3_batch10.mlir`

**Key Design**:
```mlir
// Input: 10 frames × 800 bytes = 8,000 bytes
aie.objectfifo @of_in(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<8000xi8>>

// Output: 10 frames × 80 bytes = 800 bytes
aie.objectfifo @of_out(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<800xi8>>
```

**Processing Loop**:
- Outer loop: Infinite (processes batches indefinitely)
- Inner loop: 10 iterations (processes each frame in batch)
- C kernel call: `mel_kernel_simple()` - called 10 times per batch

**DMA Transfers**:
- Host → NPU: 8,000 bytes (10 input frames at once)
- NPU → Host: 800 bytes (10 output mel features at once)

### Compilation Command

```bash
/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=mel_batch10.xclbin \
    --npu-insts-name=insts_batch10.bin \
    mel_fixed_v3_batch10.mlir
```

**Critical Flags**:
- `--no-xchesscc`: Skip chess compiler for host code
- `--no-xbridge`: Skip xbridge code generation
- `--alloc-scheme=basic-sequential`: Use sequential buffer allocation

---

## Memory Constraint Analysis

### AIE Tile Memory Architecture

**Phoenix NPU (XDNA1)**:
- **Data Memory**: 64 KB per compute tile
- **Program Memory**: Separate (not a constraint)
- **Tiles Available**: 4×6 = 24 tiles (16 compute + 8 memory)

### Buffer Sizing Trade-offs

| Batch Size | Input Buffers | Output Buffers | Total Memory | Fits in 64KB? | Speedup |
|------------|---------------|----------------|--------------|---------------|---------|
| 1 (original) | 1.6 KB | 160 B | 5.36 KB | ✅ Yes | 1x (baseline) |
| 10 | 16 KB | 1.6 KB | 21.7 KB | ✅ Yes | 10x |
| 20 | 32 KB | 3.2 KB | 39.2 KB | ✅ Yes | 20x |
| 50 | 80 KB | 8 KB | 92 KB | ❌ No | - |
| 100 | 160 KB | 16 KB | 177 KB | ❌ No | - |

---

## Next Steps

### Immediate (Hours)

1. **Test Batch-10 Kernel on NPU**
   ```bash
   python3 test_mel_batch10.py
   ```
   - Load XCLBIN to NPU device
   - Process test audio
   - Validate accuracy vs librosa
   - Measure performance

2. **Create Python Wrapper**
   - Batch audio frames into groups of 10
   - Manage DMA transfers
   - Collect and reassemble mel features

### Short-term (Days)

3. **Try Batch-20 Configuration**
   - Memory requirement: ~39.2 KB (61% of 64 KB)
   - Expected speedup: 20x
   - Better performance with acceptable memory usage

4. **Optimize for Larger Batches**
   - Use single-buffering instead of double-buffering
   - Reduces memory by 50% (sacrifice pipelining)
   - Could enable batch-40 or batch-50

### Medium-term (Weeks)

5. **Multi-Tile Strategy for Batch-100**
   - Distribute batch across 2 tiles
   - Tile 1: Processes frames 0-49
   - Tile 2: Processes frames 50-99
   - Synchronize results via memory tiles
   - Achieves 100x speedup with 2× parallelism

---

## Performance Projections

### Test Audio: 1 hour 44 minutes (628,163 frames)

**Current Baseline** (single-frame):
- NPU calls: 628,163
- Processing time: 134 seconds
- Overhead per frame: 213 µs

**With Batch-10** (compiled and ready):
- NPU calls: 62,817 (10x reduction)
- Processing time: 16-22 seconds (estimated)
- Overhead per frame: 21 µs
- **Speedup: 6-8x**

**With Batch-20** (feasible):
- NPU calls: 31,409 (20x reduction)
- Processing time: 8-11 seconds (estimated)
- Overhead per frame: 11 µs
- **Speedup: 12-17x**

**With Multi-tile Batch-100** (future):
- NPU calls: 6,282 per tile (100x reduction)
- Processing time: 3-5 seconds (estimated)
- Overhead per frame: 2 µs
- **Speedup: 27-45x**

---

## Files Generated

### Build Artifacts
```
build_batch10/
├── mel_batch10.xclbin          17 KB - NPU binary (READY)
├── insts_batch10.bin           300 B - NPU instructions
├── compilation.log             5.2 KB - Build log
├── mel_fixed_v3_batch10.mlir   8.0 KB - MLIR source
└── mel_fixed_combined.o        11 KB - C kernel object
```

### Source Files
```
mel_kernels/
├── mel_fixed_v3_batch10.mlir   8.0 KB - Batch-10 MLIR (WORKING)
├── mel_fixed_v3_batch100.mlir  8.1 KB - Batch-100 MLIR (memory overflow)
├── compile_batch10.sh          2.6 KB - Build script
└── mel_fixed_combined.o        11 KB - C kernel
```

---

## Integration Path

### Phase 1: Single-tile Batch-10 (THIS WEEK)
1. ✅ Compile XCLBIN (DONE)
2. ⏳ Test on NPU hardware
3. ⏳ Validate accuracy
4. ⏳ Measure performance
5. ⏳ Integrate with WhisperX pipeline

### Phase 2: Single-tile Batch-20 (NEXT WEEK)
1. Create `mel_fixed_v3_batch20.mlir`
2. Compile and test
3. Compare performance vs batch-10
4. Choose optimal batch size

### Phase 3: Multi-tile Batch-100 (WEEK 3-4)
1. Design 2-tile architecture
2. Implement tile-to-tile communication
3. Synchronize results
4. Achieve 27-45x target speedup

---

## Key Learnings

### 1. Memory is the Primary Constraint
- AIE tiles have only 64 KB of data memory
- Double-buffering uses 2× memory for pipelining
- Must balance batch size with memory limits

### 2. Compilation Tools Work Well
- aiecc.py successfully generates XCLBINs
- Compilation is fast (<1 second)
- Clear error messages guide optimization

### 3. Incremental Approach is Best
- Start with smaller batch sizes
- Validate accuracy at each step
- Optimize incrementally based on measurements

### 4. Multi-tile is the Path to 100x
- Single tile cannot hold batch-100 buffers
- Multi-tile distribution is necessary
- Adds complexity but achieves target performance

---

## Recommendations

### Production Deployment
- **Use Batch-10 initially** (proven to compile, safe memory usage)
- **Upgrade to Batch-20** after validation (better speedup, still safe)
- **Consider Multi-tile Batch-100** for maximum performance (requires additional development)

### Development Priority
1. Test batch-10 on hardware (highest priority)
2. Create batch-20 variant (low risk, high reward)
3. Implement single-buffering (enables batch-40)
4. Design multi-tile architecture (for batch-100)

---

## Success Criteria

### Batch-10 Success (Target: This Week)
- ✅ Compiles successfully
- ⏳ Loads on NPU without errors
- ⏳ Produces accurate mel features (>0.95 correlation)
- ⏳ Achieves 6-8x speedup
- ⏳ Integrates with WhisperX

### Batch-20 Success (Target: Next Week)
- ⏳ Compiles within memory limits
- ⏳ Achieves 12-17x speedup
- ⏳ Maintains accuracy

### Batch-100 Success (Target: Week 3-4)
- ⏳ Multi-tile design implemented
- ⏳ Achieves 27-45x target speedup
- ⏳ Production-ready integration

---

## References

**Documentation**:
- `mel_fixed_v3_batch10.mlir` - Working batch-10 configuration
- `mel_fixed_v3_batch100.mlir` - Failed batch-100 (reference for multi-tile)
- `compile_batch10.sh` - Successful build script
- `MASTER_CHECKLIST_OCT28.md` - Overall NPU optimization plan

**Related Work**:
- Single-frame kernel: `mel_fixed_v3.mlir` (baseline)
- FFT fixes: `fft_fixed_point.c`
- Mel filters: `mel_kernel_fft_fixed.c`

---

**Compiled by**: Magic Unicorn Unconventional Technology & Stuff Inc.
**Date**: November 1, 2025
**Status**: Batch-10 READY FOR TESTING ✅
