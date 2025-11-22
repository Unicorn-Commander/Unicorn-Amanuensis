# Batched Tiling Optimization Analysis

## Problem Discovery

The initial batched tiling implementation **does not reduce NPU kernel calls** as intended.

### Current Implementation Issue

```python
for i in range(num_tiles_M):  # 47 iterations
    for j_batch_start in range(0, num_tiles_N, batch_col_tiles):  # 1 iteration (when batch=8)
        for k in range(num_tiles_K):  # 8 iterations
            for j_idx, j in enumerate(range(j_batch_start, j_batch_end)):  # 8 iterations
                # THIS STILL MAKES 47 × 1 × 8 × 8 = 3,008 NPU CALLS!
                partial_result = self._matmul_npu_64x64(A_tile, B_tile)
```

**Problem**: The batching only affects **CPU-side memory layout**, not NPU kernel invocations.

### Benchmark Results

```
Batch size: 1 → 3,008 calls → 163.7s
Batch size: 8 → 376 expected, but 3,008 actual → 163.7s (same!)
```

## Root Cause

The NPU kernel `_matmul_npu_64x64()` is **hardcoded to process 64×64 tiles**:
- Input buffers: 8192 bytes each (64×64 BF16)
- Output buffer: 8192 bytes (64×64 BF16)
- Cannot process larger matrices in a single call

To truly batch operations, we need one of these solutions:

## Solution Options

### Option 1: Larger NPU Kernel (RECOMMENDED)

Compile new XCLBIN kernels for larger tile sizes:

**New kernels needed**:
- `matmul_64x512.xclbin` - Process (64, 64) @ (64, 512) → (64, 512)
- `matmul_128x512.xclbin` - Process (128, 64) @ (64, 512) → (128, 512)

**Implementation**:
```python
def _matmul_npu_64x512(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Process full row: (64, K_pad) @ (K_pad, 512) → (64, 512)
    Reduces calls for (3001, 512) @ (512, 512) from 3,008 to 376
    """
    assert A.shape == (64, 64) and B.shape == (64, 512)

    # Convert to BF16
    A_bf16 = self._float_to_bf16(A.flatten())
    B_bf16 = self._float_to_bf16(B.flatten())  # 64×512 = 32KB

    # Allocate larger buffers
    buffer_A = 8192  # 64×64 BF16
    buffer_B = 65536  # 64×512 BF16 (64KB)
    buffer_C = 65536  # 64×512 BF16 (64KB)

    # Execute larger kernel
    # ... (XRT buffer allocation and kernel execution)
```

**Pros**:
- True reduction in kernel calls (8×)
- Reduces overhead from 163s to ~20s
- No algorithm changes needed

**Cons**:
- Requires compiling new NPU kernels
- Larger memory buffers (65KB vs 8KB)
- More complex MLIR-AIE2 kernel code

**Estimated Effort**: 2-4 hours to compile and test

---

### Option 2: Kernel Fusion at XRT Level

Batch multiple 64×64 operations into a single XRT kernel invocation:

**Implementation**:
```python
def _matmul_npu_batched_64x64(self, A_tiles: list, B_tiles: list) -> list:
    """
    Execute multiple 64×64 matmuls in a single XRT kernel call

    Args:
        A_tiles: List of 64×64 matrices
        B_tiles: List of 64×64 matrices

    Returns:
        List of 64×64 result matrices
    """
    batch_size = len(A_tiles)

    # Pack all tiles into single buffer
    A_packed = np.concatenate([tile.flatten() for tile in A_tiles])
    B_packed = np.concatenate([tile.flatten() for tile in B_tiles])

    # Convert to BF16
    A_bf16 = self._float_to_bf16(A_packed)
    B_bf16 = self._float_to_bf16(B_packed)

    # Allocate buffers for entire batch
    buffer_size = 8192 * batch_size

    # Execute kernel once with all data
    # ... (kernel processes batch_size tiles sequentially)

    # Unpack results
    return [result_packed[i*4096:(i+1)*4096].reshape(64, 64)
            for i in range(batch_size)]
```

**Pros**:
- Reuses existing 64×64 kernel
- Reduces XRT overhead (setup/teardown)
- Flexible batch sizes

**Cons**:
- Still processes tiles sequentially on NPU
- May not reduce actual compute time
- Complex buffer management

**Estimated Effort**: 4-6 hours

---

### Option 3: CPU-Side Loop Optimization (CURRENT)

Keep current implementation but optimize CPU-side overhead:

**What we have now**:
```python
for i in range(num_tiles_M):
    for j in range(num_tiles_N):
        for k in range(num_tiles_K):
            result = self._matmul_npu_64x64(A_tile, B_tile)
```

**Optimization**:
- Pre-allocate all buffers (reuse across calls)
- Reduce Python overhead (use Cython/numba)
- Pipeline CPU and NPU operations

**Pros**:
- No kernel changes needed
- Works with existing infrastructure

**Cons**:
- Doesn't reduce kernel calls
- Limited speedup (maybe 10-20%)
- Still ~140s for projection

**Estimated Effort**: 2-3 hours

---

## Recommendation

**Implement Option 1: Compile `matmul_64x512.xclbin` kernel**

### Why Option 1 is Best

1. **Maximum Speedup**: 8× reduction (163s → 20s)
2. **Proven Approach**: Standard tiling optimization
3. **Reasonable Effort**: 2-4 hours for single kernel
4. **Future-Proof**: Can compile other sizes as needed

### Implementation Plan

1. **Create MLIR-AIE2 kernel** for 64×512 matmul (~1 hour)
   - Based on existing `matmul_bf16.mlir`
   - Change tile dimensions
   - Update buffer sizes

2. **Compile XCLBIN** (~30 min)
   ```bash
   cd kernels_xdna1/build_matmul
   # Update matmul_bf16.mlir with new dimensions
   make clean && make
   ```

3. **Add Python wrapper** (~30 min)
   ```python
   def _matmul_npu_64x512(self, A, B):
       # Implementation similar to _matmul_npu_64x64
       pass
   ```

4. **Update tiling logic** (~1 hour)
   ```python
   def _matmul_npu_tiled(self, A, B):
       # Use _matmul_npu_64x512 for full rows
       for i in range(num_tiles_M):
           for k in range(num_tiles_K):
               # Single call: (64, 64) @ (64, 512) → (64, 512)
               row_result = self._matmul_npu_64x512(A_tile, B_row)
   ```

5. **Test and validate** (~1 hour)
   - Accuracy: max_error < 0.004
   - Performance: ~20s for (3001, 512) @ (512, 512)
   - Memory: verify 64KB buffers fit

### Expected Results

**Before** (current):
- Kernel calls: 3,008
- Time: 163.7s
- Per call: 54.4ms

**After** (with 64×512 kernel):
- Kernel calls: 376 (47 row tiles × 8 K tiles)
- Time: ~20s (376 × 54ms = 20.4s)
- Speedup: 8×

## Alternative: Accept Current Performance

If compiling new kernels is not feasible:

**Option**: Use current implementation but understand limitations:
- Performance: ~163s for encoder projection
- This is still usable for offline processing
- Pipeline test will take 3+ minutes (acceptable for batch jobs)

**When to use**:
- Rapid prototyping
- Non-realtime applications
- While waiting for optimized kernels

## Conclusion

The initial "batching" implementation was a **conceptual error** - it batched CPU-side operations but didn't reduce NPU kernel calls. To achieve true 8× speedup:

1. **Compile larger NPU kernels** (64×512 or 128×512)
2. **OR accept current performance** (~3 minutes for pipeline test)

The code structure is correct for using larger kernels once they're compiled - we just need the actual XCLBIN files.

---

## Files Modified

- `attention_npu.py` - Added batched tiling logic (lines 215-313)
- `quick_benchmark.py` - Benchmark script
- `test_single_call.py` - Single call timing test

## Key Insights

1. **54ms per kernel call** - Higher overhead than expected
2. **Batching in Python doesn't help** - Must batch at XRT/kernel level
3. **Need larger kernels** - Current 64×64 is the bottleneck
4. **Memory is fine** - 64KB buffers are acceptable

## Next Steps

**If proceeding with optimization**:
1. Locate matmul MLIR source (`matmul_bf16.mlir`)
2. Create `matmul_64x512.mlir` variant
3. Compile new XCLBIN
4. Integrate with Python wrapper
5. Validate performance and accuracy

**If accepting current performance**:
1. Document that 3-minute pipeline test is expected
2. Focus on other optimizations (BF16 accuracy, etc.)
3. Revisit when time allows for kernel recompilation

---

**Date**: November 21, 2025
**Status**: Analysis complete, implementation strategy defined
**Decision needed**: Compile new kernels OR accept current performance
