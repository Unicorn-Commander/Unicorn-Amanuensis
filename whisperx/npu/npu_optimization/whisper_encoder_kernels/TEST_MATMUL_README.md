# NPU Matmul Tiled - Test Infrastructure

## Overview

`test_npu_matmul_tiled.py` provides comprehensive testing for the tiled matmul implementation on the AMD Phoenix NPU. The test suite validates accuracy, performance, and edge cases for matrix multiplication operations used in Whisper encoder attention and projection layers.

## Test Coverage

### 1. Single Tile Test (64×64)
- **Purpose**: Validate exact tile size operation
- **Test Size**: 64×64 @ 64×64 = 64×64 output
- **Tiles**: 1×1 = 1 tile (no padding)
- **Use Case**: NPU kernels process one tile at a time
- **Expected**: Perfect accuracy, deterministic performance

### 2. Whisper Weights Test (512×512)
- **Purpose**: Test typical Whisper weight matrix dimensions
- **Test Size**: 512×512 @ 512×512 = 512×512 output
- **Tiles**: 8×8 = 64 tiles total (512 ÷ 64 = 8 in each dimension)
- **Computation**: 8×8×8 = 512 individual tile operations
- **Use Case**: All Whisper weight matrices are 512×512 (n_dims=512, n_heads=8, head_dim=64)
- **Expected**: Tile-aligned, perfect accuracy

### 3. Q,K,V Projection Test (3001×512 × 512×512)
- **Purpose**: Real-world attention projection sizes
- **Input**:
  - A: 3001×512 (sequence × hidden_dim, max Whisper sequence length)
  - B: 512×512 (weight matrix)
  - C: 3001×512 output
- **Tiling**:
  - M dimension: ⌈3001/64⌉ = 47 tiles
  - K dimension: ⌈512/64⌉ = 8 tiles
  - N dimension: ⌈512/64⌉ = 8 tiles
  - Total computation: 47×8×8 = 3008 tile operations
  - Padded size: 3008×512 (adds 7 rows)
- **Use Case**: Q = X @ W_q, K = X @ W_k, V = X @ W_v (attention projections)
- **Expected**: Requires padding, tolerance for padded elements

### 4. Non-Tile-Aligned Test
- **Purpose**: Validate padding and boundary handling
- **Test Cases**:
  - 100×100
  - 200×150→300
  - 99×99 (prime-ish)
  - 256×128→256
  - 1000×512→512
- **Challenge**: Requires proper padding to nearest tile boundary
- **Expected**: All produce correct results within tolerance

### 5. Edge Cases
- **Purpose**: Validate correctness at boundaries
- **Test Cases**:
  - 1×1 (minimal)
  - 64×128 (1×2 tiles in output)
  - 128×64 (2×1 tiles in output)
  - 32×32 (half tile)
  - 192×192 (3×3 tiles)
- **Expected**: Correct handling of sub-tile operations

### 6. Benchmarking
- **Purpose**: Measure performance vs CPU
- **Test Sizes**:
  - 64×64 (single tile)
  - 128×128 (2×2 tiles)
  - 256×256 (4×4 tiles)
  - 512×512 (8×8 tiles - Whisper)
  - 1024×512 (large sequence)
  - 3001×512 (max Whisper size)
- **Metrics**: CPU time, NPU time, speedup factor
- **Expected**: NPU speedup increases with size

## Running Tests

### Run Full Test Suite
```bash
python3 test_npu_matmul_tiled.py
```

**Output**: Complete test results with accuracy verification and performance metrics.

### Run CPU-Only Mode (No NPU Required)
```bash
python3 test_npu_matmul_tiled.py --cpu-only
```

**Output**: Validates test infrastructure works without NPU (useful for CI/CD).

### Run Individual Tests Programmatically
```python
from test_npu_matmul_tiled import NPUMatmulTester

# Create tester
tester = NPUMatmulTester(use_npu=True, verbose=True)

# Run specific test
tester.test_single_tile()
tester.test_whisper_weights()
tester.test_qkv_projection()
tester.test_edge_cases()

# Get tiling info
tiling = tester.compute_tiling_info(M=3001, N=512, K=512)
print(f"Tiles needed: {tiling['num_tiles_M']}×{tiling['num_tiles_N']}")

# Print summary
tester.print_summary()
```

## Accuracy Criteria

All tests use the same accuracy tolerance:

**Passing Criteria**:
- Relative error < 0.01% (0.0001)
- Compared against CPU NumPy reference: `C_cpu = A @ B`

**Formula**:
```
max_error = max(|C_npu - C_cpu|)
relative_error = max_error / max(|C_cpu|)
pass = relative_error < 0.0001
```

**Why 0.01%?**
- Float32 precision: ~7 decimal digits
- Allows for rounding errors in:
  - Data movement (DMA precision)
  - Computation (tile operations)
  - Format conversions (BF16 if used)

## Tiling Mechanism

### How Tiling Works

The NPU processes matmul in 64×64 tiles:

```
For C[i,j] = A[i,:] @ B[:,j]:
  For each 64×64 tile:
    - Load 64×64 from A (or padding)
    - Load 64×64 from B (or padding)
    - Compute on NPU core
    - Store 64×64 result to C
```

### Padding

Non-tile-aligned dimensions are padded:

```
Original: 3001×512
Tiled:    64×64
Tiles needed: ⌈3001/64⌉ × ⌈512/64⌉ = 47×8
Padded:   47*64 × 8*64 = 3008×512
```

Padded region contains zeros and results are ignored.

### Tiling Information

The `compute_tiling_info()` method returns:

```python
{
    'M': 3001,              # Original M dimension
    'N': 512,               # Original N dimension
    'K': 512,               # Original K dimension
    'tile_size': 64,        # Tile size (fixed)
    'num_tiles_M': 47,      # Tiles in M dimension
    'num_tiles_N': 8,       # Tiles in N dimension
    'num_tiles_K': 8,       # Tiles in K dimension
    'total_tiles': 376,     # Total M×N output tiles
    'padded_M': 3008,       # Padded M dimension
    'padded_K': 512,        # Padded K dimension
    'padded_N': 512,        # Padded N dimension
    'tile_aligned': False,  # Whether all dims are tile multiples
}
```

## Expected Performance

### Baseline (Current Implementation)
- **Mode**: CPU reference (until NPU integration complete)
- **Performance**: NumPy matmul speed
- **Intel CPU**: ~10-50ms per operation

### Target (With Custom NPU Kernels)

| Operation | CPU Time | NPU Time | Speedup |
|-----------|----------|----------|---------|
| 64×64 | 0.5ms | 0.05ms | 10x |
| 256×256 | 8ms | 0.4ms | 20x |
| 512×512 (Whisper) | 32ms | 1ms | 32x |
| 1024×512 | 64ms | 2ms | 32x |
| 3001×512 | 180ms | 5ms | 36x |

*Note: Actual speedup depends on:*
- NPU kernel implementation quality
- Data transfer overhead (DMA)
- Memory bandwidth utilization
- Tile configuration

## Implementation Notes

### Current Status
- ✅ **Test framework**: Complete and operational
- ✅ **Tiling calculations**: Validated
- ✅ **Accuracy checking**: Implemented
- ✅ **Benchmarking**: Ready
- ⏳ **NPU execution**: Placeholder (calls CPU matmul)

### What's Needed (Other Agent)
The other agent (attention_npu.py maintainer) should:

1. **Implement tiled NPU execution**:
   - Accept tiling info
   - Load A tile into NPU buffer
   - Load B tile into NPU buffer
   - Execute matmul kernel on NPU
   - Store result to C buffer

2. **Handle padding**:
   - Pad A/B as needed
   - Mark padded regions
   - Extract valid output only

3. **Manage memory**:
   - Allocate DMA buffers
   - Handle transfers efficiently
   - Reuse buffers across tiles

### Integration Example

```python
class MultiHeadAttentionNPU:
    def matmul_npu(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiply on NPU with tiling"""

        # Get tiling info
        tiling = self.compute_tiling_info(A.shape[0], B.shape[1], A.shape[1])

        # Initialize output
        C = np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)

        # Tile-wise computation
        for tile_i in range(tiling['num_tiles_M']):
            for tile_j in range(tiling['num_tiles_N']):
                # Get tile boundaries
                m_start = tile_i * 64
                m_end = min((tile_i+1)*64, A.shape[0])
                n_start = tile_j * 64
                n_end = min((tile_j+1)*64, B.shape[1])

                # Extract tiles (with zero padding)
                A_tile = np.zeros((64, 64), dtype=A.dtype)
                A_tile[:m_end-m_start, :] = A[m_start:m_end, :]

                # Execute on NPU
                C_tile = self._npu_tile_matmul(A_tile, B_tile)

                # Store result
                C[m_start:m_end, n_start:n_end] = C_tile[:m_end-m_start, :n_end-n_start]

        return C
```

## Troubleshooting

### Test Shows 0% Accuracy
- Check: Is `attention_npu.matmul_npu()` implemented or calling CPU fallback?
- Solution: Implement actual NPU kernel execution

### NPU Device Not Found
- Check: `/dev/accel/accel0` exists?
- Check: `xrt-smi examine` shows NPU?
- Solution: Run `setup_env.sh` to initialize NPU environment

### High Relative Error (>0.01%)
- Check: Is BF16 conversion causing precision loss?
- Check: Are tile boundaries handled correctly?
- Check: Is output alignment correct after padding?
- Solution: Add per-tile accuracy logging

### Timeout or Hang
- Check: Is NPU kernel infinite loop?
- Check: Are DMA transfers completing?
- Solution: Add timeout handling, test with smaller tiles first

## Test Output Example

```
========================================================================
                NPU MATMUL TILED - COMPREHENSIVE TEST SUITE
========================================================================

======================================================================
TEST 1: Single Tile (64×64)
======================================================================

Tiling info:
  Tile-aligned: True
  Total tiles: 1
  Padded dimensions: 64×64→64

📊 Running CPU reference...
   CPU time: 0.456ms

🚀 Running NPU matmul...
   NPU time: 0.456ms

✅ Accuracy Check:
   Max absolute error: 0.00e+00
   Relative error: 0.0000%
   Status: PASS

⚡ Performance:
   Speedup: 1.00x

======================================================================
TEST 2: Whisper Weights (512×512 = 8×8 tiles)
======================================================================
...
```

## References

- **NPU Hardware**: AMD Phoenix XDNA1 (16 TOPS INT8)
- **Tile Size**: 64×64 (fixed for AIE2 architecture)
- **XRT Version**: 2.20.0+
- **Whisper Model**: Base (512 hidden dims, 8 heads, 64 head_dim)

## Files

- **`test_npu_matmul_tiled.py`** (570 lines) - Complete test suite
- **`TEST_MATMUL_README.md`** (this file) - Documentation
- **`attention_npu.py`** - Integration point (other agent)

## Support

For issues or questions:
1. Check tiling info output (compute_tiling_info)
2. Verify CPU reference is correct
3. Enable verbose mode for detailed logs
4. Test with smaller sizes first
5. Check NPU device availability
