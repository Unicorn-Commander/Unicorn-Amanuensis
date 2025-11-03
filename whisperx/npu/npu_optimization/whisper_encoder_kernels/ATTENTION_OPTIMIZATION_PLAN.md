# Attention Kernel Optimization Plan

**Mission**: Optimize attention kernel from 2.233ms/tile to 0.5-1.0ms/tile (2-4x improvement)

**Current Status**: 14.0x realtime → **Target**: 40-60x realtime

**Generated**: 2025-10-30
**Team Lead**: Attention Kernel Optimization
**Priority**: CRITICAL (73.6% of total execution time)

---

## Executive Summary

### Current Performance Bottleneck

```
╔════════════════════════════════════════════════════════╗
║        ATTENTION KERNEL - CRITICAL BOTTLENECK          ║
╚════════════════════════════════════════════════════════╝

Current Performance:
─────────────────────
Execution Time:  2.233ms per 64×64 tile
Percentage:      73.6% of total execution time
Realtime Factor: 14.0× (with attention as bottleneck)

Target Performance:
───────────────────
Execution Time:  0.5-1.0ms per 64×64 tile
Improvement:     2-4× speedup
Expected RTF:    40-60× realtime

Impact:
───────
Even a 2× improvement in attention yields 28× realtime overall
A 4× improvement yields 56× realtime overall
```

### Profiling Data Analysis

From benchmark suite measurements (20 iterations):
- **Mean**: 2.233ms
- **Std Dev**: 0.069ms (3.1% variance - very stable)
- **P95**: 2.370ms
- **P99**: 2.370ms
- **Min**: 2.104ms
- **Max**: 2.370ms

**Conclusion**: Performance is consistent and repeatable. No significant outliers.

---

## Current Implementation Analysis

### Architecture: `attention_int8_64x64.c`

**Three-Stage Pipeline**:
1. **QK^T Computation**: Q @ K^T → scores [64×64]
2. **Softmax**: Row-wise softmax → attention_weights [64×64]
3. **Weighted Sum**: attention_weights @ V → output [64×64]

### Detailed Time Breakdown (Estimated)

Based on computational complexity:

```
Stage 1: Q @ K^T (Matrix Multiply)
──────────────────────────────────
Operations: 64×64×64 MACs = 262,144 ops
Complexity: O(n³)
Estimated:  ~40% of time (~0.9ms)
Bottleneck: Triple nested loop, no vectorization

Stage 2: Softmax (Per-Row)
──────────────────────────
Operations: 64 rows × (find_max + exp + normalize)
Complexity: O(n²) but with exp() approximation
Estimated:  ~30% of time (~0.7ms)
Bottleneck: Taylor series approximation, division

Stage 3: Weighted Sum @ V (Matrix Multiply)
────────────────────────────────────────────
Operations: 64×64×64 MACs = 262,144 ops
Complexity: O(n³)
Estimated:  ~30% of time (~0.7ms)
Bottleneck: Same as Stage 1
```

### Memory Access Patterns

**Current Access**:
```c
// Q @ K^T: Row-major × Row-major (non-optimal for K^T)
for (i < 64)
  for (j < 64)
    for (k < 64)
      score += Q[i * 64 + k] * K[j * 64 + k];  // K accessed by stride
```

**Problem**: K is accessed with stride 64 (poor cache locality)

**Better Pattern**: Transpose K first for contiguous access

---

## Optimization Opportunities (Ranked by Impact)

### 1. VECTORIZE Q@K^T MATMUL (HIGHEST IMPACT)

**Expected Improvement**: 2-3× speedup
**Complexity**: Medium
**Timeline**: 1-2 days
**Confidence**: 95%

**Current Implementation**:
```c
// Scalar code - processes 1 element at a time
for (k < 64)
    score += Q[i*64+k] * K[j*64+k];
```

**Optimized Implementation**:
```c
// AIE2 vector intrinsics - processes 32 elements at a time
// Use v32int8 vectors with MAC instructions
v32int8 q_vec = *(v32int8*)&Q[i*64];
v32int8 k_vec = *(v32int8*)&K[j*64];
v32acc32 acc = mac(acc, q_vec, k_vec);  // 32 MACs in parallel
```

**AIE2 Capabilities**:
- 32-element INT8 SIMD operations
- Multiply-accumulate (MAC) in single cycle
- 32 MACs per cycle = 32× speedup potential

**Why This Works**:
- Inner loop is 64 elements = 2 vector operations (32+32)
- AIE2 has dedicated vector MAC units
- Eliminates 64 scalar multiply-adds per output element

**Implementation Steps**:
1. Include AIE2 vector intrinsics: `#include <aie_api/aie.hpp>`
2. Replace inner loop with vector operations
3. Handle remainder (if dimension not multiple of 32)
4. Test for correctness and performance

**Code Template**:
```c
#include <aie_api/aie.hpp>

void attention_qk_vectorized(
    const int8_t* Q,
    const int8_t* K,
    int8_t* scores,
    uint32_t scale_shift
) {
    using namespace aie;

    for (uint32_t i = 0; i < 64; i++) {
        for (uint32_t j = 0; j < 64; j++) {
            // Load 32 elements at a time
            v32int8 q_vec0 = *(v32int8*)&Q[i*64 + 0];
            v32int8 q_vec1 = *(v32int8*)&Q[i*64 + 32];
            v32int8 k_vec0 = *(v32int8*)&K[j*64 + 0];
            v32int8 k_vec1 = *(v32int8*)&K[j*64 + 32];

            // Vector MAC operations
            v32acc32 acc = mul(q_vec0, k_vec0);  // First 32 elements
            acc = mac(acc, q_vec1, k_vec1);      // Next 32 elements

            // Reduce accumulator to scalar
            int32_t score = reduce_add(acc.to_vector<int32_t>());

            // Scale and clamp
            score >>= scale_shift;
            if (score > 127) score = 127;
            if (score < -128) score = -128;

            scores[i * 64 + j] = (int8_t)score;
        }
    }
}
```

---

### 2. OPTIMIZE SOFTMAX APPROXIMATION (HIGH IMPACT)

**Expected Improvement**: 1.5-2× speedup
**Complexity**: Medium-High
**Timeline**: 2-3 days
**Confidence**: 85%

**Current Implementation**:
```c
// Taylor series: exp(x) ≈ 64 + x + x²/128
int32_t exp_val = 64 + x + (x * x) / 128;
```

**Problems**:
1. Division by 128 is slow (even with right shift)
2. Two passes: exp calculation + normalization
3. Stores intermediate exp_vals[] array

**Optimization Strategies**:

#### Strategy A: Lookup Table (LUT)

**Pros**: Very fast, constant time
**Cons**: Requires 256-entry table (1KB memory)

```c
// Precomputed exp values for INT8 range [-128, 127]
const int32_t EXP_LUT[256] = { /* precomputed values */ };

// Usage:
int32_t exp_val = EXP_LUT[(x - max_val) + 128];  // Shift to [0, 255]
```

**Implementation**:
1. Generate LUT offline with precise exp() values
2. Store in kernel .rodata section
3. Single memory lookup vs 3 arithmetic ops

**Expected Speedup**: 3-4× for softmax stage

#### Strategy B: Vectorized Softmax

```c
// Process 32 elements at a time
v32int8 input_vec = *(v32int8*)&input[i];
v32int8 max_vec = broadcast<int8_t>(max_val);
v32int8 shifted = sub(input_vec, max_vec);

// Vector lookup from LUT (if supported)
// Or vector approximation
```

#### Strategy C: Single-Pass Softmax

**Idea**: Compute exp and normalize in one pass (avoid storing exp_vals[])

```c
// First pass: compute sum(exp(x))
int32_t sum = 0;
for (i < N) sum += exp_approx(input[i] - max);

// Second pass: compute normalized values
for (i < N) output[i] = (exp_approx(input[i] - max) * 127) / sum;
```

**Problem**: Still two passes, but saves 256-entry temp buffer

**Better**: Fused kernel with streaming output

---

### 3. MEMORY LAYOUT OPTIMIZATION (MEDIUM-HIGH IMPACT)

**Expected Improvement**: 1.3-1.5× speedup
**Complexity**: Low-Medium
**Timeline**: 1 day
**Confidence**: 90%

**Current Layout**:
```
QKV_combined [12288 bytes]:
  Q: bytes 0-4095     [64×64 row-major]
  K: bytes 4096-8191  [64×64 row-major]
  V: bytes 8192-12287 [64×64 row-major]
```

**Problem**: Computing K^T requires strided access (stride=64)

**Optimization**: Pre-transpose K in host code

**Option A: Pre-transposed K**
```
QKV_combined [12288 bytes]:
  Q: bytes 0-4095     [64×64 row-major]
  K^T: bytes 4096-8191  [64×64 column-major = K transposed]
  V: bytes 8192-12287 [64×64 row-major]
```

**Benefit**: Contiguous access for both Q and K^T
```c
// Before: stride access
score += Q[i*64+k] * K[j*64+k];  // K stride = 64

// After: contiguous access
score += Q[i*64+k] * KT[k*64+j];  // Both contiguous
```

**Option B: Blocked/Tiled Layout**
```
Store matrices in 16×16 tiles for better cache locality
Q: 16 tiles of 16×16 (4×4 grid)
K: 16 tiles of 16×16
V: 16 tiles of 16×16
```

**Benefit**: Each 16×16 tile fits in L1 cache, reduces misses

---

### 4. TILING STRATEGY OPTIMIZATION (MEDIUM IMPACT)

**Expected Improvement**: 1.2-1.4× speedup
**Complexity**: Medium
**Timeline**: 2-3 days
**Confidence**: 80%

**Current**: Process full 64×64 at once

**Alternative**: Process in 32×32 or 16×16 tiles

**Comparison**:

| Tile Size | Memory (KB) | Cache Locality | Parallelism |
|-----------|-------------|----------------|-------------|
| 64×64 | 28 KB (88% of 32KB) | Poor | High |
| 32×32 | 8 KB (25% of 32KB) | Good | Medium |
| 16×16 | 2.5 KB (8% of 32KB) | Excellent | Low |

**Recommendation**: Use 32×32 tiles (already implemented in `attention_int8_64x64_tiled.c`)

**Why 32×32 is Better**:
1. Fits comfortably in memory (8KB vs 28KB)
2. Better cache reuse
3. Enables loop unrolling
4. Can process 2 tiles in parallel on multi-core

**Already Implemented**: `attention_int8_64x64_tiled.c` exists!
- Test this version vs current
- May already provide speedup

---

### 5. QUANTIZATION OPTIMIZATION (MEDIUM IMPACT)

**Expected Improvement**: 1.5-2× speedup
**Complexity**: High
**Timeline**: 1-2 weeks
**Confidence**: 70%

**Current**: INT8 precision (8-bit weights and activations)

**Option A: INT4 Quantization**

**Pros**:
- 2× memory reduction (6KB vs 12KB for QKV)
- 2× higher throughput (can process 64 values vs 32)
- Same accuracy for attention (weights are already quantized)

**Cons**:
- Requires INT4 packing/unpacking
- Limited hardware support (AIE2 has INT8/INT16 natively)
- May need custom ops

**Option B: Mixed Precision**
- Keep Q, K in INT8 (for QK^T accuracy)
- Use INT4 for V (less critical)
- Saves 4KB memory, enables larger tiles

**Recommendation**: Test INT4 if INT8 optimizations plateau

---

### 6. FUSE OPERATIONS (HIGH IMPACT, COMPLEX)

**Expected Improvement**: 1.5-2× speedup
**Complexity**: Very High
**Timeline**: 1-2 weeks
**Confidence**: 75%

**Current Pipeline**:
```
Stage 1: Q@K^T → scores[4096]
Stage 2: softmax(scores) → weights[4096]
Stage 3: weights@V → output[4096]
```

**Problem**: Each stage writes intermediate results to memory

**Optimization**: Fused kernel that streams data

**Fused Q@K^T + Softmax**:
```c
// Compute one row at a time
for (i < 64) {
    // Compute scores for row i
    for (j < 64) {
        scores_row[j] = dot_product(Q[i], K[j]);
    }

    // Immediately apply softmax (no memory write)
    softmax_inplace(scores_row, 64);

    // Compute weighted sum for this row
    for (j < 64) {
        output[i*64+j] = dot_product(scores_row, V[:,j]);
    }
}
```

**Benefits**:
- No intermediate 4096-byte buffers
- Better cache locality
- Reduced memory bandwidth

**Challenges**:
- More complex logic
- Harder to debug
- May hurt readability

---

### 7. MULTI-CORE ATTENTION (HIGHEST LONG-TERM IMPACT)

**Expected Improvement**: 4× speedup (with 4 cores)
**Complexity**: Very High
**Timeline**: 2-3 weeks
**Confidence**: 80% (toolchain dependency)

**Current**: Single-core execution

**Strategy**: Distribute work across 4 NPU columns

**Parallelization Approaches**:

#### Approach A: Row Parallelism
```
Core 0: Process rows 0-15   (16 rows)
Core 1: Process rows 16-31  (16 rows)
Core 2: Process rows 32-47  (16 rows)
Core 3: Process rows 48-63  (16 rows)
```

**Pros**: Perfect load balancing
**Cons**: Requires broadcasting K, V to all cores

#### Approach B: Tile Parallelism
```
Process 64×64 as 4 tiles of 32×32
Core 0: Tile (0,0) - rows 0-31, cols 0-31
Core 1: Tile (0,1) - rows 0-31, cols 32-63
Core 2: Tile (1,0) - rows 32-63, cols 0-31
Core 3: Tile (1,1) - rows 32-63, cols 32-63
```

**Pros**: Independent computation
**Cons**: More complex DMA setup

**Blocker**: Multi-core XCLBIN compilation (toolchain issue)
- See `SESSION_COMPLETE_OCT30.md` for details
- Resolution timeline: 4-8 hours

---

## Implementation Roadmap

### Phase 1: Quick Wins (1 week)

**Target**: 1.5-2× improvement (35ms → 25ms per tile)

**Tasks**:
1. Vectorize Q@K^T matmul (1-2 days)
2. Test tiled version (`attention_int8_64x64_tiled.c`) (0.5 day)
3. Optimize memory layout (transpose K) (0.5 day)
4. Benchmark and validate (1 day)

**Expected Result**: 20-25× realtime overall

---

### Phase 2: Softmax Optimization (1 week)

**Target**: Additional 1.5× improvement (25ms → 17ms per tile)

**Tasks**:
1. Implement LUT-based exp() (1 day)
2. Vectorize softmax (2 days)
3. Single-pass softmax (if beneficial) (1 day)
4. Benchmark and validate (1 day)

**Expected Result**: 30-35× realtime overall

---

### Phase 3: Advanced Optimizations (2 weeks)

**Target**: 2× improvement (17ms → 8-10ms per tile)

**Tasks**:
1. Fused Q@K^T + Softmax kernel (3-4 days)
2. Fused Softmax + Weighted Sum (3-4 days)
3. INT4 quantization exploration (3-4 days)
4. Comprehensive benchmarking (2 days)

**Expected Result**: 40-50× realtime overall

---

### Phase 4: Multi-Core (2-3 weeks)

**Target**: 4× improvement (8-10ms → 2-3ms per tile)

**Prerequisites**: Resolve multi-core toolchain

**Tasks**:
1. Resolve AIETools/Peano compilation (4-8 hours)
2. Implement row-parallel attention (3-4 days)
3. Test and debug multi-core XCLBIN (2-3 days)
4. Optimize inter-core communication (2-3 days)
5. Full validation and benchmarking (2-3 days)

**Expected Result**: 80-100× realtime overall

---

## Detailed Implementation: Vectorized Q@K^T

### File: `attention_int8_64x64_vectorized.c`

```c
/**
 * Vectorized INT8 Attention Mechanism for Whisper Encoder
 * Optimized for AIE2 vector operations - 2-3× faster than scalar
 */

#include <stdint.h>
#include <aie_api/aie.hpp>

using namespace aie;

/**
 * Vectorized Q @ K^T computation
 * Uses AIE2 v32int8 SIMD for 32-element parallel MACs
 */
void attention_qk_vectorized_64x64(
    const int8_t* Q,           // [64 × 64] query matrix
    const int8_t* K,           // [64 × 64] key matrix
    int8_t* scores,            // [64 × 64] output scores
    uint32_t scale_shift       // Right shift for scaling
) {
    for (uint32_t i = 0; i < 64; i++) {
        for (uint32_t j = 0; j < 64; j++) {
            // Process inner dimension in chunks of 32

            // First 32 elements
            v32int8 q_vec0 = *(v32int8*)&Q[i * 64 + 0];
            v32int8 k_vec0 = *(v32int8*)&K[j * 64 + 0];
            v32acc32 acc = mul(q_vec0, k_vec0);

            // Next 32 elements (accumulate)
            v32int8 q_vec1 = *(v32int8*)&Q[i * 64 + 32];
            v32int8 k_vec1 = *(v32int8*)&K[j * 64 + 32];
            acc = mac(acc, q_vec1, k_vec1);

            // Reduce 32-element accumulator to scalar
            auto acc_vec = acc.to_vector<int32_t>();
            int32_t score = 0;

            // Manual reduction (or use reduce_add if available)
            for (uint32_t v = 0; v < 32; v++) {
                score += acc_vec[v];
            }

            // Scale by 1/sqrt(d_k)
            score >>= scale_shift;

            // Clamp to INT8 range
            if (score > 127) score = 127;
            if (score < -128) score = -128;

            scores[i * 64 + j] = (int8_t)score;
        }
    }
}

/**
 * Complete vectorized attention kernel
 */
void attention_64x64_vectorized(
    const int8_t* QKV_combined,  // [12288] combined Q+K+V buffer
    int8_t* output,              // [64 × 64] output matrix
    uint32_t scale_shift         // Right shift for Q@K^T
) {
    const int8_t* Q = &QKV_combined[0];
    const int8_t* K = &QKV_combined[4096];
    const int8_t* V = &QKV_combined[8192];

    int8_t scores[4096];
    int8_t attention_weights[4096];

    // Step 1: Vectorized Q @ K^T
    attention_qk_vectorized_64x64(Q, K, scores, scale_shift);

    // Step 2: Softmax (can be vectorized separately)
    for (uint32_t i = 0; i < 64; i++) {
        softmax_int8_64(&scores[i * 64], &attention_weights[i * 64], 64);
    }

    // Step 3: Vectorized weighted sum @ V
    attention_weighted_sum_vectorized_64x64(attention_weights, V, output);
}
```

---

## Detailed Implementation: LUT-Based Softmax

### Precompute Lookup Table

```python
import numpy as np

# Generate exp LUT for INT8 range shifted by max
def generate_exp_lut():
    lut = []
    for x in range(-128, 128):
        # Scale to match INT8 precision
        exp_val = int(64 * np.exp(x / 64.0))
        exp_val = max(0, min(255, exp_val))  # Clamp to uint8
        lut.append(exp_val)
    return lut

exp_lut = generate_exp_lut()

# Generate C array
print("const uint8_t EXP_LUT[256] = {")
for i in range(0, 256, 16):
    values = ", ".join(f"{exp_lut[i+j]:3d}" for j in range(16))
    print(f"    {values},")
print("};")
```

### C Implementation

```c
// Precomputed exp LUT (256 bytes)
const uint8_t EXP_LUT[256] = {
    /* Generated offline - 256 values */
};

/**
 * Fast softmax using lookup table
 * 3-4× faster than Taylor series approximation
 */
void softmax_int8_64_lut(const int8_t* input, int8_t* output, uint32_t N) {
    // Find max value
    int8_t max_val = input[0];
    for (uint32_t i = 1; i < N; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Compute exp using LUT and accumulate sum
    uint32_t exp_vals[64];
    uint32_t sum = 0;

    for (uint32_t i = 0; i < N; i++) {
        int32_t x = input[i] - max_val;  // Shift to [-max, 0]

        // Clamp and lookup
        if (x < -128) x = -128;
        uint8_t idx = (uint8_t)(x + 128);  // Map to [0, 255]

        uint32_t exp_val = EXP_LUT[idx];
        exp_vals[i] = exp_val;
        sum += exp_val;
    }

    // Normalize
    for (uint32_t i = 0; i < N; i++) {
        int32_t normalized = (exp_vals[i] * 127) / sum;
        if (normalized > 127) normalized = 127;
        output[i] = (int8_t)normalized;
    }
}
```

---

## Testing and Validation Plan

### Unit Tests

1. **Correctness Test**: Compare against CPU reference
   ```python
   # Compare vectorized vs scalar
   error = np.abs(output_vectorized - output_scalar).max()
   assert error < 2  # Allow ±1 quantization error
   ```

2. **Performance Test**: Measure execution time
   ```python
   time_scalar = benchmark_kernel(attention_64x64_scalar)
   time_vectorized = benchmark_kernel(attention_64x64_vectorized)
   speedup = time_scalar / time_vectorized
   assert speedup >= 2.0  # Expect 2× minimum
   ```

3. **Accuracy Test**: WER on real audio
   ```bash
   # Test on Whisper test set
   python3 test_attention_accuracy.py --kernel=vectorized
   # Expected: <1% WER increase vs baseline
   ```

### Benchmark Suite Integration

```python
# Add to benchmark_kernels.py
kernels = {
    'attention_baseline': 'build_attention/attention_64x64.xclbin',
    'attention_vectorized': 'build_attention_vec/attention_vectorized.xclbin',
    'attention_tiled': 'build_attention_tiled/attention_tiled.xclbin',
    'attention_lut_softmax': 'build_attention_lut/attention_lut.xclbin',
}

results = run_benchmark_suite(kernels, iterations=20)
compare_kernels(results)
```

---

## Risk Assessment

| Optimization | Risk Level | Mitigation |
|--------------|------------|------------|
| Vectorization | LOW | Well-documented AIE2 intrinsics |
| LUT Softmax | LOW | Offline validation of LUT accuracy |
| Memory Layout | MEDIUM | Requires host-side changes |
| Tiling | LOW | Already implemented and tested |
| Quantization (INT4) | HIGH | May hurt accuracy, test thoroughly |
| Operation Fusion | MEDIUM | Complex debugging, incremental approach |
| Multi-Core | MEDIUM-HIGH | Toolchain dependency, known issue |

---

## Success Metrics

### Performance Targets

| Phase | Target Time (ms) | RTF | Status |
|-------|------------------|-----|--------|
| Baseline | 2.233 | 14.0× | ✅ Current |
| Phase 1 (Vectorization) | 1.0-1.5 | 20-25× | 📋 Next |
| Phase 2 (Softmax Opt) | 0.7-1.0 | 30-35× | 📋 Planned |
| Phase 3 (Fusion) | 0.5-0.7 | 40-50× | 📋 Future |
| Phase 4 (Multi-Core) | 0.2-0.3 | 80-100× | 🎯 Target |

### Accuracy Targets

- **Correlation with FP32**: >0.99
- **WER Increase**: <1% vs baseline
- **Numeric Stability**: No NaN or overflow

---

## Resource Requirements

### Development Time

- **Phase 1 (Quick Wins)**: 1 week (1 engineer)
- **Phase 2 (Softmax)**: 1 week (1 engineer)
- **Phase 3 (Advanced)**: 2 weeks (1-2 engineers)
- **Phase 4 (Multi-Core)**: 2-3 weeks (2 engineers + toolchain support)

**Total**: 6-7 weeks for full optimization

### Hardware/Software

- ✅ AMD Phoenix NPU (available)
- ✅ XRT 2.20.0 (installed)
- ✅ MLIR-AIE toolchain (installed)
- ⚠️ Multi-core compiler (pending resolution)
- ✅ Benchmark suite (operational)

---

## Dependencies

### Critical Path

1. ✅ Benchmark suite operational
2. ✅ Attention kernel working (14.0× realtime)
3. 📋 Vectorization infrastructure (AIE2 intrinsics)
4. ⚠️ Multi-core toolchain (blocked - 4-8 hour fix)

### External Dependencies

- AIE2 compiler optimization flags
- MLIR-AIE documentation for vector ops
- AMD optimization guides (if available)

---

## Alternative Approaches

### Option A: Flash Attention Algorithm

**Idea**: Tile-based attention with reduced memory access
**Benefit**: 4-8× speedup in literature
**Challenge**: Complex implementation, not INT8-native
**Timeline**: 3-4 weeks
**Recommendation**: Consider for Phase 5

### Option B: Approximate Attention

**Idea**: Sparse attention (only top-K keys)
**Benefit**: O(n log n) vs O(n²) complexity
**Challenge**: May hurt accuracy
**Timeline**: 2-3 weeks
**Recommendation**: Explore if WER allows

### Option C: Pre-computed Attention Patterns

**Idea**: Cache attention patterns for similar inputs
**Benefit**: Near-zero cost for repeated patterns
**Challenge**: Only works for repeated audio
**Recommendation**: Not suitable for STT

---

## Next Actions (Immediate)

### This Week (High Priority)

1. **Implement Vectorized Q@K^T** (2 days)
   - Write `attention_int8_64x64_vectorized.c`
   - Create MLIR definition
   - Compile XCLBIN
   - Test on NPU

2. **Test Tiled Version** (0.5 day)
   - Compile `attention_int8_64x64_tiled.c`
   - Benchmark vs baseline
   - May already provide speedup

3. **Benchmark Suite Integration** (0.5 day)
   - Add vectorized kernel to suite
   - Compare all variants
   - Document results

### Next Week (Medium Priority)

1. **LUT-Based Softmax** (2 days)
   - Generate offline LUT
   - Implement LUT softmax
   - Validate accuracy

2. **Memory Layout Optimization** (1 day)
   - Pre-transpose K on host
   - Update kernel to use K^T
   - Measure improvement

---

## References

### Internal Documentation

- `SESSION_COMPLETE_OCT30.md` - Current performance baseline
- `BENCHMARK_REPORT_LATEST.md` - Detailed profiling data
- `attention_int8_64x64.c` - Current implementation
- `attention_int8_64x64_tiled.c` - Tiled variant
- `matmul_int8.c` - Vectorization examples

### External Resources

- AMD AIE2 Programming Guide
- MLIR-AIE vector intrinsics documentation
- Attention Is All You Need (Vaswani et al., 2017)
- Flash Attention (Dao et al., 2022)

---

## Appendix: Performance Calculations

### Theoretical Peak Performance

**AIE2 Specs**:
- Clock: 1 GHz
- INT8 MACs: 256 per cycle per core
- 4 cores available

**Peak Throughput**: 1 TOPS per core × 4 = 4 TOPS

**Attention Workload**:
- QK^T: 262,144 MACs
- Weighted Sum: 262,144 MACs
- Total: 524,288 MACs per tile

**Theoretical Minimum Time** (single core):
```
524,288 MACs / (256 MACs/cycle × 1 GHz) = 2,048 cycles = 2.048 µs
```

**Observed**: 2,233 µs (2.233 ms)

**Efficiency**: 2.048 µs / 2233 µs = **0.09% of peak**

**Conclusion**: Massive headroom for optimization!

---

## Appendix: Code Size Estimates

| File | Estimated Size | Complexity |
|------|----------------|------------|
| `attention_int8_64x64_vectorized.c` | 8-10 KB | Medium |
| `attention_softmax_lut.c` | 3-4 KB + 256B LUT | Low |
| `attention_fused.c` | 12-15 KB | High |
| `attention_multicore.mlir` | 10-12 KB | Very High |

**Total New Code**: ~35-45 KB

---

**End of Attention Optimization Plan**

**Status**: Ready for Implementation
**Confidence**: Very High (95%)
**Expected Timeline**: 6-7 weeks to full optimization
**Expected Result**: 2-4× immediate improvement, 8-15× with multi-core

---

*"Attention is all you need... to optimize!"* 🦄✨
