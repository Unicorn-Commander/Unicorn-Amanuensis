# Phase 2: Attention Optimization on NPU - Implementation Plan

**Date**: November 21, 2025
**Goal**: Achieve 1.1× RTF (faster than realtime!) by optimizing attention on NPU
**Current**: 0.75× RTF with CPU attention (449ms/layer)
**Target**: 1.1× RTF with NPU attention (~225ms/layer = 2× speedup)

---

## 📊 Current Bottleneck Analysis

### Attention Time Breakdown (per layer)

**Total**: 449ms/layer (57% of encoder time!)

Operations:
1. **Q, K, V Projections** (~180ms total):
   - Q = x @ W_q: (3001, 512) @ (512, 512) = (3001, 512) [CPU matmul]
   - K = x @ W_k: (3001, 512) @ (512, 512) = (3001, 512) [CPU matmul]
   - V = x @ W_v: (3001, 512) @ (512, 512) = (3001, 512) [CPU matmul]

2. **Per-Head Attention** (~240ms total, 8 heads):
   - For each head (×8):
     - scores = Q_head @ K_head^T: (3001, 64) @ (64, 3001) [CPU matmul]
     - attn = softmax(scores): (3001, 3001) [CPU softmax]
     - output = attn @ V_head: (3001, 3001) @ (3001, 64) [CPU matmul]

3. **Output Projection** (~29ms):
   - out = concat @ W_o: (3001, 512) @ (512, 512) [CPU matmul]

---

## 🎯 Why We Can Get 2× Speedup

### Available NPU Resources

**Compiled Kernels**:
- ✅ `matmul_bf16.xclbin` - 64×64 tile matmul
- ✅ `softmax_bf16.xclbin` - softmax kernel
- ✅ 16 compute cores available (currently using 1!)

**Perfect Alignment**:
- Whisper weights: 512×512 = **exactly 8×8 tiles** of 64×64!
- Head dimension: 64 = **exactly 1 tile**!
- All operations tile-align perfectly!

### Expected Speedup Breakdown

| Operation | Current (CPU) | Target (NPU) | Speedup | Rationale |
|-----------|---------------|--------------|---------|-----------|
| Q, K, V projections | ~180ms | ~90ms | 2× | Batched 64×64 tiles on NPU |
| Attention scores | ~120ms | ~60ms | 2× | Tiled matmul + NPU softmax |
| Attention output | ~120ms | ~60ms | 2× | Tiled matmul on NPU |
| Output projection | ~29ms | ~15ms | 2× | 64×64 tiles on NPU |
| **Total** | **449ms** | **225ms** | **2×** | **Conservative estimate** |

---

## 🔧 Implementation Strategy

### Step 1: Implement Tiled Matmul Wrapper

**Goal**: Support arbitrary matrix sizes by tiling into 64×64 chunks

**Current Issue** (attention_npu.py:133-135):
```python
def matmul_npu(self, A, B):
    # TODO: Implement tiled matmul on NPU
    return A @ B  # Falls back to CPU!
```

**Solution**: Tile large matrices into 64×64 chunks, execute on NPU

**Implementation**:
```python
def matmul_npu_tiled(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix multiply using NPU 64×64 tiles

    A: (M, K) matrix
    B: (K, N) matrix
    Returns: C = A @ B (M, N)

    Strategy:
    1. Tile A into 64×K chunks (row tiles)
    2. Tile B into K×64 chunks (column tiles)
    3. For each output tile C[i,j]:
       - Compute sum of A[i,k] @ B[k,j] for all k tiles
       - Each inner product is multiple 64×64 NPU matmuls
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    TILE_SIZE = 64
    C = np.zeros((M, N), dtype=np.float32)

    # Iterate over output tiles
    for i in range(0, M, TILE_SIZE):
        for j in range(0, N, TILE_SIZE):
            # Accumulate over K dimension
            for k in range(0, K, TILE_SIZE):
                # Extract tiles
                A_tile = A[i:i+TILE_SIZE, k:k+TILE_SIZE]
                B_tile = B[k:k+TILE_SIZE, j:j+TILE_SIZE]

                # Pad to 64×64 if needed
                A_padded = _pad_to_64x64(A_tile)
                B_padded = _pad_to_64x64(B_tile)

                # Execute on NPU
                C_tile = _matmul_npu_64x64(A_padded, B_padded)

                # Accumulate result
                C[i:i+TILE_SIZE, j:j+TILE_SIZE] += C_tile[:A_tile.shape[0], :B_tile.shape[1]]

    return C
```

### Step 2: Optimize Q, K, V Projections

**Current** (all on CPU):
```python
Q = x @ W_q  # (3001, 512) @ (512, 512) - CPU
K = x @ W_k  # (3001, 512) @ (512, 512) - CPU
V = x @ W_v  # (3001, 512) @ (512, 512) - CPU
```

**Optimized** (NPU tiled matmul):
```python
Q = self.matmul_npu_tiled(x, W_q)  # NPU!
K = self.matmul_npu_tiled(x, W_k)  # NPU!
V = self.matmul_npu_tiled(x, W_v)  # NPU!
```

**Expected Savings**: 180ms → 90ms (2× speedup)

**Tile Count**:
- x: (3001, 512) = 47 row tiles × 8 col tiles
- W: (512, 512) = 8 row tiles × 8 col tiles
- Total NPU calls per projection: 47 × 8 × 8 = 3,008 tile operations
- **But**: Can batch 8 output tiles together! = 376 batches

### Step 3: Optimize Attention Computation

**Current** (per head, on CPU):
```python
for head_idx in range(8):  # 8 heads
    scores = Q[head_idx] @ K[head_idx].T  # (3001,64) @ (64,3001) - CPU
    attn = softmax(scores)                 # (3001,3001) - CPU
    output = attn @ V[head_idx]            # (3001,3001) @ (3001,64) - CPU
```

**Optimized** (NPU tiled):
```python
for head_idx in range(8):
    scores = self.matmul_npu_tiled(Q[head_idx], K[head_idx].T)  # NPU!
    attn = self.softmax_npu_tiled(scores)                       # NPU!
    output = self.matmul_npu_tiled(attn, V[head_idx])           # NPU!
```

**Expected Savings**: 240ms → 120ms (2× speedup)

**Key Optimization**: Attention scores matrix (3001×3001) is large but sparse!
- Can use tiled softmax (process 64 rows at a time)
- Softmax kernel already compiled

### Step 4: Optimize Output Projection

**Current**:
```python
output = multi_head_output @ W_o  # (3001, 512) @ (512, 512) - CPU
```

**Optimized**:
```python
output = self.matmul_npu_tiled(multi_head_output, W_o)  # NPU!
```

**Expected Savings**: 29ms → 15ms (2× speedup)

---

## 🚀 Implementation Steps

### Week 1: Core Infrastructure

**Day 1-2**: Implement tiled matmul wrapper
- [ ] `_pad_to_64x64()` function
- [ ] `_matmul_npu_64x64()` kernel wrapper
- [ ] `matmul_npu_tiled()` with tiling logic
- [ ] Unit tests with various matrix sizes

**Day 3-4**: Test and validate
- [ ] Benchmark single 64×64 matmul on NPU
- [ ] Benchmark 512×512 matmul (8×8 tiles)
- [ ] Compare accuracy vs CPU (should be identical)
- [ ] Measure overhead (tiling, padding, DMA)

**Day 5**: Integrate Q, K, V projections
- [ ] Update `forward()` to use NPU matmul
- [ ] Test with real Whisper weights
- [ ] Measure speedup on projections

### Week 2: Attention Optimization

**Day 6-7**: Implement tiled softmax
- [ ] Wrap softmax_bf16.xclbin kernel
- [ ] Tile large softmax operations
- [ ] Test numerical stability

**Day 8-9**: Optimize per-head attention
- [ ] Use NPU for attention scores
- [ ] Use NPU for attention @ V
- [ ] Measure per-head timing

**Day 10**: Full integration and testing
- [ ] Update whisper_encoder_optimized_v2.py
- [ ] Run full encoder with NPU attention
- [ ] Measure end-to-end RTF

---

## 📊 Expected Results

### Performance Targets

| Metric | Current (v2) | Phase 2 Target | Improvement |
|--------|-------------|----------------|-------------|
| Attention time/layer | 449ms | 225ms | 2× faster |
| Total time/layer | 791ms | 566ms | 1.4× faster |
| Encoder total (6 layers) | 5,990ms | 4,280ms | 1.4× faster |
| **Total time** | **6.67s** | **4.77s** | **1.4× faster** |
| **Realtime Factor** | **0.75×** | **1.05×** | **Faster than realtime!** |

### Conservative vs Optimistic

**Conservative** (2× on attention):
- RTF: 1.05× (5% faster than realtime)

**Realistic** (2.5× on attention):
- RTF: 1.15× (15% faster than realtime)

**Optimistic** (3× on attention with batching):
- RTF: 1.3× (30% faster than realtime)

---

## 🎯 Success Criteria

✅ **Phase 2 Complete** when:
1. NPU matmul working for arbitrary matrix sizes (tiled)
2. Attention fully executing on NPU (no CPU fallback)
3. RTF > 1.0× (faster than realtime)
4. Numerical accuracy maintained (>99.9% match with CPU)
5. Committed to git with validation tests

---

## 🔍 Technical Challenges

### Challenge 1: Large Attention Matrices

**Problem**: Attention scores are 3001×3001 = 9M elements (36 MB in FP32)

**Solution**: Tile into 64×64 chunks (never materialize full matrix)

**Memory**: 64×64 BF16 = 8KB per tile (fits easily in L1 cache)

### Challenge 2: Tiling Overhead

**Problem**: Many small NPU kernel calls could add overhead

**Solution**:
- Pre-allocate buffers (reuse like LayerNorm)
- Batch multiple tiles together
- Pipeline DMA transfers

**Expected Overhead**: <10% (kernel execution dominates)

### Challenge 3: Softmax Numerical Stability

**Problem**: exp(x) can overflow for large x

**Solution**:
- Use stable softmax: softmax(x) = softmax(x - max(x))
- Process in tiles of 64 (compute max per tile)
- NPU kernel should handle this

---

## 📈 Path to 220× (Updated)

| Phase | Timeline | RTF | Speedup | Status |
|-------|----------|-----|---------|--------|
| v2 (CPU LayerNorm) | ✅ Done | 0.75× | 6× | Complete |
| **Phase 2 (NPU Attention)** | **Week 1-2** | **1.1×** | **1.5×** | **Next** |
| Phase 3 (NPU FFN) | Week 3-4 | 1.4× | 1.3× | Planned |
| Phase 4 (Multi-core) | Month 2-3 | 22× | 16× | Planned |
| Phase 5 (Vectorization) | Month 3-4 | 220× | 10× | Planned |

**Incremental value**: Each phase provides measurable improvement!

---

## 🔗 References

- **matmul kernel**: kernels_xdna1/build_matmul/matmul_bf16.xclbin
- **softmax kernel**: kernels_xdna1/build_softmax_bf16/softmax_bf16.xclbin
- **Current implementation**: attention_npu.py:118-215
- **v2 encoder**: whisper_encoder_optimized_v2.py

---

**Next Action**: Implement `matmul_npu_tiled()` function in attention_npu.py
