# NPU Kernel Development Session Update - November 20, 2025

## Executive Summary

**Critical Discovery**: NPU kernel launch overhead (~1.5ms per call) makes per-operation NPU calls counterproductive. The optimized CPU-vectorized encoder achieves **13x realtime** vs 0.4x with per-call NPU kernels.

---

## Key Findings

### 1. Baseline Performance (Per-Call NPU Kernels)

| Test | Seq Len | Encoder Time | RTF | Issue |
|------|---------|-------------|-----|-------|
| Small | 10 | 1,353 ms | 0.1x | 12,000+ kernel calls |
| Medium | 100 | 13,677 ms | 0.1x | Per-row softmax |
| Standard | 500 | 68,786 ms | 0.1x | Per-chunk GELU |
| Full | 1500 | 77,843 ms | 0.4x | Massive overhead |

**Bottleneck Analysis**:
- Attention (softmax): 53% of time (per-row calls)
- GELU: 35% (512-element chunks)
- LayerNorm: 11.6% (chunked)

### 2. Optimized Performance (CPU Vectorized)

| Test | Seq Len | Encoder Time | RTF | Speedup |
|------|---------|-------------|-----|---------|
| Small | 10 | 25 ms | 4.0x | 54x |
| Medium | 100 | 134 ms | 7.5x | 102x |
| Standard | 500 | 695 ms | 7.2x | 99x |
| Full | 1500 | 2,309 ms | **13.0x** | 34x |

**Why it's faster**: NumPy's vectorized operations process entire arrays at once, avoiding per-call overhead.

### 3. Root Cause Analysis

The NPU kernels themselves work well (1.5ms per operation), but:

```
Per-layer attention requires:
- 8 heads × 1500 rows = 12,000 softmax calls
- 12,000 × 1.5ms = 18 seconds overhead alone!

With CPU vectorized:
- 1 NumPy softmax call = 100ms for entire matrix
```

---

## Implications for 220x Target

### Current State
- **CPU vectorized**: 13x RTF (best achieved)
- **Per-call NPU**: 0.4x RTF (current NPU approach)
- **Target**: 220x RTF

### Gap Analysis
We are 17x away from the 220x target. The path forward requires:

1. **Batched NPU Operations**: Design kernels that process entire attention matrices or FFN blocks at once, amortizing the 1.5ms launch overhead over millions of operations.

2. **Fused Kernels**: Chain multiple operations in a single kernel (e.g., LayerNorm → Q proj → K proj → V proj → attention → O proj as one kernel call).

3. **Custom MLIR**: The encoder_layer_simple.xclbin (3.5ms for LN→SM→GELU chain) shows the right direction - need to extend to full layer.

---

## What the Individual Kernels Prove

The individual kernel benchmarks are still valid and important:

| Kernel | Time | Throughput | Use Case |
|--------|------|------------|----------|
| LayerNorm | 0.83 ms | 1.2M/s | Good for batched 1024 elements |
| Softmax | 1.54 ms | 0.66M/s | Need batched attention matrix |
| GELU | 1.52 ms | 0.67M/s | Need batched FFN output |
| MatMul (vectorized) | 0.39 ms | 235x vs scalar | Excellent, but 64x64 only |

**The kernels are fast** - the problem is calling them 10,000+ times per layer.

---

## Recommended Path Forward

### Option 1: Full-Layer NPU Kernels (Best for 220x)

Build a single XCLBIN that executes an entire encoder layer:

```
1 kernel call = 1 encoder layer
- DMA in: hidden_states (1500×512 = 768K elements)
- Compute: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
- DMA out: hidden_states (1500×512)

Estimated time: 10-20ms per layer
6 layers × 20ms = 120ms
30s audio / 0.12s = 250x RTF ✅
```

This is achievable but requires significant MLIR development.

### Option 2: Batched Kernels (Medium Effort)

Create batched versions of existing kernels:

- **Batched Softmax**: Input (8, 1500, 1500), output (8, 1500, 1500)
- **Batched MatMul**: Process entire Q/K/V projections at once
- **Batched GELU**: Input (1500, 2048), output (1500, 2048)

Estimated: 50-100x RTF (still need multiple kernel calls)

### Option 3: Hybrid Approach (Quickest Win)

Use CPU for complex ops, NPU for expensive compute:
- CPU: LayerNorm, softmax, residuals
- NPU: MatMul (large tiled), GELU (batched)

The 512×512 matmuls (Q/K/V/O projections) could benefit from NPU if tiled properly.

---

## Files Created This Session

### Test Scripts
- `test_e2e_baseline.py` - Baseline performance measurement
- `npu_whisper_encoder_optimized.py` - CPU vectorized version

### Results
- Baseline: 0.4x RTF (77.8 seconds for 30s audio)
- Optimized: 13.0x RTF (2.3 seconds for 30s audio)
- Speedup: 34x improvement

---

## Technical Lessons Learned

### 1. Kernel Launch Overhead Matters
The 1.5ms per kernel call is acceptable for large operations (millions of elements), but catastrophic for per-row/per-chunk calls.

### 2. NumPy is Surprisingly Fast
For Whisper-base dimensions (512×512 matmuls, 1500×512 arrays), NumPy's optimized BLAS achieves good performance:
- Full encoder layer: 380ms
- Full encoder: 2.3 seconds
- 13x realtime

### 3. NPU Value Proposition
The NPU shines when:
- Single operation processes large data
- Kernel launch overhead is amortized
- Multiple operations fused into one call

The NPU does NOT shine when:
- Many small operations
- Per-element or per-row processing
- High Python→XRT→NPU call frequency

---

## Comparison to Target

| Approach | RTF | Gap to 220x | Feasibility |
|----------|-----|------------|-------------|
| Per-call NPU | 0.4x | 550x | Current (broken) |
| CPU Vectorized | 13x | 17x | Achieved |
| Batched NPU | 50-100x | 2-4x | Medium effort |
| Full-layer NPU | 200-300x | Achieved! | High effort |

---

## Conclusion

The session revealed that the current NPU kernel integration approach has fundamental overhead issues. The individual kernels are well-optimized (1.5ms each), but the integration pattern is wrong.

**Key insight**: NPU efficiency requires minimizing kernel launches, not maximizing per-operation speedup.

**Immediate recommendation**: Use the `npu_whisper_encoder_optimized.py` (13x RTF) as the production baseline while developing full-layer NPU kernels.

**Path to 220x**: Requires full-layer NPU kernels where one XCLBIN call processes an entire encoder layer. This is feasible given the 3.5ms result for the LN→SM→GELU chain, but needs significant MLIR development.

---

*Session: November 20, 2025*
*Platform: AMD Phoenix NPU (XDNA1)*
*Status: Bottleneck identified, optimized baseline achieved*
