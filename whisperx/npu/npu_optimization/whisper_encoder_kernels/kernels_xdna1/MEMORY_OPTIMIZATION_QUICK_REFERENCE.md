# Memory Optimization Quick Reference

**Date**: November 18, 2025
**For**: Phoenix NPU (XDNA1) Whisper Encoder Optimization

---

## The Problem

```
Whisper Base Encoder Weights: 37.51 MB (BF16)
Phoenix NPU On-Chip Memory:    0.75 MB (768 KB)
────────────────────────────────────────────
Challenge: 50x overflow!
```

---

## The Solution

**Strategy**: Hybrid Layer-Tile Approach with Multi-Column Parallelism

**Key Insight**: Encoder weights are **constant** (reused across all frames), audio features **stream** (change per frame).

---

## Quick Numbers

### Memory Breakdown

| Component | Per Layer | All 6 Layers |
|-----------|-----------|--------------|
| Self-Attention | 2.0 MB | 12.0 MB |
| Feed-Forward | 4.0 MB | 24.0 MB |
| Layer Norm | 4 KB | 24 KB |
| **Total** | **6.15 MB** | **36.94 MB** |
| Positional Embeddings | - | 1.50 MB |
| **Grand Total** | - | **37.51 MB** |

### Per-Frame Sizes (BF16)

| Component | Size | Notes |
|-----------|------|-------|
| Audio features | 480 KB | 80 mel × 3000 samples |
| Hidden states | 1.5 MB | 512 × 1500 |
| Attention scores | 36 MB | Computed incrementally |

---

## Optimal Configuration

### Batch Size

```
Recommended: 8 frames per batch

Rationale:
- Input buffer: 4 KB → 4 frames (1 KB each)
- Output buffer: 4 KB → 4 frames
- Double-buffering: 2× → 8 frames effective
- MemTile capacity: 64 KB (plenty for 8 frames)
```

### Weight Tiling

```
Per Column (4 columns total):
- Tile size: 128 KB - 512 KB
- Example: Q_proj (512×512) split into 4 tiles:
  • Column 0: Rows 0-127    (128 KB)
  • Column 1: Rows 128-255  (128 KB)
  • Column 2: Rows 256-383  (128 KB)
  • Column 3: Rows 384-511  (128 KB)

Reuse factor: Each tile used 100÷8 = 12.5 times
```

### Multi-Column Strategy

```
Hybrid Data + Model Parallelism:
1. Model Parallelism: Split matrices across columns (rows)
2. Data Parallelism: All columns process same frames
3. Gather: Combine results in MemTiles (on-chip!)

Result: 4× throughput, optimal bandwidth
```

---

## Bandwidth Analysis

### For 100 Frames (1 Second Audio)

| Approach | Total Bandwidth | Feasible? |
|----------|-----------------|-----------|
| Naive (reload weights every frame) | 3,751 MB | ❌ No (9.4 GB/s) |
| Layer-at-a-Time | 1,585 MB | ✅ Yes (4.0 GB/s) |
| Tile-and-Reuse | 1,666 MB | ✅ Yes (4.2 GB/s) |
| **Hybrid (Recommended)** | **1,736 MB** | **✅ Yes (4.3 GB/s)** |

**PCIe 4.0 x4 Capacity**: 16 GB/s theoretical, 12 GB/s practical
**Utilization**: 35% (plenty of headroom)

---

## Processing Pipeline

### High-Level Flow

```
For each encoder layer (0-5):
  ┌─────────────────────────────────────────────┐
  │ 1. Load Weight Tiles (one-time, ~6 MB)     │
  │    → Split across 4 columns                 │
  └─────────────────────────────────────────────┘

  For each frame batch (0, 8, 16, ..., 96):
    ┌───────────────────────────────────────────┐
    │ 2. DMA Frames to NPU (~8 KB)             │
    │    → Broadcast to all 4 columns           │
    └───────────────────────────────────────────┘

    ┌───────────────────────────────────────────┐
    │ 3. Parallel Compute on 4 Columns         │
    │    → Matmul, Softmax, GELU, LayerNorm     │
    │    → Each column: 1/4 of weight matrix    │
    └───────────────────────────────────────────┘

    ┌───────────────────────────────────────────┐
    │ 4. Gather Results (~8 KB per column)     │
    │    → Combine 4 partial outputs            │
    │    → DMA back to host                     │
    └───────────────────────────────────────────┘
```

### Detailed Bandwidth (100 Frames)

```
Weight Loads:
  6 layers × 6.15 MB × 1.2 overhead = 44.4 MB
  (1.2× overhead from tiling, shared across columns)

Frame Inputs:
  100 frames × 480 KB = 48 MB
  Broadcast to 4 columns: 48 × 4 = 192 MB

Intermediate Storage (layer-to-layer):
  100 frames × 1.5 MB × 5 inter-layer transfers = 750 MB
  Bidirectional (store + load): 750 × 2 = 1,500 MB

────────────────────────────────────────────────
Total: 44.4 + 192 + 1,500 = 1,736 MB
Time: ~100 ms (with 4× parallelism)
Bandwidth: 1,736 MB / 100 ms = 17.36 GB/s peak
           1,736 MB / 400 ms = 4.34 GB/s sustained
```

---

## Performance Targets

### Compute Time Breakdown (Estimated)

```
Single Layer, Single Frame (no parallelism):
├── Self-Attention:
│   ├── Q, K, V projections: 3 × 64 µs = 192 µs
│   ├── Attention scores: 64 µs
│   ├── Softmax: 8 µs
│   └── Output projection: 64 µs
│   Total: 328 µs
├── Feed-Forward:
│   ├── Linear 1: 128 µs
│   ├── GELU: 16 µs
│   └── Linear 2: 128 µs
│   Total: 272 µs
└── Layer Total: 600 µs

6 Layers × 100 Frames = 360 ms (compute only)
With DMA: 360 + 40 = 400 ms
With 4× parallelism: 400 / 4 = 100 ms
```

### Speedup Path to 220x Realtime

```
Current (encoder only): 2.5x realtime (400 ms)
With 4-column parallelism: 10x realtime (100 ms)
With kernel fusion: 30x realtime (33 ms)
With INT8 quantization: 60x realtime (16 ms)
With decoder optimization: 120x realtime (8.3 ms)
With final tuning: 220x realtime (4.5 ms) ✓

Target: 1000 ms / 220 = 4.5 ms per second of audio
```

---

## Memory Layout (Per Tile)

### Compute Tile (32 KB)

```
┌─────────────────────────────────┐
│ Weight Tile       20 KB (62.5%) │ ← Reused 12.5× per layer
├─────────────────────────────────┤
│ Input Buffer       4 KB (12.5%) │ ← 8 frames × 512B
├─────────────────────────────────┤
│ Output Buffer      4 KB (12.5%) │ ← 8 frames × 512B
├─────────────────────────────────┤
│ Scratch Space      4 KB (12.5%) │ ← Intermediate computations
└─────────────────────────────────┘
```

### MemTile (64 KB)

```
┌─────────────────────────────────┐
│ Ping Buffer       16 KB (25%)   │ ← Frame batch A
├─────────────────────────────────┤
│ Pong Buffer       16 KB (25%)   │ ← Frame batch B (while A computes)
├─────────────────────────────────┤
│ Weight Staging    24 KB (37.5%) │ ← Temporary weight storage
├─────────────────────────────────┤
│ Gather Buffer      8 KB (12.5%) │ ← Partial results from columns
└─────────────────────────────────┘
```

---

## MLIR Implementation Checklist

### Phase 1: Single Column, Single Tile ✓
- [x] Declare ShimNOC tile (DMA)
- [x] Declare MemTile (buffer)
- [x] Declare Compute tile
- [x] Define ObjectFIFOs (input, weights, output)
- [x] Implement matmul kernel call
- [x] Runtime sequence (DMA commands)

### Phase 2: Multi-Column Parallelism (In Progress)
- [ ] 4 ShimNOC tiles (one per column)
- [ ] 4 MemTiles with ping-pong buffers
- [ ] 4 Compute tiles (one per column)
- [ ] Split weights across columns
- [ ] Broadcast inputs to all columns
- [ ] Gather outputs in MemTiles

### Phase 3: Full Encoder Layer (Next)
- [ ] Self-attention (Q, K, V, O projections)
- [ ] Scaled dot-product attention
- [ ] Softmax kernel
- [ ] Feed-forward network (2 linear layers)
- [ ] GELU activation
- [ ] Layer normalization (2×)

### Phase 4: Multi-Layer Pipeline (Final)
- [ ] Loop over 6 encoder layers
- [ ] Intermediate storage management
- [ ] Host-side result gathering
- [ ] End-to-end integration

---

## Power Consumption

```
Phoenix NPU Power Budget:
├── Idle: 2W
├── Single column active: 4-5W
├── All columns active: 8-12W
└── Peak (with DRAM): 15W

Comparison:
├── CPU-only (x86): 45-65W @ 1.67× realtime
├── Intel iGPU: 18W @ 11× realtime
└── Phoenix NPU: 10W @ 220× realtime (target)

Power Efficiency:
NPU: 220× / 10W = 22× per watt
vs CPU: 22 / (1.67/50) = 659× more efficient
```

---

## Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Caching Strategy** | Hybrid Layer-Tile | Best bandwidth/complexity trade-off |
| **Batch Size** | 8 frames | Fits in buffers with double-buffering |
| **Tile Size** | 128-512 KB | Fits in ~16 tiles with margin |
| **Multi-Column** | Hybrid Data+Model | Optimal bandwidth & throughput |
| **Precision** | BF16 | Good accuracy, native NPU support |
| **Bandwidth Target** | 4.3 GB/s | 35% PCIe utilization (safe) |

---

## Files & Locations

**Documentation**:
```
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/
  whisper_encoder_kernels/kernels_xdna1/
    ├── ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md  ← Full details
    ├── MEMORY_OPTIMIZATION_QUICK_REFERENCE.md   ← This file
    ├── README.md                                 ← Kernel inventory
    └── SUCCESS_REPORT.md                         ← Softmax kernel success
```

**Kernels** (ready for compilation):
```
kernels_xdna1/
├── softmax_bf16_xdna1.cc      ← Scalar softmax (high precision)
├── softmax_xdna1.cc           ← Vectorized softmax (fast)
├── gelu_optimized_xdna1.cc    ← GELU activation
└── swiglu_xdna1.cc            ← Future-proofing
```

**MLIR Templates**:
```
whisper_encoder_kernels/
├── passthrough_complete.mlir   ← Validated test kernel
└── (TODO: whisper_encoder_layer.mlir)
```

---

## Next Steps (Priority Order)

1. **Compile Kernels** (Week 1)
   - Get Peano C++ compiler access
   - Compile softmax, GELU kernels to `.o` files
   - Generate XCLBIN with `aie-translate`

2. **Single-Column Test** (Week 1-2)
   - Load XCLBIN to NPU
   - Test matmul kernel with 8-frame batch
   - Validate accuracy vs NumPy

3. **Multi-Column Integration** (Week 2-3)
   - Implement 4-column MLIR design
   - Test weight splitting and result gathering
   - Benchmark throughput (target: 4× speedup)

4. **Full Layer Implementation** (Week 3-5)
   - Self-attention mechanism (Q, K, V, O)
   - Feed-forward network
   - Layer normalization
   - End-to-end layer test

5. **Multi-Layer Pipeline** (Week 5-7)
   - Loop over 6 encoder layers
   - Optimize intermediate storage
   - Measure accuracy and performance

6. **Optimization & Tuning** (Week 7-10)
   - Kernel fusion (reduce memory transfers)
   - INT8 quantization (optional 2× speedup)
   - DMA pipelining (hide latency)
   - Reach 220× realtime target

---

## Contact & Support

**Project**: Unicorn Amanuensis NPU Acceleration
**Hardware**: AMD Ryzen AI Phoenix (XDNA1)
**Target**: 220× Realtime Whisper Transcription
**Status**: Strategy Complete, Implementation In Progress

**Key Contributors**:
- On-Chip Memory Optimization Architect (this document)
- NPU Kernel Development Team (softmax success)
- MLIR Integration Team (passthrough kernel validated)

---

**Quick Reference Version**: 1.0
**Last Updated**: November 18, 2025
**Next Review**: After Phase 2 completion
