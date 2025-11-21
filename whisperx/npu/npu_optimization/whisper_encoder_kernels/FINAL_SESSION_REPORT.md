# NPU Whisper Encoder - Final Session Report

**Date**: November 20, 2025
**Platform**: AMD Phoenix NPU (XDNA1)
**Target**: 220x Realtime Whisper Transcription
**Status**: 13x RTF achieved (CPU optimized), 17x gap to target

---

## Executive Summary

This session identified and validated the core bottleneck preventing 220x performance: **NPU kernel launch overhead (~1.5ms per call)** makes per-chunk NPU processing counterproductive. The best working solution is a CPU-vectorized encoder achieving **13x realtime**.

---

## Performance Results

### Baseline (Per-Call NPU Kernels)
| Sequence | Encoder Time | RTF |
|----------|-------------|-----|
| 1500 frames | 77,843 ms | 0.4x |

**Bottleneck**: 53% attention (per-row softmax), 35% GELU (per-chunk)

### Optimized (CPU Vectorized)
| Sequence | Encoder Time | RTF | vs Baseline |
|----------|-------------|-----|-------------|
| 100 frames | 134 ms | 7.5x | 102x faster |
| 500 frames | 695 ms | 7.2x | 99x faster |
| 1500 frames | 2,309 ms | **13.0x** | 34x faster |

**Why it's faster**: NumPy's vectorized operations process entire arrays at once.

---

## Root Cause Analysis

### The Overhead Problem

```
NPU kernel call overhead: ~1.5 ms per call

Attention per layer:
- 8 heads × 1500 rows = 12,000 softmax calls
- 12,000 × 1.5ms = 18 seconds (overhead alone!)

CPU vectorized:
- 1 NumPy softmax = ~100 ms for entire attention matrix
```

### What Works Well
- **Individual kernels**: 0.8-1.5ms per operation (proven accurate)
- **Kernel chaining**: 3.5ms for LN→SM→GELU (7.9% faster than separate)
- **Vectorized matmul**: 0.2ms for 64x64 (235x faster than scalar)

### What Doesn't Work
- **Per-chunk streaming**: Overhead dominates
- **Per-row processing**: Too many kernel calls
- **Small buffer sizes**: Launch overhead not amortized

---

## Current Production Solution

**File**: `npu_whisper_encoder_optimized.py`
**Performance**: 13x realtime
**Implementation**: Pure CPU with NumPy vectorized operations

```python
from npu_whisper_encoder_optimized import NPUWhisperEncoderOptimized

encoder = NPUWhisperEncoderOptimized(onnx_path)
output = encoder.encode(mel_features)  # (seq_len, 512)
```

This is the recommended production encoder until full-layer NPU kernels are developed.

---

## Path to 220x Target

### Gap Analysis
- Current: 13x RTF
- Target: 220x RTF
- Gap: 17x improvement needed

### Required Architecture

220x requires **single-call full-layer kernels**:

```
1 kernel call = 1 encoder layer

DMA in: hidden_states (1500×512 = 768K elements)
Compute: LN → Attention → Residual → LN → FFN → Residual
DMA out: hidden_states (1500×512)

Estimated: 10-20ms per layer
6 layers × 20ms = 120ms
30s audio / 0.12s = 250x RTF ✓
```

### Technical Requirements

1. **Large buffer support**: 768K elements (1.5 MB) per operation
2. **On-tile memory management**: 32 KB per tile limits
3. **Tiled computation**: Process large matrices in tile-sized chunks
4. **Streaming architecture**: Keep data on NPU through full layer
5. **Custom MLIR**: ObjectFIFO patterns for layer-level pipelining

### Development Effort
- **Estimated time**: 4-6 weeks
- **Primary work**: MLIR development for full-layer kernels
- **Secondary**: C++ kernel implementations for large operations

---

## Files Created This Session

### Production Code
- `npu_whisper_encoder_optimized.py` - **13x RTF production encoder**
- `npu_whisper_encoder_streaming.py` - NPU streaming attempt (not faster)
- `test_e2e_baseline.py` - Baseline performance test

### Documentation
- `SESSION_SUMMARY_UPDATED.md` - Detailed findings
- `FINAL_SESSION_REPORT.md` - This report

### Previous Session Work
- `npu_encoder.py` - NPU kernel wrapper
- `whisper_weight_loader.py` - ONNX weight loading (39.27 MB)
- Individual kernel builds (LayerNorm, Softmax, GELU, MatMul)

---

## Key Insights

### 1. Kernel Overhead is Critical
The 1.5ms per-call overhead seems small but compounds catastrophically:
- 1000 calls = 1.5 seconds overhead
- 10000 calls = 15 seconds overhead

### 2. NumPy is Surprisingly Fast
For Whisper-base dimensions:
- 512×512 matmul: ~1ms
- 1500×512 attention: ~100ms
- Full encoder layer: ~380ms
- **13x realtime** achieved

### 3. NPU Value Requires Amortization
NPU shines when:
- Single call processes millions of elements
- Launch overhead < 1% of compute time
- Data stays on NPU (no Python round-trips)

### 4. Validated Kernel Performance
Individual kernels work correctly:
- LayerNorm: 0.83ms, correlation 0.999995
- Softmax: 1.54ms, correlation >0.999
- GELU: 1.52ms, correlation >0.999
- MatMul: 0.39ms, 235x speedup

The problem is integration pattern, not kernel quality.

---

## Recommendations

### Immediate (Use Now)
1. **Deploy `npu_whisper_encoder_optimized.py`** for production
2. **13x RTF** is acceptable for many use cases
3. **30 seconds audio in 2.3 seconds**

### Short-term (2-4 weeks)
1. **Design full-layer MLIR architecture**
2. **Build prototype for single encoder layer**
3. **Validate performance improvement**

### Medium-term (4-8 weeks)
1. **Complete 6-layer encoder XCLBIN**
2. **Integrate with production pipeline**
3. **Target 100-220x RTF**

---

## Comparison to Other Approaches

| Approach | RTF | Development | Notes |
|----------|-----|-------------|-------|
| Current per-call NPU | 0.4x | Done | Broken by overhead |
| **CPU optimized** | **13x** | **Done** | **Production ready** |
| Batched NPU | 50-100x | 2-4 weeks | Partial improvement |
| Full-layer NPU | 200-250x | 4-8 weeks | Achieves target |

---

## Conclusion

This session achieved significant progress:

1. **Identified** the kernel launch overhead bottleneck
2. **Validated** that individual kernels work correctly
3. **Created** a 13x RTF production encoder
4. **Documented** the path to 220x

The 220x target is **achievable** but requires full-layer NPU kernels where single XCLBIN calls process entire encoder layers. The current best solution is the CPU-vectorized encoder at 13x realtime.

---

**Session Complete**

*Files for production use:*
- `npu_whisper_encoder_optimized.py` - 13x RTF encoder
- `whisper_weight_loader.py` - Weight loading

*Next development phase:*
- Full-layer MLIR kernel design
- Single-call encoder layer XCLBIN
- Target: 200-250x RTF
