# NPU Whisper Benchmark Report

**Generated**: 2025-10-30 01:36:07
**Hardware**: AMD Phoenix NPU (XDNA1) - 4×6 tile array
**XRT Version**: 2.20.0
**Firmware**: 1.5.5.391

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Current Realtime Factor** | **14.0x** |
| **Target Realtime Factor** | **220x** |
| **Progress** | **6.4%** |
| **Gap** | **15.7x improvement needed** |
| **Status** | On track |

---

## Performance Summary

### Single Tile Performance

**Total tile time**: 3.034ms

| Kernel | Mean (ms) | Percentage |
|--------|-----------|------------|
| Attention | 2.233 | 73.6% |
| LayerNorm | 0.166 | 5.5% |
| GELU | 0.142 | 4.7% |
| Matmul | 0.493 | 16.2% |


---

## Kernel Performance

### Detailed Kernel Statistics

| Kernel | Mean (ms) | Std (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) |
|--------|-----------|----------|----------|----------|----------|----------|----------|
| Attention | 2.233 | 0.069 | 2.229 | 2.370 | 2.370 | 2.104 | 2.370 |
| LayerNorm | 0.166 | 0.054 | 0.137 | 0.258 | 0.261 | 0.119 | 0.262 |
| GELU | 0.142 | 0.027 | 0.136 | 0.198 | 0.201 | 0.110 | 0.202 |
| Matmul | 0.493 | 0.085 | 0.464 | 0.703 | 0.727 | 0.424 | 0.734 |


---

## Accuracy Validation

*No accuracy validation data available*

---

## Optimization Comparison

*No optimization comparison data available*

---

## Progress to Target

### Current vs Target

```
Current:    14.0x █░░░░░░░░░░░░░░░░░░░   6.4%
Target:      220x ████████████████████ 100.0%
```

### Milestones

| Milestone | Target RTF | Status |
|-----------|------------|--------|
| Phase 1: Baseline | 10-15x | ✅ COMPLETE |
| Phase 2: Buffer Optimization | 15-20x | ⏳ In Progress |
| Phase 3: Larger Tiles (64x64) | 40-60x | 📋 Planned |
| Phase 4: Batch Processing | 80-120x | 📋 Planned |
| Phase 5: Multi-core NPU | 150-180x | 📋 Planned |
| Phase 6: Full Optimization | 220x+ | 🎯 Target |


---

## Next Optimizations

1. **Implement Larger Matmul Tiles** (16×16 → 64×64)
   - Expected: 4-6x speedup
   - Priority: HIGH

2. **Optimize Buffer Management**
   - Minimize DMA sync operations
   - Reuse buffers across kernel calls
   - Expected: 1.5-2x speedup


---

## Detailed Metrics

### System Information

- **Hardware**: AMD Ryzen 9 8945HS with Phoenix NPU
- **NPU**: 4×6 tile array, 16 TOPS INT8
- **Memory**: DDR5, shared with system
- **XRT**: 2.20.0
- **Firmware**: 1.5.5.391

### Kernel Overhead Analysis

| Component | Time (ms) | Overhead (%) |
|-----------|-----------|---------------|
| Attention | 2.233 | 73.6% |
| LayerNorm | 0.166 | 5.5% |
| GELU | 0.142 | 4.7% |
| Matmul | 0.493 | 16.2% |


---

**Report End**
