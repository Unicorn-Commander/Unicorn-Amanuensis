# Phoenix NPU Memory Optimization - Documentation Index

**Project**: Whisper Base Encoder - 220x Realtime Transcription
**Hardware**: AMD Ryzen AI Phoenix NPU (XDNA1)
**Date**: November 18, 2025
**Status**: Strategy Complete, Implementation Ready

---

## Quick Start Guide

**New to this project?** Start here:

1. 📄 Read **[EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md)** (15 min)
   - Understand the challenge and solution
   - See performance targets and timeline
   - Review success criteria

2. 📊 Browse **[ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md)** (10 min)
   - Visual understanding of tile layout
   - See data flow and memory hierarchy
   - Understand multi-column coordination

3. 📋 Reference **[MEMORY_OPTIMIZATION_QUICK_REFERENCE.md](./MEMORY_OPTIMIZATION_QUICK_REFERENCE.md)** (5 min)
   - Quick lookup for key numbers
   - Batch sizes and tile configurations
   - Bandwidth calculations

4. 📚 Deep dive **[ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md](./ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md)** (60 min)
   - Complete technical strategy
   - Detailed analysis of all approaches
   - Implementation pseudocode

---

## Document Hierarchy

### 📋 Executive Level (Management & Decision Makers)

**[EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md)** (483 lines, 16 KB)
- **Audience**: Managers, project leads, stakeholders
- **Reading Time**: 15 minutes
- **Content**:
  - Problem statement and impact
  - Solution overview and results
  - Risk assessment
  - Resource requirements
  - Timeline and success criteria
  - Next actions

**Use When**: Making go/no-go decisions, resource allocation, progress reporting

---

### 🎨 Visual Reference (Engineers & Architects)

**[ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md)** (612 lines, 47 KB)
- **Audience**: Engineers, architects, implementation team
- **Reading Time**: 10-15 minutes
- **Content**:
  - 10 comprehensive ASCII diagrams
  - Phoenix NPU tile layout
  - Memory hierarchy
  - Weight tiling strategy
  - Data flow timeline
  - Batch processing with double-buffering
  - Multi-column parallel execution
  - Bandwidth comparison
  - Power vs performance trade-off
  - Complete system architecture

**Use When**: Designing MLIR code, debugging memory issues, explaining architecture

---

### 📖 Technical Strategy (Implementation Team)

**[ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md](./ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md)** (1090 lines, 44 KB)
- **Audience**: MLIR developers, kernel engineers, NPU specialists
- **Reading Time**: 60 minutes (comprehensive)
- **Content**:
  - **Step 1**: Whisper encoder memory requirements (detailed analysis)
  - **Step 2**: Weight caching strategies (3 approaches compared)
  - **Step 3**: Optimal batch size calculation (constraints & derivation)
  - **Step 4**: Multi-tile coordination strategy (3 approaches analyzed)
  - **Step 5**: Implementation design (MLIR code examples)
  - **Step 6**: Bandwidth analysis (4 approaches compared)
  - **Step 7**: Performance projection (with optimizations)

**Use When**: Implementing MLIR kernels, optimizing memory layout, bandwidth analysis

---

### 📝 Quick Reference (Daily Use)

**[MEMORY_OPTIMIZATION_QUICK_REFERENCE.md](./MEMORY_OPTIMIZATION_QUICK_REFERENCE.md)** (382 lines, 12 KB)
- **Audience**: All team members
- **Reading Time**: 5 minutes
- **Content**:
  - The Problem (one-paragraph summary)
  - The Solution (key insight)
  - Quick Numbers (tables & formulas)
  - Optimal Configuration (batch size, tiling, multi-column)
  - Bandwidth Analysis (comparison table)
  - Processing Pipeline (high-level flow)
  - Memory Layout (per-tile breakdown)
  - MLIR Implementation Checklist
  - Power Consumption
  - Key Decisions Summary
  - Files & Locations
  - Next Steps

**Use When**: Quick lookups during coding, referencing key numbers, checking configurations

---

### 🎯 Status & Progress (Tracking)

**[SUCCESS_REPORT.md](./SUCCESS_REPORT.md)** (360 lines, 9.3 KB)
- **Audience**: All team members, stakeholders
- **Reading Time**: 10 minutes
- **Content**:
  - Softmax kernel compilation success
  - NPU execution validation
  - Performance results (1.565 ms, 99.5% accuracy)
  - Debug journey and lessons learned
  - Proof that MLIR-AIE toolchain works
  - XRT kernel invocation pattern (critical!)
  - Next steps for additional kernels

**Use When**: Verifying toolchain works, understanding kernel execution pattern

---

### 📚 Kernel Inventory (Reference)

**[README.md](./README.md)** (315 lines, 9.5 KB)
- **Audience**: Kernel developers
- **Reading Time**: 10 minutes
- **Content**:
  - Kernel inventory (4 kernels)
  - Softmax (vectorized BF16)
  - GELU optimized (tanh approximation BF16)
  - SwiGLU (modern activation)
  - Softmax BF16 (scalar high-precision)
  - Compilation instructions
  - Performance targets
  - Testing and validation
  - Known issues and limitations
  - XDNA1-specific notes

**Use When**: Understanding available kernels, compilation process, performance expectations

---

## Document Purpose Matrix

| Document | Strategy | Implementation | Reference | Status |
|----------|----------|----------------|-----------|--------|
| **EXECUTIVE_SUMMARY.md** | ✅ High-level | ❌ | ✅ Quick | ✅ Complete |
| **ARCHITECTURE_DIAGRAMS.md** | ✅ Visual | ✅ Design aid | ✅ Visual | ✅ Complete |
| **ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md** | ✅ Complete | ✅ Detailed | ❌ | ✅ Complete |
| **MEMORY_OPTIMIZATION_QUICK_REFERENCE.md** | ❌ | ❌ | ✅ Daily use | ✅ Complete |
| **SUCCESS_REPORT.md** | ❌ | ✅ Pattern | ❌ | ✅ Validated |
| **README.md** | ❌ | ✅ Compilation | ✅ Kernels | ✅ Complete |

---

## Reading Paths by Role

### 👔 Manager / Project Lead

**Goal**: Understand feasibility, timeline, and resource requirements

1. **[EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md)** - Complete overview
2. **[ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md)** - Diagram 8 (bandwidth comparison)
3. **[MEMORY_OPTIMIZATION_QUICK_REFERENCE.md](./MEMORY_OPTIMIZATION_QUICK_REFERENCE.md)** - Next Steps section

**Time Investment**: 20 minutes
**Outcome**: Can make informed decisions about project approval and resources

---

### 🏗️ System Architect

**Goal**: Understand overall design and integration points

1. **[ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md)** - All diagrams
2. **[ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md](./ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md)** - Steps 1-4
3. **[EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md)** - Technical Architecture section

**Time Investment**: 60 minutes
**Outcome**: Can design MLIR architecture and buffer management

---

### 💻 MLIR Developer

**Goal**: Implement kernels and memory management

1. **[SUCCESS_REPORT.md](./SUCCESS_REPORT.md)** - Working kernel example
2. **[ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md](./ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md)** - Step 5 (Implementation Design)
3. **[ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md)** - Diagrams 1-6
4. **[MEMORY_OPTIMIZATION_QUICK_REFERENCE.md](./MEMORY_OPTIMIZATION_QUICK_REFERENCE.md)** - MLIR checklist

**Time Investment**: 90 minutes initial, then reference as needed
**Outcome**: Can write MLIR ObjectFIFO configurations and runtime sequences

---

### 🔬 Kernel Engineer

**Goal**: Compile C++ kernels and validate on NPU

1. **[README.md](./README.md)** - Kernel inventory and compilation
2. **[SUCCESS_REPORT.md](./SUCCESS_REPORT.md)** - Proven compilation and execution pattern
3. **[MEMORY_OPTIMIZATION_QUICK_REFERENCE.md](./MEMORY_OPTIMIZATION_QUICK_REFERENCE.md)** - Memory layout per tile

**Time Investment**: 30 minutes
**Outcome**: Can compile kernels with Peano and test on NPU

---

### 🧪 Performance Engineer

**Goal**: Benchmark and optimize

1. **[ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md](./ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md)** - Steps 3, 6, 7 (batch size, bandwidth, performance)
2. **[ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md)** - Diagrams 4, 8, 9 (timeline, bandwidth, power)
3. **[MEMORY_OPTIMIZATION_QUICK_REFERENCE.md](./MEMORY_OPTIMIZATION_QUICK_REFERENCE.md)** - Performance targets

**Time Investment**: 45 minutes
**Outcome**: Can measure and optimize bandwidth and throughput

---

### 🆕 New Team Member

**Goal**: Get up to speed quickly

1. **[EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md)** - Big picture
2. **[ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md)** - Visual understanding
3. **[SUCCESS_REPORT.md](./SUCCESS_REPORT.md)** - Proof of concept
4. **[MEMORY_OPTIMIZATION_QUICK_REFERENCE.md](./MEMORY_OPTIMIZATION_QUICK_REFERENCE.md)** - Daily reference

**Time Investment**: 40 minutes
**Outcome**: Understands problem, solution, and current status

---

## Key Numbers Quick Lookup

### Memory

| Metric | Value |
|--------|-------|
| Whisper Base weights | 37.51 MB (BF16) |
| Phoenix NPU on-chip | 768 KB (24 tiles) |
| Per tile (compute) | 32 KB |
| Per tile (MemTile) | 64 KB |
| Challenge | 50× overflow |

### Performance

| Metric | Value |
|--------|-------|
| Target realtime factor | 220× |
| Target latency | 4.5 ms per second audio |
| Target power | <10W |
| Target bandwidth | <5 GB/s sustained |

### Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 8 frames |
| Weight tile size | 128-512 KB |
| Columns used | 4 (parallel) |
| Reuse factor | 12.5× per tile |

---

## Implementation Checklist

### Phase 1: Compilation ⏳

- [x] Copy kernels from XDNA2 source
- [x] Analyze memory requirements
- [x] Design optimization strategy
- [ ] **Get Peano compiler access** ← BLOCKER
- [ ] Compile softmax kernel
- [ ] Compile GELU kernel
- [ ] Compile matmul kernel
- [ ] Generate XCLBINs

### Phase 2: Single-Column Test ⏳

- [ ] Load XCLBIN to NPU
- [ ] Test with 8-frame batch
- [ ] Validate accuracy (>99%)
- [ ] Measure bandwidth (<2 GB/s)
- [ ] Measure latency

### Phase 3: Multi-Column Integration ⏳

- [ ] Implement 4-column MLIR
- [ ] Test weight splitting
- [ ] Test result gathering
- [ ] Benchmark 4× speedup
- [ ] Validate bandwidth (<5 GB/s)

### Phase 4: Full Layer ⏳

- [ ] Self-attention mechanism
- [ ] Feed-forward network
- [ ] Layer normalization
- [ ] End-to-end layer test
- [ ] Accuracy validation (<1% WER)

### Phase 5: Multi-Layer Pipeline ⏳

- [ ] Loop over 6 layers
- [ ] Intermediate storage
- [ ] End-to-end encoder
- [ ] Performance: 120-160× realtime

### Phase 6: Optimization ⏳

- [ ] Kernel fusion
- [ ] INT8 quantization (optional)
- [ ] DMA pipelining
- [ ] Final tuning
- [ ] **Target: 220× realtime achieved** ✅

---

## File Sizes and Statistics

| File | Lines | Size | Estimated Reading Time |
|------|-------|------|----------------------|
| EXECUTIVE_SUMMARY.md | 483 | 16 KB | 15 min |
| ARCHITECTURE_DIAGRAMS.md | 612 | 47 KB | 15 min |
| ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md | 1090 | 44 KB | 60 min |
| MEMORY_OPTIMIZATION_QUICK_REFERENCE.md | 382 | 12 KB | 5 min |
| SUCCESS_REPORT.md | 360 | 9.3 KB | 10 min |
| README.md | 315 | 9.5 KB | 10 min |
| **Total** | **3242** | **137.8 KB** | **115 min** |

---

## Version History

### v1.0 (November 18, 2025) - Initial Release

**Created**:
- `ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md` - Complete technical strategy
- `MEMORY_OPTIMIZATION_QUICK_REFERENCE.md` - Quick lookup guide
- `ARCHITECTURE_DIAGRAMS.md` - 10 visual diagrams
- `EXECUTIVE_SUMMARY.md` - Executive overview
- `INDEX.md` - This navigation document

**Updated**:
- `README.md` - Added references to new strategy docs
- `SUCCESS_REPORT.md` - Already existed (softmax validation)

**Status**: Strategy phase complete, implementation ready

---

## Contact & Support

**Project**: Unicorn Amanuensis NPU Acceleration
**Organization**: Magic Unicorn Unconventional Technology & Stuff Inc.
**Hardware**: AMD Ryzen AI Phoenix NPU (XDNA1)
**Repository**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1/`

**Key Contributors**:
- On-Chip Memory Optimization Architect (strategy & design)
- NPU Kernel Development Team (softmax validation)
- MLIR Integration Team (toolchain setup)

**Questions?**
- Technical: See relevant document sections
- Implementation: Start with `SUCCESS_REPORT.md` for working example
- Strategy: See `ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md`
- Quick lookup: Use `MEMORY_OPTIMIZATION_QUICK_REFERENCE.md`

---

## Additional Resources

### External Documentation

- **AMD XDNA1 Documentation**: Phoenix NPU architecture
- **MLIR-AIE Documentation**: Kernel development guide
- **XRT Documentation**: Runtime API reference
- **Whisper Paper**: "Robust Speech Recognition via Large-Scale Weak Supervision" (OpenAI, 2022)

### Related Files in Repository

```
../                                 # Parent directory
├── passthrough_complete.mlir       # Validated test kernel (working!)
├── attention_int8_64x64_tiled.c    # Existing attention kernel
├── matmul_int8_64x64.c             # Existing matmul kernel
└── gelu_int8.c                     # Existing GELU kernel

./                                  # This directory (kernels_xdna1)
├── softmax_bf16_xdna1.cc           # BF16 softmax kernel (ready to compile)
├── gelu_optimized_xdna1.cc         # BF16 GELU kernel (ready to compile)
└── softmax_xdna1.cc                # Vectorized softmax (ready to compile)
```

---

## Document Maintenance

**Update Frequency**: After each major phase completion

**Responsibility**:
- Strategy docs: Architecture team
- Success reports: Implementation team
- Quick reference: All contributors
- This index: Documentation lead

**Version Control**: All documents tracked in git

---

## Glossary of Terms

| Term | Definition |
|------|------------|
| **Phoenix NPU** | AMD Ryzen AI NPU (XDNA1 architecture) with 24 AIE2 tiles |
| **Tile** | Compute unit in NPU array (32 KB memory + processing) |
| **MemTile** | Memory tile (64 KB) for buffering (Row 1 in Phoenix) |
| **ShimNOC** | DMA controller tile (Row 0 in Phoenix) |
| **ObjectFIFO** | MLIR data movement abstraction (modern, recommended) |
| **XCLBIN** | Compiled NPU binary (loaded by XRT runtime) |
| **BF16** | Bfloat16 (16-bit floating point, 8-bit exponent) |
| **Weight Tile** | Subset of weight matrix that fits in tile memory |
| **Frame Batch** | Group of audio frames processed together (8 frames) |
| **Realtime Factor** | Speed multiplier vs real time (220× = 220× faster) |
| **PCIe 4.0 x4** | Interface between host CPU and NPU (16 GB/s theoretical) |

---

**INDEX Version**: 1.0
**Last Updated**: November 18, 2025
**Next Review**: After Phase 2 completion

**Status**: ✅ Complete and Ready for Use

---

**End of Index**
