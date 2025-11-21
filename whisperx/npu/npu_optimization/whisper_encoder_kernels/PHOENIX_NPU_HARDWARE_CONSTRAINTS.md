# AMD Phoenix NPU (XDNA1) Hardware Constraints

## Hardware Specifications

### Tile Array Configuration
- **Layout**: 4 columns × 6 rows = 24 total tiles
- **Compute Tiles**: 4×4 = 16 AIE2 cores (rows 2-5)
- **Memory Tiles**: 4 tiles (row 1)
- **Shim Tiles**: 4 tiles (row 0) - DMA/NOC interface

### Per-Tile Resources
**AIE2 Compute Core**:
- **Vector Unit**: 512-bit wide, processes 16×BF16 per cycle
- **Local Memory**: 64 KB data memory (32 KB instruction, 32 KB data)
- **Clock**: ~1.25 GHz
- **Peak**: 8 GFLOPS per core (BF16)

**Memory Tile**:
- **Capacity**: 512 KB per tile (4× 512KB = 2 MB total)
- **Bandwidth**: High-bandwidth to compute tiles

### Memory Bandwidth
- **Shim to Memory Tile**: ~50 GB/s aggregate
- **Memory Tile to Compute**: ~400 GB/s (on-chip)
- **Host to Shim (PCIe)**: ~16 GB/s

### Key Constraints

#### 1. Memory Hierarchy
```
Host DRAM (slow)
    ↓ PCIe: 16 GB/s
Shim Tiles (DMA)
    ↓ NOC: 50 GB/s
Memory Tiles (2 MB)
    ↓ High-BW: 400 GB/s
Compute Tiles (64 KB each)
```

**Implication**: Keep data on-chip as much as possible!

#### 2. Vector Processing
- AIE2 vector unit processes **16× BF16** (32 bytes) per cycle
- LayerNorm on 512 elements = 512/16 = **32 cycles minimum**
- But we're doing 3,001 frames × 512 = **1,540,512 elements**

**Implication**: Need to process multiple frames in parallel!

#### 3. Data Movement Cost
Based on our profiling:
- **Host→Device DMA**: 0.001ms per 512 elements (1 KB)
- **Host→Device for 3001 frames**: ~3ms for 1.54 MB
- **Kernel execution**: 0.452ms per 512 elements
- **Ratio**: Kernel is 450× faster than we're using it!

**Implication**: Kernel is mostly idle waiting for data!

#### 4. Parallel Processing Capacity
- **16 compute cores** available
- Currently using **1 core** for 1 frame at a time
- Could process **16 frames simultaneously**

**Implication**: 16× parallelism available!

#### 5. Memory Tile Capacity
- **2 MB total** in memory tiles
- **3001 frames × 512 × 2 bytes** = 3.08 MB (doesn't fit!)
- **Can fit ~680 frames** in memory tiles

**Implication**: Need chunked processing!

## Optimization Strategy for Phoenix NPU

### Phase 1: Buffer Reuse (Easy - 1 hour)
**Goal**: Eliminate allocation overhead

**Current overhead**: 0.045ms × 3001 = 135ms total
**After optimization**: ~5ms one-time

**Implementation**:
```python
class WhisperEncoderOptimized:
    def __init__(self):
        # Allocate buffers once
        self.bo_instr = xrt.bo(...)  # Reuse for all calls
        self.bo_input = xrt.bo(...)
        self.bo_output = xrt.bo(...)
```

**Expected gain**: 130ms (0.5% improvement)

### Phase 2: Batched DMA Transfers (Medium - 2-3 hours)
**Goal**: Reduce DMA overhead and CPU conversion

**Current approach**: 3,001 separate DMA transfers
**Optimized approach**: Process in chunks that fit memory tiles

**Chunk size calculation**:
- Memory tile capacity: 2 MB
- Per frame: 512 × 2 bytes = 1 KB
- Optimal chunk: 680 frames (fitting in memory tiles)
- Number of chunks: 3001 / 680 = 5 chunks

**Implementation**:
```python
def layernorm_batched(x_batch):  # x_batch: (680, 512)
    # Single DMA transfer for 680 frames
    # Process all frames
    # Single DMA back
    pass
```

**Expected gain**: 1.5x speedup (28s → 19s)

### Phase 3: Multi-Frame Kernel (Hard - 1-2 weeks)
**Goal**: Process multiple frames per kernel invocation

**Current**: 1 frame per kernel call
**Target**: 16 frames per kernel call (using 16 compute cores)

**MLIR Implementation**:
```mlir
// Distribute 16 frames across 16 compute tiles
for tile_id in 0..15:
  aie.tile(%tile_id) {
    aie.core(%tile_id) {
      layernorm(frame[tile_id])  // Process 1 frame per core
    }
  }
```

**Expected gain**: 16× improvement (19s → 1.2s → **4.2x RTF**)

### Phase 4: Vectorized Processing (Hard - 2-3 weeks)
**Goal**: Use AIE2 vector units for SIMD processing

**Current**: Scalar processing (1 element at a time)
**Target**: Vector processing (16 elements at a time)

**AIE2 Vector Code**:
```cpp
// Process 16 BF16 values simultaneously
v16bfloat16 vec = load_vector(input);
v16bfloat16 normalized = vector_layernorm(vec);
store_vector(output, normalized);
```

**Expected gain**: 8-10× improvement (1.2s → 0.12s → **42x RTF**)

### Phase 5: Pipeline All Encoder Layers (Very Hard - 1-2 months)
**Goal**: Keep all tiles busy with different layers

**Approach**: Stream processing
```
Tile 0-3:   Process Layer 1
Tile 4-7:   Process Layer 2
Tile 8-11:  Process Layer 3
Tile 12-15: Process Layer 4
Memory Tiles: Store Layer 5-6 weights
```

**Expected gain**: 3-4× improvement (0.12s → 0.03s → **167x RTF**)

### Phase 6: Custom Attention + FFN (Very Hard - 2-3 months)
**Goal**: Implement full encoder on NPU

**Current attention**: 453ms (CPU)
**Current FFN**: 326ms (CPU)
**Target**: All operations on NPU

**Expected gain**: 1.3× improvement (0.03s → 0.023s → **220x RTF** ✨)

## Practical Roadmap

### Week 1: Quick Wins
- [ ] Buffer reuse (Phase 1)
- [ ] Batched DMA with chunking (Phase 2)
- **Target**: 1.5x improvement → 0.28x RTF

### Weeks 2-3: Multi-Core Processing
- [ ] Modify MLIR to use multiple tiles (Phase 3)
- [ ] Implement chunked parallel processing
- **Target**: 16× improvement → 4.2x RTF

### Weeks 4-6: Vectorization
- [ ] Write vectorized LayerNorm kernel (Phase 4)
- [ ] Use AIE2 vector intrinsics
- **Target**: 8-10× improvement → 42x RTF

### Months 2-3: Full Pipeline
- [ ] Multi-layer pipelining (Phase 5)
- [ ] Custom attention/FFN kernels (Phase 6)
- **Target**: 4-5× improvement → **220x RTF**

## Memory-Conscious Implementation

### Chunked Processing Pattern
```python
def process_encoder_chunked(x):
    # x: (3001, 512)
    chunk_size = 680  # Fits in memory tiles
    num_chunks = ceil(3001 / 680)  # 5 chunks

    results = []
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, 3001)

        # Process chunk (fits in memory tiles)
        chunk_result = process_chunk_npu(x[start:end])
        results.append(chunk_result)

    return concat(results)
```

### Tile Allocation Strategy
```
Columns 0-3 (4 compute tiles per column):
  - Column 0: Process frames 0-169
  - Column 1: Process frames 170-339
  - Column 2: Process frames 340-509
  - Column 3: Process frames 510-679

Memory Tiles:
  - Store input chunk (680 × 512 × 2 = 680 KB)
  - Store output chunk (680 KB)
  - Store weights/coefficients (remaining space)
```

## Key Takeaways

1. **Memory bandwidth is NOT the bottleneck** (only 0.2% of time)
2. **Kernel invocation overhead** is the real issue (60% idle)
3. **16× parallelism available** (16 compute cores)
4. **Chunked processing required** (memory tile capacity: 2 MB)
5. **Path to 220x is clear** through progressive optimization

## Hardware Utilization Targets

| Phase | Cores Used | Memory Used | Utilization |
|-------|-----------|-------------|-------------|
| Current | 1/16 (6%) | <1 MB | ~5% |
| Phase 1-2 | 1/16 | <1 MB | ~5% |
| Phase 3 | 16/16 (100%) | 2 MB | ~80% |
| Phase 4 | 16/16 | 2 MB | ~95% |
| Phase 5-6 | 16/16 | 2 MB | ~98% |

**Goal**: Reach 95%+ hardware utilization for 220x performance.
