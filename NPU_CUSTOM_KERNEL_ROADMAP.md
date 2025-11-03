# 🚀 NPU Custom Kernel Development Roadmap to 220x Performance

**Goal**: Achieve 220x realtime Whisper transcription using custom MLIR-AIE2 kernels on AMD Phoenix NPU

**Current Status**: 6.2x realtime with NPU mel preprocessing + faster-whisper backend

**Target**: 220x realtime with full NPU pipeline

---

## Phase 1: Optimize Current Mel Processing (Week 1) ✅ IN PROGRESS

**Status**: NPU mel kernel loaded and executing successfully

**Current Performance**:
- NPU mel time: 0.404s for 5s audio
- Total time: 0.80s
- Realtime factor: 6.2x

**Optimizations Needed**:

### 1.1 Batch Processing (Priority: HIGH)
**Current**: Frame-by-frame processing (98 frames for 5s audio)
**Target**: Batch multiple frames together

**Implementation**:
```mlir
// Process 16 frames at once instead of 1
aie.tile(%tile0) {
  %buf_audio = aie.buffer(%tile0) : memref<16x400xi16>  // 16 frames
  %buf_mel = aie.buffer(%tile0) : memref<16x80xi8>      // 16 mel outputs
  
  aie.core(%tile0) {
    // Vectorized FFT across 16 frames
    // Parallel mel filterbank computation
  }
}
```

**Expected Gain**: 4-8x faster mel processing
**Timeline**: 2-3 days

### 1.2 DMA Pipeline Optimization
**Current**: Sequential DMA transfers
**Target**: Pipelined DMA (transfer while computing)

**Implementation**:
- Use ObjectFIFO ping-pong buffers
- Overlap input DMA, compute, output DMA
- Double buffering for continuous streaming

**Expected Gain**: 2-3x faster overall
**Timeline**: 2-3 days

### 1.3 Multi-Tile Parallelism
**Current**: Single tile processing
**Target**: Use all 4 Phoenix NPU columns

**Implementation**:
```mlir
// Distribute frames across tiles
aie.tile(%tile0_2) { /* Process frames 0-24  */ }
aie.tile(%tile1_2) { /* Process frames 25-49 */ }
aie.tile(%tile2_2) { /* Process frames 50-74 */ }
aie.tile(%tile3_2) { /* Process frames 75-98 */ }
```

**Expected Gain**: 3-4x faster
**Timeline**: 3-4 days

**Phase 1 Target**: 20-25x realtime (from current 6.2x)

---

## Phase 2: Matrix Multiplication Kernel (Weeks 2-3)

**Goal**: Implement INT8 quantized matrix multiplication on NPU for encoder/decoder layers

### 2.1 Basic MatMul Kernel
**Operation**: C = A × B (INT8)
**Tile Sizes**: 64×64, 32×32, 16×16

**MLIR Implementation**:
```mlir
aie.tile(%tile0_2) {
  %A = aie.buffer : memref<64x64xi8>
  %B = aie.buffer : memref<64x64xi8>
  %C = aie.buffer : memref<64x64xi32>
  
  aie.core {
    // Vectorized INT8 matmul
    // Accumulate to INT32
    // Requantize to INT8
  }
}
```

**Testing**:
- Unit tests with known matrices
- Numerical accuracy validation (>99.9%)
- Performance benchmarking

**Timeline**: 5-7 days
**Expected**: 100-200 GOPS on NPU

### 2.2 Tiled MatMul for Large Matrices
**Whisper encoder**: 384×384, 512×512, 1024×1024 matrices
**Strategy**: Tile into 64×64 blocks, distribute across NPU

**Implementation**:
- Tiling algorithm for arbitrary sizes
- Inter-tile data movement optimization
- Accumulation strategy

**Timeline**: 3-4 days
**Expected**: Handle all Whisper matrix sizes

### 2.3 Integration with ONNX Models
**Replace**: `torch.matmul` calls in encoder/decoder
**With**: NPU matmul kernel

**Implementation**:
```python
# In onnx_whisper_npu.py
def run_encoder_layer(hidden_states):
    # Q, K, V projections use NPU matmul
    Q = npu_matmul(hidden_states, W_q)  # NPU kernel
    K = npu_matmul(hidden_states, W_k)  # NPU kernel
    V = npu_matmul(hidden_states, W_v)  # NPU kernel
    # ...
```

**Timeline**: 3-4 days

**Phase 2 Target**: 30-40x realtime (major encoder/decoder acceleration)

---

## Phase 3: Attention Mechanism (Weeks 4-5)

**Goal**: Implement multi-head self-attention on NPU

### 3.1 Scaled Dot-Product Attention
**Operation**: Attention(Q, K, V) = softmax(QK^T / √d_k) V

**Components**:
1. Q × K^T matrix multiply (NPU)
2. Scale by 1/√d_k (NPU)
3. Softmax (NPU - challenging!)
4. Result × V (NPU)

**MLIR Implementation**:
```mlir
// Attention kernel for 8 heads
aie.tile(%tile0_2) {
  %Q = aie.buffer : memref<8x48x64xi8>     // 8 heads, 48 tokens, 64 dims
  %K = aie.buffer : memref<8x48x64xi8>
  %V = aie.buffer : memref<8x48x64xi8>
  %out = aie.buffer : memref<8x48x64xi8>
  
  aie.core {
    // Compute attention scores
    // Apply softmax
    // Multiply by values
  }
}
```

**Challenges**:
- Softmax requires exp() and division
- Need lookup tables or polynomial approximation
- INT8 quantization of attention scores

**Timeline**: 7-10 days

### 3.2 Multi-Head Attention
**Whisper base**: 6 attention heads
**Whisper large**: 16 attention heads

**Parallelization**: Process heads across NPU tiles

**Timeline**: 3-4 days

### 3.3 Cross-Attention (Decoder)
**Different from self-attention**: Q from decoder, K/V from encoder

**Implementation**: Extend self-attention kernel

**Timeline**: 2-3 days

**Phase 3 Target**: 60-80x realtime (attention is 60-70% of compute)

---

## Phase 4: Complete Encoder on NPU (Weeks 6-7)

**Goal**: All 6 encoder layers (Whisper base) running on NPU

### 4.1 Layer Normalization
**Operation**: LayerNorm(x) = γ(x - μ)/σ + β

**NPU Implementation**:
- Compute mean and variance (reduction)
- Normalize (vectorized)
- Scale and shift

**Timeline**: 3-4 days

### 4.2 Feed-Forward Network
**Architecture**: Linear → GELU → Linear

**Components**:
- Linear: NPU matmul ✅ (from Phase 2)
- GELU: Existing kernel (`gelu_simple.xclbin`) or custom
- Residual connection

**Timeline**: 2-3 days

### 4.3 Complete Encoder Layer
**Stack**:
1. Multi-head self-attention
2. Residual + LayerNorm
3. Feed-forward network
4. Residual + LayerNorm

**Implementation**:
- Compose all kernels
- Minimize CPU-NPU transfers
- Keep intermediate data on NPU

**Timeline**: 3-4 days

### 4.4 6-Layer Encoder
**Repeat**: Run 6 encoder layers sequentially
**Optimization**: Pipeline layer execution

**Timeline**: 2-3 days

**Phase 4 Target**: 120-150x realtime (encoder fully on NPU)

---

## Phase 5: Complete Decoder on NPU (Weeks 8-9)

**Goal**: All 6 decoder layers running on NPU with KV cache

### 5.1 KV Cache Implementation
**Purpose**: Store past key/value states for autoregressive generation
**Challenge**: Manage cache on NPU memory

**Implementation**:
```mlir
// KV cache for 6 layers
aie.tile(%tile_mem) {
  %kv_cache = aie.buffer : memref<6x2x1500x512xi8>  // layers × K/V × tokens × dims
}
```

**Timeline**: 4-5 days

### 5.2 Decoder Self-Attention with Causal Masking
**Difference from encoder**: Causal mask prevents attending to future tokens

**Implementation**: Add mask to attention scores before softmax

**Timeline**: 2-3 days

### 5.3 Decoder Cross-Attention
**Operation**: Attend to encoder outputs

**Implementation**: Use encoder output as K/V, decoder hidden state as Q

**Timeline**: 2-3 days

### 5.4 Complete 6-Layer Decoder
**Stack**: Same as encoder but with cross-attention + causal masking

**Timeline**: 3-4 days

**Phase 5 Target**: 180-200x realtime (full model on NPU)

---

## Phase 6: End-to-End Optimization (Week 10)

**Goal**: Eliminate all CPU bottlenecks and achieve 220x

### 6.1 Token Embedding and Logits
**Embedding**: Convert token IDs to embeddings (on NPU)
**Logits**: Final projection to vocabulary (NPU matmul)

**Timeline**: 2 days

### 6.2 Beam Search on NPU (Optional)
**Current**: CPU beam search
**Target**: NPU beam search for zero CPU usage

**Timeline**: 3-4 days (optional)

### 6.3 Pipeline All Operations
**Eliminate**: All CPU-NPU synchronization points
**Strategy**: Continuous streaming

**Timeline**: 2-3 days

### 6.4 Final Optimizations
- Kernel fusion (combine operations)
- Memory layout optimization
- DMA transfer reduction
- Precision tuning (INT4 where possible)

**Timeline**: 3-4 days

**Phase 6 Target**: 220x realtime ✅ **GOAL ACHIEVED**

---

## Development Tools and Infrastructure

### MLIR-AIE2 Toolchain
- ✅ `aie-opt`: MLIR optimization passes
- ⏳ Peano C++ compiler: For AIE core compilation (need to locate/install)
- ✅ `aie-translate`: XCLBIN generation
- ✅ XRT 2.20.0: NPU runtime

### Testing Framework
```bash
# Unit test for each kernel
cd whisperx/npu/npu_optimization/whisper_encoder_kernels
pytest test_npu_matmul.py -v
pytest test_npu_attention.py -v
pytest test_npu_layernorm.py -v
pytest test_npu_encoder.py -v
pytest test_npu_decoder.py -v
```

### Benchmarking
```python
# Performance testing
from whisperx.npu.npu_optimization import NPUBenchmark

bench = NPUBenchmark()
bench.test_matmul_performance()      # GOPS, latency
bench.test_attention_performance()   # Tokens/sec
bench.test_encoder_performance()     # Frames/sec
bench.test_full_pipeline()           # Realtime factor
```

---

## Risk Mitigation

### Risk 1: Kernel Complexity
**Mitigation**: Start with simple kernels, validate thoroughly, incrementally add features

### Risk 2: NPU Resource Limits
**Mitigation**: Profile memory usage, use tiling, optimize data movement

### Risk 3: Numerical Accuracy
**Mitigation**: Extensive testing vs CPU baseline, accept <1% WER degradation

### Risk 4: DRM IOCTL Errors (Like GELU-2048, Attention)
**Mitigation**: 
- Sequential kernel loading (don't load all at once)
- Investigate context limits
- May need kernel module reload between compilations

---

## Success Metrics

### Performance Targets by Phase
| Phase | Target RTF | Description |
|-------|-----------|-------------|
| Current | 6.2x | NPU mel + faster-whisper |
| Phase 1 | 20-25x | Optimized mel |
| Phase 2 | 30-40x | + NPU matmul |
| Phase 3 | 60-80x | + NPU attention |
| Phase 4 | 120-150x | + Full encoder |
| Phase 5 | 180-200x | + Full decoder |
| Phase 6 | **220x** | **Full pipeline** ✅ |

### Accuracy Targets
- WER degradation: <1% vs CPU baseline
- Numerical correlation: >0.99 for all kernels

### Resource Utilization
- NPU utilization: >80%
- CPU usage: <5%
- Memory usage: <2GB

---

## Timeline Summary

**Total Duration**: 10 weeks
**Checkpoints**: End of each phase
**Deliverables**: Working kernel + tests + benchmarks per phase

**Week-by-Week**:
- Week 1: Mel optimization → 20-25x ✅
- Weeks 2-3: MatMul kernel → 30-40x
- Weeks 4-5: Attention → 60-80x
- Weeks 6-7: Full encoder → 120-150x
- Weeks 8-9: Full decoder → 180-200x
- Week 10: Final optimization → **220x** 🎯

---

## Next Immediate Steps (Week 1)

1. **Locate/Install Peano Compiler** (30-60 min)
2. **Test XCLBIN generation** with passthrough kernel (15 min)
3. **Implement mel batch processing** (2-3 days)
4. **DMA pipeline optimization** (2-3 days)
5. **Multi-tile parallelism** (3-4 days)

**Expected by end of Week 1**: 20-25x realtime performance!

---

**Status**: Ready to begin Phase 1 optimizations
**Current**: 6.2x realtime with NPU mel preprocessing ✅
**Target**: 220x realtime in 10 weeks 🚀
