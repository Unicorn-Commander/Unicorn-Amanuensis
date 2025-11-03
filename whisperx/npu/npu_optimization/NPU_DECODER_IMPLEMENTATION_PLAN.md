# NPU Whisper Decoder - Implementation Plan

**Date**: November 2, 2025
**Project Lead**: NPU Decoder Implementation Team
**Status**: Ready for Implementation
**Timeline**: 8-10 weeks to 220x target

---

## Executive Summary

This document provides a **detailed, phased implementation plan** to achieve 220x realtime Whisper decoder performance on AMD Phoenix NPU.

**Approach**: Incremental development with measurable milestones every 2 weeks
**Strategy**: Hybrid NPU/CPU initially, progressing to full NPU
**Risk Management**: Maintain accuracy and CPU fallback at each phase

---

## Phase 1: Foundation - Accurate Decoder (Week 1-2)

### Objective
Get the decoder producing **accurate transcriptions** on NPU, even if not yet optimized for speed.

**Target Performance**: 20-30x realtime
**Target Accuracy**: Match CPU baseline (2.5% WER)

### Tasks

#### Week 1: Fix Existing Decoder Issues

**1.1 Debug Garbled Output** (2 days)
- **File**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/onnx_whisper_npu.py`
- **Issues**: Limited to 20 tokens, incorrect token sequences
- **Actions**:
  - Review `WhisperForConditionalGeneration` configuration
  - Fix decoder input/output tensor shapes
  - Verify tokenizer integration
  - Test with known audio samples

**Deliverable**: Decoder produces coherent text (not garbled)

**1.2 Implement KV Cache** (3 days)
- **Current**: No caching → O(n²) complexity
- **Target**: Proper KV cache → O(n) complexity
- **Implementation**:
  ```python
  class DecoderKVCache:
      def __init__(self, num_layers=6, max_length=250):
          self.cache = {}
          for layer in range(num_layers):
              self.cache[layer] = {
                  'key': np.zeros((max_length, 512), dtype=np.float16),
                  'value': np.zeros((max_length, 512), dtype=np.float16),
                  'length': 0
              }

      def update(self, layer, new_k, new_v):
          idx = self.cache[layer]['length']
          self.cache[layer]['key'][idx] = new_k
          self.cache[layer]['value'][idx] = new_v
          self.cache[layer]['length'] += 1

      def get(self, layer):
          length = self.cache[layer]['length']
          return (
              self.cache[layer]['key'][:length],
              self.cache[layer]['value'][:length]
          )
  ```

**Deliverable**: KV cache working, 25-50x speedup vs non-cached

**1.3 Connect to NPU Matmul Kernel** (2 days)
- **File**: `whisper_encoder_kernels/whisper_npu_decoder_matmul.py`
- **Action**: Replace `NPUMatmul` wrapper with actual XCLBIN kernel
- **Test**: Single matmul operation on NPU
  ```python
  # Test script
  from npu_matmul_wrapper import NPUMatmul

  npu_matmul = NPUMatmul(xclbin_path="build/matmul_simple.xclbin")

  # Test data
  A = np.random.randint(-32, 32, (16, 16), dtype=np.int8)
  B = np.random.randint(-32, 32, (16, 16), dtype=np.int8)

  # Run on NPU
  C_npu = npu_matmul(A, B)

  # Verify against CPU
  C_cpu = (A @ B).astype(np.int8)

  assert np.allclose(C_npu, C_cpu, atol=2)
  print("✅ NPU matmul working!")
  ```

**Deliverable**: At least one decoder matmul running on NPU hardware

#### Week 2: Self-Attention with Causal Masking

**2.1 Modify Attention Kernel** (3 days)
- **Source**: `whisper_encoder_kernels/attention_int8.c`
- **Add**: Causal mask support
  ```c
  // Add causal masking to attention kernel
  void decoder_self_attention_int8(
      const int8_t* q,
      const int8_t* kv_cache,
      int8_t* out,
      int current_step
  ) {
      // ... existing code ...

      // Apply causal mask
      for (int i = 0; i <= current_step; i++) {
          // Only attend to current and past tokens
          int16_t score = compute_attention_score(q, k[i]);

          // Tokens beyond current_step are masked (implicitly by loop)
          scores[i] = score;
      }

      // ... softmax and weighted sum ...
  }
  ```

- **MLIR**: Update `attention_simple.mlir` for causal pattern
- **Compile**: `./compile_decoder_attention.sh`

**Deliverable**: `decoder_self_attention.xclbin` (12-15 KB)

**2.2 Test Autoregressive Generation** (2 days)
- **Test**: Token-by-token generation with causal attention
- **Verify**: Each token can only see past tokens
- **Measure**: Latency per token

**Deliverable**: Working causal self-attention on NPU

**2.3 End-to-End Integration** (2 days)
- **Integrate**: All Phase 1 components
- **Test**: Real audio transcription
- **Measure**: Accuracy and performance

**Deliverable**: Complete decoder pipeline producing accurate text

### Phase 1 Success Criteria
- ✅ Decoder produces accurate transcriptions (match CPU baseline)
- ✅ KV cache implemented and working
- ✅ At least 2 NPU kernels running (matmul + attention)
- ✅ Performance: 20-30x realtime
- ✅ No garbled output or token limit issues

---

## Phase 2: Performance Optimization (Week 3-4)

### Objective
Optimize decoder performance while maintaining accuracy.

**Target Performance**: 60-80x realtime
**Target Accuracy**: <1% degradation from CPU baseline

### Tasks

#### Week 3: Vocabulary Projection Optimization

**3.1 Implement Sparse Vocabulary** (2 days)
- **Build Top-5K vocabulary**:
  ```python
  # analyze_vocabulary_frequency.py
  from collections import Counter
  import json

  def build_sparse_vocab(tokenizer, training_transcripts, top_k=5000):
      token_freq = Counter()

      for text in training_transcripts:
          tokens = tokenizer.encode(text)
          token_freq.update(tokens)

      # Select top-K most frequent
      sparse_vocab = [tok for tok, count in token_freq.most_common(top_k)]

      # Create mapping
      sparse_to_full = {i: full_id for i, full_id in enumerate(sparse_vocab)}
      full_to_sparse = {full_id: i for i, full_id in enumerate(sparse_vocab)}

      # Save
      np.save('sparse_vocab_indices.npy', sparse_vocab)
      with open('sparse_vocab_mapping.json', 'w') as f:
          json.dump({'sparse_to_full': sparse_to_full}, f)

      return sparse_vocab, sparse_to_full, full_to_sparse
  ```

- **Extract Sparse Weights**:
  ```python
  # Full LM head: [51865, 512]
  full_lm_head = model.lm_head.weight.detach().numpy()

  # Sparse LM head: [5000, 512]
  sparse_lm_head = full_lm_head[sparse_vocab, :]

  # Quantize to INT8
  sparse_lm_head_int8, scale = quantize_weights(sparse_lm_head)

  # Save
  sparse_lm_head_int8.tofile('decoder_weights_int8/lm_head_sparse.bin')
  ```

**Deliverable**: Sparse vocabulary with 10× faster projection

**3.2 Sparse Vocabulary NPU Kernel** (3 days)
- **C Kernel**: `sparse_vocab_int8.c`
- **MLIR Wrapper**: `sparse_vocab.mlir`
- **Compilation**: `./compile_sparse_vocab.sh`
- **Testing**: Verify accuracy vs full vocabulary

**Deliverable**: `sparse_vocab.xclbin` with <10ms latency per token

**3.3 Cross-Attention Optimization** (2 days)
- **Pre-compute** encoder K,V once:
  ```python
  # After encoder completes
  encoder_kv_cache = {}
  for layer in range(6):
      k = npu_matmul(encoder_output, W_k[layer])
      v = npu_matmul(encoder_output, W_v[layer])
      encoder_kv_cache[layer] = {'k': k, 'v': v}

  # Pin in NPU-accessible memory
  npu_pin_memory(encoder_kv_cache)
  ```

- **Reuse** across all decoder steps (no recomputation)

**Deliverable**: Cross-attention 10x faster (amortized)

#### Week 4: Fused Kernels and DMA Optimization

**4.1 Fused FFN Kernel** (3 days)
- **Combine**: Linear(512→2048) + GELU + Linear(2048→512)
- **C Kernel**: `fused_ffn_int8.c`
- **Benefits**: Eliminate intermediate transfers
- **Expected Speedup**: 1.3x over separate kernels

**Deliverable**: `fused_ffn.xclbin` (15-20 KB)

**4.2 DMA Pipelining** (2 days)
- **Overlap** DMA transfers with computation:
  ```python
  for layer in range(6):
      # Start async DMA for next layer
      if layer < 5:
          dma_async_load(weights[layer+1])

      # Compute current layer (overlapped)
      output = npu_decoder_layer(input, weights[layer])

      # Wait for DMA
      if layer < 5:
          dma_wait()
  ```

**Deliverable**: 20-30% latency reduction from DMA overlap

**4.3 End-to-End Benchmark** (2 days)
- **Test**: Multiple audio files
- **Measure**: RTF, accuracy, latency breakdown
- **Profile**: Identify remaining bottlenecks

**Deliverable**: Performance report with profiling data

### Phase 2 Success Criteria
- ✅ Performance: 60-80x realtime
- ✅ Sparse vocabulary working (<1% accuracy loss)
- ✅ Fused FFN kernel deployed
- ✅ DMA pipelining implemented
- ✅ All decoder layers on NPU

---

## Phase 3: Scaling (Week 5-6)

### Objective
Scale tile sizes and optimize memory layout for production workloads.

**Target Performance**: 120-150x realtime

### Tasks

#### Week 5: Tile Size Optimization

**5.1 Scale to 64×64 Tiles** (3 days)
- **Current**: 16×16 tiles
- **Target**: 64×64 tiles (16× more work per kernel call)
- **Memory**: Requires tiling strategy for NPU 32KB limit
- **Action**: Implement blocked matmul
  ```c
  // Blocked matmul for large matrices
  for (int i = 0; i < M; i += 64) {
      for (int j = 0; j < N; j += 64) {
          for (int k = 0; k < K; k += 64) {
              npu_matmul_64x64(
                  A + i*K + k,
                  B + k*N + j,
                  C + i*N + j
              );
          }
      }
  }
  ```

**Deliverable**: 64×64 kernels with 3-4x speedup

**5.2 Multi-Head Parallel Attention** (4 days)
- **Current**: Process 8 heads sequentially
- **Target**: Process 4 heads per NPU core (use 2 cores)
- **Parallelization**:
  ```python
  # Assign heads to cores
  core_0_heads = [0, 1, 2, 3]  # Heads 0-3
  core_1_heads = [4, 5, 6, 7]  # Heads 4-7

  # Launch in parallel
  future_0 = npu_attention_async(q, kv, core_0_heads, core=0)
  future_1 = npu_attention_async(q, kv, core_1_heads, core=1)

  # Wait for both
  result_0 = future_0.wait()
  result_1 = future_1.wait()

  # Concatenate
  output = np.concatenate([result_0, result_1], axis=1)
  ```

**Deliverable**: 2× speedup from 2-core parallelism

#### Week 6: Memory and Cache Optimization

**6.1 Encoder KV Chunking** (2 days)
- **Problem**: 1500-frame encoder output too large for NPU L3
- **Solution**: Process in chunks
  ```python
  chunk_size = 256
  for chunk_start in range(0, 1500, chunk_size):
      chunk_kv = encoder_kv[chunk_start:chunk_start+chunk_size]
      partial_scores = npu_cross_attention_chunk(q, chunk_kv)
      all_scores.append(partial_scores)

  # Combine scores
  final_scores = np.concatenate(all_scores, axis=-1)
  ```

**Deliverable**: Cross-attention with chunked encoder KV

**6.2 Dynamic Weight Loading** (2 days)
- **Problem**: Can't fit all 6 layers' weights in NPU memory
- **Solution**: Load weights on-demand per layer
- **Optimization**: Cache most recently used layer

**Deliverable**: Efficient weight swapping strategy

**6.3 Profiling and Tuning** (3 days)
- **Tools**: XRT profiling, custom timers
- **Identify**: Hotspots and bottlenecks
- **Tune**: Kernel parameters, buffer sizes, DMA patterns

**Deliverable**: Optimized configuration for 120-150x target

### Phase 3 Success Criteria
- ✅ Performance: 120-150x realtime
- ✅ 64×64 tile kernels working
- ✅ Multi-head parallelism implemented
- ✅ Encoder KV chunking working
- ✅ All optimizations integrated

---

## Phase 4: Multi-Core Parallelism (Week 7-8)

### Objective
Utilize all 4 Phoenix NPU cores for maximum performance.

**Target Performance**: 200-220x realtime ✨

### Tasks

#### Week 7: 4-Core Decoder Pipeline

**7.1 Layer Parallelism** (4 days)
- **Strategy**: Distribute 6 decoder layers across 4 cores
  ```
  Core 0: Layers 0, 1
  Core 1: Layers 2, 3
  Core 2: Layers 4, 5
  Core 3: Vocabulary projection
  ```

- **Pipeline**: Overlap layer computation
  ```python
  # Layer pipeline
  while generating:
      # All cores work simultaneously
      future_0 = npu_layers_async(input, [0, 1], core=0)
      future_1 = npu_layers_async(input, [2, 3], core=1)
      future_2 = npu_layers_async(input, [4, 5], core=2)

      # Wait for layers 0-1
      x = future_0.wait()

      # Pass to layers 2-3 (core 1)
      # ... continue pipeline
  ```

**Deliverable**: 4-core decoder with 1.5-2x speedup

**7.2 Batch Processing** (3 days)
- **Process** multiple audio files in parallel
- **Batch size**: 4 (one per core)
- **Benefits**: Amortize overhead, better utilization

**Deliverable**: Batched inference with 1.3-1.5x additional speedup

#### Week 8: Final Optimization and Polish

**8.1 Beam Search Implementation** (3 days)
- **CPU-coordinated** beam search
- **NPU-accelerated** compute for each beam
- **Beam width**: 5 (typical)
  ```python
  beams = [[START_TOKEN]]  # Initial beam

  for step in range(max_length):
      candidates = []

      for beam in beams:
          # Run decoder for this beam (NPU)
          logits = npu_decoder_step(beam)

          # Get top-K next tokens
          top_k_tokens = np.argsort(logits)[-10:]

          for token in top_k_tokens:
              candidates.append(beam + [token])

      # Keep top 5 beams by score
      beams = select_top_beams(candidates, beam_width=5)
  ```

**Deliverable**: Beam search with 10-20% accuracy improvement

**8.2 Accuracy Validation** (2 days)
- **Test Suite**: 100+ audio files with ground truth
- **Metrics**: WER, accuracy, RTF
- **Comparison**: CPU baseline vs NPU decoder

**Deliverable**: Validation report

**8.3 Production Integration** (2 days)
- **Integrate** with `server_dynamic.py`
- **Fallback**: Graceful degradation to CPU if NPU unavailable
- **Monitoring**: Performance metrics and logging

**Deliverable**: Production-ready NPU decoder

### Phase 4 Success Criteria
- ✅ Performance: 200-220x realtime ✨ **TARGET ACHIEVED**
- ✅ All 4 NPU cores utilized
- ✅ Beam search working
- ✅ Accuracy validated (< 1% degradation)
- ✅ Production integration complete

---

## Testing Strategy

### Unit Tests
```python
# tests/test_npu_decoder.py

def test_causal_attention():
    """Test causal self-attention masks future tokens"""
    # ... test implementation

def test_kv_cache_update():
    """Test KV cache grows correctly"""
    # ... test implementation

def test_cross_attention():
    """Test cross-attention with encoder KV"""
    # ... test implementation

def test_sparse_vocabulary():
    """Test sparse vocab produces correct tokens"""
    # ... test implementation

def test_fused_ffn():
    """Test fused FFN matches separate ops"""
    # ... test implementation
```

### Integration Tests
```python
# tests/test_decoder_integration.py

def test_single_token_generation():
    """Test generating one token"""
    # ... test implementation

def test_full_transcription():
    """Test full audio transcription accuracy"""
    # ... test implementation

def test_long_output():
    """Test 300+ token generation"""
    # ... test implementation
```

### Performance Tests
```python
# tests/benchmark_decoder.py

def benchmark_per_token_latency():
    """Measure latency per token"""
    # ... benchmark implementation

def benchmark_full_transcription():
    """Measure end-to-end RTF"""
    # ... benchmark implementation

def benchmark_memory_usage():
    """Measure NPU and host memory"""
    # ... benchmark implementation
```

---

## Risk Mitigation Plan

### High-Priority Risks

**Risk 1**: KV cache doesn't fit in NPU memory
- **Mitigation**: Use host memory with DMA prefetch
- **Fallback**: Hybrid NPU/host memory approach
- **Impact**: Adds 50-100µs latency (acceptable)

**Risk 2**: Cross-attention too slow (1.5ms per token)
- **Mitigation**: Chunked encoder KV processing
- **Fallback**: Keep cross-attention on CPU (still 100x overall)
- **Impact**: Reduces target to 150-180x (still excellent)

**Risk 3**: Sequential decoder limits NPU utilization
- **Mitigation**: Multi-stream batching
- **Fallback**: Accept single-stream limitation
- **Impact**: May not reach 220x for single audio file

### Medium-Priority Risks

**Risk 4**: INT8 quantization degrades accuracy
- **Mitigation**: Mixed precision (INT8 matmul, FP16 softmax)
- **Fallback**: Use FP16 for critical operations
- **Impact**: 2x slower but still 100x+ realtime

**Risk 5**: DMA bandwidth becomes bottleneck
- **Mitigation**: Aggressive pipelining and caching
- **Fallback**: Keep frequently used data in NPU L3
- **Impact**: May limit to 150-180x

---

## Success Metrics

### Phase 1 (Week 2)
- ✅ Accuracy: Match CPU baseline
- ✅ RTF: 20-30x
- ✅ NPU Kernels: 2+ running
- ✅ KV Cache: Working

### Phase 2 (Week 4)
- ✅ Accuracy: < 1% degradation
- ✅ RTF: 60-80x
- ✅ NPU Kernels: 5+ running
- ✅ Optimizations: Sparse vocab, fused FFN, DMA pipelining

### Phase 3 (Week 6)
- ✅ Accuracy: < 1% degradation
- ✅ RTF: 120-150x
- ✅ Tile Size: 64×64
- ✅ Cores Used: 2+

### Phase 4 (Week 8)
- ✅ Accuracy: < 1% degradation (with beam search)
- ✅ RTF: 200-220x ✨ **TARGET ACHIEVED**
- ✅ Cores Used: 4
- ✅ Production: Deployed and tested

---

## Resource Requirements

### Hardware
- ✅ AMD Phoenix NPU (available)
- ✅ XRT 2.20.0 (installed)
- ✅ 16GB+ RAM (available)

### Software
- ✅ MLIR-AIE2 toolchain (installed)
- ✅ Peano compiler (available)
- ✅ Python 3.10+ (available)
- ✅ PyTorch, NumPy (installed)

### Development Time
- **Total**: 8-10 weeks
- **Full-time equivalent**: 1 engineer
- **Can be parallelized** with encoder optimization work

---

## Timeline Visualization

```
Week 1-2: Phase 1 - Foundation
├─ Fix decoder bugs
├─ Implement KV cache
└─ Connect NPU kernels
   → Deliverable: 20-30x RTF ✅

Week 3-4: Phase 2 - Optimization
├─ Sparse vocabulary
├─ Fused FFN
└─ DMA pipelining
   → Deliverable: 60-80x RTF ✅

Week 5-6: Phase 3 - Scaling
├─ 64×64 tiles
├─ Multi-head parallel
└─ Memory optimization
   → Deliverable: 120-150x RTF ✅

Week 7-8: Phase 4 - Multi-Core
├─ 4-core parallelism
├─ Beam search
└─ Production integration
   → Deliverable: 200-220x RTF ✨ **GOAL ACHIEVED**
```

---

## Conclusion

This implementation plan provides a **clear, achievable path** to 220x realtime Whisper decoder performance on AMD Phoenix NPU.

**Key Strengths**:
- ✅ Incremental approach with value at each phase
- ✅ Realistic timelines based on proven encoder work
- ✅ Risk mitigation at every step
- ✅ Clear success criteria and milestones

**Confidence Level**: **HIGH**
- All foundational components proven
- UC-Meeting-Ops demonstrated 220x on same hardware
- Hybrid approach provides safety net

**Next Steps**:
1. Review and approve this plan
2. Begin Phase 1 implementation
3. Set up weekly progress checkpoints
4. Prepare test audio dataset

---

**Plan Prepared By**: NPU Decoder Implementation Team
**Date**: November 2, 2025
**Status**: Ready for Execution
**Next Document**: NPU_DECODER_INTEGRATION.md (encoder-decoder connection details)
