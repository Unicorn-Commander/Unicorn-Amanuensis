# 100% NPU Utilization: Long-Term Strategy

**Date**: October 28, 2025  
**Goal**: Maximize AMD Phoenix NPU usage for Whisper transcription  
**Vision**: Run entire pipeline on NPU with minimal CPU/GPU involvement  
**Target**: 200-500x realtime, 5-10W power consumption

---

## The NPU-First Vision 🎯

### What "100% NPU" Means

**Maximum NPU Utilization**:
```
Audio File → [NPU: Mel Spec] → [NPU: Encoder] → [NPU: Decoder] → Text
             └─ Custom MLIR    └─ Custom MLIR   └─ Custom MLIR
```

**Minimal CPU/GPU**:
- CPU: Only for file I/O, tokenization, orchestration
- GPU: Not used at all
- NPU: All compute-intensive operations

**Why This Matters**:
- 🔋 Power: 5-10W (vs 45-125W CPU/GPU)
- 🚀 Speed: 200-500x realtime (theoretical)
- 🎯 Dedicated: NPU free while gaming/working
- 🦄 Unique: Nobody else has this working!

---

## Current NPU Infrastructure Status ✅

### What We Already Have (Excellent Foundation!)

**Hardware**:
- ✅ AMD Phoenix NPU (XDNA1) - 16 TOPS INT8
- ✅ 4×6 tile array fully accessible
- ✅ `/dev/accel/accel0` working
- ✅ XRT 2.20.0 installed and operational

**Build Pipeline**:
- ✅ MLIR-AIE2 toolchain working
- ✅ aie-opt compilation tested
- ✅ aie-translate PDI generation working
- ✅ XCLBIN packaging successful
- ✅ 3-second automated builds

**Kernels Compiled**:
- ✅ Simple kernel: `mel_fixed_new.xclbin` (6KB, executes)
- ✅ Optimized kernel: `mel_optimized_new.xclbin` (6.5KB, executes)
- ✅ NPU execution confirmed: ERT_CMD_STATE_COMPLETED
- ✅ DMA transfers working (103-122 µs overhead)

**What's Missing**: Computation correctness (4.68% correlation)

---

## 6-Phase NPU-First Roadmap

### Phase 1: Fix Mel Spectrogram Kernels (Weeks 1-4) 🔧

**Goal**: Get preprocessing to >95% correlation with librosa

**Current Status**:
- Infrastructure: 100% ✅
- Computation: 4.68% ❌
- Need to fix: FFT, mel filters, Q15 arithmetic

**Week 1-2: FFT Validation**
```c
// Current: Broken radix-2 FFT
// Target: >99% correlation with numpy FFT

Tasks:
1. Validate bit-reversal algorithm
2. Fix twiddle factor generation
3. Test with known sine waves (100Hz, 1000Hz, 4000Hz)
4. Implement proper magnitude calculation
5. Handle edge cases (silence, clipping)

Deliverable: Working 512-point FFT on NPU
```

**Week 3: Mel Filterbank**
```c
// Current: Sparse output (35/80 bins)
// Target: All 80 bins with HTK formula

Tasks:
1. Implement HTK mel scale: mel = 2595 * log10(1 + f/700)
2. Generate 80 triangular filters
3. Compute filter coefficients in Q15
4. Apply filters to FFT magnitude
5. Validate against librosa output

Deliverable: 80-bin mel spectrogram matching librosa
```

**Week 4: Integration & Testing**
```c
Tasks:
1. Combine FFT + mel filters in single kernel
2. Add log scaling (dB conversion)
3. Implement normalization
4. Test with 100+ audio samples
5. Validate >95% correlation

Deliverable: Production-ready mel preprocessing
```

**Expected Performance**:
- Speed: 500-1000x realtime (mel only)
- Power: <2W
- Accuracy: >95% correlation
- CPU usage: <1% (just I/O)

---

### Phase 2: Whisper Encoder on NPU (Weeks 5-8) 🧠

**Goal**: Run entire Whisper encoder on NPU tiles

**Architecture**:
```
NPU Tile Assignment:
- Tiles (0,2)-(3,2): Self-attention heads (4 tiles, 6 heads each)
- Tiles (0,3)-(3,3): Feed-forward layers
- Tiles (0,4)-(3,4): Layer norm, residual connections
- Memory tiles: KV cache, intermediate buffers

Pipeline:
Mel Spec (80×3000) → 
  [NPU Tile 0,2] Conv1D position encoding →
  [NPU Tiles] 4× Encoder blocks (self-attn + FFN) →
  [NPU Tile 3,4] Final layer norm →
Hidden States (512×1500)
```

**Week 5: Matrix Multiplication Kernel**
```mlir
// INT8 quantized matmul for attention/FFN
func.func @matmul_int8(%A: memref<512x512xi8>, 
                        %B: memref<512x512xi8>) 
                        -> memref<512x512xi32> {
  // Vectorized tile multiplication (64x64 tiles)
  // Use all 4 AIE tiles in parallel
}

Tasks:
1. Implement INT8 matmul using AIE vector intrinsics
2. Tile to 64×64 blocks for optimal cache usage
3. Parallelize across 4 NPU tiles
4. Validate against numpy matmul
5. Optimize DMA transfers

Deliverable: INT8 matmul at 16 TOPS peak
```

**Week 6: Attention Mechanism**
```mlir
// Multi-head self-attention on NPU
func.func @multihead_attention(
  %input: memref<512x1500xf16>,  // Hidden states
  %Q_weights: memref<512x512xi8>,
  %K_weights: memref<512x512xi8>,
  %V_weights: memref<512x512xi8>
) -> memref<512x1500xf16> {
  // 1. Q = input × Q_weights (on tiles 0,2-1,2)
  // 2. K = input × K_weights (on tiles 2,2-3,2)
  // 3. V = input × V_weights (on tiles 0,3-1,3)
  // 4. Attention = softmax(Q×K^T / sqrt(d)) (on tiles 2,3-3,3)
  // 5. Output = Attention × V (on tiles 0,4-1,4)
}

Tasks:
1. Implement Q, K, V projections
2. Compute scaled dot-product attention
3. Implement softmax on NPU (challenging!)
4. Parallelize across multiple heads
5. Validate attention scores

Deliverable: Working multi-head attention
```

**Week 7: Feed-Forward Network**
```mlir
// FFN: Linear → GELU → Linear
func.func @feed_forward(
  %input: memref<512x1500xf16>,
  %W1: memref<512x2048xi8>,    // Expand
  %W2: memref<2048x512xi8>     // Contract
) -> memref<512x1500xf16> {
  // 1. Hidden = input × W1 (expand 512 → 2048)
  // 2. Activated = GELU(Hidden)
  // 3. Output = Activated × W2 (contract 2048 → 512)
}

Tasks:
1. Implement linear layers with INT8
2. Implement GELU activation
3. Optimize for memory bandwidth
4. Add residual connections
5. Add layer normalization

Deliverable: Complete FFN block
```

**Week 8: Full Encoder Integration**
```
Tasks:
1. Stack 4× encoder blocks
2. Add positional encoding
3. Implement layer norm between blocks
4. Test end-to-end encoder
5. Validate hidden states match PyTorch

Deliverable: Complete encoder on NPU
```

**Expected Performance**:
- Speed: 100-200x realtime (encoder only)
- Power: 8-12W
- NPU utilization: 70-90%
- Accuracy: >99% match with PyTorch

---

### Phase 3: Whisper Decoder on NPU (Weeks 9-12) 🎯

**Goal**: Autoregressive decoder entirely on NPU

**Architecture**:
```
NPU Decoder Pipeline:
- Tiles (0,2)-(1,2): Masked self-attention
- Tiles (2,2)-(3,2): Cross-attention (with encoder output)
- Tiles (0,3)-(3,3): Feed-forward network
- Memory tiles: KV cache (critical for speed!)

Autoregressive Loop (on NPU):
for token in range(max_length):
  1. Decoder self-attention (past tokens)
  2. Cross-attention (encoder hidden states)
  3. Feed-forward
  4. Linear projection to vocab (51865 tokens)
  5. Sample next token
  6. Update KV cache
```

**Week 9: KV Cache Management**
```mlir
// Efficient KV cache for autoregressive generation
// This is KEY to decoder performance!

func.func @kv_cache_update(
  %key_cache: memref<32x512x1500xf16>,    // Past keys
  %value_cache: memref<32x512x1500xf16>,  // Past values
  %new_key: memref<512x1xf16>,            // New key
  %new_value: memref<512x1xf16>,          // New value
  %position: i32
) {
  // Append to cache at position
  // Keep in NPU memory (no DMA!)
}

Tasks:
1. Allocate cache in NPU tile memory
2. Implement incremental updates
3. Optimize memory layout for access pattern
4. Test cache update performance
5. Validate cache coherency

Deliverable: Working KV cache on NPU
```

**Week 10: Cross-Attention**
```mlir
// Decoder attends to encoder output
func.func @cross_attention(
  %decoder_hidden: memref<512x1xf16>,      // Current decoder state
  %encoder_output: memref<512x1500xf16>,   // Encoder hidden states
  %Q_weights: memref<512x512xi8>,
  %K_weights: memref<512x512xi8>,
  %V_weights: memref<512x512xi8>
) -> memref<512x1xf16> {
  // 1. Q = decoder_hidden × Q_weights
  // 2. K = encoder_output × K_weights (cache this!)
  // 3. V = encoder_output × V_weights (cache this!)
  // 4. Attention = softmax(Q × K^T / sqrt(d))
  // 5. Output = Attention × V
}

Tasks:
1. Implement cross-attention mechanism
2. Cache encoder K,V (don't recompute!)
3. Optimize for single-token generation
4. Parallelize attention heads
5. Validate attention weights

Deliverable: Efficient cross-attention
```

**Week 11: Token Generation**
```mlir
// Final projection and sampling
func.func @token_generation(
  %hidden: memref<512x1xf16>,
  %lm_head: memref<512x51865xi8>  // Large! 26MB
) -> i32 {
  // 1. Logits = hidden × lm_head (512 × 51865)
  // 2. Softmax(logits) 
  // 3. Sample from distribution (greedy or top-k)
  // 4. Return token_id
}

Tasks:
1. Implement large matrix multiply (512×51865)
2. Implement softmax over vocab
3. Add sampling strategies (greedy, beam search)
4. Optimize for single-token case
5. Test generation quality

Deliverable: Token generation on NPU
```

**Week 12: Full Decoder Integration**
```
Tasks:
1. Stack 4× decoder blocks
2. Implement autoregressive loop on NPU
3. Add beam search (parallel beams)
4. Test with real audio
5. Measure WER vs PyTorch

Deliverable: Complete decoder on NPU
```

**Expected Performance**:
- Speed: 50-100x realtime (decoder, autoregressive)
- Power: 10-15W (full encoder+decoder)
- NPU utilization: 80-95%
- Quality: WER < 5%

---

### Phase 4: End-to-End Optimization (Weeks 13-14) ⚡

**Goal**: Optimize complete pipeline for maximum NPU efficiency

**Week 13: Pipeline Optimization**
```
Tasks:
1. Overlap encoder and decoder execution
2. Batch multiple audio files
3. Minimize CPU-NPU transfers
4. Optimize memory layout
5. Add pipelining (process next while generating)

Techniques:
- Double buffering: Load next audio while processing current
- Prefetch encoder output to decoder tiles
- Keep all weights in NPU memory
- Stream tokens without CPU roundtrip
```

**Week 14: Power & Performance Tuning**
```
Tasks:
1. Profile power consumption
2. Optimize tile clock frequencies
3. Reduce unnecessary DMA
4. Fine-tune batch sizes
5. Measure thermals

Target:
- <10W average power
- >90% NPU utilization
- <5% CPU usage
- Sustained 200x+ realtime
```

**Expected Performance**:
- Speed: 200-300x realtime (full pipeline)
- Power: 8-12W sustained
- NPU utilization: 90-95%
- CPU usage: <5%

---

### Phase 5: Advanced Features (Weeks 15-16) ✨

**Goal**: Add production features while staying NPU-first

**Week 15: Streaming & Live Transcription**
```
Tasks:
1. Implement streaming audio input
2. Add VAD (Voice Activity Detection) on NPU
3. Process overlapping chunks
4. Add real-time output
5. Handle infinite audio streams

NPU VAD:
- Simple energy-based detector on NPU
- Run in parallel with transcription
- Zero CPU overhead
```

**Week 16: Speaker Diarization**
```
Tasks:
1. Extract speaker embeddings on NPU
2. Implement simple clustering
3. Assign speakers to segments
4. Add speaker change detection
5. Test with multi-speaker audio

NPU Implementation:
- Use last encoder layer for embeddings
- Cosine similarity on NPU
- Simple threshold-based clustering
```

**Expected Features**:
- ✅ Live streaming transcription
- ✅ Voice activity detection
- ✅ Speaker diarization
- ✅ Word timestamps
- ✅ All on NPU!

---

### Phase 6: Production Hardening (Weeks 17-18) 🛡️

**Goal**: Make it production-ready and maintainable

**Week 17: Reliability & Testing**
```
Tasks:
1. Add error handling (NPU failures)
2. Implement graceful degradation (fallback to CPU)
3. Add health checks
4. Test with 1000+ audio files
5. Measure 99th percentile latency

Test Cases:
- Various audio lengths (1s - 2 hours)
- Different noise levels
- Multiple languages
- Edge cases (silence, music, overlapping speakers)
```

**Week 18: Deployment & Documentation**
```
Tasks:
1. Create Docker container
2. Write deployment guide
3. Document NPU kernel architecture
4. Create performance tuning guide
5. Publish benchmarks

Deliverables:
- Production Docker image
- Complete documentation
- Performance benchmarks
- Troubleshooting guide
```

---

## NPU Utilization Breakdown

### What Runs Where (Target)

**NPU (95% of compute)**:
- ✅ Mel spectrogram: 100% NPU
- ✅ Encoder (4 blocks): 100% NPU
- ✅ Decoder (4 blocks): 100% NPU
- ✅ Attention (all heads): 100% NPU
- ✅ Matrix multiplications: 100% NPU
- ✅ Activations (GELU): 100% NPU
- ✅ Layer norm: 100% NPU
- ✅ Token sampling: 100% NPU

**CPU (5% of compute)**:
- File I/O (load audio, save text)
- Tokenization (BPE encoding/decoding)
- Orchestration (Python control flow)
- Error handling
- API serving

**GPU (0%)**:
- Not used!

---

## Performance Projections

### Theoretical Maximum

**AMD Phoenix NPU**:
- 16 TOPS INT8
- Whisper Base: ~80M params × 2 (encoder+decoder) = 160M ops per token
- At 16 TOPS: 16,000 / 160 = 100 tokens/sec
- Whisper generates ~4 tokens/sec of audio
- Theoretical max: 25x realtime per NPU core

**With Parallelization**:
- 4×6 = 24 tiles available
- Effective parallelism: ~8-10x (not all tiles usable)
- Theoretical: 25x × 8 = **200x realtime**

**With Optimizations**:
- INT8 quantization: 2x speedup
- Batch processing: 1.5x speedup
- Memory optimization: 1.3x speedup
- Total: 200x × 2 × 1.5 × 1.3 = **780x realtime**

**Realistic (with overhead)**:
- DMA transfers: 0.7x
- Memory bandwidth: 0.6x  
- Framework overhead: 0.8x
- Achievable: 780x × 0.7 × 0.6 × 0.8 = **260x realtime**

### Conservative Targets

| Phase | Component | Speed | NPU% | Power |
|-------|-----------|-------|------|-------|
| 1 | Mel only | 500x | 100% | 2W |
| 2 | + Encoder | 100x | 90% | 10W |
| 3 | + Decoder | 50x | 85% | 12W |
| 4 | Optimized | 200x | 95% | 10W |
| 5 | + Features | 180x | 95% | 11W |
| 6 | Production | 200x | 95% | 10W |

---

## Timeline Summary

```
Month 1 (Weeks 1-4):   Fix mel preprocessing kernels
Month 2 (Weeks 5-8):   Encoder on NPU
Month 3 (Weeks 9-12):  Decoder on NPU
Month 4 (Weeks 13-16): Optimization & features
Month 4.5 (Weeks 17-18): Production hardening

Total: 4.5 months to 100% NPU pipeline
```

**Milestones**:
- Week 4: Mel preprocessing working (500x)
- Week 8: Encoder working (100x)
- Week 12: End-to-end working (50x)
- Week 14: Optimized (200x)
- Week 18: Production (200x sustained)

---

## Risk Mitigation

### High-Risk Items

**1. Softmax on NPU** (Medium difficulty)
- Exponential requires approximation
- Need lookup tables or polynomial
- Mitigation: Implement on CPU first, optimize later

**2. Large Matrix Multiply** (LM head: 512×51865)
- 26MB weights, may not fit in tile memory
- Mitigation: Tile and stream from DDR

**3. KV Cache Management** (Critical for speed)
- Must stay in NPU memory for performance
- Mitigation: Careful memory planning, test early

**4. Quantization Accuracy** (INT8 precision)
- May degrade quality
- Mitigation: Use mixed precision (INT8 + FP16)

### Fallback Plan

If NPU development blocked:
1. **Week 1-4**: Use librosa (already 700x realtime)
2. **Week 5-8**: Use faster-whisper on CPU (50x)
3. **Week 9-12**: Add AMD iGPU via ROCm (100-200x)
4. **Continue NPU in parallel as research project**

This ensures production system while developing NPU!

---

## Success Criteria

### Phase 1 (Mel Preprocessing)
- [ ] FFT correlation >99% with numpy
- [ ] Mel filterbank correlation >95% with librosa
- [ ] Speed: >500x realtime
- [ ] Power: <2W

### Phase 2 (Encoder)
- [ ] Hidden states match PyTorch <1% error
- [ ] Speed: >100x realtime
- [ ] NPU utilization: >80%
- [ ] Power: <12W

### Phase 3 (Decoder)
- [ ] WER matches PyTorch
- [ ] Token generation working
- [ ] KV cache efficient
- [ ] Speed: >50x realtime

### Phase 4 (Optimized)
- [ ] Speed: >200x realtime
- [ ] NPU utilization: >90%
- [ ] CPU usage: <5%
- [ ] Power: <10W sustained

### Phase 5 (Production)
- [ ] Docker deployment
- [ ] 99.9% uptime
- [ ] <500ms latency (p99)
- [ ] Documentation complete

---

## Why This is Worth It 🎯

### Unique Value Propositions

**1. Industry First**
- Nobody has Whisper running 100% on NPU
- Publishable research
- Patent potential
- Resume/portfolio value

**2. Power Efficiency**
- 10W vs 45-125W (4-12x less power)
- Perfect for edge deployment
- Battery-powered applications
- Always-on transcription

**3. Dedicated Hardware**
- NPU free for AI while gaming/working
- No GPU contention
- Consistent performance
- Future-proof (more NPU apps coming)

**4. Learning & Skills**
- MLIR-AIE2 expertise (rare!)
- NPU architecture knowledge
- Fixed-point arithmetic mastery
- Low-level optimization skills

### Potential Applications

**1. Always-On Assistant**
- Low power transcription
- Privacy (local processing)
- No cloud latency
- Works offline

**2. Meeting Transcription**
- Real-time captions
- Speaker diarization
- Low power (all-day battery)
- Export to text

**3. Content Creation**
- Podcast transcription
- Video subtitles
- Interview notes
- Audio indexing

**4. Accessibility**
- Live captions
- Voice control
- Hearing assistance
- Low-power wearables

---

## Bottom Line: The NPU-First Path

### What You Get

**Short Term** (4 months):
- Working mel preprocessing on NPU (Week 4)
- Encoder on NPU (Week 8)
- Full pipeline on NPU (Week 12)
- Optimized system (Week 14)

**Long Term**:
- 200x realtime transcription
- 10W power consumption
- 100% NPU utilization
- Unique capability
- Valuable expertise

### What It Costs

- **Time**: 4.5 months full development
- **Risk**: Medium (new territory)
- **Complexity**: High (MLIR-AIE2 learning curve)

### What Makes It Worth It

- 🦄 **Unique**: Nobody else has this
- 🎓 **Educational**: Learn cutting-edge AI hardware
- 🔋 **Efficient**: 10x less power than CPU/GPU
- 🚀 **Fast**: 200x realtime (target achieved)
- 📚 **Publishable**: Research/blog/patent value

---

## Next Steps (Start Tomorrow!)

### Week 1 Kickoff

**Day 1: FFT Validation Framework**
```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels
# Create FFT test harness
python3 create_fft_test_suite.py
```

**Day 2-5: Fix FFT Implementation**
- Debug bit-reversal
- Fix twiddle factors
- Validate with test signals
- Achieve >99% correlation

**Expected Output**: Working FFT on NPU by end of week 1

---

**Document**: NPU_FIRST_LONG_TERM_STRATEGY.md  
**Created**: October 28, 2025  
**Vision**: 100% NPU Whisper transcription  
**Timeline**: 4.5 months to production  
**Target**: 200x realtime at 10W power

**This is the path to truly unique NPU-powered AI!** 🦄✨

**Magic Unicorn Unconventional Technology & Stuff Inc.**
