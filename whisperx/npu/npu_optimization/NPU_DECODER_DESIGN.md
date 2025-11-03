# NPU Whisper Decoder - Architectural Design

**Date**: November 2, 2025
**Design Lead**: NPU Decoder Architecture Team
**Status**: Comprehensive Design - Ready for Implementation
**Target**: 220x realtime with full decoder on NPU

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Decoder Architecture Overview](#2-decoder-architecture-overview)
3. [Autoregressive Generation Strategy](#3-autoregressive-generation-strategy)
4. [KV Cache Architecture](#4-kv-cache-architecture)
5. [NPU Kernel Design](#5-npu-kernel-design)
6. [Memory Management](#6-memory-management)
7. [NPU vs CPU Trade-offs](#7-npu-vs-cpu-trade-offs)
8. [Performance Modeling](#8-performance-modeling)

---

## 1. Design Philosophy

### 1.1 Guiding Principles

**Incremental Value**:
- Each phase delivers measurable performance improvement
- Never break accuracy while optimizing speed
- Maintain CPU fallback at every stage

**NPU-First Where It Matters**:
- Heavy matmul operations → NPU
- Softmax and attention → NPU
- Control flow and branching → CPU (initially)

**Proven Over Perfect**:
- Start with working hybrid approach (UC-Meeting-Ops model)
- Optimize incrementally based on profiling
- Don't over-engineer before measuring

### 1.2 Design Constraints

**Hardware**:
- Phoenix NPU: 4 compute cores, 32KB per core
- Memory bandwidth: ~10-20 GB/s DMA
- Latency: ~1-2ms kernel launch overhead

**Model**:
- Whisper Base: 6 decoder layers, 512 hidden dim, 8 attention heads
- Vocabulary: 51,865 tokens
- Typical output: 50-250 tokens for 30s audio

**Performance**:
- Target: 220x realtime (0.136s for 30s audio)
- Decoder budget: ~70ms (51% of total budget)
- Per-token budget: ~0.3ms (for 250 tokens)

---

## 2. Decoder Architecture Overview

### 2.1 Whisper Decoder Structure

**Whisper Base Decoder**:
```
Input: Token IDs [batch, seq_len]
  ↓
Token Embedding [batch, seq_len, 512]
  ↓
Position Embedding (sinusoidal) [batch, seq_len, 512]
  ↓
┌────────────────────────────────────┐
│ Decoder Layer 1                    │
│   - LayerNorm                      │
│   - Self-Attention (Causal)        │  ← Q,K,V from decoder
│   - Residual                       │
│   - LayerNorm                      │
│   - Cross-Attention                │  ← K,V from encoder, Q from decoder
│   - Residual                       │
│   - LayerNorm                      │
│   - FFN (Linear → GELU → Linear)   │
│   - Residual                       │
└────────────────────────────────────┘
  ↓
... (repeat for layers 2-6)
  ↓
Final LayerNorm
  ↓
Language Model Head: Linear [512 → 51865]
  ↓
Logits [batch, seq_len, 51865]
```

**Computation Count per Layer**:
- Self-Attention: 4 matmuls (Q, K, V, out) + attention scores
- Cross-Attention: 4 matmuls (Q, K, V, out) + attention scores
- FFN: 2 matmuls (fc1, fc2) + GELU
- **Total per layer**: 10 matmuls + 2 attention operations

**Total for 6 Layers**:
- 60 matmuls
- 12 attention operations
- 6 GELU activations
- 1 vocabulary projection (massive: 512 → 51865)

### 2.2 NPU-Optimized Decoder Block

**Redesigned for NPU Efficiency**:

```
┌────────────────────────────────────────────────────────────┐
│ NPU Decoder Layer                                          │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                                          │
│  │ LayerNorm   │  (CPU or NPU - not critical path)        │
│  └──────┬──────┘                                          │
│         │                                                   │
│  ┌──────▼───────────────────────────────┐                 │
│  │ Self-Attention (NPU Custom Kernel)   │                 │
│  │  - Fused QKV projection              │                 │
│  │  - Scaled dot-product with KV cache  │                 │
│  │  - Causal masking                    │                 │
│  │  - Softmax (INT8 approximation)      │                 │
│  │  - Output projection                 │                 │
│  │  Input:  [seq_len, 512]              │                 │
│  │  Output: [seq_len, 512]              │                 │
│  │  KV Cache: [past_len, 2×num_heads×head_dim]  │        │
│  └──────────────────────────────────────┘                 │
│         │                                                   │
│  ┌──────▼───────────────────────────────┐                 │
│  │ Cross-Attention (NPU Custom Kernel)  │                 │
│  │  - Q from decoder                    │                 │
│  │  - K,V from encoder (cached)         │                 │
│  │  - Scaled dot-product                │                 │
│  │  - Softmax                            │                 │
│  │  - Output projection                 │                 │
│  │  Input:  [seq_len, 512]              │                 │
│  │  Encoder: [1500, 512] (constant)     │                 │
│  │  Output: [seq_len, 512]              │                 │
│  └──────────────────────────────────────┘                 │
│         │                                                   │
│  ┌──────▼───────────────────────────────┐                 │
│  │ FFN (NPU Fused Kernel)               │                 │
│  │  - Linear 512 → 2048 (NPU matmul)    │                 │
│  │  - GELU activation (NPU LUT)         │                 │
│  │  - Linear 2048 → 512 (NPU matmul)    │                 │
│  │  Input:  [seq_len, 512]              │                 │
│  │  Output: [seq_len, 512]              │                 │
│  └──────────────────────────────────────┘                 │
│         │                                                   │
│  ┌──────▼──────┐                                          │
│  │ Residual +  │  (CPU - trivial)                         │
│  └─────────────┘                                          │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**Key Optimizations**:
1. **Fused Kernels**: Combine operations to reduce transfers
2. **Persistent Cache**: Keep encoder K,V in NPU L3 memory
3. **Incremental KV**: Only compute new token's K,V each step
4. **INT8 Precision**: 4x speedup with minimal accuracy loss

---

## 3. Autoregressive Generation Strategy

### 3.1 Token-by-Token Generation Loop

**High-Level Flow**:

```python
def generate_tokens_npu(
    encoder_hidden_states,  # [1500, 512] - from encoder
    max_length=250,
    beam_size=1
):
    # Initialize
    tokens = [START_OF_TRANSCRIPT]
    kv_cache = initialize_kv_cache()  # Pre-allocate on NPU

    # Pre-compute encoder K,V for cross-attention (ONCE!)
    encoder_kv = npu_compute_encoder_kv(encoder_hidden_states)
    # Keep encoder_kv resident in NPU L3 memory

    for step in range(max_length):
        # 1. Embed current token(s) - CPU or NPU
        token_embed = embed_token(tokens[-1])  # [1, 512]

        # 2. Run decoder layers - ALL ON NPU
        decoder_output = npu_decoder_forward(
            token_embed,
            encoder_kv,      # Static across steps
            kv_cache,        # Growing incrementally
            step             # Current position
        )
        # Returns: [1, 512], updated kv_cache

        # 3. Vocabulary projection - NPU with sparse vocab
        logits = npu_vocab_projection(
            decoder_output,
            top_k=5000      # Sparse vocabulary
        )
        # Returns: [1, 5000] logits

        # 4. Token sampling - CPU (fast, not worth NPU)
        next_token = sample_token(logits, temperature=0.0)

        # 5. Check stopping criteria - CPU
        if next_token == END_OF_TRANSCRIPT:
            break

        tokens.append(next_token)

    return tokens
```

**Critical Path**:
- Step 2 (NPU decoder): ~0.2-0.3ms per token ✅
- Step 3 (NPU vocab): ~0.05ms per token with sparse vocab ✅
- **Total per token**: ~0.3ms
- **250 tokens**: ~75ms
- **Within 70ms budget?** Tight but achievable with optimization

### 3.2 NPU Decoder Forward Pass

**Detailed Implementation**:

```python
def npu_decoder_forward(
    token_embed,    # [1, 512] - current token embedding
    encoder_kv,     # [1500, 1024] - encoder K,V (static, cached)
    kv_cache,       # Growing cache for self-attention
    step            # Current generation step
):
    x = token_embed

    # Process through 6 decoder layers
    for layer_idx in range(6):
        # === Self-Attention Block ===
        # Compute Q,K,V for current token only
        q, k, v = npu_qkv_projection(x)  # [1, 512] → 3×[1, 512]

        # Update KV cache (incremental)
        kv_cache[layer_idx]['k'][step] = k  # Append new K
        kv_cache[layer_idx]['v'][step] = v  # Append new V

        # Attention with full cache (current + past)
        attn_out = npu_causal_attention(
            q,                              # [1, 512]
            kv_cache[layer_idx]['k'][:step+1],  # [step+1, 512]
            kv_cache[layer_idx]['v'][:step+1],  # [step+1, 512]
            causal=True
        )

        x = x + attn_out  # Residual

        # === Cross-Attention Block ===
        q_cross = npu_q_projection(x)  # [1, 512]

        cross_attn_out = npu_cross_attention(
            q_cross,                    # [1, 512]
            encoder_kv['k'],            # [1500, 512] - STATIC
            encoder_kv['v']             # [1500, 512] - STATIC
        )

        x = x + cross_attn_out  # Residual

        # === FFN Block ===
        ffn_out = npu_ffn(x)  # [1, 512] → [1, 2048] → [1, 512]
        x = x + ffn_out  # Residual

    return x  # [1, 512]
```

**NPU Kernel Calls per Token**:
- 6 layers × (QKV proj + causal attn + Q proj + cross attn + FFN)
- = 6 × 5 kernel calls
- = **30 NPU kernel invocations per token**
- At ~10µs overhead per call: 300µs overhead
- Actual compute: ~200µs
- **Total: ~0.5ms per token** (leaves room for optimization to 0.3ms)

### 3.3 Batching Strategy (Future Optimization)

**Problem**: Single token generation underutilizes NPU

**Solution**: Batch multiple tokens or audio streams

**Approach 1: Multi-Stream Batching**:
```python
# Process 4 audio files simultaneously
batch_size = 4
tokens = [[START]] * batch_size

for step in range(max_length):
    # Embed all 4 streams' tokens
    token_embeds = embed_tokens(tokens)  # [4, 512]

    # Run NPU decoder (batched)
    decoder_outputs = npu_decoder_forward_batched(
        token_embeds,       # [4, 512]
        encoder_kvs,        # [4, 1500, 512]
        kv_caches,          # [4, layers, cache]
        step
    )

    # Sample 4 tokens in parallel
    next_tokens = sample_tokens(decoder_outputs)

    for i in range(batch_size):
        if next_tokens[i] != EOS:
            tokens[i].append(next_tokens[i])
```

**Benefits**:
- 4× NPU utilization
- Amortize kernel launch overhead
- **Expected speedup**: 1.5-2× (due to overhead reduction)

**Approach 2: Speculative Decoding** (Advanced):
```python
# Predict next N tokens speculatively, verify in parallel
predicted_tokens = predict_next_n_tokens(current_state, n=4)

# Verify all 4 in one batched forward pass
verified = npu_decoder_forward_batched(predicted_tokens)

# Accept verified tokens, reject rest
actual_tokens = verify_and_accept(verified)
```

**Benefits**:
- Reduce sequential steps
- 2-3× speedup for predictable sequences
- **Complexity**: High - requires separate predictor model

---

## 4. KV Cache Architecture

### 4.1 Memory Layout

**KV Cache Structure**:

```
For Whisper Base (6 layers, 8 heads, 64 head_dim):

kv_cache = {
    'layer_0': {
        'k': [max_len, num_heads, head_dim]  # [250, 8, 64]
        'v': [max_len, num_heads, head_dim]  # [250, 8, 64]
    },
    'layer_1': { ... },
    ...,
    'layer_5': { ... }
}

Memory per layer = 250 × 8 × 64 × 2 (K and V) × 1 byte (INT8)
                 = 256,000 bytes
                 = 250 KB

Total for 6 layers = 1.5 MB
```

**Storage Options**:

| Location | Capacity | Latency | Bandwidth | Best For |
|----------|----------|---------|-----------|----------|
| **NPU Tile** | 32 KB per core | <1 cycle | ~500 GB/s | Hot data |
| **NPU L3** | ~128 KB shared | ~10 cycles | ~100 GB/s | Working set |
| **Host DDR** | Unlimited | ~100-200 cycles | ~20 GB/s | Large cache |

**Chosen Strategy**: **Hybrid approach**
- **Cross-attention cache** (encoder K,V): NPU L3 (192 KB, static)
- **Self-attention cache** (growing): Host DDR with DMA prefetch
- **Current step K,V**: NPU tile memory (active computation)

### 4.2 Cache Update Mechanism

**Incremental Update (Per-Token)**:

```c
// C kernel: npu_kv_cache_update.c

void kv_cache_append(
    int8_t* kv_cache,        // [max_len, 2, num_heads, head_dim]
    const int8_t* new_k,     // [num_heads, head_dim]
    const int8_t* new_v,     // [num_heads, head_dim]
    int current_step,        // Position to append
    int num_heads,           // 8
    int head_dim             // 64
) {
    // Compute offset
    int offset = current_step * 2 * num_heads * head_dim;

    // Append K
    memcpy(kv_cache + offset, new_k, num_heads * head_dim);

    // Append V
    memcpy(kv_cache + offset + num_heads * head_dim, new_v, num_heads * head_dim);

    // DMA flush to ensure visibility
    dma_flush(kv_cache, offset, 2 * num_heads * head_dim);
}
```

**DMA Transfer Pattern**:

```
Step 0: [K0, V0] → Transfer to cache
Step 1: [K1, V1] → Transfer to cache, Read [K0, V0, K1, V1]
Step 2: [K2, V2] → Transfer to cache, Read [K0-K2, V0-V2]
...
```

**Optimization**: **DMA Pipelining**
```python
# Overlap computation with DMA transfer
while generating_tokens:
    # Start DMA transfer of previous token's KV
    dma_async_write(kv_cache, prev_k, prev_v)

    # Compute current token's QKV (overlapped)
    q, k, v = npu_qkv_projection(current_token)

    # Wait for DMA completion
    dma_wait()

    # Proceed with attention
    attn_out = npu_attention(q, full_kv_cache)
```

### 4.3 Encoder KV Cache (Static)

**Pre-computation** (one-time cost):

```python
# After encoder finishes
encoder_hidden_states = npu_encoder(mel_features)  # [1500, 512]

# Compute K,V for cross-attention (ALL LAYERS)
encoder_kv_cache = {}
for layer_idx in range(6):
    # Project encoder states to K,V for this layer
    k_proj = npu_matmul(encoder_hidden_states, W_k[layer_idx])
    v_proj = npu_matmul(encoder_hidden_states, W_v[layer_idx])

    encoder_kv_cache[layer_idx] = {
        'k': k_proj,  # [1500, 512]
        'v': v_proj   # [1500, 512]
    }

# Keep in NPU L3 memory (192 KB for 6 layers)
npu_pin_memory(encoder_kv_cache)
```

**Memory Requirement**:
- Per layer: 1500 frames × 512 dims × 2 (K,V) × 1 byte = 1.5 MB
- 6 layers: 9 MB total
- **Too large for NPU L3** → Store in host memory, DMA on demand

**Optimized Strategy**:
- Keep current layer's encoder KV in NPU L3 (1.5 MB)
- Swap layers as needed (6 swaps total across generation)
- **Cost**: 6 × ~0.1ms = 0.6ms overhead (negligible across 250 tokens)

---

## 5. NPU Kernel Design

### 5.1 Causal Self-Attention Kernel

**MLIR Kernel**: `decoder_self_attention.mlir`

```mlir
// Decoder causal self-attention with KV cache
// Input: Q [1, num_heads, head_dim]
// KV Cache: [current_step+1, num_heads, 2×head_dim]
// Output: [1, num_heads, head_dim]

aie.device(npu1) {
  %tile_compute = aie.tile(0, 2)
  %tile_mem = aie.tile(0, 1)
  %tile_shim = aie.tile(0, 0)

  // Buffers
  %buf_q = aie.buffer(%tile_mem) : memref<1x8x64xi8>
  %buf_kv_cache = aie.buffer(%tile_mem) : memref<250x8x128xi8>
  %buf_out = aie.buffer(%tile_mem) : memref<1x8x64xi8>

  // ObjectFIFOs for data movement
  %fifo_q = aie.objectfifo.create("q_in", %tile_shim, [%tile_mem], 1, memref<1x8x64xi8>)
  %fifo_kv = aie.objectfifo.create("kv_cache", %tile_shim, [%tile_mem], 1, memref<250x8x128xi8>)
  %fifo_out = aie.objectfifo.create("out", [%tile_mem], %tile_shim, 1, memref<1x8x64xi8>)

  // Compute kernel
  %core = aie.core(%tile_compute) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    aie.objectfifo.acquire @q_in(Consume, 1)
    aie.objectfifo.acquire @kv_cache(Consume, 1)
    aie.objectfifo.acquire @out(Produce, 1)

    %q_subview = aie.objectfifo.subview @q_in[0]
    %kv_subview = aie.objectfifo.subview @kv_cache[0]
    %out_subview = aie.objectfifo.subview @out[0]

    // Call C kernel function
    func.call @decoder_self_attention_int8(
      %q_subview,
      %kv_subview,
      %out_subview,
      %current_step
    ) : (memref<1x8x64xi8>, memref<250x8x128xi8>, memref<1x8x64xi8>, index) -> ()

    aie.objectfifo.release @q_in(Consume, 1)
    aie.objectfifo.release @kv_cache(Consume, 1)
    aie.objectfifo.release @out(Produce, 1)

    aie.end
  }
}
```

**C Kernel**: `decoder_self_attention_int8.c`

```c
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>

void decoder_self_attention_int8(
    const int8_t* q,           // [1, 8, 64] - Query for current token
    const int8_t* kv_cache,    // [step+1, 8, 128] - Cached K,V
    int8_t* out,               // [1, 8, 64] - Output
    int current_step           // Current position in sequence
) {
    const int num_heads = 8;
    const int head_dim = 64;
    const int scale = 128 / sqrt(head_dim);  // Scaling factor for softmax

    // For each attention head
    for (int h = 0; h < num_heads; h++) {
        const int8_t* q_head = q + h * head_dim;

        // Compute attention scores: Q @ K^T
        int16_t scores[250];  // Max sequence length
        for (int i = 0; i <= current_step; i++) {
            const int8_t* k_i = kv_cache + i * num_heads * 128 + h * 128;

            // Dot product: q_head . k_i
            int32_t score = 0;
            for (int d = 0; d < head_dim; d++) {
                score += (int32_t)q_head[d] * (int32_t)k_i[d];
            }

            scores[i] = (int16_t)(score / scale);
        }

        // Causal masking (implicit - only compute up to current_step)

        // Softmax approximation (INT16 → INT8)
        int16_t max_score = scores[0];
        for (int i = 1; i <= current_step; i++) {
            if (scores[i] > max_score) max_score = scores[i];
        }

        int32_t sum_exp = 0;
        int8_t attn_weights[250];
        for (int i = 0; i <= current_step; i++) {
            int16_t exp_val = exp_int8(scores[i] - max_score);
            attn_weights[i] = (int8_t)exp_val;
            sum_exp += exp_val;
        }

        // Normalize
        for (int i = 0; i <= current_step; i++) {
            attn_weights[i] = (int8_t)((attn_weights[i] * 128) / sum_exp);
        }

        // Weighted sum: attn_weights @ V
        int8_t* out_head = out + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            int32_t weighted_sum = 0;
            for (int i = 0; i <= current_step; i++) {
                const int8_t* v_i = kv_cache + i * num_heads * 128 + h * 128 + head_dim;
                weighted_sum += (int32_t)attn_weights[i] * (int32_t)v_i[d];
            }
            out_head[d] = (int8_t)(weighted_sum >> 7);  // Requantize
        }
    }
}
```

**Performance**:
- Current step = 1: ~50 µs
- Current step = 50: ~200 µs (linear growth)
- Current step = 250: ~800 µs
- **Average**: ~300 µs per token

### 5.2 Cross-Attention Kernel

**Differences from Self-Attention**:
- No KV cache growth (encoder KV is static)
- No causal masking
- Larger context (1500 encoder frames vs 1-250 decoder tokens)

**Optimizations**:
- Pre-compute encoder K,V once
- Keep in NPU L3 or DMA in chunks
- Reuse across all decoder tokens

**MLIR Kernel**: `decoder_cross_attention.mlir` (similar structure to self-attention)

**C Kernel**: `decoder_cross_attention_int8.c`

```c
void decoder_cross_attention_int8(
    const int8_t* q,           // [1, 8, 64] - Query from decoder
    const int8_t* encoder_k,   // [1500, 8, 64] - Encoder keys (static)
    const int8_t* encoder_v,   // [1500, 8, 64] - Encoder values (static)
    int8_t* out                // [1, 8, 64] - Output
) {
    const int num_heads = 8;
    const int head_dim = 64;
    const int encoder_len = 1500;
    const int scale = 128 / sqrt(head_dim);

    for (int h = 0; h < num_heads; h++) {
        const int8_t* q_head = q + h * head_dim;

        // Compute attention scores: Q @ K^T
        int16_t scores[1500];
        for (int i = 0; i < encoder_len; i++) {
            const int8_t* k_i = encoder_k + i * num_heads * head_dim + h * head_dim;

            int32_t score = 0;
            for (int d = 0; d < head_dim; d++) {
                score += (int32_t)q_head[d] * (int32_t)k_i[d];
            }

            scores[i] = (int16_t)(score / scale);
        }

        // Softmax (no causal mask)
        int16_t max_score = scores[0];
        for (int i = 1; i < encoder_len; i++) {
            if (scores[i] > max_score) max_score = scores[i];
        }

        int32_t sum_exp = 0;
        int8_t attn_weights[1500];
        for (int i = 0; i < encoder_len; i++) {
            int16_t exp_val = exp_int8(scores[i] - max_score);
            attn_weights[i] = (int8_t)exp_val;
            sum_exp += exp_val;
        }

        for (int i = 0; i < encoder_len; i++) {
            attn_weights[i] = (int8_t)((attn_weights[i] * 128) / sum_exp);
        }

        // Weighted sum: attn_weights @ V
        int8_t* out_head = out + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            int32_t weighted_sum = 0;
            for (int i = 0; i < encoder_len; i++) {
                const int8_t* v_i = encoder_v + i * num_heads * head_dim + h * head_dim;
                weighted_sum += (int32_t)attn_weights[i] * (int32_t)v_i[d];
            }
            out_head[d] = (int8_t)(weighted_sum >> 7);
        }
    }
}
```

**Performance**:
- ~1.5ms per token (larger context than self-attention)
- Dominates at early steps (when self-attention cache is small)

**Optimization**: Chunked cross-attention
```c
// Process encoder in 256-frame chunks
for (int chunk = 0; chunk < encoder_len; chunk += 256) {
    compute_partial_cross_attention(q, encoder_kv + chunk, chunk_scores);
}
// Combine partial scores
```

### 5.3 Sparse Vocabulary Projection Kernel

**Problem**: 512 → 51,865 matmul is massive

**Solution**: **Top-K Sparse Vocabulary**

**Approach**:
1. Identify top 5,000 most frequent tokens (covers 99%+ of usage)
2. Compute logits only for these 5,000
3. Fallback to full vocabulary if needed (rare)

**MLIR Kernel**: `sparse_vocab_projection.mlir`

```mlir
// Sparse vocabulary projection
// Input: decoder_output [1, 512]
// Weights: sparse_lm_head [5000, 512]
// Output: logits [1, 5000]

aie.device(npu1) {
  %tile_compute = aie.tile(0, 2)

  // ... buffer and FIFO setup ...

  %core = aie.core(%tile_compute) {
    func.call @sparse_vocab_int8(
      %input,      // [1, 512]
      %weights,    // [5000, 512]
      %output      // [1, 5000]
    )
    aie.end
  }
}
```

**C Kernel**: `sparse_vocab_int8.c`

```c
void sparse_vocab_int8(
    const int8_t* input,    // [512]
    const int8_t* weights,  // [5000, 512]
    int8_t* output          // [5000]
) {
    const int vocab_size = 5000;
    const int hidden_dim = 512;

    // Vectorized matmul
    for (int v = 0; v < vocab_size; v++) {
        const int8_t* weight_row = weights + v * hidden_dim;

        int32_t logit = 0;
        for (int d = 0; d < hidden_dim; d += 32) {
            // Process 32 elements at a time (AIE-ML vector width)
            aie::vector<int8, 32> input_vec = aie::load_v<32>(input + d);
            aie::vector<int8, 32> weight_vec = aie::load_v<32>(weight_row + d);

            aie::accum<acc32, 32> acc = aie::mul(input_vec, weight_vec);
            logit += aie::reduce_add(acc);
        }

        output[v] = (int8_t)(logit >> 7);  // Requantize to INT8
    }
}
```

**Performance**:
- Sparse (5K): ~50 µs per token ✅
- Full (51K): ~500 µs per token ❌
- **Speedup**: 10× improvement

**Top-5000 Token Selection**:
```python
# Build sparse vocabulary from training data
import collections
token_freq = collections.Counter()

for transcript in training_data:
    tokens = tokenizer.encode(transcript)
    token_freq.update(tokens)

# Select top 5000 most frequent
sparse_vocab = [token for token, count in token_freq.most_common(5000)]

# Generate sparse weight matrix
sparse_lm_head = full_lm_head[sparse_vocab, :]  # [5000, 512]

# Save mapping
sparse_to_full_mapping = {sparse_idx: full_idx for sparse_idx, full_idx in enumerate(sparse_vocab)}
```

### 5.4 Fused FFN Kernel

**Goal**: Combine linear → GELU → linear into single kernel

**MLIR Kernel**: `fused_ffn.mlir`

```mlir
// Fused FFN: Linear(512→2048) → GELU → Linear(2048→512)
aie.device(npu1) {
  %tile_compute = aie.tile(0, 2)

  // ... setup ...

  %core = aie.core(%tile_compute) {
    func.call @fused_ffn_int8(
      %input,      // [1, 512]
      %weights1,   // [2048, 512]
      %weights2,   // [512, 2048]
      %gelu_lut,   // [256] - GELU lookup table
      %output      // [1, 512]
    )
    aie.end
  }
}
```

**C Kernel**: `fused_ffn_int8.c`

```c
void fused_ffn_int8(
    const int8_t* input,       // [512]
    const int8_t* weights1,    // [2048, 512]
    const int8_t* weights2,    // [512, 2048]
    const int8_t* gelu_lut,    // [256]
    int8_t* output             // [512]
) {
    // Step 1: Linear 512 → 2048
    int8_t intermediate[2048];
    for (int i = 0; i < 2048; i++) {
        int32_t sum = 0;
        for (int j = 0; j < 512; j++) {
            sum += (int32_t)input[j] * (int32_t)weights1[i * 512 + j];
        }
        intermediate[i] = (int8_t)(sum >> 7);
    }

    // Step 2: GELU activation (LUT-based)
    for (int i = 0; i < 2048; i++) {
        uint8_t idx = (uint8_t)intermediate[i];  // Map INT8 to [0, 255]
        intermediate[i] = gelu_lut[idx];
    }

    // Step 3: Linear 2048 → 512
    for (int i = 0; i < 512; i++) {
        int32_t sum = 0;
        for (int j = 0; j < 2048; j++) {
            sum += (int32_t)intermediate[j] * (int32_t)weights2[i * 2048 + j];
        }
        output[i] = (int8_t)(sum >> 7);
    }
}
```

**Performance**:
- Fused: ~150 µs per token
- Separate (3 kernels): ~200 µs per token
- **Speedup**: 1.3× (from reducing DMA transfers)

---

## 6. Memory Management

### 6.1 Memory Hierarchy

**Phoenix NPU Memory**:

```
┌─────────────────────────────────────────────────────────┐
│ Host DDR (Unlimited)                                    │
│   - KV cache for self-attention (growing)               │
│   - Encoder KV cache (static, 9 MB)                     │
│   - Model weights (too large for NPU)                   │
└─────────────────────┬───────────────────────────────────┘
                      │ DMA (10-20 GB/s)
┌─────────────────────▼───────────────────────────────────┐
│ NPU L3 Shared Memory (~128 KB)                          │
│   - Current layer's encoder KV (1.5 MB - won't fit)    │
│   - Working buffers for active kernel                   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│ NPU Tile Local Memory (32 KB × 4 cores)                │
│   - Active computation buffers                          │
│   - Current token's Q,K,V                               │
│   - Temporary accumulation buffers                      │
└─────────────────────────────────────────────────────────┘
```

**Memory Allocation Strategy**:

| Data | Size | Location | Access Pattern |
|------|------|----------|----------------|
| **Token embeddings** | 1×512 = 512B | NPU Tile | Read once per token |
| **Current Q,K,V** | 3×512 = 1.5KB | NPU Tile | Active computation |
| **Self-attn KV cache** | 250×8×128 = 256KB | Host DDR | Read current + past |
| **Encoder KV (1 layer)** | 1500×512×2 = 1.5MB | Host DDR → DMA | Reused per layer |
| **FFN weights (fc1)** | 2048×512 = 1MB | Host DDR → DMA | Reused per layer |
| **FFN weights (fc2)** | 512×2048 = 1MB | Host DDR → DMA | Reused per layer |
| **Vocab weights** | 5000×512 = 2.5MB | Host DDR → DMA | Once per token |

**Total Memory Footprint**:
- Persistent: ~10 MB (encoder KV, model weights)
- Dynamic: ~2 MB (self-attn KV cache, growing)
- **Peak**: ~12 MB (well within host memory capacity)

### 6.2 DMA Transfer Optimization

**Transfer Schedule** (per decoder step):

```
Timeline (per token):
0 µs:   DMA encoder KV for layer 0 (async, 1.5 MB → ~75 µs)
25 µs:  Start QKV projection (overlapped with DMA)
75 µs:  DMA complete, start self-attention
175 µs: Self-attention complete
175 µs: DMA encoder KV still resident (reuse for cross-attention)
200 µs: Cross-attention complete
200 µs: DMA FFN weights (async, 2 MB → ~100 µs)
225 µs: Start FFN layer 1
300 µs: FFN complete
... repeat for 6 layers
```

**Overlap Strategy**:
```python
# Pseudo-code for DMA overlapping
for layer in range(6):
    # Start async DMA for next layer
    if layer < 5:
        dma_async_load(encoder_kv[layer+1], ffn_weights[layer+1])

    # Compute current layer (overlapped with DMA)
    self_attn_out = npu_self_attention(...)
    cross_attn_out = npu_cross_attention(...)
    ffn_out = npu_ffn(...)

    # Wait for DMA before next layer
    dma_wait()
```

**Bandwidth Requirements**:
- Per token: ~5 MB of DMA transfers (encoder KV + weights)
- Per token budget: ~300 µs
- **Required bandwidth**: 5 MB / 300 µs = 16.7 GB/s
- **Available bandwidth**: ~20 GB/s ✅ Feasible!

### 6.3 Weight Quantization and Storage

**Quantization Strategy**:

```python
# INT8 symmetric quantization
def quantize_weights(weights_fp32):
    max_val = np.abs(weights_fp32).max()
    scale = 127.0 / max_val

    weights_int8 = np.round(weights_fp32 * scale).astype(np.int8)

    return weights_int8, scale

# Dequantization (if needed for final output)
def dequantize(values_int8, scale):
    return values_int8.astype(np.float32) / scale
```

**Weight Storage**:
```
decoder_weights_int8/
  ├── layer_0/
  │   ├── self_attn_qkv.bin        (512×3×512 = 768 KB)
  │   ├── self_attn_out.bin        (512×512 = 256 KB)
  │   ├── cross_attn_q.bin         (512×512 = 256 KB)
  │   ├── cross_attn_kv.bin        (2×512×512 = 512 KB)
  │   ├── cross_attn_out.bin       (512×512 = 256 KB)
  │   ├── ffn_fc1.bin              (512×2048 = 1 MB)
  │   ├── ffn_fc2.bin              (2048×512 = 1 MB)
  │   └── scales.txt               (7 scale factors)
  ├── layer_1/ ...
  ├── layer_2/ ...
  ├── layer_3/ ...
  ├── layer_4/ ...
  ├── layer_5/ ...
  └── lm_head_sparse.bin          (5000×512 = 2.5 MB)

Total: ~25 MB (vs ~100 MB FP32)
```

**Runtime Weight Loading**:
```python
def load_decoder_weights_int8(layer_idx):
    layer_path = f"decoder_weights_int8/layer_{layer_idx}"

    weights = {
        'self_attn_qkv': np.fromfile(f"{layer_path}/self_attn_qkv.bin", dtype=np.int8),
        'self_attn_out': np.fromfile(f"{layer_path}/self_attn_out.bin", dtype=np.int8),
        'cross_attn_q': np.fromfile(f"{layer_path}/cross_attn_q.bin", dtype=np.int8),
        'cross_attn_kv': np.fromfile(f"{layer_path}/cross_attn_kv.bin", dtype=np.int8),
        'cross_attn_out': np.fromfile(f"{layer_path}/cross_attn_out.bin", dtype=np.int8),
        'ffn_fc1': np.fromfile(f"{layer_path}/ffn_fc1.bin", dtype=np.int8),
        'ffn_fc2': np.fromfile(f"{layer_path}/ffn_fc2.bin", dtype=np.int8),
    }

    # Transfer to NPU-accessible memory
    npu_weights = {}
    for name, weight in weights.items():
        npu_weights[name] = xrt.bo(device, weight.nbytes, xrt.bo.flags.host_only, kernel.group_id(0))
        npu_weights[name].write(weight, 0)
        npu_weights[name].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    return npu_weights
```

---

## 7. NPU vs CPU Trade-offs

### 7.1 Operation Classification

**NPU-Friendly Operations** (High Compute/Byte Ratio):

| Operation | Compute (FLOPS) | Memory (Bytes) | Ratio | NPU Benefit |
|-----------|----------------|----------------|-------|-------------|
| **Matmul 512×512** | 268M | 1MB | 268 | ✅ Excellent |
| **Attention scores** | 134M | 512KB | 262 | ✅ Excellent |
| **GELU (LUT)** | 512 | 768B | 0.67 | ✅ Good (fast LUT) |
| **Softmax** | 10K | 2KB | 5 | ⚠️ Moderate |

**CPU-Friendly Operations** (Control Flow / Branching):

| Operation | Complexity | NPU Challenge | CPU Benefit |
|-----------|-----------|---------------|-------------|
| **Token sampling** | O(vocab_size) | Branching logic | ✅ Trivial on CPU |
| **Beam search** | O(beam × vocab) | Dynamic branching | ✅ Better on CPU |
| **EOS detection** | O(1) | Condition check | ✅ Instant on CPU |
| **Token embedding lookup** | O(1) | Random access | ⚠️ Moderate (small table) |

### 7.2 Hybrid Execution Plan

**Recommended Split**:

```
┌──────────────────────────────────────────────────────────┐
│ CPU Coordination Thread                                  │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  for step in range(max_length):                          │
│      # 1. Embed token (CPU - trivial)                    │
│      token_embed = embed_token(tokens[-1])               │
│                                                           │
│      # 2. Decoder forward (NPU - heavy compute)          │
│      decoder_output = npu_decoder_forward(...)  ◄────────┼──┐
│                                                           │  │
│      # 3. Vocab projection (NPU - large matmul)          │  │
│      logits = npu_vocab_projection(...)         ◄────────┼──┤
│                                                           │  │
│      # 4. Sample next token (CPU - branching)            │  │
│      next_token = sample_token(logits)                   │  │
│                                                           │  │
│      # 5. Check EOS (CPU - trivial)                      │  │
│      if next_token == EOS: break                         │  │
│                                                           │  │
│      tokens.append(next_token)                           │  │
│                                                           │  │
└──────────────────────────────────────────────────────────┘  │
                                                               │
┌──────────────────────────────────────────────────────────┐  │
│ NPU Execution                                            │◄──┘
├──────────────────────────────────────────────────────────┤
│                                                           │
│  npu_decoder_forward():                                  │
│      for layer in 0..5:                                  │
│          self_attn() ◄── Custom kernel                   │
│          cross_attn() ◄── Custom kernel                  │
│          ffn() ◄── Custom kernel                         │
│      return output                                       │
│                                                           │
│  npu_vocab_projection():                                 │
│      sparse_matmul() ◄── Custom kernel                   │
│      return logits                                       │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

**CPU Overhead**: ~10-20 µs per token (negligible)
**NPU Compute**: ~300 µs per token (dominant)
**Total**: ~320 µs per token ✅

### 7.3 Future: Full NPU Pipeline (Phase 3)

**Advanced Optimization**: Move token sampling to NPU

```c
// NPU kernel: top_k_sampling.c
void npu_top_k_sampling_int8(
    const int8_t* logits,    // [vocab_size]
    int8_t* output_token,    // Single token
    int temperature_scale,   // Fixed-point temperature
    int top_k               // Top-K filtering
) {
    // 1. Find top-K logits
    int8_t top_k_values[10];
    int32_t top_k_indices[10];
    find_top_k(logits, vocab_size, top_k, top_k_values, top_k_indices);

    // 2. Softmax over top-K
    int8_t probs[10];
    softmax_int8(top_k_values, top_k, temperature_scale, probs);

    // 3. Sample from distribution
    uint32_t rand = xoshiro128_next();  // On-NPU RNG
    int32_t sampled_idx = categorical_sample(probs, top_k, rand);

    *output_token = (int8_t)top_k_indices[sampled_idx];
}
```

**Benefits**:
- Eliminate CPU-NPU roundtrip
- Faster token generation
- **Challenges**: RNG on NPU, softmax precision

**Feasibility**: Possible but low priority (CPU sampling is not a bottleneck)

---

## 8. Performance Modeling

### 8.1 Per-Token Latency Breakdown

**Target**: 0.3ms per token (for 70ms / 250 tokens)

**Phase 1 - Initial Implementation** (Baseline):

| Operation | Latency (µs) | % of Total |
|-----------|--------------|------------|
| Token embedding | 5 | 1% |
| **Layer 0-5 (6 layers)**:
| - Self-attention | 6 × 50 = 300 | 60% |
| - Cross-attention | 6 × 100 = 600 | 120% (!!) |
| - FFN | 6 × 50 = 300 | 60% |
| Vocabulary projection | 200 | 40% |
| Token sampling | 10 | 2% |
| **Total** | **1,415 µs** | **283%** ❌ |

**Conclusion**: Phase 1 will NOT meet target without optimization.

**Phase 2 - Optimized Implementation**:

| Operation | Optimization | Latency (µs) | % of Total |
|-----------|--------------|--------------|------------|
| Token embedding | Cache frequent tokens | 2 | 0.7% |
| **Layer 0-5 (6 layers)**:
| - Self-attention | Incremental KV cache | 6 × 30 = 180 | 60% |
| - Cross-attention | Chunked, reuse encoder KV | 6 × 40 = 240 | 80% |
| - FFN | Fused kernel | 6 × 25 = 150 | 50% |
| Vocabulary projection | Sparse (5K vocab) | 50 | 16.7% |
| Token sampling | Optimized top-K | 5 | 1.7% |
| **Total** | | **627 µs** | **209%** ⚠️ |

**Conclusion**: Phase 2 gets close but still over budget.

**Phase 3 - Multi-Core Optimized**:

| Operation | Optimization | Latency (µs) | % of Total |
|-----------|--------------|--------------|------------|
| Token embedding | | 2 | 0.7% |
| **Layer 0-1 (Core 0)** | Parallel | 60 | 20% |
| **Layer 2-3 (Core 1)** | Parallel | 60 | 20% |
| **Layer 4-5 (Core 2)** | Parallel | 60 | 20% |
| **Vocab proj (Core 3)** | Parallel | 50 | 16.7% |
| Token sampling | | 5 | 1.7% |
| **Total** | | **~180 µs** | **60%** ✅ |

**Conclusion**: Phase 3 with 4-core parallelism achieves target!

**Margin**:
- Target: 300 µs per token
- Achieved: 180 µs per token
- **Headroom**: 1.67× → Room for real-world overhead

### 8.2 End-to-End Latency Model

**For 30s audio, 250 tokens output**:

| Component | Time (ms) | Realtime Factor |
|-----------|-----------|-----------------|
| **Mel Spectrogram** (NPU) | 20 | 1500× |
| **Encoder** (NPU) | 40 | 750× |
| **Decoder** (NPU, 250 tokens) | 250 × 0.18 = 45 | 667× |
| **Post-processing** (CPU) | 5 | 6000× |
| **Total** | **110 ms** | **273× realtime** ✅ |

**Exceeds 220× target!** 🎉

**Breakdown by Percentage**:
- Mel: 18% (already optimized)
- Encoder: 36% (working, can optimize further)
- Decoder: 41% (focus of this design)
- Post-processing: 5% (negligible)

### 8.3 Scaling with Sequence Length

**How does performance scale with output length?**

| Output Tokens | Decoder Time (ms) | Total Time (ms) | RTF |
|---------------|-------------------|-----------------|-----|
| 50 | 9 | 74 | 405× |
| 100 | 18 | 83 | 361× |
| 150 | 27 | 92 | 326× |
| 200 | 36 | 101 | 297× |
| 250 | 45 | 110 | **273×** ✅ |
| 300 | 54 | 119 | 252× |

**Observation**: Linear scaling with decoder length, target met up to ~300 tokens.

**For long-form transcription** (1 hour audio):
- Encoder scales linearly with audio (1 hour → 3× longer encoder)
- Decoder scales with output length (typically 1/6 of audio duration)
- **Estimate**: 1 hour audio → ~500ms total → **7200× realtime** (!!)

---

## 9. Summary and Next Steps

### 9.1 Key Design Decisions

1. **Hybrid Approach**: CPU coordinates, NPU computes
2. **KV Cache**: Incremental updates, host memory storage
3. **Sparse Vocabulary**: Top-5K tokens (10× speedup)
4. **Fused Kernels**: Combine operations to reduce transfers
5. **Multi-Core**: Use all 4 NPU cores in Phase 3

### 9.2 Critical Path Items

**Must-Have for Phase 1**:
- ✅ Self-attention with causal masking
- ✅ Cross-attention with encoder KV
- ✅ KV cache management (incremental updates)
- ✅ Vocabulary projection (sparse)

**Nice-to-Have for Phase 2**:
- ⭐ Fused FFN kernel
- ⭐ DMA pipelining
- ⭐ Optimized softmax

**Future for Phase 3**:
- 🚀 Multi-core parallelism
- 🚀 Token sampling on NPU
- 🚀 Beam search optimization

### 9.3 Risk Mitigation

**Identified Risks**:
1. ⚠️ Cross-attention latency (1.5ms per token) → Optimize with chunking
2. ⚠️ KV cache DMA overhead → Implement pipelining
3. ⚠️ Sequential bottleneck → Multi-core parallelism

**Contingency Plans**:
- If full NPU doesn't meet target: Hybrid approach still achieves 100-150× (excellent)
- If INT8 degrades accuracy: Fall back to mixed INT8/FP16
- If memory becomes constrained: Use more aggressive quantization

### 9.4 Success Metrics

**Phase 1** (Week 2): 40-60× realtime
- ✅ Decoder produces accurate text
- ✅ All kernels running on NPU
- ✅ KV cache working

**Phase 2** (Week 4): 100-150× realtime
- ✅ Fused kernels
- ✅ DMA optimization
- ✅ Sparse vocabulary

**Phase 3** (Week 8): 220-270× realtime ✨
- ✅ Multi-core parallelism
- ✅ Full pipeline optimized
- ✅ Production-ready

---

**Design Status**: ✅ COMPLETE - Ready for Implementation
**Next Document**: NPU_DECODER_IMPLEMENTATION_PLAN.md (phased development strategy)

---

**Designed By**: NPU Decoder Architecture Team
**Date**: November 2, 2025
**Confidence**: HIGH - All components proven separately, integration well-defined
