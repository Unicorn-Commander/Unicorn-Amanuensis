# NPU Whisper Decoder - Encoder Integration Guide

**Date**: November 2, 2025
**Integration Lead**: NPU Decoder Team
**Status**: Integration Design Complete
**Purpose**: How encoder and decoder work together on NPU

---

## Overview

This document describes how the NPU encoder and decoder integrate to form a complete Whisper transcription pipeline.

**Key Principle**: Encoder runs **once** per audio file, decoder runs **iteratively** (token-by-token).

**Data Flow**:
```
Audio → Mel (NPU) → Encoder (NPU) → [Encoder Hidden States] → Decoder (NPU, iterative) → Text
```

---

## End-to-End Pipeline

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Audio Processing (One-time)                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Audio File (30s)                                               │
│       ↓                                                          │
│  [NPU] Mel Spectrogram (36.1x realtime)                        │
│       → Output: [1500, 80] mel features                         │
│       ↓                                                          │
│  [NPU] Encoder (6 layers)                                       │
│       → Self-attention, FFN for all frames                      │
│       → Output: [1500, 512] encoder hidden states               │
│       ↓                                                          │
│  [NPU] Pre-compute Encoder K,V for Cross-Attention             │
│       → For each layer: project to K,V                          │
│       → Output: 6 × [1500, 512] K,V pairs                       │
│       → Cache in NPU-accessible memory                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Text Generation (Iterative, 250 steps)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [CPU] Initialize: tokens = [START_TOKEN]                       │
│                                                                  │
│  for step in range(max_length):                                 │
│      │                                                           │
│      [CPU] Embed current token                                  │
│      │   → token_embed = embedding[tokens[-1]]                  │
│      │                                                           │
│      [NPU] Decoder Forward Pass                                 │
│      │   → Self-attention (with KV cache)                       │
│      │   → Cross-attention (with encoder K,V)  ◄───────┐       │
│      │   → FFN                                          │       │
│      │   → Output: [1, 512]                             │       │
│      │                                           Uses encoder   │
│      [NPU] Vocabulary Projection                   hidden states│
│      │   → Linear: [1, 512] → [1, 5000]                │       │
│      │   → Output: logits                               │       │
│      │                                                   │       │
│      [CPU] Sample next token                            │       │
│      │   → next_token = sample(logits)                 │       │
│      │                                                   │       │
│      [CPU] Check EOS                                    │       │
│      │   → if next_token == EOS: break                 │       │
│      │                                                   │       │
│      tokens.append(next_token)                          │       │
│                                                          │       │
│  Output: tokens = [START, tok1, tok2, ..., tokN, EOS]  │       │
│                                                          │       │
│  [CPU] Decode to text                                   │       │
│       → text = tokenizer.decode(tokens)                 │       │
│                                                          │       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Transfer Points

### 1. Encoder → Decoder: Hidden States

**What**: Encoder output (contextualized audio features)
**Shape**: `[num_frames, hidden_dim]` = `[1500, 512]`
**Size**: 1500 × 512 × 1 byte (INT8) = 768 KB
**Transfer**: Once per audio file
**Storage**: Host memory or NPU L3 (if fits)

**Code**:
```python
# After encoder completes
encoder_hidden_states = npu_encoder(mel_features)  # [1500, 512]

# Store for decoder access
np.save('encoder_output.npy', encoder_hidden_states)

# Or keep in memory
decoder_input = {'encoder_states': encoder_hidden_states}
```

### 2. Encoder → Decoder: Cross-Attention K,V Cache

**What**: Pre-computed Key and Value matrices for cross-attention
**Why**: Computed once, reused for all 250 decoder steps (huge savings!)
**Shape**: `[num_layers, num_frames, hidden_dim]` = `[6, 1500, 512]`
**Size**: 6 × 1500 × 512 × 2 (K and V) × 1 byte = 9 MB
**Transfer**: Once per audio file
**Storage**: Host memory with DMA on-demand

**Code**:
```python
def precompute_encoder_kv(encoder_hidden_states, decoder_layers):
    """
    Pre-compute K,V for cross-attention (one-time cost)

    This eliminates recomputation for every decoder step!
    """
    encoder_kv_cache = {}

    for layer_idx in range(len(decoder_layers)):
        layer = decoder_layers[layer_idx]

        # Project encoder states to K,V for cross-attention
        k = npu_matmul(encoder_hidden_states, layer.cross_attn.W_k)
        v = npu_matmul(encoder_hidden_states, layer.cross_attn.W_v)

        encoder_kv_cache[layer_idx] = {
            'key': k,      # [1500, 512]
            'value': v     # [1500, 512]
        }

    return encoder_kv_cache

# Usage
encoder_kv = precompute_encoder_kv(encoder_hidden_states, decoder.layers)

# Now reuse for all 250 decoder steps (no recomputation!)
for step in range(250):
    decoder_output = decoder_forward(..., encoder_kv=encoder_kv)
```

**Savings**:
- Without caching: 6 layers × 2 matmuls × 250 steps = **3,000 matmuls**
- With caching: 6 layers × 2 matmuls × 1 = **12 matmuls**
- **Speedup**: 250× faster cross-attention!

### 3. Decoder Self-Attention KV Cache

**What**: Growing cache of past decoder tokens' K,V
**Why**: Avoid recomputing attention for past tokens
**Shape**: `[num_layers, current_step+1, hidden_dim × 2]`
**Size**: Grows from 0 to `6 × 250 × 1024` = 1.5 MB
**Update**: Every decoder step (incremental)

**Code**:
```python
class DecoderKVCache:
    def __init__(self, num_layers=6, max_length=250, hidden_dim=512):
        self.cache = {}
        for layer in range(num_layers):
            self.cache[layer] = {
                'key': np.zeros((max_length, hidden_dim), dtype=np.int8),
                'value': np.zeros((max_length, hidden_dim), dtype=np.int8),
                'length': 0  # Current cache size
            }

    def append(self, layer, new_k, new_v):
        """Append new token's K,V to cache"""
        idx = self.cache[layer]['length']
        self.cache[layer]['key'][idx] = new_k
        self.cache[layer]['value'][idx] = new_v
        self.cache[layer]['length'] += 1

    def get(self, layer):
        """Get current cache (all past + current)"""
        length = self.cache[layer]['length']
        return (
            self.cache[layer]['key'][:length],   # [current_step+1, 512]
            self.cache[layer]['value'][:length]
        )

# Usage in decoder
kv_cache = DecoderKVCache()

for step in range(max_length):
    # Generate Q,K,V for current token
    q, k, v = decoder_layer.compute_qkv(current_token_embed)

    # Append to cache
    kv_cache.append(layer_idx, k, v)

    # Get full cache (current + past)
    cached_k, cached_v = kv_cache.get(layer_idx)

    # Compute attention with full cache
    attn_output = npu_attention(q, cached_k, cached_v)
```

---

## Memory Layout

### NPU Memory Allocation

**Phoenix NPU Memory**:
```
Row 2 (Compute Cores): 4 × 32KB = 128KB
Row 1 (Memory Tiles):   4 × 32KB = 128KB
Row 0 (ShimNOC/DMA):    Minimal
─────────────────────────────────────────
Total NPU Memory:       ~256KB
```

**What Fits in NPU**:
- ✅ Current token embedding (512 bytes)
- ✅ Current layer weights (loaded on-demand, ~1MB → tiled)
- ✅ Intermediate activations (512 bytes → 2KB)
- ❌ Full encoder hidden states (768KB) → Too large
- ❌ Encoder KV cache (9MB) → Too large
- ❌ Decoder KV cache (1.5MB) → Too large

**Memory Strategy**:
```
┌────────────────────────────────────────────────┐
│ NPU Tile Memory (32KB)                         │
│   - Current token Q,K,V                        │
│   - Attention scores (working buffer)          │
│   - FFN intermediate activations               │
└────────────────────────────────────────────────┘
                    ↕ DMA (frequent)
┌────────────────────────────────────────────────┐
│ Host Memory (DDR, ~16GB)                       │
│   - Encoder hidden states (768KB)              │
│   - Encoder KV cache (9MB)                     │
│   - Decoder KV cache (1.5MB, growing)          │
│   - Model weights (25MB INT8)                  │
│   - Audio buffer                               │
└────────────────────────────────────────────────┘
```

---

## Integration Code Example

### Complete Pipeline

```python
#!/usr/bin/env python3
"""
Complete NPU Whisper Pipeline
Encoder → Decoder Integration
"""

import numpy as np
import torch
from pathlib import Path

# Import NPU components
from whisper_npu_encoder_matmul import WhisperNPUEncoderMatmul
from whisper_npu_decoder_matmul import WhisperNPUDecoderMatmul
from npu_mel_preprocessing import NPUMelPreprocessor

class NPUWhisperPipeline:
    def __init__(self, model_name="base", device_id=0):
        """Complete Whisper pipeline on NPU"""

        # Model config
        config = {
            "base": {"layers": 6, "d_model": 512, "heads": 8, "d_ff": 2048}
        }[model_name]

        # Initialize components
        self.mel_processor = NPUMelPreprocessor()
        self.encoder = WhisperNPUEncoderMatmul(**config, device_id=device_id)
        self.decoder = WhisperNPUDecoderMatmul(**config, device_id=device_id)

        # Cache
        self.encoder_kv_cache = None

    def transcribe(self, audio_path: str, max_length: int = 250):
        """
        Full transcription pipeline

        Args:
            audio_path: Path to audio file
            max_length: Maximum output tokens

        Returns:
            Transcription text
        """
        print(f"\n{'='*70}")
        print(f"TRANSCRIBING: {audio_path}")
        print(f"{'='*70}\n")

        # ═══════════════════════════════════════════════════════════
        # Step 1: Audio → Mel Spectrogram (NPU)
        # ═══════════════════════════════════════════════════════════
        print("[1/4] Computing mel spectrogram (NPU)...")
        mel_features = self.mel_processor.process_audio(audio_path)
        # Shape: [num_frames, 80]
        print(f"      Mel features: {mel_features.shape}")

        # ═══════════════════════════════════════════════════════════
        # Step 2: Mel → Encoder Hidden States (NPU)
        # ═══════════════════════════════════════════════════════════
        print("[2/4] Running encoder (NPU, 6 layers)...")
        encoder_hidden_states = self.encoder(mel_features)
        # Shape: [num_frames, 512]
        print(f"      Encoder output: {encoder_hidden_states.shape}")

        # ═══════════════════════════════════════════════════════════
        # Step 3: Pre-compute Encoder K,V for Cross-Attention (NPU)
        # ═══════════════════════════════════════════════════════════
        print("[3/4] Pre-computing encoder K,V cache (NPU)...")
        self.encoder_kv_cache = self._precompute_encoder_kv(
            encoder_hidden_states
        )
        print(f"      Cached K,V for {len(self.encoder_kv_cache)} layers")

        # ═══════════════════════════════════════════════════════════
        # Step 4: Autoregressive Decoder (NPU + CPU coordination)
        # ═══════════════════════════════════════════════════════════
        print("[4/4] Generating text (NPU decoder, autoregressive)...")
        tokens = self._generate_tokens(max_length)
        print(f"      Generated {len(tokens)} tokens")

        # Decode tokens to text
        text = self._decode_tokens(tokens)

        print(f"\n{'='*70}")
        print(f"TRANSCRIPTION COMPLETE")
        print(f"{'='*70}")
        print(f"Text: {text}")
        print(f"{'='*70}\n")

        return text

    def _precompute_encoder_kv(self, encoder_hidden_states):
        """
        Pre-compute K,V for cross-attention

        Computed ONCE per audio file, reused for ALL 250 decoder steps!

        Args:
            encoder_hidden_states: [num_frames, 512]

        Returns:
            encoder_kv_cache: {layer_idx: {'key': K, 'value': V}}
        """
        num_frames, hidden_dim = encoder_hidden_states.shape
        encoder_kv_cache = {}

        for layer_idx in range(self.decoder.num_layers):
            decoder_layer = self.decoder.layers[layer_idx]

            # Project encoder states to K,V
            # Use cross-attention K,V projection weights
            k = decoder_layer.encoder_attn.k_proj(encoder_hidden_states)
            v = decoder_layer.encoder_attn.v_proj(encoder_hidden_states)

            encoder_kv_cache[layer_idx] = {
                'key': k,      # [num_frames, 512]
                'value': v     # [num_frames, 512]
            }

        return encoder_kv_cache

    def _generate_tokens(self, max_length: int):
        """
        Autoregressive token generation

        Args:
            max_length: Maximum tokens to generate

        Returns:
            List of token IDs
        """
        tokens = [self._get_start_token()]
        decoder_kv_cache = self._init_decoder_kv_cache()

        for step in range(max_length):
            # Embed current token (CPU)
            current_token_id = tokens[-1]
            token_embed = self._embed_token(current_token_id)

            # Decoder forward pass (NPU)
            decoder_output, decoder_kv_cache = self._decoder_step(
                token_embed,
                decoder_kv_cache,
                step
            )

            # Vocabulary projection (NPU)
            logits = self._vocab_projection(decoder_output)

            # Sample next token (CPU)
            next_token = self._sample_token(logits)

            # Check stopping criteria (CPU)
            if next_token == self._get_end_token():
                break

            tokens.append(next_token)

        return tokens

    def _decoder_step(self, token_embed, decoder_kv_cache, step):
        """
        Single decoder forward pass

        Args:
            token_embed: [1, 512] - Current token embedding
            decoder_kv_cache: KV cache for self-attention
            step: Current generation step

        Returns:
            decoder_output: [1, 512]
            updated_kv_cache: Updated KV cache
        """
        x = token_embed

        # Pass through 6 decoder layers
        for layer_idx, layer in enumerate(self.decoder.layers):
            # Self-attention (with KV cache)
            x = layer.self_attn(
                x,
                past_kv=decoder_kv_cache[layer_idx],
                step=step
            )

            # Update decoder KV cache
            decoder_kv_cache[layer_idx] = layer.self_attn.get_kv_cache()

            # Cross-attention (with encoder K,V)
            x = layer.encoder_attn(
                x,
                encoder_kv=self.encoder_kv_cache[layer_idx]
            )

            # FFN
            x = layer.ffn(x)

        return x, decoder_kv_cache

    def _init_decoder_kv_cache(self):
        """Initialize empty KV cache for decoder"""
        return {layer_idx: None for layer_idx in range(self.decoder.num_layers)}

    def _embed_token(self, token_id):
        """Embed token (CPU array lookup)"""
        # In real implementation, load from model
        return torch.randn(1, 512)  # Placeholder

    def _vocab_projection(self, decoder_output):
        """Project decoder output to vocabulary (NPU)"""
        # In real implementation, use sparse vocab projection kernel
        return torch.randn(1, 5000)  # Placeholder logits

    def _sample_token(self, logits):
        """Sample next token from logits (CPU)"""
        # Greedy decoding for now
        return int(torch.argmax(logits))

    def _decode_tokens(self, tokens):
        """Decode token IDs to text (CPU)"""
        # In real implementation, use tokenizer
        return f"[Generated {len(tokens)} tokens]"

    def _get_start_token(self):
        return 50258  # <|startoftranscript|>

    def _get_end_token(self):
        return 50257  # <|endoftext|>


# ═══════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = NPUWhisperPipeline(model_name="base", device_id=0)

    # Transcribe
    text = pipeline.transcribe("test_audio.wav", max_length=250)

    print(f"\nFinal transcription: {text}")
```

---

## Performance Optimization

### Encoder-Decoder Interface Optimization

**1. Pre-compute Encoder K,V** (already discussed)
- **Savings**: 250× fewer matmuls
- **Cost**: One-time 12 matmuls
- **Status**: ✅ Critical optimization

**2. Pin Encoder K,V in Memory**
```python
# Keep encoder K,V resident (avoid repeated DMA)
encoder_kv_pinned = {}
for layer_idx, kv in encoder_kv_cache.items():
    encoder_kv_pinned[layer_idx] = {
        'key': pin_memory(kv['key']),
        'value': pin_memory(kv['value'])
    }
```
- **Savings**: Avoid 250 × 6 = 1,500 DMA transfers
- **Cost**: 9MB pinned memory
- **Status**: ✅ Recommended

**3. Chunked Cross-Attention**
```python
# Process encoder in 256-frame chunks (fits in NPU L3)
chunk_size = 256
for chunk_start in range(0, num_frames, chunk_size):
    chunk_kv = encoder_kv[chunk_start:chunk_start+chunk_size]
    partial_scores = npu_cross_attention(q, chunk_kv)
    all_scores.append(partial_scores)

# Combine
final_output = combine_attention_outputs(all_scores)
```
- **Benefit**: Fits in NPU L3 memory
- **Cost**: Slightly more complex kernel
- **Status**: ⭐ Phase 2 optimization

---

## Monitoring and Debugging

### Integration Validation

```python
def validate_encoder_decoder_integration():
    """Test encoder → decoder data flow"""

    # 1. Test encoder output shape
    encoder_output = encoder(mel_features)
    assert encoder_output.shape == (1500, 512), "Encoder shape mismatch"

    # 2. Test encoder KV cache
    encoder_kv = precompute_encoder_kv(encoder_output)
    assert len(encoder_kv) == 6, "Should have 6 layers"
    for layer_kv in encoder_kv.values():
        assert layer_kv['key'].shape == (1500, 512)
        assert layer_kv['value'].shape == (1500, 512)

    # 3. Test decoder accepts encoder output
    decoder_output = decoder.forward_single_step(
        token_embed=torch.randn(1, 512),
        encoder_kv=encoder_kv
    )
    assert decoder_output.shape == (1, 512), "Decoder shape mismatch"

    print("✅ Encoder-decoder integration validated!")
```

### Performance Profiling

```python
import time

def profile_encoder_decoder():
    """Profile encoder → decoder pipeline"""

    # Encoder
    start = time.perf_counter()
    encoder_output = encoder(mel_features)
    encoder_time = time.perf_counter() - start

    # Encoder KV
    start = time.perf_counter()
    encoder_kv = precompute_encoder_kv(encoder_output)
    encoder_kv_time = time.perf_counter() - start

    # Decoder (single step)
    start = time.perf_counter()
    decoder_output = decoder.forward_single_step(token_embed, encoder_kv)
    decoder_time = time.perf_counter() - start

    print(f"Encoder:        {encoder_time*1000:.2f}ms")
    print(f"Encoder K,V:    {encoder_kv_time*1000:.2f}ms")
    print(f"Decoder (step): {decoder_time*1000:.2f}ms")
    print(f"Decoder (250):  {decoder_time*250*1000:.2f}ms")
    print(f"Total:          {(encoder_time + encoder_kv_time + decoder_time*250)*1000:.2f}ms")

    rtf = 30000 / ((encoder_time + encoder_kv_time + decoder_time*250) * 1000)
    print(f"Realtime factor: {rtf:.1f}x")
```

---

## Conclusion

**Key Takeaways**:
1. ✅ Encoder runs **once**, decoder runs **250 times** (iterative)
2. ✅ Pre-computing encoder K,V saves **250× cross-attention compute**
3. ✅ KV caching is **critical** for both cross-attention and self-attention
4. ✅ Hybrid NPU/CPU approach leverages strengths of each
5. ✅ Memory management strategy: NPU for compute, host for large storage

**Performance Target**:
- Encoder: 40ms
- Encoder KV: 10ms
- Decoder: 250 × 0.2ms = 50ms
- **Total**: 100ms for 30s audio = **300× realtime** ✨

**Next Steps**:
1. Implement integration in `npu_whisper_integration_example.py`
2. Test with real audio
3. Profile and optimize
4. Deploy to production

---

**Document Prepared By**: NPU Integration Team
**Date**: November 2, 2025
**Status**: ✅ Integration Design Complete
**Ready for**: Implementation (Phase 1, Week 1-2)
