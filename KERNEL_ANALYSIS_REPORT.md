# Whisper Encoder Kernel Analysis Report
## AMD Phoenix NPU (XDNA1) Kernel Implementation Gap Analysis

**Date**: November 20, 2025  
**Analysis Target**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1/`  
**Hardware**: AMD Phoenix NPU (XDNA1, 4×6 tile array, 16 TOPS INT8)

---

## Executive Summary

**Current Status**: Foundation kernels exist, but full encoder layer requires additional functions
- **3 Basic Kernels Available**: LayerNorm, Softmax, GELU
- **2 MatMul Implementations**: Vectorized and scalar versions
- **Critical Gaps**: Attention mechanism, residual connections, multi-head orchestration
- **Memory Bottleneck**: Most kernels hardcoded for 1024-element buffers

**Gap Assessment**:
- **Immediate Needs** (High Priority): Q@K^T attention matrix, residual add, scaled division
- **Critical Dependencies**: Element-wise operations, reduction operations, memory management
- **Expected Development**: 2-4 weeks to implement full single-layer encoder pipeline

---

## 1. CURRENT KERNEL ANALYSIS

### 1.1 LayerNorm (layernorm_bf16_xdna1.cc)

#### Current Implementation
```
File: layernorm_bf16_xdna1.cc (3.1 KB)
Functions: layernorm_bf16(), layernorm_simple_bf16()
Version: Scalar implementation (scalar operations, no vectorization)
```

#### Current Capabilities
| Aspect | Details |
|--------|---------|
| **Input Size** | 1024 elements (hardcoded) |
| **Data Type** | BF16 (2 bytes each = 2 KB buffer) |
| **Parameters** | Yes: gamma, beta (learnable) |
| **Algorithm** | 3-pass: mean, variance, normalization |
| **Approximation** | Fast inverse square root (Quake III) |
| **Vectorization** | None (scalar only) |

#### Algorithm Breakdown
```c
Pass 1: sum = Σ(x[i])                        → mean = sum / N
Pass 2: var_sum = Σ((x[i] - mean)²)          → var = var_sum / N
Pass 3: inv_std = rsqrt(var + eps)            → y[i] = (x[i] - mean) * inv_std * gamma + beta
```

#### Performance Characteristics
- **Flops**: 1024 + 1024 + (2×1024) = 4,096 scalar operations
- **Memory**: 2 KB read (input) + 2 KB write (output) + 2 KB gamma + 2 KB beta = 8 KB
- **Estimated Time**: ~1 ms (scalar operations on AIE2 core)

#### Required Modifications for Full Encoder

**Problem 1: Hardcoded Buffer Size**
- Current: Fixed 1024 elements
- Needed: Support variable sizes or at least these dimensions:
  - 512 elements (common hidden dim)
  - 1024 elements (Whisper base/medium)
  - 1536 elements (Whisper large)
  - 2048 elements (expansion layers, FFN)

**Problem 2: Slow Scalar Implementation**
- Current: ~4,096 scalar ops per 1024 elements
- Better: Vectorize with 16-element BF16 vectors (matches `gelu_optimized_xdna1.cc`)
- Potential: 64× speedup with proper vectorization

**Solution Required**:
```c
// Parameterizable LayerNorm
void layernorm_bf16_param(bfloat16 *input, bfloat16 *output,
                          bfloat16 *gamma, bfloat16 *beta,
                          int32_t size);  // <-- Add size parameter

// Vectorized version (16-element vectors)
void layernorm_bf16_vec(bfloat16 *input, bfloat16 *output,
                        bfloat16 *gamma, bfloat16 *beta,
                        int32_t size);
```

#### Buffer Size Requirements
```
For sequence of 1500 tokens with hidden_dim=512:
  Input:  1500×512 = 768,000 elements → 1.5 MB
  Output: 1500×512 = 768,000 elements → 1.5 MB
  Gamma:  512 elements → 1 KB
  Beta:   512 elements → 1 KB
  ────────────────────────────────────
  Total:  3+ MB (exceeds local AIE memory, needs tiling)
```

---

### 1.2 Softmax (softmax_bf16_xdna1.cc, softmax_bf16_xdna1_batched.cc)

#### Current Implementation
```
File 1: softmax_bf16_xdna1.cc (2.5 KB)
  - softmax_simple_bf16(): Single input
  - softmax_bf16(): Wrapper for 1024 elements
  
File 2: softmax_bf16_xdna1_batched.cc (2.5 KB)
  - softmax_batched_bf16_4(): 4 inputs simultaneously
```

#### Current Capabilities
| Aspect | Details |
|--------|---------|
| **Input Size** | 1024 elements (fixed) or 4×1024 (batched) |
| **Data Type** | BF16 |
| **Algorithm** | Exp approximation + normalize |
| **Batching** | 4 frames per kernel call (reduces overhead) |
| **Vectorization** | None (scalar) |
| **Precision** | FP32 accumulation for stability |

#### Algorithm Breakdown
```
Pass 1: max_val = max(input[i])              (numerical stability)
Pass 2: exp[i] = exp(input[i] - max_val)    (approximation via 2^x)
        sum = Σ(exp[i])
Pass 3: output[i] = exp[i] / sum             (normalize)
```

#### Exp Approximation (Key Innovation)
```c
// Input: x (float)
// Output: exp(x) approximated as 2^(log2(e) * x)

ix = (int32_t)(x * 1.442695f);              // log2(e) = 1.442695
fx = x * 1.442695f - ix;                    // Fractional part

pow2_ix = (ix + 127) << 23;                 // Convert to float bits
pow2_fx = 1.0f + 0.693147f*fx + 0.240160f*fx²;  // Polynomial for 2^fx

result = pow2_ix * pow2_fx;
```

**Error**: ~1-2% relative error, acceptable for attention

#### Performance Characteristics
- **Single**: 2×1024 + 1024 = ~3,072 scalar operations
- **Batched-4**: Amortizes overhead → ~2.1ms for 4 inputs (vs 1.565ms×4 = 6.26ms)
- **Speedup**: ~3× with batching (overhead reduction)

#### Required Modifications for Full Encoder

**Problem 1: Hardcoded for 1024 Elements**
- Attention score matrix: T×T where T=1500 (sequence length)
- Per-token softmax needs: 1500 elements
- Current: Max 1024 (loss of accuracy from truncation)

**Problem 2: No Scaling Support**
- Attention softmax uses: softmax(Q@K^T / sqrt(d_k))
- Need to divide by sqrt(d_k) before softmax
- Currently no scale parameter

**Problem 3: Single Output Only**
- Works for 1D softmax
- Attention needs: batch × heads × sequence × sequence
- Requires 3D tensor support or reshaping

**Solution Required**:
```c
// Softmax with scaling (for attention)
void softmax_bf16_scaled(bfloat16 *input, bfloat16 *output,
                         int32_t size,
                         float scale);  // scale = 1.0f / sqrt(d_k)

// 2D softmax (for attention score matrix)
void softmax_2d_bf16(bfloat16 *input, bfloat16 *output,
                     int32_t rows, int32_t cols,
                     float scale);  // Apply softmax to each row

// Batched 3D softmax (batch × seq × seq)
void softmax_batched_3d_bf16(bfloat16 *input, bfloat16 *output,
                             int32_t batch_size,
                             int32_t seq_len,
                             float scale);
```

#### Buffer Size Requirements
```
Attention score matrix (1500×1500 for one head):
  Input:  1500×1500 = 2.25M elements → 4.5 MB
  Output: 1500×1500 = 2.25M elements → 4.5 MB
  ────────────────────────────────────
  Total:  9 MB (way exceeds AIE local memory: ~16-32 KB)
  
Solution: Tile-based softmax (process 64×64 or 64×128 tiles)
```

---

### 1.3 GELU (gelu_simple_xdna1.cc, gelu_optimized_xdna1.cc)

#### Current Implementation
```
File 1: gelu_simple_xdna1.cc (1.7 KB)
  - Scalar fast_tanh() approximation
  - No vectorization, simple loops
  
File 2: gelu_optimized_xdna1.cc (2.9 KB)
  - Vectorized: 16-element BF16 vectors
  - Loop unrolling with pipeline pragmas
  - LUT-based tanh lookup (getTanhBf16)
```

#### Current Capabilities
| Aspect | Details |
|--------|---------|
| **Input Size** | 1024 elements |
| **Data Type** | BF16 |
| **Algorithm** | GELU(x) = 0.5×x×(1+tanh(√(2/π)×(x+0.044715×x³))) |
| **Vectorization** | 16 elements per iteration (optimized version) |
| **Performance** | ~16× faster than simple version |

#### Algorithm Breakdown
```
Step 1: x² = x * x
Step 2: x³ = x * x²
Step 3: inner = √(2/π) * (x + 0.044715 * x³)
Step 4: tanh_out = tanh(inner)                    [LUT or approximation]
Step 5: result = 0.5 * x * (1 + tanh_out)
```

#### Vectorization Analysis (Optimized Version)
```c
// Process 16 elements at a time:
for (int i = 0; i < vector_size; i += 16) {
    auto x = *it_in++;                           // Load 16 BF16s
    auto x2 = aie::mul(x, x);                    // 16 multiplies
    auto x3 = aie::mul(x, x2);                   // 16 multiplies
    auto inner = aie::add(x, aie::mul(x3, vBeta)); // 16 ops
    auto tanh_out = getTanhBf16(inner);          // Vectorized tanh
    auto result = aie::mul(x, aie::mul(v05, aie::add(tanh_out, v1))); // 32 ops
    *it_out++ = result;
}
```

#### Performance Characteristics
- **Simple**: 1024 × (7 scalar ops) = ~7,168 scalar operations
- **Optimized**: 64 iterations × (8 vector ops on width-16) = ~256 vector ops
- **Speedup**: ~16-30× with vectorization and LUT

#### Required Modifications for Full Encoder

**Problem 1: Hardcoded 1024 Size**
- FFN hidden layer can be 2048-4096 elements
- Current: Truncates at 1024

**Problem 2: No Parameterization**
- Fixed constants only
- Some architectures use α or β scaling

**Problem 3: No Alternative Activations**
- Encoder uses GELU, but support for other activations needed
- SwiGLU (LLaMA variant) available separately

**Solution Required**:
```c
// Parameterizable GELU
void gelu_bf16(bfloat16 *input, bfloat16 *output, int32_t size);

// For FFN expansion layer
void gelu_bf16_expanded(bfloat16 *input, bfloat16 *output,
                        int32_t size);  // Support larger sizes

// Alternative: SwiGLU (already exists in swiglu_xdna1.cc)
void swiglu_bf16(bfloat16 *input, bfloat16 *w1, bfloat16 *w2,
                 bfloat16 *output, int32_t size);
```

#### Buffer Size Requirements
```
FFN layer: hidden_dim×4 = 512×4 = 2048 elements
  Input:  2048 elements → 4 KB
  Output: 2048 elements → 4 KB
  ──────────────────────────────
  Total:  8 KB (fits in AIE local memory)
  
BUT: If processing full batch:
  Batch×seq_len×2048 = 1×1500×2048 = 3.07M elements → 6.1 MB (needs tiling)
```

---

### 1.4 MatMul (matmul_bf16_xdna1.cc, matmul_bf16_vectorized_xdna1.cc)

#### Current Implementation
```
File 1: matmul_bf16_xdna1.cc (4.3 KB)
  - Scalar implementation
  - Tiled approach (64×64 tiles)
  - FP32 accumulation for stability
  
File 2: matmul_bf16_vectorized_xdna1.cc (7.6 KB)
  - 3 versions: basic vectorized, unrolled, tiled
  - Vector width: 16 BF16 elements
  - Accum type: accfloat (FP32)
```

#### Current Capabilities
| Aspect | Details |
|--------|---------|
| **Tile Size** | 64×64 matrix (8,192 bytes) |
| **VecWidth** | 16 BF16 elements |
| **Unroll Factor** | 4 rows processed simultaneously |
| **Accumulation** | FP32 (accfloat) for precision |
| **Algorithm** | Broadcast A scalar, multiply with B vector row |

#### Algorithm Breakdown (Vectorized)
```c
// For each output row i, columns j to j+15:
for (int k = 0; k < K; k++) {
    a_val = A[i, k];                              // 1 scalar
    a_vec = broadcast(a_val, 16);                 // 16-wide vector
    b_vec = load_v(&B[k, j]);                     // Load 16 BF16s
    acc += a_vec * b_vec;                         // 16 MACs
}
output[i, j:j+15] = acc.to_vector<bfloat16>();   // Convert & store
```

#### Unrolled Version (4 rows at once)
```c
// Process 4 rows in parallel:
// acc0, acc1, acc2, acc3 (4 independent accumulators)
for (int k = 0; k < K; k++) {
    a_vec0 = broadcast(A[i+0, k], 16);
    a_vec1 = broadcast(A[i+1, k], 16);
    a_vec2 = broadcast(A[i+2, k], 16);
    a_vec3 = broadcast(A[i+3, k], 16);
    
    b_vec = load_v(&B[k, j]);                     // Load once, share
    
    acc0 += a_vec0 * b_vec;  // 16 MACs
    acc1 += a_vec1 * b_vec;  // 16 MACs
    acc2 += a_vec2 * b_vec;  // 16 MACs
    acc3 += a_vec3 * b_vec;  // 16 MACs
}
```

**Result**: Better instruction-level parallelism, ~2× speedup

#### Performance Characteristics (64×64)
- **Naive**: 64³ = 262,144 scalar multiplications
- **Vectorized**: 262,144 / 16 = 16,384 vector operations
- **Unrolled**: 16,384 ops but with better parallelism
- **With Tiling**: Cache-friendly, better memory access patterns

#### Current Limitations

**Problem 1: Fixed Tile Size (64×64)**
- Attention Q@K^T: (1500 / 64) = ~24 tile iterations per dimension
- FFN: (2048 / 64) = ~32 tiles
- 24×24×64 = 36,864 small matmuls needed for full sequence

**Problem 2: Limited to Square Matrices**
- Encoder needs: (seq_len × hidden_dim) @ (hidden_dim × hidden_dim)
- Projection: (seq_len × hidden_dim) @ (hidden_dim × hidden_dim)
- Typical: 1500×512 @ 512×512 (non-square)

**Problem 3: No Support for Batched Operations**
- Multi-head attention: batch × heads × seq × seq
- Would need reshape or batch dimension support

**Problem 4: FP32 Accumulation Overflow**
- Large K (1024+) can cause accumulator overflow
- Need scaling or intermediate reduction

#### Required Modifications for Full Encoder

**Solution 1: Parameterizable Dimensions**
```c
void matmul_bf16(bfloat16 *A, bfloat16 *B, bfloat16 *C,
                 int32_t M, int32_t N, int32_t K);
```
✅ Already exists, uses adaptive tiling

**Solution 2: Larger Tile Support**
```c
// For larger operations: 128×128 or 256×256 with better memory layout
void matmul_bf16_large_tile(bfloat16 *A, bfloat16 *B, bfloat16 *C,
                            int32_t M, int32_t N, int32_t K,
                            int32_t tile_m, int32_t tile_n, int32_t tile_k);
```

**Solution 3: Batched MatMul**
```c
// Process multiple matrices in batch
void matmul_bf16_batched(bfloat16 *A, bfloat16 *B, bfloat16 *C,
                         int32_t batch_size,
                         int32_t M, int32_t N, int32_t K);
```

#### Buffer Size Requirements
```
Q@K^T: (1500×512) @ (512×1500) → 1500×1500
  A: 1500×512 = 768,000 elements → 1.5 MB
  B: 512×1500 = 768,000 elements → 1.5 MB
  C: 1500×1500 = 2.25M elements → 4.5 MB
  ────────────────────────────────────
  Total: 7.5 MB (EXCEEDS AIE LOCAL MEMORY)
  
With 64×64 tiling: Process in chunks, reuse buffers
  Tile A: 64×512 → 256 KB
  Tile B: 512×64 → 256 KB
  Tile C: 64×64 → 8 KB
  ──────────────────────────────
  Per-tile: ~520 KB (manageable)
  Iterations: (1500/64) × (1500/64) = 546 tiles
```

---

## 2. WHISPER ENCODER ARCHITECTURE

### 2.1 Whisper Dimensions Reference

```
Whisper Encoder Architecture (Base Model):
├── Input: Audio spectrogram (80 mel bins × ~1500 frames)
├── Conv1D × 2: 80 → 512 hidden dim (kernel_size=3, stride=1)
├── Positional Encoding: (seq_len, 512) learned embeddings
│
└── Transformer Encoder: 12 layers of:
    ├── LayerNorm: (seq, 512) → (seq, 512)
    ├── MultiHeadAttention: 
    │   ├── Q, K, V projections: (seq, 512) @ (512, 512) → (seq, 512)
    │   ├── Q@K^T: (seq, 512) @ (512, seq) → (seq, seq) [BOTTLENECK]
    │   ├── Softmax: (seq, seq) [1500×1500 per head, 8 heads]
    │   ├── Attention@V: (seq, seq) @ (seq, 512) → (seq, 512)
    │   └── Output projection: (seq, 512) @ (512, 512) → (seq, 512)
    │
    ├── Residual Add: attention_output + input → (seq, 512)
    ├── LayerNorm: (seq, 512) → (seq, 512)
    ├── FFN (feed-forward network):
    │   ├── Linear: (seq, 512) @ (512, 2048) → (seq, 2048)
    │   ├── GELU: (seq, 2048) → (seq, 2048)
    │   └── Linear: (seq, 2048) @ (2048, 512) → (seq, 512)
    │
    └── Residual Add: ffn_output + attention_output+input → (seq, 512)

Typical Dimensions:
  Hidden dim (d_model): 512
  Attention heads: 8
  Head dim (d_k): 512 / 8 = 64
  FFN inner dim: 4 × 512 = 2048
  Max sequence length: 1500 (for 30-second audio @ 50ms frames)
  Batch size: 1 (or small, for inference)
```

### 2.2 Critical Operations in Full Layer

| Operation | Input Shape | Output Shape | Memory | Priority |
|-----------|------------|-------------|--------|----------|
| LayerNorm | (seq, 512) | (seq, 512) | 512B in | High |
| Q proj | (seq, 512)@(512, 512) | (seq, 512) | 1.5MB | High |
| K proj | (seq, 512)@(512, 512) | (seq, 512) | 1.5MB | High |
| V proj | (seq, 512)@(512, 512) | (seq, 512) | 1.5MB | High |
| **Q@K^T** | (seq, 512)@(512, seq) | (seq, seq) | **4.5MB** | **CRITICAL** |
| Softmax | (seq, seq) | (seq, seq) | 4.5MB | High |
| **Attn@V** | (seq, seq)@(seq, 512) | (seq, 512) | 4.5MB | **CRITICAL** |
| Output proj | (seq, 512)@(512, 512) | (seq, 512) | 1.5MB | High |
| **Residual add** | (seq, 512) + (seq, 512) | (seq, 512) | 1.5KB | **CRITICAL** |
| Linear 1 | (seq, 512)@(512, 2048) | (seq, 2048) | 3MB | High |
| GELU | (seq, 2048) | (seq, 2048) | 4KB in | High |
| Linear 2 | (seq, 2048)@(2048, 512) | (seq, 512) | 3MB | High |

---

## 3. MISSING KERNEL FUNCTIONS

### 3.1 CRITICAL - Must Implement First

#### 3.1.1 Q@K^T Attention Matrix (Scaled Dot-Product)

**Purpose**: Compute attention scores before softmax  
**Computation**: `scores[i,j] = Q[i,:] · K[j,:] / sqrt(d_k)`

**Dimensions**:
- Q: (batch, seq_len, heads, d_k) = (1, 1500, 8, 64)
- K: (batch, seq_len, heads, d_k) = (1, 1500, 8, 64)
- Output: (batch, heads, seq_len, seq_len) = (1, 8, 1500, 1500)

**Per-head operation**:
- Input Q: 1500×64
- Input K: 1500×64 (but use K^T: 64×1500)
- Output: 1500×1500
- Scale factor: 1 / sqrt(64) = 0.125

**Algorithm**:
```
Step 1: Transpose K → (64, 1500)
Step 2: MatMul Q@K^T → (1500, 1500)
Step 3: Scale by 1/sqrt(d_k)
Step 4: Repeat for all 8 heads
```

**Implementation Estimate**:
```
void attention_qkt_bf16(bfloat16 *Q,           // (seq, d_k)
                        bfloat16 *K,           // (seq, d_k)
                        bfloat16 *scores,      // (seq, seq)
                        int32_t seq_len,       // 1500
                        int32_t d_k,           // 64
                        int32_t d_model,       // 512
                        int32_t num_heads) {   // 8
  
  // For each head:
  for (int h = 0; h < num_heads; h++) {
    bfloat16 *Q_h = Q + h * d_k;              // Head h of Q
    bfloat16 *K_h = K + h * d_k;              // Head h of K
    bfloat16 *scores_h = scores + h * seq_len * seq_len;
    
    // MatMul: Q_h@K_h^T → (seq, seq)
    matmul_bf16(Q_h, K_h_transposed, scores_h, seq_len, seq_len, d_k);
    
    // Scale by sqrt(d_k) inverse
    float scale = 1.0f / sqrt((float)d_k);
    for (int i = 0; i < seq_len * seq_len; i++) {
      scores_h[i] = (bfloat16)((float)scores_h[i] * scale);
    }
  }
}
```

**Memory Requirements**:
```
Single head:
  Q: 1500×64 = 96K elements → 192 KB
  K: 1500×64 = 96K elements → 192 KB
  Output: 1500×1500 = 2.25M elements → 4.5 MB
  ─────────────────────────────────
  Total: ~5 MB per head (EXCEEDS AIE LOCAL MEMORY)
  
Solution: Tile-based computation
  For 64×64 tile: 64×64 @ 64×64 → 64×64 output (8 KB)
  Iterations: (1500/64)² = 546 outer iterations
  Per-tile memory: ~512 KB (manageable)
```

**Critical Issue**: This is the main bottleneck
- MatMul: 1500×1500×64 = 144M MACs per head
- For 8 heads: 1.15B MACs total
- At 16 TOPS (NPU spec): 71ms per head, 568ms total
- **Unacceptable for realtime transcription**

**Optimization Strategy**:
1. Use vectorized matmul (256 ops/cycle)
2. Tile-based computation (reuse memory)
3. Parallel heads (if NPU tile array allows)
4. Integer quantization (INT8 = 4× speedup potential)

---

#### 3.1.2 Residual Add (Element-wise Addition)

**Purpose**: Add residual connections: x + attention_output  
**Computation**: `output[i] = input[i] + residual[i]`

**Dimensions**:
- Input A: (seq_len, hidden_dim) = (1500, 512)
- Input B: (seq_len, hidden_dim) = (1500, 512)
- Output: (seq_len, hidden_dim) = (1500, 512)

**Algorithm**:
```c
void residual_add_bf16(bfloat16 *input,      // (seq, hidden)
                       bfloat16 *residual,    // (seq, hidden)
                       bfloat16 *output,      // (seq, hidden)
                       int32_t size) {        // seq × hidden
  for (int i = 0; i < size; i++) {
    output[i] = (bfloat16)((float)input[i] + (float)residual[i]);
  }
}
```

**Vectorized Version**:
```c
void residual_add_bf16_vec(bfloat16 *input, bfloat16 *residual,
                           bfloat16 *output, int32_t size) {
  // Process 16 elements at a time
  for (int i = 0; i < size; i += 16) {
    auto v1 = aie::load_v<16>(input + i);
    auto v2 = aie::load_v<16>(residual + i);
    auto result = aie::add(v1, v2);
    aie::store_v(output + i, result);
  }
}
```

**Performance**:
- Simple: 1 addition per element = 768,000 additions
- Vectorized: 768,000 / 16 = 48,000 vector additions
- Time: <1ms on NPU
- **This is TRIVIAL compared to MatMul, but ESSENTIAL**

**Memory**:
```
Input: 1500×512 → 1.5 MB
Residual: 1500×512 → 1.5 MB
Output: 1500×512 → 1.5 MB
────────────────────────────
Total: 4.5 MB (would need streaming/tiling)
```

**Status**: **MUST IMPLEMENT IMMEDIATELY** (very simple, very important)

---

#### 3.1.3 Scaled Division (Scale Factor for Softmax)

**Purpose**: Apply scale factor before softmax: `scaled = input / sqrt(d_k)`  
**Computation**: `output[i] = input[i] * (1.0 / sqrt(d_k))`

**Dimensions**:
- Input: (seq, seq) = (1500, 1500) = 2.25M elements
- Output: (seq, seq) = (1500, 1500) = 2.25M elements
- Scale: Single float (1 / sqrt(64) = 0.125)

**Algorithm**:
```c
void scale_bf16(bfloat16 *input, bfloat16 *output,
                int32_t size, float scale) {
  bfloat16 scale_bf16 = (bfloat16)scale;
  for (int i = 0; i < size; i++) {
    output[i] = (bfloat16)((float)input[i] * scale);
  }
}
```

**Vectorized**:
```c
void scale_bf16_vec(bfloat16 *input, bfloat16 *output,
                    int32_t size, float scale) {
  auto scale_vec = aie::broadcast<bfloat16, 16>((bfloat16)scale);
  for (int i = 0; i < size; i += 16) {
    auto v = aie::load_v<16>(input + i);
    auto result = aie::mul(v, scale_vec);
    aie::store_v(output + i, result);
  }
}
```

**Performance**: <1ms for 2.25M elements (very fast, overlappable with softmax)

**Status**: **TRIVIAL, but can be fused with softmax**

---

### 3.2 HIGH PRIORITY - Required for Multiple Operations

#### 3.2.1 Attention Matrix × Values (Attn@V)

**Purpose**: Apply attention scores to values: `output[i] = Σ_j scores[i,j] * V[j,:]`  
**Computation**: MatMul with transposed attention scores

**Dimensions**:
- Attention scores: (seq_len, seq_len) = (1500, 1500)
- V: (seq_len, d_k) = (1500, 64)
- Output: (seq_len, d_k) = (1500, 64)

**Algorithm**:
```
MatMul: scores @ V → (1500, 1500) @ (1500, 64) → (1500, 64)
```

**Implementation**:
```c
void attention_values_bf16(bfloat16 *scores,      // (seq, seq)
                           bfloat16 *values,      // (seq, d_k)
                           bfloat16 *output,      // (seq, d_k)
                           int32_t seq_len,
                           int32_t d_k) {
  matmul_bf16(scores, values, output, seq_len, d_k, seq_len);
}
```

**Status**: Reuses existing matmul kernel - **NO NEW CODE NEEDED**

---

#### 3.2.2 Multi-Head Attention Orchestration

**Purpose**: Manage 8 parallel attention heads  
**Process**:
1. Split input into 8 heads
2. Compute Q, K, V for each head
3. Compute attention for each head in parallel (or sequential)
4. Concatenate outputs

**Pseudo-code**:
```c
void multi_head_attention_bf16(bfloat16 *input,      // (seq, hidden=512)
                               bfloat16 *W_q, *W_k, *W_v,  // (hidden, hidden)
                               bfloat16 *W_out,       // (hidden, hidden)
                               bfloat16 *output,      // (seq, hidden=512)
                               int32_t seq_len,
                               int32_t hidden_dim,    // 512
                               int32_t num_heads) {   // 8
  
  int32_t d_k = hidden_dim / num_heads;  // 64
  
  // Project input to Q, K, V
  bfloat16 Q[seq_len * hidden_dim];
  bfloat16 K[seq_len * hidden_dim];
  bfloat16 V[seq_len * hidden_dim];
  
  matmul_bf16(input, W_q, Q, seq_len, hidden_dim, hidden_dim);
  matmul_bf16(input, W_k, K, seq_len, hidden_dim, hidden_dim);
  matmul_bf16(input, W_v, V, seq_len, hidden_dim, hidden_dim);
  
  // Compute attention for each head
  bfloat16 head_outputs[num_heads * seq_len * d_k];
  for (int h = 0; h < num_heads; h++) {
    // Extract head h
    bfloat16 *Q_h = Q + h * d_k;
    bfloat16 *K_h = K + h * d_k;
    bfloat16 *V_h = V + h * d_k;
    bfloat16 *out_h = head_outputs + h * seq_len * d_k;
    
    // Compute attention
    attention_qkt_bf16(Q_h, K_h, scores_h, seq_len, d_k, hidden_dim, 1);
    softmax_bf16_scaled(scores_h, scores_h, seq_len * seq_len, 1.0f / sqrt(d_k));
    attention_values_bf16(scores_h, V_h, out_h, seq_len, d_k);
  }
  
  // Concatenate heads
  concat_heads_bf16(head_outputs, concat, seq_len, hidden_dim, num_heads);
  
  // Output projection
  matmul_bf16(concat, W_out, output, seq_len, hidden_dim, hidden_dim);
}
```

**Status**: Needs **Q@K^T kernel**, rest uses existing kernels

---

#### 3.2.3 Attention Softmax with Scaling

**Purpose**: Apply softmax to attention scores with scale factor  
**Computation**: softmax(scale * input)

**Implementation**:
```c
void softmax_bf16_scaled(bfloat16 *input, bfloat16 *output,
                         int32_t size, float scale) {
  // Combine scaling + softmax
  // 1. Scale input
  // 2. Find max (numerically stable)
  // 3. Exp and sum
  // 4. Normalize
  
  float max_val = -FLT_MAX;
  for (int i = 0; i < size; i++) {
    float val = (float)input[i] * scale;
    if (val > max_val) max_val = val;
  }
  
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    float x = (float)input[i] * scale - max_val;
    float exp_val = exp(x);  // Use fast_exp approximation
    output[i] = (bfloat16)exp_val;
    sum += exp_val;
  }
  
  float inv_sum = 1.0f / sum;
  for (int i = 0; i < size; i++) {
    output[i] = (bfloat16)((float)output[i] * inv_sum);
  }
}
```

**Status**: **CAN BE FUSED** with existing softmax_bf16 kernel (modify to add scale parameter)

---

### 3.3 MEDIUM PRIORITY - Performance Optimizations

#### 3.3.1 BatchNorm (if used in encoder)

**Status**: Whisper uses LayerNorm only - **NOT NEEDED**

#### 3.3.2 Dropout (for training)

**Status**: Inference only - **NOT NEEDED**

#### 3.3.3 Concatenate Heads

**Purpose**: Merge attention heads back to hidden dim  
**Computation**: Reshape/reorganize tensors

**Simple implementation**:
```c
void concat_heads_bf16(bfloat16 *heads,      // (seq, heads, d_k)
                       bfloat16 *output,     // (seq, hidden)
                       int32_t seq_len,
                       int32_t hidden_dim,
                       int32_t num_heads) {
  int32_t d_k = hidden_dim / num_heads;
  for (int s = 0; s < seq_len; s++) {
    for (int h = 0; h < num_heads; h++) {
      for (int d = 0; d < d_k; d++) {
        int src_idx = h * seq_len * d_k + s * d_k + d;
        int dst_idx = s * hidden_dim + h * d_k + d;
        output[dst_idx] = heads[src_idx];
      }
    }
  }
}
```

**Status**: Trivial, can be CPU operation - **LOW PRIORITY**

---

## 4. BUFFER SIZE REQUIREMENTS SUMMARY

### 4.1 Per-Kernel Memory Profile

| Kernel | Current Max | Full Encoder | Required |
|--------|------------|-------------|----------|
| LayerNorm | 1024 (2 KB) | 1500×512 (1.5 MB) | Support parameterizable sizes |
| Softmax | 1024 (2 KB) | 1500×1500 (4.5 MB) | Tile-based 64×64 blocks |
| GELU | 1024 (2 KB) | 2048 (4 KB) | ✓ Expandable to 2048 |
| MatMul-64x64 | 64×64 (8 KB) | 1500×1500 (4.5 MB) | ✓ Tiling support exists |
| **NEW: Attn-QKT** | N/A | 1500×1500 (4.5 MB) | MatMul wrapper |
| **NEW: ResidualAdd** | N/A | 1500×512 (1.5 MB) | Element-wise add |

### 4.2 Total System Memory for One Layer

```
Processing one full encoder layer (1500 seq, 512 hidden, 8 heads):

Activations in flight:
  Input:                1500×512 = 1.5 MB
  After LayerNorm:      1500×512 = 1.5 MB
  Q, K, V:              3×1.5 MB = 4.5 MB
  Attention scores:     1500×1500 = 4.5 MB per head × 8 = 36 MB (!)
  Head outputs:         1500×512 = 1.5 MB
  After attention:      1500×512 = 1.5 MB
  FFN hidden:           1500×2048 = 6 MB
  ─────────────────────────────────────────
  Peak: ~50 MB (for full processing without reuse)
  
With buffer reuse (streaming):
  Tile-based: Process 64×seq at a time
  Active buffers: ~512 KB → 5 MB
  ─────────────────────────────────────────
  With careful management: ~10 MB (acceptable)
```

### 4.3 AIE Local Memory Constraints

```
Phoenix NPU Tile Memory:
  Per compute tile: ~32 KB instruction + ~32 KB data = 64 KB
  Shared memory tiles: 256 KB each
  Total system memory: ~1-2 MB accessible
  
Implication: Must use DMA + tiling
  - Load 64×64 tiles from main memory
  - Process in AIE cores
  - Write results back
  - Overlap compute with memory transfer
```

---

## 5. DETAILED IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Week 1)

**Goal**: Get one full encoder layer working end-to-end

**Tasks**:
1. **Residual Add Kernel** ✓ (trivial, ~30 min)
   - File: `residual_add_bf16_xdna1.cc`
   - Vectorized version with 16-element operations
   - Support parameterizable size

2. **Modified Softmax** ✓ (30 min)
   - Update `softmax_bf16_xdna1.cc`
   - Add `scale` parameter
   - Support larger sizes (up to 1500)

3. **Parameterizable LayerNorm** ✓ (1 hour)
   - Update `layernorm_bf16_xdna1.cc`
   - Make size parameter flexible
   - Keep fast_rsqrt optimization

4. **Expandable GELU** ✓ (30 min)
   - Update `gelu_optimized_xdna1.cc`
   - Support up to 4096 elements
   - Use batch size parameter

5. **Test Single Layer** (2-3 hours)
   - Write test harness
   - Feed sample data through encoder
   - Validate outputs vs PyTorch reference

**Estimated Time**: 8-10 hours

---

### Phase 2: Critical Attention (Week 2)

**Goal**: Implement full multi-head attention mechanism

**Tasks**:
1. **Q@K^T Attention Kernel** (3-4 hours)
   - Wrapper around existing matmul
   - Handle head splitting/merging
   - Scale factor application
   - Tiling for large sequences

2. **Multi-Head Orchestration** (2-3 hours)
   - Coordinate 8 heads
   - Parallel or sequential execution
   - Concatenation logic

3. **Attention Softmax** (1 hour)
   - Fuse scaled softmax
   - Optimize for attention dimensions

4. **Full Layer Test** (2-3 hours)
   - Test multi-head attention
   - Compare with PyTorch
   - Measure latency per operation

**Estimated Time**: 10-12 hours

---

### Phase 3: Performance Optimization (Week 3)

**Goal**: Optimize for 220x realtime target

**Tasks**:
1. **Kernel Fusion** (2-3 hours)
   - Fuse LayerNorm + next operation
   - Fuse attention score + softmax
   - Eliminate intermediate writes

2. **Memory Layout Optimization** (2-3 hours)
   - Align data for DMA transfers
   - Optimize tile sizes
   - Improve cache utilization

3. **Vectorization Improvements** (2-3 hours)
   - Use 32-element vectors where possible
   - Reduce scalar operations
   - Better instruction scheduling

4. **Benchmarking Suite** (2 hours)
   - Measure each kernel
   - Identify bottlenecks
   - Profile memory bandwidth

**Estimated Time**: 10-12 hours

---

### Phase 4: Full Integration (Week 4)

**Goal**: Complete 12-layer encoder on NPU

**Tasks**:
1. **Layer Stacking** (2-3 hours)
   - Chain 12 encoder layers
   - Handle inter-layer data flow
   - Optimize batch processing

2. **System Integration** (2-3 hours)
   - Connect to mel-spectrogram input
   - Integrate with ONNX decoder
   - End-to-end pipeline

3. **Final Testing & Optimization** (3-4 hours)
   - Full transcription test
   - Latency analysis
   - Power consumption measurement

**Estimated Time**: 10-12 hours

**Total**: 40-50 hours (1 week full-time, 2-3 weeks part-time)

---

## 6. CODE TEMPLATES

### 6.1 Residual Add Template

```c
//===- residual_add_bf16_xdna1.cc --------------------------------*- C++ -*-===//
//
// XDNA1 (Phoenix NPU) - Residual Add Kernel
// Element-wise addition for skip connections
//
// result[i] = input[i] + residual[i]
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

// Vectorized residual add: 16 elements at a time
void residual_add_bf16(bfloat16 *restrict input,
                       bfloat16 *restrict residual,
                       bfloat16 *restrict output,
                       int32_t size) {
  event0();
  
  // Process 16 elements per iteration
  for (int32_t i = 0; i < size; i += 16) {
    // Load two vectors
    aie::vector<bfloat16, 16> v_input = aie::load_v<16>(&input[i]);
    aie::vector<bfloat16, 16> v_residual = aie::load_v<16>(&residual[i]);
    
    // Add element-wise
    aie::vector<bfloat16, 16> result = aie::add(v_input, v_residual);
    
    // Store result
    aie::store_v(&output[i], result);
  }
  
  event1();
}

// Handle remaining elements if size not multiple of 16
void residual_add_bf16_general(bfloat16 *restrict input,
                               bfloat16 *restrict residual,
                               bfloat16 *restrict output,
                               int32_t size) {
  event0();
  
  int32_t chunks = size / 16;
  
  // Vectorized part
  for (int32_t i = 0; i < chunks; i++) {
    aie::vector<bfloat16, 16> v1 = aie::load_v<16>(&input[i * 16]);
    aie::vector<bfloat16, 16> v2 = aie::load_v<16>(&residual[i * 16]);
    aie::vector<bfloat16, 16> result = aie::add(v1, v2);
    aie::store_v(&output[i * 16], result);
  }
  
  // Scalar tail for remaining elements
  for (int32_t i = chunks * 16; i < size; i++) {
    float a = (float)input[i];
    float b = (float)residual[i];
    output[i] = (bfloat16)(a + b);
  }
  
  event1();
}

} // extern "C"
```

---

### 6.2 Scaled Softmax Template

```c
//===- softmax_scaled_bf16_xdna1.cc ----------------------------*- C++ -*-===//
//
// XDNA1 (Phoenix NPU) - Scaled BF16 Softmax Kernel
// Softmax with scale factor: softmax(scale * input)
//
// Used for attention: softmax(Q@K^T / sqrt(d_k))
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void softmax_scaled_bf16(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector,
                         const int32_t vector_size,
                         const float scale) {
  event0();
  
  // Find maximum for numerical stability
  float max_val = (float)input_vector[0] * scale;
  for (int32_t i = 1; i < vector_size; i++) {
    float val = (float)input_vector[i] * scale;
    if (val > max_val) {
      max_val = val;
    }
  }

  // First pass: compute exp(x) and sum
  float sum = 0.0f;
  for (int32_t i = 0; i < vector_size; i++) {
    float x = (float)input_vector[i] * scale - max_val;

    // Exp approximation
    int32_t ix = (int32_t)(x * 1.442695040888963f);
    float fx = x * 1.442695040888963f - ix;

    ix = (ix + 127) << 23;
    float pow2_ix;
    memcpy(&pow2_ix, &ix, sizeof(float));

    float pow2_fx = 1.0f + 0.6931471805599453f * fx + 0.2401598148889220f * fx * fx;
    float result = pow2_ix * pow2_fx;
    output_vector[i] = (bfloat16)result;
    sum += result;
  }

  // Second pass: normalize
  const float eps = 1e-7f;
  sum = sum + eps;
  float inv_sum = 1.0f / sum;

  for (int32_t i = 0; i < vector_size; i++) {
    float val = (float)output_vector[i] * inv_sum;
    output_vector[i] = (bfloat16)val;
  }
  
  event1();
}

} // extern "C"
```

---

### 6.3 Q@K^T Attention Kernel Template

```c
//===- attention_qkt_bf16_xdna1.cc ------------------------------*- C++ -*-===//
//
// XDNA1 (Phoenix NPU) - Q@K^T Attention Matrix Kernel
// Compute scaled dot-product attention scores
//
// scores[i,j] = (Q[i,:] · K[j,:]) / sqrt(d_k)
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <cmath>

extern "C" {

// Compute Q@K^T for one attention head
// Q: (seq_len, d_k)
// K_T: (d_k, seq_len) [K transposed]
// scores: (seq_len, seq_len)
void attention_qkt_bf16(bfloat16 *restrict Q,
                        bfloat16 *restrict K_T,
                        bfloat16 *restrict scores,
                        int32_t seq_len,
                        int32_t d_k) {
  event0();
  
  float scale = 1.0f / sqrt((float)d_k);
  
  // Matrix multiply: Q @ K_T
  // Use existing matmul_bf16 or inline for efficiency
  for (int32_t i = 0; i < seq_len; i++) {
    for (int32_t j = 0; j < seq_len; j++) {
      float acc = 0.0f;
      for (int32_t k = 0; k < d_k; k++) {
        float q_val = (float)Q[i * d_k + k];
        float k_val = (float)K_T[k * seq_len + j];
        acc += q_val * k_val;
      }
      // Store with scale applied
      scores[i * seq_len + j] = (bfloat16)(acc * scale);
    }
  }
  
  event1();
}

// Tiled version for large sequences
void attention_qkt_bf16_tiled(bfloat16 *restrict Q,
                              bfloat16 *restrict K_T,
                              bfloat16 *restrict scores,
                              int32_t seq_len,
                              int32_t d_k,
                              int32_t tile_size) {
  event0();
  
  float scale = 1.0f / sqrt((float)d_k);
  const int32_t TILE = 64;  // Tile size for cache efficiency
  
  // Process in tiles
  for (int32_t i0 = 0; i0 < seq_len; i0 += TILE) {
    for (int32_t j0 = 0; j0 < seq_len; j0 += TILE) {
      int32_t i_max = (i0 + TILE < seq_len) ? i0 + TILE : seq_len;
      int32_t j_max = (j0 + TILE < seq_len) ? j0 + TILE : seq_len;
      
      for (int32_t i = i0; i < i_max; i++) {
        for (int32_t j = j0; j < j_max; j++) {
          float acc = 0.0f;
          for (int32_t k = 0; k < d_k; k++) {
            float q_val = (float)Q[i * d_k + k];
            float k_val = (float)K_T[k * seq_len + j];
            acc += q_val * k_val;
          }
          scores[i * seq_len + j] = (bfloat16)(acc * scale);
        }
      }
    }
  }
  
  event1();
}

} // extern "C"
```

---

## 7. INTEGRATION CHECKLIST

### Pre-Implementation
- [ ] Review XDNA1 hardware specs (memory, execution units)
- [ ] Understand AIE2 vector ISA (intrinsics used)
- [ ] Set up build environment for AIE2 compilation
- [ ] Create test harness for single kernel
- [ ] Establish performance baseline

### Phase 1: Basic Operations
- [ ] Implement residual_add_bf16
- [ ] Update softmax to support scaling and larger sizes
- [ ] Update layernorm to be parameterizable
- [ ] Update gelu to support larger sizes
- [ ] Test each kernel individually

### Phase 2: Attention Operations
- [ ] Implement attention_qkt_bf16
- [ ] Implement multi_head_attention orchestrator
- [ ] Test multi-head logic
- [ ] Validate against PyTorch reference

### Phase 3: Integration
- [ ] Chain kernels into full layer
- [ ] Test single layer end-to-end
- [ ] Stack 12 layers
- [ ] Test full encoder

### Phase 4: Optimization
- [ ] Profile each operation
- [ ] Identify bottlenecks
- [ ] Implement kernel fusion
- [ ] Optimize memory layout
- [ ] Measure final performance vs 220x target

### Validation
- [ ] Accuracy validation (WER, output correlation)
- [ ] Latency measurement per operation
- [ ] Power consumption profiling
- [ ] Scalability testing (variable sequence lengths)
- [ ] Production readiness

---

## 8. CRITICAL SUCCESS FACTORS

### Must-Have
1. **Q@K^T Matmul**: Bottleneck operation, needs optimization
2. **Residual Add**: Essential for skip connections
3. **Softmax Scaling**: Required for attention correctness

### Performance Targets
- Single layer: <100ms
- Full 12-layer encoder: <1200ms
- Full transcription (encoder + decoder): <5 seconds for 30-second audio

### Risk Mitigation
- Keep CPU fallback ready if NPU stalls
- Validate each kernel against PyTorch
- Use incremental integration (test at each phase)
- Monitor memory usage strictly (tile-based processing mandatory)

---

## 9. REFERENCES

**Current Kernel Files**:
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1/`
- layernorm_bf16_xdna1.cc
- softmax_bf16_xdna1.cc
- gelu_simple_xdna1.cc
- gelu_optimized_xdna1.cc
- matmul_bf16_xdna1.cc
- matmul_bf16_vectorized_xdna1.cc
- swiglu_xdna1.cc (alternative activation)

**Related Code**:
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/npu_encoder.py` - Encoder interface
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/npu_attention_wrapper.py` - Attention logic

**Hardware Documentation**:
- AMD Phoenix NPU (XDNA1): 16 TOPS INT8, 4×6 tile array
- XRT 2.20.0: AMD runtime for NPU
- MLIR-AIE: MLIR dialect for AIE2 kernels

---

**Report Generated**: November 20, 2025
**Analysis Depth**: Comprehensive
**Ready for Implementation**: Yes
**Estimated Development Time**: 40-50 hours

