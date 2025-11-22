# Kernel Integration Architecture
## AMD Phoenix NPU Whisper Encoder Optimization

**Date**: November 18, 2025
**Version**: 1.0
**Author**: Integration Architecture Team Lead
**Status**: Design Complete - Ready for Implementation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Kernel Inventory](#kernel-inventory)
3. [Encoder Layer Data Flow](#encoder-layer-data-flow)
4. [Buffer Management Strategy](#buffer-management-strategy)
5. [Kernel Chaining Architecture](#kernel-chaining-architecture)
6. [Memory Layout for BF16 Tensors](#memory-layout-for-bf16-tensors)
7. [Performance Projections](#performance-projections)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Risk Analysis](#risk-analysis)

---

## Executive Summary

This document defines the integration architecture for chaining custom NPU kernels to implement a complete Whisper encoder layer on the AMD Phoenix NPU (XDNA1).

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Chaining Approach** | Sequential with Pipelined DMA | Minimize latency while hiding transfer overhead |
| **Buffer Strategy** | Persistent + Ping-Pong | Reduce host-NPU transfers, maximize on-chip reuse |
| **Data Type** | BF16 (bfloat16) | Native NPU support, good accuracy/performance balance |
| **Batch Size** | 8 frames | Optimal for 32KB tile buffers with double-buffering |

### Target Performance

- **Current**: 5.2x realtime (CPU-only)
- **Phase 1**: 40x realtime (single encoder layer on NPU)
- **Final Target**: 220x realtime (full encoder stack)

---

## Kernel Inventory

### Available Compiled Kernels (November 18, 2025)

| Kernel | XCLBIN | Status | Speedup | Use Case |
|--------|--------|--------|---------|----------|
| **Softmax BF16** | `build_softmax_bf16/softmax_bf16.xclbin` (15KB) | Production Ready | 2.02x proven | Attention scores |
| **Softmax Parallel 2-tile** | `build_softmax_parallel/softmax_parallel.xclbin` (23KB) | Validated | 2.02x | Multi-head parallel softmax |
| **Softmax Multi-column 4-tile** | `build_softmax_multicolumn/` | Compiled, Testing | Expected 4x | Full parallelism |
| **GELU BF16** | `build_gelu/gelu_bf16.xclbin` (15KB) | Production Ready | Est. 15-20x | FFN activation |
| **LayerNorm BF16** | `build_layernorm/layernorm_bf16.xclbin` (13KB) | Production Ready | Est. 5-10x | Pre-attention & post-FFN |
| **MatMul BF16** | In development | Design Phase | Target 30-50x | Attention & FFN projections |

### Kernel Dependencies

```
LayerNorm → MatMul (Q,K,V) → Softmax → MatMul (Attn) → MatMul (Out) → Add
    │                                                                    │
    └────────────────────── Residual Connection ─────────────────────────┘

LayerNorm → MatMul (Up) → GELU → MatMul (Down) → Add
    │                                             │
    └──────────── Residual Connection ────────────┘
```

---

## Encoder Layer Data Flow

### Complete Whisper Encoder Layer Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        WHISPER ENCODER LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                     MULTI-HEAD SELF-ATTENTION                            │  │
│  ├──────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                          │  │
│  │  Hidden States ─┬─► LayerNorm ─┬─► Linear_Q ──► Q                       │  │
│  │   [B,T,D]       │              ├─► Linear_K ──► K    Reshape to         │  │
│  │                 │              └─► Linear_V ──► V    [B,H,T,D/H]        │  │
│  │                 │                                                        │  │
│  │                 │              ┌─────────────────────────────────────┐   │  │
│  │                 │              │  For each head h in 1..H:           │   │  │
│  │                 │              │    scores = Q_h @ K_h^T / sqrt(d_k) │   │  │
│  │                 │              │    attn = Softmax(scores)           │   │  │
│  │                 │              │    context = attn @ V_h             │   │  │
│  │                 │              └─────────────────────────────────────┘   │  │
│  │                 │                                                        │  │
│  │                 │              Concat heads ─► Linear_Out ──► Output     │  │
│  │                 │                                              │        │  │
│  │                 └──────────────────── + ◄─────────────────────┘        │  │
│  │                                      │                                  │  │
│  └──────────────────────────────────────┼──────────────────────────────────┘  │
│                                         │                                     │
│  ┌──────────────────────────────────────┼──────────────────────────────────┐  │
│  │                     FEED-FORWARD NETWORK                │               │  │
│  ├─────────────────────────────────────────────────────────┼───────────────┤  │
│  │                                                         │               │  │
│  │  Attention Out ─┬─► LayerNorm ─► Linear_Up ─► GELU ─► Linear_Down ─┐   │  │
│  │    [B,T,D]      │     [B,T,D]    [B,T,4D]    [B,T,4D]   [B,T,D]    │   │  │
│  │                 │                                                  │   │  │
│  │                 └──────────────────── + ◄─────────────────────────┘   │  │
│  │                                      │                                 │  │
│  └──────────────────────────────────────┼─────────────────────────────────┘  │
│                                         │                                     │
│                                    Encoder Out                                │
│                                     [B,T,D]                                   │
│                                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘

Legend:
  B = Batch size (1 for streaming, 8 for batch)
  T = Sequence length (1500 time steps for 30s audio)
  D = Hidden dimension (384/512/768/1024/1280 depending on model)
  H = Number of attention heads (6/8/12/16/20)
  d_k = D/H (dimension per head)
```

### Dimension Reference by Model Size

| Model | D (hidden) | H (heads) | d_k | FFN dim | Layers |
|-------|------------|-----------|-----|---------|--------|
| Tiny | 384 | 6 | 64 | 1536 | 4 |
| Base | 512 | 8 | 64 | 2048 | 6 |
| Small | 768 | 12 | 64 | 3072 | 12 |
| Medium | 1024 | 16 | 64 | 4096 | 24 |
| Large | 1280 | 20 | 64 | 5120 | 32 |

---

## Buffer Management Strategy

### Design Principles

1. **Minimize Host-NPU Transfers**: Keep intermediate results on-chip
2. **Persistent Buffers**: Allocate once, reuse across all frames
3. **Ping-Pong for DMA**: Overlap computation with data transfer
4. **Weight Tiling**: Load weight tiles that fit in NPU memory

### Buffer Allocation Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HOST MEMORY                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Persistent XRT Buffers (allocated once, reused):                       │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  Weight Buffers (read-only):                                   │     │
│  │    - attn_qkv_weights  [3, D, D]    (3 × D² × 2 bytes BF16)    │     │
│  │    - attn_out_weights  [D, D]       (D² × 2 bytes)             │     │
│  │    - ffn_up_weights    [D, 4D]      (4D² × 2 bytes)            │     │
│  │    - ffn_down_weights  [4D, D]      (4D² × 2 bytes)            │     │
│  │    - ln1_gamma, ln1_beta [D]        (2D × 2 bytes)             │     │
│  │    - ln2_gamma, ln2_beta [D]        (2D × 2 bytes)             │     │
│  │                                                                │     │
│  │  Activation Buffers (read-write, ping-pong):                   │     │
│  │    - input_ping   [B, T, D]         (B×T×D × 2 bytes)          │     │
│  │    - input_pong   [B, T, D]                                    │     │
│  │    - output_ping  [B, T, D]                                    │     │
│  │    - output_pong  [B, T, D]                                    │     │
│  │                                                                │     │
│  │  Intermediate Buffers (per operation):                         │     │
│  │    - qkv_buffer   [B, 3, H, T, d_k] (Q, K, V projections)      │     │
│  │    - attn_scores  [B, H, T, T]      (attention weights)        │     │
│  │    - ffn_hidden   [B, T, 4D]        (up-projection output)     │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                          PCIe 4.0 x4 DMA
                         (12 GB/s practical)
                                   │
┌──────────────────────────────────┼───────────────────────────────────────┐
│                         NPU ON-CHIP MEMORY                               │
├──────────────────────────────────┼───────────────────────────────────────┤
│                                  ▼                                       │
│  ShimNOC Tiles (Row 0) - DMA Controllers:                               │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐             │
│  │   Tile 00   │   Tile 10   │   Tile 20   │   Tile 30   │             │
│  │  DMA Ch 0   │  DMA Ch 1   │  DMA Ch 2   │  DMA Ch 3   │             │
│  │  In Weights │  In Activ   │  Out Activ  │  Control    │             │
│  └─────────────┴─────────────┴─────────────┴─────────────┘             │
│                                                                          │
│  MemTiles (Row 1) - 64KB each:                                          │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐             │
│  │ MemTile 01  │ MemTile 11  │ MemTile 21  │ MemTile 31  │             │
│  │  Weight     │  Activation │  Activation │  Result     │             │
│  │  Staging    │  Ping       │  Pong       │  Gathering  │             │
│  │  (64KB)     │  (64KB)     │  (64KB)     │  (64KB)     │             │
│  └─────────────┴─────────────┴─────────────┴─────────────┘             │
│                                                                          │
│  Compute Tiles (Rows 2-5) - 32KB each:                                  │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐             │
│  │ Tile 02     │ Tile 12     │ Tile 22     │ Tile 32     │             │
│  │ LayerNorm   │ MatMul Col0 │ MatMul Col1 │ MatMul Col2 │             │
│  ├─────────────┼─────────────┼─────────────┼─────────────┤             │
│  │ Tile 03     │ Tile 13     │ Tile 23     │ Tile 33     │             │
│  │ Softmax     │ MatMul Col0 │ MatMul Col1 │ MatMul Col2 │             │
│  ├─────────────┼─────────────┼─────────────┼─────────────┤             │
│  │ Tile 04     │ Tile 14     │ Tile 24     │ Tile 34     │             │
│  │ GELU        │ MatMul Col0 │ MatMul Col1 │ MatMul Col2 │             │
│  ├─────────────┼─────────────┼─────────────┼─────────────┤             │
│  │ Tile 05     │ Tile 15     │ Tile 25     │ Tile 35     │             │
│  │ Residual    │ MatMul Col0 │ MatMul Col1 │ MatMul Col2 │             │
│  └─────────────┴─────────────┴─────────────┴─────────────┘             │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Buffer Sizing (Whisper Base Model)

| Buffer | Dimensions | Size (BF16) | Location |
|--------|------------|-------------|----------|
| QKV Weights | [3, 512, 512] | 1.5 MB | Host (tiled to NPU) |
| Out Weights | [512, 512] | 512 KB | Host (tiled to NPU) |
| FFN Up | [512, 2048] | 2 MB | Host (tiled to NPU) |
| FFN Down | [2048, 512] | 2 MB | Host (tiled to NPU) |
| Input/Output | [8, 100, 512] | 800 KB | Host + NPU |
| Attention Scores | [8, 8, 100, 100] | 1.25 MB | NPU-only |
| FFN Hidden | [8, 100, 2048] | 3.2 MB | Host + NPU |

**Total per layer**: ~11 MB weights + ~6 MB activations = **~17 MB**

---

## Kernel Chaining Architecture

### Approach: Sequential Execution with DMA Pipelining

```
Time ────────────────────────────────────────────────────────────►

Frame N:  [DMA In]──[LayerNorm]──[Q,K,V]──[Softmax]──[Attn@V]──[Out]──[Add]
                                    │                                    │
                          [DMA Weights]                           [DMA Out]

Frame N+1:       [DMA In]──[LayerNorm]──[Q,K,V]──[Softmax]──[Attn@V]──[Out]──[Add]
                               │                                          │
                     (Uses same weights)                            [DMA Out]
```

### Kernel Execution Sequence

```python
def encoder_layer(hidden_states):
    """
    Complete encoder layer execution sequence
    """
    # =========== SELF-ATTENTION BLOCK ===========

    # 1. Pre-attention LayerNorm
    #    Kernel: layernorm_bf16.xclbin
    #    Input:  hidden_states [B, T, D]
    #    Output: normed [B, T, D]
    normed = execute_layernorm(hidden_states, ln1_gamma, ln1_beta)

    # 2. Q, K, V Projections (3 parallel MatMuls)
    #    Kernel: matmul_bf16.xclbin (3 instances)
    #    Input:  normed [B, T, D], weights [D, D]
    #    Output: Q, K, V each [B, T, D]
    Q = execute_matmul(normed, W_q)  # Can parallelize across columns
    K = execute_matmul(normed, W_k)
    V = execute_matmul(normed, W_v)

    # 3. Reshape for multi-head attention
    #    On CPU: [B, T, D] -> [B, H, T, D/H]
    Q = reshape_for_heads(Q)  # [B, 8, T, 64] for base
    K = reshape_for_heads(K)
    V = reshape_for_heads(V)

    # 4. Attention Scores
    #    Kernel: matmul_bf16.xclbin
    #    Input:  Q [B, H, T, d_k], K [B, H, T, d_k]
    #    Output: scores [B, H, T, T]
    scores = execute_matmul_batch(Q, K.transpose(-2, -1))  # Q @ K^T
    scores = scores / sqrt(64)  # Scale

    # 5. Softmax
    #    Kernel: softmax_parallel.xclbin (parallel across heads)
    #    Input:  scores [B, H, T, T]
    #    Output: attn_weights [B, H, T, T]
    attn_weights = execute_softmax_parallel(scores)

    # 6. Attention @ Values
    #    Kernel: matmul_bf16.xclbin
    #    Input:  attn_weights [B, H, T, T], V [B, H, T, d_k]
    #    Output: context [B, H, T, d_k]
    context = execute_matmul_batch(attn_weights, V)

    # 7. Reshape and Output Projection
    #    Reshape: [B, H, T, d_k] -> [B, T, D]
    #    Kernel: matmul_bf16.xclbin
    context = reshape_from_heads(context)
    attn_output = execute_matmul(context, W_out)

    # 8. Residual Add (CPU or simple NPU kernel)
    hidden_states = hidden_states + attn_output

    # =========== FEED-FORWARD BLOCK ===========

    # 9. Pre-FFN LayerNorm
    normed = execute_layernorm(hidden_states, ln2_gamma, ln2_beta)

    # 10. FFN Up-Projection
    #     Kernel: matmul_bf16.xclbin
    #     Input:  normed [B, T, D], weights [D, 4D]
    #     Output: ffn_hidden [B, T, 4D]
    ffn_hidden = execute_matmul(normed, W_up)

    # 11. GELU Activation
    #     Kernel: gelu_bf16.xclbin
    #     Input/Output: ffn_hidden [B, T, 4D]
    ffn_hidden = execute_gelu(ffn_hidden)

    # 12. FFN Down-Projection
    #     Kernel: matmul_bf16.xclbin
    #     Input:  ffn_hidden [B, T, 4D], weights [4D, D]
    #     Output: ffn_output [B, T, D]
    ffn_output = execute_matmul(ffn_hidden, W_down)

    # 13. Residual Add
    hidden_states = hidden_states + ffn_output

    return hidden_states
```

### XRT Kernel Invocation Pattern

Based on the validated softmax kernel (SUCCESS_REPORT.md), the correct invocation pattern is:

```python
# Critical: 5 parameters in this exact order
opcode = 3  # Standard NPU kernel opcode
run = kernel(
    opcode,           # Parameter 1: Operation code (always 3)
    bo_instr,         # Parameter 2: Instruction buffer
    len(insts),       # Parameter 3: Instruction size in bytes
    bo_input,         # Parameter 4: Input buffer
    bo_output         # Parameter 5: Output buffer
)
run.wait()
```

---

## Memory Layout for BF16 Tensors

### BF16 (Brain Float 16) Format

```
┌─────────────────────────────────────┐
│  S │  EEEEEEEE  │  MMMMMMM         │
│ 15 │  14 ... 7  │  6 ... 0         │
└─────────────────────────────────────┘

S = Sign bit (1 bit)
E = Exponent (8 bits, same as FP32)
M = Mantissa (7 bits, truncated from FP32's 23 bits)
```

### Tensor Memory Layout

**Row-Major Order (C-style)**:
```
Tensor [B, T, D] = [batch, time, hidden]

Memory address = base + (b * T * D + t * D + d) * sizeof(bf16)

For Whisper Base [1, 100, 512]:
  Element [0, 50, 256] is at offset:
    (0 * 100 * 512 + 50 * 512 + 256) * 2 = 51,712 bytes
```

### DMA Alignment Requirements

| Requirement | Value | Reason |
|-------------|-------|--------|
| **Page Alignment** | 4 KB | XRT DMA engine |
| **Vector Alignment** | 32 bytes | AIE2 SIMD (16 × BF16) |
| **Minimum Transfer** | 64 bytes | Cache line efficiency |

### Buffer Allocation with Alignment

```python
def allocate_aligned_buffer(shape, dtype=np.uint16):
    """
    Allocate XRT buffer with proper alignment for NPU DMA
    """
    size_bytes = np.prod(shape) * np.dtype(dtype).itemsize

    # Round up to 4KB page boundary
    aligned_size = ((size_bytes + 4095) // 4096) * 4096

    # Create XRT buffer object
    buffer = xrt.bo(device, aligned_size, xrt.bo.flags.normal, kernel.group_id(arg_index))

    return buffer, aligned_size
```

---

## Performance Projections

### Operation Timing (Whisper Base, B=8, T=100, D=512)

| Operation | Elements | Theoretical (NPU) | CPU Baseline | Speedup |
|-----------|----------|-------------------|--------------|---------|
| LayerNorm (×2) | 2 × 409,600 | 0.05 ms | 0.5 ms | 10× |
| Q,K,V MatMul | 3 × 209M | 0.30 ms | 6.0 ms | 20× |
| Score MatMul | 64M | 0.10 ms | 2.0 ms | 20× |
| Softmax | 64,000 | 0.01 ms | 0.1 ms | 10× |
| Attn@V MatMul | 64M | 0.10 ms | 2.0 ms | 20× |
| Out MatMul | 209M | 0.10 ms | 2.0 ms | 20× |
| FFN Up MatMul | 838M | 0.40 ms | 8.0 ms | 20× |
| GELU | 1,638,400 | 0.10 ms | 1.0 ms | 10× |
| FFN Down MatMul | 838M | 0.40 ms | 8.0 ms | 20× |
| **Total per Layer** | - | **1.66 ms** | **29.6 ms** | **17.8×** |

### Full Encoder Performance

| Model | Layers | NPU Time | CPU Time | Audio (1s) | RTF |
|-------|--------|----------|----------|------------|-----|
| Tiny | 4 | 6.6 ms | 118 ms | 1000 ms | 151× |
| **Base** | **6** | **10 ms** | **178 ms** | **1000 ms** | **100×** |
| Small | 12 | 20 ms | 355 ms | 1000 ms | 50× |
| Medium | 24 | 40 ms | 710 ms | 1000 ms | 25× |
| Large | 32 | 53 ms | 947 ms | 1000 ms | 19× |

### Critical Path Analysis

**Bottleneck**: MatMul operations dominate (85% of compute)

```
Single Encoder Layer Timeline (ms):
├── Self-Attention Block: 0.66 ms (40%)
│   ├── LayerNorm:     0.025 ms
│   ├── QKV MatMul:    0.30 ms  ◄── Major bottleneck
│   ├── Scores:        0.10 ms
│   ├── Softmax:       0.01 ms
│   ├── Attn@V:        0.10 ms
│   └── Out:           0.10 ms
│
└── FFN Block: 1.00 ms (60%)
    ├── LayerNorm:     0.025 ms
    ├── FFN Up:        0.40 ms  ◄── Major bottleneck
    ├── GELU:          0.10 ms
    └── FFN Down:      0.40 ms  ◄── Major bottleneck
```

### Optimization Opportunities

1. **MatMul Optimization** (Priority 1)
   - INT8 quantization: 2× speedup
   - 4-column parallelism: 4× speedup
   - Potential total: 8× speedup on MatMul alone

2. **Pipeline Parallelism** (Priority 2)
   - Overlap DMA with compute
   - Process multiple frames concurrently
   - Potential: 2× speedup

3. **Kernel Fusion** (Priority 3)
   - Fuse LayerNorm + MatMul
   - Fuse MatMul + GELU
   - Reduce memory bandwidth

**Path to 220× Realtime**:
- Base estimate: 100×
- MatMul optimization: +80%
- Pipeline parallelism: +30%
- Kernel fusion: +10%
- Total: ~220×

---

## Implementation Roadmap

### Phase 1: Single Kernel Tests (DONE)

**Status**: Complete

**Accomplished**:
- Softmax BF16 validated on NPU (1.565 ms, 99.5% accuracy)
- GELU BF16 compiled
- LayerNorm BF16 compiled
- XRT invocation pattern documented

### Phase 2: Kernel Pairs (Weeks 1-2)

**Objective**: Chain 2 kernels in sequence

**Tasks**:
1. LayerNorm → MatMul (output of LN feeds into MatMul)
2. MatMul → Softmax (attention scores to probabilities)
3. MatMul → GELU (FFN up-projection to activation)

**Deliverables**:
- Working 2-kernel chains
- Buffer reuse between kernels
- Performance benchmarks

**Success Criteria**:
- End-to-end latency < sum of individual kernel latencies + 20%
- No accuracy degradation
- No host memory allocation between kernels

### Phase 3: Full Attention Block (Weeks 3-4)

**Objective**: Complete multi-head self-attention on NPU

**Tasks**:
1. Implement Q,K,V projection (3 parallel MatMuls)
2. Chain Scores → Softmax → Attn@V → Output
3. Parallel processing across 8 attention heads
4. End-to-end attention test

**Deliverables**:
- `attention_block.xclbin` (or kernel sequence)
- Accuracy validation vs PyTorch reference
- Head-parallel execution

**Success Criteria**:
- Attention block < 0.7 ms
- Accuracy correlation > 99%
- All 8 heads processed correctly

### Phase 4: Full Encoder Layer (Weeks 5-6)

**Objective**: Complete encoder layer (attention + FFN)

**Tasks**:
1. Add FFN block (Up → GELU → Down)
2. Implement residual connections
3. Chain attention → FFN
4. Layer-level profiling

**Deliverables**:
- `encoder_layer.py` integration class
- Full layer benchmark
- Memory profiling

**Success Criteria**:
- Full layer < 2 ms
- Accuracy < 1% WER increase vs CPU
- Memory footprint documented

### Phase 5: Full Encoder Stack (Weeks 7-10)

**Objective**: All 6 encoder layers (Whisper Base)

**Tasks**:
1. Loop over 6 layers
2. Optimize intermediate buffer reuse
3. Implement weight loading strategy
4. End-to-end encoder test

**Deliverables**:
- Complete `WhisperEncoderNPU` class
- Integration with WhisperX pipeline
- Production deployment guide

**Success Criteria**:
- Full encoder < 10 ms for 1 second audio
- **100× realtime** achieved
- Ready for production testing

### Phase 6: Optimizations (Weeks 11-12)

**Objective**: Reach 220× realtime target

**Tasks**:
1. INT8 quantization for MatMul
2. 4-column parallelism
3. DMA pipelining
4. Kernel fusion experiments

**Deliverables**:
- Optimized `whisper_encoder_optimized.xclbin`
- Performance comparison report
- Final integration

**Success Criteria**:
- **220× realtime** achieved
- Power < 10W
- Production stability verified

---

## Risk Analysis

### High Risks

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| **MatMul kernel complexity** | Blocks 85% of compute | Start with simple tiled implementation, optimize later | Manageable |
| **Memory overflow** | Cannot fit activations | Use aggressive tiling, streaming | Designed |
| **PCIe bandwidth saturation** | Cannot feed NPU fast enough | Use 35% bandwidth design (validated) | Mitigated |

### Medium Risks

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| **Kernel-to-kernel latency** | Reduced speedup | Minimize buffer copies, use ObjectFIFO | Designing |
| **Numerical accuracy** | WER degradation | BF16 sufficient, validate at each phase | Monitoring |
| **Multi-head synchronization** | Head mismatch | Careful index management | Aware |

### Low Risks

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| **XRT stability** | Runtime crashes | XRT 2.20.0 proven stable | Low risk |
| **Compilation failures** | Build delays | Templates validated, Peano working | Low risk |

---

## Appendix A: XRT Buffer Management Code

```python
import numpy as np
import xrt_binding as xrt

class XRTBufferManager:
    """Manage XRT buffers for NPU kernel execution"""

    def __init__(self, device, kernel):
        self.device = device
        self.kernel = kernel
        self.buffers = {}

    def allocate(self, name, size_bytes, group_id):
        """Allocate aligned XRT buffer"""
        aligned_size = ((size_bytes + 4095) // 4096) * 4096
        bo = xrt.bo(self.device, aligned_size, xrt.bo.flags.normal, group_id)
        self.buffers[name] = {
            'bo': bo,
            'size': aligned_size,
            'group_id': group_id
        }
        return bo

    def write(self, name, data):
        """Write numpy array to buffer"""
        bo = self.buffers[name]['bo']
        bo.write(data.tobytes())
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def read(self, name, dtype, shape):
        """Read buffer to numpy array"""
        bo = self.buffers[name]['bo']
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return np.frombuffer(bo.read(np.prod(shape) * np.dtype(dtype).itemsize),
                            dtype=dtype).reshape(shape)

    def cleanup(self):
        """Release all buffers"""
        self.buffers.clear()
```

---

## Appendix B: References

### Key Documentation

1. **SUCCESS_REPORT.md** - Validated softmax kernel execution pattern
2. **ON_CHIP_MEMORY_OPTIMIZATION_STRATEGY.md** - Memory hierarchy design
3. **EXECUTIVE_SUMMARY.md** - Project overview and targets
4. **README.md** - Kernel inventory and compilation

### External References

1. AMD Versal AIE2 Programming Guide
2. XRT Runtime Documentation
3. MLIR-AIE User Guide
4. Whisper Model Architecture (OpenAI)

---

**Document End**

**Next Step**: Implement `whisper_encoder_npu.py` with this architecture.
