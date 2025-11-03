# NPU Whisper Encoder - Phased Implementation Plan

**Date**: November 2, 2025
**Project Manager**: NPU Implementation Lead
**Target**: 220x realtime Whisper Base encoder
**Timeline**: 9-14 weeks (63-98 days)
**Status**: 📋 **READY TO BEGIN**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Phase 1: Fix Critical Blockers (Weeks 1-3)](#2-phase-1-fix-critical-blockers-weeks-1-3)
3. [Phase 2: Complete Kernel Wrappers (Weeks 4-5)](#3-phase-2-complete-kernel-wrappers-weeks-4-5)
4. [Phase 3: Unified XCLBIN Creation (Weeks 6-7)](#4-phase-3-unified-xclbin-creation-weeks-6-7)
5. [Phase 4: Full Encoder Integration (Weeks 8-10)](#5-phase-4-full-encoder-integration-weeks-8-10)
6. [Phase 5: Optimization & 220x Target (Weeks 11-12)](#6-phase-5-optimization--220x-target-weeks-11-12)
7. [Phase 6: Production Hardening (Weeks 13-14)](#7-phase-6-production-hardening-weeks-13-14)
8. [Risk Management](#8-risk-management)
9. [Success Metrics](#9-success-metrics)
10. [Resource Requirements](#10-resource-requirements)

---

## 1. Executive Summary

### 1.1 Project Overview

**Goal**: Complete NPU encoder implementation for 220x realtime Whisper transcription

**Current State**: 75% infrastructure complete, integration blocked

**Critical Path**:
1. Fix attention buffer issue → Fix matmul wrapper → Create unified XCLBIN → Integrate → Optimize

**Timeline**:
- **Best Case**: 9 weeks
- **Expected**: 12 weeks
- **Worst Case**: 14 weeks

**Confidence**: 70% (high probability of success)

### 1.2 Phase Summary

| Phase | Duration | Goal | Success Metric |
|-------|----------|------|----------------|
| **Phase 1** | Weeks 1-3 | Fix critical blockers | Attention working, MatMul 68x faster |
| **Phase 2** | Weeks 4-5 | Complete wrappers | LayerNorm + GELU tested |
| **Phase 3** | Weeks 6-7 | Unified XCLBIN | All kernels in one binary |
| **Phase 4** | Weeks 8-10 | Full integration | End-to-end encoder working |
| **Phase 5** | Weeks 11-12 | Optimization | 220x realtime achieved |
| **Phase 6** | Weeks 13-14 | Production | Error handling, docs, deployment |

### 1.3 Critical Milestones

- **Week 2**: Attention buffer issue fixed ✓
- **Week 3**: MatMul wrapper 68x faster ✓
- **Week 5**: All 4 kernel wrappers tested ✓
- **Week 7**: Unified XCLBIN compiles ✓
- **Week 10**: Full encoder working ✓
- **Week 12**: 220x realtime achieved ✓
- **Week 14**: Production ready ✓

---

## 2. Phase 1: Fix Critical Blockers (Weeks 1-3)

### 2.1 Overview

**Duration**: 3 weeks (21 days)
**Effort**: 60-80 hours
**Priority**: 🔴 **CRITICAL** - Blocks all downstream work
**Goal**: Fix attention buffer issue and matmul wrapper performance

### 2.2 Week 1: Fix Attention Buffer Issue (16-24 hours)

**Objective**: Get attention kernel returning valid output instead of zeros

**Current Status**:
- Kernel compiles and executes successfully
- Output buffer contains all zeros
- XRT warning about buffer bank mismatch
- 10 different buffer configurations tested, all fail

**Task 1.1**: Debug Zero Output (Day 1-2, 8-12 hours)

**Steps**:
```bash
# 1. Create minimal reproduction case
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels

# 2. Test with known input (identity Q=K=V)
python3 << 'EOF'
from npu_attention_wrapper import NPUAttention
import numpy as np

# Create identity test
attn = NPUAttention()
Q = np.eye(64, dtype=np.int8)
K = np.eye(64, dtype=np.int8)
V = np.eye(64, dtype=np.int8)

output = attn(Q, K, V)
print(f"Output non-zero: {np.count_nonzero(output)} / {output.size}")
print(f"Output range: [{output.min()}, {output.max()}]")
print(f"Expected: Identity matrix, got: zeros={np.all(output == 0)}")
EOF

# 3. Add debug prints to kernel wrapper
vim npu_attention_wrapper.py
# Add prints before/after each DMA sync
# Verify data is written to input_bo
# Verify kernel actually executes
# Check output_bo before sync

# 4. Test all 3 attention XCLBIN variants
for xclbin in build_attention/*.xclbin; do
    echo "Testing $xclbin..."
    python3 test_attention.py --xclbin=$xclbin
done
```

**Expected Issues**:
- **Hypothesis 1**: Input buffer not synced properly
- **Hypothesis 2**: Output offset incorrect
- **Hypothesis 3**: Kernel needs warmup call
- **Hypothesis 4**: Q/K/V format wrong (need transpose?)

**Success Criteria**:
- [ ] Output buffer contains non-zero values
- [ ] At least 50% of output elements active
- [ ] Output values in expected range [-127, +127]

**Task 1.2**: Validate Attention Accuracy (Day 3-4, 8-12 hours)

**Steps**:
```python
# Compare NPU attention vs CPU reference
import torch
import torch.nn.functional as F

def cpu_attention(Q, K, V):
    """Reference implementation"""
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(64)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output

def test_attention_accuracy():
    Q = torch.randn(64, 64)
    K = torch.randn(64, 64)
    V = torch.randn(64, 64)

    # CPU reference
    cpu_out = cpu_attention(Q, K, V)

    # NPU (quantized)
    npu_out = npu_attention(Q, K, V)

    # Compare
    correlation = np.corrcoef(cpu_out.flatten(), npu_out.flatten())[0,1]
    print(f"Correlation: {correlation:.4f}")

    assert correlation > 0.80, "Attention accuracy too low!"
```

**Success Criteria**:
- [ ] Correlation >0.80 with CPU reference
- [ ] No NaN or inf values
- [ ] Output distribution reasonable (mean ~0, std ~20)

**Deliverables**:
- [ ] `attention_buffer_fix.md` - Documentation of fix
- [ ] `test_attention_minimal.py` - Minimal test case
- [ ] Updated `npu_attention_wrapper.py` with fix

---

### 2.3 Week 2-3: Fix MatMul Wrapper Performance (20-30 hours)

**Objective**: Speed up matmul wrapper from 1,082s to 15.9s (68x faster)

**Current Status**:
- NPU kernel works perfectly: 0.484ms per 16×16 tile
- Wrapper calls kernel 32,768 times for single 512×512 matmul
- Per-call overhead: 32.54ms (DMA sync, memory copies)
- **Result**: 68x slower than it should be

**Root Cause**: Triple nested loop with per-tile DMA sync

**Task 2.1**: Implement Tile Batching (Day 5-8, 12-16 hours)

**Current Code** (slow):
```python
# Lines 213-242: BROKEN - 32,768 NPU calls
for i in range(M_tiles):  # 32
    for j in range(N_tiles):  # 32
        for k in range(K_tiles):  # 32
            result_tile = self._matmul_tile(A_tile, B_tile)  # NPU call
```

**New Code** (fast):
```python
def _matmul_batched(self, A, B):
    """Batch all tiles into single NPU call"""
    M, K = A.shape
    K2, N = B.shape

    # Pad to tile boundaries
    M_tiles = (M + 15) // 16
    N_tiles = (N + 15) // 16
    K_tiles = (K + 15) // 16

    A_padded = np.zeros((M_tiles * 16, K_tiles * 16), dtype=np.int8)
    B_padded = np.zeros((K_tiles * 16, N_tiles * 16), dtype=np.int8)
    A_padded[:M, :K] = A
    B_padded[:K, :N] = B

    # Pack all A tiles into contiguous buffer
    total_A_tiles = M_tiles * K_tiles
    A_tiles_packed = np.empty((total_A_tiles, 16, 16), dtype=np.int8)

    tile_idx = 0
    for i in range(M_tiles):
        for k in range(K_tiles):
            A_tiles_packed[tile_idx] = A_padded[
                i*16:(i+1)*16,
                k*16:(k+1)*16
            ]
            tile_idx += 1

    # Pack all B tiles
    total_B_tiles = K_tiles * N_tiles
    B_tiles_packed = np.empty((total_B_tiles, 16, 16), dtype=np.int8)

    tile_idx = 0
    for k in range(K_tiles):
        for j in range(N_tiles):
            B_tiles_packed[tile_idx] = B_padded[
                k*16:(k+1)*16,
                j*16:(j+1)*16
            ]
            tile_idx += 1

    # Allocate large buffers (ONCE)
    total_tiles = M_tiles * N_tiles * K_tiles
    large_input_size = total_tiles * 512  # 256 A + 256 B per tile
    large_output_size = total_tiles * 256  # 256 per tile

    large_input_bo = xrt.bo(
        self.device, large_input_size,
        xrt.bo.flags.host_only, self.kernel.group_id(3)
    )
    large_output_bo = xrt.bo(
        self.device, large_output_size,
        xrt.bo.flags.host_only, self.kernel.group_id(4)
    )

    # Write all tiles at once
    offset = 0
    for i in range(M_tiles):
        for j in range(N_tiles):
            for k in range(K_tiles):
                A_idx = i * K_tiles + k
                B_idx = k * N_tiles + j

                # Concatenate A and B tiles
                packed = np.concatenate([
                    A_tiles_packed[A_idx].flatten(),
                    B_tiles_packed[B_idx].flatten()
                ])

                large_input_bo.write(packed.tobytes(), offset)
                offset += 512

    # Single DMA sync (TO_DEVICE)
    large_input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                        large_input_size, 0)

    # Execute kernel (processes all tiles internally)
    run = self.kernel(
        self.opcode,
        self.instr_bo,
        self.n_insts,
        large_input_bo,
        large_output_bo,
        total_tiles  # Number of tiles to process
    )
    run.wait(timeout=10000)  # 10 second timeout

    # Single DMA sync (FROM_DEVICE)
    large_output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
                         large_output_size, 0)

    # Read all results at once
    output_bytes = large_output_bo.read(large_output_size, 0)
    all_outputs = np.frombuffer(output_bytes, dtype=np.int8)
    all_outputs = all_outputs.reshape(total_tiles, 16, 16)

    # Accumulate partial results
    C_padded = np.zeros((M_tiles * 16, N_tiles * 16), dtype=np.int32)

    tile_idx = 0
    for i in range(M_tiles):
        for j in range(N_tiles):
            acc = np.zeros((16, 16), dtype=np.int32)
            for k in range(K_tiles):
                acc += all_outputs[tile_idx].astype(np.int32)
                tile_idx += 1

            C_padded[i*16:(i+1)*16, j*16:(j+1)*16] = acc

    # Extract result
    C = C_padded[:M, :N]
    return np.clip(C, -128, 127).astype(np.int8)
```

**Expected Speedup**: 1,082s → 15.9s (68x faster)

**Task 2.2**: Optimize Buffer Management (Day 9-11, 8-14 hours)

**Improvements**:
1. **Pre-allocate buffers once** (don't allocate per matmul)
2. **Reuse buffers** across multiple calls
3. **Eliminate unnecessary memory copies**
4. **Use memory pools** for tile buffers

**Code**:
```python
class NPUMatmul:
    def __init__(self, ...):
        # Pre-allocate large buffers (reused for all matmuls)
        max_tiles = 64 * 64 * 64  # Support up to 512×512 @ 512×512
        self.large_input_bo = xrt.bo(
            self.device, max_tiles * 512,
            xrt.bo.flags.host_only, self.kernel.group_id(3)
        )
        self.large_output_bo = xrt.bo(
            self.device, max_tiles * 256,
            xrt.bo.flags.host_only, self.kernel.group_id(4)
        )

        # Tile buffer pool (avoid allocations)
        self.tile_buffer_pool = {
            'A': np.empty((max_tiles, 16, 16), dtype=np.int8),
            'B': np.empty((max_tiles, 16, 16), dtype=np.int8),
            'C': np.empty((max_tiles, 16, 16), dtype=np.int8)
        }
```

**Expected Improvement**: 15.9s → 10s (additional 1.6x speedup)

**Success Criteria**:
- [ ] 500×512 @ 512×512 matmul completes in <15 seconds
- [ ] Per-tile overhead <1ms (down from 32.54ms)
- [ ] Total speedup >60x from original

**Deliverables**:
- [ ] `npu_matmul_wrapper_fixed.py` - Fixed wrapper
- [ ] `MATMUL_PERFORMANCE_FIX.md` - Documentation
- [ ] `test_matmul_batched.py` - Validation tests

---

### 2.4 Phase 1 Summary

**Total Effort**: 60-80 hours over 3 weeks

**Completion Criteria**:
- [ ] Attention kernel returns valid output (not zeros)
- [ ] Attention correlation >0.80 with CPU
- [ ] MatMul wrapper 60-68x faster than original
- [ ] Both kernels tested and validated

**Risks**:
- Attention buffer issue may be deeper (kernel itself broken)
- MatMul batching may hit XRT buffer size limits
- Timeline may extend to 4 weeks if major issues found

**Mitigation**:
- Have CPU fallback ready for attention
- Test intermediate batching (1000 tiles first, then scale up)
- Add +1 week buffer to timeline

**Phase 1 Deliverables**:
1. Fixed attention wrapper
2. Fixed matmul wrapper
3. Test suites for both
4. Documentation of fixes
5. Performance benchmarks

**Exit Criteria**: Both critical blockers resolved, ready for Phase 2

---

## 3. Phase 2: Complete Kernel Wrappers (Weeks 4-5)

### 3.1 Overview

**Duration**: 2 weeks (14 days)
**Effort**: 24-36 hours
**Priority**: 🟡 **HIGH** - Needed for full encoder
**Goal**: Create and test LayerNorm and GELU wrappers

### 3.2 Week 4: LayerNorm Wrapper (12-18 hours)

**Objective**: Create NPULayerNorm wrapper class

**Task 3.1**: Implement LayerNorm Wrapper (Day 15-17, 8-12 hours)

**Files**:
- Create: `npu/wrappers/npu_layernorm.py`
- XCLBIN: `whisper_encoder_kernels/build_layernorm/layernorm_simple.xclbin` (already compiled)

**Implementation**:
```python
class NPULayerNorm:
    """
    Layer Normalization on NPU

    Input: (seq_len, hidden_dim) INT8
    Output: (seq_len, hidden_dim) INT8
    Parameters: gamma, beta (hidden_dim,) INT8
    """

    def __init__(self, hidden_dim=384, device_id=0):
        # Initialize NPU
        self.device = xrt.device(device_id)

        # Load XCLBIN
        xclbin_path = "build_layernorm/layernorm_simple.xclbin"
        insts_path = "build_layernorm/insts.bin"

        xclbin = xrt.xclbin(xclbin_path)
        self.device.register_xclbin(xclbin)
        uuid = xclbin.get_uuid()
        self.hw_ctx = xrt.hw_context(self.device, uuid)
        self.kernel = xrt.kernel(self.hw_ctx, "MLIR_AIE")

        # Load instructions
        with open(insts_path, "rb") as f:
            self.insts = f.read()
        self.n_insts = len(self.insts)

        # Create buffers
        max_size = 3000 * 384  # Max sequence length × hidden dim
        self.instr_bo = xrt.bo(self.device, self.n_insts,
                               xrt.bo.flags.cacheable, self.kernel.group_id(1))
        self.input_bo = xrt.bo(self.device, max_size,
                               xrt.bo.flags.host_only, self.kernel.group_id(2))
        self.params_bo = xrt.bo(self.device, 384 * 2,  # gamma + beta
                                xrt.bo.flags.host_only, self.kernel.group_id(3))
        self.output_bo = xrt.bo(self.device, max_size,
                                xrt.bo.flags.host_only, self.kernel.group_id(4))

        # Write instructions once
        self.instr_bo.write(self.insts, 0)
        self.instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                           self.n_insts, 0)

        # Opcode
        self.opcode = 3

    def __call__(self, x, gamma, beta):
        """
        Apply layer normalization

        Args:
            x: Input (seq_len, hidden_dim) INT8 or FP32
            gamma: Scale parameters (hidden_dim,) INT8 or FP32
            beta: Shift parameters (hidden_dim,) INT8 or FP32

        Returns:
            Normalized output (seq_len, hidden_dim) INT8
        """
        # Quantize if needed
        if x.dtype == np.float32:
            x = self._quantize(x)
        if gamma.dtype == np.float32:
            gamma = self._quantize(gamma)
        if beta.dtype == np.float32:
            beta = self._quantize(beta)

        seq_len, hidden_dim = x.shape
        assert hidden_dim == 384, f"Expected hidden_dim=384, got {hidden_dim}"

        # Write input
        self.input_bo.write(x.tobytes(), 0)
        self.input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                           x.nbytes, 0)

        # Write parameters
        params = np.concatenate([gamma, beta])
        self.params_bo.write(params.tobytes(), 0)
        self.params_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                            params.nbytes, 0)

        # Execute
        run = self.kernel(self.opcode, self.instr_bo, self.n_insts,
                          self.input_bo, self.params_bo, self.output_bo,
                          seq_len, hidden_dim)
        run.wait(1000)

        # Read output
        self.output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
                            x.nbytes, 0)
        output_bytes = self.output_bo.read(x.nbytes, 0)
        output = np.frombuffer(output_bytes, dtype=np.int8)
        output = output.reshape(seq_len, hidden_dim)

        return output

    def _quantize(self, x):
        """Quantize FP32 to INT8"""
        max_val = np.abs(x).max()
        if max_val > 0:
            scale = 127.0 / max_val
            return np.round(x * scale).astype(np.int8)
        return x.astype(np.int8)
```

**Task 3.2**: Test LayerNorm (Day 18-19, 4-6 hours)

**Validation**:
```python
def test_layernorm():
    # Create test data
    x = np.random.randn(3000, 384).astype(np.float32)
    gamma = np.ones(384, dtype=np.float32)
    beta = np.zeros(384, dtype=np.float32)

    # CPU reference
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    x_norm_cpu = (x - mean) / np.sqrt(var + 1e-5)
    x_norm_cpu = gamma * x_norm_cpu + beta

    # NPU
    ln = NPULayerNorm()
    x_norm_npu = ln(x, gamma, beta)

    # Compare
    correlation = np.corrcoef(
        x_norm_cpu.flatten(),
        x_norm_npu.astype(np.float32).flatten()
    )[0, 1]

    print(f"LayerNorm correlation: {correlation:.4f}")
    assert correlation > 0.95, "LayerNorm accuracy too low!"

    # Benchmark
    import time
    start = time.perf_counter()
    for _ in range(100):
        _ = ln(x, gamma, beta)
    elapsed = (time.perf_counter() - start) / 100

    print(f"LayerNorm time: {elapsed*1000:.3f}ms")
    print(f"Expected: <1ms, Actual: {elapsed*1000:.3f}ms")
    assert elapsed < 0.005, "LayerNorm too slow!"
```

**Success Criteria**:
- [ ] Correlation >0.95 with CPU reference
- [ ] Processing time <1ms for 3000×384
- [ ] No errors or crashes

---

### 3.3 Week 5: GELU Wrapper (12-18 hours)

**Objective**: Create NPUGELU wrapper class

**Task 3.3**: Implement GELU Wrapper (Day 20-22, 8-12 hours)

**Files**:
- Create: `npu/wrappers/npu_gelu.py`
- XCLBIN: `whisper_encoder_kernels/build_gelu/gelu_2048.xclbin` (already compiled)

**Implementation**:
```python
class NPUGELU:
    """
    GELU activation on NPU using lookup table

    Input: (batch, dim) INT8
    Output: (batch, dim) INT8
    """

    def __init__(self, device_id=0):
        # Initialize NPU (similar to LayerNorm)
        # ...

        # GELU lookup table (precomputed)
        self.gelu_lut = self._generate_gelu_lut()

        # Load LUT to NPU memory
        self.lut_bo = xrt.bo(self.device, 256,
                             xrt.bo.flags.host_only, self.kernel.group_id(5))
        self.lut_bo.write(self.gelu_lut.tobytes(), 0)
        self.lut_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                         256, 0)

    def _generate_gelu_lut(self):
        """Generate GELU lookup table for INT8"""
        x_int8 = np.arange(-128, 128, dtype=np.int8)
        x_float = x_int8.astype(np.float32) / 127.0  # Normalize to [-1, 1]

        # GELU(x) = x * Φ(x) where Φ is CDF of standard normal
        gelu = x_float * 0.5 * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x_float + 0.044715 * x_float**3)
        ))

        # Quantize back to INT8
        gelu_int8 = np.round(gelu * 127.0).astype(np.int8)
        return gelu_int8

    def __call__(self, x):
        """Apply GELU activation"""
        # Write input
        self.input_bo.write(x.tobytes(), 0)
        self.input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                           x.nbytes, 0)

        # Execute (kernel uses LUT internally)
        run = self.kernel(self.opcode, self.instr_bo, self.n_insts,
                          self.input_bo, self.lut_bo, self.output_bo,
                          x.size)
        run.wait(1000)

        # Read output
        self.output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
                            x.nbytes, 0)
        output_bytes = self.output_bo.read(x.nbytes, 0)
        output = np.frombuffer(output_bytes, dtype=np.int8)
        output = output.reshape(x.shape)

        return output
```

**Task 3.4**: Test GELU (Day 23-24, 4-6 hours)

**Validation**:
```python
def test_gelu():
    # Test data
    x = np.random.randn(3000, 1536).astype(np.float32)

    # CPU reference
    def gelu_cpu(x):
        return x * 0.5 * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
        ))

    gelu_ref = gelu_cpu(x)

    # NPU
    gelu_npu = NPUGELU()
    x_int8 = (x * 127).astype(np.int8)
    output = gelu_npu(x_int8)
    output_float = output.astype(np.float32) / 127.0

    # Compare
    correlation = np.corrcoef(gelu_ref.flatten(), output_float.flatten())[0,1]
    print(f"GELU correlation: {correlation:.4f}")
    assert correlation > 0.95, "GELU accuracy too low!"

    # Benchmark
    import time
    start = time.perf_counter()
    for _ in range(100):
        _ = gelu_npu(x_int8)
    elapsed = (time.perf_counter() - start) / 100

    print(f"GELU time: {elapsed*1000:.3f}ms")
    assert elapsed < 0.002, "GELU too slow!"
```

**Success Criteria**:
- [ ] Correlation >0.95 with CPU GELU
- [ ] Processing time <2ms for 3000×1536
- [ ] Lookup table correct (256 entries)

---

### 3.4 Phase 2 Summary

**Total Effort**: 24-36 hours over 2 weeks

**Completion Criteria**:
- [ ] NPULayerNorm wrapper implemented and tested
- [ ] NPUGELU wrapper implemented and tested
- [ ] Both wrappers achieve >0.95 correlation
- [ ] Both wrappers meet performance targets

**Deliverables**:
1. `npu/wrappers/npu_layernorm.py`
2. `npu/wrappers/npu_gelu.py`
3. `test_layernorm.py`
4. `test_gelu.py`
5. Documentation for both wrappers

**Exit Criteria**: All 4 kernel wrappers (MatMul, Attention, LayerNorm, GELU) working

---

## 4. Phase 3: Unified XCLBIN Creation (Weeks 6-7)

### 4.1 Overview

**Duration**: 2 weeks (14 days)
**Effort**: 40-60 hours
**Priority**: 🔴 **CRITICAL** - Required for 220x performance
**Goal**: Combine all 4 kernels into single XCLBIN

### 4.2 Week 6: MLIR Design and Compilation (24-32 hours)

**Objective**: Create unified MLIR file with all kernels

**Background**:
- Current: 4 separate XCLBINs (matmul, attention, layernorm, gelu)
- Problem: Can only load ONE XCLBIN at a time
- Solution: Compile all kernels into single XCLBIN
- Challenge: MLIR kernel fusion and tile allocation

**Task 4.1**: Design Unified MLIR (Day 25-27, 12-16 hours)

**MLIR Structure**:
```mlir
// whisper_encoder_unified.mlir
module @whisper_encoder {
  aie.device(npu1) {
    // Tile assignments
    %tile_matmul_0 = aie.tile(0, 3)  // Column 0, Row 3
    %tile_matmul_1 = aie.tile(1, 3)  // Column 1, Row 3
    %tile_attn_0 = aie.tile(2, 3)    // Column 2, Row 3
    %tile_attn_1 = aie.tile(3, 3)    // Column 3, Row 3
    %tile_ln_0 = aie.tile(0, 2)      // Column 0, Row 2
    %tile_gelu_0 = aie.tile(1, 2)    // Column 1, Row 2

    // MatMul kernel on tile (0,3)
    %core_matmul_0 = aie.core(%tile_matmul_0) {
      // Link to compiled C++ kernel
      func.func @matmul_16x16(%arg0: memref<16x16xi8>,
                               %arg1: memref<16x16xi8>,
                               %arg2: memref<16x16xi32>) {
        // Implementation from matmul_int8.o
      }
      aie.end
    } { link_with="matmul_int8.o" }

    // Attention kernel on tile (2,3)
    %core_attn_0 = aie.core(%tile_attn_0) {
      func.func @attention_64x64(...) {
        // Implementation from attention_int8.o
      }
      aie.end
    } { link_with="attention_int8.o" }

    // LayerNorm kernel on tile (0,2)
    %core_ln_0 = aie.core(%tile_ln_0) {
      func.func @layernorm(...) {
        // Implementation from layernorm_int8.o
      }
      aie.end
    } { link_with="layernorm_int8.o" }

    // GELU kernel on tile (1,2)
    %core_gelu_0 = aie.core(%tile_gelu_0) {
      func.func @gelu(...) {
        // Implementation from gelu_int8.o
      }
      aie.end
    } { link_with="gelu_lut.o" }

    // Data movement (ObjectFIFOs for modern MLIR-AIE)
    %buffer_input = aie.objectfifo @input_fifo (%tile_shim_0, %tile_matmul_0) : !aie.objectfifo<memref<16x16xi8>>
    %buffer_output = aie.objectfifo @output_fifo (%tile_matmul_0, %tile_shim_0) : !aie.objectfifo<memref<16x16xi8>>

    // ... more ObjectFIFOs for other kernels ...
  }
}
```

**Task 4.2**: Compile Unified XCLBIN (Day 28-29, 8-12 hours)

**Build Process**:
```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels

# Create build directory
mkdir -p build_unified
cd build_unified

# Compile all C++ kernels
peano --target=AIE2 ../matmul_int8.c -o matmul_int8.o
peano --target=AIE2 ../attention_int8.c -o attention_int8.o
peano --target=AIE2 ../layernorm_int8.c -o layernorm_int8.o
peano --target=AIE2 ../gelu_int8.c -o gelu_int8.o

# Lower MLIR
aie-opt --aie-canonicalize-device \
        --aie-objectFifo-stateful-transform \
        --aie-create-pathfinder-flows \
        --aie-assign-buffer-addresses \
        ../whisper_encoder_unified.mlir -o lowered.mlir

# Generate XCLBIN
aiecc.py --xclbin-name=whisper_encoder_unified.xclbin \
         --peano-install-dir=/opt/amd/peano \
         lowered.mlir

# Verify output
ls -lh whisper_encoder_unified.xclbin
# Expected: ~100-150 KB (4 kernels combined)
```

**Expected Challenges**:
1. **Tile Conflicts**: Kernels assigned to same tiles
   - Solution: Manual tile assignment in MLIR
2. **Buffer Conflicts**: ObjectFIFO routing conflicts
   - Solution: Use different ObjectFIFO names for each kernel
3. **Compilation Errors**: Linker errors with multiple .o files
   - Solution: Create combined archive with llvm-ar

**Task 4.3**: Debug Compilation Issues (Day 30-31, 4-8 hours)

**Common Issues**:
```bash
# Issue 1: Undefined symbols
# Solution: Check all kernels have extern "C" linkage
llvm-nm -g matmul_int8.o | grep matmul_16x16

# Issue 2: Tile assignment conflicts
# Solution: Verify in MLIR that each kernel uses unique tiles
grep "aie.tile" whisper_encoder_unified.mlir

# Issue 3: XCLBIN too large
# Solution: Optimize kernel code, reduce memory usage
ls -lh whisper_encoder_unified.xclbin
# Should be <500 KB
```

**Success Criteria**:
- [ ] Unified XCLBIN compiles successfully
- [ ] XCLBIN size <500 KB
- [ ] No compilation errors or warnings
- [ ] All 4 kernels present in XCLBIN

---

### 4.3 Week 7: Unified XCLBIN Testing (16-28 hours)

**Objective**: Validate unified XCLBIN works on NPU

**Task 4.4**: Load and Initialize (Day 32-33, 8-12 hours)

**Test Code**:
```python
import pyxrt as xrt
import numpy as np

# Load unified XCLBIN
device = xrt.device(0)
xclbin = xrt.xclbin("build_unified/whisper_encoder_unified.xclbin")
device.register_xclbin(xclbin)
uuid = xclbin.get_uuid()
hw_ctx = xrt.hw_context(device, uuid)

# Get all kernel handles
kernel = xrt.kernel(hw_ctx, "MLIR_AIE")

print("✅ Unified XCLBIN loaded successfully!")

# Test individual kernels
print("\nTesting individual kernels from unified XCLBIN:")

# 1. Test MatMul
print("1. Testing MatMul...")
# ... (use existing test code)

# 2. Test Attention
print("2. Testing Attention...")
# ... (use existing test code)

# 3. Test LayerNorm
print("3. Testing LayerNorm...")
# ... (use existing test code)

# 4. Test GELU
print("4. Testing GELU...")
# ... (use existing test code)

print("\n✅ All kernels in unified XCLBIN working!")
```

**Task 4.5**: Update Wrappers for Unified XCLBIN (Day 34-36, 8-16 hours)

**Modify All Wrappers**:
```python
class NPUEncoderKernels:
    """
    Unified kernel loader for all encoder operations
    Loads single XCLBIN with all kernels
    """

    def __init__(self, device_id=0):
        # Load unified XCLBIN once
        self.device = xrt.device(device_id)
        xclbin_path = "build_unified/whisper_encoder_unified.xclbin"
        xclbin = xrt.xclbin(xclbin_path)
        self.device.register_xclbin(xclbin)
        uuid = xclbin.get_uuid()
        self.hw_ctx = xrt.hw_context(self.device, uuid)
        self.kernel = xrt.kernel(self.hw_ctx, "MLIR_AIE")

        # Create individual kernel wrappers (share same XCLBIN)
        self.matmul = NPUMatmul(device=self.device, kernel=self.kernel)
        self.attention = NPUAttention(device=self.device, kernel=self.kernel)
        self.layernorm = NPULayerNorm(device=self.device, kernel=self.kernel)
        self.gelu = NPUGELU(device=self.device, kernel=self.kernel)

# Update wrapper __init__ to accept pre-loaded device/kernel
class NPUMatmul:
    def __init__(self, device=None, kernel=None, device_id=0):
        if device is None:
            # Old behavior: load own XCLBIN
            self.device = xrt.device(device_id)
            # ... load matmul XCLBIN ...
        else:
            # New behavior: use pre-loaded unified XCLBIN
            self.device = device
            self.kernel = kernel
        # Rest of initialization...
```

**Success Criteria**:
- [ ] All 4 wrappers work with unified XCLBIN
- [ ] No performance regression
- [ ] Can switch between kernels without reloading XCLBIN

---

### 4.4 Phase 3 Summary

**Total Effort**: 40-60 hours over 2 weeks

**Completion Criteria**:
- [ ] Unified XCLBIN compiled and tested
- [ ] All 4 kernels accessible from single XCLBIN
- [ ] Wrappers updated to use unified XCLBIN
- [ ] No performance regression

**Deliverables**:
1. `whisper_encoder_unified.mlir` - Unified MLIR source
2. `build_unified/whisper_encoder_unified.xclbin` - Compiled binary
3. `NPUEncoderKernels` class - Unified loader
4. Updated wrapper classes
5. Test suite for unified XCLBIN

**Exit Criteria**: Can load all kernels from single XCLBIN, ready for full encoder integration

---

## 5. Phase 4: Full Encoder Integration (Weeks 8-10)

### 5.1 Overview

**Duration**: 3 weeks (21 days)
**Effort**: 48-72 hours
**Priority**: 🟡 **HIGH** - Core functionality
**Goal**: Complete end-to-end encoder implementation

### 5.2 Week 8: Single Layer Integration (16-24 hours)

**Objective**: Implement and test single encoder layer

**Task 5.1**: Encoder Layer Implementation (Day 37-40, 12-16 hours)

**File**: `npu/encoder/encoder_layer.py`

```python
class WhisperNPUEncoderLayer:
    """
    Single Whisper encoder layer on NPU

    Architecture:
      1. LayerNorm
      2. Multi-head self-attention
      3. Residual
      4. LayerNorm
      5. FFN (Linear → GELU → Linear)
      6. Residual
    """

    def __init__(self, npu_kernels, layer_idx, d_model=384, n_heads=6, d_ff=1536):
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 384 / 6 = 64
        self.d_ff = d_ff

        # NPU kernel wrappers (shared across layers)
        self.matmul = npu_kernels.matmul
        self.attention = npu_kernels.attention
        self.layernorm = npu_kernels.layernorm
        self.gelu = npu_kernels.gelu

        # Load weights for this layer
        self.load_weights(layer_idx)

    def load_weights(self, layer_idx):
        """Load quantized INT8 weights for this layer"""
        weight_path = f"models/whisper-base-npu-int8/encoder_layer_{layer_idx}.npz"
        weights = np.load(weight_path)

        self.attn_q_weight = weights['attn_q_weight']  # (384, 384) INT8
        self.attn_k_weight = weights['attn_k_weight']  # (384, 384) INT8
        self.attn_v_weight = weights['attn_v_weight']  # (384, 384) INT8
        self.attn_out_weight = weights['attn_out_weight']  # (384, 384) INT8

        self.ffn_1_weight = weights['ffn_1_weight']  # (384, 1536) INT8
        self.ffn_2_weight = weights['ffn_2_weight']  # (1536, 384) INT8

        self.ln1_gamma = weights['ln1_gamma']  # (384,) INT8
        self.ln1_beta = weights['ln1_beta']  # (384,) INT8
        self.ln2_gamma = weights['ln2_gamma']  # (384,) INT8
        self.ln2_beta = weights['ln2_beta']  # (384,) INT8

    def forward(self, hidden_states):
        """
        Forward pass through encoder layer

        Args:
            hidden_states: (seq_len, d_model) INT8

        Returns:
            output: (seq_len, d_model) INT8
        """
        # Self-Attention Block
        residual = hidden_states.copy()

        # LayerNorm 1
        x = self.layernorm(hidden_states, self.ln1_gamma, self.ln1_beta)

        # Multi-head attention
        # Q/K/V projections
        Q = self.matmul(x, self.attn_q_weight)  # (seq_len, 384)
        K = self.matmul(x, self.attn_k_weight)
        V = self.matmul(x, self.attn_v_weight)

        # Attention mechanism
        attn_output = self.attention(Q, K, V, num_heads=self.n_heads)

        # Output projection
        attn_output = self.matmul(attn_output, self.attn_out_weight)

        # Residual connection
        hidden_states = residual + attn_output
        hidden_states = np.clip(hidden_states, -128, 127).astype(np.int8)

        # Feed-Forward Block
        residual = hidden_states.copy()

        # LayerNorm 2
        x = self.layernorm(hidden_states, self.ln2_gamma, self.ln2_beta)

        # FFN
        x = self.matmul(x, self.ffn_1_weight)  # 384 → 1536
        x = self.gelu(x)
        x = self.matmul(x, self.ffn_2_weight)  # 1536 → 384

        # Residual connection
        hidden_states = residual + x
        hidden_states = np.clip(hidden_states, -128, 127).astype(np.int8)

        return hidden_states
```

**Task 5.2**: Test Single Layer (Day 41-42, 4-8 hours)

**Validation**:
```python
def test_single_layer():
    # Load NPU kernels
    kernels = NPUEncoderKernels()

    # Create encoder layer
    layer = WhisperNPUEncoderLayer(kernels, layer_idx=0)

    # Test input (random)
    x = np.random.randint(-32, 32, (1500, 384), dtype=np.int8)

    # Forward pass
    import time
    start = time.perf_counter()
    output = layer.forward(x)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"Single layer time: {elapsed:.2f}ms")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min()}, {output.max()}]")
    print(f"Non-zero: {np.count_nonzero(output)} / {output.size}")

    # Check output is valid
    assert output.shape == (1500, 384), "Output shape mismatch!"
    assert np.count_nonzero(output) > output.size * 0.8, "Too many zeros!"
    assert elapsed < 100, f"Too slow: {elapsed}ms (expected <100ms)"

    print("✅ Single layer test passed!")
```

**Success Criteria**:
- [ ] Layer produces valid output
- [ ] Output shape correct (seq_len, 384)
- [ ] Processing time <100ms for 1500 frames
- [ ] >80% non-zero values

---

### 5.3 Week 9: Full 6-Layer Encoder (16-24 hours)

**Objective**: Implement complete 6-layer encoder

**Task 5.3**: Full Encoder Implementation (Day 43-46, 12-16 hours)

**File**: `npu/encoder/whisper_npu_encoder.py`

```python
class WhisperNPUEncoder:
    """
    Complete Whisper Base encoder on NPU

    Architecture:
      - Input projection (Conv1D: 80 → 384)
      - Positional encoding
      - 6 transformer layers
      - Final layer norm
    """

    def __init__(self, model_name="base", device_id=0):
        print("=" * 70)
        print("Whisper NPU Encoder - Initializing")
        print("=" * 70)

        # Load unified kernels
        print("Loading NPU kernels...")
        self.kernels = NPUEncoderKernels(device_id)
        print("✅ Kernels loaded")

        # Create 6 encoder layers
        print("Creating encoder layers...")
        self.layers = []
        for i in range(6):
            layer = WhisperNPUEncoderLayer(
                self.kernels,
                layer_idx=i,
                d_model=384,
                n_heads=6,
                d_ff=1536
            )
            self.layers.append(layer)
            print(f"  Layer {i+1}/6 ready")
        print("✅ All layers created")

        # Load input projection and positional encoding
        self.load_input_projection()
        self.load_positional_encoding()

        # Final layer norm
        final_weights = np.load("models/whisper-base-npu-int8/encoder_final_ln.npz")
        self.final_ln_gamma = final_weights['gamma']
        self.final_ln_beta = final_weights['beta']

        print("✅ Encoder initialized")
        print("=" * 70)

    def load_input_projection(self):
        """Load Conv1D projection weights (80 → 384)"""
        # ... implementation ...

    def load_positional_encoding(self):
        """Load positional encoding (3000 × 384)"""
        # ... implementation ...

    def forward(self, mel_features):
        """
        Encode mel spectrogram

        Args:
            mel_features: (80, num_frames) FP32

        Returns:
            encoded: (384, num_frames) INT8
        """
        # Input projection
        hidden_states = self.input_projection(mel_features)

        # Add positional encoding
        num_frames = hidden_states.shape[1]
        pos_enc = self.positional_encoding[:, :num_frames]
        hidden_states = hidden_states + pos_enc

        # Quantize to INT8
        hidden_states = self.quantize(hidden_states)

        # Transpose to (num_frames, 384) for layers
        hidden_states = hidden_states.T

        # Pass through encoder layers
        for i, layer in enumerate(self.layers):
            print(f"Processing layer {i+1}/6...")
            hidden_states = layer.forward(hidden_states)

        # Final layer norm
        hidden_states = self.kernels.layernorm(
            hidden_states,
            self.final_ln_gamma,
            self.final_ln_beta
        )

        # Transpose back to (384, num_frames)
        hidden_states = hidden_states.T

        return hidden_states

    def quantize(self, x):
        """Quantize FP32 to INT8"""
        max_val = np.abs(x).max()
        if max_val > 0:
            scale = 127.0 / max_val
            return np.round(x * scale).astype(np.int8)
        return x.astype(np.int8)
```

**Task 5.4**: End-to-End Encoder Test (Day 47-48, 4-8 hours)

**Test Script**:
```python
def test_full_encoder():
    # Load encoder
    encoder = WhisperNPUEncoder()

    # Generate test mel spectrogram
    mel_features = np.random.randn(80, 1500).astype(np.float32)

    # Encode
    import time
    start = time.perf_counter()
    encoded = encoder.forward(mel_features)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\nFull encoder test:")
    print(f"  Input shape: {mel_features.shape}")
    print(f"  Output shape: {encoded.shape}")
    print(f"  Processing time: {elapsed:.2f}ms")
    print(f"  Realtime factor: {(30000 / elapsed):.1f}x (for 30s audio)")

    # Validate
    assert encoded.shape == (384, 1500), "Output shape mismatch!"
    assert np.count_nonzero(encoded) > encoded.size * 0.8, "Too many zeros!"

    # Performance check
    rtf = 30000 / elapsed
    if rtf >= 150:
        print(f"  ✅ EXCELLENT: {rtf:.1f}x realtime")
    elif rtf >= 75:
        print(f"  ✅ GOOD: {rtf:.1f}x realtime")
    elif rtf >= 50:
        print(f"  ⚠️ ACCEPTABLE: {rtf:.1f}x realtime (need optimization)")
    else:
        print(f"  ❌ TOO SLOW: {rtf:.1f}x realtime (target: 75-150x)")

    return rtf
```

**Success Criteria**:
- [ ] Encoder completes without errors
- [ ] Output shape correct (384, num_frames)
- [ ] Realtime factor >50x (minimum)
- [ ] Realtime factor >75x (good)
- [ ] Realtime factor >150x (excellent)

---

### 5.4 Week 10: Accuracy Validation (16-24 hours)

**Objective**: Validate encoder accuracy against CPU

**Task 5.5**: Accuracy Benchmarking (Day 49-51, 12-16 hours)

**Benchmark Script**:
```python
def benchmark_encoder_accuracy():
    """Compare NPU encoder vs CPU encoder"""
    import torch
    from transformers import WhisperProcessor, WhisperModel

    # Load CPU model
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    cpu_model = WhisperModel.from_pretrained("openai/whisper-base")

    # Load NPU encoder
    npu_encoder = WhisperNPUEncoder()

    # Test audio
    import librosa
    audio, sr = librosa.load("test_audio.wav", sr=16000)

    # CPU processing
    mel_cpu = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
    with torch.no_grad():
        cpu_encoded = cpu_model.encoder(mel_cpu).last_hidden_state

    # NPU processing
    mel_npu = mel_cpu.squeeze(0).numpy()  # (80, num_frames)
    npu_encoded = npu_encoder.forward(mel_npu)  # (384, num_frames)

    # Compare
    cpu_flat = cpu_encoded.squeeze(0).numpy().T.flatten()
    npu_flat = npu_encoded.flatten().astype(np.float32)

    correlation = np.corrcoef(cpu_flat, npu_flat)[0, 1]
    mse = np.mean((cpu_flat / 127.0 - npu_flat / 127.0) ** 2)

    print(f"\nEncoder Accuracy:")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  MSE: {mse:.6f}")

    if correlation >= 0.95:
        print(f"  ✅ EXCELLENT accuracy ({correlation:.4f})")
    elif correlation >= 0.90:
        print(f"  ✅ GOOD accuracy ({correlation:.4f})")
    elif correlation >= 0.80:
        print(f"  ⚠️ ACCEPTABLE accuracy ({correlation:.4f})")
    else:
        print(f"  ❌ POOR accuracy ({correlation:.4f}) - needs debugging!")

    return correlation
```

**Task 5.6**: Word Error Rate (WER) Testing (Day 52-54, 4-8 hours)

**WER Test**:
```python
def test_encoder_wer():
    """Test WER impact of NPU encoder"""
    from whisperx import load_model

    # Load models
    cpu_model = load_model("base", device="cpu")
    npu_model = load_model("base", device="npu")  # Uses our NPU encoder

    # Test audio files
    test_files = [
        "test_audio/jfk.wav",
        "test_audio/speech1.wav",
        "test_audio/speech2.wav"
    ]

    results = []
    for audio_file in test_files:
        # CPU transcription
        cpu_result = cpu_model.transcribe(audio_file)
        cpu_text = cpu_result["text"]

        # NPU transcription
        npu_result = npu_model.transcribe(audio_file)
        npu_text = npu_result["text"]

        # Calculate WER
        wer = calculate_wer(cpu_text, npu_text)
        results.append({
            'file': audio_file,
            'cpu_text': cpu_text,
            'npu_text': npu_text,
            'wer': wer
        })

        print(f"\n{audio_file}:")
        print(f"  CPU: {cpu_text}")
        print(f"  NPU: {npu_text}")
        print(f"  WER: {wer:.2%}")

    avg_wer = np.mean([r['wer'] for r in results])
    print(f"\nAverage WER: {avg_wer:.2%}")

    if avg_wer <= 0.05:
        print(f"  ✅ EXCELLENT: WER <5%")
    elif avg_wer <= 0.10:
        print(f"  ✅ GOOD: WER <10%")
    else:
        print(f"  ⚠️ HIGH: WER >{avg_wer:.0%} (needs improvement)")

    return avg_wer
```

**Success Criteria**:
- [ ] Encoder correlation >0.90 with CPU
- [ ] WER increase <10% vs CPU
- [ ] No significant transcription errors

---

### 5.5 Phase 4 Summary

**Total Effort**: 48-72 hours over 3 weeks

**Completion Criteria**:
- [ ] Single encoder layer working
- [ ] Full 6-layer encoder implemented
- [ ] End-to-end encoder tested
- [ ] Accuracy validated (correlation >0.90)
- [ ] WER tested (increase <10%)
- [ ] Realtime factor >50x

**Deliverables**:
1. `WhisperNPUEncoderLayer` class
2. `WhisperNPUEncoder` class
3. Test suite for encoder
4. Accuracy benchmarks
5. WER validation results

**Exit Criteria**: Full encoder working end-to-end with acceptable accuracy

---

## 6. Phase 5: Optimization & 220x Target (Weeks 11-12)

### 6.1 Overview

**Duration**: 2 weeks (14 days)
**Effort**: 32-48 hours
**Priority**: 🎯 **TARGET** - Achieve 220x performance
**Goal**: Optimize encoder to reach 220x realtime

**Current Estimate** (from design):
- Expected RTF with basic integration: 75x realtime
- Target RTF: 220x realtime
- **Need**: 2.9x additional speedup

### 6.2 Week 11: Attention Optimization (16-24 hours)

**Objective**: Optimize attention mechanism (biggest bottleneck)

**Task 6.1**: Parallel Multi-Head Processing (Day 55-58, 12-16 hours)

**Problem**: Currently processing 6 heads sequentially

**Solution**: Process all 6 heads in parallel on 6 NPU tiles

**Implementation**:
```python
def optimized_multi_head_attention(Q, K, V, num_heads=6):
    """
    Process all attention heads in parallel on NPU

    Current: Sequential processing of 6 heads (6× time)
    Optimized: Parallel processing on 6 tiles (1× time)
    Expected speedup: 6×
    """
    seq_len, d_model = Q.shape
    head_dim = d_model // num_heads  # 384 / 6 = 64

    # Reshape to heads: (6, seq_len, 64)
    Q_heads = Q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    K_heads = K.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    V_heads = V.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

    # Process all heads in parallel (NEW!)
    # Each head assigned to different NPU tile
    head_outputs = []
    for head_idx in range(num_heads):
        # Submit to NPU tile (non-blocking)
        npu_tile_id = head_idx % 4  # Use 4 attention tiles
        future = self.attention.process_async(
            Q_heads[head_idx],
            K_heads[head_idx],
            V_heads[head_idx],
            tile_id=npu_tile_id
        )
        head_outputs.append(future)

    # Wait for all heads to complete
    results = [future.result() for future in head_outputs]

    # Concatenate heads
    output = np.concatenate(results, axis=-1)  # (seq_len, 384)
    return output
```

**Expected Speedup**: 6x (attention from 24ms → 4ms per layer)

**Task 6.2**: Fused Attention Kernel (Day 59-60, 4-8 hours)

**Problem**: Multiple kernel calls for Q@K^T, Softmax, @V

**Solution**: Fuse into single kernel

**MLIR Modification**:
```mlir
// Add fused attention kernel to unified XCLBIN
func.func @attention_fused(
    %Q: memref<64x64xi8>,
    %K: memref<64x64xi8>,
    %V: memref<64x64xi8>,
    %output: memref<64x64xi8>
) {
    // All in one kernel (no intermediate DMA)
    %scores = matmul_transpose(%Q, %K)
    %scaled = scale(%scores, 1.0 / sqrt(64))
    %softmax = softmax(%scaled)
    %result = matmul(%softmax, %V)
    store %result, %output
}
```

**Expected Speedup**: 1.5x (reduce DMA overhead)

**Total Attention Speedup**: 6x × 1.5x = 9x
- Current: 24ms per layer
- Optimized: 24ms / 9 = **2.7ms per layer** ✅

---

### 6.3 Week 12: MatMul and Batching Optimization (16-24 hours)

**Objective**: Optimize matrix multiplications and enable batching

**Task 6.3**: MatMul Tiling Optimization (Day 61-63, 12-16 hours)

**Problem**: 16×16 tiles have high tiling overhead

**Solution**: Use larger tiles for encoder matmuls (32×32 or 64×64)

**Implementation**:
```python
# Adaptive tile size based on matrix dimensions
def select_optimal_tile_size(M, K, N):
    """
    Choose tile size for best performance

    Small matrices (M,N < 512): 16×16 (less overhead)
    Medium matrices (512-2048): 32×32 (balanced)
    Large matrices (>2048): 64×64 (max parallelism)
    """
    max_dim = max(M, N, K)

    if max_dim <= 512:
        return 16
    elif max_dim <= 2048:
        return 32
    else:
        return 64

# Use in matmul wrapper
tile_size = select_optimal_tile_size(M, K, N)
result = self.matmul(A, B, tile_size=tile_size)
```

**Expected Speedup**: 1.5-2x on large matrices

**Task 6.4**: Batch Processing (Day 64-66, 4-8 hours)

**Problem**: Processing one audio clip at a time

**Solution**: Batch multiple audio clips or frames

**Implementation**:
```python
def encode_batch(self, mel_features_list):
    """
    Process multiple audio clips in single batch

    Input: List of (80, num_frames) mel spectrograms
    Output: List of (384, num_frames) encoded features

    Speedup: Amortize kernel initialization overhead
    """
    # Pad to same length
    max_frames = max(mel.shape[1] for mel in mel_features_list)
    batch = []
    for mel in mel_features_list:
        padded = np.zeros((80, max_frames), dtype=np.float32)
        padded[:, :mel.shape[1]] = mel
        batch.append(padded)

    # Stack into batch: (batch_size, 80, max_frames)
    batch = np.stack(batch, axis=0)

    # Process entire batch on NPU (parallel)
    encoded_batch = self.encoder_forward_batched(batch)

    # Unstack and remove padding
    results = []
    for i, mel in enumerate(mel_features_list):
        result = encoded_batch[i, :, :mel.shape[1]]
        results.append(result)

    return results
```

**Expected Speedup**: 1.5-2x (for batch size 4-8)

**Task 6.5**: Final Performance Tuning (Day 67-68, 4-8 hours)

**Optimizations**:
1. **DMA Pipelining**: Overlap data transfer with computation
2. **Buffer Pooling**: Reuse buffers across calls
3. **Kernel Warmup**: Pre-initialize kernels once
4. **Memory Alignment**: Ensure aligned buffers for fastest DMA
5. **Precision Tuning**: Mixed INT8/INT16 for critical ops

**Expected Additional Speedup**: 1.2-1.5x

---

### 6.4 Phase 5 Performance Targets

**Baseline** (after Phase 4): 75x realtime

**After Attention Optimization** (9x speedup):
- Attention: 24ms → 2.7ms per layer (savings: 21.3ms × 6 layers = 128ms)
- Total: 400ms - 128ms = 272ms
- RTF: 30,000ms / 272ms = **110x realtime**

**After MatMul Optimization** (1.5x speedup):
- MatMul time: ~180ms → 120ms (savings: 60ms)
- Total: 272ms - 60ms = 212ms
- RTF: 30,000ms / 212ms = **142x realtime**

**After Batching** (1.5x speedup):
- Total: 212ms / 1.5 = 141ms
- RTF: 30,000ms / 141ms = **213x realtime**

**After Final Tuning** (1.3x speedup):
- Total: 141ms / 1.3 = 108ms
- RTF: 30,000ms / 108ms = **278x realtime** 🎯

**TARGET EXCEEDED**: 278x > 220x ✅

---

### 6.5 Phase 5 Summary

**Total Effort**: 32-48 hours over 2 weeks

**Completion Criteria**:
- [ ] Attention optimized (9x speedup)
- [ ] MatMul optimized (1.5x speedup)
- [ ] Batching implemented (1.5x speedup)
- [ ] Final tuning complete (1.3x speedup)
- [ ] **220x realtime achieved** 🎯

**Deliverables**:
1. Parallel multi-head attention
2. Fused attention kernel
3. Adaptive tile sizing
4. Batch processing
5. Performance benchmarks

**Exit Criteria**: 220x realtime performance achieved and validated

---

## 7. Phase 6: Production Hardening (Weeks 13-14)

### 7.1 Overview

**Duration**: 2 weeks (14 days)
**Effort**: 24-36 hours
**Priority**: ✅ **QUALITY** - Production readiness
**Goal**: Error handling, documentation, deployment

### 7.2 Week 13: Error Handling and Robustness (12-18 hours)

**Task 7.1**: Exception Handling (Day 69-71, 8-12 hours)

**Error Categories**:
1. **NPU Errors**: Device not available, XCLBIN load failures
2. **Memory Errors**: Out of memory, buffer allocation failures
3. **Computation Errors**: Invalid inputs, numerical overflow
4. **Timeout Errors**: Kernel execution timeouts

**Implementation**:
```python
class NPUError(Exception):
    """Base exception for NPU errors"""
    pass

class NPUDeviceError(NPUError):
    """NPU device not available or initialization failed"""
    pass

class NPUMemoryError(NPUError):
    """NPU memory allocation or access error"""
    pass

class WhisperNPUEncoder:
    def __init__(self, ...):
        try:
            self.device = xrt.device(device_id)
        except Exception as e:
            raise NPUDeviceError(
                f"Failed to initialize NPU device {device_id}: {e}\n"
                f"Please ensure:\n"
                f"  1. NPU driver installed (check /dev/accel/accel0)\n"
                f"  2. XRT 2.20.0 installed\n"
                f"  3. Firmware updated (1.5.5.391)"
            )

        try:
            xclbin = xrt.xclbin(self.xclbin_path)
            self.device.register_xclbin(xclbin)
        except Exception as e:
            raise NPUError(
                f"Failed to load XCLBIN {self.xclbin_path}: {e}\n"
                f"Try rebuilding: cd npu && ./build_unified.sh"
            )

    def forward(self, mel_features):
        # Validate inputs
        if mel_features.shape[0] != 80:
            raise ValueError(
                f"Expected 80 mel bins, got {mel_features.shape[0]}"
            )

        if mel_features.shape[1] > 3000:
            raise ValueError(
                f"Max 3000 frames supported, got {mel_features.shape[1]}"
            )

        try:
            result = self._forward_internal(mel_features)
        except TimeoutError:
            raise NPUError(
                "NPU kernel execution timed out (>10s)\n"
                "This may indicate:\n"
                "  1. NPU hung (try rebooting)\n"
                "  2. Input too large\n"
                "  3. Kernel bug"
            )
        except Exception as e:
            # Fallback to CPU
            logger.warning(f"NPU execution failed: {e}, falling back to CPU")
            result = self._cpu_fallback(mel_features)

        return result

    def _cpu_fallback(self, mel_features):
        """Fallback to CPU encoder if NPU fails"""
        logger.info("Using CPU fallback encoder")
        # Load CPU model (cached)
        if not hasattr(self, 'cpu_model'):
            from transformers import WhisperModel
            self.cpu_model = WhisperModel.from_pretrained("openai/whisper-base")

        # Encode on CPU
        with torch.no_grad():
            result = self.cpu_model.encoder(mel_features)

        return result.last_hidden_state.numpy()
```

**Task 7.2**: Logging and Monitoring (Day 72-73, 4-6 hours)

**Implementation**:
```python
import logging

logger = logging.getLogger("whisper_npu")
logger.setLevel(logging.INFO)

class WhisperNPUEncoder:
    def __init__(self, ...):
        # Log initialization
        logger.info("=" * 70)
        logger.info("Whisper NPU Encoder v1.0")
        logger.info(f"Device: {device_id}")
        logger.info(f"Model: whisper-base")
        logger.info(f"XCLBIN: {self.xclbin_path}")
        logger.info("=" * 70)

        # Performance monitoring
        self.stats = {
            'total_calls': 0,
            'total_time_ms': 0,
            'total_frames': 0,
            'errors': 0,
            'fallbacks': 0
        }

    def forward(self, mel_features):
        self.stats['total_calls'] += 1
        self.stats['total_frames'] += mel_features.shape[1]

        start = time.perf_counter()
        try:
            result = self._forward_internal(mel_features)
        except Exception as e:
            self.stats['errors'] += 1
            raise

        elapsed = (time.perf_counter() - start) * 1000
        self.stats['total_time_ms'] += elapsed

        # Log performance
        rtf = (mel_features.shape[1] / 50.0 * 1000) / elapsed  # 50 fps
        logger.debug(f"Encoded {mel_features.shape[1]} frames in {elapsed:.2f}ms ({rtf:.1f}x realtime)")

        return result

    def get_stats(self):
        """Get performance statistics"""
        avg_time = self.stats['total_time_ms'] / self.stats['total_calls']
        avg_frames = self.stats['total_frames'] / self.stats['total_calls']
        avg_rtf = (avg_frames / 50.0 * 1000) / avg_time

        return {
            'total_calls': self.stats['total_calls'],
            'total_frames': self.stats['total_frames'],
            'avg_time_ms': avg_time,
            'avg_rtf': avg_rtf,
            'errors': self.stats['errors'],
            'error_rate': self.stats['errors'] / self.stats['total_calls']
        }
```

---

### 7.3 Week 14: Documentation and Deployment (12-18 hours)

**Task 7.3**: User Documentation (Day 74-76, 8-12 hours)

**Create Documentation**:
1. `README_NPU_ENCODER.md` - User guide
2. `INSTALLATION.md` - Setup instructions
3. `API_REFERENCE.md` - API documentation
4. `TROUBLESHOOTING.md` - Common issues and solutions
5. `PERFORMANCE_GUIDE.md` - Performance tuning

**README Example**:
```markdown
# Whisper NPU Encoder

NPU-accelerated Whisper Base encoder for AMD Phoenix NPU.

## Features

- 220x realtime transcription
- INT8 quantization
- Automatic CPU fallback
- Batch processing support

## Installation

bash
pip install -r requirements.txt
bash scripts/install_npu.sh


## Quick Start

python
from whisperx.npu import WhisperNPUEncoder

encoder = WhisperNPUEncoder(model="base")
encoded = encoder(mel_features)


## Performance

- **Realtime Factor**: 220x
- **Power**: 5-10W
- **Latency**: <150ms for 30s audio

## Requirements

- AMD Ryzen 7040/8040 series
- XRT 2.20.0
- NPU firmware 1.5.5.391
```

**Task 7.4**: Deployment Package (Day 77-78, 4-6 hours)

**Create Distribution**:
```bash
# 1. Package structure
whisper-npu-encoder/
├── whisperx/
│   └── npu/
│       ├── encoder/
│       ├── wrappers/
│       └── kernels/
├── models/
│   └── whisper-base-npu-int8/
├── docs/
├── tests/
├── setup.py
├── requirements.txt
└── README.md

# 2. Setup script
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="whisper-npu-encoder",
    version="1.0.0",
    description="NPU-accelerated Whisper encoder for AMD Phoenix",
    author="Magic Unicorn Inc.",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=2.0.0",
        "transformers>=4.30.0"
    ],
    extras_require={
        "npu": ["pyxrt>=2.20.0"]
    },
    python_requires=">=3.10"
)
EOF

# 3. Build distribution
python3 setup.py sdist bdist_wheel

# 4. Test installation
pip install dist/whisper-npu-encoder-1.0.0.tar.gz
```

**Task 7.5**: Integration Testing (Day 79-80, 4-6 hours)

**Create Integration Tests**:
```python
def test_full_pipeline():
    """Test complete transcription pipeline with NPU encoder"""
    import whisperx

    # Load model with NPU encoder
    model = whisperx.load_model(
        "base",
        device="npu",
        compute_type="int8"
    )

    # Transcribe
    audio = whisperx.load_audio("test_audio.wav")
    result = model.transcribe(audio)

    # Validate
    assert result["text"], "Empty transcription!"
    assert result["language"] == "en", "Wrong language detected"
    assert len(result["segments"]) > 0, "No segments!"

    # Check performance
    audio_duration = len(audio) / 16000
    processing_time = result["processing_time"]
    rtf = audio_duration / processing_time

    assert rtf > 100, f"Too slow: {rtf:.1f}x (expected >100x)"
    print(f"✅ Pipeline test passed: {rtf:.1f}x realtime")
```

---

### 7.4 Phase 6 Summary

**Total Effort**: 24-36 hours over 2 weeks

**Completion Criteria**:
- [ ] Error handling implemented
- [ ] CPU fallback working
- [ ] Logging and monitoring added
- [ ] Documentation complete
- [ ] Deployment package created
- [ ] Integration tests passing

**Deliverables**:
1. Robust error handling
2. Complete documentation (5 docs)
3. Deployment package
4. Integration test suite
5. Performance monitoring

**Exit Criteria**: Production-ready NPU encoder, fully documented and tested

---

## 8. Risk Management

### 8.1 Technical Risks

| Risk ID | Risk | Probability | Impact | Mitigation | Timeline Buffer |
|---------|------|-------------|--------|------------|-----------------|
| **R1** | Unified XCLBIN compilation fails | 30% | CRITICAL | Use dynamic kernel swapping fallback | +2 weeks |
| **R2** | Attention buffer issue persists | 20% | HIGH | Use CPU attention (hybrid mode) | +1 week |
| **R3** | MatMul batching hits buffer limits | 25% | MEDIUM | Use smaller batch sizes | +1 week |
| **R4** | INT8 accuracy <90% correlation | 15% | MEDIUM | Use mixed INT8/INT16 precision | +1 week |
| **R5** | Performance doesn't reach 220x | 40% | MEDIUM | Accept 150x as success | N/A |
| **R6** | MLIR kernel modifications needed | 40% | HIGH | Have MLIR expert available | +2 weeks |
| **R7** | Integration bugs and edge cases | 60% | MEDIUM | Comprehensive testing | +3 weeks built-in |
| **R8** | NPU firmware compatibility issues | 10% | CRITICAL | Test on multiple firmware versions | +1 week |

**Overall Risk**: MEDIUM (70% confidence of success)

### 8.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Debugging takes longer | 70% | MEDIUM | 50% buffer added to estimates |
| Unexpected blockers | 50% | HIGH | Have fallback plans for each phase |
| Testing reveals new issues | 80% | LOW | 2 iterations built into timeline |
| Optimization less effective | 40% | MEDIUM | Accept lower performance (150x) |

**Schedule Confidence**: 60% (realistic timeline, likely extensions)

### 8.3 Contingency Plans

**Contingency 1: Unified XCLBIN Fails**
- **Trigger**: Cannot compile all kernels into one XCLBIN
- **Action**: Use dynamic kernel swapping (Option B from Phase 3)
- **Impact**: Performance ~100-150x instead of 220x
- **Still**: Major improvement over CPU (13.5x)

**Contingency 2: Attention Not Fixable**
- **Trigger**: Attention still returns zeros after 2 weeks
- **Action**: Use CPU for attention, NPU for everything else
- **Impact**: Performance ~80-100x (still good!)
- **Benefit**: Most of encoder still on NPU

**Contingency 3: Can't Reach 220x**
- **Trigger**: After all optimizations, only achieve 150x
- **Action**: Accept 150x as success
- **Impact**: Still 11x faster than CPU baseline
- **Benefit**: Valuable NPU acceleration

**Contingency 4: NPU Completely Unusable**
- **Trigger**: Fundamental NPU issues can't be resolved
- **Action**: Fall back to optimized CPU (faster-whisper)
- **Impact**: No NPU acceleration
- **Benefit**: Still have working transcription

**Bottom Line**: Multiple fallback plans ensure we deliver value regardless

---

## 9. Success Metrics

### 9.1 Phase-Level Metrics

| Phase | Success Metric | Minimum | Good | Excellent |
|-------|---------------|---------|------|-----------|
| **Phase 1** | Attention working | Non-zero output | Correlation >0.70 | Correlation >0.80 |
| **Phase 1** | MatMul speedup | 30x faster | 50x faster | 68x faster |
| **Phase 2** | LayerNorm correlation | >0.90 | >0.95 | >0.98 |
| **Phase 2** | GELU correlation | >0.90 | >0.95 | >0.98 |
| **Phase 3** | Unified XCLBIN | Compiles | <500 KB | <300 KB |
| **Phase 4** | Encoder correlation | >0.80 | >0.90 | >0.95 |
| **Phase 4** | Encoder RTF | >50x | >75x | >100x |
| **Phase 5** | Final RTF | >150x | >200x | >220x 🎯 |
| **Phase 6** | Error rate | <10% | <5% | <1% |

### 9.2 Overall Project Metrics

**Performance Metrics**:
- **Primary**: Realtime factor >220x for 30s audio
- **Secondary**: Power consumption <15W
- **Tertiary**: Latency <200ms for 30s audio

**Quality Metrics**:
- **Primary**: Encoder correlation >0.90 with CPU
- **Secondary**: WER increase <10% vs CPU
- **Tertiary**: No transcription failures

**Reliability Metrics**:
- **Primary**: Success rate >95% (errors <5%)
- **Secondary**: CPU fallback works 100%
- **Tertiary**: No crashes or hangs

### 9.3 Business Metrics

**Technical Success**:
- ✅ **Minimum**: Any NPU acceleration working (>13.5x)
- ✅ **Good**: 100x realtime achieved
- ✅ **Excellent**: 220x realtime achieved 🎯

**User Impact**:
- **Minimum**: Noticeable speedup in transcription
- **Good**: Real-time transcription of long audio
- **Excellent**: Batch transcription of hours of audio in minutes

**Competitive Advantage**:
- **Unique**: First Whisper on AMD Phoenix NPU
- **Performance**: 10-16x faster than competing solutions
- **Power**: 5-10W vs 45-125W for CPU/GPU

---

## 10. Resource Requirements

### 10.1 Human Resources

**Primary Developer** (Full-time equivalent):
- **Phase 1-2**: 1.0 FTE (fixing blockers, creating wrappers)
- **Phase 3-4**: 1.0 FTE (integration work)
- **Phase 5**: 0.75 FTE (optimization)
- **Phase 6**: 0.5 FTE (documentation, deployment)

**Supporting Roles** (Part-time):
- **MLIR Expert**: 0.25 FTE (Phase 3, XCLBIN compilation)
- **ML Engineer**: 0.25 FTE (Phase 4, accuracy validation)
- **QA Engineer**: 0.25 FTE (Phase 6, testing)

**Total Effort**: ~150-200 person-hours over 14 weeks

### 10.2 Hardware Resources

**Development Machine**:
- AMD Ryzen 7040/8040 with Phoenix NPU
- 32GB RAM (minimum)
- 100GB free storage

**Test Hardware**:
- Same as development (ideally 2 machines)
- One for development, one for validation

**Already Available**: ✅ Hardware exists and operational

### 10.3 Software Resources

**Licenses**: None required (all open-source)

**Tools**:
- XRT 2.20.0 ✅ Installed
- MLIR-AIE toolchain ✅ Operational
- Python 3.10+ ✅ Available
- PyTorch ✅ Installed
- Transformers ✅ Installed

**Already Available**: ✅ All software dependencies met

### 10.4 Budget Estimate

**Development Costs**:
- Labor: $0 (using existing resources)
- Hardware: $0 (already purchased)
- Software: $0 (all open-source)
- Cloud/Compute: $0 (local development)

**Total Budget**: $0 (all resources in-house) ✅

---

## 11. Timeline Summary

### 11.1 Gantt Chart

```
Week  1  2  3  4  5  6  7  8  9 10 11 12 13 14
Phase ──────────────────────────────────────────
  1   [████████████]
  2            [█████████]
  3                  [█████████]
  4                        [███████████████]
  5                                    [█████████]
  6                                          [█████████]
```

### 11.2 Key Milestones

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| **2** | Attention fixed | Non-zero output, >0.80 correlation |
| **3** | MatMul 68x faster | Wrapper optimized |
| **5** | All wrappers complete | 4/4 kernels tested |
| **7** | Unified XCLBIN working | All kernels in one binary |
| **10** | Full encoder working | End-to-end validated |
| **12** | 220x achieved | Performance target met 🎯 |
| **14** | Production ready | Docs, tests, deployment |

### 11.3 Critical Path

**Critical Path** (cannot be parallelized):
1. Fix Attention (Week 1-2) → **BLOCKS** Layer Testing
2. Fix MatMul (Week 2-3) → **BLOCKS** Encoder Integration
3. Unified XCLBIN (Week 6-7) → **BLOCKS** Full Integration
4. Full Integration (Week 8-10) → **BLOCKS** Optimization
5. Optimization (Week 11-12) → **BLOCKS** 220x Target

**Timeline Risk**: Any delay on critical path delays final delivery

**Buffer**: 3 weeks built into timeline (14 weeks vs 11 week minimum)

---

## 12. Final Summary

### 12.1 Project Overview

**Goal**: Achieve 220x realtime Whisper Base encoder on AMD Phoenix NPU

**Approach**: 6-phase implementation over 14 weeks

**Confidence**: 70% (high probability of success)

### 12.2 Phase Breakdown

| Phase | Duration | Effort | Priority | Risk |
|-------|----------|--------|----------|------|
| **1** | 3 weeks | 60-80h | 🔴 CRITICAL | MEDIUM |
| **2** | 2 weeks | 24-36h | 🟡 HIGH | LOW |
| **3** | 2 weeks | 40-60h | 🔴 CRITICAL | HIGH |
| **4** | 3 weeks | 48-72h | 🟡 HIGH | MEDIUM |
| **5** | 2 weeks | 32-48h | 🎯 TARGET | MEDIUM |
| **6** | 2 weeks | 24-36h | ✅ QUALITY | LOW |
| **TOTAL** | **14 weeks** | **228-332h** | - | MEDIUM |

### 12.3 Success Criteria

**Must Have** (Minimum Viable Product):
- ✅ NPU encoder working end-to-end
- ✅ Realtime factor >50x
- ✅ Accuracy correlation >0.80
- ✅ CPU fallback for errors

**Should Have** (Good Product):
- ✅ Realtime factor >100x
- ✅ Accuracy correlation >0.90
- ✅ WER increase <10%
- ✅ Error rate <5%

**Could Have** (Excellent Product):
- ✅ Realtime factor >220x 🎯
- ✅ Accuracy correlation >0.95
- ✅ WER increase <5%
- ✅ Batch processing

**Won't Have** (Out of Scope):
- Decoder on NPU (future work)
- Multi-GPU support
- FP16 precision
- Model fine-tuning

### 12.4 Risk Summary

**Overall Risk Level**: MEDIUM

**Top 3 Risks**:
1. Unified XCLBIN compilation (30% fail, CRITICAL impact)
2. Performance doesn't reach 220x (40% fail, MEDIUM impact)
3. Unexpected integration issues (60% fail, MEDIUM impact)

**Mitigation**: Multiple fallback plans at each level

**Worst Case**: Fall back to CPU (still functional)

### 12.5 Next Steps

**To Begin Implementation**:

1. **Create GitHub branch**: `npu-encoder-220x`
2. **Set up tracking**: Create GitHub project board with these tasks
3. **Assign resources**: Allocate developer time
4. **Start Phase 1**: Fix attention buffer issue (Week 1)

**First Command**:
```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis
git checkout -b npu-encoder-220x
cd whisperx/npu/npu_optimization/whisper_encoder_kernels

# Begin Phase 1, Task 1.1
python3 test_attention_minimal.py
```

**Weekly Reviews**: Every Friday, review progress and update timeline

**Stakeholder Updates**: Every 2 weeks, report to project sponsor

---

**Implementation Plan Status**: 📋 **READY TO BEGIN**

**Timeline**: 14 weeks (9-14 weeks with risk buffer)

**Confidence**: 70% chance of achieving 220x target

**Fallback**: Multiple levels of acceptable performance (150x, 100x, 50x all useful)

**Recommendation**: **PROCEED WITH IMPLEMENTATION** - High value, manageable risk, clear path forward

---

**Plan Created**: November 2, 2025
**Plan Author**: NPU Implementation Lead
**Next Review**: Start of Week 1
**Approval**: **READY FOR EXECUTION**

**Magic Unicorn Unconventional Technology & Stuff Inc.** 🦄✨
