# Phase 3 Implementation Plan: Encoder BFP16 Integration

**Project**: Unicorn-Amanuensis XDNA2
**Date**: October 30, 2025
**Status**: Ready to start (Phase 2 complete)
**Estimated Duration**: 8-12 hours

---

## Executive Summary

Phase 3 will integrate the BFP16 quantization system (completed in Phase 2) into the full 6-layer encoder. This phase focuses on **replacing INT8 with BFP16** across all 6 matmul call sites in each layer, updating weight loading, and ensuring memory buffers are correctly sized.

### Goals
- **Replace 6 matmul calls per layer** (36 total) with BFP16 equivalents
- **Update weight loading** to convert FP32 → BFP16 and store in new format
- **Update memory buffers** to accommodate BFP16 storage (1.125× overhead)
- **Validate accuracy** >99% vs PyTorch reference
- **Maintain performance** 18-20× realtime target

---

## Current State Analysis

### Phase 2 Deliverables (Available)
- ✅ `BFP16Quantizer` class fully implemented
- ✅ `prepare_for_npu()` and `read_from_npu()` high-level API
- ✅ Conversion, shuffle, unshuffle operations working
- ✅ Unit tests passing (conversion accuracy validated)

### Phase 3 Starting Point
- ❌ Encoder still uses INT8 buffers and scales
- ❌ 6 matmul call sites use old `run_npu_linear()` signature
- ❌ Weight loading uses `Quantizer` (INT8)
- ❌ Memory buffers sized for INT8 (8 bits per value)

---

## Detailed Implementation Plan

### Task 1: Update encoder_layer.hpp (2-3 hours)

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/include/encoder_layer.hpp`

#### 1.1 Update Includes (5 minutes)

**Location**: Lines 1-8

**Changes**:
```cpp
// BEFORE:
#include "quantization.hpp"

// AFTER:
#include "bfp16_quantization.hpp"  // BFP16 quantizer
#include "quantization.hpp"         // Keep for legacy support (optional)
```

#### 1.2 Update Weight Storage Members (30 minutes)

**Location**: Lines 157-170

**BEFORE** (INT8 + scales):
```cpp
// Quantized weights (INT8) and scales
Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> q_weight_int8_;
Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> k_weight_int8_;
Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> v_weight_int8_;
Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> out_weight_int8_;
Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> fc1_weight_int8_;
Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> fc2_weight_int8_;

float q_weight_scale_;
float k_weight_scale_;
float v_weight_scale_;
float out_weight_scale_;
float fc1_weight_scale_;
float fc2_weight_scale_;
```

**AFTER** (BFP16, no separate scales):
```cpp
// BFP16 weights (shuffled, ready for NPU)
// Format: 8 mantissas (8 bits each) + 1 shared exponent (8 bits) per block
// Storage overhead: 1.125× (9 bytes per 8 values)
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> q_weight_bfp16_;
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> k_weight_bfp16_;
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> v_weight_bfp16_;
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> out_weight_bfp16_;
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> fc1_weight_bfp16_;
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> fc2_weight_bfp16_;

// NOTE: Scales are now embedded in BFP16 block exponents
// Each block has its own shared exponent (stored in 9th byte)
```

**Memory Impact**:
- **INT8**: 512×512 = 262,144 bytes + 4 bytes scale = 262,148 bytes per weight
- **BFP16**: 512×576 (1.125×) = 294,912 bytes (includes exponents)
- **Overhead**: +12.5% per weight matrix

#### 1.3 Update Activation Buffer Types (20 minutes)

**Location**: Lines 193-195

**BEFORE** (INT8):
```cpp
// Quantized buffers
Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> input_int8_;
Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> matmul_output_int32_;
```

**AFTER** (BFP16):
```cpp
// BFP16 buffers (temporary, reused across matmuls)
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> input_bfp16_;          // Input before shuffle
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> input_bfp16_shuffled_; // Input after shuffle (NPU layout)
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> output_bfp16_shuffled_; // Output from NPU (shuffled)
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> output_bfp16_;         // Output after unshuffle

// NOTE: NPU outputs BFP16, not INT32 like INT8 mode
```

**Memory Impact**:
- **Worst case** (FC1 output): 512×2048 values → 512×2304 bytes = 1,179,648 bytes
- **Typical** (512×512): 294,912 bytes
- **Total per layer**: ~2.5 MB (vs 3.1 MB for INT8)

#### 1.4 Update run_npu_linear() Signature (15 minutes)

**Location**: Lines 206-212

**BEFORE**:
```cpp
void run_npu_linear(
    const Eigen::MatrixXf& input,
    const Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>& weight_int8,
    float weight_scale,
    const Eigen::VectorXf& bias,
    Eigen::MatrixXf& output
);
```

**AFTER**:
```cpp
void run_npu_linear(
    const Eigen::MatrixXf& input,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& weight_bfp16,
    const Eigen::VectorXf& bias,
    Eigen::MatrixXf& output
);
```

**Key Changes**:
- Remove `float weight_scale` parameter (embedded in BFP16 blocks)
- Change weight type: `int8_t` → `uint8_t`
- Same FP32 input/output interface (quantization happens inside)

#### 1.5 Add BFP16 Quantizer Instance (5 minutes)

**Location**: After line 148 (inside private section)

**Add**:
```cpp
// BFP16 quantizer instance
BFP16Quantizer bfp16_quantizer_;
```

#### Verification Checklist
- [ ] All INT8 weight members replaced with BFP16 equivalents
- [ ] All scale members removed (6 floats)
- [ ] Activation buffers updated to BFP16
- [ ] `run_npu_linear()` signature updated
- [ ] BFP16Quantizer instance added
- [ ] File compiles without errors

---

### Task 2: Update load_weights() (2-3 hours)

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/src/encoder_layer.cpp`

**Location**: Lines 22-74

#### 2.1 Replace INT8 Quantization with BFP16 Conversion

**BEFORE** (per weight, 6× total):
```cpp
Quantizer quantizer;

q_weight_scale_ = quantizer.compute_scale(q_weight);
quantizer.quantize_tensor(q_weight, q_weight_int8_, q_weight_scale_);

k_weight_scale_ = quantizer.compute_scale(k_weight);
quantizer.quantize_tensor(k_weight, k_weight_int8_, k_weight_scale_);

// ... (repeat for 4 more weights)
```

**AFTER** (simplified, no scales):
```cpp
// Convert all weights to BFP16 (with shuffle for NPU)
bfp16_quantizer_.prepare_for_npu(q_weight, q_weight_bfp16_);
bfp16_quantizer_.prepare_for_npu(k_weight, k_weight_bfp16_);
bfp16_quantizer_.prepare_for_npu(v_weight, v_weight_bfp16_);
bfp16_quantizer_.prepare_for_npu(out_weight, out_weight_bfp16_);
bfp16_quantizer_.prepare_for_npu(fc1_weight, fc1_weight_bfp16_);
bfp16_quantizer_.prepare_for_npu(fc2_weight, fc2_weight_bfp16_);
```

**Changes**:
1. Remove `Quantizer quantizer;` instance
2. Remove 6 scale computation calls
3. Remove 6 `quantize_tensor()` calls
4. Add 6 `prepare_for_npu()` calls
5. Update variable names (`_int8` → `_bfp16`)

**Result**:
- **Code reduction**: 12 lines → 6 lines
- **Simplification**: No explicit scale management
- **Memory overhead**: +12.5% per weight (acceptable)

#### 2.2 Keep Biases and LayerNorm Parameters (no changes)

**Lines 61-73** remain unchanged (FP32 storage):
```cpp
// Store biases (FP32) - NO CHANGES
q_bias_ = q_bias;
k_bias_ = k_bias;
v_bias_ = v_bias;
out_bias_ = out_bias;
fc1_bias_ = fc1_bias;
fc2_bias_ = fc2_bias;

// Store layer norm parameters (FP32) - NO CHANGES
attn_ln_weight_ = attn_ln_weight;
attn_ln_bias_ = attn_ln_bias;
ffn_ln_weight_ = ffn_ln_weight;
ffn_ln_bias_ = ffn_ln_bias;
```

#### Verification Checklist
- [ ] All 6 weights converted using `prepare_for_npu()`
- [ ] No scale computation or storage
- [ ] Biases remain FP32
- [ ] LayerNorm params remain FP32
- [ ] Function compiles and links

---

### Task 3: Update run_npu_linear() (3-4 hours)

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/src/encoder_layer.cpp`

**Location**: Lines 163-223

#### 3.1 Update Function Signature

**BEFORE**:
```cpp
void EncoderLayer::run_npu_linear(
    const Eigen::MatrixXf& input,
    const Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>& weight_int8,
    float weight_scale,
    const Eigen::VectorXf& bias,
    Eigen::MatrixXf& output
) {
```

**AFTER**:
```cpp
void EncoderLayer::run_npu_linear(
    const Eigen::MatrixXf& input,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& weight_bfp16,
    const Eigen::VectorXf& bias,
    Eigen::MatrixXf& output
) {
```

#### 3.2 Rewrite Function Body (Complete Replacement)

**OLD LOGIC** (INT8 path):
```cpp
// [1] Quantize input (FP32 → INT8)
Quantizer quantizer;
float input_scale = quantizer.compute_scale(input);
quantizer.quantize_tensor(input, input_int8_, input_scale);

// [2] NPU matmul (INT8 @ INT8 → INT32)
npu_callback_fn_(..., input_int8_, weight_int8, matmul_output_int32_, ...);

// [3] Dequantize output (INT32 → FP32)
quantizer.dequantize_matmul_output(matmul_output_int32_, output, input_scale, weight_scale);

// [4] Add bias
output += bias;
```

**NEW LOGIC** (BFP16 path):
```cpp
void EncoderLayer::run_npu_linear(
    const Eigen::MatrixXf& input,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& weight_bfp16,
    const Eigen::VectorXf& bias,
    Eigen::MatrixXf& output
) {
    const int M = input.rows();
    const int K = input.cols();
    const int N = weight_bfp16.rows();  // Weight is (N, K_bfp16) in BFP16 bytes

    // [1] Convert input FP32 → BFP16 and shuffle for NPU
    bfp16_quantizer_.prepare_for_npu(input, input_bfp16_shuffled_);

    // [2] Calculate BFP16 output buffer size
    // N values = N/8 blocks = (N/8) × 9 bytes
    const int N_blocks = (N + BFP16Config::BLOCK_SIZE - 1) / BFP16Config::BLOCK_SIZE;
    const int N_bfp16_bytes = N_blocks * BFP16Config::BYTES_PER_BLOCK;

    // Resize output buffer if needed
    if (output_bfp16_shuffled_.rows() != M ||
        output_bfp16_shuffled_.cols() != N_bfp16_bytes) {
        output_bfp16_shuffled_.resize(M, N_bfp16_bytes);
    }

    // [3] Run NPU matmul: BFP16 @ BFP16 → BFP16
    if (npu_callback_fn_) {
        // NPU path (via Python callback)
        typedef int (*NPUCallbackBFP16)(
            void*,          // user_data
            const uint8_t*, // A (M × K_bfp16)
            const uint8_t*, // B (N × K_bfp16)
            uint8_t*,       // C (M × N_bfp16)
            size_t,         // M
            size_t,         // K (original dimension)
            size_t          // N (original dimension)
        );
        auto callback = reinterpret_cast<NPUCallbackBFP16>(npu_callback_fn_);

        int result = callback(
            npu_user_data_,
            input_bfp16_shuffled_.data(),
            weight_bfp16.data(),
            output_bfp16_shuffled_.data(),
            M, K, N
        );

        if (result != 0) {
            throw std::runtime_error("NPU callback failed (BFP16)");
        }
    } else if (npu_matmul_fn_) {
        // C++ std::function path (for testing)
        // NOTE: npu_matmul_fn_ signature needs update to support BFP16
        throw std::runtime_error("BFP16 std::function path not yet implemented");
    } else {
        // CPU fallback not available for BFP16
        throw std::runtime_error("BFP16 requires NPU - no CPU fallback available");
    }

    // [4] Convert output BFP16 → FP32 (with unshuffle)
    bfp16_quantizer_.read_from_npu(output_bfp16_shuffled_, output, M, N);

    // [5] Add bias (FP32)
    for (int i = 0; i < M; ++i) {
        output.row(i) += bias.transpose();
    }
}
```

**Key Differences**:
1. **No scale computation** (embedded in BFP16 blocks)
2. **Shuffle before NPU** (via `prepare_for_npu()`)
3. **Unshuffle after NPU** (via `read_from_npu()`)
4. **BFP16 buffer sizing** (1.125× original dimensions)
5. **No CPU fallback** (BFP16 conversion is fast enough to do on CPU if needed, but matmul must be NPU)

#### 3.3 Update NPU Callback Signature

The Python callback signature changes:

**BEFORE** (INT8):
```python
def npu_callback(user_data, A_int8, B_int8, C_int32, M, K, N):
    # A: (M, K) int8
    # B: (N, K) int8
    # C: (M, N) int32
```

**AFTER** (BFP16):
```python
def npu_callback_bfp16(user_data, A_bfp16, B_bfp16, C_bfp16, M, K, N):
    # A: (M, K_bfp16) uint8 (shuffled BFP16)
    # B: (N, K_bfp16) uint8 (shuffled BFP16)
    # C: (M, N_bfp16) uint8 (shuffled BFP16)
    # M, K, N: Original FP32 dimensions (for reference)
```

**Note**: K_bfp16 = ((K + 7) // 8) * 9 (bytes)

#### Verification Checklist
- [ ] Function signature updated
- [ ] Input conversion uses `prepare_for_npu()`
- [ ] Output buffer correctly sized (1.125× formula)
- [ ] NPU callback signature updated
- [ ] Output conversion uses `read_from_npu()`
- [ ] Bias addition remains FP32
- [ ] No references to INT8 or scales
- [ ] CPU fallback removed (with clear error)

---

### Task 4: Update All Matmul Call Sites (1-2 hours)

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/src/encoder_layer.cpp`

**Location**: 6 call sites across attention and FFN blocks

#### 4.1 Update Attention Block (lines 124-133)

**BEFORE**:
```cpp
// Q/K/V projections (NPU matmuls)
run_npu_linear(ln_output_, q_weight_int8_, q_weight_scale_, q_bias_, Q_);
run_npu_linear(ln_output_, k_weight_int8_, k_weight_scale_, k_bias_, K_);
run_npu_linear(ln_output_, v_weight_int8_, v_weight_scale_, v_bias_, V_);

// ... attention computation (CPU) ...

// Output projection (NPU matmul)
run_npu_linear(attn_heads, out_weight_int8_, out_weight_scale_, out_bias_, output);
```

**AFTER**:
```cpp
// Q/K/V projections (NPU matmuls with BFP16)
run_npu_linear(ln_output_, q_weight_bfp16_, q_bias_, Q_);
run_npu_linear(ln_output_, k_weight_bfp16_, k_bias_, K_);
run_npu_linear(ln_output_, v_weight_bfp16_, v_bias_, V_);

// ... attention computation (CPU) - NO CHANGES ...

// Output projection (NPU matmul with BFP16)
run_npu_linear(attn_heads, out_weight_bfp16_, out_bias_, output);
```

**Changes per call**:
- Remove `_scale_` parameter
- Update variable name: `_int8_` → `_bfp16_`

#### 4.2 Update FFN Block (lines 154-160)

**BEFORE**:
```cpp
// FC1: (seq_len, n_state) @ (n_state, ffn_dim) -> (seq_len, ffn_dim)
run_npu_linear(ln_output_, fc1_weight_int8_, fc1_weight_scale_, fc1_bias_, fc1_output_);

// GELU activation (CPU)
FeedForward::gelu(fc1_output_);

// FC2: (seq_len, ffn_dim) @ (ffn_dim, n_state) -> (seq_len, n_state)
run_npu_linear(fc1_output_, fc2_weight_int8_, fc2_weight_scale_, fc2_bias_, output);
```

**AFTER**:
```cpp
// FC1: (seq_len, n_state) @ (n_state, ffn_dim) -> (seq_len, ffn_dim)
run_npu_linear(ln_output_, fc1_weight_bfp16_, fc1_bias_, fc1_output_);

// GELU activation (CPU) - NO CHANGES
FeedForward::gelu(fc1_output_);

// FC2: (seq_len, ffn_dim) @ (ffn_dim, n_state) -> (seq_len, n_state)
run_npu_linear(fc1_output_, fc2_weight_bfp16_, fc2_bias_, output);
```

#### Summary: 6 Call Sites Updated
1. ✅ Q projection (line 124)
2. ✅ K projection (line 125)
3. ✅ V projection (line 126)
4. ✅ Out projection (line 133)
5. ✅ FC1 (line 154)
6. ✅ FC2 (line 160)

#### Verification Checklist
- [ ] All 6 call sites updated
- [ ] No `_scale_` parameters
- [ ] All `_int8_` → `_bfp16_`
- [ ] No other changes to surrounding code
- [ ] Attention and GELU remain on CPU

---

### Task 5: Update Build System (15 minutes)

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/CMakeLists.txt`

#### 5.1 Verify BFP16 Sources Are Included

Check that these files are in the build:
```cmake
add_library(whisper_xdna2
    src/encoder_layer.cpp
    src/bfp16_converter.cpp
    src/bfp16_quantization.cpp  # Should already be added
    src/quantization.cpp         # Keep for legacy support
    # ... other sources
)
```

#### 5.2 Update Test Targets (if needed)

Ensure test executables link against updated library:
```cmake
add_executable(test_encoder_layer_bfp16
    tests/test_encoder_layer_bfp16.cpp
)
target_link_libraries(test_encoder_layer_bfp16
    whisper_xdna2
    GTest::gtest_main
    Eigen3::Eigen
)
```

#### Verification Checklist
- [ ] All BFP16 sources included in build
- [ ] Library compiles without errors
- [ ] Test targets build successfully
- [ ] No linker errors

---

### Task 6: Create Integration Tests (2-3 hours)

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/tests/test_encoder_layer_bfp16.cpp`

#### 6.1 Test Single Layer Forward Pass

```cpp
TEST(EncoderLayerBFP16, SingleLayerForward) {
    // Setup
    EncoderLayer layer(0, 8, 512, 2048);

    // Load random weights (FP32)
    Eigen::MatrixXf q_weight = Eigen::MatrixXf::Random(512, 512);
    // ... (load all weights)

    layer.load_weights(q_weight, k_weight, ...);

    // Mock NPU callback (CPU fallback for testing)
    auto mock_callback = [](void*, const uint8_t*, const uint8_t*, uint8_t*,
                           size_t, size_t, size_t) -> int {
        // For testing: just fill output with zeros
        // Real NPU will do actual matmul
        return 0;
    };
    layer.set_npu_callback(reinterpret_cast<void*>(mock_callback), nullptr);

    // Run forward pass
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(512, 512);
    Eigen::MatrixXf output;
    layer.forward(input, output);

    // Verify output shape
    EXPECT_EQ(output.rows(), 512);
    EXPECT_EQ(output.cols(), 512);
}
```

#### 6.2 Test Weight Loading

```cpp
TEST(EncoderLayerBFP16, WeightLoadingBFP16) {
    EncoderLayer layer(0, 8, 512, 2048);

    // Create known weights
    Eigen::MatrixXf q_weight = Eigen::MatrixXf::Ones(512, 512);
    // ... (all weights as ones)

    layer.load_weights(q_weight, ...);

    // Verify weights are stored internally
    // (access via reflection or public getter if available)
    SUCCEED();  // Weight loading doesn't throw
}
```

#### 6.3 Test Accuracy vs FP32 Reference

```cpp
TEST(EncoderLayerBFP16, AccuracyVsFP32) {
    // Setup BFP16 layer
    EncoderLayer layer_bfp16(0, 8, 512, 2048);

    // Setup FP32 reference layer (using PyTorch via Python)
    // ... (load same weights)

    // Run both
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(512, 512);
    Eigen::MatrixXf output_bfp16, output_fp32;

    layer_bfp16.forward(input, output_bfp16);
    // reference_fp32.forward(input, output_fp32);  // via Python

    // Compute cosine similarity
    float cosine_sim = compute_cosine_similarity(output_bfp16, output_fp32);

    // Expect >99% accuracy
    EXPECT_GT(cosine_sim, 0.99);
}
```

#### Verification Checklist
- [ ] 3 core tests written
- [ ] All tests compile
- [ ] Tests use BFP16 weights
- [ ] Mock NPU callback for testing
- [ ] Accuracy test framework ready

---

### Task 7: Python Integration Test (1 hour)

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/test_cpp_bfp16_encoder.py`

```python
#!/usr/bin/env python3
"""
Test full 6-layer encoder with BFP16 NPU integration
"""

import numpy as np
import time

# Import C++ encoder wrapper (pybind11)
import whisper_encoder_cpp

def test_6layer_encoder_bfp16():
    """Test full encoder with BFP16"""

    # Create 6-layer encoder
    encoder = whisper_encoder_cpp.WhisperEncoder(
        n_layers=6,
        n_heads=8,
        n_state=512,
        ffn_dim=2048
    )

    # Load real Whisper Base weights (FP32)
    print("[1/4] Loading weights...")
    for i in range(6):
        # Load from weights/whisper_base_fp32/
        q_weight = np.load(f"weights/whisper_base_fp32/encoder.blocks.{i}.attn.query.weight.npy")
        # ... (load all weights)

        encoder.load_layer_weights(i, q_weight, k_weight, ...)

    # Setup NPU callback
    print("[2/4] Setting up NPU callback...")
    def npu_callback_bfp16(user_data, A, B, C, M, K, N):
        # A: (M, K_bfp16) uint8
        # B: (N, K_bfp16) uint8
        # C: (M, N_bfp16) uint8

        # Call actual NPU kernel
        # ... (XRT dispatch)
        return 0

    encoder.set_npu_callback(npu_callback_bfp16)

    # Run encoder
    print("[3/4] Running encoder...")
    input_tensor = np.random.randn(512, 512).astype(np.float32)

    start = time.time()
    output = encoder.forward(input_tensor)
    latency = (time.time() - start) * 1000

    print(f"[4/4] Results:")
    print(f"  Latency: {latency:.2f} ms")
    print(f"  Output shape: {output.shape}")
    print(f"  Realtime factor: {10240 / latency:.2f}×")

    # Verify
    assert output.shape == (512, 512)
    assert latency < 650  # Target: 520-580ms, allow 10% margin

    print("✅ BFP16 encoder test passed!")

if __name__ == "__main__":
    test_6layer_encoder_bfp16()
```

#### Verification Checklist
- [ ] Test script created
- [ ] Loads real Whisper weights
- [ ] Calls BFP16 encoder
- [ ] Measures latency
- [ ] Validates output shape
- [ ] Passes basic sanity checks

---

## Risk Analysis

### Risk 1: Memory Buffer Sizing Errors
**Probability**: Medium (30%)
**Impact**: High (crashes or incorrect results)

**Description**: BFP16 buffer sizing is complex (1.125× formula). Off-by-one errors could cause segfaults or data corruption.

**Mitigation**:
- Use `BFP16Config::BYTES_PER_BLOCK` constant everywhere
- Add assertions for buffer sizes before NPU calls
- Test with various matrix dimensions (512×512, 512×2048, edge cases)
- Use valgrind to detect memory errors

**Code Example**:
```cpp
// Defensive buffer sizing
const int K_blocks = (K + BFP16Config::BLOCK_SIZE - 1) / BFP16Config::BLOCK_SIZE;
const int K_bfp16_bytes = K_blocks * BFP16Config::BYTES_PER_BLOCK;

assert(input_bfp16_shuffled_.cols() == K_bfp16_bytes);  // Check before use
```

### Risk 2: NPU Callback Signature Mismatch
**Probability**: High (60%)
**Impact**: High (runtime errors, NPU failures)

**Description**: The NPU callback signature changes from INT8 to BFP16. Python-side callback must match exactly.

**Mitigation**:
- Document new signature clearly in code comments
- Add runtime type checks in Python callback
- Test callback with mock data before real NPU
- Print buffer shapes in callback for debugging

**Code Example**:
```python
def npu_callback_bfp16(user_data, A_ptr, B_ptr, C_ptr, M, K, N):
    # Calculate expected BFP16 buffer sizes
    K_bfp16 = ((K + 7) // 8) * 9
    N_bfp16 = ((N + 7) // 8) * 9

    # Verify buffer shapes
    A = np.ctypeslib.as_array(A_ptr, shape=(M, K_bfp16))
    assert A.dtype == np.uint8, "Expected uint8 for BFP16"
    # ... (continue with NPU call)
```

### Risk 3: Accuracy Below 99% Target
**Probability**: Low (15%)
**Impact**: High (requires redesign)

**Description**: BFP16 quantization error may accumulate across 6 layers, resulting in <99% accuracy.

**Mitigation**:
- Test accuracy layer-by-layer (isolate error source)
- Compare intermediate outputs vs PyTorch
- Verify BFP16 converter is working correctly (already tested in Phase 1)
- Consider per-channel exponents if per-tensor insufficient

**Contingency**:
- If accuracy is 95-99%: May be acceptable for production
- If accuracy <95%: Investigate block size tuning or hybrid approach

### Risk 4: Performance Slower Than Expected
**Probability**: Medium (40%)
**Impact**: Medium (may need optimization)

**Description**: Shuffle/unshuffle overhead or BFP16 conversion could slow down inference beyond 20% estimate.

**Mitigation**:
- Profile conversion and shuffle times separately
- Optimize critical paths (SIMD, vectorization)
- Consider moving shuffle to NPU kernel (Phase 4)
- Batch multiple conversions together

**Acceptance Criteria**:
- Target: 18-20× realtime (520-580ms)
- Acceptable: 15-20× realtime (520-680ms)
- Fallback: Use INT8 if >680ms (though accuracy suffers)

### Risk 5: Build/Link Errors
**Probability**: Low (20%)
**Impact**: Low (easy to fix)

**Description**: Missing includes, undefined symbols, or CMake configuration issues.

**Mitigation**:
- Compile incrementally (after each major change)
- Use `-Werror` to catch warnings early
- Check linker output for missing symbols
- Keep INT8 code alongside for reference

---

## Testing Strategy

### Unit Tests (2 hours)
1. **Weight Loading Test**
   - Load FP32 weights
   - Verify BFP16 conversion
   - Check buffer sizes

2. **Single Matmul Test**
   - Run one `run_npu_linear()` call
   - Mock NPU callback
   - Verify output shape

3. **Single Layer Test**
   - Run full layer forward pass
   - Check attention + FFN
   - Validate intermediate shapes

### Integration Tests (2 hours)
4. **6-Layer Encoder Test**
   - Load real Whisper weights
   - Run all 6 layers
   - Measure total latency

5. **Accuracy Test**
   - Compare vs PyTorch FP32
   - Compute cosine similarity
   - Target: >99%

6. **Performance Test**
   - 100-iteration stability
   - Measure average latency
   - Check consistency >99%

### Acceptance Criteria
- [ ] All unit tests pass (100%)
- [ ] Accuracy: >99% cosine similarity
- [ ] Performance: 18-20× realtime (520-580ms)
- [ ] Stability: >99% consistency
- [ ] No memory leaks (valgrind clean)
- [ ] No crashes (200 iterations)

---

## Timeline & Milestones

### Optimistic (8 hours)
- Task 1: encoder_layer.hpp (2 hours)
- Task 2: load_weights() (2 hours)
- Task 3: run_npu_linear() (2 hours)
- Task 4: Call sites (1 hour)
- Tasks 5-7: Build & test (1 hour)

### Realistic (10 hours)
- Task 1: encoder_layer.hpp (2.5 hours)
- Task 2: load_weights() (2.5 hours)
- Task 3: run_npu_linear() (3 hours)
- Task 4: Call sites (1.5 hours)
- Tasks 5-7: Build & test (1.5 hours)

### Pessimistic (12 hours)
- Task 1: encoder_layer.hpp (3 hours)
- Task 2: load_weights() (3 hours)
- Task 3: run_npu_linear() (4 hours)
- Task 4: Call sites (2 hours)
- Tasks 5-7: Build & test (2 hours)

**Expected**: 10 hours (realistic)

---

## Success Criteria

### Code Quality
- [ ] All INT8 references removed
- [ ] No scale parameters remaining
- [ ] Clean BFP16 abstractions
- [ ] Comprehensive error handling
- [ ] Memory-safe (valgrind clean)

### Functionality
- [ ] 6 layers compile and link
- [ ] All 36 matmuls use BFP16
- [ ] Weights load correctly
- [ ] Forward pass executes
- [ ] Output shape correct

### Performance
- [ ] Latency: 520-580ms (target)
- [ ] Realtime: 18-20× (target)
- [ ] Consistency: >99%
- [ ] Memory: <250MB

### Accuracy
- [ ] Cosine similarity: >99%
- [ ] Relative error: <1%
- [ ] Max error: <2%

---

## Next Steps (Phase 4)

After Phase 3 is complete:

1. **Compile BFP16 NPU Kernels**
   - Adapt MLIR-AIE examples
   - Generate XCLBin files
   - Validate on real hardware

2. **Update Python NPU Runtime**
   - Implement `npu_callback_bfp16()`
   - Load BFP16 XCLBin kernels
   - Test buffer transfers

3. **End-to-End Validation**
   - Run full encoder on NPU
   - Measure real latency
   - Validate accuracy

---

## Conclusion

Phase 3 will transform the encoder from INT8 to BFP16, unlocking >99% accuracy while maintaining 18-20× realtime performance. The plan is detailed, tested incrementally, and includes comprehensive risk mitigation.

**Key Wins**:
- Simpler API (no explicit scales)
- Higher accuracy (64.6% → >99%)
- Acceptable overhead (+12.5% memory)
- Clear migration path

**Ready to start!** 🚀

---

**Document Version**: 1.0
**Last Updated**: October 30, 2025
**Status**: Ready for Implementation
**Estimated Completion**: 8-12 hours

**Built with 💪 by Team BRO**
**Powered by AMD XDNA2 NPU (32 tiles, 50 TOPS)**
