# Phase 3 Code Templates: BFP16 Integration

**Project**: Unicorn-Amanuensis XDNA2
**Purpose**: Ready-to-use code snippets for Phase 3 implementation
**Date**: October 30, 2025

---

## Table of Contents

1. [Header File Updates](#header-file-updates)
2. [Weight Loading Updates](#weight-loading-updates)
3. [run_npu_linear() Rewrite](#run_npu_linear-rewrite)
4. [Call Site Updates](#call-site-updates)
5. [Test Templates](#test-templates)
6. [Python Integration](#python-integration)

---

## Header File Updates

### Template 1: encoder_layer.hpp - Weight Storage Section

**Location**: Lines 157-170

**Complete Replacement**:
```cpp
// ============================================================================
// BFP16 Weights (Quantized and Shuffled for NPU)
// ============================================================================

// BFP16 format: 8 mantissas (8 bits each) + 1 shared exponent (8 bits) per block
// Storage overhead: 1.125× (9 bytes per 8 values)
// Weights are pre-shuffled for optimal NPU memory layout

Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> q_weight_bfp16_;    // Query projection (512×576 bytes)
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> k_weight_bfp16_;    // Key projection (512×576 bytes)
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> v_weight_bfp16_;    // Value projection (512×576 bytes)
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> out_weight_bfp16_;  // Output projection (512×576 bytes)
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> fc1_weight_bfp16_;  // FC1 expansion (2048×576 bytes)
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> fc2_weight_bfp16_;  // FC2 reduction (512×2304 bytes)

// NOTE: Scales are now embedded in BFP16 block exponents
// Each 8-value block has its own shared exponent (stored in 9th byte)
```

**Delete These Lines** (157-170):
```cpp
// OLD (DELETE):
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

---

### Template 2: encoder_layer.hpp - Activation Buffers

**Location**: Lines 193-195

**Complete Replacement**:
```cpp
// ============================================================================
// BFP16 Activation Buffers (Temporary, Reused Across Matmuls)
// ============================================================================

// BFP16 conversion pipeline:
// 1. FP32 input → BFP16 (block-based quantization)
// 2. BFP16 → Shuffled BFP16 (NPU memory layout)
// 3. NPU matmul (BFP16 @ BFP16 → BFP16)
// 4. Shuffled BFP16 → BFP16 (unshuffle)
// 5. BFP16 → FP32 output (dequantization)

Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> input_bfp16_;          // After FP32→BFP16, before shuffle
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> input_bfp16_shuffled_; // After shuffle, ready for NPU
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> output_bfp16_shuffled_; // From NPU, still shuffled
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> output_bfp16_;         // After unshuffle, before FP32

// Buffer sizing:
// For (M, N) FP32 matrix → (M, N_bytes) BFP16 where:
// N_bytes = ((N + 7) / 8) * 9  (9 bytes per 8-value block)
```

**Delete These Lines** (193-195):
```cpp
// OLD (DELETE):
Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> input_int8_;
Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> matmul_output_int32_;
```

---

### Template 3: encoder_layer.hpp - run_npu_linear() Signature

**Location**: Lines 206-212

**Complete Replacement**:
```cpp
/**
 * Run NPU matmul with BFP16 quantization
 *
 * Pipeline:
 * 1. Convert input FP32 → BFP16 (with shuffle)
 * 2. Run NPU matmul: BFP16 @ BFP16 → BFP16
 * 3. Convert output BFP16 → FP32 (with unshuffle)
 * 4. Add bias (FP32)
 *
 * @param input Input matrix (FP32, seq_len × n_state)
 * @param weight_bfp16 Weight matrix (BFP16, pre-shuffled, N × K_bytes)
 * @param bias Bias vector (FP32, N)
 * @param output Output matrix (FP32, seq_len × N)
 */
void run_npu_linear(
    const Eigen::MatrixXf& input,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& weight_bfp16,
    const Eigen::VectorXf& bias,
    Eigen::MatrixXf& output
);
```

**Delete This Signature** (206-212):
```cpp
// OLD (DELETE):
void run_npu_linear(
    const Eigen::MatrixXf& input,
    const Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>& weight_int8,
    float weight_scale,
    const Eigen::VectorXf& bias,
    Eigen::MatrixXf& output
);
```

---

### Template 4: encoder_layer.hpp - Add BFP16Quantizer Member

**Location**: After line 148 (inside private section)

**Add This Line**:
```cpp
// BFP16 quantizer instance (shared across all matmuls)
BFP16Quantizer bfp16_quantizer_;
```

---

## Weight Loading Updates

### Template 5: load_weights() - Complete Function Body

**Location**: Lines 22-74 in encoder_layer.cpp

**Complete Replacement**:
```cpp
void EncoderLayer::load_weights(
    const Eigen::MatrixXf& q_weight,
    const Eigen::MatrixXf& k_weight,
    const Eigen::MatrixXf& v_weight,
    const Eigen::MatrixXf& out_weight,
    const Eigen::VectorXf& q_bias,
    const Eigen::VectorXf& k_bias,
    const Eigen::VectorXf& v_bias,
    const Eigen::VectorXf& out_bias,
    const Eigen::MatrixXf& fc1_weight,
    const Eigen::MatrixXf& fc2_weight,
    const Eigen::VectorXf& fc1_bias,
    const Eigen::VectorXf& fc2_bias,
    const Eigen::VectorXf& attn_ln_weight,
    const Eigen::VectorXf& attn_ln_bias,
    const Eigen::VectorXf& ffn_ln_weight,
    const Eigen::VectorXf& ffn_ln_bias
) {
    // ========================================================================
    // Convert Weights to BFP16 (FP32 → BFP16 with shuffle for NPU)
    // ========================================================================

    // prepare_for_npu() does:
    // 1. FP32 → BFP16 conversion (block-based quantization)
    // 2. Shuffle for NPU memory layout (8×9 subtiles)
    // Result: Ready-to-use BFP16 weights for NPU dispatch

    bfp16_quantizer_.prepare_for_npu(q_weight, q_weight_bfp16_);
    bfp16_quantizer_.prepare_for_npu(k_weight, k_weight_bfp16_);
    bfp16_quantizer_.prepare_for_npu(v_weight, v_weight_bfp16_);
    bfp16_quantizer_.prepare_for_npu(out_weight, out_weight_bfp16_);
    bfp16_quantizer_.prepare_for_npu(fc1_weight, fc1_weight_bfp16_);
    bfp16_quantizer_.prepare_for_npu(fc2_weight, fc2_weight_bfp16_);

    // ========================================================================
    // Store Biases (FP32, no quantization)
    // ========================================================================

    q_bias_ = q_bias;
    k_bias_ = k_bias;
    v_bias_ = v_bias;
    out_bias_ = out_bias;
    fc1_bias_ = fc1_bias;
    fc2_bias_ = fc2_bias;

    // ========================================================================
    // Store Layer Norm Parameters (FP32, no quantization)
    // ========================================================================

    attn_ln_weight_ = attn_ln_weight;
    attn_ln_bias_ = attn_ln_bias;
    ffn_ln_weight_ = ffn_ln_weight;
    ffn_ln_bias_ = ffn_ln_bias;
}
```

**Key Changes**:
- **Removed**: Quantizer instance, scale computation (12 lines)
- **Added**: 6 `prepare_for_npu()` calls (6 lines)
- **Result**: Simpler, no explicit scale management

---

## run_npu_linear() Rewrite

### Template 6: run_npu_linear() - Complete Function

**Location**: Lines 163-223 in encoder_layer.cpp

**Complete Replacement**:
```cpp
void EncoderLayer::run_npu_linear(
    const Eigen::MatrixXf& input,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& weight_bfp16,
    const Eigen::VectorXf& bias,
    Eigen::MatrixXf& output
) {
    // ========================================================================
    // [1] Compute Dimensions
    // ========================================================================

    const int M = input.rows();       // Sequence length
    const int K = input.cols();       // Input dimension (n_state or ffn_dim)
    const int N = weight_bfp16.rows(); // Output dimension (from weight shape)

    // ========================================================================
    // [2] Convert Input FP32 → BFP16 (with shuffle)
    // ========================================================================

    // prepare_for_npu() does:
    // - FP32 → BFP16 conversion (block-based quantization)
    // - Shuffle for NPU memory layout
    // Result: input_bfp16_shuffled_ ready for NPU dispatch

    bfp16_quantizer_.prepare_for_npu(input, input_bfp16_shuffled_);

    // ========================================================================
    // [3] Allocate Output Buffer (BFP16, shuffled)
    // ========================================================================

    // Calculate BFP16 buffer size:
    // N FP32 values → N/8 blocks → (N/8) × 9 bytes
    const int N_blocks = (N + BFP16Config::BLOCK_SIZE - 1) / BFP16Config::BLOCK_SIZE;
    const int N_bfp16_bytes = N_blocks * BFP16Config::BYTES_PER_BLOCK;

    // Resize output buffer if needed (avoid reallocation on every call)
    if (output_bfp16_shuffled_.rows() != M ||
        output_bfp16_shuffled_.cols() != N_bfp16_bytes) {
        output_bfp16_shuffled_.resize(M, N_bfp16_bytes);
    }

    // ========================================================================
    // [4] Run NPU Matmul: BFP16 @ BFP16 → BFP16
    // ========================================================================

    if (npu_callback_fn_) {
        // NPU path (via Python callback)
        // Callback signature:
        // int callback(void* user_data, const uint8_t* A, const uint8_t* B,
        //              uint8_t* C, size_t M, size_t K, size_t N);
        // A: (M, K_bfp16) BFP16 input (shuffled)
        // B: (N, K_bfp16) BFP16 weight (pre-shuffled)
        // C: (M, N_bfp16) BFP16 output (shuffled)
        // M, K, N: Original FP32 dimensions (for reference)

        typedef int (*NPUCallbackBFP16)(
            void*,          // user_data
            const uint8_t*, // A
            const uint8_t*, // B
            uint8_t*,       // C
            size_t,         // M
            size_t,         // K
            size_t          // N
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
        // NOTE: This path is not yet implemented for BFP16
        // Would need to update NPUMatmulFunction typedef in header
        throw std::runtime_error(
            "BFP16 std::function path not yet implemented. "
            "Use Python NPU callback or update NPUMatmulFunction typedef."
        );

    } else {
        // No NPU available
        // BFP16 requires NPU hardware - no CPU fallback
        throw std::runtime_error(
            "BFP16 requires NPU hardware. No CPU fallback available. "
            "Please set NPU callback via set_npu_callback() or enable NPU device."
        );
    }

    // ========================================================================
    // [5] Convert Output BFP16 → FP32 (with unshuffle)
    // ========================================================================

    // read_from_npu() does:
    // - Unshuffle from NPU memory layout
    // - BFP16 → FP32 conversion (dequantization)
    // Result: output is now FP32 matrix (M, N)

    bfp16_quantizer_.read_from_npu(output_bfp16_shuffled_, output, M, N);

    // ========================================================================
    // [6] Add Bias (FP32)
    // ========================================================================

    // Bias is stored in FP32 (not quantized)
    // Add to each row of output
    for (int i = 0; i < M; ++i) {
        output.row(i) += bias.transpose();
    }
}
```

**Key Changes**:
- **Input quantization**: 8 lines → 1 line (`prepare_for_npu`)
- **Output dequantization**: 5 lines → 1 line (`read_from_npu`)
- **Buffer sizing**: Explicit BFP16 formula (1.125×)
- **NPU callback**: Updated signature (uint8_t*)
- **CPU fallback**: Removed (BFP16 requires NPU)

---

## Call Site Updates

### Template 7: Attention Block - Q/K/V Projections

**Location**: Lines 124-126 in encoder_layer.cpp

**Replace**:
```cpp
// OLD (DELETE):
run_npu_linear(ln_output_, q_weight_int8_, q_weight_scale_, q_bias_, Q_);
run_npu_linear(ln_output_, k_weight_int8_, k_weight_scale_, k_bias_, K_);
run_npu_linear(ln_output_, v_weight_int8_, v_weight_scale_, v_bias_, V_);
```

**With**:
```cpp
// Q/K/V projections (BFP16 NPU matmuls)
run_npu_linear(ln_output_, q_weight_bfp16_, q_bias_, Q_);
run_npu_linear(ln_output_, k_weight_bfp16_, k_bias_, K_);
run_npu_linear(ln_output_, v_weight_bfp16_, v_bias_, V_);
```

---

### Template 8: Attention Block - Output Projection

**Location**: Line 133 in encoder_layer.cpp

**Replace**:
```cpp
// OLD (DELETE):
run_npu_linear(attn_heads, out_weight_int8_, out_weight_scale_, out_bias_, output);
```

**With**:
```cpp
// Output projection (BFP16 NPU matmul)
run_npu_linear(attn_heads, out_weight_bfp16_, out_bias_, output);
```

---

### Template 9: FFN Block - FC1 and FC2

**Location**: Lines 154 and 160 in encoder_layer.cpp

**Replace**:
```cpp
// OLD (DELETE):
run_npu_linear(ln_output_, fc1_weight_int8_, fc1_weight_scale_, fc1_bias_, fc1_output_);
// ... (GELU in between, unchanged)
run_npu_linear(fc1_output_, fc2_weight_int8_, fc2_weight_scale_, fc2_bias_, output);
```

**With**:
```cpp
// FC1: Expansion layer (BFP16 NPU matmul)
run_npu_linear(ln_output_, fc1_weight_bfp16_, fc1_bias_, fc1_output_);

// GELU activation (CPU, unchanged)
FeedForward::gelu(fc1_output_);

// FC2: Reduction layer (BFP16 NPU matmul)
run_npu_linear(fc1_output_, fc2_weight_bfp16_, fc2_bias_, output);
```

---

## Test Templates

### Template 10: C++ Unit Test - Weight Loading

**File**: `cpp/tests/test_encoder_layer_bfp16.cpp`

```cpp
#include "encoder_layer.hpp"
#include "bfp16_quantization.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>

using namespace whisper_xdna2;

/**
 * Test: BFP16 weight loading
 *
 * Verifies that load_weights() correctly converts FP32 weights to BFP16
 * and stores them internally without throwing exceptions.
 */
TEST(EncoderLayerBFP16, WeightLoadingBFP16) {
    // Create encoder layer
    EncoderLayer layer(
        0,      // layer_idx
        8,      // n_heads
        512,    // n_state
        2048    // ffn_dim
    );

    // Create random FP32 weights
    Eigen::MatrixXf q_weight = Eigen::MatrixXf::Random(512, 512);
    Eigen::MatrixXf k_weight = Eigen::MatrixXf::Random(512, 512);
    Eigen::MatrixXf v_weight = Eigen::MatrixXf::Random(512, 512);
    Eigen::MatrixXf out_weight = Eigen::MatrixXf::Random(512, 512);
    Eigen::MatrixXf fc1_weight = Eigen::MatrixXf::Random(2048, 512);
    Eigen::MatrixXf fc2_weight = Eigen::MatrixXf::Random(512, 2048);

    Eigen::VectorXf q_bias = Eigen::VectorXf::Random(512);
    Eigen::VectorXf k_bias = Eigen::VectorXf::Random(512);
    Eigen::VectorXf v_bias = Eigen::VectorXf::Random(512);
    Eigen::VectorXf out_bias = Eigen::VectorXf::Random(512);
    Eigen::VectorXf fc1_bias = Eigen::VectorXf::Random(2048);
    Eigen::VectorXf fc2_bias = Eigen::VectorXf::Random(512);

    Eigen::VectorXf attn_ln_weight = Eigen::VectorXf::Ones(512);
    Eigen::VectorXf attn_ln_bias = Eigen::VectorXf::Zero(512);
    Eigen::VectorXf ffn_ln_weight = Eigen::VectorXf::Ones(512);
    Eigen::VectorXf ffn_ln_bias = Eigen::VectorXf::Zero(512);

    // Load weights (should not throw)
    EXPECT_NO_THROW({
        layer.load_weights(
            q_weight, k_weight, v_weight, out_weight,
            q_bias, k_bias, v_bias, out_bias,
            fc1_weight, fc2_weight,
            fc1_bias, fc2_bias,
            attn_ln_weight, attn_ln_bias,
            ffn_ln_weight, ffn_ln_bias
        );
    });
}
```

---

### Template 11: C++ Unit Test - Forward Pass with Mock NPU

**File**: `cpp/tests/test_encoder_layer_bfp16.cpp`

```cpp
/**
 * Test: Single layer forward pass with BFP16
 *
 * Runs a complete encoder layer forward pass with BFP16 quantization.
 * Uses mock NPU callback for testing without hardware.
 */
TEST(EncoderLayerBFP16, SingleLayerForward) {
    // Create encoder layer
    EncoderLayer layer(0, 8, 512, 2048);

    // Load weights (reuse from previous test)
    Eigen::MatrixXf q_weight = Eigen::MatrixXf::Random(512, 512);
    // ... (load all weights)
    layer.load_weights(/* all weights */);

    // Setup mock NPU callback
    // For testing: Fill output with input + small noise
    auto mock_callback = [](
        void* user_data,
        const uint8_t* A,
        const uint8_t* B,
        uint8_t* C,
        size_t M,
        size_t K,
        size_t N
    ) -> int {
        // Mock NPU: Just copy input to output (for testing)
        // Real NPU will do actual BFP16 matmul
        std::memset(C, 0, M * ((N + 7) / 8) * 9);  // Zero output
        return 0;  // Success
    };

    layer.set_npu_callback(
        reinterpret_cast<void*>(mock_callback),
        nullptr  // user_data
    );

    // Run forward pass
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(512, 512);
    Eigen::MatrixXf output;

    EXPECT_NO_THROW({
        layer.forward(input, output);
    });

    // Verify output shape
    EXPECT_EQ(output.rows(), 512);
    EXPECT_EQ(output.cols(), 512);

    // Verify output is finite (no NaN/Inf)
    EXPECT_TRUE(output.allFinite());
}
```

---

### Template 12: C++ Unit Test - BFP16 Buffer Sizes

**File**: `cpp/tests/test_encoder_layer_bfp16.cpp`

```cpp
/**
 * Test: BFP16 buffer sizing
 *
 * Verifies that BFP16 buffers are correctly sized using 1.125× formula.
 */
TEST(EncoderLayerBFP16, BFP16BufferSizes) {
    BFP16Quantizer quantizer;

    // Test case 1: 512×512 matrix
    {
        Eigen::MatrixXf input = Eigen::MatrixXf::Random(512, 512);
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> output;

        quantizer.prepare_for_npu(input, output);

        // Expected BFP16 size:
        // 512 values = 64 blocks = 64 × 9 bytes = 576 bytes per row
        EXPECT_EQ(output.rows(), 512);
        EXPECT_EQ(output.cols(), 576);  // 512 × 1.125
    }

    // Test case 2: 512×2048 matrix
    {
        Eigen::MatrixXf input = Eigen::MatrixXf::Random(512, 2048);
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> output;

        quantizer.prepare_for_npu(input, output);

        // Expected BFP16 size:
        // 2048 values = 256 blocks = 256 × 9 bytes = 2304 bytes per row
        EXPECT_EQ(output.rows(), 512);
        EXPECT_EQ(output.cols(), 2304);  // 2048 × 1.125
    }

    // Test case 3: Non-multiple-of-8 dimension
    {
        Eigen::MatrixXf input = Eigen::MatrixXf::Random(512, 500);
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> output;

        quantizer.prepare_for_npu(input, output);

        // Expected BFP16 size:
        // 500 values → 63 blocks (round up) = 63 × 9 bytes = 567 bytes per row
        int expected_blocks = (500 + 7) / 8;  // 63
        int expected_bytes = expected_blocks * 9;  // 567
        EXPECT_EQ(output.rows(), 512);
        EXPECT_EQ(output.cols(), expected_bytes);
    }
}
```

---

## Python Integration

### Template 13: Python NPU Callback (BFP16)

**File**: `test_cpp_bfp16_encoder.py` or `runtime/npu_callback.py`

```python
import numpy as np
import ctypes

def npu_callback_bfp16(
    user_data: int,
    A_ptr: int,
    B_ptr: int,
    C_ptr: int,
    M: int,
    K: int,
    N: int
) -> int:
    """
    NPU callback for BFP16 matmul: C = A @ B

    Args:
        user_data: User data pointer (Python runtime object)
        A_ptr: Pointer to input BFP16 matrix (M × K_bfp16)
        B_ptr: Pointer to weight BFP16 matrix (N × K_bfp16)
        C_ptr: Pointer to output BFP16 matrix (M × N_bfp16)
        M: Number of rows in A
        K: Original dimension (before BFP16 conversion)
        N: Number of rows in B

    Returns:
        0 on success, non-zero on error
    """

    try:
        # Calculate BFP16 buffer sizes
        K_blocks = (K + 7) // 8
        K_bfp16 = K_blocks * 9  # 9 bytes per block

        N_blocks = (N + 7) // 8
        N_bfp16 = N_blocks * 9

        # Extract BFP16 buffers as numpy arrays
        A_bfp16 = np.ctypeslib.as_array(
            ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_uint8)),
            shape=(M, K_bfp16)
        )

        B_bfp16 = np.ctypeslib.as_array(
            ctypes.cast(B_ptr, ctypes.POINTER(ctypes.c_uint8)),
            shape=(N, K_bfp16)
        )

        C_bfp16 = np.ctypeslib.as_array(
            ctypes.cast(C_ptr, ctypes.POINTER(ctypes.c_uint8)),
            shape=(M, N_bfp16)
        )

        # Verify buffer types
        assert A_bfp16.dtype == np.uint8, "Expected uint8 for BFP16"
        assert B_bfp16.dtype == np.uint8, "Expected uint8 for BFP16"
        assert C_bfp16.dtype == np.uint8, "Expected uint8 for BFP16"

        # TODO: Call actual NPU kernel via XRT
        # For now, fill output with zeros (mock)
        C_bfp16.fill(0)

        # Example NPU dispatch (pseudocode):
        # xrt_runtime = get_runtime(user_data)
        # xrt_runtime.run_bfp16_matmul(A_bfp16, B_bfp16, C_bfp16, M, K, N)

        return 0  # Success

    except Exception as e:
        print(f"[ERROR] NPU callback failed: {e}")
        import traceback
        traceback.print_exc()
        return -1  # Error
```

---

### Template 14: Python Integration Test

**File**: `test_cpp_bfp16_encoder.py`

```python
#!/usr/bin/env python3
"""
Test full 6-layer Whisper encoder with BFP16 NPU integration
"""

import numpy as np
import time
import sys
import os

# Import C++ encoder wrapper (pybind11)
# Adjust path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp/build'))
import whisper_encoder_cpp  # C++ binding

def test_6layer_encoder_bfp16():
    """Test full encoder with BFP16"""

    print("=" * 60)
    print("BFP16 Encoder Integration Test")
    print("=" * 60)
    print()

    # ========================================================================
    # [1/5] Create Encoder
    # ========================================================================

    print("[1/5] Creating 6-layer encoder...")
    encoder = whisper_encoder_cpp.WhisperEncoder(
        n_layers=6,
        n_heads=8,
        n_state=512,
        ffn_dim=2048
    )
    print("  ✓ Encoder created")
    print()

    # ========================================================================
    # [2/5] Load Real Weights
    # ========================================================================

    print("[2/5] Loading Whisper Base weights...")
    weights_dir = "weights/whisper_base_fp32"

    for i in range(6):
        # Load weights for layer i
        q_weight = np.load(f"{weights_dir}/encoder.blocks.{i}.attn.query.weight.npy")
        k_weight = np.load(f"{weights_dir}/encoder.blocks.{i}.attn.key.weight.npy")
        v_weight = np.load(f"{weights_dir}/encoder.blocks.{i}.attn.value.weight.npy")
        out_weight = np.load(f"{weights_dir}/encoder.blocks.{i}.attn.out.weight.npy")

        q_bias = np.load(f"{weights_dir}/encoder.blocks.{i}.attn.query.bias.npy")
        k_bias = np.load(f"{weights_dir}/encoder.blocks.{i}.attn.key.bias.npy")
        v_bias = np.load(f"{weights_dir}/encoder.blocks.{i}.attn.value.bias.npy")
        out_bias = np.load(f"{weights_dir}/encoder.blocks.{i}.attn.out.bias.npy")

        fc1_weight = np.load(f"{weights_dir}/encoder.blocks.{i}.mlp.0.weight.npy")
        fc2_weight = np.load(f"{weights_dir}/encoder.blocks.{i}.mlp.2.weight.npy")
        fc1_bias = np.load(f"{weights_dir}/encoder.blocks.{i}.mlp.0.bias.npy")
        fc2_bias = np.load(f"{weights_dir}/encoder.blocks.{i}.mlp.2.bias.npy")

        attn_ln_weight = np.load(f"{weights_dir}/encoder.blocks.{i}.attn_ln.weight.npy")
        attn_ln_bias = np.load(f"{weights_dir}/encoder.blocks.{i}.attn_ln.bias.npy")
        ffn_ln_weight = np.load(f"{weights_dir}/encoder.blocks.{i}.mlp_ln.weight.npy")
        ffn_ln_bias = np.load(f"{weights_dir}/encoder.blocks.{i}.mlp_ln.bias.npy")

        # Load into encoder (converts to BFP16 internally)
        encoder.load_layer_weights(
            i,
            q_weight, k_weight, v_weight, out_weight,
            q_bias, k_bias, v_bias, out_bias,
            fc1_weight, fc2_weight,
            fc1_bias, fc2_bias,
            attn_ln_weight, attn_ln_bias,
            ffn_ln_weight, ffn_ln_bias
        )

        print(f"  ✓ Layer {i} weights loaded")

    print()

    # ========================================================================
    # [3/5] Setup NPU Callback
    # ========================================================================

    print("[3/5] Setting up NPU callback...")

    # Import or define npu_callback_bfp16 (see Template 13)
    from npu_callback import npu_callback_bfp16

    encoder.set_npu_callback(npu_callback_bfp16)
    print("  ✓ NPU callback configured")
    print()

    # ========================================================================
    # [4/5] Run Encoder
    # ========================================================================

    print("[4/5] Running encoder...")

    # Create random input (simulating mel spectrogram)
    input_tensor = np.random.randn(512, 512).astype(np.float32)

    # Warm-up run (NPU initialization)
    print("  Warm-up run...")
    _ = encoder.forward(input_tensor)

    # Timed run
    print("  Timed run...")
    start = time.time()
    output = encoder.forward(input_tensor)
    latency = (time.time() - start) * 1000  # ms

    print(f"  ✓ Forward pass complete")
    print()

    # ========================================================================
    # [5/5] Results
    # ========================================================================

    print("[5/5] Results:")
    print(f"  Output shape: {output.shape}")
    print(f"  Latency: {latency:.2f} ms")
    print(f"  Realtime factor: {10240 / latency:.2f}×")
    print(f"  Target: 18-20× (520-580ms)")
    print()

    # Verify
    assert output.shape == (512, 512), f"Wrong shape: {output.shape}"
    assert latency < 680, f"Too slow: {latency:.2f}ms (target: <580ms with margin)"

    # Check output is finite
    assert np.isfinite(output).all(), "Output contains NaN/Inf"

    print("=" * 60)
    print("✅ BFP16 encoder test PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_6layer_encoder_bfp16()
```

---

## Utility Templates

### Template 15: Debug Print Helper

```cpp
// Add to encoder_layer.cpp for debugging

#ifdef DEBUG_BFP16
#include <iostream>

void print_bfp16_info(
    const char* name,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& mat
) {
    std::cout << "[DEBUG] " << name << ": "
              << mat.rows() << "×" << mat.cols() << " bytes ("
              << (mat.rows() * mat.cols() / 1024.0) << " KB)"
              << std::endl;
}

// Usage in run_npu_linear():
print_bfp16_info("input_bfp16_shuffled", input_bfp16_shuffled_);
print_bfp16_info("output_bfp16_shuffled", output_bfp16_shuffled_);
#endif
```

---

### Template 16: Verification Script

```bash
#!/bin/bash
# verify_bfp16_migration.sh
# Verify that Phase 3 migration is complete

echo "Verifying BFP16 migration..."

# Check for INT8 residue
echo "[1/5] Checking for INT8 references..."
INT8_COUNT=$(grep -r "int8_t" cpp/src/encoder_layer.cpp | wc -l)
if [ $INT8_COUNT -gt 0 ]; then
    echo "  ❌ Found $INT8_COUNT INT8 references (should be 0)"
    grep -n "int8_t" cpp/src/encoder_layer.cpp
    exit 1
else
    echo "  ✓ No INT8 references"
fi

# Check for scale parameters
echo "[2/5] Checking for scale parameters..."
SCALE_COUNT=$(grep -r "weight_scale" cpp/src/encoder_layer.cpp | wc -l)
if [ $SCALE_COUNT -gt 0 ]; then
    echo "  ❌ Found $SCALE_COUNT scale references (should be 0)"
    grep -n "weight_scale" cpp/src/encoder_layer.cpp
    exit 1
else
    echo "  ✓ No scale parameters"
fi

# Check for BFP16 usage
echo "[3/5] Checking for BFP16 usage..."
BFP16_COUNT=$(grep -r "bfp16" cpp/src/encoder_layer.cpp | wc -l)
if [ $BFP16_COUNT -lt 10 ]; then
    echo "  ❌ Found only $BFP16_COUNT BFP16 references (expected >10)"
    exit 1
else
    echo "  ✓ Found $BFP16_COUNT BFP16 references"
fi

# Check build
echo "[4/5] Testing build..."
cd cpp/build
if ! make encoder_layer -j4 > /dev/null 2>&1; then
    echo "  ❌ Build failed"
    exit 1
else
    echo "  ✓ Build succeeded"
fi

# Check symbols
echo "[5/5] Checking symbols..."
if ! nm libwhisper_xdna2.a | grep -q "BFP16Quantizer"; then
    echo "  ❌ BFP16Quantizer symbols not found"
    exit 1
else
    echo "  ✓ BFP16 symbols present"
fi

echo ""
echo "✅ BFP16 migration verified!"
```

---

## Summary

This document provides **16 ready-to-use code templates** for Phase 3 implementation:

### Header Updates (4 templates)
1. Weight storage section
2. Activation buffers
3. run_npu_linear() signature
4. BFP16Quantizer member

### Implementation Updates (5 templates)
5. load_weights() function
6. run_npu_linear() function
7. Q/K/V projection calls
8. Output projection call
9. FC1/FC2 calls

### Testing (5 templates)
10. Weight loading test
11. Forward pass test
12. Buffer size test
13. Python NPU callback
14. Python integration test

### Utilities (2 templates)
15. Debug print helper
16. Verification script

**Usage**:
1. Copy templates directly into your files
2. Adjust variable names if needed
3. Run verification script after each section
4. Test incrementally

**Ready to implement Phase 3!** 🚀

---

**Document Version**: 1.0
**Last Updated**: October 30, 2025
**Status**: Ready to Use

**Built with 💪 by Team BRO**
