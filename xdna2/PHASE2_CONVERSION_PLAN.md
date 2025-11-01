# Phase 2: BFP16 Conversion Plan

**Date**: October 30, 2025
**Project**: Unicorn-Amanuensis XDNA2
**Goal**: Map INT8 quantization workflow to BFP16 workflow
**Estimated Duration**: 6-8 hours

---

## Executive Summary

This document provides a **line-by-line mapping** of all code changes required to migrate from INT8 quantization to BFP16 block floating point format.

**Key Changes**:
- Remove per-tensor scales (embedded in block exponents)
- Change buffer types: `int8_t` → `uint8_t`, `int32_t` → `uint8_t`
- Add shuffle/unshuffle operations before/after NPU calls
- Update NPU callback signature for BFP16
- Adjust buffer size calculations (1.0x → 1.125x)

**Files Modified**: 3
**Lines Changed**: ~150 lines
**Complexity**: Easy (mostly type changes and API updates)

---

## Table of Contents

1. [Current INT8 Workflow](#current-int8-workflow)
2. [Target BFP16 Workflow](#target-bfp16-workflow)
3. [File-by-File Changes](#file-by-file-changes)
4. [Function Mapping](#function-mapping)
5. [Buffer Size Changes](#buffer-size-changes)
6. [NPU Callback Changes](#npu-callback-changes)
7. [Testing Strategy](#testing-strategy)

---

## Current INT8 Workflow

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   Current INT8 Pipeline                      │
└─────────────────────────────────────────────────────────────┘

Input (FP32)
    ↓
[1] Quantizer::quantize_tensor()
    - Compute scale: max(abs(tensor)) / 127
    - Quantize: round(tensor / scale).clip(-127, 127)
    - Store scale: float
    ↓
INT8 Buffer (Eigen::Matrix<int8_t>)
    ↓
[2] NPU Matmul (INT8 @ INT8 → INT32)
    - npu_callback_fn_(input_int8, weight_int8, output_int32, M, K, N)
    - XCLBin kernel: matmul_4tile_int8.xclbin
    ↓
INT32 Buffer (Eigen::Matrix<int32_t>)
    ↓
[3] Quantizer::dequantize_matmul_output()
    - output_fp32 = output_int32 * scale_A * scale_B
    ↓
Output (FP32)
```

### Key Functions

**File**: `cpp/src/quantization.cpp`

1. **compute_scale()** (line 5-8):
   ```cpp
   float Quantizer::compute_scale(const Eigen::MatrixXf& tensor) {
       float max_val = tensor.cwiseAbs().maxCoeff();
       return std::max(max_val / 127.0f, QuantizationConfig::MIN_SCALE);
   }
   ```

2. **quantize_tensor()** (line 10-24):
   ```cpp
   void Quantizer::quantize_tensor(
       const Eigen::MatrixXf& input,
       Eigen::Matrix<int8_t, Dynamic, Dynamic>& output,
       float& scale
   ) {
       scale = compute_scale(input);
       output.resize(input.rows(), input.cols());
       float inv_scale = 1.0f / scale;
       for (int i = 0; i < input.rows(); i++) {
           for (int j = 0; j < input.cols(); j++) {
               output(i, j) = quantization_helpers::quantize_value(input(i, j), inv_scale);
           }
       }
   }
   ```

3. **dequantize_matmul_output()** (line 41-55):
   ```cpp
   void Quantizer::dequantize_matmul_output(
       const Eigen::Matrix<int32_t, Dynamic, Dynamic>& input,
       Eigen::MatrixXf& output,
       float input_scale,
       float weight_scale
   ) {
       float combined_scale = input_scale * weight_scale;
       output.resize(input.rows(), input.cols());
       for (int i = 0; i < input.rows(); i++) {
           for (int j = 0; j < input.cols(); j++) {
               output(i, j) = quantization_helpers::dequantize_matmul_value(
                   input(i, j), combined_scale
               );
           }
       }
   }
   ```

### Buffer Types

```cpp
// Weights (stored)
Eigen::Matrix<int8_t, Dynamic, Dynamic> q_weight_int8_;
float q_weight_scale_;

// Activations (temporary)
Eigen::Matrix<int8_t, Dynamic, Dynamic> input_int8_;
Eigen::Matrix<int32_t, Dynamic, Dynamic> matmul_output_int32_;
```

---

## Target BFP16 Workflow

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   Target BFP16 Pipeline                      │
└─────────────────────────────────────────────────────────────┘

Input (FP32)
    ↓
[1] BFP16Quantizer::convert_to_bfp16()
    - Find block exponents (per 8 values)
    - Quantize to 8-bit mantissas
    - Pack: 8 mantissas + 1 exponent = 9 bytes per block
    - NO SCALE NEEDED (embedded in exponents)
    ↓
BFP16 Buffer (Eigen::Matrix<uint8_t>)
    ↓
[2] BFP16Quantizer::shuffle_bfp16()
    - Rearrange 8×9 subtiles for AIE layout
    ↓
BFP16 Shuffled Buffer (Eigen::Matrix<uint8_t>)
    ↓
[3] NPU Matmul (BFP16 @ BFP16 → BFP16)
    - npu_callback_fn_(input_bfp16, weight_bfp16, output_bfp16, M, K, N)
    - XCLBin kernel: matmul_bfp16_512x512x512.xclbin
    ↓
BFP16 Shuffled Buffer (Eigen::Matrix<uint8_t>)
    ↓
[4] BFP16Quantizer::unshuffle_bfp16()
    - Restore row-major layout
    ↓
BFP16 Buffer (Eigen::Matrix<uint8_t>)
    ↓
[5] BFP16Quantizer::convert_from_bfp16()
    - Extract block exponents
    - Dequantize mantissas to FP32
    ↓
Output (FP32)
```

### Key Functions

**File**: `cpp/src/bfp16_quantization.cpp`

1. **prepare_for_npu()** (replaces quantize_tensor):
   ```cpp
   void BFP16Quantizer::prepare_for_npu(
       const Eigen::MatrixXf& input,
       Eigen::Matrix<uint8_t, Dynamic, Dynamic>& output
   ) {
       // [1] Convert FP32 → BFP16
       Eigen::Matrix<uint8_t, Dynamic, Dynamic> bfp16_temp;
       convert_to_bfp16(input, bfp16_temp);

       // [2] Shuffle for NPU layout
       shuffle_bfp16(bfp16_temp, output, input.rows(), bfp16_temp.cols());
   }
   ```

2. **read_from_npu()** (replaces dequantize_matmul_output):
   ```cpp
   void BFP16Quantizer::read_from_npu(
       const Eigen::Matrix<uint8_t, Dynamic, Dynamic>& input,
       Eigen::MatrixXf& output,
       size_t rows,
       size_t cols
   ) {
       // [1] Unshuffle from NPU layout
       Eigen::Matrix<uint8_t, Dynamic, Dynamic> bfp16_unshuffled;
       unshuffle_bfp16(input, bfp16_unshuffled, rows, input.cols());

       // [2] Convert BFP16 → FP32
       convert_from_bfp16(bfp16_unshuffled, output, rows, cols);
   }
   ```

### Buffer Types

```cpp
// Weights (stored) - NO SCALES
Eigen::Matrix<uint8_t, Dynamic, Dynamic> q_weight_bfp16_;

// Activations (temporary)
Eigen::Matrix<uint8_t, Dynamic, Dynamic> input_bfp16_;
Eigen::Matrix<uint8_t, Dynamic, Dynamic> input_bfp16_shuffled_;
Eigen::Matrix<uint8_t, Dynamic, Dynamic> output_bfp16_shuffled_;
```

---

## File-by-File Changes

### File 1: `cpp/include/encoder_layer.hpp`

**Changes**: Update member variables and function signatures

#### Change 1.1: Include BFP16 header (line 6)

**BEFORE**:
```cpp
#include "quantization.hpp"
```

**AFTER**:
```cpp
#include "quantization.hpp"
#include "bfp16_quantization.hpp"
```

#### Change 1.2: Update weight buffer types (lines 157-170)

**BEFORE**:
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

**AFTER**:
```cpp
// BFP16 weights (no scales needed!)
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> q_weight_bfp16_;
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> k_weight_bfp16_;
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> v_weight_bfp16_;
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> out_weight_bfp16_;
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> fc1_weight_bfp16_;
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> fc2_weight_bfp16_;

// NO SCALES - embedded in block exponents
```

**Lines Removed**: 6 scale floats (24 bytes saved)
**Lines Added**: 0 (cleaner API!)

#### Change 1.3: Update activation buffer types (lines 193-195)

**BEFORE**:
```cpp
// Quantized buffers
Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> input_int8_;
Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> matmul_output_int32_;
```

**AFTER**:
```cpp
// BFP16 buffers
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> input_bfp16_shuffled_;
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> output_bfp16_shuffled_;
```

**Lines Removed**: 2
**Lines Added**: 2

#### Change 1.4: Update run_npu_linear signature (lines 206-212)

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

**Parameters Removed**: `weight_scale` (no longer needed)
**Type Changed**: `int8_t` → `uint8_t`

---

### File 2: `cpp/src/encoder_layer.cpp`

**Changes**: Update load_weights() and run_npu_linear()

#### Change 2.1: Update load_weights() (lines 40-59)

**BEFORE**:
```cpp
void EncoderLayer::load_weights(...) {
    // Quantize weights to INT8
    Quantizer quantizer;

    q_weight_scale_ = quantizer.compute_scale(q_weight);
    quantizer.quantize_tensor(q_weight, q_weight_int8_, q_weight_scale_);

    k_weight_scale_ = quantizer.compute_scale(k_weight);
    quantizer.quantize_tensor(k_weight, k_weight_int8_, k_weight_scale_);

    v_weight_scale_ = quantizer.compute_scale(v_weight);
    quantizer.quantize_tensor(v_weight, v_weight_int8_, v_weight_scale_);

    out_weight_scale_ = quantizer.compute_scale(out_weight);
    quantizer.quantize_tensor(out_weight, out_weight_int8_, out_weight_scale_);

    fc1_weight_scale_ = quantizer.compute_scale(fc1_weight);
    quantizer.quantize_tensor(fc1_weight, fc1_weight_int8_, fc1_weight_scale_);

    fc2_weight_scale_ = quantizer.compute_scale(fc2_weight);
    quantizer.quantize_tensor(fc2_weight, fc2_weight_int8_, fc2_weight_scale_);

    // Store biases (FP32) - UNCHANGED
    // ...
}
```

**AFTER**:
```cpp
void EncoderLayer::load_weights(...) {
    // Convert weights to BFP16 (with shuffle for NPU)
    BFP16Quantizer bfp16_quantizer;

    bfp16_quantizer.prepare_for_npu(q_weight, q_weight_bfp16_);
    bfp16_quantizer.prepare_for_npu(k_weight, k_weight_bfp16_);
    bfp16_quantizer.prepare_for_npu(v_weight, v_weight_bfp16_);
    bfp16_quantizer.prepare_for_npu(out_weight, out_weight_bfp16_);
    bfp16_quantizer.prepare_for_npu(fc1_weight, fc1_weight_bfp16_);
    bfp16_quantizer.prepare_for_npu(fc2_weight, fc2_weight_bfp16_);

    // Biases and layer norms stay FP32 (UNCHANGED)
    // ...
}
```

**Lines Changed**: 20 → 12 (40% reduction!)
**Complexity**: Much simpler (no scale management)

#### Change 2.2: Update run_attention() calls (lines 124-126, 133)

**BEFORE**:
```cpp
// Q/K/V projections (NPU matmuls)
run_npu_linear(ln_output_, q_weight_int8_, q_weight_scale_, q_bias_, Q_);
run_npu_linear(ln_output_, k_weight_int8_, k_weight_scale_, k_bias_, K_);
run_npu_linear(ln_output_, v_weight_int8_, v_weight_scale_, v_bias_, V_);

// ...

// Output projection (NPU matmul)
run_npu_linear(attn_heads, out_weight_int8_, out_weight_scale_, out_bias_, output);
```

**AFTER**:
```cpp
// Q/K/V projections (NPU matmuls with BFP16)
run_npu_linear(ln_output_, q_weight_bfp16_, q_bias_, Q_);
run_npu_linear(ln_output_, k_weight_bfp16_, k_bias_, K_);
run_npu_linear(ln_output_, v_weight_bfp16_, v_bias_, V_);

// ...

// Output projection (NPU matmul with BFP16)
run_npu_linear(attn_heads, out_weight_bfp16_, out_bias_, output);
```

**Lines Changed**: 4 (remove scale parameter)

#### Change 2.3: Update run_ffn() calls (lines 154, 160)

**BEFORE**:
```cpp
// FC1: (seq_len, n_state) @ (n_state, ffn_dim) -> (seq_len, ffn_dim)
run_npu_linear(ln_output_, fc1_weight_int8_, fc1_weight_scale_, fc1_bias_, fc1_output_);

// ...

// FC2: (seq_len, ffn_dim) @ (ffn_dim, n_state) -> (seq_len, n_state)
run_npu_linear(fc1_output_, fc2_weight_int8_, fc2_weight_scale_, fc2_bias_, output);
```

**AFTER**:
```cpp
// FC1: (seq_len, n_state) @ (n_state, ffn_dim) -> (seq_len, ffn_dim)
run_npu_linear(ln_output_, fc1_weight_bfp16_, fc1_bias_, fc1_output_);

// ...

// FC2: (seq_len, ffn_dim) @ (ffn_dim, n_state) -> (seq_len, n_state)
run_npu_linear(fc1_output_, fc2_weight_bfp16_, fc2_bias_, output);
```

**Lines Changed**: 2 (remove scale parameter)

#### Change 2.4: Rewrite run_npu_linear() (lines 163-223)

**BEFORE** (INT8):
```cpp
void EncoderLayer::run_npu_linear(
    const Eigen::MatrixXf& input,
    const Eigen::Matrix<int8_t, Dynamic, Dynamic>& weight_int8,
    float weight_scale,
    const Eigen::VectorXf& bias,
    Eigen::MatrixXf& output
) {
    const int M = input.rows();
    const int K = input.cols();
    const int N = weight_int8.rows();  // Weight is transposed: (N, K)

    // Quantize input
    Quantizer quantizer;
    float input_scale = quantizer.compute_scale(input);

    if (input_int8_.rows() != M || input_int8_.cols() != K) {
        input_int8_.resize(M, K);
    }
    quantizer.quantize_tensor(input, input_int8_, input_scale);

    // Allocate output buffer
    if (matmul_output_int32_.rows() != M || matmul_output_int32_.cols() != N) {
        matmul_output_int32_.resize(M, N);
    }

    // Run NPU matmul: C_int32 = A_int8 @ B_int8^T
    if (npu_callback_fn_) {
        // NPU path (via Python callback)
        typedef int (*NPUCallback)(void*, const int8_t*, const int8_t*, int32_t*, size_t, size_t, size_t);
        auto callback = reinterpret_cast<NPUCallback>(npu_callback_fn_);

        int result = callback(
            npu_user_data_,
            input_int8_.data(),
            weight_int8.data(),
            matmul_output_int32_.data(),
            M, K, N
        );

        if (result != 0) {
            throw std::runtime_error("NPU callback failed");
        }
    } else if (npu_matmul_fn_) {
        // NPU path (C++ std::function)
        npu_matmul_fn_(input_int8_, weight_int8, matmul_output_int32_);
    } else {
        // CPU fallback (for testing without NPU)
        matmul_output_int32_ = (input_int8_.cast<int32_t>() * weight_int8.transpose().cast<int32_t>());
    }

    // Dequantize: C_fp32 = C_int32 * scale_A * scale_B
    if (output.rows() != M || output.cols() != N) {
        output.resize(M, N);
    }
    quantizer.dequantize_matmul_output(matmul_output_int32_, output, input_scale, weight_scale);

    // Add bias
    for (int i = 0; i < M; ++i) {
        output.row(i) += bias.transpose();
    }
}
```

**AFTER** (BFP16):
```cpp
void EncoderLayer::run_npu_linear(
    const Eigen::MatrixXf& input,
    const Eigen::Matrix<uint8_t, Dynamic, Dynamic>& weight_bfp16,
    const Eigen::VectorXf& bias,
    Eigen::MatrixXf& output
) {
    const int M = input.rows();
    const int K = input.cols();
    const int N = weight_bfp16.rows();  // Weight is (N, K_bfp16) in BFP16

    BFP16Quantizer bfp16_quantizer;

    // [1] Convert input FP32 → BFP16 and shuffle
    bfp16_quantizer.prepare_for_npu(input, input_bfp16_shuffled_);

    // [2] Allocate output buffer (BFP16, shuffled)
    // Calculate size: M × N values → M × (N * 1.125 bytes)
    const int output_cols_bfp16 = BFP16Quantizer::calculate_bfp16_cols(N);
    if (output_bfp16_shuffled_.rows() != M ||
        output_bfp16_shuffled_.cols() != output_cols_bfp16) {
        output_bfp16_shuffled_.resize(M, output_cols_bfp16);
    }

    // [3] Run NPU matmul: BFP16 @ BFP16 → BFP16
    if (npu_callback_fn_) {
        typedef int (*NPUCallbackBFP16)(
            void*, const uint8_t*, const uint8_t*, uint8_t*,
            size_t, size_t, size_t
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
        // BFP16 does not support C++ std::function path
        throw std::runtime_error("BFP16 requires Python NPU callback");
    } else {
        // CPU fallback: Not implemented for BFP16
        throw std::runtime_error("BFP16 CPU fallback not available");
    }

    // [4] Convert output BFP16 → FP32 (with unshuffle)
    bfp16_quantizer.read_from_npu(output_bfp16_shuffled_, output, M, N);

    // [5] Add bias (FP32)
    for (int i = 0; i < M; i++) {
        output.row(i) += bias.transpose();
    }
}
```

**Key Changes**:
- Remove `weight_scale` parameter
- Change types: `int8_t` → `uint8_t`, `int32_t` → `uint8_t`
- Add shuffle before NPU call
- Add unshuffle after NPU call
- Update callback signature
- Remove CPU fallback (BFP16 requires NPU)

**Lines Changed**: ~60 lines rewritten

---

## Function Mapping

### Quantization Functions

| INT8 Function | BFP16 Function | Notes |
|---------------|----------------|-------|
| `compute_scale()` | *N/A* | No scales needed (embedded in block exponents) |
| `quantize_tensor()` | `convert_to_bfp16()` | Block-based quantization |
| `quantize_tensor_with_scale()` | *N/A* | No pre-computed scales |
| `dequantize_matmul_output()` | `convert_from_bfp16()` | Output is BFP16, not INT32 |
| `dequantize_tensor()` | `convert_from_bfp16()` | Same function for all BFP16 → FP32 |
| *N/A* | `shuffle_bfp16()` | **NEW**: Shuffle for NPU layout |
| *N/A* | `unshuffle_bfp16()` | **NEW**: Unshuffle from NPU layout |
| *N/A* | `prepare_for_npu()` | **NEW**: All-in-one convert + shuffle |
| *N/A* | `read_from_npu()` | **NEW**: All-in-one unshuffle + convert |

### High-Level API Changes

| Operation | INT8 API | BFP16 API |
|-----------|----------|-----------|
| **Load weights** | `quantize_tensor(weight, weight_int8, scale)` | `prepare_for_npu(weight, weight_bfp16)` |
| **Convert input** | `quantize_tensor(input, input_int8, scale)` | `prepare_for_npu(input, input_bfp16_shuffled)` |
| **NPU matmul** | `matmul(A_int8, B_int8) → C_int32` | `matmul(A_bfp16, B_bfp16) → C_bfp16` |
| **Convert output** | `dequantize_matmul_output(C_int32, output, scale_A, scale_B)` | `read_from_npu(C_bfp16_shuffled, output, M, N)` |

---

## Buffer Size Changes

### INT8 Buffer Sizes

```cpp
// 512×512 matrix
int M = 512, K = 512;

// INT8 storage
size_t input_int8_bytes = M * K * sizeof(int8_t) = 512 * 512 * 1 = 262,144 bytes
size_t weight_int8_bytes = K * N * sizeof(int8_t) = 512 * 512 * 1 = 262,144 bytes
size_t output_int32_bytes = M * N * sizeof(int32_t) = 512 * 512 * 4 = 1,048,576 bytes

// Plus scales
size_t scale_bytes = 2 * sizeof(float) = 8 bytes

// Total per matmul: 1,572,872 bytes (1.5 MB)
```

### BFP16 Buffer Sizes

```cpp
// 512×512 matrix
int M = 512, K = 512;

// BFP16 storage (1.125 bytes per value)
size_t input_bfp16_bytes = M * K * 1.125 = 512 * 512 * 1.125 = 294,912 bytes
size_t weight_bfp16_bytes = K * N * 1.125 = 512 * 512 * 1.125 = 294,912 bytes
size_t output_bfp16_bytes = M * N * 1.125 = 512 * 512 * 1.125 = 294,912 bytes

// NO SCALES

// Total per matmul: 884,736 bytes (864 KB)
```

### Memory Savings

**Per Matmul**:
- INT8: 1.5 MB
- BFP16: 864 KB
- **Savings: 43% less memory** (1.5 MB → 864 KB)

**Per Layer** (6 matmuls: Q/K/V/Out/FC1/FC2):
- INT8: 9 MB + 24 bytes (scales)
- BFP16: 5.2 MB (no scales)
- **Savings: 42% less memory**

**Total Encoder** (6 layers):
- INT8: 54 MB + 144 bytes (scales)
- BFP16: 31 MB (no scales)
- **Savings: 43% less memory**

---

## NPU Callback Changes

### INT8 Callback Signature

```cpp
typedef int (*NPUCallbackINT8)(
    void* user_data,      // NPU runtime object
    const int8_t* A,      // Input (M × K)
    const int8_t* B,      // Weight (N × K, transposed)
    int32_t* C,           // Output (M × N)
    size_t M,
    size_t K,
    size_t N
);
```

### BFP16 Callback Signature

```cpp
typedef int (*NPUCallbackBFP16)(
    void* user_data,      // NPU runtime object
    const uint8_t* A,     // Input (M × K_bfp16, shuffled)
    const uint8_t* B,     // Weight (N × K_bfp16, shuffled)
    uint8_t* C,           // Output (M × N_bfp16, shuffled)
    size_t M,             // Rows (FP32 count, not BFP16 bytes)
    size_t K,             // Cols (FP32 count, not BFP16 bytes)
    size_t N              // Cols (FP32 count, not BFP16 bytes)
);
```

**Key Differences**:
- Pointer types: `int8_t*` → `uint8_t*`, `int32_t*` → `uint8_t*`
- Data layout: Shuffled BFP16 (not row-major)
- Dimensions: Still FP32 counts (M, K, N), not byte sizes
- Kernel: Uses `matmul_bfp16_512x512x512.xclbin`, not `matmul_4tile_int8.xclbin`

### Python Runtime Changes

**File**: `runtime/npu_runtime.py`

**BEFORE** (INT8):
```python
def matmul_int8(self, A_int8, B_int8):
    """INT8 matmul: A (M×K) @ B (K×N) → C (M×N)"""
    # Write INT8 buffers
    self.buffers[3].write(A_int8.astype(np.int8))
    self.buffers[4].write(B_int8.astype(np.int8))

    # Execute
    self.app.run()

    # Read INT32 output
    C_int32 = self.buffers[5].read().astype(np.int32)
    return C_int32
```

**AFTER** (BFP16):
```python
def matmul_bfp16(self, A_bfp16, B_bfp16):
    """BFP16 matmul: A (M×K_bfp16) @ B (K×N_bfp16) → C (M×N_bfp16)"""
    # Write BFP16 buffers (already shuffled)
    self.buffers[3].write(A_bfp16.astype(np.uint8))
    self.buffers[4].write(B_bfp16.astype(np.uint8))

    # Execute
    self.app.run()

    # Read BFP16 output (shuffled)
    C_bfp16 = self.buffers[5].read().astype(np.uint8)
    return C_bfp16
```

**Changes**:
- Type: `int8` → `uint8`, `int32` → `uint8`
- Data: Expects shuffled BFP16 input/output
- Kernel: Uses `matmul_bfp16_512x512x512.xclbin`

---

## Testing Strategy

### Unit Tests

**File**: `cpp/test/test_bfp16_quantization.cpp`

1. **Test convert_to_bfp16()**:
   - Input: 512×512 random FP32 matrix
   - Output: 512×576 BFP16 matrix (1.125x size)
   - Validate: Output size correct, no crashes

2. **Test convert_from_bfp16()**:
   - Input: BFP16 matrix from test 1
   - Output: 512×512 FP32 matrix
   - Validate: Round-trip error < 1%

3. **Test shuffle/unshuffle**:
   - Input: BFP16 matrix
   - Output: Shuffled → Unshuffled
   - Validate: Identity transformation (shuffle then unshuffle = original)

4. **Test prepare_for_npu() / read_from_npu()**:
   - Input: 512×512 FP32 matrix
   - Process: FP32 → BFP16 → Shuffled → Unshuffled → FP32
   - Validate: Round-trip error < 1%

### Integration Tests

**File**: `test_encoder_layer_bfp16.py`

1. **Test single matmul**:
   - Input: 512×512 FP32 activations + 512×512 FP32 weights
   - Process: Convert to BFP16, run NPU matmul, convert back
   - Validate: Output accuracy > 99% vs FP32 baseline

2. **Test single layer**:
   - Input: 512×512 FP32 encoder input
   - Process: Full encoder layer with BFP16 NPU calls
   - Validate: Output accuracy > 99% vs PyTorch

3. **Test 6-layer encoder**:
   - Input: 1500×512 mel spectrogram
   - Process: Full 6-layer encoder with BFP16
   - Validate:
     - Output accuracy > 99% vs PyTorch
     - Latency 517-565 ms (18-20× realtime)
     - No crashes, no memory leaks

### Accuracy Validation

**File**: `test_accuracy_bfp16_vs_pytorch.py`

**Metrics**:
```python
def validate_accuracy(pytorch_output, bfp16_output):
    # Cosine similarity
    cosine_sim = np.dot(pytorch_output.flat, bfp16_output.flat) / \
                 (np.linalg.norm(pytorch_output) * np.linalg.norm(bfp16_output))

    # Relative error
    rel_error = np.abs(pytorch_output - bfp16_output).mean() / \
                np.abs(pytorch_output).mean()

    # Max error
    max_error = np.abs(pytorch_output - bfp16_output).max()

    print(f"Cosine similarity: {cosine_sim:.4f} (target: >0.99)")
    print(f"Relative error: {rel_error:.4%} (target: <1%)")
    print(f"Max error: {max_error:.4f} (target: <0.1)")

    assert cosine_sim > 0.99, "Accuracy too low"
    assert rel_error < 0.01, "Error too high"
```

**Expected Results**:
- **Cosine similarity**: > 99% (vs 64.6% for INT8)
- **Relative error**: < 1% (vs 7.7% for INT8)
- **Max error**: < 0.1 (vs 12% for INT8)

---

## Summary

### Files Modified

1. **cpp/include/bfp16_quantization.hpp** (NEW): 200 lines
2. **cpp/src/bfp16_quantization.cpp** (NEW): 250 lines
3. **cpp/include/encoder_layer.hpp**: ~15 lines changed
4. **cpp/src/encoder_layer.cpp**: ~80 lines changed
5. **runtime/npu_runtime.py**: ~30 lines changed

**Total**: ~575 lines (mostly new BFP16 quantizer)

### Key Improvements

1. **Simpler API**: No scale management (60 lines → 12 lines in load_weights)
2. **Higher Accuracy**: 99% vs 64.6% cosine similarity
3. **Less Memory**: 43% reduction (54 MB → 31 MB for 6 layers)
4. **Cleaner Code**: Block-based quantization is more intuitive

### Performance Impact

**Expected** (based on BFP16_INTEGRATION_ROADMAP.md):
- **Latency**: 470 ms → 517-565 ms (10-20% slower)
- **Realtime Factor**: 21.79× → 18-20× (still above 17× minimum)
- **Accuracy**: 64.6% → >99% (+34.4% improvement)

**Trade-off**: Slight performance loss for massive accuracy gain

---

**Next Steps**: See `PHASE2_CHECKLIST.md` for implementation tasks

---

**Document Version**: 1.0
**Last Updated**: October 30, 2025
**Status**: Ready for Implementation
