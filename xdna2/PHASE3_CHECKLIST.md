# Phase 3 Checklist: Encoder BFP16 Integration

**Project**: Unicorn-Amanuensis XDNA2
**Phase**: 3 of 5
**Date**: October 30, 2025
**Estimated Duration**: 8-12 hours

---

## Quick Start

```bash
# Navigate to project
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2

# Read detailed plan
cat PHASE3_IMPLEMENTATION_PLAN.md

# Start with Task 1
vim cpp/include/encoder_layer.hpp
```

---

## Task 1: Update encoder_layer.hpp (2-3 hours)

### Subtask 1.1: Update Includes (5 min)
- [ ] Open `cpp/include/encoder_layer.hpp`
- [ ] Add `#include "bfp16_quantization.hpp"` after line 5
- [ ] Verify file compiles: `cd cpp/build && make encoder_layer -j4`

**Expected**:
```cpp
#include "bfp16_quantization.hpp"  // NEW
#include "quantization.hpp"         // Optional: keep for legacy
```

### Subtask 1.2: Update Weight Storage (30 min)
- [ ] Navigate to lines 157-170
- [ ] Delete all 6 `_int8_` weight members
- [ ] Delete all 6 `_scale_` members (floats)
- [ ] Add 6 `_bfp16_` weight members (uint8_t)
- [ ] Add comment explaining BFP16 storage format
- [ ] Verify: 12 lines deleted, 6 lines added

**Before** (lines 157-170):
```cpp
Eigen::Matrix<int8_t, ...> q_weight_int8_;
// ... 5 more int8 weights
float q_weight_scale_;
// ... 5 more scales
```

**After**:
```cpp
// BFP16 weights (shuffled, ready for NPU)
Eigen::Matrix<uint8_t, ...> q_weight_bfp16_;
// ... 5 more bfp16 weights
```

### Subtask 1.3: Update Activation Buffers (20 min)
- [ ] Navigate to lines 193-195
- [ ] Delete `input_int8_` and `matmul_output_int32_`
- [ ] Add 4 BFP16 buffers: `input_bfp16_`, `input_bfp16_shuffled_`, `output_bfp16_shuffled_`, `output_bfp16_`
- [ ] Add comments explaining shuffle/unshuffle
- [ ] Verify: 2 lines deleted, 4 lines added

**After**:
```cpp
Eigen::Matrix<uint8_t, ...> input_bfp16_;          // Before shuffle
Eigen::Matrix<uint8_t, ...> input_bfp16_shuffled_; // After shuffle (NPU layout)
Eigen::Matrix<uint8_t, ...> output_bfp16_shuffled_; // From NPU (shuffled)
Eigen::Matrix<uint8_t, ...> output_bfp16_;         // After unshuffle
```

### Subtask 1.4: Update run_npu_linear() Signature (15 min)
- [ ] Navigate to lines 206-212
- [ ] Delete `float weight_scale` parameter (line 209)
- [ ] Change weight type: `int8_t` → `uint8_t` (line 208)
- [ ] Update comment to mention BFP16
- [ ] Verify: Signature has 4 parameters (was 5)

**After**:
```cpp
void run_npu_linear(
    const Eigen::MatrixXf& input,
    const Eigen::Matrix<uint8_t, ...>& weight_bfp16,  // Changed from int8_t
    const Eigen::VectorXf& bias,
    Eigen::MatrixXf& output
);
```

### Subtask 1.5: Add BFP16Quantizer Instance (5 min)
- [ ] Navigate to line 148 (inside private section)
- [ ] Add line: `BFP16Quantizer bfp16_quantizer_;`
- [ ] Verify: 1 line added

### Task 1 Verification
- [ ] File compiles without errors: `make encoder_layer -j4`
- [ ] No INT8 members remaining (grep check)
- [ ] No scale members remaining (grep check)
- [ ] BFP16 buffers added (4 buffers)
- [ ] `run_npu_linear()` signature updated

**Test Command**:
```bash
cd cpp/build
make encoder_layer -j4 2>&1 | tee task1_build.log
grep -n "error:" task1_build.log  # Should be empty
```

---

## Task 2: Update load_weights() (2-3 hours)

### Subtask 2.1: Replace Quantizer with BFP16Quantizer (30 min)
- [ ] Open `cpp/src/encoder_layer.cpp`
- [ ] Navigate to line 40 (inside `load_weights()`)
- [ ] Delete line: `Quantizer quantizer;`
- [ ] Verify: `bfp16_quantizer_` member will be used instead

### Subtask 2.2: Update Q Weight Conversion (15 min)
- [ ] Navigate to lines 43-44
- [ ] Delete 2 lines:
  ```cpp
  q_weight_scale_ = quantizer.compute_scale(q_weight);
  quantizer.quantize_tensor(q_weight, q_weight_int8_, q_weight_scale_);
  ```
- [ ] Add 1 line:
  ```cpp
  bfp16_quantizer_.prepare_for_npu(q_weight, q_weight_bfp16_);
  ```
- [ ] Verify: 2 lines → 1 line, no scale

### Subtask 2.3: Update K Weight Conversion (5 min)
- [ ] Navigate to lines 46-47
- [ ] Replace with: `bfp16_quantizer_.prepare_for_npu(k_weight, k_weight_bfp16_);`
- [ ] Verify: 2 lines → 1 line

### Subtask 2.4: Update V Weight Conversion (5 min)
- [ ] Navigate to lines 49-50
- [ ] Replace with: `bfp16_quantizer_.prepare_for_npu(v_weight, v_weight_bfp16_);`
- [ ] Verify: 2 lines → 1 line

### Subtask 2.5: Update Out Weight Conversion (5 min)
- [ ] Navigate to lines 52-53
- [ ] Replace with: `bfp16_quantizer_.prepare_for_npu(out_weight, out_weight_bfp16_);`
- [ ] Verify: 2 lines → 1 line

### Subtask 2.6: Update FC1 Weight Conversion (5 min)
- [ ] Navigate to lines 55-56
- [ ] Replace with: `bfp16_quantizer_.prepare_for_npu(fc1_weight, fc1_weight_bfp16_);`
- [ ] Verify: 2 lines → 1 line

### Subtask 2.7: Update FC2 Weight Conversion (5 min)
- [ ] Navigate to lines 58-59
- [ ] Replace with: `bfp16_quantizer_.prepare_for_npu(fc2_weight, fc2_weight_bfp16_);`
- [ ] Verify: 2 lines → 1 line

### Subtask 2.8: Verify Biases Unchanged (5 min)
- [ ] Navigate to lines 61-73
- [ ] Verify biases and layer norm params are unchanged (FP32)
- [ ] No modifications needed

### Task 2 Verification
- [ ] Total changes: 13 lines deleted, 6 lines added
- [ ] All weights use `prepare_for_npu()`
- [ ] No scale computations
- [ ] Biases remain FP32
- [ ] File compiles: `make encoder_layer -j4`

**Test Command**:
```bash
cd cpp/build
make encoder_layer -j4
grep "prepare_for_npu" ../src/encoder_layer.cpp | wc -l  # Should be 6
```

---

## Task 3: Rewrite run_npu_linear() (3-4 hours)

### Subtask 3.1: Update Function Signature (10 min)
- [ ] Navigate to line 163
- [ ] Update signature to match header (4 parameters, uint8_t weight)
- [ ] Update parameter names: `weight_int8` → `weight_bfp16`
- [ ] Remove `weight_scale` parameter
- [ ] Verify: Matches header exactly

### Subtask 3.2: Replace Quantize Input Section (30 min)
- [ ] Navigate to lines 174-181 (quantize input)
- [ ] Delete all lines (8 lines):
  - Quantizer instantiation
  - Scale computation
  - Buffer resize
  - quantize_tensor call
- [ ] Add new code (1 line):
  ```cpp
  bfp16_quantizer_.prepare_for_npu(input, input_bfp16_shuffled_);
  ```
- [ ] Verify: 8 lines → 1 line

### Subtask 3.3: Update Output Buffer Allocation (45 min)
- [ ] Navigate to lines 183-186 (output buffer)
- [ ] Replace with BFP16 buffer sizing:
  ```cpp
  // Calculate BFP16 output buffer size
  const int N_blocks = (N + BFP16Config::BLOCK_SIZE - 1) / BFP16Config::BLOCK_SIZE;
  const int N_bfp16_bytes = N_blocks * BFP16Config::BYTES_PER_BLOCK;

  if (output_bfp16_shuffled_.rows() != M ||
      output_bfp16_shuffled_.cols() != N_bfp16_bytes) {
      output_bfp16_shuffled_.resize(M, N_bfp16_bytes);
  }
  ```
- [ ] Verify: Uses 1.125× formula

### Subtask 3.4: Update NPU Callback Section (60 min)
- [ ] Navigate to lines 188-211 (NPU dispatch)
- [ ] Update callback typedef to BFP16:
  ```cpp
  typedef int (*NPUCallbackBFP16)(
      void*, const uint8_t*, const uint8_t*, uint8_t*,
      size_t, size_t, size_t
  );
  ```
- [ ] Update callback arguments:
  - `input_int8_.data()` → `input_bfp16_shuffled_.data()`
  - `weight_int8.data()` → `weight_bfp16.data()`
  - `matmul_output_int32_.data()` → `output_bfp16_shuffled_.data()`
- [ ] Remove CPU fallback (lines 209-210)
- [ ] Add error for BFP16 without NPU
- [ ] Verify: Callback signature matches Python

### Subtask 3.5: Replace Dequantize Output Section (30 min)
- [ ] Navigate to lines 213-217 (dequantize output)
- [ ] Delete all lines (5 lines):
  - Output resize
  - dequantize_matmul_output call
- [ ] Add new code (1 line):
  ```cpp
  bfp16_quantizer_.read_from_npu(output_bfp16_shuffled_, output, M, N);
  ```
- [ ] Verify: 5 lines → 1 line

### Subtask 3.6: Verify Bias Addition Unchanged (5 min)
- [ ] Navigate to lines 219-222 (bias addition)
- [ ] Verify code is unchanged (FP32 bias)
- [ ] No modifications needed

### Task 3 Verification
- [ ] Function signature matches header
- [ ] Input conversion: 8 lines → 1 line
- [ ] Output conversion: 5 lines → 1 line
- [ ] NPU callback uses uint8_t*
- [ ] No references to INT8 or scales
- [ ] File compiles: `make encoder_layer -j4`

**Test Command**:
```bash
cd cpp/build
make encoder_layer -j4
nm libwhisper_xdna2.a | grep "run_npu_linear"  # Verify symbol exists
```

---

## Task 4: Update Matmul Call Sites (1-2 hours)

### Subtask 4.1: Update Q Projection Call (10 min)
- [ ] Navigate to line 124
- [ ] Change:
  ```cpp
  run_npu_linear(ln_output_, q_weight_int8_, q_weight_scale_, q_bias_, Q_);
  ```
  To:
  ```cpp
  run_npu_linear(ln_output_, q_weight_bfp16_, q_bias_, Q_);
  ```
- [ ] Verify: Removed scale parameter, updated variable name

### Subtask 4.2: Update K Projection Call (5 min)
- [ ] Navigate to line 125
- [ ] Update: `k_weight_int8_` → `k_weight_bfp16_`, remove scale
- [ ] Verify: 4 parameters

### Subtask 4.3: Update V Projection Call (5 min)
- [ ] Navigate to line 126
- [ ] Update: `v_weight_int8_` → `v_weight_bfp16_`, remove scale
- [ ] Verify: 4 parameters

### Subtask 4.4: Update Out Projection Call (10 min)
- [ ] Navigate to line 133
- [ ] Update: `out_weight_int8_` → `out_weight_bfp16_`, remove scale
- [ ] Verify: 4 parameters

### Subtask 4.5: Update FC1 Call (10 min)
- [ ] Navigate to line 154
- [ ] Update: `fc1_weight_int8_` → `fc1_weight_bfp16_`, remove scale
- [ ] Verify: 4 parameters

### Subtask 4.6: Update FC2 Call (10 min)
- [ ] Navigate to line 160
- [ ] Update: `fc2_weight_int8_` → `fc2_weight_bfp16_`, remove scale
- [ ] Verify: 4 parameters

### Task 4 Verification
- [ ] All 6 call sites updated
- [ ] All use `_bfp16_` weights
- [ ] All have 4 parameters (no scale)
- [ ] No other changes to surrounding code
- [ ] File compiles: `make encoder_layer -j4`

**Test Command**:
```bash
cd cpp/build
make encoder_layer -j4
grep "run_npu_linear.*bfp16" ../src/encoder_layer.cpp | wc -l  # Should be 6
grep "weight_scale_" ../src/encoder_layer.cpp  # Should be empty (no matches)
```

---

## Task 5: Update Build System (15 minutes)

### Subtask 5.1: Verify CMakeLists.txt (5 min)
- [ ] Open `cpp/CMakeLists.txt`
- [ ] Verify `bfp16_quantization.cpp` is in sources list
- [ ] Verify `bfp16_converter.cpp` is in sources list
- [ ] No changes needed (should already be added from Phase 1)

### Subtask 5.2: Clean Build (5 min)
- [ ] Delete build directory: `rm -rf cpp/build`
- [ ] Recreate build: `mkdir cpp/build && cd cpp/build`
- [ ] Configure: `cmake .. -DCMAKE_BUILD_TYPE=Release`
- [ ] Build: `make -j4`
- [ ] Verify: No errors

### Subtask 5.3: Verify Linking (5 min)
- [ ] Check library: `ls -lh libwhisper_xdna2.a`
- [ ] Check symbols: `nm libwhisper_xdna2.a | grep BFP16`
- [ ] Verify: BFP16 symbols present

### Task 5 Verification
- [ ] Clean build succeeds
- [ ] All libraries link
- [ ] BFP16 symbols present
- [ ] No undefined references

**Test Command**:
```bash
cd cpp/build
make clean && make -j4 2>&1 | tee build.log
grep -E "error:|undefined" build.log  # Should be empty
```

---

## Task 6: Create Integration Test (2 hours)

### Subtask 6.1: Create Test File (10 min)
- [ ] Create `cpp/tests/test_encoder_layer_bfp16.cpp`
- [ ] Add includes:
  ```cpp
  #include "encoder_layer.hpp"
  #include "bfp16_quantization.hpp"
  #include <gtest/gtest.h>
  ```
- [ ] Add namespace: `using namespace whisper_xdna2;`

### Subtask 6.2: Test Weight Loading (30 min)
- [ ] Write `TEST(EncoderLayerBFP16, WeightLoadingBFP16)`
- [ ] Create random weights (512×512)
- [ ] Call `layer.load_weights(...)`
- [ ] Verify no exceptions thrown
- [ ] EXPECT_NO_THROW()

### Subtask 6.3: Test Single Forward Pass (45 min)
- [ ] Write `TEST(EncoderLayerBFP16, SingleLayerForward)`
- [ ] Create layer (8 heads, 512 state, 2048 ffn)
- [ ] Load weights
- [ ] Create mock NPU callback (returns zeros)
- [ ] Run `layer.forward(input, output)`
- [ ] Verify output shape (512, 512)

### Subtask 6.4: Test Buffer Sizes (30 min)
- [ ] Write `TEST(EncoderLayerBFP16, BFP16BufferSizes)`
- [ ] Create known input (512×512)
- [ ] Convert to BFP16
- [ ] Verify buffer size: 512 × 576 (1.125×)
- [ ] Test multiple dimensions

### Task 6 Verification
- [ ] Test file created (3 tests)
- [ ] Tests compile
- [ ] Tests use BFP16 weights
- [ ] Mock NPU callback works
- [ ] All tests pass (with mock)

**Test Command**:
```bash
cd cpp/build
make test_encoder_layer_bfp16 -j4
./test_encoder_layer_bfp16 --gtest_output=xml:test_results.xml
cat test_results.xml  # Verify all tests passed
```

---

## Task 7: Python Integration Test (1 hour)

### Subtask 7.1: Create Test Script (10 min)
- [ ] Create `test_cpp_bfp16_encoder.py`
- [ ] Add imports: `numpy`, `time`, `whisper_encoder_cpp`
- [ ] Add docstring explaining test

### Subtask 7.2: Test Weight Loading (20 min)
- [ ] Load real Whisper weights from `weights/whisper_base_fp32/`
- [ ] Create 6-layer encoder
- [ ] Call `encoder.load_layer_weights()` for each layer
- [ ] Verify no errors

### Subtask 7.3: Test Forward Pass (20 min)
- [ ] Create random input (512, 512)
- [ ] Setup mock NPU callback
- [ ] Run `encoder.forward(input)`
- [ ] Measure latency
- [ ] Verify output shape

### Subtask 7.4: Test Accuracy Placeholder (10 min)
- [ ] Add TODO for PyTorch comparison
- [ ] Placeholder for cosine similarity
- [ ] Comment: "Phase 4: Real NPU validation"

### Task 7 Verification
- [ ] Test script created
- [ ] Loads real weights
- [ ] Runs forward pass
- [ ] Measures latency
- [ ] Validates output shape

**Test Command**:
```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2
python3 test_cpp_bfp16_encoder.py
# Expected: Output shape validated, latency printed
```

---

## Overall Phase 3 Verification

### Code Quality
- [ ] All INT8 references removed
- [ ] All scale parameters removed
- [ ] No compiler warnings (-Wall -Wextra)
- [ ] No memory leaks (valgrind clean)
- [ ] Clean code (consistent style)

### Functionality
- [ ] 6 layers compile
- [ ] All 36 matmuls use BFP16
- [ ] Weights load without errors
- [ ] Forward pass executes
- [ ] Output shapes correct

### Testing
- [ ] C++ unit tests pass (3/3)
- [ ] Python integration test runs
- [ ] No crashes (mock NPU)
- [ ] Buffer sizes validated

### Documentation
- [ ] Code comments updated
- [ ] BFP16 format documented
- [ ] API changes noted
- [ ] Phase 3 completion report written

---

## Final Validation Commands

```bash
# Navigate to project
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2

# Clean build
rm -rf cpp/build
mkdir cpp/build && cd cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Run C++ tests
./test_encoder_layer_bfp16 --gtest_color=yes

# Run Python test
cd ..
python3 test_cpp_bfp16_encoder.py

# Check for INT8 residue (should be empty)
grep -r "int8_t" cpp/src/encoder_layer.cpp
grep -r "weight_scale" cpp/src/encoder_layer.cpp

# Check BFP16 usage (should find matches)
grep -r "bfp16" cpp/src/encoder_layer.cpp | wc -l  # Should be >10

# Memory check (optional, requires valgrind)
valgrind --leak-check=full ./test_encoder_layer_bfp16
```

---

## Time Tracking

| Task | Estimated | Actual | Notes |
|------|-----------|--------|-------|
| Task 1: encoder_layer.hpp | 2-3h | | |
| Task 2: load_weights() | 2-3h | | |
| Task 3: run_npu_linear() | 3-4h | | |
| Task 4: Call sites | 1-2h | | |
| Task 5: Build system | 15min | | |
| Task 6: C++ tests | 2h | | |
| Task 7: Python test | 1h | | |
| **TOTAL** | **8-12h** | | |

---

## Common Issues and Solutions

### Issue 1: Compiler Error "int8_t not found"
**Cause**: Didn't update weight type in header
**Solution**: Change all `int8_t` → `uint8_t` in encoder_layer.hpp

### Issue 2: Linker Error "undefined reference to BFP16Quantizer"
**Cause**: Missing bfp16_quantization.cpp in build
**Solution**: Check CMakeLists.txt, add source file

### Issue 3: Runtime Error "Buffer size mismatch"
**Cause**: Incorrect BFP16 buffer sizing (1.125× formula)
**Solution**: Use `BFP16Config::BYTES_PER_BLOCK` constant

### Issue 4: Segfault in run_npu_linear()
**Cause**: Uninitialized BFP16 buffers
**Solution**: Resize buffers before use, add bounds checks

### Issue 5: Test Failure "Output shape wrong"
**Cause**: Forgot to unshuffle NPU output
**Solution**: Verify `read_from_npu()` is called

---

## Next Steps (Phase 4)

After completing this checklist:

1. **Create Phase 3 Completion Report**
   - Document all changes made
   - List test results
   - Note any issues encountered
   - Provide recommendations for Phase 4

2. **Move to Phase 4: NPU Integration**
   - Compile BFP16 NPU kernels (XCLBin)
   - Update Python NPU callback
   - Test on real hardware
   - Measure actual latency

3. **Update MASTER_CHECKLIST.md**
   - Mark Phase 3 as complete
   - Update progress tracker
   - Set Phase 4 start date

---

**Document Version**: 1.0
**Last Updated**: October 30, 2025
**Status**: Ready to Use
**Estimated Completion**: 8-12 hours

**Built with 💪 by Team BRO**
