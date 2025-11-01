# Phase 2 Implementation Checklist

**Date**: October 30, 2025
**Project**: Unicorn-Amanuensis XDNA2 BFP16 Integration
**Estimated Duration**: 6-8 hours
**Status**: Ready to Start

---

## Overview

This checklist tracks the implementation of Phase 2: Quantization Layer Updates for BFP16 integration.

**Goal**: Replace INT8 quantization with BFP16 block floating point format
**Success Criteria**: All tests pass, accuracy >99%, encoder compiles

---

## Pre-Implementation Checklist

- [x] Phase 1 complete (BFP16 converter functions implemented)
- [x] BFP16_INTEGRATION_ROADMAP.md reviewed
- [x] PHASE2_CONVERSION_PLAN.md reviewed
- [x] Reference files reviewed (mm_bfp.cc, BFP16_FORMAT.md)
- [x] Git branch created: `feature/phase2-bfp16-quantization`
- [x] Backup created: `cpp/src/quantization_int8_backup.cpp`

---

## Implementation Tasks

### Task 1: Create BFP16 Quantizer Implementation (3-4 hours)

**File**: `cpp/src/bfp16_quantization.cpp` (STUB ALREADY CREATED)

#### Subtask 1.1: Implement find_block_exponent()
- [ ] Extract max absolute value from 8-value block
- [ ] Extract FP32 exponent bits (23-30)
- [ ] Add BFP16 bias (127) and clamp to [0, 255]
- [ ] Handle edge case: all zeros (exponent = 0)
- [ ] Test with: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
- [ ] Expected exponent: ~130 (max value 8.0 → exp = 3 + 127)

**Reference**: BFP16_FORMAT.md, section "Find Block Exponent"

#### Subtask 1.2: Implement quantize_to_8bit_mantissa()
- [ ] Extract sign, exponent, mantissa from FP32 (bit manipulation)
- [ ] Calculate relative exponent (value_exp - block_exp)
- [ ] Shift mantissa based on relative exponent
- [ ] Extract top 7 bits (+ sign bit)
- [ ] Clamp to [-127, 127]
- [ ] Test with: value=4.0, block_exp=130
- [ ] Expected mantissa: ~64 (4.0 = 0x40800000 → mantissa ~0.5)

**Reference**: BFP16_FORMAT.md, section "Quantize Mantissas"

#### Subtask 1.3: Implement dequantize_from_8bit_mantissa()
- [ ] Extract sign from mantissa (bit 7)
- [ ] Calculate FP32 value: sign × mantissa × 2^(block_exp - 7 - 127)
- [ ] Test round-trip: quantize then dequantize
- [ ] Expected error: < 1% for typical values

**Reference**: Inverse of quantize_to_8bit_mantissa()

#### Subtask 1.4: Test convert_to_bfp16() / convert_from_bfp16()
- [ ] Create 512×512 random FP32 matrix
- [ ] Convert to BFP16 (expect 512×576 output)
- [ ] Convert back to FP32 (expect 512×512 output)
- [ ] Measure round-trip error (expect < 1%)
- [ ] Test edge cases: zeros, small values, large values

**Test Command**:
```bash
cd cpp/build
./test_bfp16_quantization --gtest_filter=*RoundTrip*
```

#### Subtask 1.5: Validate shuffle/unshuffle
- [ ] Create 512×576 BFP16 matrix (row-major)
- [ ] Shuffle for NPU layout
- [ ] Unshuffle back to row-major
- [ ] Verify identity: shuffle → unshuffle = original
- [ ] Visual inspection: check subtile arrangement

**Test Command**:
```bash
./test_bfp16_quantization --gtest_filter=*Shuffle*
```

**Completion Criteria**:
- [ ] All BFP16 conversion tests pass
- [ ] Round-trip error < 1%
- [ ] Shuffle is identity transformation
- [ ] No memory leaks (valgrind clean)

---

### Task 2: Update encoder_layer.hpp (30 minutes)

**File**: `cpp/include/encoder_layer.hpp`

#### Subtask 2.1: Add BFP16 header include
- [ ] Line 6: Add `#include "bfp16_quantization.hpp"`
- [ ] Verify: Header compiles without errors

#### Subtask 2.2: Update weight buffer declarations (lines 157-170)
- [ ] Remove 6 INT8 weight buffers (q/k/v/out/fc1/fc2)
- [ ] Remove 6 scale floats
- [ ] Add 6 BFP16 weight buffers (uint8_t type)
- [ ] Add comment: "// NO SCALES - embedded in block exponents"

**Before**:
```cpp
Eigen::Matrix<int8_t, Dynamic, Dynamic> q_weight_int8_;
float q_weight_scale_;
// ... (6 weights × 2 fields)
```

**After**:
```cpp
Eigen::Matrix<uint8_t, Dynamic, Dynamic> q_weight_bfp16_;
// ... (6 weights, no scales)
```

#### Subtask 2.3: Update activation buffer declarations (lines 193-195)
- [ ] Remove `input_int8_` and `matmul_output_int32_`
- [ ] Add `input_bfp16_shuffled_` and `output_bfp16_shuffled_`
- [ ] Verify: All buffer types are uint8_t

#### Subtask 2.4: Update run_npu_linear() signature (lines 206-212)
- [ ] Remove `weight_scale` parameter
- [ ] Change weight type: `int8_t` → `uint8_t`
- [ ] Update function comment to mention BFP16

**Before**:
```cpp
void run_npu_linear(
    const Eigen::MatrixXf& input,
    const Eigen::Matrix<int8_t, Dynamic, Dynamic>& weight_int8,
    float weight_scale,
    const Eigen::VectorXf& bias,
    Eigen::MatrixXf& output
);
```

**After**:
```cpp
void run_npu_linear(
    const Eigen::MatrixXf& input,
    const Eigen::Matrix<uint8_t, Dynamic, Dynamic>& weight_bfp16,
    const Eigen::VectorXf& bias,
    Eigen::MatrixXf& output
);
```

**Completion Criteria**:
- [ ] encoder_layer.hpp compiles without errors
- [ ] No INT8 types remaining in encoder layer
- [ ] No scale members remaining

---

### Task 3: Update encoder_layer.cpp (2-3 hours)

**File**: `cpp/src/encoder_layer.cpp`

#### Subtask 3.1: Update load_weights() (lines 40-59)
- [ ] Remove `Quantizer quantizer;`
- [ ] Add `BFP16Quantizer bfp16_quantizer;`
- [ ] Replace 6 quantize_tensor() calls with prepare_for_npu() calls
- [ ] Remove scale computation and storage
- [ ] Verify: Biases and layer norms unchanged (still FP32)

**Before** (20 lines):
```cpp
Quantizer quantizer;
q_weight_scale_ = quantizer.compute_scale(q_weight);
quantizer.quantize_tensor(q_weight, q_weight_int8_, q_weight_scale_);
// ... (5 more weights)
```

**After** (6 lines):
```cpp
BFP16Quantizer bfp16_quantizer;
bfp16_quantizer.prepare_for_npu(q_weight, q_weight_bfp16_);
// ... (5 more weights)
```

**Test Command**:
```bash
# Test weight loading
./test_encoder_layer_bfp16 --gtest_filter=*LoadWeights*
```

#### Subtask 3.2: Update run_attention() calls (lines 124-126, 133)
- [ ] Line 124: Remove `q_weight_scale_` parameter
- [ ] Line 125: Remove `k_weight_scale_` parameter
- [ ] Line 126: Remove `v_weight_scale_` parameter
- [ ] Line 133: Remove `out_weight_scale_` parameter
- [ ] Change weight buffer names: `*_int8_` → `*_bfp16_`

**Before**:
```cpp
run_npu_linear(ln_output_, q_weight_int8_, q_weight_scale_, q_bias_, Q_);
```

**After**:
```cpp
run_npu_linear(ln_output_, q_weight_bfp16_, q_bias_, Q_);
```

#### Subtask 3.3: Update run_ffn() calls (lines 154, 160)
- [ ] Line 154: Remove `fc1_weight_scale_` parameter
- [ ] Line 160: Remove `fc2_weight_scale_` parameter
- [ ] Change weight buffer names: `*_int8_` → `*_bfp16_`

#### Subtask 3.4: Rewrite run_npu_linear() (lines 163-223)
- [ ] Remove `weight_scale` parameter from function
- [ ] Change `weight_int8` to `weight_bfp16` (type: uint8_t)
- [ ] Remove `Quantizer quantizer;`
- [ ] Add `BFP16Quantizer bfp16_quantizer;`
- [ ] Replace quantize_tensor() with prepare_for_npu()
- [ ] Update NPU callback signature (int8_t* → uint8_t*)
- [ ] Remove dequantize_matmul_output() call
- [ ] Add read_from_npu() call
- [ ] Update buffer resize logic (1.125x size for BFP16)
- [ ] Remove CPU fallback (BFP16 requires NPU)

**Key changes**:
```cpp
// OLD: Quantize input to INT8
quantizer.quantize_tensor(input, input_int8_, input_scale);

// NEW: Convert input to BFP16 and shuffle
bfp16_quantizer.prepare_for_npu(input, input_bfp16_shuffled_);

// OLD: Dequantize INT32 output
quantizer.dequantize_matmul_output(matmul_output_int32_, output,
                                   input_scale, weight_scale);

// NEW: Unshuffle and convert BFP16 output
bfp16_quantizer.read_from_npu(output_bfp16_shuffled_, output, M, N);
```

**Test Command**:
```bash
# Test single layer forward pass
./test_encoder_layer_bfp16 --gtest_filter=*SingleLayer*
```

**Completion Criteria**:
- [ ] encoder_layer.cpp compiles without errors
- [ ] No INT8 types or scale variables remaining
- [ ] All 6 NPU calls updated to BFP16
- [ ] Single-layer test passes (accuracy >99%)

---

### Task 4: Update CMakeLists.txt (15 minutes)

**File**: `cpp/CMakeLists.txt`

#### Subtask 4.1: Add BFP16 quantization source
- [ ] Add `src/bfp16_quantization.cpp` to `add_library(whisper_xdna2_lib ...)`
- [ ] Verify: Library builds successfully

**Before**:
```cmake
add_library(whisper_xdna2_lib
    src/quantization.cpp
    src/encoder_layer.cpp
    # ...
)
```

**After**:
```cmake
add_library(whisper_xdna2_lib
    src/quantization.cpp           # Keep for reference
    src/bfp16_quantization.cpp    # NEW
    src/encoder_layer.cpp
    # ...
)
```

#### Subtask 4.2: Add BFP16 tests
- [ ] Add `test/test_bfp16_quantization.cpp` to test executable
- [ ] Add `test/test_encoder_layer_bfp16.cpp` to test executable

**Test Command**:
```bash
cd cpp/build
cmake ..
make -j$(nproc)
```

**Completion Criteria**:
- [ ] CMakeLists.txt updated
- [ ] Project builds without errors
- [ ] Test executables link successfully

---

### Task 5: Create Unit Tests (1-2 hours)

**File**: `cpp/test/test_bfp16_quantization.cpp` (NEW)

#### Subtask 5.1: Test find_block_exponent()
- [ ] Test with [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
- [ ] Verify exponent is ~130 (max 8.0 → exp = 3 + 127)
- [ ] Test with all zeros → exponent = 0
- [ ] Test with very small values (1e-6) → low exponent
- [ ] Test with very large values (1e6) → high exponent

#### Subtask 5.2: Test quantize/dequantize round-trip
- [ ] Quantize value 4.0 with block_exp 130
- [ ] Dequantize back
- [ ] Verify error < 1%
- [ ] Test 1000 random values
- [ ] Measure average error, max error

#### Subtask 5.3: Test convert_to_bfp16()
- [ ] Create 512×512 random FP32 matrix (range [-1, 1])
- [ ] Convert to BFP16
- [ ] Verify output size: 512 × 576 (1.125x)
- [ ] Check no crashes
- [ ] Check reasonable values (not all zeros)

#### Subtask 5.4: Test convert_from_bfp16()
- [ ] Use BFP16 output from previous test
- [ ] Convert back to FP32
- [ ] Verify output size: 512 × 512
- [ ] Measure round-trip error (expect < 1%)
- [ ] Test cosine similarity (expect > 0.99)

#### Subtask 5.5: Test shuffle/unshuffle
- [ ] Create 512×576 BFP16 matrix
- [ ] Shuffle
- [ ] Unshuffle
- [ ] Verify: unshuffle(shuffle(x)) == x
- [ ] Check byte-by-byte equality

#### Subtask 5.6: Test prepare_for_npu() / read_from_npu()
- [ ] Create 512×512 FP32 input
- [ ] prepare_for_npu()
- [ ] read_from_npu()
- [ ] Verify round-trip error < 1%

**Test Command**:
```bash
cd cpp/build
./test_bfp16_quantization
```

**Expected Output**:
```
[==========] Running 6 tests
[ RUN      ] BFP16Test.FindBlockExponent
[       OK ] BFP16Test.FindBlockExponent (0 ms)
[ RUN      ] BFP16Test.QuantizeDequantize
[       OK ] BFP16Test.QuantizeDequantize (1 ms)
[ RUN      ] BFP16Test.ConvertToBFP16
[       OK ] BFP16Test.ConvertToBFP16 (50 ms)
[ RUN      ] BFP16Test.ConvertFromBFP16
[       OK ] BFP16Test.ConvertFromBFP16 (50 ms)
[ RUN      ] BFP16Test.ShuffleUnshuffle
[       OK ] BFP16Test.ShuffleUnshuffle (20 ms)
[ RUN      ] BFP16Test.PrepareReadNPU
[       OK ] BFP16Test.PrepareReadNPU (60 ms)
[==========] 6 tests passed (181 ms total)
```

**Completion Criteria**:
- [ ] All 6 tests pass
- [ ] Round-trip error < 1%
- [ ] No memory leaks (valgrind)

---

### Task 6: Create Integration Tests (1 hour)

**File**: `cpp/test/test_encoder_layer_bfp16.cpp` (NEW)

#### Subtask 6.1: Test load_weights() with BFP16
- [ ] Create 6 random FP32 weight matrices (512×512, 512×2048, etc.)
- [ ] Load weights via load_weights()
- [ ] Verify all 6 BFP16 weight buffers allocated
- [ ] Verify buffer sizes correct (1.125x)
- [ ] Verify no crashes

#### Subtask 6.2: Test run_npu_linear() (single matmul)
- [ ] Mock NPU callback (return dummy BFP16 output)
- [ ] Create 512×512 FP32 input
- [ ] Call run_npu_linear() with BFP16 weights
- [ ] Verify callback receives correct BFP16 pointers
- [ ] Verify output dimensions correct

#### Subtask 6.3: Test single layer forward pass
- [ ] Create 512×512 FP32 encoder input
- [ ] Run forward() (full layer with BFP16)
- [ ] Compare output vs FP32 baseline (CPU)
- [ ] Measure accuracy (cosine similarity)
- [ ] Expect > 99% accuracy

**Test Command**:
```bash
./test_encoder_layer_bfp16
```

**Expected Output**:
```
[==========] Running 3 tests
[ RUN      ] EncoderLayerBFP16Test.LoadWeights
[       OK ] EncoderLayerBFP16Test.LoadWeights (10 ms)
[ RUN      ] EncoderLayerBFP16Test.RunNPULinear
[       OK ] EncoderLayerBFP16Test.RunNPULinear (50 ms)
[ RUN      ] EncoderLayerBFP16Test.SingleLayerForward
         Cosine similarity: 0.9912
         Relative error: 0.67%
[       OK ] EncoderLayerBFP16Test.SingleLayerForward (200 ms)
[==========] 3 tests passed (260 ms total)
```

**Completion Criteria**:
- [ ] All integration tests pass
- [ ] Accuracy > 99%
- [ ] No crashes
- [ ] No memory leaks

---

## Verification Checklist

### Code Quality
- [ ] No compiler warnings (`-Wall -Wextra`)
- [ ] No memory leaks (valgrind clean)
- [ ] Code follows project style guide
- [ ] All TODOs removed from production code
- [ ] Comments updated to reflect BFP16 logic

### Functionality
- [ ] All unit tests pass (6/6)
- [ ] All integration tests pass (3/3)
- [ ] Round-trip error < 1%
- [ ] Cosine similarity > 99%
- [ ] Encoder layer compiles and links

### Performance
- [ ] Buffer allocations reasonable (~1.125x FP32 size)
- [ ] No unexpected memory overhead
- [ ] Conversion time < 5ms per 512×512 matrix
- [ ] Shuffle time < 2ms per 512×576 matrix

### Documentation
- [ ] All functions have docstrings
- [ ] PHASE2_CONVERSION_PLAN.md reflects actual changes
- [ ] README updated with BFP16 status
- [ ] Known issues documented

---

## Post-Implementation Checklist

### Testing
- [ ] Run full test suite: `make test`
- [ ] Run valgrind: `valgrind ./test_bfp16_quantization`
- [ ] Run coverage: `gcov bfp16_quantization.cpp`
- [ ] Test on real hardware (if available)

### Documentation
- [ ] Update BFP16_INTEGRATION_ROADMAP.md (mark Phase 2 complete)
- [ ] Create `PHASE2_COMPLETE.md` with results
- [ ] Document any deviations from plan
- [ ] List known issues or limitations

### Git Workflow
- [ ] Commit changes: `git commit -m "feat: Phase 2 BFP16 quantization complete"`
- [ ] Push branch: `git push origin feature/phase2-bfp16-quantization`
- [ ] Create pull request
- [ ] Code review by team
- [ ] Merge to main

### Next Steps
- [ ] Begin Phase 3: Update encoder_layer for full BFP16 pipeline
- [ ] Compile BFP16 NPU kernels (matmul_bfp16_512x512x512.xclbin)
- [ ] Update Python NPU runtime for BFP16 callbacks
- [ ] End-to-end accuracy validation vs PyTorch

---

## Common Issues and Solutions

### Issue 1: Round-trip error > 1%

**Symptoms**: convert_to_bfp16 → convert_from_bfp16 has error > 1%

**Possible Causes**:
- Incorrect exponent calculation
- Mantissa shifting bugs
- Sign bit handling

**Solution**:
- Add debug prints to see intermediate values
- Test with simple inputs (1.0, 2.0, 4.0, 8.0)
- Compare vs reference Python implementation

### Issue 2: Shuffle is not identity transformation

**Symptoms**: shuffle → unshuffle ≠ original

**Possible Causes**:
- Incorrect subtile size (should be 8×9)
- Off-by-one errors in indexing
- Bounds checking issues

**Solution**:
- Add visual debugging (print first 72 bytes)
- Compare vs mm_bfp.cc reference implementation
- Test with small matrices (64×72 first)

### Issue 3: Encoder layer compilation errors

**Symptoms**: Cannot find BFP16Quantizer, type mismatches

**Possible Causes**:
- Missing #include "bfp16_quantization.hpp"
- Type mismatch (int8_t vs uint8_t)
- Missing const qualifiers

**Solution**:
- Check header includes
- Verify all INT8 types changed to uint8_t
- Review compiler error messages carefully

### Issue 4: NPU callback crashes

**Symptoms**: Segfault or bus error in NPU callback

**Possible Causes**:
- Incorrect pointer types (int8_t* vs uint8_t*)
- Wrong buffer sizes
- Uninitialized pointers

**Solution**:
- Add null pointer checks
- Verify buffer sizes match expectations
- Test with CPU fallback first (if available)

### Issue 5: Accuracy lower than expected

**Symptoms**: Cosine similarity < 99%

**Possible Causes**:
- BFP16 conversion bugs
- Shuffle/unshuffle bugs
- NPU kernel issues

**Solution**:
- Test each step independently
- Compare intermediate values vs FP32 baseline
- Verify NPU kernel is using correct matmul_bfp16_*.xclbin

---

## Time Tracking

| Task | Estimated | Actual | Notes |
|------|-----------|--------|-------|
| Task 1: BFP16 Quantizer | 3-4h | ___ h | |
| Task 2: encoder_layer.hpp | 0.5h | ___ h | |
| Task 3: encoder_layer.cpp | 2-3h | ___ h | |
| Task 4: CMakeLists.txt | 0.25h | ___ h | |
| Task 5: Unit Tests | 1-2h | ___ h | |
| Task 6: Integration Tests | 1h | ___ h | |
| **Total** | **6-8h** | **___ h** | |

---

## Sign-Off

- [ ] **Implementation Complete**: All tasks checked off
- [ ] **Tests Passing**: All unit and integration tests pass
- [ ] **Documentation Complete**: All docs updated
- [ ] **Code Review**: Reviewed by at least 1 team member
- [ ] **Ready for Phase 3**: Quantization layer ready for encoder integration

**Completed By**: _______________
**Date**: _______________
**Reviewer**: _______________
**Review Date**: _______________

---

**Document Version**: 1.0
**Last Updated**: October 30, 2025
**Status**: Ready to Start
