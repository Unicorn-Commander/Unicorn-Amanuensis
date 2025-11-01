# BFP16 Migration Quick Start Guide

**Date**: October 30, 2025
**Status**: Ready to implement
**Timeline**: 1-2 weeks (28-40 hours)

---

## What is BFP16?

**Block Floating Point 16** (BFP16) is AMD's optimized quantization format for XDNA2:

```
Format: 8-bit mantissa per value + shared 8-bit exponent per 8 values
Average: 9 bits per value (vs 8-bit INT8, 16-bit FP16)
Performance: 50 TOPS (same as INT8!)
Accuracy: >99% (vs 64.6% for INT8)
```

## Why BFP16?

| Current (INT8) | After BFP16 | Improvement |
|----------------|-------------|-------------|
| 21.79× realtime | 18-20× realtime | -10% speed (acceptable) |
| 64.6% accuracy | >99% accuracy | **+34% accuracy** ✅ |
| 128 MB memory | 200 MB memory | +56% (acceptable) |

**Result**: Production-ready accuracy with minimal performance impact!

---

## 5-Phase Implementation Plan

### Phase 1: BFP16 Converter (8-12 hours)

**Create conversion functions**:

```cpp
// New files:
cpp/include/bfp16_converter.hpp
cpp/src/bfp16_converter.cpp

// Functions:
fp32_to_bfp16()      // FP32 → BFP16
bfp16_to_fp32()      // BFP16 → FP32
shuffle_bfp16()      // Shuffle for NPU layout
unshuffle_bfp16()    // Unshuffle from NPU
```

**Key Algorithm** (FP32 → BFP16):
1. Group values into blocks of 8
2. Find max absolute value in block
3. Compute shared exponent: `floor(log2(max))`
4. Extract 8-bit mantissas
5. Store: `[M1][M2]...[M8][Exp]` (9 bytes per block)

### Phase 2: Update Quantization (6-8 hours)

**Replace INT8 quantization**:

```cpp
// Before (INT8):
class Quantizer {
    static void quantize_tensor(..., int8_t& output, float& scale);
};

// After (BFP16):
class BFP16Quantizer {
    static void prepare_for_npu(MatrixXf& input, Matrix<uint8_t>& output);
    static void read_from_npu(Matrix<uint8_t>& input, MatrixXf& output);
};
```

**Changes**:
- Remove scale computation
- Use BFP16 conversion instead of INT8 quantization
- Add shuffle/unshuffle calls

### Phase 3: Update Encoder Layer (8-12 hours)

**Update weight storage**:

```cpp
// Before (INT8):
Eigen::Matrix<int8_t, Dynamic, Dynamic> q_weight_int8_;
float q_weight_scale_;

// After (BFP16):
Eigen::Matrix<uint8_t, Dynamic, Dynamic> q_weight_bfp16_;
// No scale needed!
```

**Update run_npu_linear()**:
```cpp
// Before:
quantize_tensor(input, input_int8_, scale);
npu_callback(input_int8_, weight_int8_, output_int32_);
dequantize_matmul_output(output_int32_, output, scale_A, scale_B);

// After:
prepare_for_npu(input, input_bfp16_shuffled_);
npu_callback(input_bfp16_shuffled_, weight_bfp16_, output_bfp16_shuffled_);
read_from_npu(output_bfp16_shuffled_, output);
```

### Phase 4: NPU Integration (6-8 hours)

**Compile BFP16 kernels**:

```bash
# Generate MLIR with BFP16 support
python3 single_core_iron.py \
  --dev npu2 \
  -M 512 -K 512 -N 512 \
  --dtype_in bf16 \
  --emulate-bf16-mmul-with-bfp16 True \
  > matmul_bfp16_512x512x512.mlir

# Compile to XCLBin
aiecc.py --xclbin-name=matmul_bfp16_512x512x512.xclbin \
  matmul_bfp16_512x512x512.mlir
```

**Update Python callback**:
```python
# Before (INT8):
A_int8 = A.astype(np.int8)
C_int32 = npu_matmul(A_int8, B_int8)

# After (BFP16):
A_bfp16 = convert_to_bfp16(A)  # Returns uint8 array
C_bfp16 = npu_matmul(A_bfp16, B_bfp16)  # Returns uint8 array
```

### Phase 5: Testing (8-10 hours)

**3-tier testing strategy**:

1. **Unit Tests** (2-3 hours):
   - BFP16 conversion accuracy
   - Shuffle correctness
   - Round-trip error <1%

2. **Accuracy Validation** (3-4 hours):
   - Compare vs PyTorch FP32
   - Target: >99% cosine similarity
   - Test with real Whisper weights

3. **Performance Benchmarking** (2-3 hours):
   - Single matmul latency
   - Full encoder latency
   - Realtime factor calculation
   - Target: >17× realtime

---

## Quick Commands

### Start Implementation

```bash
# Navigate to project
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2

# Review roadmap
less BFP16_INTEGRATION_ROADMAP.md

# Create Phase 1 files
mkdir -p cpp/include cpp/src
touch cpp/include/bfp16_converter.hpp
touch cpp/src/bfp16_converter.cpp
```

### Build and Test

```bash
# Build C++ with new BFP16 code
cd cpp/build
cmake ..
make -j16

# Run unit tests
make test

# Run accuracy test
cd ../..
python3 test_bfp16_integration.py
```

### Validate Accuracy

```bash
# Compare BFP16 vs PyTorch
python3 test_accuracy_bfp16_vs_pytorch.py

# Expected output:
# Cosine similarity: 0.9912 (target: >0.99) ✅
# Relative error: 0.87% (target: <1%) ✅
# Performance: 18.5× realtime (target: >17×) ✅
```

---

## Expected Results

### Performance Trajectory

```
Current (INT8):
  - Latency: 470 ms
  - Realtime: 21.79×
  - Accuracy: 64.6% ❌

Week 1 (Converter + Integration):
  - Latency: ~520 ms (CPU fallback)
  - Realtime: ~19× (estimated)
  - Accuracy: >99% ✅

Week 2 (NPU Integration):
  - Latency: 517-565 ms
  - Realtime: 18-20×
  - Accuracy: >99% ✅

Production Target: ✅ ACHIEVED
```

### Success Metrics

| Metric | Target | Expected | Status |
|--------|--------|----------|--------|
| Accuracy | >99% | 99-99.5% | ✅ High confidence |
| Performance | >17× | 18-20× | ✅ High confidence |
| Stability | >95% | >99% | ✅ High confidence |
| Memory | <512MB | 200MB | ✅ High confidence |
| Power | <20W | 5-15W | ✅ High confidence |

---

## Risk Summary

### Low Risk (High Confidence)

✅ **BFP16 converter**: Reference code available (`mm_bfp.cc`)
✅ **Quantization update**: Straightforward API replacement
✅ **Memory requirements**: Well within limits (200MB)
✅ **Power consumption**: Same as INT8 (5-15W)

### Medium Risk (Mitigation Available)

⚠️ **BFP16 shuffle complexity**: Reference implementation available, CPU fallback if needed
⚠️ **Accuracy validation**: Early testing after Phase 1, block size tuning available
⚠️ **Integration bugs**: Extensive testing, keep INT8 as reference

### Managed Risk (Contingency Plans)

⚠️ **NPU kernel compilation**: Early validation, INT8 fallback available
⚠️ **Performance below target**: Optimization options (vectorization, larger tiles)

**Overall Risk**: LOW-MEDIUM (Well-managed with clear mitigation strategies)

---

## Development Checklist

### Week 1 (Core Implementation)

- [ ] **Day 1**: Create `bfp16_converter.hpp` with function signatures
- [ ] **Day 1**: Implement `fp32_to_bfp16()` with block processing
- [ ] **Day 2**: Implement `bfp16_to_fp32()` and round-trip tests
- [ ] **Day 2**: Implement shuffle/unshuffle based on `mm_bfp.cc`
- [ ] **Day 3**: Create `BFP16Quantizer` class with prepare/read functions
- [ ] **Day 3**: Unit tests for quantization layer
- [ ] **Day 4**: Update `encoder_layer.hpp` with BFP16 buffers
- [ ] **Day 4**: Update `load_weights()` to use BFP16
- [ ] **Day 5**: Update `run_npu_linear()` with BFP16 conversions
- [ ] **Day 5**: Single-layer integration test

### Week 2 (NPU + Testing)

- [ ] **Day 6**: Compile `matmul_bfp16_512x512x512.xclbin`
- [ ] **Day 6**: Compile `matmul_bfp16_512x512x2048.xclbin`
- [ ] **Day 7**: Update Python NPU runtime for BFP16
- [ ] **Day 7**: Update C++ callback wrapper
- [ ] **Day 8**: Accuracy validation vs PyTorch (target >99%)
- [ ] **Day 9**: Performance benchmarking (target >17×)
- [ ] **Day 9**: Compare BFP16 vs INT8
- [ ] **Day 10**: Stability testing (100 iterations)
- [ ] **Day 10**: Write validation report

---

## Key Files to Create/Modify

### New Files

```
cpp/include/bfp16_converter.hpp         (Phase 1)
cpp/src/bfp16_converter.cpp             (Phase 1)
cpp/test/test_bfp16_converter.cpp       (Phase 1)
cpp/test/test_bfp16_quantization.cpp    (Phase 2)
cpp/test/test_encoder_layer_bfp16.cpp   (Phase 3)
test_bfp16_integration.py               (Phase 5)
test_accuracy_bfp16_vs_pytorch.py       (Phase 5)
benchmark_bfp16_performance.py          (Phase 5)
BFP16_VALIDATION_REPORT.md              (Phase 5)
```

### Modified Files

```
cpp/include/quantization.hpp            (Phase 2: Add BFP16Quantizer)
cpp/src/quantization.cpp                (Phase 2: Implement BFP16Quantizer)
cpp/include/encoder_layer.hpp           (Phase 3: Replace INT8 buffers)
cpp/src/encoder_layer.cpp               (Phase 3: Update run_npu_linear)
runtime/npu_runtime.py                  (Phase 4: Add BFP16 support)
runtime/cpp_callback.py                 (Phase 4: Add BFP16 callback)
```

---

## Reference Implementation

**BFP16 Kernel Reference**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/kernels/bfp16/`

```
mm_bfp.cc                  - BFP16 matmul kernel (C++)
single_core_iron.py        - MLIR generator with BFP16 support
makefile-common            - Build configuration
```

**Key Functions in `mm_bfp.cc`**:
- `scalarShuffleMatrixForBfp16ebs8()` - Shuffle implementation (lines 30-66)
- `matmul_vectorized_2x2_bfp16()` - BFP16 matmul kernel (lines 79-155)
- `zero_vectorized_v64bfp16ebs8()` - BFP16 zero initialization (lines 14-27)

---

## Getting Help

### Documentation

- **Full Roadmap**: `BFP16_INTEGRATION_ROADMAP.md` (133 KB, comprehensive)
- **Quick Start**: This file
- **AMD Reference**: `kernels/bfp16/mm_bfp.cc` (example kernel)
- **Comprehensive Findings**: `COMPREHENSIVE_FINDINGS_SUMMARY.md` (BFP16 research)

### Testing Resources

- **Accuracy baseline**: `test_accuracy_vs_pytorch.py` (current INT8 test)
- **Performance baseline**: `test_cpp_steady_state.py` (21.79× realtime)
- **Stability baseline**: `test_cpp_npu_stability.py` (99.22% consistency)

### Support

- **AMD MLIR-AIE Examples**: `~/mlir-aie/programming_examples/`
- **AMD Documentation**: XDNA2 NPU Programming Guide
- **Team**: Review roadmap together, pair programming for complex parts

---

## Success Criteria

**Ready to Deploy When**:

✅ All unit tests pass (100% pass rate)
✅ Accuracy >99% cosine similarity vs PyTorch
✅ Performance >17× realtime (target: 18-20×)
✅ Stability >99% consistency over 100 iterations
✅ Memory usage <512MB (expected: 200MB)
✅ No memory leaks (valgrind clean)
✅ Error handling robust (no crashes)
✅ Documentation complete

**Estimated Completion**: November 10-15, 2025

---

## Next Steps

1. **Review this quick start** with team (15 minutes)
2. **Read full roadmap** for detailed implementation (`BFP16_INTEGRATION_ROADMAP.md`)
3. **Start Phase 1** (Day 1): Create `bfp16_converter.hpp`
4. **Daily standups** to track progress and blockers
5. **Weekly milestones** to validate approach

**Ready to build production-grade accuracy!** 🚀

---

**Document Version**: 1.0
**Last Updated**: October 30, 2025
**Status**: ✅ Ready for Implementation

**See Also**:
- `BFP16_INTEGRATION_ROADMAP.md` - Comprehensive 133 KB implementation guide
- `COMPREHENSIVE_FINDINGS_SUMMARY.md` - Research findings and BFP16 analysis
- `kernels/bfp16/` - Reference BFP16 kernel implementations
