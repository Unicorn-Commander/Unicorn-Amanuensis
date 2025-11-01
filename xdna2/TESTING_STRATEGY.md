# Phase 5 Testing & Validation Strategy
## Whisper Encoder BFP16 NPU Acceleration

**Date**: October 30, 2025
**Project**: CC-1L Whisper Encoder NPU Acceleration
**Phase**: 5 - Real NPU Integration Testing
**Status**: READY FOR EXECUTION

---

## Executive Summary

This document outlines the comprehensive testing and validation strategy for Phase 5 of the Whisper encoder BFP16 NPU acceleration project. The goal is to validate that real NPU execution maintains the accuracy and performance targets achieved in Phase 0-4 mock tests.

**Key Targets**:
- **Accuracy**: >99% cosine similarity vs PyTorch reference
- **Performance**: 400-500× realtime speech-to-text throughput
- **Stability**: Zero crashes in 10,000+ iteration stress tests
- **Latency**: <50ms full 6-layer encoder, <8ms per layer

---

## 1. Test Pyramid Strategy

```
                    /\
                   /  \
                  / E2E \ ← 10 tests (full 6-layer encoder)
                 /______\
                /        \
               /Integration\ ← 50 tests (layer + NPU + XRT)
              /____________\
             /              \
            /  Unit Tests    \ ← 200 tests (quantizer, kernels)
           /________________\
```

### Test Distribution

| Layer | Test Count | Coverage | Purpose |
|-------|-----------|----------|---------|
| **Unit Tests** | 200+ | BFP16 quantizer, kernels, buffers | Component validation |
| **Integration Tests** | 50+ | Layer + NPU + XRT | System integration |
| **E2E Tests** | 10+ | Full 6-layer encoder | End-to-end validation |
| **Stress Tests** | 5+ | Long-running, edge cases | Stability validation |
| **Benchmarks** | 20+ | Performance characterization | Performance validation |
| **Total** | **285+** | **Full stack** | **Production readiness** |

---

## 2. Test Categories

### 2.1 Unit Tests (200+ tests)

**Components Under Test**:
- BFP16 quantization (convert_to_bfp16, convert_from_bfp16)
- Shuffle/unshuffle operations
- Buffer management (allocation, deallocation, memory pools)
- Kernel selection and loading
- Error handling and edge cases

**Success Criteria**:
- Round-trip error <1% (BFP16 quantization)
- Shuffle/unshuffle is exact reversal (byte-for-byte)
- No memory leaks in 10,000 iterations
- All edge cases handled gracefully (zeros, infinities, NaN)

**Status**: ✅ 11/11 BFP16 unit tests passing (Phase 4 complete)

---

### 2.2 Integration Tests (50+ tests)

**Components Under Test**:
- EncoderLayer + NPU callback
- XRT runtime integration
- Matmul operations (Q, K, V projections, FFN layers)
- Memory transfer (CPU ↔ NPU)
- Callback synchronization

**Test Matrix**:

| Test Case | Input Size | Expected Similarity | Max Error |
|-----------|-----------|---------------------|-----------|
| Small matmul | 64×64×64 | >99.99% | <0.5% |
| Whisper Q projection | 512×512×512 | >99.9% | <1% |
| Whisper K projection | 512×512×512 | >99.9% | <1% |
| Whisper V projection | 512×512×512 | >99.9% | <1% |
| Whisper out projection | 512×512×512 | >99.9% | <1% |
| Whisper FFN1 | 512×512×2048 | >99.9% | <1% |
| Whisper FFN2 | 512×2048×512 | >99.9% | <1% |
| Single encoder layer | (1,1500,512) | >99.5% | <2% |
| Batch of 2 | (2,1500,512) | >99.5% | <2% |
| Batch of 4 | (4,1500,512) | >99.5% | <2% |

**Success Criteria**:
- NPU callback executes without errors
- Data transfer CPU ↔ NPU is correct (validated vs CPU reference)
- Synchronization is correct (no race conditions)
- Memory is properly managed (no leaks, no corruption)

---

### 2.3 End-to-End Tests (10+ tests)

**Components Under Test**:
- Full 6-layer Whisper encoder
- All 6 layers stacked sequentially
- Real audio input → mel spectrogram → encoder → embeddings
- PyTorch reference comparison

**Test Cases**:
1. **Single utterance** (1 second audio)
2. **Medium utterance** (10 seconds audio)
3. **Long utterance** (30 seconds audio)
4. **Batch processing** (4× 10-second utterances)
5. **Edge cases**:
   - Silent audio (all zeros)
   - Loud audio (near clipping)
   - Music (non-speech)
   - Noisy speech (SNR 10dB)
6. **Multiple speakers**
7. **Different languages** (English, Spanish, Mandarin)
8. **Different accents**
9. **Continuous processing** (100 utterances back-to-back)
10. **Cold start** (first run after boot)

**Success Criteria**:
- Cosine similarity >99% vs PyTorch reference
- Relative error <3% (full 6-layer encoder)
- WER impact <1% (word error rate on LibriSpeech test set)
- No crashes or memory leaks

---

### 2.4 Stress Tests (5+ tests)

**Purpose**: Validate stability under extreme conditions

**Test Cases**:

#### 2.4.1 Consistency Test
- **Duration**: 1,000 iterations
- **Input**: Same mel spectrogram repeated
- **Validation**: Outputs are **identical** (bit-for-bit deterministic)
- **Metrics**: Standard deviation across runs (should be 0.0)

#### 2.4.2 Memory Leak Test
- **Duration**: 10,000 iterations
- **Monitoring**: Memory usage (RSS, heap allocations)
- **Validation**: Memory usage is stable (no growth)
- **Tools**: Valgrind, AddressSanitizer

#### 2.4.3 Error Recovery Test
- **Injected Errors**:
  - Bad inputs (NaN, Inf, out-of-range)
  - Device resets (simulated NPU failure)
  - Memory allocation failures
- **Validation**: Graceful error handling (no crashes)

#### 2.4.4 Extended Runtime Test
- **Duration**: 1 hour continuous processing
- **Load**: 10 utterances/second
- **Validation**: No performance degradation, no crashes
- **Metrics**: Throughput, latency (p50, p95, p99)

#### 2.4.5 Thermal Throttling Test
- **Conditions**: Run under thermal load
- **Monitoring**: NPU temperature, frequency scaling
- **Validation**: Performance degrades gracefully (no crashes)

---

### 2.5 Performance Benchmarks (20+ tests)

**Metrics**:
- **Latency**: Time to process one utterance
- **Throughput**: Utterances per second
- **GFLOPS**: Computational throughput
- **Realtime Factor**: Processing time / audio duration
- **Power**: Watts consumed during inference

**Benchmark Matrix**:

| Operation | Target Latency | Target Throughput | Target GFLOPS |
|-----------|---------------|-------------------|---------------|
| 512×512×512 matmul | <2 ms | >500 matmuls/sec | >100 GFLOPS |
| Single encoder layer | <8 ms | >125 layers/sec | >50 GFLOPS |
| Full encoder (6 layers) | <50 ms | >20 encodes/sec | >40 GFLOPS |
| Whisper Base (30s audio) | <75 ms | >13 utterances/sec | **>400× realtime** |

**Scaling Benchmarks**:
- Batch size scaling (1, 2, 4, 8)
- Sequence length scaling (512, 1024, 1500, 3000)
- Layer scaling (1, 2, 3, 6 layers)
- Warmup effect (cold start vs warm)

---

## 3. PyTorch Reference Implementation

**Purpose**: Golden reference for accuracy validation

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/tests/pytorch_reference.py`

**Capabilities**:
- Load official Whisper model from HuggingFace
- Extract weights for C++ comparison
- Encode mel spectrograms (full encoder)
- Encode layer-by-layer (for detailed comparison)
- Export reference outputs as `.npy` files

**API**:
```python
class WhisperEncoderReference:
    def __init__(self, model_name="openai/whisper-base")
    def extract_weights(self) -> dict
    def encode(self, mel_spectrogram: np.ndarray) -> np.ndarray
    def encode_layer(self, layer_idx: int, input: np.ndarray) -> np.ndarray
```

**Status**: ✅ Partially complete (`test_accuracy_vs_pytorch.py` exists)

---

## 4. NPU Accuracy Test Suite

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/tests/test_npu_accuracy.py`

**Test Implementation**:
```python
def test_npu_vs_pytorch_accuracy():
    """
    Compare NPU encoder output vs PyTorch reference

    Steps:
    1. Load same Whisper weights in both
    2. Generate test mel spectrogram (seed=42)
    3. Run PyTorch encoder → output_pytorch
    4. Run NPU encoder → output_npu
    5. Compare:
       - Cosine similarity >99%
       - Relative error <3%
       - Element-wise accuracy >99%
    """
    pass

def test_npu_layer_by_layer():
    """
    Compare layer-by-layer (more granular debugging)

    For each layer i=0..5:
    1. Run PyTorch layer_i(input)
    2. Run NPU layer_i(input)
    3. Assert similarity >99.5%
    4. Use PyTorch output as input to next layer
    """
    pass
```

---

## 5. Performance Benchmark Suite

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/tests/benchmark_npu_performance.py`

**Implementation**:
```python
def benchmark_matmul_512x512x512():
    """Measure latency and GFLOPS for 512×512×512 matmul"""
    # Warmup: 10 iterations
    # Benchmark: 100 iterations
    # Report: mean, median, p95, p99 latency
    #         GFLOPS, ops/sec
    pass

def benchmark_single_layer():
    """Measure single encoder layer performance"""
    pass

def benchmark_full_encoder():
    """Measure 6-layer encoder performance"""
    # Report: ms/encode, encodes/sec, realtime factor
    pass

def benchmark_batch_scaling():
    """Measure throughput vs batch size (1, 2, 4, 8)"""
    # Expected: Near-linear scaling up to batch=4
    pass
```

---

## 6. Stability Test Suite

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/tests/test_npu_stability.py`

**Test Cases** (already implemented):
- `test_cpp_npu_stability.py` - Basic stability
- `test_cpp_real_weights_stability.py` - Real weights stability
- `test_cpp_steady_state.py` - Long-running stability

**To Add**:
- Memory leak detection (with Valgrind)
- Error injection and recovery
- Thermal stress test

---

## 7. Comparative Analysis Framework

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/tests/comparative_analysis.py`

**Comparisons**:
1. **BFP16 NPU vs INT8 NPU** (accuracy improvement)
2. **BFP16 NPU vs FP32 CPU** (speed vs accuracy tradeoff)
3. **BFP16 NPU vs FP32 GPU** (if available)
4. **BFP16 NPU vs FP16 NPU** (storage vs accuracy)

**Report Format**:
```markdown
# BFP16 NPU Performance Report

## Accuracy Comparison
| Format | Cosine Sim | Rel Error | WER Impact | Storage |
|--------|-----------|-----------|------------|---------|
| BFP16 NPU | 99.5% | 1.2% | 0.5% | 28.1% |
| INT8 NPU | 98.0% | 3.5% | 2.0% | 25.0% |
| FP32 CPU | 100.0% | 0.0% | 0.0% | 100% |

## Performance Comparison
| Backend | Latency | GFLOPS | Power | Realtime Factor |
|---------|---------|--------|-------|----------------|
| BFP16 NPU | 45 ms | 120 | 12W | 450× |
| INT8 NPU | 40 ms | 140 | 10W | 500× |
| FP32 CPU | 8000 ms | 8 | 45W | 2.5× |
```

---

## 8. Regression Test Database

**Directory**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/tests/regression_database/`

**Structure**:
```
regression_database/
├── test_vectors/
│   ├── mel_spec_001.npy         # Known input (1s audio)
│   ├── mel_spec_002.npy         # Known input (10s audio)
│   ├── mel_spec_003.npy         # Known input (30s audio)
│   └── ...
├── reference_outputs/
│   ├── pytorch_001.npy          # Known good output (PyTorch)
│   ├── pytorch_002.npy
│   └── ...
├── whisper_weights/
│   ├── layer_0_q_weight.npy
│   ├── layer_0_k_weight.npy
│   └── ... (already exists in weights/)
└── benchmarks/
    └── baseline_performance.json
```

**Purpose**:
- Detect regression in accuracy (compare vs known good outputs)
- Detect regression in performance (compare vs baseline metrics)
- Provide reproducible test cases for debugging

---

## 9. CI/CD Integration

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/.github/workflows/npu_validation.yml`

**Workflow**:
```yaml
name: NPU Validation

on: [push, pull_request]

jobs:
  test-npu:
    runs-on: [self-hosted, xdna2]

    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Build C++ Library
        run: |
          cd cpp/build
          cmake .. -DCMAKE_BUILD_TYPE=Release
          make -j16

      - name: Run Unit Tests
        run: cd cpp/build && ctest --output-on-failure

      - name: Run Accuracy Tests
        run: python tests/test_npu_accuracy.py

      - name: Run Performance Benchmarks
        run: python tests/benchmark_npu_performance.py

      - name: Run Stability Tests
        run: python tests/test_npu_stability.py

      - name: Generate Report
        run: python tests/generate_validation_report.py

      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: tests/results/
```

---

## 10. Success Criteria

### Phase 5 Completion Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| **Accuracy** | >99% cosine similarity | ⏳ To validate |
| **Performance** | >400× realtime | ⏳ To measure |
| **Latency** | <50ms full encoder | ⏳ To measure |
| **Stability** | 10,000 iterations no crash | ⏳ To test |
| **Memory** | No leaks | ⏳ To verify |
| **Unit Tests** | 200+ passing | ✅ 11/11 BFP16 tests |
| **Integration Tests** | 50+ passing | ⏳ To create |
| **E2E Tests** | 10+ passing | ⏳ To create |
| **Benchmarks** | 20+ completed | ⏳ To run |

---

## 11. Timeline

**Total Estimated Time**: 4-6 hours (Testing & Validation Team)

| Task | Duration | Priority |
|------|----------|----------|
| 1. PyTorch reference implementation | 1 hour | HIGH |
| 2. NPU accuracy test suite | 2 hours | CRITICAL |
| 3. Performance benchmark suite | 1.5 hours | HIGH |
| 4. Stability test suite | 1 hour | MEDIUM |
| 5. Comparative analysis | 1 hour | MEDIUM |
| 6. Regression database setup | 0.5 hours | LOW |
| 7. Documentation (TESTING_GUIDE.md) | 1 hour | MEDIUM |
| 8. CI/CD integration | 0.5 hours | LOW |

---

## 12. Dependencies

**Team 1 (Kernel Compilation)**:
- [ ] XCLBin files for 512×512, 512×2048 matmul
- [ ] Kernel metadata (tile count, memory requirements)
- [ ] Performance profiling data

**Team 2 (XRT Integration)**:
- [ ] XRT runtime library (`libwhisper_xdna2_cpp.so`)
- [ ] NPU callback implementation
- [ ] Buffer management API
- [ ] Error handling implementation

**Independent Work**:
- [x] PyTorch reference (can start immediately)
- [x] Test data generation (can start immediately)
- [ ] Accuracy metrics implementation
- [ ] Benchmark infrastructure

---

## 13. Risk Mitigation

### Potential Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **NPU accuracy <99%** | High | Detailed layer-by-layer debugging, adjust quantization |
| **Performance <400×** | High | Profile bottlenecks, optimize memory transfer, kernel tuning |
| **Stability issues** | Medium | Extensive stress testing, Valgrind analysis |
| **Memory leaks** | Medium | AddressSanitizer, Valgrind, ref counting |
| **XRT integration bugs** | High | Mock testing first, gradual rollout |

---

## 14. Deliverables

**Documentation**:
- [x] `TESTING_STRATEGY.md` (this document)
- [ ] `TESTING_GUIDE.md` (user guide)
- [ ] `PHASE5_VALIDATION_REPORT.md` (final results)

**Code**:
- [ ] `tests/pytorch_reference.py` (PyTorch reference)
- [ ] `tests/test_npu_accuracy.py` (accuracy tests)
- [ ] `tests/benchmark_npu_performance.py` (performance benchmarks)
- [ ] `tests/test_npu_stability.py` (stability tests)
- [ ] `tests/comparative_analysis.py` (comparative analysis)
- [ ] `tests/generate_validation_report.py` (report generator)

**Data**:
- [ ] `tests/regression_database/` (test vectors + reference outputs)
- [ ] `tests/results/` (test results, benchmarks, reports)

---

## 15. Next Steps

**Immediate Actions**:
1. ✅ Create `TESTING_STRATEGY.md` (this document)
2. ⏳ Enhance `tests/pytorch_reference.py` (add layer-by-layer API)
3. ⏳ Create `tests/test_npu_accuracy.py` (NPU vs PyTorch comparison)
4. ⏳ Create `tests/benchmark_npu_performance.py` (performance suite)
5. ⏳ Create `TESTING_GUIDE.md` (user documentation)

**Coordination**:
- Sync with Team 1: Get XCLBin status, expected performance
- Sync with Team 2: Get XRT API documentation, integration timeline
- Provide test vectors to both teams for validation

---

**Status**: READY FOR EXECUTION
**Owner**: Testing & Validation Team (Team 3)
**Last Updated**: October 30, 2025

---

Generated with Claude Code (Anthropic)
Project: CC-1L Whisper Encoder NPU Acceleration
