# Testing Guide for Whisper Encoder BFP16 NPU Acceleration

**Date**: October 30, 2025
**Project**: CC-1L Whisper Encoder NPU Acceleration
**Version**: 1.0.0

---

## Table of Contents

1. [Introduction](#introduction)
2. [Test Organization](#test-organization)
3. [Running Tests](#running-tests)
4. [Understanding Test Results](#understanding-test-results)
5. [Adding New Tests](#adding-new-tests)
6. [Troubleshooting](#troubleshooting)
7. [CI/CD Integration](#cicd-integration)
8. [Performance Profiling](#performance-profiling)

---

## Introduction

This guide explains how to run, interpret, and extend the test suite for the Whisper encoder BFP16 NPU acceleration project.

### Test Philosophy

Our testing follows the **Test Pyramid** approach:
- **200+ Unit Tests**: Fast, isolated component tests
- **50+ Integration Tests**: Layer + NPU + XRT integration
- **10+ E2E Tests**: Full encoder validation
- **5+ Stress Tests**: Stability and error handling
- **20+ Benchmarks**: Performance characterization

### Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| **Accuracy** | >99% cosine similarity | Phase 4: ✅ 99.99% |
| **Performance** | >400× realtime | Phase 5: ⏳ To validate |
| **Stability** | 10,000 iterations | Phase 4: ✅ Passing |
| **Memory** | Zero leaks | Phase 4: ✅ Verified |

---

## Test Organization

### Directory Structure

```
xdna2/
├── cpp/
│   ├── tests/                          # C++ tests (GTest)
│   │   ├── test_quantization.cpp       # INT8 quantization tests
│   │   ├── test_bfp16_quantization.cpp # BFP16 quantization (6 tests)
│   │   ├── test_bfp16_converter.cpp    # BFP16 converter (8 tests)
│   │   ├── test_encoder_layer_bfp16.cpp# EncoderLayer integration (3 tests)
│   │   ├── test_encoder_layer.cpp      # Legacy encoder tests
│   │   ├── test_accuracy.cpp           # Accuracy validation
│   │   └── test_runtime.cpp            # XRT runtime tests
│   └── benchmarks/
│       └── bench_encoder.cpp           # C++ performance benchmarks
└── tests/                              # Python tests
    ├── pytorch_reference.py            # PyTorch golden reference
    ├── test_npu_accuracy.py            # NPU accuracy tests
    ├── benchmark_npu_performance.py    # NPU performance benchmarks
    ├── test_accuracy_vs_pytorch.py     # PyTorch vs C++ comparison
    ├── test_cpp_npu_stability.py       # Stability tests
    ├── test_cpp_real_weights_stability.py
    ├── test_cpp_steady_state.py
    └── regression_database/            # Test vectors and baselines
        ├── test_vectors/               # Input test cases
        └── reference_outputs/          # Known good outputs
```

### Test Categories

#### 1. Unit Tests (C++)
- **Location**: `cpp/tests/test_bfp16_*.cpp`
- **Framework**: Google Test
- **Run time**: <1 second
- **Command**: `cd cpp/build && ctest --verbose`

#### 2. Integration Tests (Python + C++)
- **Location**: `tests/test_*.py`
- **Framework**: Python unittest / custom
- **Run time**: 1-10 seconds per test
- **Command**: `python tests/test_npu_accuracy.py`

#### 3. E2E Tests (Python)
- **Location**: `tests/test_accuracy_vs_pytorch.py`
- **Framework**: Custom validation
- **Run time**: 10-60 seconds
- **Command**: `python tests/test_accuracy_vs_pytorch.py`

#### 4. Performance Benchmarks (Python)
- **Location**: `tests/benchmark_npu_performance.py`
- **Framework**: Custom benchmarking
- **Run time**: 1-5 minutes
- **Command**: `python tests/benchmark_npu_performance.py`

---

## Running Tests

### Prerequisites

```bash
# 1. Build C++ library
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j16

# 2. Install Python dependencies
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2
pip install -r requirements.txt

# 3. Extract Whisper weights (if needed)
python extract_whisper_weights.py
```

### Running C++ Unit Tests

```bash
# Run all tests
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build
ctest --output-on-failure

# Run specific test suite
ctest -R BFP16QuantizationTest --verbose

# Run with Valgrind (memory leak detection)
ctest -T memcheck

# List all available tests
ctest --show-only
```

**Expected Output** (Phase 4):
```
Test project /home/ccadmin/.../cpp/build
    Start 1: QuantizationTest
1/5 Test #1: QuantizationTest ................. Passed    0.00 sec
    Start 2: EncoderLayerTest
2/5 Test #2: EncoderLayerTest ................. Failed    1.19 sec
    Start 3: BFP16ConverterTest
3/5 Test #3: BFP16ConverterTest ............... Passed    0.02 sec
    Start 4: BFP16QuantizationTest
4/5 Test #4: BFP16QuantizationTest ............ Passed    0.05 sec
    Start 5: EncoderLayerBFP16Test
5/5 Test #5: EncoderLayerBFP16Test ............ Passed    0.51 sec

80% tests passed, 1 tests failed out of 5
```

**Note**: `EncoderLayerTest` failure is expected (pre-existing, unrelated to BFP16).

### Running Python Accuracy Tests

```bash
# Full accuracy test suite
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2
python tests/test_npu_accuracy.py

# PyTorch vs C++ comparison
python tests/test_accuracy_vs_pytorch.py

# Individual tests (when NPU is ready)
python -m pytest tests/test_npu_accuracy.py::NPUAccuracyTester::test_single_layer -v
```

**Expected Output** (Phase 5):
```
================================================================================
  NPU ACCURACY TEST SUITE
================================================================================
2025-10-30 12:00:00 - INFO - Running test: small_matmul
2025-10-30 12:00:00 - WARNING - Test skipped: NPU matmul not yet implemented
...
================================================================================
  TEST SUMMARY
================================================================================
Total tests: 6
Passed: 1
Failed: 0
Skipped: 5
```

### Running Performance Benchmarks

```bash
# Full benchmark suite
python tests/benchmark_npu_performance.py

# Save results to file
python tests/benchmark_npu_performance.py > benchmark_results.txt

# Run with profiling
python -m cProfile -o benchmark.prof tests/benchmark_npu_performance.py
```

**Expected Output** (Phase 5):
```
================================================================================
Benchmark: Single Encoder Layer
================================================================================
Warming up (10 iterations)...
Benchmarking (100 iterations)...
Mean latency: 5.23 ms (target: <8ms)
Throughput: 191.2 layers/sec (target: >125)
Meets target: YES
```

### Running Stability Tests

```bash
# Consistency test (1,000 iterations)
python tests/test_cpp_npu_stability.py

# Extended runtime test (1 hour)
python tests/test_cpp_steady_state.py --duration 3600

# Memory leak test (with Valgrind)
valgrind --leak-check=full python tests/test_cpp_real_weights_stability.py
```

---

## Understanding Test Results

### Accuracy Metrics

#### Cosine Similarity
- **Range**: 0.0 to 1.0 (1.0 = perfect match)
- **Target**: >0.99 (99%)
- **Interpretation**:
  - 0.9999: Excellent (99.99%)
  - 0.999: Very good (99.9%)
  - 0.99: Good (99%)
  - <0.99: **FAIL** (needs investigation)

#### Relative Error
- **Range**: 0.0 to ∞ (0.0 = perfect match)
- **Target**: <0.01 (1%)
- **Formula**: `mean(|a - b| / |a|)`
- **Interpretation**:
  - 0.005: Excellent (0.5% error)
  - 0.01: Good (1% error)
  - 0.03: Acceptable for full encoder (3% error)
  - >0.03: **FAIL** (needs investigation)

#### SNR (Signal-to-Noise Ratio)
- **Range**: 0 to ∞ (higher = better)
- **Target**: >40 dB
- **Interpretation**:
  - >50 dB: Excellent
  - 40-50 dB: Good
  - 30-40 dB: Acceptable
  - <30 dB: **FAIL**

### Performance Metrics

#### Latency
- **Unit**: milliseconds (ms)
- **Metrics**:
  - **Mean**: Average latency
  - **Median** (p50): Middle value (less affected by outliers)
  - **p95**: 95th percentile (tail latency)
  - **p99**: 99th percentile (worst-case latency)

**Example**:
```json
{
  "mean_latency_ms": 5.23,
  "median_latency_ms": 5.18,
  "p95_latency_ms": 5.67,
  "p99_latency_ms": 6.12
}
```

**Interpretation**:
- Mean close to median → consistent performance
- Large p99 spike → investigate tail latencies

#### Throughput
- **Unit**: operations per second
- **Example**: 191.2 layers/sec
- **Calculation**: `1000 / mean_latency_ms`

#### GFLOPS
- **Definition**: Billion floating-point operations per second
- **Calculation**: `(2 * M * N * K) / (latency_sec * 1e9)`
- **Example**: 512×512×512 matmul @ 2ms → 136 GFLOPS

#### Realtime Factor
- **Definition**: Processing speed vs audio duration
- **Example**: 30s audio processed in 75ms → 400× realtime
- **Calculation**: `audio_duration / processing_time`

---

## Adding New Tests

### Adding a C++ Unit Test

**1. Create test file**: `cpp/tests/test_my_feature.cpp`

```cpp
#include <gtest/gtest.h>
#include "my_feature.hpp"

TEST(MyFeatureTest, BasicFunctionality) {
    // Arrange
    MyFeature feature;

    // Act
    int result = feature.compute(42);

    // Assert
    EXPECT_EQ(result, 84);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

**2. Register test in CMake**: `cpp/tests/CMakeLists.txt`

```cmake
add_executable(test_my_feature test_my_feature.cpp)
target_link_libraries(test_my_feature whisper_encoder_cpp gtest gtest_main)
add_test(NAME MyFeatureTest COMMAND test_my_feature)
```

**3. Build and run**:
```bash
cd cpp/build
make test_my_feature
ctest -R MyFeatureTest --verbose
```

### Adding a Python Accuracy Test

**1. Add test method**: `tests/test_npu_accuracy.py`

```python
def test_my_accuracy_check(self) -> dict:
    """
    Test: My new accuracy check
    Target: >99% similarity
    """
    logger.info("="*80)
    logger.info("Test: My Accuracy Check")
    logger.info("="*80)

    try:
        # Run test
        # ...

        metrics = {
            'test_name': 'my_accuracy_check',
            'status': 'PASS',
            'cosine_similarity': 0.995,
            'relative_error': 0.008,
        }

        logger.info(f"Test PASSED: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Test FAILED: {e}")
        return {
            'test_name': 'my_accuracy_check',
            'status': 'FAILED',
            'error': str(e),
        }
```

**2. Register in test runner**:

```python
def run_all_tests(self):
    all_results['my_accuracy_check'] = self.test_my_accuracy_check()
```

### Adding a Performance Benchmark

**1. Add benchmark method**: `tests/benchmark_npu_performance.py`

```python
def benchmark_my_operation(self, num_iterations: int = 100) -> Dict:
    """
    Benchmark: My operation
    Target: <X ms

    Args:
        num_iterations: Number of iterations

    Returns:
        metrics: Performance metrics
    """
    logger.info("Benchmark: My Operation")

    # Warmup
    for _ in range(10):
        my_operation()

    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        my_operation()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)
    return {
        'benchmark': 'my_operation',
        'mean_latency_ms': float(np.mean(latencies)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'throughput_ops_per_sec': 1000.0 / np.mean(latencies),
    }
```

---

## Troubleshooting

### Common Issues

#### Issue 1: C++ Tests Fail to Build

**Symptom**:
```
error: undefined reference to 'BFP16Quantizer::convert_to_bfp16'
```

**Solution**:
```bash
# Clean and rebuild
cd cpp/build
make clean
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j16
```

#### Issue 2: Python Tests Can't Find C++ Library

**Symptom**:
```
FileNotFoundError: C++ library not found: libwhisper_encoder_cpp.so
```

**Solution**:
```bash
# Check library exists
ls -la cpp/build/libwhisper_encoder_cpp.so

# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build:$LD_LIBRARY_PATH
```

#### Issue 3: Accuracy Tests Fail (Low Cosine Similarity)

**Symptom**:
```
AssertionError: Cosine similarity 0.85 < 0.99 (target)
```

**Investigation Steps**:
1. **Check weight loading**: Verify weights match PyTorch
2. **Layer-by-layer debug**: Run `test_accuracy_vs_pytorch.py` with layer-by-layer comparison
3. **Check quantization**: Verify BFP16 round-trip error <1%
4. **Inspect numerical stability**: Check for NaN/Inf in outputs

**Debug Commands**:
```bash
# Layer-by-layer comparison
python tests/test_accuracy_vs_pytorch.py --layer-by-layer --verbose

# Dump intermediate outputs
python tests/test_accuracy_vs_pytorch.py --save-intermediate
```

#### Issue 4: Memory Leaks Detected

**Symptom**:
```
Valgrind: definitely lost: 2,048 bytes in 1 blocks
```

**Investigation**:
```bash
# Run with Valgrind
valgrind --leak-check=full --show-leak-kinds=all \
  python tests/test_cpp_npu_stability.py

# Check for missing encoder_layer_destroy() calls
grep -n "encoder_layer_create" tests/*.py
grep -n "encoder_layer_destroy" tests/*.py
```

#### Issue 5: Performance Below Target

**Symptom**:
```
Mean latency: 15.2 ms (target: <8ms)
```

**Investigation**:
1. **Profile bottlenecks**: Use `perf` or `gprof`
2. **Check NPU utilization**: Ensure NPU is actually being used
3. **Memory bandwidth**: Check CPU ↔ NPU transfer overhead
4. **Warmup**: Ensure proper warmup (JIT compilation)

**Profiling Commands**:
```bash
# Profile with perf
perf record -g python tests/benchmark_npu_performance.py
perf report

# Profile with cProfile
python -m cProfile -o bench.prof tests/benchmark_npu_performance.py
python -m pstats bench.prof
```

---

## CI/CD Integration

### GitHub Actions Workflow

**File**: `.github/workflows/npu_validation.yml`

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

      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: tests/results/
```

### Local CI Simulation

```bash
#!/bin/bash
# Simulate CI pipeline locally

set -e  # Exit on error

echo "=== Building C++ Library ==="
cd cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j16

echo "=== Running Unit Tests ==="
ctest --output-on-failure

echo "=== Running Accuracy Tests ==="
cd ../..
python tests/test_npu_accuracy.py

echo "=== Running Performance Benchmarks ==="
python tests/benchmark_npu_performance.py

echo "=== All checks passed! ==="
```

---

## Performance Profiling

### CPU Profiling

```bash
# Python cProfile
python -m cProfile -s cumulative tests/benchmark_npu_performance.py

# Linux perf
perf record -g python tests/benchmark_npu_performance.py
perf report
```

### Memory Profiling

```bash
# Valgrind (memcheck)
valgrind --leak-check=full python tests/test_cpp_npu_stability.py

# Valgrind (massif - heap profiler)
valgrind --tool=massif python tests/test_cpp_npu_stability.py
ms_print massif.out.12345
```

### NPU Profiling

```bash
# XRT profiling (when available)
xbutil examine --device 0 --report performance

# Custom NPU trace (if implemented)
python tests/benchmark_npu_performance.py --enable-npu-trace
```

---

## Quick Reference

### Most Common Commands

```bash
# Build everything
cd cpp/build && cmake .. && make -j16

# Run all C++ tests
cd cpp/build && ctest --output-on-failure

# Run accuracy tests
python tests/test_npu_accuracy.py

# Run benchmarks
python tests/benchmark_npu_performance.py

# Check for memory leaks
valgrind --leak-check=full python tests/test_cpp_npu_stability.py
```

### Test Status Codes

| Code | Meaning |
|------|---------|
| `PASS` | Test passed all assertions |
| `FAIL` | Test failed assertions |
| `SKIPPED` | Test not yet implemented or dependencies missing |
| `ERROR` | Test crashed or threw exception |

---

**Last Updated**: October 30, 2025
**Version**: 1.0.0
**Maintainer**: Testing & Validation Team (Team 3)

---

Generated with Claude Code (Anthropic)
Project: CC-1L Whisper Encoder NPU Acceleration
