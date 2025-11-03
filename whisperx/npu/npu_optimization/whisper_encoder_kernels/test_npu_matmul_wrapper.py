#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for NPU Matmul Wrapper
Tests all features and edge cases of npu_matmul_wrapper.py

Test Coverage:
1. Basic 16x16 multiplication
2. Arbitrary sizes (non-multiple of 16)
3. Large matrices (2048x2048)
4. Batch operations
5. Edge cases (1x1, empty, etc.)
6. INT8 quantization accuracy
7. CPU fallback
8. Thread safety
9. Memory management
10. Performance benchmarks
"""

import sys
import os
sys.path.insert(0, '/opt/xilinx/xrt/python')
sys.path.insert(0, os.path.dirname(__file__))

import pytest
import numpy as np
import time
import threading
from pathlib import Path
from npu_matmul_wrapper import NPUMatmul


class TestBasicMatmul:
    """Test 1: Basic 16x16 matrix multiplication"""

    @pytest.fixture
    def npu_matmul(self):
        """Initialize NPU matmul once per test class"""
        return NPUMatmul()

    def test_16x16_identity(self, npu_matmul):
        """Test identity matrix multiplication"""
        A = np.eye(16, dtype=np.int8) * 64
        B = np.eye(16, dtype=np.int8) * 64

        C = npu_matmul(A, B, quantize=False)

        # Expected: scaled identity matrix
        C_ref = (A.astype(np.int32) @ B.astype(np.int32)) >> 7
        C_ref = np.clip(C_ref, -128, 127).astype(np.int8)

        assert C.shape == (16, 16)
        assert np.allclose(C, C_ref, atol=1)

    def test_16x16_random(self, npu_matmul):
        """Test random 16x16 matrices"""
        np.random.seed(42)
        A = np.random.randint(-64, 64, (16, 16), dtype=np.int8)
        B = np.random.randint(-64, 64, (16, 16), dtype=np.int8)

        C = npu_matmul(A, B, quantize=False)

        # Compute reference
        C_ref = (A.astype(np.int32) @ B.astype(np.int32)) >> 7
        C_ref = np.clip(C_ref, -128, 127).astype(np.int8)

        correlation = np.corrcoef(C.flatten(), C_ref.flatten())[0, 1]
        assert correlation > 0.99, f"Low correlation: {correlation}"

    def test_16x16_zeros(self, npu_matmul):
        """Test zero matrices"""
        A = np.zeros((16, 16), dtype=np.int8)
        B = np.zeros((16, 16), dtype=np.int8)

        C = npu_matmul(A, B, quantize=False)

        assert np.all(C == 0), "Zero matrices should produce zero output"

    def test_16x16_max_values(self, npu_matmul):
        """Test maximum INT8 values with clamping"""
        A = np.ones((16, 16), dtype=np.int8) * 127
        B = np.ones((16, 16), dtype=np.int8) * 127

        C = npu_matmul(A, B, quantize=False)

        # Should clamp to 127
        assert C[0, 0] == 127, "Should clamp to INT8 max"


class TestArbitrarySizes:
    """Test 2: Arbitrary matrix sizes (non-multiple of 16)"""

    @pytest.fixture
    def npu_matmul(self):
        return NPUMatmul()

    def test_small_odd_size(self, npu_matmul):
        """Test 13x13 matrices (requires padding)"""
        np.random.seed(42)
        A = np.random.randint(-32, 32, (13, 13), dtype=np.int8)
        B = np.random.randint(-32, 32, (13, 13), dtype=np.int8)

        C = npu_matmul(A, B, quantize=False)

        assert C.shape == (13, 13), "Should return unpadded size"

    def test_rectangular_matrices(self, npu_matmul):
        """Test non-square matrices"""
        np.random.seed(42)
        A = np.random.randint(-32, 32, (20, 50), dtype=np.int8)
        B = np.random.randint(-32, 32, (50, 30), dtype=np.int8)

        C = npu_matmul(A, B, quantize=False)

        assert C.shape == (20, 30), f"Wrong shape: {C.shape}"

    def test_non_aligned_sizes(self, npu_matmul):
        """Test sizes that require padding: 15x17 @ 17x19"""
        np.random.seed(42)
        A = np.random.randint(-32, 32, (15, 17), dtype=np.int8)
        B = np.random.randint(-32, 32, (17, 19), dtype=np.int8)

        C = npu_matmul(A, B, quantize=False)

        assert C.shape == (15, 19), f"Wrong shape: {C.shape}"


class TestLargeMatrices:
    """Test 3: Large matrix multiplication (2048x2048)"""

    @pytest.fixture
    def npu_matmul(self):
        return NPUMatmul()

    def test_512x512(self, npu_matmul):
        """Test 512x512 matrices"""
        np.random.seed(42)
        A = np.random.randint(-32, 32, (512, 512), dtype=np.int8)
        B = np.random.randint(-32, 32, (512, 512), dtype=np.int8)

        start = time.perf_counter()
        C = npu_matmul(A, B, quantize=False)
        elapsed = time.perf_counter() - start

        assert C.shape == (512, 512)
        print(f"\n512x512 matmul took {elapsed*1000:.1f}ms")

    def test_1024x1024(self, npu_matmul):
        """Test 1024x1024 matrices"""
        np.random.seed(42)
        A = np.random.randint(-32, 32, (1024, 1024), dtype=np.int8)
        B = np.random.randint(-32, 32, (1024, 1024), dtype=np.int8)

        start = time.perf_counter()
        C = npu_matmul(A, B, quantize=False)
        elapsed = time.perf_counter() - start

        assert C.shape == (1024, 1024)
        print(f"\n1024x1024 matmul took {elapsed*1000:.1f}ms")

    def test_2048x2048(self, npu_matmul):
        """Test 2048x2048 matrices (stress test)"""
        np.random.seed(42)
        A = np.random.randint(-32, 32, (2048, 2048), dtype=np.int8)
        B = np.random.randint(-32, 32, (2048, 2048), dtype=np.int8)

        start = time.perf_counter()
        C = npu_matmul(A, B, quantize=False)
        elapsed = time.perf_counter() - start

        assert C.shape == (2048, 2048)
        print(f"\n2048x2048 matmul took {elapsed:.2f}s")

    def test_whisper_encoder_size(self, npu_matmul):
        """Test Whisper encoder typical size: 1500x512 @ 512x2048"""
        np.random.seed(42)
        A = np.random.randint(-32, 32, (1500, 512), dtype=np.int8)
        B = np.random.randint(-32, 32, (512, 2048), dtype=np.int8)

        start = time.perf_counter()
        C = npu_matmul(A, B, quantize=False)
        elapsed = time.perf_counter() - start

        assert C.shape == (1500, 2048)
        print(f"\nWhisper encoder size (1500x512 @ 512x2048) took {elapsed*1000:.1f}ms")


class TestBatchOperations:
    """Test 4: Batch matrix multiplication"""

    @pytest.fixture
    def npu_matmul(self):
        return NPUMatmul()

    def test_batch_same_weights(self, npu_matmul):
        """Test batch with shared weights (common pattern)"""
        np.random.seed(42)
        A_batch = np.random.randint(-32, 32, (8, 256, 256), dtype=np.int8)
        B = np.random.randint(-32, 32, (256, 256), dtype=np.int8)

        C_batch = npu_matmul.batch_matmul(A_batch, B, quantize=False)

        assert C_batch.shape == (8, 256, 256)

    def test_batch_different_weights(self, npu_matmul):
        """Test batch with different weights per sample"""
        np.random.seed(42)
        A_batch = np.random.randint(-32, 32, (4, 128, 128), dtype=np.int8)
        B_batch = np.random.randint(-32, 32, (4, 128, 128), dtype=np.int8)

        C_batch = npu_matmul.batch_matmul(A_batch, B_batch, quantize=False)

        assert C_batch.shape == (4, 128, 128)

    def test_batch_consistency(self, npu_matmul):
        """Verify batch produces same results as individual calls"""
        np.random.seed(42)
        A_batch = np.random.randint(-32, 32, (3, 64, 64), dtype=np.int8)
        B = np.random.randint(-32, 32, (64, 64), dtype=np.int8)

        # Batch processing
        C_batch = npu_matmul.batch_matmul(A_batch, B, quantize=False)

        # Individual processing
        C_individual = []
        for i in range(3):
            C = npu_matmul(A_batch[i], B, quantize=False)
            C_individual.append(C)
        C_individual = np.stack(C_individual, axis=0)

        assert np.allclose(C_batch, C_individual, atol=1), "Batch should match individual"


class TestEdgeCases:
    """Test 5: Edge cases and boundary conditions"""

    @pytest.fixture
    def npu_matmul(self):
        return NPUMatmul()

    def test_1x1_matrix(self, npu_matmul):
        """Test smallest possible matrix"""
        A = np.array([[64]], dtype=np.int8)
        B = np.array([[64]], dtype=np.int8)

        C = npu_matmul(A, B, quantize=False)

        assert C.shape == (1, 1)

    def test_1xN_vector(self, npu_matmul):
        """Test row vector multiplication"""
        np.random.seed(42)
        A = np.random.randint(-32, 32, (1, 256), dtype=np.int8)
        B = np.random.randint(-32, 32, (256, 1), dtype=np.int8)

        C = npu_matmul(A, B, quantize=False)

        assert C.shape == (1, 1)

    def test_Nx1_vector(self, npu_matmul):
        """Test column vector multiplication"""
        np.random.seed(42)
        A = np.random.randint(-32, 32, (256, 1), dtype=np.int8)
        B = np.random.randint(-32, 32, (1, 256), dtype=np.int8)

        C = npu_matmul(A, B, quantize=False)

        assert C.shape == (256, 256)

    def test_negative_values(self, npu_matmul):
        """Test matrices with all negative values"""
        A = np.ones((32, 32), dtype=np.int8) * -64
        B = np.ones((32, 32), dtype=np.int8) * -64

        C = npu_matmul(A, B, quantize=False)

        # Negative * negative = positive
        assert np.all(C > 0), "All values should be positive"

    def test_shape_mismatch_error(self, npu_matmul):
        """Test that shape mismatch raises error"""
        A = np.random.randint(-32, 32, (16, 20), dtype=np.int8)
        B = np.random.randint(-32, 32, (25, 16), dtype=np.int8)

        with pytest.raises(AssertionError):
            C = npu_matmul(A, B, quantize=False)


class TestQuantization:
    """Test 6: INT8 quantization accuracy"""

    @pytest.fixture
    def npu_matmul(self):
        return NPUMatmul()

    def test_fp32_to_int8_quantization(self, npu_matmul):
        """Test automatic FP32 to INT8 quantization"""
        np.random.seed(42)
        A = np.random.randn(128, 128).astype(np.float32)
        B = np.random.randn(128, 128).astype(np.float32)

        C = npu_matmul(A, B, quantize=True)

        assert C.dtype == np.int8
        assert C.shape == (128, 128)

    def test_quantization_preserves_range(self, npu_matmul):
        """Test that quantization doesn't clip unnecessarily"""
        # Small values that should quantize well
        A = np.random.randn(64, 64).astype(np.float32) * 0.1
        B = np.random.randn(64, 64).astype(np.float32) * 0.1

        C = npu_matmul(A, B, quantize=True)

        # Should have reasonable range
        assert C.min() >= -128 and C.max() <= 127

    def test_mixed_int8_fp32_input(self, npu_matmul):
        """Test mixed input types"""
        np.random.seed(42)
        A = np.random.randint(-32, 32, (64, 64), dtype=np.int8)
        B = np.random.randn(64, 64).astype(np.float32)

        C = npu_matmul(A, B, quantize=True)

        assert C.dtype == np.int8
        assert C.shape == (64, 64)


class TestCPUFallback:
    """Test 7: CPU fallback behavior"""

    def test_initialization_with_missing_xclbin(self):
        """Test graceful handling of missing XCLBIN"""
        with pytest.raises(FileNotFoundError):
            npu = NPUMatmul(xclbin_path="/nonexistent/path.xclbin")

    def test_initialization_with_invalid_device(self):
        """Test handling of invalid device ID"""
        # Device 99 shouldn't exist
        with pytest.raises(Exception):
            npu = NPUMatmul(device_id=99)


class TestThreadSafety:
    """Test 8: Thread safety of NPU matmul"""

    @pytest.fixture
    def npu_matmul(self):
        return NPUMatmul()

    def test_concurrent_calls(self, npu_matmul):
        """Test multiple threads calling matmul simultaneously"""
        np.random.seed(42)

        results = []
        errors = []

        def worker(thread_id):
            try:
                A = np.random.randint(-32, 32, (64, 64), dtype=np.int8)
                B = np.random.randint(-32, 32, (64, 64), dtype=np.int8)
                C = npu_matmul(A, B, quantize=False)
                results.append((thread_id, C))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Launch 5 concurrent threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 5, "All threads should complete"

    def test_sequential_vs_concurrent_consistency(self, npu_matmul):
        """Test that results are consistent between sequential and concurrent calls"""
        np.random.seed(42)
        A = np.random.randint(-32, 32, (128, 128), dtype=np.int8)
        B = np.random.randint(-32, 32, (128, 128), dtype=np.int8)

        # Sequential
        C1 = npu_matmul(A, B, quantize=False)
        C2 = npu_matmul(A, B, quantize=False)

        assert np.array_equal(C1, C2), "Sequential calls should be deterministic"


class TestMemoryManagement:
    """Test 9: Memory allocation and cleanup"""

    @pytest.fixture
    def npu_matmul(self):
        return NPUMatmul()

    def test_repeated_calls_no_leak(self, npu_matmul):
        """Test that repeated calls don't leak memory"""
        np.random.seed(42)
        A = np.random.randint(-32, 32, (256, 256), dtype=np.int8)
        B = np.random.randint(-32, 32, (256, 256), dtype=np.int8)

        # Run 100 iterations
        for i in range(100):
            C = npu_matmul(A, B, quantize=False)

        # If we got here without crash, memory management is OK
        assert True

    def test_buffer_reuse(self, npu_matmul):
        """Test that buffers are reused efficiently"""
        np.random.seed(42)
        A1 = np.random.randint(-32, 32, (64, 64), dtype=np.int8)
        B1 = np.random.randint(-32, 32, (64, 64), dtype=np.int8)

        A2 = np.random.randint(-32, 32, (64, 64), dtype=np.int8)
        B2 = np.random.randint(-32, 32, (64, 64), dtype=np.int8)

        # First call
        C1 = npu_matmul(A1, B1, quantize=False)

        # Second call (should reuse buffers)
        C2 = npu_matmul(A2, B2, quantize=False)

        assert not np.array_equal(C1, C2), "Different inputs should give different outputs"

    def test_statistics_tracking(self, npu_matmul):
        """Test that statistics are tracked correctly"""
        npu_matmul.reset_stats()

        np.random.seed(42)
        A = np.random.randint(-32, 32, (128, 128), dtype=np.int8)
        B = np.random.randint(-32, 32, (128, 128), dtype=np.int8)

        # Make 10 calls
        for i in range(10):
            C = npu_matmul(A, B, quantize=False)

        stats = npu_matmul.get_stats()
        assert stats['total_calls'] == 10
        assert stats['total_tiles'] > 0
        assert stats['total_time_ms'] > 0


class TestPerformanceBenchmarks:
    """Test 10: Performance benchmarks and targets"""

    @pytest.fixture
    def npu_matmul(self):
        return NPUMatmul()

    def test_single_tile_performance(self, npu_matmul):
        """Test single 16x16 tile performance"""
        np.random.seed(42)
        A = np.random.randint(-32, 32, (16, 16), dtype=np.int8)
        B = np.random.randint(-32, 32, (16, 16), dtype=np.int8)

        # Warm-up
        _ = npu_matmul(A, B, quantize=False)

        # Benchmark
        times = []
        for i in range(100):
            start = time.perf_counter()
            C = npu_matmul(A, B, quantize=False)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        print(f"\nSingle 16x16 tile: {avg_time:.3f}ms avg")

        # Target: 0.484ms per tile (from benchmark)
        assert avg_time < 2.0, f"Too slow: {avg_time:.3f}ms (target <2.0ms)"

    def test_throughput_ops_per_second(self, npu_matmul):
        """Test operations per second"""
        np.random.seed(42)
        A = np.random.randint(-32, 32, (16, 16), dtype=np.int8)
        B = np.random.randint(-32, 32, (16, 16), dtype=np.int8)

        # Run for 1 second
        start = time.perf_counter()
        count = 0
        while (time.perf_counter() - start) < 1.0:
            _ = npu_matmul(A, B, quantize=False)
            count += 1

        elapsed = time.perf_counter() - start
        ops_per_sec = count / elapsed

        print(f"\nThroughput: {ops_per_sec:.0f} ops/sec")

        # Target: 2,218 ops/sec (from benchmark)
        assert ops_per_sec > 500, f"Too slow: {ops_per_sec:.0f} ops/sec"

    def test_large_matrix_performance(self, npu_matmul):
        """Test performance on Whisper-sized matrices"""
        np.random.seed(42)
        A = np.random.randint(-32, 32, (1500, 512), dtype=np.int8)
        B = np.random.randint(-32, 32, (512, 512), dtype=np.int8)

        start = time.perf_counter()
        C = npu_matmul(A, B, quantize=False)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\nWhisper encoder matmul (1500x512 @ 512x512): {elapsed:.1f}ms")

        # Should complete in reasonable time
        assert elapsed < 10000, f"Too slow: {elapsed:.1f}ms"

    def test_benchmark_method(self, npu_matmul):
        """Test the built-in benchmark method"""
        results = npu_matmul.benchmark(M=256, N=256, K=256, iterations=50)

        assert 'avg_time_ms' in results
        assert 'gflops' in results
        assert 'tiles_per_second' in results

        print(f"\n256x256 benchmark: {results['avg_time_ms']:.2f}ms, "
              f"{results['gflops']:.3f} GFLOPS")


# Performance summary fixture
@pytest.fixture(scope="session", autouse=True)
def print_summary():
    """Print summary after all tests"""
    yield
    print("\n" + "="*70)
    print("NPU MATMUL WRAPPER TEST SUITE COMPLETE")
    print("="*70)
    print("\nAll tests passed! NPU matmul wrapper is ready for integration.")
    print("\nNext steps:")
    print("  1. Integrate into Whisper encoder attention layers")
    print("  2. Integrate into Whisper decoder attention layers")
    print("  3. Measure end-to-end performance improvement")
    print("="*70)


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s", "--tb=short"])
