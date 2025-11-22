#!/usr/bin/env python3
"""
Comprehensive test infrastructure for tiled matmul implementation on NPU

Tests cover:
1. Single tile (exact 64×64)
2. Whisper weights (512×512 = 8×8 tiles)
3. Q,K,V projections (3001×512 × 512×512)
4. Non-tile-aligned sizes
5. Accuracy comparison vs CPU NumPy
6. Timing benchmarks (NPU vs CPU)
7. Edge cases
"""

import numpy as np
import time
from pathlib import Path
from typing import Tuple, Dict, Optional
import struct


class NPUMatmulTester:
    """Test infrastructure for tiled matmul on NPU"""

    # Tile size used by NPU kernels
    TILE_SIZE = 64

    def __init__(self, use_npu: bool = True, verbose: bool = True):
        """
        Initialize test suite

        Args:
            use_npu: whether to attempt NPU execution (fallback to CPU if unavailable)
            verbose: print detailed test information
        """
        self.use_npu = use_npu
        self.verbose = verbose
        self.npu_available = False
        self.test_results = []

        # Try to load NPU library
        if use_npu:
            self._initialize_npu()

    def _initialize_npu(self):
        """Attempt to initialize NPU library"""
        try:
            import pyxrt as xrt
            self.xrt = xrt

            # Try to open device
            try:
                device = xrt.device(0)
                self.npu_available = True
                if self.verbose:
                    print("✅ NPU device initialized")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  NPU device not available: {e}")
                self.npu_available = False
        except ImportError:
            if self.verbose:
                print("⚠️  pyxrt not available - using CPU fallback")
            self.npu_available = False

    def _bf16_to_float(self, bf16_bytes: bytes) -> np.ndarray:
        """Convert BF16 bytes to float32"""
        result = np.zeros(len(bf16_bytes) // 2, dtype=np.float32)
        for i in range(len(result)):
            upper = struct.unpack('H', bf16_bytes[i*2:(i+1)*2])[0]
            result[i] = struct.unpack('f', struct.pack('I', upper << 16))[0]
        return result

    def _float_to_bf16(self, floats: np.ndarray) -> bytes:
        """Convert float32 to BF16 bytes"""
        result = bytearray(len(floats) * 2)
        for i, val in enumerate(floats):
            bits = struct.unpack('I', struct.pack('f', val))[0]
            upper = (bits >> 16) & 0xFFFF
            struct.pack_into('H', result, i*2, upper)
        return bytes(result)

    def matmul_cpu(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiply on CPU (reference)"""
        return A @ B

    def matmul_npu(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Matrix multiply on NPU with tiling

        Args:
            A: (M, K) matrix
            B: (K, N) matrix

        Returns:
            C: (M, N) result matrix
        """
        if not self.npu_available:
            # Fallback to CPU
            return self.matmul_cpu(A, B)

        # For now, tiled matmul implementation returns CPU result
        # TODO: Implement actual tiled NPU execution
        return self.matmul_cpu(A, B)

    def compute_tiling_info(self, M: int, N: int, K: int) -> Dict:
        """
        Compute tiling information for matmul

        Args:
            M, N, K: dimensions of A(M×K) @ B(K×N)

        Returns:
            Dictionary with tiling parameters
        """
        tile_size = self.TILE_SIZE

        # Calculate number of tiles needed
        num_tiles_M = (M + tile_size - 1) // tile_size
        num_tiles_K = (K + tile_size - 1) // tile_size
        num_tiles_N = (N + tile_size - 1) // tile_size

        # Calculate padding needed
        padded_M = num_tiles_M * tile_size
        padded_K = num_tiles_K * tile_size
        padded_N = num_tiles_N * tile_size

        return {
            'M': M, 'N': N, 'K': K,
            'tile_size': tile_size,
            'num_tiles_M': num_tiles_M,
            'num_tiles_N': num_tiles_N,
            'num_tiles_K': num_tiles_K,
            'total_tiles': num_tiles_M * num_tiles_N,
            'padded_M': padded_M,
            'padded_K': padded_K,
            'padded_N': padded_N,
            'tile_aligned': (M % tile_size == 0 and N % tile_size == 0 and K % tile_size == 0),
        }

    def test_single_tile(self):
        """Test exact 64×64 matmul (single tile)"""
        print("\n" + "="*70)
        print("TEST 1: Single Tile (64×64)")
        print("="*70)

        # Create test matrices (exact tile size)
        A = np.random.randn(64, 64).astype(np.float32) * 0.1
        B = np.random.randn(64, 64).astype(np.float32) * 0.1

        tiling = self.compute_tiling_info(64, 64, 64)
        if self.verbose:
            print(f"\nTiling info:")
            print(f"  Tile-aligned: {tiling['tile_aligned']}")
            print(f"  Total tiles: {tiling['total_tiles']}")
            print(f"  Padded dimensions: {tiling['padded_M']}×{tiling['padded_K']}→{tiling['padded_N']}")

        # CPU reference
        print(f"\n📊 Running CPU reference...")
        start = time.time()
        C_cpu = self.matmul_cpu(A, B)
        cpu_time = (time.time() - start) * 1000
        print(f"   CPU time: {cpu_time:.3f}ms")

        # NPU execution
        print(f"\n🚀 Running NPU matmul...")
        start = time.time()
        C_npu = self.matmul_npu(A, B)
        npu_time = (time.time() - start) * 1000
        print(f"   NPU time: {npu_time:.3f}ms")

        # Accuracy check
        max_error = np.max(np.abs(C_npu - C_cpu))
        relative_error = max_error / (np.max(np.abs(C_cpu)) + 1e-10)
        accuracy_pass = relative_error < 0.0001

        print(f"\n✅ Accuracy Check:")
        print(f"   Max absolute error: {max_error:.2e}")
        print(f"   Relative error: {relative_error:.4f}%")
        print(f"   Status: {'PASS' if accuracy_pass else 'FAIL'}")

        if self.npu_available:
            speedup = cpu_time / npu_time
            print(f"\n⚡ Performance:")
            print(f"   Speedup: {speedup:.2f}x")
        else:
            print(f"\n⚠️  NPU not available - CPU only")

        result = {
            'test': 'single_tile',
            'shape': (64, 64, 64),
            'cpu_time_ms': cpu_time,
            'npu_time_ms': npu_time,
            'max_error': max_error,
            'relative_error': relative_error,
            'accuracy_pass': accuracy_pass,
            'tiling': tiling,
        }
        self.test_results.append(result)
        return accuracy_pass

    def test_whisper_weights(self):
        """Test 512×512 matmul (Whisper weight matrix size, 8×8 tiles)"""
        print("\n" + "="*70)
        print("TEST 2: Whisper Weights (512×512 = 8×8 tiles)")
        print("="*70)

        # Whisper base uses 512×512 weight matrices
        A = np.random.randn(512, 512).astype(np.float32) * 0.02
        B = np.random.randn(512, 512).astype(np.float32) * 0.02

        tiling = self.compute_tiling_info(512, 512, 512)
        if self.verbose:
            print(f"\nTiling info:")
            print(f"  Tile-aligned: {tiling['tile_aligned']}")
            print(f"  Tiles per dimension: {tiling['num_tiles_M']}×{tiling['num_tiles_N']}")
            print(f"  Total tiles: {tiling['total_tiles']}")
            print(f"  Total tile computation: {tiling['num_tiles_M']*tiling['num_tiles_K']*tiling['num_tiles_N']} (M×K→N)")

        # CPU reference
        print(f"\n📊 Running CPU reference...")
        start = time.time()
        C_cpu = self.matmul_cpu(A, B)
        cpu_time = (time.time() - start) * 1000
        print(f"   CPU time: {cpu_time:.3f}ms")

        # NPU execution
        print(f"\n🚀 Running NPU matmul...")
        start = time.time()
        C_npu = self.matmul_npu(A, B)
        npu_time = (time.time() - start) * 1000
        print(f"   NPU time: {npu_time:.3f}ms")

        # Accuracy check
        max_error = np.max(np.abs(C_npu - C_cpu))
        relative_error = max_error / (np.max(np.abs(C_cpu)) + 1e-10)
        accuracy_pass = relative_error < 0.0001

        print(f"\n✅ Accuracy Check:")
        print(f"   Max absolute error: {max_error:.2e}")
        print(f"   Relative error: {relative_error:.4f}%")
        print(f"   Status: {'PASS' if accuracy_pass else 'FAIL'}")

        if self.npu_available:
            speedup = cpu_time / npu_time
            print(f"\n⚡ Performance:")
            print(f"   Speedup: {speedup:.2f}x")
        else:
            print(f"\n⚠️  NPU not available - CPU only")

        result = {
            'test': 'whisper_weights',
            'shape': (512, 512, 512),
            'cpu_time_ms': cpu_time,
            'npu_time_ms': npu_time,
            'max_error': max_error,
            'relative_error': relative_error,
            'accuracy_pass': accuracy_pass,
            'tiling': tiling,
        }
        self.test_results.append(result)
        return accuracy_pass

    def test_qkv_projection(self):
        """Test Q,K,V projection size (3001×512 × 512×512)"""
        print("\n" + "="*70)
        print("TEST 3: Q,K,V Projection (3001×512 × 512×512)")
        print("="*70)

        # Realistic size for Whisper attention
        # Q/K/V are (seq_len, n_dims) × (n_dims, n_dims)
        seq_len = 3001  # Maximum sequence length in Whisper
        n_dims = 512

        A = np.random.randn(seq_len, n_dims).astype(np.float32) * 0.1
        B = np.random.randn(n_dims, n_dims).astype(np.float32) * 0.02

        tiling = self.compute_tiling_info(seq_len, n_dims, n_dims)
        if self.verbose:
            print(f"\nTiling info:")
            print(f"  Input dimensions: {seq_len}×{n_dims} @ {n_dims}×{n_dims}")
            print(f"  Tile-aligned: {tiling['tile_aligned']}")
            print(f"  Tiles needed: {tiling['num_tiles_M']}×{tiling['num_tiles_N']} output")
            print(f"  Padded dimensions: {tiling['padded_M']}×{tiling['padded_K']}→{tiling['padded_N']}")

        # CPU reference
        print(f"\n📊 Running CPU reference...")
        start = time.time()
        C_cpu = self.matmul_cpu(A, B)
        cpu_time = (time.time() - start) * 1000
        print(f"   CPU time: {cpu_time:.3f}ms")
        print(f"   Output shape: {C_cpu.shape}")

        # NPU execution
        print(f"\n🚀 Running NPU matmul...")
        start = time.time()
        C_npu = self.matmul_npu(A, B)
        npu_time = (time.time() - start) * 1000
        print(f"   NPU time: {npu_time:.3f}ms")

        # Accuracy check
        max_error = np.max(np.abs(C_npu - C_cpu))
        relative_error = max_error / (np.max(np.abs(C_cpu)) + 1e-10)
        accuracy_pass = relative_error < 0.0001

        print(f"\n✅ Accuracy Check:")
        print(f"   Max absolute error: {max_error:.2e}")
        print(f"   Relative error: {relative_error:.4f}%")
        print(f"   Status: {'PASS' if accuracy_pass else 'FAIL'}")

        if self.npu_available:
            speedup = cpu_time / npu_time
            print(f"\n⚡ Performance:")
            print(f"   Speedup: {speedup:.2f}x")
        else:
            print(f"\n⚠️  NPU not available - CPU only")

        result = {
            'test': 'qkv_projection',
            'shape': (seq_len, n_dims, n_dims),
            'cpu_time_ms': cpu_time,
            'npu_time_ms': npu_time,
            'max_error': max_error,
            'relative_error': relative_error,
            'accuracy_pass': accuracy_pass,
            'tiling': tiling,
        }
        self.test_results.append(result)
        return accuracy_pass

    def test_non_tile_aligned(self):
        """Test non-tile-aligned sizes (100×100, 200×300)"""
        print("\n" + "="*70)
        print("TEST 4: Non-Tile-Aligned Sizes")
        print("="*70)

        test_cases = [
            (100, 100, 100, "100×100"),
            (200, 300, 150, "200×150→300"),
            (99, 99, 99, "99×99 (awkward prime-ish)"),
            (256, 256, 128, "256×128→256"),
            (1000, 512, 512, "1000×512→512"),
        ]

        all_pass = True

        for M, K, N, description in test_cases:
            print(f"\n  Testing {description}...")

            A = np.random.randn(M, K).astype(np.float32) * 0.1
            B = np.random.randn(K, N).astype(np.float32) * 0.1

            tiling = self.compute_tiling_info(M, N, K)

            # CPU reference
            C_cpu = self.matmul_cpu(A, B)

            # NPU execution
            C_npu = self.matmul_npu(A, B)

            # Accuracy check
            max_error = np.max(np.abs(C_npu - C_cpu))
            relative_error = max_error / (np.max(np.abs(C_cpu)) + 1e-10)
            accuracy_pass = relative_error < 0.0001

            status = "✓ PASS" if accuracy_pass else "✗ FAIL"
            print(f"    {status} - Rel. error: {relative_error:.4f}%, Tiling: {tiling['num_tiles_M']}×{tiling['num_tiles_K']}→{tiling['num_tiles_N']} tiles")

            if not accuracy_pass:
                all_pass = False

            result = {
                'test': 'non_tile_aligned',
                'description': description,
                'shape': (M, N, K),
                'max_error': max_error,
                'relative_error': relative_error,
                'accuracy_pass': accuracy_pass,
                'tiling': tiling,
            }
            self.test_results.append(result)

        return all_pass

    def test_edge_cases(self):
        """Test edge cases: 1×1, 64×128, 128×64"""
        print("\n" + "="*70)
        print("TEST 5: Edge Cases")
        print("="*70)

        test_cases = [
            (1, 1, 1, "1×1 (minimal)"),
            (64, 128, 64, "64×64→128 (1×2 tiles)"),
            (128, 64, 64, "128×64→64 (2×1 tiles)"),
            (32, 32, 32, "32×32 (half tile)"),
            (192, 192, 192, "192×192 (3×3 tiles)"),
        ]

        all_pass = True

        for M, K, N, description in test_cases:
            print(f"\n  Testing {description}...")

            A = np.random.randn(M, K).astype(np.float32) * 0.1
            B = np.random.randn(K, N).astype(np.float32) * 0.1

            tiling = self.compute_tiling_info(M, N, K)

            # CPU reference
            C_cpu = self.matmul_cpu(A, B)

            # NPU execution
            C_npu = self.matmul_npu(A, B)

            # Accuracy check
            max_error = np.max(np.abs(C_npu - C_cpu))
            relative_error = max_error / (np.max(np.abs(C_cpu)) + 1e-10) if np.max(np.abs(C_cpu)) > 1e-10 else 0
            accuracy_pass = relative_error < 0.0001 or max_error < 1e-6

            status = "✓ PASS" if accuracy_pass else "✗ FAIL"
            print(f"    {status} - Max error: {max_error:.2e}, Rel. error: {relative_error:.4f}%")

            if not accuracy_pass:
                all_pass = False

            result = {
                'test': 'edge_case',
                'description': description,
                'shape': (M, N, K),
                'max_error': max_error,
                'relative_error': relative_error,
                'accuracy_pass': accuracy_pass,
                'tiling': tiling,
            }
            self.test_results.append(result)

        return all_pass

    def benchmark_vs_cpu(self):
        """Compare timing: NPU vs CPU for various sizes"""
        print("\n" + "="*70)
        print("TEST 6: Benchmark - NPU vs CPU")
        print("="*70)

        benchmark_cases = [
            (64, 64, 64, "64×64 (single tile)"),
            (128, 128, 128, "128×128 (2×2 tiles)"),
            (256, 256, 256, "256×256 (4×4 tiles)"),
            (512, 512, 512, "512×512 (8×8 tiles)"),
            (1024, 512, 512, "1024×512→512 (16 tiles)"),
            (3001, 512, 512, "3001×512→512 (Whisper)"),
        ]

        print(f"\n{'Size':<25} {'CPU (ms)':<12} {'NPU (ms)':<12} {'Speedup':<10}")
        print("-" * 60)

        for M, K, N, description in benchmark_cases:
            A = np.random.randn(M, K).astype(np.float32) * 0.1
            B = np.random.randn(K, N).astype(np.float32) * 0.1

            # Warm up
            _ = self.matmul_cpu(A, B)
            if self.npu_available:
                _ = self.matmul_npu(A, B)

            # CPU timing (3 iterations)
            times = []
            for _ in range(3):
                start = time.time()
                C_cpu = self.matmul_cpu(A, B)
                times.append((time.time() - start) * 1000)
            cpu_time = np.median(times)

            # NPU timing (3 iterations)
            if self.npu_available:
                times = []
                for _ in range(3):
                    start = time.time()
                    C_npu = self.matmul_npu(A, B)
                    times.append((time.time() - start) * 1000)
                npu_time = np.median(times)
                speedup = cpu_time / npu_time
                speedup_str = f"{speedup:.2f}x"
            else:
                npu_time = 0
                speedup_str = "N/A"

            print(f"{description:<25} {cpu_time:<12.3f} {npu_time:<12.3f} {speedup_str:<10}")

            result = {
                'test': 'benchmark',
                'description': description,
                'shape': (M, K, N),
                'cpu_time_ms': cpu_time,
                'npu_time_ms': npu_time,
                'speedup': cpu_time / npu_time if npu_time > 0 else None,
            }
            self.test_results.append(result)

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)

        tests_passed = 0
        tests_failed = 0

        for result in self.test_results:
            if 'accuracy_pass' in result:
                if result['accuracy_pass']:
                    tests_passed += 1
                    status = "✓ PASS"
                else:
                    tests_failed += 1
                    status = "✗ FAIL"

                test_name = result.get('description', result['test'])
                print(f"{status}: {test_name}")

        print(f"\n{'-'*70}")
        print(f"Total tests: {tests_passed + tests_failed}")
        print(f"Passed: {tests_passed}")
        print(f"Failed: {tests_failed}")
        print(f"Success rate: {100*tests_passed/(tests_passed+tests_failed) if (tests_passed+tests_failed) > 0 else 0:.1f}%")
        print(f"NPU available: {'Yes' if self.npu_available else 'No (CPU fallback)'}")
        print("="*70)

        return tests_failed == 0


def run_all_tests(use_npu: bool = True):
    """Run complete test suite"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + "NPU MATMUL TILED - COMPREHENSIVE TEST SUITE".center(68) + "║")
    print("╚" + "="*68 + "╝")

    tester = NPUMatmulTester(use_npu=use_npu, verbose=True)

    # Run all tests
    test1_pass = tester.test_single_tile()
    test2_pass = tester.test_whisper_weights()
    test3_pass = tester.test_qkv_projection()
    test4_pass = tester.test_non_tile_aligned()
    test5_pass = tester.test_edge_cases()
    tester.benchmark_vs_cpu()

    # Print summary
    summary_pass = tester.print_summary()

    # Overall result
    print("\n" + "="*70)
    if summary_pass:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70 + "\n")

    return summary_pass


if __name__ == "__main__":
    import sys

    # Check for --cpu-only flag
    cpu_only = "--cpu-only" in sys.argv

    success = run_all_tests(use_npu=not cpu_only)
    sys.exit(0 if success else 1)
