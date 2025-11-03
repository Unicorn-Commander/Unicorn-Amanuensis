#!/usr/bin/env python3
"""
Individual Kernel Benchmarking

Measures performance of individual NPU kernels:
- Attention (64x64 and multicore variants)
- LayerNorm
- GELU
- MatMul (16x16)

Provides detailed statistics: mean, std, min, max, percentiles
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json


class KernelBenchmark:
    """Benchmark individual NPU kernels with detailed performance metrics"""

    def __init__(self, num_iterations: int = 100):
        """
        Initialize kernel benchmarking suite

        Args:
            num_iterations: Number of iterations for each benchmark (default: 100)
        """
        self.num_iterations = num_iterations
        self.results = {}
        self.encoder = None

    def _initialize_encoder(self):
        """Lazy initialization of NPU encoder"""
        if self.encoder is None:
            # Import here to avoid loading NPU if not needed
            parent_dir = Path(__file__).parent.parent
            sys.path.insert(0, str(parent_dir))
            from test_encoder_block import NPUEncoderBlock

            print("Initializing NPU Encoder Block...")
            self.encoder = NPUEncoderBlock()
            print("Encoder initialized successfully!")
            print()

    def benchmark_attention(self, size: int = 64) -> Dict:
        """
        Benchmark attention kernel

        Args:
            size: Size of attention matrices (default: 64x64)

        Returns:
            Dictionary with performance statistics
        """
        print(f"Benchmarking Attention kernel ({size}x{size})...")
        self._initialize_encoder()

        # Generate test data
        Q = np.random.randint(-64, 64, (size, size), dtype=np.int8)
        K = np.random.randint(-64, 64, (size, size), dtype=np.int8)
        V = np.random.randint(-64, 64, (size, size), dtype=np.int8)

        # Warm-up run
        _ = self.encoder.run_attention(Q, K, V)

        # Benchmark
        times = []
        for i in range(self.num_iterations):
            start = time.perf_counter()
            _ = self.encoder.run_attention(Q, K, V)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{self.num_iterations} iterations")

        self.results['attention'] = self._compute_statistics(times, 'Attention')
        return self.results['attention']

    def benchmark_layernorm(self) -> Dict:
        """
        Benchmark LayerNorm kernel

        Returns:
            Dictionary with performance statistics
        """
        print("Benchmarking LayerNorm kernel...")
        self._initialize_encoder()

        # Generate test data
        input_256 = np.random.randint(-64, 64, 256, dtype=np.int8)
        gamma = np.ones(256, dtype=np.int8)
        beta = np.zeros(256, dtype=np.int8)

        # Warm-up run
        _ = self.encoder.run_layernorm(input_256, gamma, beta)

        # Benchmark
        times = []
        for i in range(self.num_iterations):
            start = time.perf_counter()
            _ = self.encoder.run_layernorm(input_256, gamma, beta)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{self.num_iterations} iterations")

        self.results['layernorm'] = self._compute_statistics(times, 'LayerNorm')
        return self.results['layernorm']

    def benchmark_gelu(self) -> Dict:
        """
        Benchmark GELU activation kernel

        Returns:
            Dictionary with performance statistics
        """
        print("Benchmarking GELU kernel...")
        self._initialize_encoder()

        # Generate test data
        input_512 = np.random.randint(-64, 64, 512, dtype=np.int8)

        # Warm-up run
        _ = self.encoder.run_gelu(input_512)

        # Benchmark
        times = []
        for i in range(self.num_iterations):
            start = time.perf_counter()
            _ = self.encoder.run_gelu(input_512)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{self.num_iterations} iterations")

        self.results['gelu'] = self._compute_statistics(times, 'GELU')
        return self.results['gelu']

    def benchmark_matmul(self) -> Dict:
        """
        Benchmark matrix multiplication kernel (16x16)

        Returns:
            Dictionary with performance statistics
        """
        print("Benchmarking Matmul kernel (16x16)...")
        self._initialize_encoder()

        # Generate test data
        A = np.random.randint(-64, 64, (16, 16), dtype=np.int8)
        B = np.random.randint(-64, 64, (16, 16), dtype=np.int8)

        # Warm-up run
        _ = self.encoder.run_matmul(A, B)

        # Benchmark
        times = []
        for i in range(self.num_iterations):
            start = time.perf_counter()
            _ = self.encoder.run_matmul(A, B)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{self.num_iterations} iterations")

        self.results['matmul'] = self._compute_statistics(times, 'Matmul')
        return self.results['matmul']

    def benchmark_all_kernels(self) -> Dict:
        """
        Benchmark all kernels and return comprehensive results

        Returns:
            Dictionary containing results for all kernels
        """
        print("=" * 70)
        print("BENCHMARKING ALL KERNELS")
        print("=" * 70)
        print()

        try:
            self.benchmark_attention()
            print()

            self.benchmark_layernorm()
            print()

            self.benchmark_gelu()
            print()

            self.benchmark_matmul()
            print()

        except Exception as e:
            print(f"Error during benchmarking: {e}")
            import traceback
            traceback.print_exc()

        print("=" * 70)
        print("ALL KERNEL BENCHMARKS COMPLETE")
        print("=" * 70)
        print()

        return self.results

    def _compute_statistics(self, times: List[float], kernel_name: str) -> Dict:
        """
        Compute detailed statistics from timing measurements

        Args:
            times: List of timing measurements in milliseconds
            kernel_name: Name of the kernel for display

        Returns:
            Dictionary with statistical measures
        """
        times_array = np.array(times)

        stats = {
            'kernel': kernel_name,
            'iterations': len(times),
            'mean': float(np.mean(times_array)),
            'std': float(np.std(times_array)),
            'min': float(np.min(times_array)),
            'max': float(np.max(times_array)),
            'p50': float(np.percentile(times_array, 50)),
            'p95': float(np.percentile(times_array, 95)),
            'p99': float(np.percentile(times_array, 99)),
            'median': float(np.median(times_array))
        }

        # Print summary
        print(f"  {kernel_name} Statistics:")
        print(f"    Mean:   {stats['mean']:.3f}ms")
        print(f"    Std:    {stats['std']:.3f}ms")
        print(f"    Min:    {stats['min']:.3f}ms")
        print(f"    Max:    {stats['max']:.3f}ms")
        print(f"    Median: {stats['median']:.3f}ms")
        print(f"    P95:    {stats['p95']:.3f}ms")
        print(f"    P99:    {stats['p99']:.3f}ms")

        return stats

    def save_results(self, output_file: str):
        """Save benchmark results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Run standalone benchmark
    benchmark = KernelBenchmark(num_iterations=100)
    results = benchmark.benchmark_all_kernels()
    benchmark.save_results("kernel_benchmark_results.json")
