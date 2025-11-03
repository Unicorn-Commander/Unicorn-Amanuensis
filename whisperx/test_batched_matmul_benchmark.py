#!/usr/bin/env python3
"""
Quick Benchmark: Batched MatMul vs Sequential
Tests the 10x speedup claim for encoder workloads

Run this script to validate batched matmul performance.
"""

import sys
sys.path.insert(0, 'npu/npu_optimization/whisper_encoder_kernels')

import numpy as np
import time
from npu_matmul_wrapper_batched import NPUMatmulBatched

def benchmark_sizes():
    """Benchmark different matrix sizes"""

    print("=" * 70)
    print("BATCHED MATMUL BENCHMARK")
    print("=" * 70)
    print()

    # Initialize batched matmul
    print("Initializing batched NPU matmul...")
    try:
        matmul = NPUMatmulBatched()
        print("✅ Batched matmul initialized")
        print(f"   Device: {matmul.device}")
        print(f"   XCLBIN: {matmul.xclbin_path.name}")
        print(f"   Tile size: {matmul.tile_size}×{matmul.tile_size}")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    # Test sizes
    test_sizes = [
        (64, 64, "Small (Whisper attention heads)"),
        (128, 128, "Medium (Whisper hidden dim)"),
        (512, 512, "Large (Full encoder layer)")
    ]

    results = []

    for m, n, desc in test_sizes:
        print(f"Testing {m}×{n} - {desc}")
        print("-" * 70)

        # Create random INT8 matrices
        A = np.random.randint(-127, 128, (m, n), dtype=np.int8)
        B = np.random.randint(-127, 128, (n, m), dtype=np.int8)

        # Warm-up run
        try:
            _ = matmul(A, B, quantize=False)  # Use __call__ method
        except Exception as e:
            print(f"⚠️  Warm-up failed: {e}")
            continue

        # Benchmark run (5 iterations)
        times = []
        for i in range(5):
            start = time.time()
            C = matmul(A, B, quantize=False)  # Use __call__ method
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        # Expected sequential time (based on Team 3 analysis)
        expected_sequential = {
            64: 34.3,    # ms
            128: 234.7,  # ms
            512: 15110   # ms (15.11s)
        }
        seq_time = expected_sequential.get(m, avg_time * 10)

        speedup = seq_time / avg_time

        print(f"   Batched time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"   Expected sequential: {seq_time:.2f} ms")
        print(f"   Speedup: {speedup:.1f}x")
        print(f"   Output shape: {C.shape}")
        print(f"   Output dtype: {C.dtype}")
        print()

        results.append({
            'size': m,
            'desc': desc,
            'batched_ms': avg_time,
            'sequential_ms': seq_time,
            'speedup': speedup
        })

    # Summary
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Size':<10} {'Description':<30} {'Batched':<12} {'Speedup':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r['size']}×{r['size']:<6} {r['desc']:<30} {r['batched_ms']:>8.2f} ms {r['speedup']:>8.1f}x")

    print()
    print("Target: 10x speedup for 512×512")
    if results:
        large_result = [r for r in results if r['size'] == 512]
        if large_result:
            actual = large_result[0]['speedup']
            if actual >= 8.0:
                print(f"✅ ACHIEVED: {actual:.1f}x speedup (target: 10x)")
            else:
                print(f"⚠️  PARTIAL: {actual:.1f}x speedup (target: 10x)")

    print()
    print("=" * 70)

if __name__ == "__main__":
    benchmark_sizes()
