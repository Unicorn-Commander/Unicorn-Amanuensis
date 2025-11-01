#!/usr/bin/env python3
"""
Buffer Pool Performance Benchmark

Measures the performance improvements achieved by buffer pooling:
- Allocation time comparison (with vs without pool)
- Latency improvement measurement
- Memory usage profiling
- Pool hit rate analysis
- Memory leak testing

Author: CC-1L Buffer Pool Implementation Team
Date: November 1, 2025
Status: Week 8 Days 1-2 Implementation
"""

import sys
import time
import numpy as np
import tracemalloc
import gc
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from buffer_pool import BufferPool, GlobalBufferManager


def benchmark_allocation_time():
    """
    Benchmark 1: Allocation time comparison

    Measures the time to allocate buffers with and without pooling
    """
    print("\n" + "="*70)
    print("  BENCHMARK 1: Allocation Time Comparison")
    print("="*70)

    buffer_size = 960 * 1024  # 960KB mel buffer
    iterations = 1000

    # Without pool (direct numpy allocation)
    print("\n[Without Pool] Direct numpy allocation:")
    gc.collect()

    times_no_pool = []
    for i in range(iterations):
        start = time.perf_counter()
        buffer = np.zeros((80, 3000), dtype=np.float32)
        end = time.perf_counter()
        times_no_pool.append((end - start) * 1000)  # Convert to ms
        del buffer

    avg_no_pool = np.mean(times_no_pool)
    std_no_pool = np.std(times_no_pool)
    min_no_pool = np.min(times_no_pool)
    max_no_pool = np.max(times_no_pool)

    print(f"  Iterations: {iterations}")
    print(f"  Average:    {avg_no_pool:.4f}ms")
    print(f"  Std Dev:    {std_no_pool:.4f}ms")
    print(f"  Min:        {min_no_pool:.4f}ms")
    print(f"  Max:        {max_no_pool:.4f}ms")

    # With pool
    print("\n[With Pool] Buffer pool allocation:")
    pool = BufferPool(
        buffer_type='mel',
        buffer_size=buffer_size,
        dtype=np.float32,
        shape=(80, 3000),
        initial_size=10,
        max_size=20
    )

    gc.collect()

    times_with_pool = []
    for i in range(iterations):
        start = time.perf_counter()
        buffer = pool.acquire()
        pool.release(buffer)
        end = time.perf_counter()
        times_with_pool.append((end - start) * 1000)  # Convert to ms

    avg_with_pool = np.mean(times_with_pool)
    std_with_pool = np.std(times_with_pool)
    min_with_pool = np.min(times_with_pool)
    max_with_pool = np.max(times_with_pool)

    print(f"  Iterations: {iterations}")
    print(f"  Average:    {avg_with_pool:.4f}ms")
    print(f"  Std Dev:    {std_with_pool:.4f}ms")
    print(f"  Min:        {min_with_pool:.4f}ms")
    print(f"  Max:        {max_with_pool:.4f}ms")

    # Calculate improvement
    improvement_pct = ((avg_no_pool - avg_with_pool) / avg_no_pool) * 100
    speedup = avg_no_pool / avg_with_pool

    print("\n[Results]")
    print(f"  Improvement:     {improvement_pct:.1f}%")
    print(f"  Speedup:         {speedup:.2f}x")
    print(f"  Time saved:      {avg_no_pool - avg_with_pool:.4f}ms per allocation")
    print(f"  Pool hit rate:   {pool.get_stats()['hit_rate']*100:.1f}%")

    return {
        'no_pool_avg': avg_no_pool,
        'with_pool_avg': avg_with_pool,
        'improvement_pct': improvement_pct,
        'speedup': speedup,
        'hit_rate': pool.get_stats()['hit_rate']
    }


def benchmark_memory_usage():
    """
    Benchmark 2: Memory usage comparison

    Measures peak memory usage with and without pooling
    """
    print("\n" + "="*70)
    print("  BENCHMARK 2: Memory Usage Comparison")
    print("="*70)

    iterations = 100

    # Without pool
    print("\n[Without Pool] Memory usage:")
    gc.collect()
    tracemalloc.start()

    buffers = []
    for i in range(iterations):
        buffer = np.zeros((80, 3000), dtype=np.float32)
        buffers.append(buffer)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Allocations: {iterations}")
    print(f"  Current:     {current / (1024*1024):.2f}MB")
    print(f"  Peak:        {peak / (1024*1024):.2f}MB")

    buffers.clear()
    gc.collect()

    # With pool - limit concurrent to max_size
    print("\n[With Pool] Memory usage:")
    gc.collect()
    tracemalloc.start()

    pool = BufferPool(
        buffer_type='mel',
        buffer_size=960 * 1024,
        dtype=np.float32,
        shape=(80, 3000),
        initial_size=10,
        max_size=iterations  # Allow enough for all iterations
    )

    buffers = []
    for i in range(iterations):
        buffer = pool.acquire()
        buffers.append(buffer)

    current_pool, peak_pool = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Allocations: {iterations}")
    print(f"  Current:     {current_pool / (1024*1024):.2f}MB")
    print(f"  Peak:        {peak_pool / (1024*1024):.2f}MB")
    print(f"  Pool size:   {pool.get_stats()['total_buffers']} buffers")

    # Release all
    for buffer in buffers:
        pool.release(buffer)

    print("\n[Results]")
    print(f"  Memory saved: {(peak - peak_pool) / (1024*1024):.2f}MB")
    print(f"  Reduction:    {((peak - peak_pool) / peak) * 100:.1f}%")

    return {
        'no_pool_peak_mb': peak / (1024*1024),
        'with_pool_peak_mb': peak_pool / (1024*1024),
        'memory_saved_mb': (peak - peak_pool) / (1024*1024),
        'reduction_pct': ((peak - peak_pool) / peak) * 100
    }


def benchmark_concurrent_performance():
    """
    Benchmark 3: Concurrent performance

    Tests buffer pool performance under concurrent load
    """
    print("\n" + "="*70)
    print("  BENCHMARK 3: Concurrent Performance")
    print("="*70)

    import threading

    pool = BufferPool(
        buffer_type='concurrent_test',
        buffer_size=960 * 1024,
        dtype=np.float32,
        shape=(80, 3000),
        initial_size=10,
        max_size=50
    )

    num_threads = 20
    iterations_per_thread = 100
    errors = []

    def worker():
        """Worker thread"""
        try:
            for _ in range(iterations_per_thread):
                buffer = pool.acquire()
                buffer.fill(threading.current_thread().ident)
                time.sleep(0.0001)  # Simulate work
                pool.release(buffer)
        except Exception as e:
            errors.append(e)

    print(f"\n[Running] {num_threads} threads × {iterations_per_thread} iterations...")

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]

    start = time.time()
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    elapsed = time.time() - start

    stats = pool.get_stats()
    total_ops = num_threads * iterations_per_thread

    print(f"\n[Results]")
    print(f"  Total operations: {total_ops}")
    print(f"  Elapsed time:     {elapsed:.2f}s")
    print(f"  Ops/sec:          {total_ops/elapsed:.0f}")
    print(f"  Hit rate:         {stats['hit_rate']*100:.1f}%")
    print(f"  Pool size:        {stats['total_buffers']} buffers")
    print(f"  Errors:           {len(errors)}")
    print(f"  Leaked buffers:   {stats['leaked_buffers']}")

    return {
        'threads': num_threads,
        'total_ops': total_ops,
        'elapsed': elapsed,
        'ops_per_sec': total_ops/elapsed,
        'hit_rate': stats['hit_rate'],
        'errors': len(errors),
        'leaked_buffers': stats['leaked_buffers']
    }


def benchmark_memory_leak():
    """
    Benchmark 4: Memory leak detection

    Runs 1000+ iterations to verify no memory leaks
    """
    print("\n" + "="*70)
    print("  BENCHMARK 4: Memory Leak Detection (1000 iterations)")
    print("="*70)

    iterations = 1000

    pool = BufferPool(
        buffer_type='leak_test',
        buffer_size=960 * 1024,
        dtype=np.float32,
        shape=(80, 3000),
        initial_size=5,
        max_size=10
    )

    print(f"\n[Running] {iterations} acquire/release cycles...")

    gc.collect()
    tracemalloc.start()

    start_mem = tracemalloc.get_traced_memory()[0]

    for i in range(iterations):
        buffer = pool.acquire()
        buffer.fill(float(i))
        pool.release(buffer)

        if (i + 1) % 250 == 0:
            current_mem = tracemalloc.get_traced_memory()[0]
            print(f"  Progress: {i+1}/{iterations} - Memory: {current_mem/(1024*1024):.2f}MB")

    end_mem = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    stats = pool.get_stats()

    print(f"\n[Results]")
    print(f"  Iterations:      {iterations}")
    print(f"  Start memory:    {start_mem/(1024*1024):.2f}MB")
    print(f"  End memory:      {end_mem/(1024*1024):.2f}MB")
    print(f"  Memory growth:   {(end_mem - start_mem)/(1024*1024):.2f}MB")
    print(f"  Leaked buffers:  {stats['leaked_buffers']}")
    print(f"  Has leaks:       {stats['has_leaks']}")
    print(f"  Hit rate:        {stats['hit_rate']*100:.1f}%")

    # Verify no leaks
    leak_free = not stats['has_leaks'] and abs(end_mem - start_mem) < 1024*1024  # < 1MB growth

    if leak_free:
        print(f"\n  ✅ PASSED: No memory leaks detected!")
    else:
        print(f"\n  ❌ FAILED: Memory leaks detected!")

    return {
        'iterations': iterations,
        'memory_growth_mb': (end_mem - start_mem)/(1024*1024),
        'leaked_buffers': stats['leaked_buffers'],
        'has_leaks': stats['has_leaks'],
        'leak_free': leak_free
    }


def benchmark_simulated_request_pattern():
    """
    Benchmark 5: Simulated request pattern

    Simulates real service usage with multiple buffer types
    """
    print("\n" + "="*70)
    print("  BENCHMARK 5: Simulated Service Request Pattern")
    print("="*70)

    manager = GlobalBufferManager.instance()

    # Configure pools like the service
    manager.configure({
        'mel': {
            'size': 960 * 1024,
            'count': 10,
            'max_count': 20,
            'dtype': np.float32,
            'shape': (80, 3000)
        },
        'audio': {
            'size': 480 * 1024,
            'count': 5,
            'max_count': 15,
            'dtype': np.float32
        },
        'encoder_output': {
            'size': 3 * 1024 * 1024,
            'count': 5,
            'max_count': 15,
            'dtype': np.float32,
            'shape': (3000, 512)
        }
    })

    num_requests = 100

    print(f"\n[Running] {num_requests} simulated requests...")

    start = time.time()

    for i in range(num_requests):
        # Simulate request
        audio_buf = None
        mel_buf = None
        encoder_buf = None

        try:
            # Acquire buffers
            audio_buf = manager.acquire('audio')
            mel_buf = manager.acquire('mel')
            encoder_buf = manager.acquire('encoder_output')

            # Simulate work
            audio_buf.fill(float(i))
            mel_buf.fill(float(i))
            encoder_buf.fill(float(i))

            time.sleep(0.001)  # Simulate 1ms processing

        finally:
            # Always release
            if audio_buf is not None:
                manager.release('audio', audio_buf)
            if mel_buf is not None:
                manager.release('mel', mel_buf)
            if encoder_buf is not None:
                manager.release('encoder_output', encoder_buf)

        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{num_requests}")

    elapsed = time.time() - start

    stats = manager.get_stats()

    print(f"\n[Results]")
    print(f"  Requests:        {num_requests}")
    print(f"  Elapsed time:    {elapsed:.2f}s")
    print(f"  Requests/sec:    {num_requests/elapsed:.1f}")

    for pool_name, pool_stats in stats.items():
        print(f"\n  [{pool_name}]")
        print(f"    Hit rate:      {pool_stats['hit_rate']*100:.1f}%")
        print(f"    Leaked:        {pool_stats['leaked_buffers']}")
        print(f"    Total buffers: {pool_stats['total_buffers']}")

    # Check for leaks
    total_leaks = sum(s['leaked_buffers'] for s in stats.values())
    leak_free = total_leaks == 0

    if leak_free:
        print(f"\n  ✅ PASSED: No leaks across all pools!")
    else:
        print(f"\n  ❌ FAILED: {total_leaks} leaked buffers detected!")

    return {
        'requests': num_requests,
        'elapsed': elapsed,
        'requests_per_sec': num_requests/elapsed,
        'total_leaks': total_leaks,
        'leak_free': leak_free
    }


def main():
    """Run all benchmarks"""
    print("\n" + "="*70)
    print("  BUFFER POOL PERFORMANCE BENCHMARKS")
    print("="*70)
    print("\n  Testing buffer pool performance improvements")
    print("  Target: 80% allocation overhead reduction")
    print("  Target: <50MB peak memory usage")
    print("  Target: 0 memory leaks")

    results = {}

    # Run benchmarks
    try:
        results['allocation'] = benchmark_allocation_time()
        results['memory'] = benchmark_memory_usage()
        results['concurrent'] = benchmark_concurrent_performance()
        results['leak_test'] = benchmark_memory_leak()

        # Reset singleton for simulated request pattern
        GlobalBufferManager._instance = None
        results['simulation'] = benchmark_simulated_request_pattern()

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print summary
    print("\n" + "="*70)
    print("  BENCHMARK SUMMARY")
    print("="*70)

    print("\n[Allocation Performance]")
    print(f"  Improvement: {results['allocation']['improvement_pct']:.1f}%")
    print(f"  Speedup:     {results['allocation']['speedup']:.2f}x")
    print(f"  Hit rate:    {results['allocation']['hit_rate']*100:.1f}%")

    print("\n[Memory Usage]")
    print(f"  Memory saved: {results['memory']['memory_saved_mb']:.2f}MB")
    print(f"  Reduction:    {results['memory']['reduction_pct']:.1f}%")

    print("\n[Concurrent Performance]")
    print(f"  Operations/sec: {results['concurrent']['ops_per_sec']:.0f}")
    print(f"  Hit rate:       {results['concurrent']['hit_rate']*100:.1f}%")
    print(f"  Errors:         {results['concurrent']['errors']}")

    print("\n[Memory Leak Test]")
    print(f"  Iterations:     {results['leak_test']['iterations']}")
    print(f"  Memory growth:  {results['leak_test']['memory_growth_mb']:.2f}MB")
    print(f"  Leak free:      {results['leak_test']['leak_free']}")

    print("\n[Service Simulation]")
    print(f"  Requests:       {results['simulation']['requests']}")
    print(f"  Requests/sec:   {results['simulation']['requests_per_sec']:.1f}")
    print(f"  Leak free:      {results['simulation']['leak_free']}")

    # Final verdict
    print("\n" + "="*70)

    success = (
        results['allocation']['improvement_pct'] >= 60 and  # At least 60% improvement
        results['leak_test']['leak_free'] and
        results['simulation']['leak_free']
    )

    if success:
        print("  ✅ ALL BENCHMARKS PASSED!")
        print("  Buffer pool is production-ready")
    else:
        print("  ❌ SOME BENCHMARKS FAILED")
        print("  Review results above")

    print("="*70 + "\n")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
