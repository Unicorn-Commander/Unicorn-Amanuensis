#!/usr/bin/env python3
"""
Comprehensive Pipelined Encoder Benchmark Suite

This script measures the actual performance improvement from DMA pipelined execution
across different batch sizes and configurations.

Usage:
    python3 benchmark_pipelined.py
    python3 benchmark_pipelined.py --verbose
    python3 benchmark_pipelined.py --tiles 20
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
import time
import argparse
from test_encoder_block import NPUEncoderBlock, PipelinedNPUExecutor


def benchmark_pipelined_performance(num_tiles=10, pipeline_depths=[2, 3], verbose=False):
    """Run comprehensive pipeline benchmarks"""

    print("\n")
    print("=" * 70)
    print("COMPREHENSIVE PIPELINED ENCODER BENCHMARK")
    print("=" * 70)
    print()

    # Initialize encoder
    print("Initializing NPU encoder...")
    encoder = NPUEncoderBlock()
    print()

    # Prepare test data
    print(f"Preparing test data for {num_tiles} tiles...")
    tiles = []
    for i in range(num_tiles):
        Q = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
        K = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
        V = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
        tiles.append((Q, K, V))
    print(f"  Test data ready: {num_tiles} tiles")
    print()

    # Baseline: Sequential execution
    print("=" * 70)
    print("BASELINE: Sequential Execution")
    print("=" * 70)
    print()

    pipeline_seq = PipelinedNPUExecutor(encoder, pipeline_depth=2, verbose=verbose)

    # Warm-up
    _ = pipeline_seq.process_attention_tiles_pipelined(tiles[:2], sync_per_tile=True)

    # Benchmark
    pipeline_seq.reset_statistics()
    seq_start = time.perf_counter()
    seq_results = pipeline_seq.process_attention_tiles_pipelined(tiles, sync_per_tile=True)
    seq_time = (time.perf_counter() - seq_start) * 1000
    seq_stats = pipeline_seq.get_statistics()

    print(f"Results:")
    print(f"  Total time:         {seq_time:.2f}ms")
    print(f"  Avg time per tile:  {seq_stats['avg_time_per_tile_ms']:.2f}ms")
    print()

    # Test different pipeline depths
    results = {
        'sequential': {
            'time_ms': seq_stats['avg_time_per_tile_ms'],
            'speedup': 1.0,
            'stalls': 0
        }
    }

    for depth in pipeline_depths:
        print("=" * 70)
        print(f"PIPELINED EXECUTION: Depth {depth}")
        print("=" * 70)
        print()

        pipeline = PipelinedNPUExecutor(encoder, pipeline_depth=depth, verbose=verbose)

        # Warm-up
        _ = pipeline.process_attention_tiles_pipelined(tiles[:depth], sync_per_tile=False)

        # Benchmark
        pipeline.reset_statistics()
        pipe_start = time.perf_counter()
        pipe_results = pipeline.process_attention_tiles_pipelined(tiles, sync_per_tile=False)
        pipe_time = (time.perf_counter() - pipe_start) * 1000
        pipe_stats = pipeline.get_statistics()

        speedup = seq_time / pipe_time

        print(f"Results:")
        print(f"  Total time:         {pipe_time:.2f}ms")
        print(f"  Avg time per tile:  {pipe_stats['avg_time_per_tile_ms']:.2f}ms")
        print(f"  Pipeline stalls:    {pipe_stats['pipeline_stalls']}")
        print(f"  DMA overhead:       {pipe_stats['dma_overhead_percent']:.1f}%")
        print(f"  Speedup vs seq:     {speedup:.2f}x")
        print()

        results[f'depth_{depth}'] = {
            'time_ms': pipe_stats['avg_time_per_tile_ms'],
            'speedup': speedup,
            'stalls': pipe_stats['pipeline_stalls'],
            'dma_overhead': pipe_stats['dma_overhead_percent']
        }

    # Summary table
    print("=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Configuration':<20} {'Time/Tile (ms)':<15} {'Speedup':<10} {'Stalls':<10}")
    print("-" * 70)

    for config, data in results.items():
        print(f"{config:<20} {data['time_ms']:<15.2f} {data['speedup']:<10.2f}x {data['stalls']:<10}")

    print("-" * 70)
    print()

    # Best configuration
    best_config = max(results.items(), key=lambda x: x[1]['speedup'])
    print(f"Best Configuration: {best_config[0]}")
    print(f"  Speedup: {best_config[1]['speedup']:.2f}x")
    print(f"  Time per tile: {best_config[1]['time_ms']:.2f}ms")
    print()

    # Projected realtime factor
    baseline_rtf = 14.0
    tiles_per_block = 23.4
    blocks = 6
    mel_time = 304.7  # ms

    best_time = best_config[1]['time_ms']
    encoder_time = best_time * tiles_per_block * blocks
    total_time = mel_time + encoder_time
    projected_rtf = 11000 / total_time

    print(f"Projected Performance (11-second audio):")
    print(f"  Encoder time:    {encoder_time:.1f}ms")
    print(f"  Total time:      {total_time:.1f}ms")
    print(f"  Realtime factor: {projected_rtf:.1f}x")
    print(f"  vs Baseline:     {baseline_rtf:.1f}x → {projected_rtf:.1f}x ({projected_rtf/baseline_rtf:.2f}x improvement)")
    print()

    if projected_rtf >= 23:
        print(f"TARGET ACHIEVED: {projected_rtf:.1f}x >= 23x realtime")
    elif projected_rtf >= 20:
        print(f"NEAR TARGET: {projected_rtf:.1f}x (target: 23-26x)")
    else:
        print(f"PROGRESS: {projected_rtf:.1f}x (target: 23-26x)")

    print("=" * 70)
    print()

    return results


def benchmark_batch_sizes(batch_sizes=[1, 2, 4, 8], verbose=False):
    """Benchmark different batch sizes"""

    print("\n")
    print("=" * 70)
    print("BATCH SIZE ANALYSIS")
    print("=" * 70)
    print()

    # Initialize encoder
    encoder = NPUEncoderBlock()
    pipeline = PipelinedNPUExecutor(encoder, pipeline_depth=2, verbose=verbose)

    results = {}

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size} tiles")

        # Prepare test data
        tiles = []
        for i in range(batch_size):
            Q = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
            K = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
            V = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
            tiles.append((Q, K, V))

        # Sequential
        pipeline.reset_statistics()
        seq_start = time.perf_counter()
        _ = pipeline.process_attention_tiles_pipelined(tiles, sync_per_tile=True)
        seq_time = (time.perf_counter() - seq_start) * 1000

        # Pipelined
        pipeline.reset_statistics()
        pipe_start = time.perf_counter()
        _ = pipeline.process_attention_tiles_pipelined(tiles, sync_per_tile=False)
        pipe_time = (time.perf_counter() - pipe_start) * 1000
        pipe_stats = pipeline.get_statistics()

        speedup = seq_time / pipe_time

        results[batch_size] = {
            'sequential_ms': seq_time,
            'pipelined_ms': pipe_time,
            'speedup': speedup,
            'stalls': pipe_stats['pipeline_stalls']
        }

        print(f"  Sequential: {seq_time:.2f}ms")
        print(f"  Pipelined:  {pipe_time:.2f}ms")
        print(f"  Speedup:    {speedup:.2f}x")
        print(f"  Stalls:     {pipe_stats['pipeline_stalls']}")
        print()

    # Summary
    print("=" * 70)
    print("BATCH SIZE SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Batch Size':<12} {'Sequential (ms)':<18} {'Pipelined (ms)':<18} {'Speedup':<10}")
    print("-" * 70)

    for batch_size, data in results.items():
        print(f"{batch_size:<12} {data['sequential_ms']:<18.2f} {data['pipelined_ms']:<18.2f} {data['speedup']:<10.2f}x")

    print("-" * 70)
    print()

    return results


def main():
    """Main benchmark entry point"""
    parser = argparse.ArgumentParser(description='Pipelined Encoder Benchmark Suite')
    parser.add_argument('--tiles', type=int, default=10,
                       help='Number of tiles to process (default: 10)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed output')
    parser.add_argument('--batch-analysis', action='store_true',
                       help='Run batch size analysis')

    args = parser.parse_args()

    if args.batch_analysis:
        batch_results = benchmark_batch_sizes(verbose=args.verbose)
    else:
        pipeline_results = benchmark_pipelined_performance(
            num_tiles=args.tiles,
            verbose=args.verbose
        )

    print("Benchmark complete!")
    print()


if __name__ == "__main__":
    main()
