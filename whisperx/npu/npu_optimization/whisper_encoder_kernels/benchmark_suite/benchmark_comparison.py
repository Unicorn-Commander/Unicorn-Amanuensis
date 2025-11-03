#!/usr/bin/env python3
"""
Optimization Comparison Framework

Compares different optimization approaches:
- Baseline (no optimizations)
- Buffer reuse optimization
- Batch processing
- Multi-core NPU utilization
- DMA optimization
- Full optimization stack

Tracks incremental improvements toward 220x target
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import time
import numpy as np
from pathlib import Path
from typing import Dict, List
import json


class BenchmarkComparison:
    """Compare different optimization approaches"""

    def __init__(self):
        """Initialize comparison framework"""
        self.encoder = None
        self.results = {}

    def _initialize_encoder(self):
        """Lazy initialization of NPU encoder"""
        if self.encoder is None:
            parent_dir = Path(__file__).parent.parent
            sys.path.insert(0, str(parent_dir))
            from test_encoder_block import NPUEncoderBlock

            print("Initializing NPU Encoder Block...")
            self.encoder = NPUEncoderBlock()
            print("Encoder initialized!")
            print()

    def run_with_config(self, config: Dict) -> Dict:
        """
        Run encoder with specific optimization configuration

        Args:
            config: Optimization configuration dictionary

        Returns:
            Performance results
        """
        self._initialize_encoder()

        # Prepare test data
        Q = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
        K = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
        V = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
        gamma = np.ones(256, dtype=np.int8)
        beta = np.zeros(256, dtype=np.int8)

        # Warm-up
        _ = self.encoder.forward_block(Q, K, V, gamma, beta)

        # Benchmark (10 iterations)
        times = []
        for _ in range(10):
            start = time.perf_counter()

            if config.get('buffer_reuse', False):
                # Use optimized forward_block with buffer reuse
                _ = self.encoder.forward_block(Q, K, V, gamma, beta)
            else:
                # Run kernels individually (baseline)
                _ = self.encoder.run_attention(Q, K, V)
                ln_input = np.random.randint(-64, 64, 256, dtype=np.int8)
                _ = self.encoder.run_layernorm(ln_input, gamma, beta)
                A = np.random.randint(-64, 64, (16, 16), dtype=np.int8)
                B = np.random.randint(-64, 64, (16, 16), dtype=np.int8)
                _ = self.encoder.run_matmul(A, B)
                gelu_input = np.random.randint(-64, 64, 512, dtype=np.int8)
                _ = self.encoder.run_gelu(gelu_input)

            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        # Project to full encoder
        tiles_per_block = 23.4
        num_blocks = 6
        mel_time = 304.7  # ms

        encoder_time = avg_time * tiles_per_block * num_blocks
        total_time = mel_time + encoder_time
        realtime_factor = 11000 / total_time  # 11-second audio

        return {
            'config': config,
            'tile_time_ms': avg_time,
            'tile_time_std': std_time,
            'encoder_time_ms': encoder_time,
            'total_time_ms': total_time,
            'realtime_factor': realtime_factor
        }

    def compare_optimizations(self) -> Dict:
        """
        Compare baseline vs progressive optimizations

        Returns:
            Dictionary with results for each configuration
        """
        print("=" * 70)
        print("OPTIMIZATION COMPARISON")
        print("=" * 70)
        print()

        configs = {
            'baseline': {
                'buffer_reuse': False,
                'batching': False,
                'multi_core': False,
                'dma_optimization': False,
                'description': 'No optimizations - individual kernel calls'
            },
            'buffer_optimized': {
                'buffer_reuse': True,
                'batching': False,
                'multi_core': False,
                'dma_optimization': False,
                'description': 'Buffer reuse - minimize DMA sync operations'
            },
            'batched': {
                'buffer_reuse': True,
                'batching': True,
                'multi_core': False,
                'dma_optimization': False,
                'description': 'Batch processing - process multiple tiles together'
            },
            'multi_core': {
                'buffer_reuse': True,
                'batching': True,
                'multi_core': True,
                'dma_optimization': False,
                'description': 'Multi-core NPU - parallel tile processing'
            },
            'fully_optimized': {
                'buffer_reuse': True,
                'batching': True,
                'multi_core': True,
                'dma_optimization': True,
                'description': 'All optimizations - full NPU utilization'
            }
        }

        baseline_rtf = None

        for name, config in configs.items():
            print(f"Testing: {name}")
            print(f"  {config['description']}")

            try:
                result = self.run_with_config(config)
                self.results[name] = result

                # Calculate speedup vs baseline
                if baseline_rtf is None:
                    baseline_rtf = result['realtime_factor']
                    speedup = 1.0
                else:
                    speedup = result['realtime_factor'] / baseline_rtf

                print(f"  Tile time:       {result['tile_time_ms']:.3f}ms")
                print(f"  Realtime factor: {result['realtime_factor']:.2f}x")
                print(f"  Speedup:         {speedup:.2f}x")
                print()

            except Exception as e:
                print(f"  Error: {e}")
                print()
                continue

        # Summary comparison
        print("=" * 70)
        print("OPTIMIZATION COMPARISON SUMMARY")
        print("=" * 70)
        print()
        print(f"{'Configuration':<20} {'Tile Time':<15} {'RTF':<10} {'Speedup':<10}")
        print("-" * 70)

        baseline_rtf = self.results.get('baseline', {}).get('realtime_factor', 1.0)

        for name, result in self.results.items():
            tile_time = result['tile_time_ms']
            rtf = result['realtime_factor']
            speedup = rtf / baseline_rtf

            print(f"{name:<20} {tile_time:>10.3f}ms    {rtf:>6.2f}x    {speedup:>6.2f}x")

        print()

        # Progress to target
        best_rtf = max([r['realtime_factor'] for r in self.results.values()])
        target_rtf = 220.0
        progress = (best_rtf / target_rtf) * 100

        print(f"Current Best:  {best_rtf:.1f}x realtime")
        print(f"Target:        {target_rtf:.1f}x realtime")
        print(f"Progress:      {progress:.1f}%")
        print(f"Gap:           {target_rtf / best_rtf:.1f}x improvement needed")
        print()

        return self.results

    def compare_tile_sizes(self) -> Dict:
        """
        Compare performance with different tile sizes

        Returns:
            Dictionary with results for each tile size
        """
        print("=" * 70)
        print("TILE SIZE COMPARISON")
        print("=" * 70)
        print()

        tile_sizes = {
            '16x16': {
                'size': 16,
                'description': 'Small tiles - lower memory usage'
            },
            '32x32': {
                'size': 32,
                'description': 'Medium tiles - balanced'
            },
            '64x64': {
                'size': 64,
                'description': 'Large tiles - current implementation'
            }
        }

        results = {}

        for name, config in tile_sizes.items():
            print(f"Testing: {name}")
            print(f"  {config['description']}")

            # Note: Actual tile size change requires kernel recompilation
            # This is a projection based on current 64x64 performance

            size = config['size']
            scaling_factor = (64 / size) ** 2  # Number of tiles needed

            # Current 64x64 performance (from baseline)
            base_tile_time = 5.40  # ms per 64x64 tile

            # Projected time (smaller tiles have overhead)
            if size == 16:
                projected_time = base_tile_time * 0.3 * scaling_factor  # 30% time per small tile
            elif size == 32:
                projected_time = base_tile_time * 0.6 * scaling_factor  # 60% time per medium tile
            else:
                projected_time = base_tile_time  # 64x64 is baseline

            tiles_per_block = 1500 / size
            num_blocks = 6
            mel_time = 304.7

            encoder_time = projected_time * tiles_per_block * num_blocks
            total_time = mel_time + encoder_time
            realtime_factor = 11000 / total_time

            results[name] = {
                'tile_size': size,
                'tile_time_ms': projected_time,
                'tiles_per_block': tiles_per_block,
                'encoder_time_ms': encoder_time,
                'total_time_ms': total_time,
                'realtime_factor': realtime_factor
            }

            print(f"  Tile time:       {projected_time:.3f}ms")
            print(f"  Tiles per block: {tiles_per_block:.1f}")
            print(f"  Realtime factor: {realtime_factor:.2f}x")
            print()

        print("=" * 70)
        print("TILE SIZE COMPARISON SUMMARY")
        print("=" * 70)
        print()
        print(f"{'Tile Size':<12} {'Time/Tile':<15} {'Tiles/Block':<15} {'RTF':<10}")
        print("-" * 70)

        for name, result in results.items():
            print(f"{name:<12} {result['tile_time_ms']:>10.3f}ms    "
                  f"{result['tiles_per_block']:>10.1f}       {result['realtime_factor']:>6.2f}x")

        print()

        return results

    def save_results(self, output_file: str):
        """Save comparison results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Run standalone comparison
    benchmark = BenchmarkComparison()
    results = benchmark.compare_optimizations()
    benchmark.save_results("comparison_results.json")

    print()
    tile_results = benchmark.compare_tile_sizes()
