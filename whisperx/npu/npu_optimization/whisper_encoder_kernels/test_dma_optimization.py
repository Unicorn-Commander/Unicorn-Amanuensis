#!/usr/bin/env python3
"""
DMA Optimization Benchmark Suite

This benchmark suite measures the impact of various DMA optimization strategies:
1. Baseline: Per-kernel sync (current implementation)
2. Buffer Pooling: Pre-allocated buffer reuse
3. Pipelined Execution: Overlapped DMA and compute
4. Batch DMA: Batched sync operations

Goal: Achieve 1.3-1.5x improvement through DMA optimization

Usage:
    python3 test_dma_optimization.py
    python3 test_dma_optimization.py --verbose
    python3 test_dma_optimization.py --num-tiles 20
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Import our optimization modules
from npu_buffer_pool import NPUBufferPool, BufferCache
from npu_pipeline_executor import PipelinedNPUExecutor
from test_encoder_block import NPUEncoderBlock


class DMABenchmark:
    """Comprehensive DMA optimization benchmark suite"""

    def __init__(self, encoder: NPUEncoderBlock, num_tiles: int = 10, verbose: bool = False):
        """
        Initialize benchmark

        Args:
            encoder: NPUEncoderBlock instance
            num_tiles: Number of tiles to process in benchmarks
            verbose: Print detailed output
        """
        self.encoder = encoder
        self.num_tiles = num_tiles
        self.verbose = verbose

        # Create test data
        self.test_tiles = self._create_test_data()

        # Results storage
        self.results = {}

        if self.verbose:
            print(f"Benchmark initialized with {num_tiles} tiles")
            print()

    def _create_test_data(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Create test data for benchmarks"""
        tiles = []

        for i in range(self.num_tiles):
            Q = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
            K = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
            V = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
            tiles.append((Q, K, V))

        return tiles

    def benchmark_baseline(self) -> Dict[str, float]:
        """
        Baseline: Per-kernel sync (current implementation)

        Each tile: write -> sync_to_device -> execute -> wait -> sync_from_device -> read
        """
        print("=" * 70)
        print("BENCHMARK 1: Baseline (Per-Kernel Sync)")
        print("=" * 70)
        print()

        times = []
        dma_write_times = []
        dma_read_times = []
        compute_times = []

        for i in range(self.num_tiles):
            Q, K, V = self.test_tiles[i]

            # Measure DMA write
            dma_write_start = time.perf_counter()
            QKV_combined = np.concatenate([Q.flatten(), K.flatten(), V.flatten()])
            self.encoder.attn_input_bo.write(QKV_combined.tobytes(), 0)
            self.encoder.attn_input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 12288, 0)
            dma_write_time = (time.perf_counter() - dma_write_start) * 1000
            dma_write_times.append(dma_write_time)

            # Measure compute
            compute_start = time.perf_counter()
            opcode = 3
            run = self.encoder.attn_kernel(opcode, self.encoder.attn_instr_bo,
                                          self.encoder.attn_n_insts,
                                          self.encoder.attn_input_bo,
                                          self.encoder.attn_output_bo)
            run.wait(1000)
            compute_time = (time.perf_counter() - compute_start) * 1000
            compute_times.append(compute_time)

            # Measure DMA read
            dma_read_start = time.perf_counter()
            self.encoder.attn_output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 4096, 0)
            output = np.frombuffer(self.encoder.attn_output_bo.read(4096, 0), dtype=np.int8)
            dma_read_time = (time.perf_counter() - dma_read_start) * 1000
            dma_read_times.append(dma_read_time)

            total_time = dma_write_time + compute_time + dma_read_time
            times.append(total_time)

        avg_total = np.mean(times)
        avg_dma_write = np.mean(dma_write_times)
        avg_compute = np.mean(compute_times)
        avg_dma_read = np.mean(dma_read_times)

        dma_overhead = avg_dma_write + avg_dma_read
        dma_overhead_percent = 100 * dma_overhead / avg_total

        print(f"Results (average over {self.num_tiles} tiles):")
        print(f"  DMA write:     {avg_dma_write:.3f}ms")
        print(f"  Compute:       {avg_compute:.3f}ms")
        print(f"  DMA read:      {avg_dma_read:.3f}ms")
        print(f"  Total:         {avg_total:.3f}ms")
        print(f"  DMA overhead:  {dma_overhead:.3f}ms ({dma_overhead_percent:.1f}%)")
        print()

        results = {
            'strategy': 'baseline',
            'avg_time_ms': avg_total,
            'dma_write_ms': avg_dma_write,
            'compute_ms': avg_compute,
            'dma_read_ms': avg_dma_read,
            'dma_overhead_ms': dma_overhead,
            'dma_overhead_percent': dma_overhead_percent
        }

        self.results['baseline'] = results
        return results

    def benchmark_buffer_pooling(self) -> Dict[str, float]:
        """
        Buffer Pooling: Pre-allocated buffer reuse

        Eliminates buffer allocation overhead through reuse.
        """
        print("=" * 70)
        print("BENCHMARK 2: Buffer Pooling")
        print("=" * 70)
        print()

        # Create buffer pool
        pool = NPUBufferPool(self.encoder.device, num_buffers=4, verbose=self.verbose)

        # Pre-allocate buffers (this is the key optimization)
        input_bo = pool.allocate_buffer("attn_input", 12288, self.encoder.attn_kernel.group_id(3))
        output_bo = pool.allocate_buffer("attn_output", 4096, self.encoder.attn_kernel.group_id(4))

        if self.verbose:
            print()

        times = []

        for i in range(self.num_tiles):
            Q, K, V = self.test_tiles[i]

            tile_start = time.perf_counter()

            # Use pooled buffers
            QKV_combined = np.concatenate([Q.flatten(), K.flatten(), V.flatten()])
            input_bo.write(QKV_combined.tobytes(), 0)
            input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 12288, 0)

            opcode = 3
            run = self.encoder.attn_kernel(opcode, self.encoder.attn_instr_bo,
                                          self.encoder.attn_n_insts,
                                          input_bo, output_bo)
            run.wait(1000)

            output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 4096, 0)
            output = np.frombuffer(output_bo.read(4096, 0), dtype=np.int8)

            tile_time = (time.perf_counter() - tile_start) * 1000
            times.append(tile_time)

        avg_time = np.mean(times)

        if self.verbose:
            pool.print_statistics()

        print(f"Results (average over {self.num_tiles} tiles):")
        print(f"  Avg time per tile:  {avg_time:.3f}ms")
        print(f"  Buffer reuse ratio: {pool.stats['reuses']}/{pool.stats['allocations']}")
        print()

        baseline_time = self.results.get('baseline', {}).get('avg_time_ms', avg_time)
        improvement = baseline_time / avg_time

        print(f"Improvement vs baseline: {improvement:.3f}x")
        print()

        results = {
            'strategy': 'buffer_pooling',
            'avg_time_ms': avg_time,
            'improvement_vs_baseline': improvement,
            'buffer_reuses': pool.stats['reuses'],
            'buffer_allocations': pool.stats['allocations']
        }

        self.results['buffer_pooling'] = results
        return results

    def benchmark_pipelined_execution(self) -> Dict[str, float]:
        """
        Pipelined Execution: Overlapped DMA and compute

        While NPU processes tile N, CPU prepares tile N+1.
        """
        print("=" * 70)
        print("BENCHMARK 3: Pipelined Execution")
        print("=" * 70)
        print()

        # Create pipeline executor
        pipeline = PipelinedNPUExecutor(self.encoder, pipeline_depth=2, verbose=self.verbose)

        # Process all tiles with pipelining
        start_time = time.perf_counter()
        results = pipeline.process_attention_tiles_pipelined(self.test_tiles, sync_per_tile=False)
        total_time = (time.perf_counter() - start_time) * 1000

        avg_time = total_time / self.num_tiles

        if self.verbose:
            print()
            pipeline.print_statistics()

        print(f"Results ({self.num_tiles} tiles):")
        print(f"  Total time:         {total_time:.3f}ms")
        print(f"  Avg time per tile:  {avg_time:.3f}ms")
        print(f"  Pipeline stalls:    {pipeline.stats['pipeline_stalls']}")
        print()

        baseline_time = self.results.get('baseline', {}).get('avg_time_ms', avg_time)
        improvement = baseline_time / avg_time

        print(f"Improvement vs baseline: {improvement:.3f}x")
        print()

        results_dict = {
            'strategy': 'pipelined',
            'avg_time_ms': avg_time,
            'total_time_ms': total_time,
            'improvement_vs_baseline': improvement,
            'pipeline_stalls': pipeline.stats['pipeline_stalls']
        }

        self.results['pipelined'] = results_dict
        return results_dict

    def benchmark_batch_dma(self) -> Dict[str, float]:
        """
        Batch DMA: Multiple writes before sync

        Batch multiple DMA operations to reduce sync overhead.
        """
        print("=" * 70)
        print("BENCHMARK 4: Batch DMA")
        print("=" * 70)
        print()

        batch_size = 4
        times = []

        for batch_start in range(0, self.num_tiles, batch_size):
            batch_end = min(batch_start + batch_size, self.num_tiles)
            batch_tiles = self.test_tiles[batch_start:batch_end]

            batch_time_start = time.perf_counter()

            # Batch writes (no sync yet)
            for i, (Q, K, V) in enumerate(batch_tiles):
                QKV_combined = np.concatenate([Q.flatten(), K.flatten(), V.flatten()])
                self.encoder.attn_input_bo.write(QKV_combined.tobytes(), 0)

                # Process immediately (in real batch DMA, we'd defer this)
                self.encoder.attn_input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 12288, 0)

                opcode = 3
                run = self.encoder.attn_kernel(opcode, self.encoder.attn_instr_bo,
                                              self.encoder.attn_n_insts,
                                              self.encoder.attn_input_bo,
                                              self.encoder.attn_output_bo)
                run.wait(1000)

                self.encoder.attn_output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 4096, 0)
                output = np.frombuffer(self.encoder.attn_output_bo.read(4096, 0), dtype=np.int8)

            batch_time = (time.perf_counter() - batch_time_start) * 1000
            times.extend([batch_time / len(batch_tiles)] * len(batch_tiles))

        avg_time = np.mean(times)

        print(f"Results (batch size {batch_size}, {self.num_tiles} tiles):")
        print(f"  Avg time per tile:  {avg_time:.3f}ms")
        print()

        baseline_time = self.results.get('baseline', {}).get('avg_time_ms', avg_time)
        improvement = baseline_time / avg_time

        print(f"Improvement vs baseline: {improvement:.3f}x")
        print()

        results = {
            'strategy': 'batch_dma',
            'avg_time_ms': avg_time,
            'batch_size': batch_size,
            'improvement_vs_baseline': improvement
        }

        self.results['batch_dma'] = results
        return results

    def print_summary(self):
        """Print comprehensive summary of all benchmarks"""
        print("\n")
        print("=" * 70)
        print("DMA OPTIMIZATION SUMMARY")
        print("=" * 70)
        print()

        if 'baseline' not in self.results:
            print("No baseline results available")
            return

        baseline_time = self.results['baseline']['avg_time_ms']

        print(f"{'Strategy':<20} {'Time (ms)':<12} {'Improvement':<12} {'Cumulative RTF'}")
        print("-" * 70)

        # Calculate cumulative RTF (assuming 16.2x baseline)
        baseline_rtf = 16.2  # From current reports

        strategies = ['baseline', 'buffer_pooling', 'pipelined', 'batch_dma']
        cumulative_improvement = 1.0

        for strategy in strategies:
            if strategy not in self.results:
                continue

            result = self.results[strategy]
            time_ms = result['avg_time_ms']

            if strategy == 'baseline':
                improvement = 1.0
                improvement_str = "1.0x"
            else:
                improvement = result.get('improvement_vs_baseline', 1.0)
                cumulative_improvement *= improvement
                improvement_str = f"{improvement:.2f}x"

            rtf = baseline_rtf * cumulative_improvement
            rtf_str = f"{rtf:.1f}x RT"

            print(f"{strategy:<20} {time_ms:<12.3f} {improvement_str:<12} {rtf_str}")

        print("-" * 70)
        print()

        # DMA overhead analysis
        if 'baseline' in self.results:
            baseline = self.results['baseline']
            print(f"DMA Overhead Analysis (Baseline):")
            print(f"  DMA write:        {baseline['dma_write_ms']:.3f}ms")
            print(f"  DMA read:         {baseline['dma_read_ms']:.3f}ms")
            print(f"  Total DMA:        {baseline['dma_overhead_ms']:.3f}ms")
            print(f"  DMA percentage:   {baseline['dma_overhead_percent']:.1f}%")
            print(f"  Compute:          {baseline['compute_ms']:.3f}ms")
            print()

        # Best optimization
        best_strategy = max(
            [s for s in strategies if s != 'baseline' and s in self.results],
            key=lambda s: self.results[s].get('improvement_vs_baseline', 0.0),
            default=None
        )

        if best_strategy:
            best_result = self.results[best_strategy]
            print(f"Best Optimization: {best_strategy}")
            print(f"  Improvement: {best_result['improvement_vs_baseline']:.2f}x")
            print(f"  Time per tile: {best_result['avg_time_ms']:.3f}ms")
            print()

        # Recommendation
        print("Recommendation:")
        if cumulative_improvement >= 1.3:
            print(f"  ✅ Target achieved: {cumulative_improvement:.2f}x improvement")
            print(f"  ✅ New RTF: {baseline_rtf * cumulative_improvement:.1f}x realtime")
        else:
            print(f"  ⚠️  Target not fully met: {cumulative_improvement:.2f}x improvement")
            print(f"     (target: 1.3-1.5x)")
            print(f"  Current RTF: {baseline_rtf * cumulative_improvement:.1f}x realtime")

        print()

    def run_all_benchmarks(self):
        """Run all benchmarks in sequence"""
        print("\n")
        print("=" * 70)
        print("DMA OPTIMIZATION BENCHMARK SUITE")
        print("=" * 70)
        print()
        print(f"Configuration:")
        print(f"  Number of tiles:  {self.num_tiles}")
        print(f"  Tile size:        64x64 (Q, K, V)")
        print(f"  Data type:        INT8")
        print()

        # Run benchmarks
        self.benchmark_baseline()
        self.benchmark_buffer_pooling()
        self.benchmark_pipelined_execution()
        self.benchmark_batch_dma()

        # Print summary
        self.print_summary()


def main():
    """Main benchmark entry point"""
    parser = argparse.ArgumentParser(description='DMA Optimization Benchmark Suite')
    parser.add_argument('--num-tiles', type=int, default=10,
                       help='Number of tiles to process (default: 10)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed output')

    args = parser.parse_args()

    print("Initializing NPU Encoder...")
    encoder = NPUEncoderBlock()
    print()

    benchmark = DMABenchmark(encoder, num_tiles=args.num_tiles, verbose=args.verbose)
    benchmark.run_all_benchmarks()

    print("Benchmark complete!")
    print()


if __name__ == "__main__":
    main()
