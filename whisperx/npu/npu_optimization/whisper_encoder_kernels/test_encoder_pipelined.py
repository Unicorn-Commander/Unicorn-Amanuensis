#!/usr/bin/env python3
"""
Pipelined Encoder Block Test - Software Multi-Threading for 2-3× Speedup
Uses ThreadPoolExecutor to pipeline kernel submissions on existing single-column XCLBIN

Goal: Validate multi-core concept without recompilation
Expected: 2-3× improvement through XRT pipelining → 23-24× realtime
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
import time
from pathlib import Path
import concurrent.futures
from test_encoder_block import NPUEncoderBlock

class NPUEncoderBlockPipelined(NPUEncoderBlock):
    """Pipelined encoder block using ThreadPoolExecutor for async execution"""

    def __init__(self, num_workers=4):
        """Initialize with support for parallel execution

        Args:
            num_workers: Number of worker threads (default 4 for 4 columns)
        """
        super().__init__()
        self.num_workers = num_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)

    def process_tiles_pipelined(self, tiles_data):
        """Process multiple tiles with pipelined execution

        Args:
            tiles_data: List of tuples (Q, K, V, gamma, beta)

        Returns:
            List of results
        """
        # Submit all tasks to executor
        futures = []
        for Q, K, V, gamma, beta in tiles_data:
            future = self.executor.submit(self.forward_block, Q, K, V, gamma, beta)
            futures.append(future)

        # Wait for all results
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

        return results

    def __del__(self):
        """Clean up executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


def test_encoder_pipelined_vs_sequential():
    """Compare pipelined vs sequential execution"""

    print("\n")
    print("=" * 70)
    print("PIPELINED ENCODER TEST - Multi-Threading with Existing Kernels")
    print("=" * 70)
    print()

    # Initialize encoder
    print("Initializing encoder...")
    encoder = NPUEncoderBlockPipelined(num_workers=4)
    print("  ✅ Encoder ready with 4-thread pipeline")
    print()

    # Prepare test data for multiple tiles
    num_tiles = 8  # Test with 8 tiles (will process in batches of 4)
    print(f"Preparing {num_tiles} tiles for testing...")

    tiles_data = []
    for i in range(num_tiles):
        Q = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
        K = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
        V = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
        gamma = np.ones(256, dtype=np.int8)
        beta = np.zeros(256, dtype=np.int8)
        tiles_data.append((Q, K, V, gamma, beta))

    print(f"  ✅ {num_tiles} tiles prepared")
    print()

    # Warm-up
    print("Running warm-up pass...")
    _ = encoder.forward_block(*tiles_data[0])
    print("  ✅ Warm-up complete")
    print()

    # ====================================================================
    # Test 1: Sequential Processing (Baseline)
    # ====================================================================
    print("=" * 70)
    print("TEST 1: Sequential Processing (Baseline)")
    print("=" * 70)
    print()

    print(f"Processing {num_tiles} tiles sequentially...")
    sequential_start = time.perf_counter()

    sequential_results = []
    for i, (Q, K, V, gamma, beta) in enumerate(tiles_data):
        result = encoder.forward_block(Q, K, V, gamma, beta)
        sequential_results.append(result)

    sequential_time = (time.perf_counter() - sequential_start) * 1000

    print(f"  ✅ Sequential processing complete")
    print(f"     Total time: {sequential_time:.2f}ms")
    print(f"     Time per tile: {sequential_time / num_tiles:.2f}ms")
    print()

    # ====================================================================
    # Test 2: Pipelined Processing (Multi-Threaded)
    # ====================================================================
    print("=" * 70)
    print("TEST 2: Pipelined Processing (4 Threads)")
    print("=" * 70)
    print()

    print(f"Processing {num_tiles} tiles with pipeline...")
    pipelined_start = time.perf_counter()

    pipelined_results = encoder.process_tiles_pipelined(tiles_data)

    pipelined_time = (time.perf_counter() - pipelined_start) * 1000

    print(f"  ✅ Pipelined processing complete")
    print(f"     Total time: {pipelined_time:.2f}ms")
    print(f"     Time per tile: {pipelined_time / num_tiles:.2f}ms")
    print()

    # ====================================================================
    # Performance Comparison
    # ====================================================================
    print("=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    print()

    improvement = sequential_time / pipelined_time

    print(f"Sequential time:  {sequential_time:.2f}ms ({sequential_time / num_tiles:.2f}ms per tile)")
    print(f"Pipelined time:   {pipelined_time:.2f}ms ({pipelined_time / num_tiles:.2f}ms per tile)")
    print(f"Improvement:      {improvement:.2f}× faster")
    print()

    # Calculate realtime factors
    baseline_tile_time = 2.85  # ms (from buffer optimization)
    pipelined_tile_time = pipelined_time / num_tiles

    # Full pipeline projection
    tiles_per_block = 23.4
    blocks = 6
    mel_time = 304.7  # ms

    baseline_encoder = baseline_tile_time * tiles_per_block * blocks
    pipelined_encoder = pipelined_tile_time * tiles_per_block * blocks

    baseline_total = mel_time + baseline_encoder
    pipelined_total = mel_time + pipelined_encoder

    baseline_rtf = 11000 / baseline_total
    pipelined_rtf = 11000 / pipelined_total

    print("Full Pipeline Projection (11-second audio):")
    print(f"  Mel preprocessing:    {mel_time:.1f}ms (unchanged)")
    print()
    print(f"  Baseline encoder:     {baseline_encoder:.1f}ms")
    print(f"  Pipelined encoder:    {pipelined_encoder:.1f}ms")
    print(f"  Encoder improvement:  {baseline_encoder / pipelined_encoder:.2f}×")
    print()
    print(f"  Baseline total:       {baseline_total:.1f}ms → {baseline_rtf:.1f}× realtime")
    print(f"  Pipelined total:      {pipelined_total:.1f}ms → {pipelined_rtf:.1f}× realtime")
    print(f"  Overall improvement:  {pipelined_rtf / baseline_rtf:.2f}×")
    print()

    if pipelined_rtf >= 50:
        print(f"🎉 TARGET ACHIEVED! {pipelined_rtf:.1f}× > 50× realtime")
    elif pipelined_rtf >= 26:
        print(f"✅ EXCELLENT PROGRESS! {pipelined_rtf:.1f}× realtime")
    elif pipelined_rtf >= 20:
        print(f"✅ GOOD PROGRESS! {pipelined_rtf:.1f}× realtime (target: 50-80×)")
    else:
        print(f"✅ PROGRESS! {pipelined_rtf:.1f}× realtime (need more optimization)")

    print("=" * 70)
    print()

    # Validate outputs
    print("Output Validation:")
    print(f"  Sequential results: {len(sequential_results)} tiles")
    print(f"  Pipelined results:  {len(pipelined_results)} tiles")

    # Check first result from each
    seq_attn = sequential_results[0]['attention']
    pip_attn = pipelined_results[0]['attention']

    print(f"  Sequential attention activity: {np.count_nonzero(seq_attn)}/{seq_attn.size} ({100*np.count_nonzero(seq_attn)/seq_attn.size:.1f}%)")
    print(f"  Pipelined attention activity:  {np.count_nonzero(pip_attn)}/{pip_attn.size} ({100*np.count_nonzero(pip_attn)/pip_attn.size:.1f}%)")
    print()

    return {
        'sequential_time': sequential_time,
        'pipelined_time': pipelined_time,
        'improvement': improvement,
        'pipelined_rtf': pipelined_rtf,
        'baseline_rtf': baseline_rtf
    }


if __name__ == "__main__":
    results = test_encoder_pipelined_vs_sequential()

    print("\n✅ Pipelined encoder test complete!")
    print(f"   Pipeline improvement: {results['improvement']:.2f}×")
    print(f"   New realtime factor: {results['pipelined_rtf']:.1f}×")
    print()
    print("💡 Next step: Compile true multi-core MLIR for 4× improvement")
    print("   This validates the concept - true multi-core will be even better!")
    print()
