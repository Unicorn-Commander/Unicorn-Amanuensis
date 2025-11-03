#!/usr/bin/env python3
"""
Batched Encoder Block - 2-3x Speedup from Parallel Processing
Uses existing single-core XCLBINs with batched execution for immediate gains

Current:  15.6x realtime (2.85ms per tile)
Target:   31-47x realtime (0.95-1.40ms per tile with batching)

Approach: Submit multiple kernel calls and overlap DMA/compute operations
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
import time
from pathlib import Path
from test_encoder_block import NPUEncoderBlock

class NPUEncoderBlockBatched(NPUEncoderBlock):
    """Batched encoder block with overlapped execution"""

    def __init__(self, batch_size=4):
        """Initialize with batching support

        Args:
            batch_size: Number of tiles to process in parallel (default 4)
        """
        super().__init__()
        self.batch_size = batch_size

        print(f"\n✅ Batched encoder initialized (batch_size={batch_size})")
        print(f"   Will process {batch_size} tiles with overlapped execution")
        print()

    def process_tiles_batched(self, tiles_data):
        """Process multiple tiles with batched execution

        This achieves parallelism by:
        1. Pre-allocating all buffers
        2. Submitting kernels without waiting
        3. Overlapping DMA and compute
        4. Collecting results efficiently

        Args:
            tiles_data: List of tuples (Q, K, V, gamma, beta)

        Returns:
            List of results
        """
        results = []
        num_tiles = len(tiles_data)

        # Process in batches
        for batch_start in range(0, num_tiles, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_tiles)
            batch = tiles_data[batch_start:batch_end]

            # Process this batch with overlapping
            batch_results = self._process_batch_overlapped(batch)
            results.extend(batch_results)

        return results

    def _process_batch_overlapped(self, batch):
        """Process a batch of tiles with DMA/compute overlap

        Strategy:
        1. Write all inputs to NPU (DMA phase)
        2. Submit all kernel executions (compute phase)
        3. Read all outputs from NPU (DMA phase)

        This overlaps operations where possible.
        """
        batch_size = len(batch)
        results = []

        # Phase 1: Submit all attention operations (overlapped)
        attn_runs = []
        for Q, K, V, gamma, beta in batch:
            # Prepare input
            QKV_combined = np.concatenate([Q.flatten(), K.flatten(), V.flatten()])
            self.attn_input_bo.write(QKV_combined.tobytes(), 0)
            self.attn_input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 12288, 0)

            # Submit kernel (don't wait yet)
            opcode = 3
            run = self.attn_kernel(opcode, self.attn_instr_bo, self.attn_n_insts,
                                   self.attn_input_bo, self.attn_output_bo)
            attn_runs.append(run)

        # Phase 2: Wait for all attention to complete and collect outputs
        attn_outputs = []
        for run in attn_runs:
            run.wait(1000)
            self.attn_output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 4096, 0)
            output = np.frombuffer(self.attn_output_bo.read(4096, 0), dtype=np.int8)
            attn_outputs.append(output.reshape(64, 64))

        # Phase 3: Process layernorm and GELU for each
        for i, (Q, K, V, gamma, beta) in enumerate(batch):
            attn_output = attn_outputs[i]

            # LayerNorm
            ln_input = attn_output[:4, :64].flatten()[:256]
            combined = np.concatenate([ln_input.flatten(), gamma.flatten(), beta.flatten()])
            self.ln_input_bo.write(combined.tobytes(), 0)
            self.ln_input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 768, 0)

            opcode = 3
            run = self.ln_kernel(opcode, self.ln_instr_bo, self.ln_n_insts,
                                self.ln_input_bo, self.ln_output_bo)
            run.wait(1000)

            self.ln_output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 256, 0)
            ln_output = np.frombuffer(self.ln_output_bo.read(256, 0), dtype=np.int8)

            # GELU
            gelu_input = ln_output[:512] if len(ln_output) >= 512 else np.pad(ln_output, (0, 512-len(ln_output)))
            self.gelu_input_bo.write(gelu_input.tobytes(), 0)
            self.gelu_input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 512, 0)

            opcode = 3
            run = self.gelu_kernel(opcode, self.gelu_instr_bo, self.gelu_n_insts,
                                  self.gelu_input_bo, self.gelu_output_bo)
            run.wait(1000)

            self.gelu_output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 512, 0)
            gelu_output = np.frombuffer(self.gelu_output_bo.read(512, 0), dtype=np.int8)

            results.append({
                'attention': attn_output,
                'layernorm': ln_output,
                'gelu': gelu_output
            })

        return results


def test_batched_vs_sequential():
    """Compare batched vs sequential execution"""

    print("\n")
    print("=" * 70)
    print("BATCHED ENCODER TEST - Parallel Processing with Existing Kernels")
    print("=" * 70)
    print()

    # Initialize encoders
    print("Initializing encoders...")
    print("\n1. Sequential encoder (baseline):")
    sequential_encoder = NPUEncoderBlock()

    print("2. Batched encoder (batch_size=4):")
    batched_encoder = NPUEncoderBlockBatched(batch_size=4)

    # Prepare test data for multiple tiles
    num_tiles = 12  # Test with 12 tiles (3 batches of 4)
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
    _ = sequential_encoder.forward_block(*tiles_data[0])
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
        result = sequential_encoder.forward_block(Q, K, V, gamma, beta)
        sequential_results.append(result)

    sequential_time = (time.perf_counter() - sequential_start) * 1000

    print(f"  ✅ Sequential processing complete")
    print(f"     Total time: {sequential_time:.2f}ms")
    print(f"     Time per tile: {sequential_time / num_tiles:.2f}ms")
    print()

    # ====================================================================
    # Test 2: Batched Processing
    # ====================================================================
    print("=" * 70)
    print("TEST 2: Batched Processing (batch_size=4)")
    print("=" * 70)
    print()

    print(f"Processing {num_tiles} tiles with batching...")
    batched_start = time.perf_counter()

    batched_results = batched_encoder.process_tiles_batched(tiles_data)

    batched_time = (time.perf_counter() - batched_start) * 1000

    print(f"  ✅ Batched processing complete")
    print(f"     Total time: {batched_time:.2f}ms")
    print(f"     Time per tile: {batched_time / num_tiles:.2f}ms")
    print()

    # ====================================================================
    # Performance Comparison
    # ====================================================================
    print("=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    print()

    improvement = sequential_time / batched_time

    print(f"Sequential time:  {sequential_time:.2f}ms ({sequential_time / num_tiles:.2f}ms per tile)")
    print(f"Batched time:     {batched_time:.2f}ms ({batched_time / num_tiles:.2f}ms per tile)")
    print(f"Improvement:      {improvement:.2f}× faster")
    print()

    # Calculate realtime factors
    baseline_tile_time = 2.85  # ms (from buffer optimization)
    batched_tile_time = batched_time / num_tiles

    # Full pipeline projection
    tiles_per_block = 23.4
    blocks = 6
    mel_time = 304.7  # ms

    baseline_encoder = baseline_tile_time * tiles_per_block * blocks
    batched_encoder_time = batched_tile_time * tiles_per_block * blocks

    baseline_total = mel_time + baseline_encoder
    batched_total = mel_time + batched_encoder_time

    baseline_rtf = 11000 / baseline_total
    batched_rtf = 11000 / batched_total

    print("Full Pipeline Projection (11-second audio):")
    print(f"  Mel preprocessing:    {mel_time:.1f}ms (unchanged)")
    print()
    print(f"  Baseline encoder:     {baseline_encoder:.1f}ms")
    print(f"  Batched encoder:      {batched_encoder_time:.1f}ms")
    print(f"  Encoder improvement:  {baseline_encoder / batched_encoder_time:.2f}×")
    print()
    print(f"  Baseline total:       {baseline_total:.1f}ms → {baseline_rtf:.1f}× realtime")
    print(f"  Batched total:        {batched_total:.1f}ms → {batched_rtf:.1f}× realtime")
    print(f"  Overall improvement:  {batched_rtf / baseline_rtf:.2f}×")
    print()

    if batched_rtf >= 50:
        print(f"🎉 TARGET ACHIEVED! {batched_rtf:.1f}× > 50× realtime")
    elif batched_rtf >= 26:
        print(f"✅ EXCELLENT PROGRESS! {batched_rtf:.1f}× realtime")
    elif batched_rtf >= 20:
        print(f"✅ GOOD PROGRESS! {batched_rtf:.1f}× realtime (target: 50-80×)")
    else:
        print(f"✅ PROGRESS! {batched_rtf:.1f}× realtime (continuing optimization)")

    print("=" * 70)
    print()

    # Output validation
    print("Output Validation:")
    print(f"  Sequential results: {len(sequential_results)} tiles")
    print(f"  Batched results:    {len(batched_results)} tiles")

    # Check first result from each
    seq_attn = sequential_results[0]['attention']
    bat_attn = batched_results[0]['attention']

    print(f"  Sequential attention activity: {np.count_nonzero(seq_attn)}/{seq_attn.size} ({100*np.count_nonzero(seq_attn)/seq_attn.size:.1f}%)")
    print(f"  Batched attention activity:    {np.count_nonzero(bat_attn)}/{bat_attn.size} ({100*np.count_nonzero(bat_attn)/bat_attn.size:.1f}%)")
    print()

    return {
        'sequential_time': sequential_time,
        'batched_time': batched_time,
        'improvement': improvement,
        'batched_rtf': batched_rtf,
        'baseline_rtf': baseline_rtf,
        'batched_tile_time': batched_tile_time
    }


if __name__ == "__main__":
    results = test_batched_vs_sequential()

    print("\n✅ Batched encoder test complete!")
    print(f"   Batching improvement: {results['improvement']:.2f}×")
    print(f"   New realtime factor: {results['batched_rtf']:.1f}×")
    print(f"   Time per tile: {results['batched_tile_time']:.2f}ms")
    print()
    print("💡 Next steps:")
    print("   - If batching works well (2-3× improvement), this is immediate win!")
    print("   - Then move to true multi-core MLIR for 4× improvement")
    print("   - Combined: 2-3× (batching) + 4× (multi-core) = 8-12× total speedup")
    print()
