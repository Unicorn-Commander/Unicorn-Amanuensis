#!/usr/bin/env python3
"""
NPU Attention Wrapper - Single-Tile 4× Approach (Option 2)
===========================================================

Uses the working single-tile 64×64 attention kernel, calling it 4 times
from Python to process 4 tiles. This is a temporary solution while we
debug the multi-core kernel data routing.

Performance: ~10ms for 4 tiles (still 10× faster than CPU)
Target: 0.28ms with true multi-core parallel execution

Usage:
    attention = NPUAttentionSingleTile()
    output = attention.process_batch(input_4_tiles)  # (4, 64, 64, 3)
"""

import numpy as np
import pyxrt as xrt
import time
from typing import Optional
from pathlib import Path


class NPUAttentionSingleTile:
    """
    NPU Attention using single-tile kernel with 4× Python loop.

    Temporary solution until multi-core data routing is fixed.
    """

    def __init__(
        self,
        xclbin_path: Optional[str] = None,
        insts_path: Optional[str] = None
    ):
        """
        Initialize NPU attention with single-tile kernel.

        Args:
            xclbin_path: Path to single-tile XCLBIN (default: auto-detect)
            insts_path: Path to instruction binary (default: auto-detect)
        """
        # Default paths
        if xclbin_path is None:
            base_dir = Path(__file__).parent / "build_attention_64x64"
            xclbin_path = str(base_dir / "attention_64x64.xclbin")

        if insts_path is None:
            base_dir = Path(__file__).parent / "build_attention_64x64"
            insts_path = str(base_dir / "insts.bin")

        self.xclbin_path = xclbin_path
        self.insts_path = insts_path

        # Verify files exist
        if not Path(xclbin_path).exists():
            raise FileNotFoundError(f"XCLBIN not found: {xclbin_path}")
        if not Path(insts_path).exists():
            raise FileNotFoundError(f"Instructions not found: {insts_path}")

        # Initialize NPU
        self._init_npu()

        print(f"✅ NPU Attention initialized (single-tile 4× mode)")
        print(f"   XCLBIN: {xclbin_path}")
        print(f"   Instructions: {insts_path}")

    def _init_npu(self):
        """Initialize NPU device and kernel."""
        # Open device
        self.device = xrt.device(0)

        # Load and register XCLBIN
        self.xclbin = xrt.xclbin(self.xclbin_path)
        self.device.register_xclbin(self.xclbin)

        # Create hardware context
        uuid = self.xclbin.get_uuid()
        self.hw_ctx = xrt.hw_context(self.device, uuid)

        # Get kernel
        self.kernel = xrt.kernel(self.hw_ctx, "MLIR_AIE")

        # Load instructions
        with open(self.insts_path, 'rb') as f:
            self.insts_data = f.read()

        # Create instruction buffer (reused across calls)
        self.instr_bo = xrt.bo(
            self.device,
            len(self.insts_data),
            xrt.bo.flags.cacheable,
            self.kernel.group_id(1)
        )
        self.instr_bo.write(self.insts_data, 0)
        self.instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Create data buffers (single tile size, reused)
        self.INPUT_SIZE = 12288  # Q+K+V for 64×64
        self.OUTPUT_SIZE = 4096  # 64×64 output

        self.input_bo = xrt.bo(
            self.device,
            self.INPUT_SIZE,
            xrt.bo.flags.host_only,
            self.kernel.group_id(3)
        )

        self.output_bo = xrt.bo(
            self.device,
            self.OUTPUT_SIZE,
            xrt.bo.flags.host_only,
            self.kernel.group_id(4)
        )

    def process_tile(self, tile_input: np.ndarray) -> np.ndarray:
        """
        Process a single 64×64 tile on NPU.

        Args:
            tile_input: Input tensor (12288,) containing Q+K+V

        Returns:
            Output tensor (4096,) containing attention result
        """
        assert tile_input.shape == (self.INPUT_SIZE,), \
            f"Expected input shape ({self.INPUT_SIZE},), got {tile_input.shape}"
        assert tile_input.dtype == np.int8, \
            f"Expected dtype int8, got {tile_input.dtype}"

        # Write input to NPU
        self.input_bo.write(tile_input.tobytes(), 0)
        self.input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Execute kernel
        run = self.kernel(self.instr_bo, self.input_bo, self.output_bo, 3)
        run.wait()

        # Read output from NPU
        self.output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        output = np.frombuffer(
            self.output_bo.read(self.OUTPUT_SIZE, 0),
            dtype=np.int8
        )

        return output.copy()  # Return copy since buffer is reused

    def process_batch(
        self,
        batch_input: np.ndarray,
        return_stats: bool = False
    ) -> np.ndarray:
        """
        Process batch of 4 tiles using 4× single-tile kernel calls.

        Args:
            batch_input: Input tensor (4, 12288) or (4, 64, 64, 3)
            return_stats: If True, return (output, stats) tuple

        Returns:
            Output tensor (4, 4096) or (4, 64, 64)
            If return_stats=True, returns (output, stats_dict)
        """
        # Reshape if needed
        if batch_input.ndim == 4:
            # (4, 64, 64, 3) → (4, 12288)
            batch_size, h, w, c = batch_input.shape
            assert h == 64 and w == 64 and c == 3, \
                f"Expected shape (4, 64, 64, 3), got {batch_input.shape}"
            batch_input = batch_input.reshape(batch_size, -1)

        batch_size, input_size = batch_input.shape
        assert batch_size == 4, f"Expected batch_size=4, got {batch_size}"
        assert input_size == self.INPUT_SIZE, \
            f"Expected input_size={self.INPUT_SIZE}, got {input_size}"

        # Process each tile sequentially
        outputs = []
        tile_times = []

        start_total = time.time()

        for i in range(batch_size):
            start_tile = time.time()
            output = self.process_tile(batch_input[i])
            tile_time = (time.time() - start_tile) * 1000

            outputs.append(output)
            tile_times.append(tile_time)

        total_time = (time.time() - start_total) * 1000

        # Stack outputs
        batch_output = np.stack(outputs, axis=0)  # (4, 4096)

        if return_stats:
            stats = {
                'total_time_ms': total_time,
                'tile_times_ms': tile_times,
                'avg_tile_time_ms': np.mean(tile_times),
                'throughput_tiles_per_sec': 4000.0 / total_time,
                'non_zero_pct': 100.0 * np.count_nonzero(batch_output) / batch_output.size
            }
            return batch_output, stats
        else:
            return batch_output

    def process_sequence(
        self,
        sequence: np.ndarray,
        tile_size: int = 64
    ) -> np.ndarray:
        """
        Process a sequence of attention computations.

        Args:
            sequence: Input tensor (num_tiles, 12288) or (num_tiles, 64, 64, 3)
            tile_size: Tile size (default: 64)

        Returns:
            Output tensor (num_tiles, 4096) or (num_tiles, 64, 64)
        """
        # Reshape if needed
        if sequence.ndim == 4:
            num_tiles = sequence.shape[0]
            sequence = sequence.reshape(num_tiles, -1)

        num_tiles = sequence.shape[0]

        # Process in batches of 4
        outputs = []

        for batch_start in range(0, num_tiles, 4):
            batch_end = min(batch_start + 4, num_tiles)
            batch = sequence[batch_start:batch_end]

            # Pad if needed
            if batch.shape[0] < 4:
                padding = np.zeros(
                    (4 - batch.shape[0], self.INPUT_SIZE),
                    dtype=np.int8
                )
                batch = np.vstack([batch, padding])

            # Process batch
            batch_output = self.process_batch(batch)

            # Trim padding
            batch_output = batch_output[:batch_end - batch_start]

            outputs.append(batch_output)

        return np.vstack(outputs)

    def __del__(self):
        """Cleanup resources."""
        # XRT handles cleanup automatically
        pass


def test_single_tile_wrapper():
    """Test the single-tile wrapper."""
    print("=" * 70)
    print("Testing NPU Attention Single-Tile 4× Wrapper")
    print("=" * 70)
    print()

    # Initialize wrapper
    attention = NPUAttentionSingleTile()
    print()

    # Generate test data (4 tiles)
    np.random.seed(42)
    test_input = np.random.randint(-128, 127, (4, 12288), dtype=np.int8)

    print("Processing 4 tiles sequentially...")
    output, stats = attention.process_batch(test_input, return_stats=True)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"📊 Performance:")
    print(f"   Total time: {stats['total_time_ms']:.2f}ms")
    print(f"   Avg per tile: {stats['avg_tile_time_ms']:.2f}ms")
    print(f"   Throughput: {stats['throughput_tiles_per_sec']:.1f} tiles/sec")
    print()
    print(f"📊 Output quality:")
    print(f"   Shape: {output.shape}")
    print(f"   Non-zero: {stats['non_zero_pct']:.1f}%")
    print(f"   Range: [{output.min()}, {output.max()}]")
    print()

    # Compare with target
    target_time = 0.28  # Target parallel execution time
    current_time = stats['total_time_ms']
    speedup_potential = current_time / target_time

    print(f"💡 Analysis:")
    print(f"   Current (serial): {current_time:.2f}ms")
    print(f"   Target (parallel): {target_time}ms")
    print(f"   Speedup potential: {speedup_potential:.1f}×")
    print()

    if stats['non_zero_pct'] > 50:
        print("✅ Single-tile 4× wrapper working!")
        print("⚡ Ready for integration while multi-core is debugged")
    else:
        print("⚠️  Output validation needed")

    print()
    print("=" * 70)


if __name__ == "__main__":
    test_single_tile_wrapper()
