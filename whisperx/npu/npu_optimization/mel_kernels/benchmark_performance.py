#!/usr/bin/env python3
"""
NPU Mel Spectrogram Performance Benchmarking Suite

Comprehensive performance comparison between simple and optimized NPU kernels.
Measures:
- Processing time per frame (µs precision)
- Throughput (frames per second)
- Realtime factor (vs audio duration)
- Statistical analysis (mean, std dev, min, max)
- Memory footprint
- XCLBIN size comparison

Author: Magic Unicorn Inc. - Performance Metrics Lead
Date: October 28, 2025
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import pyxrt as xrt
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import statistics


@dataclass
class BenchmarkResult:
    """Performance benchmark result for single kernel"""
    kernel_name: str
    xclbin_path: str
    xclbin_size_bytes: int
    num_iterations: int

    # Timing metrics (seconds)
    execution_times: List[float]
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    median_time: float

    # Per-frame metrics (microseconds)
    mean_time_us: float
    std_time_us: float
    min_time_us: float
    max_time_us: float

    # Throughput metrics
    frames_per_second: float
    realtime_factor: float

    # NPU utilization (estimated)
    estimated_npu_utilization_pct: float

    # Timestamps
    timestamp: str

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class NPUPerformanceBenchmark:
    """Benchmark NPU mel spectrogram performance"""

    # Audio configuration (matches Whisper preprocessing)
    SAMPLE_RATE = 16000  # Hz
    FRAME_SIZE = 400     # samples (25ms at 16kHz)
    FRAME_DURATION_MS = (FRAME_SIZE / SAMPLE_RATE) * 1000  # 25ms

    # Buffer sizes (in bytes)
    INPUT_SIZE = 800   # 400 INT16 samples = 800 bytes
    OUTPUT_SIZE = 80   # 80 INT8 mel bins = 80 bytes

    def __init__(self, xclbin_path: str, kernel_name: str):
        """Initialize NPU benchmark for specific kernel

        Args:
            xclbin_path: Path to XCLBIN file
            kernel_name: Name for reporting (e.g., "simple", "optimized")
        """
        self.xclbin_path = Path(xclbin_path)
        self.kernel_name = kernel_name

        if not self.xclbin_path.exists():
            raise FileNotFoundError(f"XCLBIN not found: {xclbin_path}")

        # Get XCLBIN size
        self.xclbin_size = self.xclbin_path.stat().st_size

        print(f"Initializing NPU for {kernel_name}...")
        print(f"  XCLBIN: {xclbin_path}")
        print(f"  Size: {self.xclbin_size:,} bytes ({self.xclbin_size/1024:.2f} KB)")

        # Initialize NPU
        self.device = xrt.device(0)
        self.xclbin = xrt.xclbin(str(self.xclbin_path))
        uuid = self.xclbin.get_uuid()
        self.device.register_xclbin(self.xclbin)

        # Create hardware context
        self.hw_ctx = xrt.hw_context(self.device, uuid)
        self.kernel = xrt.kernel(self.hw_ctx, "MLIR_AIE")

        # Allocate buffers (reuse for all iterations)
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

        # Instruction buffer (for NPU instructions)
        insts_path = self.xclbin_path.parent / "insts.bin"
        if insts_path.exists():
            with open(insts_path, "rb") as f:
                self.insts_bin = f.read()
            self.num_instr = len(self.insts_bin) // 4
        else:
            # Empty instruction buffer for kernels without insts.bin
            self.insts_bin = bytes(300)
            self.num_instr = 0

        self.instr_bo = xrt.bo(
            self.device,
            max(len(self.insts_bin), 300),
            xrt.bo.flags.cacheable,
            self.kernel.group_id(1)
        )

        # Write instructions
        instr_map = self.instr_bo.map()
        instr_map[:len(self.insts_bin)] = self.insts_bin
        self.instr_bo.sync(
            xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
            len(self.insts_bin),
            0
        )

        print(f"✅ NPU initialized for {kernel_name}")

    def generate_test_audio(self) -> np.ndarray:
        """Generate synthetic test audio (400 INT16 samples)

        Returns:
            audio_int16: INT16 audio samples
        """
        # Generate realistic audio signal (sine wave + harmonics + noise)
        t = np.linspace(0, self.FRAME_DURATION_MS / 1000, self.FRAME_SIZE)

        signal = (
            0.3 * np.sin(2 * np.pi * 200 * t) +   # 200Hz fundamental
            0.2 * np.sin(2 * np.pi * 400 * t) +   # 400Hz harmonic
            0.15 * np.sin(2 * np.pi * 800 * t) +  # 800Hz harmonic
            0.1 * np.random.randn(self.FRAME_SIZE)  # Noise
        )

        # Normalize to INT16 range
        signal = signal / np.max(np.abs(signal)) * 32000
        audio_int16 = signal.astype(np.int16)

        return audio_int16

    def run_single_iteration(self, audio_int16: np.ndarray) -> float:
        """Run single kernel execution and measure time

        Args:
            audio_int16: Input audio samples

        Returns:
            execution_time: Time in seconds
        """
        # Write input data
        input_data = audio_int16.tobytes()
        self.input_bo.write(input_data, 0)
        self.input_bo.sync(
            xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
            self.INPUT_SIZE,
            0
        )

        # Execute kernel with high-precision timing
        start = time.perf_counter()

        opcode = 3  # Standard opcode for NPU execution
        run = self.kernel(opcode, self.instr_bo, self.num_instr, self.input_bo, self.output_bo)
        state = run.wait(5000)  # 5 second timeout

        end = time.perf_counter()

        if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"Kernel execution failed: {state}")

        # Read output (for completeness, time not critical here)
        self.output_bo.sync(
            xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
            self.OUTPUT_SIZE,
            0
        )

        return end - start

    def run_benchmark(self, num_iterations: int = 100, warmup_iterations: int = 10) -> BenchmarkResult:
        """Run complete benchmark with statistical analysis

        Args:
            num_iterations: Number of timed iterations
            warmup_iterations: Number of warmup iterations (not timed)

        Returns:
            BenchmarkResult with comprehensive metrics
        """
        print(f"\nRunning benchmark: {self.kernel_name}")
        print(f"  Warmup: {warmup_iterations} iterations")
        print(f"  Timed: {num_iterations} iterations")
        print()

        # Generate test audio once
        audio_int16 = self.generate_test_audio()

        # Warmup phase (stabilize NPU, cache effects)
        print("  Warmup phase...", end=' ', flush=True)
        for _ in range(warmup_iterations):
            self.run_single_iteration(audio_int16)
        print("✅")

        # Benchmark phase
        print(f"  Benchmark phase ({num_iterations} iterations)...")
        execution_times = []

        for i in range(num_iterations):
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i+1}/{num_iterations}", flush=True)

            exec_time = self.run_single_iteration(audio_int16)
            execution_times.append(exec_time)

        print("  ✅ Benchmark complete!")
        print()

        # Calculate statistics
        mean_time = statistics.mean(execution_times)
        std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        min_time = min(execution_times)
        max_time = max(execution_times)
        median_time = statistics.median(execution_times)

        # Convert to microseconds
        mean_time_us = mean_time * 1_000_000
        std_time_us = std_time * 1_000_000
        min_time_us = min_time * 1_000_000
        max_time_us = max_time * 1_000_000

        # Calculate throughput
        frames_per_second = 1.0 / mean_time if mean_time > 0 else 0

        # Calculate realtime factor
        # One frame = 25ms of audio, if we process in X ms, RTF = 25/X
        audio_duration_s = self.FRAME_SIZE / self.SAMPLE_RATE
        realtime_factor = audio_duration_s / mean_time if mean_time > 0 else 0

        # Estimate NPU utilization (very rough)
        # Assume ideal processing time is 10µs (pure compute without overhead)
        ideal_time_us = 10.0
        estimated_utilization = (ideal_time_us / mean_time_us) * 100 if mean_time_us > 0 else 0
        estimated_utilization = min(100.0, estimated_utilization)  # Cap at 100%

        # Create result
        result = BenchmarkResult(
            kernel_name=self.kernel_name,
            xclbin_path=str(self.xclbin_path),
            xclbin_size_bytes=self.xclbin_size,
            num_iterations=num_iterations,
            execution_times=execution_times,
            mean_time=mean_time,
            std_time=std_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            mean_time_us=mean_time_us,
            std_time_us=std_time_us,
            min_time_us=min_time_us,
            max_time_us=max_time_us,
            frames_per_second=frames_per_second,
            realtime_factor=realtime_factor,
            estimated_npu_utilization_pct=estimated_utilization,
            timestamp=datetime.now().isoformat()
        )

        return result

    def cleanup(self):
        """Clean up NPU resources"""
        # XRT automatically cleans up when objects go out of scope
        pass


def print_benchmark_summary(result: BenchmarkResult):
    """Print formatted benchmark summary

    Args:
        result: BenchmarkResult to print
    """
    print("=" * 70)
    print(f"BENCHMARK RESULTS: {result.kernel_name.upper()}")
    print("=" * 70)
    print()

    print("XCLBIN Information:")
    print(f"  Path: {result.xclbin_path}")
    print(f"  Size: {result.xclbin_size_bytes:,} bytes ({result.xclbin_size_bytes/1024:.2f} KB)")
    print()

    print("Timing Statistics (microseconds):")
    print(f"  Mean:     {result.mean_time_us:8.2f} µs")
    print(f"  Std Dev:  {result.std_time_us:8.2f} µs")
    print(f"  Median:   {result.median_time*1e6:8.2f} µs")
    print(f"  Min:      {result.min_time_us:8.2f} µs")
    print(f"  Max:      {result.max_time_us:8.2f} µs")
    print(f"  CV:       {(result.std_time_us/result.mean_time_us)*100:8.2f} % (lower is better)")
    print()

    print("Performance Metrics:")
    print(f"  Throughput:       {result.frames_per_second:,.1f} frames/second")
    print(f"  Realtime Factor:  {result.realtime_factor:.1f}x")
    print(f"  NPU Utilization:  {result.estimated_npu_utilization_pct:.1f}%")
    print()

    print("Audio Processing:")
    frame_duration_ms = (NPUPerformanceBenchmark.FRAME_SIZE / NPUPerformanceBenchmark.SAMPLE_RATE) * 1000
    print(f"  Frame Duration:   {frame_duration_ms:.2f} ms (audio)")
    print(f"  Processing Time:  {result.mean_time*1000:.4f} ms (compute)")
    print(f"  Speedup:          {result.realtime_factor:.1f}x faster than realtime")
    print()

    print(f"Iterations: {result.num_iterations}")
    print(f"Timestamp:  {result.timestamp}")
    print()


def compare_kernels(simple_result: BenchmarkResult, optimized_result: BenchmarkResult):
    """Print comparison between simple and optimized kernels

    Args:
        simple_result: Results from simple kernel
        optimized_result: Results from optimized kernel
    """
    print("=" * 70)
    print("KERNEL COMPARISON: SIMPLE vs OPTIMIZED")
    print("=" * 70)
    print()

    # XCLBIN size comparison
    print("XCLBIN Size:")
    print(f"  Simple:     {simple_result.xclbin_size_bytes:6,} bytes ({simple_result.xclbin_size_bytes/1024:.2f} KB)")
    print(f"  Optimized:  {optimized_result.xclbin_size_bytes:6,} bytes ({optimized_result.xclbin_size_bytes/1024:.2f} KB)")
    size_ratio = optimized_result.xclbin_size_bytes / simple_result.xclbin_size_bytes
    print(f"  Ratio:      {size_ratio:.2f}x (optimized/simple)")
    print()

    # Processing time comparison
    print("Processing Time (microseconds):")
    print(f"  Simple:     {simple_result.mean_time_us:8.2f} µs ± {simple_result.std_time_us:.2f} µs")
    print(f"  Optimized:  {optimized_result.mean_time_us:8.2f} µs ± {optimized_result.std_time_us:.2f} µs")
    time_ratio = optimized_result.mean_time_us / simple_result.mean_time_us
    overhead_pct = (time_ratio - 1.0) * 100

    if overhead_pct > 0:
        print(f"  Overhead:   +{overhead_pct:.1f}% ({optimized_result.mean_time_us - simple_result.mean_time_us:.2f} µs slower)")
    else:
        print(f"  Speedup:    {-overhead_pct:.1f}% ({simple_result.mean_time_us - optimized_result.mean_time_us:.2f} µs faster)")
    print()

    # Throughput comparison
    print("Throughput (frames/second):")
    print(f"  Simple:     {simple_result.frames_per_second:10,.1f} fps")
    print(f"  Optimized:  {optimized_result.frames_per_second:10,.1f} fps")
    print()

    # Realtime factor comparison
    print("Realtime Factor:")
    print(f"  Simple:     {simple_result.realtime_factor:8.1f}x")
    print(f"  Optimized:  {optimized_result.realtime_factor:8.1f}x")
    print()

    # Variance comparison
    print("Timing Variance (lower is better):")
    simple_cv = (simple_result.std_time_us / simple_result.mean_time_us) * 100
    optimized_cv = (optimized_result.std_time_us / optimized_result.mean_time_us) * 100
    print(f"  Simple:     {simple_cv:.2f}% CV")
    print(f"  Optimized:  {optimized_cv:.2f}% CV")

    if optimized_cv < simple_cv:
        print(f"  Winner:     Optimized ({simple_cv - optimized_cv:.2f}% more consistent)")
    else:
        print(f"  Winner:     Simple ({optimized_cv - simple_cv:.2f}% more consistent)")
    print()

    # Overall recommendation
    print("RECOMMENDATION:")
    print("-" * 70)

    # Decision criteria:
    # 1. If optimized is significantly faster (>10%) and size increase is acceptable (<2x)
    # 2. If optimized has better variance (more predictable)
    # 3. If simple is faster but only slightly (<10%), prefer optimized for features

    if overhead_pct < -10:  # Optimized is >10% faster
        print("✅ USE OPTIMIZED: Significant performance improvement")
        print(f"   Optimized is {-overhead_pct:.1f}% faster with better features")
    elif overhead_pct < 10 and optimized_cv < simple_cv:  # Similar speed but more consistent
        print("✅ USE OPTIMIZED: More consistent performance")
        print(f"   Similar speed with {simple_cv - optimized_cv:.2f}% better consistency")
    elif overhead_pct < 20:  # <20% overhead
        print("⚠️  USE OPTIMIZED (with caution): Acceptable overhead for features")
        print(f"   +{overhead_pct:.1f}% overhead, but {optimized_result.xclbin_size_bytes/1024:.1f} KB size")
    else:  # >20% overhead
        print("❌ USE SIMPLE: Optimized kernel has too much overhead")
        print(f"   +{overhead_pct:.1f}% overhead not justified by features")

    print()


def save_results(results: Dict[str, BenchmarkResult], output_dir: Path):
    """Save benchmark results to JSON

    Args:
        results: Dictionary of benchmark results
        output_dir: Output directory
    """
    output_dir.mkdir(exist_ok=True)

    # Convert results to JSON-serializable format
    results_dict = {
        name: result.to_dict()
        for name, result in results.items()
    }

    # Save to file
    output_file = output_dir / "performance_benchmarks.json"
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"✅ Results saved: {output_file}")


def main():
    """Main benchmark execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark NPU mel spectrogram performance"
    )
    parser.add_argument(
        '--simple-xclbin',
        default='build/mel_simple.xclbin',
        help='Path to simple kernel XCLBIN'
    )
    parser.add_argument(
        '--optimized-xclbin',
        default='build/mel_int8_optimized.xclbin',
        help='Path to optimized kernel XCLBIN'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of benchmark iterations (default: 100)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Number of warmup iterations (default: 10)'
    )
    parser.add_argument(
        '--output-dir',
        default='benchmark_results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NPU MEL KERNEL PERFORMANCE BENCHMARK")
    print("=" * 70)
    print()
    print(f"Simple Kernel:    {args.simple_xclbin}")
    print(f"Optimized Kernel: {args.optimized_xclbin}")
    print(f"Iterations:       {args.iterations}")
    print(f"Warmup:           {args.warmup}")
    print()

    results = {}

    try:
        # Benchmark simple kernel
        print("BENCHMARKING SIMPLE KERNEL")
        print("-" * 70)
        simple_bench = NPUPerformanceBenchmark(args.simple_xclbin, "simple")
        simple_result = simple_bench.run_benchmark(args.iterations, args.warmup)
        simple_bench.cleanup()
        results['simple'] = simple_result
        print_benchmark_summary(simple_result)

        # Benchmark optimized kernel
        print("BENCHMARKING OPTIMIZED KERNEL")
        print("-" * 70)
        optimized_bench = NPUPerformanceBenchmark(args.optimized_xclbin, "optimized")
        optimized_result = optimized_bench.run_benchmark(args.iterations, args.warmup)
        optimized_bench.cleanup()
        results['optimized'] = optimized_result
        print_benchmark_summary(optimized_result)

        # Compare results
        compare_kernels(simple_result, optimized_result)

        # Save results
        output_dir = Path(args.output_dir)
        save_results(results, output_dir)

        print()
        print("=" * 70)
        print("BENCHMARK COMPLETE!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Review results: benchmark_results/performance_benchmarks.json")
        print("  2. Generate charts: python3 create_performance_charts.py")
        print("  3. Create report: python3 generate_performance_report.py")
        print()

    except Exception as e:
        print()
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
