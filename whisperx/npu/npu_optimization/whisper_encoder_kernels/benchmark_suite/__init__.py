"""
NPU Whisper Benchmark Suite

Comprehensive benchmarking and validation framework for tracking
progress toward 220x realtime target on AMD Phoenix NPU.

Components:
- KernelBenchmark: Individual kernel performance measurement
- PipelineBenchmark: End-to-end pipeline benchmarking
- AccuracyBenchmark: Output quality validation
- BenchmarkComparison: Optimization comparison framework
- BenchmarkReport: Report generation and visualization

Usage:
    from benchmark_suite import KernelBenchmark, PipelineBenchmark

    kernel_bench = KernelBenchmark()
    results = kernel_bench.benchmark_all_kernels()
"""

__version__ = "1.0.0"
__author__ = "Magic Unicorn Unconventional Technology & Stuff Inc."

from .benchmark_kernels import KernelBenchmark
from .benchmark_pipeline import PipelineBenchmark
from .benchmark_accuracy import AccuracyBenchmark
from .benchmark_comparison import BenchmarkComparison
from .benchmark_report import BenchmarkReport

__all__ = [
    'KernelBenchmark',
    'PipelineBenchmark',
    'AccuracyBenchmark',
    'BenchmarkComparison',
    'BenchmarkReport'
]
