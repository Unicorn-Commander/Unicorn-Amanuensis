#!/usr/bin/env python3
"""
NPU Performance Benchmark Suite

Comprehensive performance benchmarking for BFP16 NPU implementation.

Benchmarks:
- Matmul operations (512×512×512): Latency, GFLOPS, ops/sec
- Single encoder layer: ms/layer, layers/sec
- Full 6-layer encoder: ms/encode, encodes/sec, realtime factor
- Batch scaling: Throughput vs batch size (1, 2, 4, 8)
- Warmup effect: Cold start vs warm performance
- Memory bandwidth: GB/s transfer rates

Performance Targets:
- 512×512×512 matmul: <2ms, >100 GFLOPS
- Single layer: <8ms, >125 layers/sec
- Full encoder: <50ms, >20 encodes/sec
- Whisper Base (30s audio): <75ms, >400× realtime
"""

import numpy as np
import ctypes
from ctypes import c_void_p, c_float, c_size_t, POINTER
import time
import sys
from pathlib import Path
from typing import Dict, List
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NPUPerformanceBenchmark:
    """NPU performance benchmarking framework"""

    def __init__(self, lib_path: str = None):
        """
        Initialize NPU performance benchmark

        Args:
            lib_path: Path to C++ library (default: auto-detect)
        """
        if lib_path is None:
            lib_path = "/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build/libwhisper_encoder_cpp.so"

        if not Path(lib_path).exists():
            raise FileNotFoundError(f"C++ library not found: {lib_path}")

        logger.info(f"Loading C++ library: {lib_path}")
        self.lib = ctypes.CDLL(lib_path)

        # Define C API
        self._define_c_api()

        logger.info("NPU performance benchmark initialized")

    def _define_c_api(self):
        """Define C API bindings"""
        # EncoderLayer API
        self.lib.encoder_layer_create.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t]
        self.lib.encoder_layer_create.restype = c_void_p

        self.lib.encoder_layer_destroy.argtypes = [c_void_p]

        self.lib.encoder_layer_load_weights.argtypes = [
            c_void_p,
            POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
            POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
            POINTER(c_float), POINTER(c_float),
            POINTER(c_float), POINTER(c_float),
            POINTER(c_float), POINTER(c_float),
            POINTER(c_float), POINTER(c_float),
            c_size_t, c_size_t
        ]
        self.lib.encoder_layer_load_weights.restype = ctypes.c_int

        self.lib.encoder_layer_forward.argtypes = [
            c_void_p,
            POINTER(c_float),
            POINTER(c_float),
            c_size_t,
            c_size_t
        ]
        self.lib.encoder_layer_forward.restype = ctypes.c_int

    def benchmark_matmul_512x512x512(self, num_iterations: int = 100, warmup: int = 10) -> Dict:
        """
        Benchmark: 512×512×512 matmul
        Target: <2ms, >100 GFLOPS

        Args:
            num_iterations: Number of iterations for benchmark
            warmup: Number of warmup iterations

        Returns:
            metrics: Performance metrics dictionary
        """
        logger.info("="*80)
        logger.info("Benchmark: Matmul 512×512×512")
        logger.info("="*80)

        # TODO: Implement when NPU matmul is available
        metrics = {
            'benchmark': 'matmul_512x512x512',
            'status': 'SKIPPED',
            'reason': 'NPU matmul not yet implemented',
            'target_latency_ms': 2.0,
            'target_gflops': 100.0,
        }

        logger.warning(f"Benchmark skipped: {metrics['reason']}")
        return metrics

    def benchmark_single_layer(self, num_iterations: int = 100, warmup: int = 10) -> Dict:
        """
        Benchmark: Single encoder layer
        Target: <8ms, >125 layers/sec

        Args:
            num_iterations: Number of iterations for benchmark
            warmup: Number of warmup iterations

        Returns:
            metrics: Performance metrics dictionary
        """
        logger.info("="*80)
        logger.info("Benchmark: Single Encoder Layer")
        logger.info("="*80)

        try:
            # Create encoder layer
            layer = self.lib.encoder_layer_create(0, 8, 512, 2048)
            if not layer:
                raise RuntimeError("Failed to create encoder layer")

            # Load weights
            n_state = 512
            ffn_dim = 2048

            # Generate random weights (lightweight for benchmarking)
            q_w = np.random.randn(n_state, n_state).astype(np.float32)
            k_w = np.random.randn(n_state, n_state).astype(np.float32)
            v_w = np.random.randn(n_state, n_state).astype(np.float32)
            out_w = np.random.randn(n_state, n_state).astype(np.float32)
            fc1_w = np.random.randn(ffn_dim, n_state).astype(np.float32)
            fc2_w = np.random.randn(n_state, ffn_dim).astype(np.float32)

            q_b = np.zeros(n_state, dtype=np.float32)
            k_b = np.zeros(n_state, dtype=np.float32)
            v_b = np.zeros(n_state, dtype=np.float32)
            out_b = np.zeros(n_state, dtype=np.float32)
            fc1_b = np.zeros(ffn_dim, dtype=np.float32)
            fc2_b = np.zeros(n_state, dtype=np.float32)

            attn_ln_w = np.ones(n_state, dtype=np.float32)
            attn_ln_b = np.zeros(n_state, dtype=np.float32)
            ffn_ln_w = np.ones(n_state, dtype=np.float32)
            ffn_ln_b = np.zeros(n_state, dtype=np.float32)

            result = self.lib.encoder_layer_load_weights(
                layer,
                q_w.ctypes.data_as(POINTER(c_float)),
                k_w.ctypes.data_as(POINTER(c_float)),
                v_w.ctypes.data_as(POINTER(c_float)),
                out_w.ctypes.data_as(POINTER(c_float)),
                q_b.ctypes.data_as(POINTER(c_float)),
                k_b.ctypes.data_as(POINTER(c_float)),
                v_b.ctypes.data_as(POINTER(c_float)),
                out_b.ctypes.data_as(POINTER(c_float)),
                fc1_w.ctypes.data_as(POINTER(c_float)),
                fc2_w.ctypes.data_as(POINTER(c_float)),
                fc1_b.ctypes.data_as(POINTER(c_float)),
                fc2_b.ctypes.data_as(POINTER(c_float)),
                attn_ln_w.ctypes.data_as(POINTER(c_float)),
                attn_ln_b.ctypes.data_as(POINTER(c_float)),
                ffn_ln_w.ctypes.data_as(POINTER(c_float)),
                ffn_ln_b.ctypes.data_as(POINTER(c_float)),
                n_state,
                ffn_dim
            )

            if result != 0:
                raise RuntimeError("Failed to load weights")

            # Prepare input/output
            input_np = np.random.randn(1500, 512).astype(np.float32)
            output_np = np.zeros((1500, 512), dtype=np.float32)

            # Warmup
            logger.info(f"Warming up ({warmup} iterations)...")
            for _ in range(warmup):
                self.lib.encoder_layer_forward(
                    layer,
                    input_np.ctypes.data_as(POINTER(c_float)),
                    output_np.ctypes.data_as(POINTER(c_float)),
                    1500,
                    512
                )

            # Benchmark
            logger.info(f"Benchmarking ({num_iterations} iterations)...")
            latencies = []

            for _ in range(num_iterations):
                start = time.perf_counter()
                result = self.lib.encoder_layer_forward(
                    layer,
                    input_np.ctypes.data_as(POINTER(c_float)),
                    output_np.ctypes.data_as(POINTER(c_float)),
                    1500,
                    512
                )
                end = time.perf_counter()

                if result != 0:
                    raise RuntimeError("Forward pass failed")

                latencies.append((end - start) * 1000)  # Convert to ms

            # Compute statistics
            latencies = np.array(latencies)
            mean_latency = np.mean(latencies)
            median_latency = np.median(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)

            throughput = 1000.0 / mean_latency  # layers/sec

            metrics = {
                'benchmark': 'single_layer',
                'status': 'PASS',
                'iterations': num_iterations,
                'warmup': warmup,
                'mean_latency_ms': float(mean_latency),
                'median_latency_ms': float(median_latency),
                'p95_latency_ms': float(p95_latency),
                'p99_latency_ms': float(p99_latency),
                'min_latency_ms': float(min_latency),
                'max_latency_ms': float(max_latency),
                'throughput_layers_per_sec': float(throughput),
                'target_latency_ms': 8.0,
                'target_throughput': 125.0,
                'meets_target': mean_latency < 8.0,
            }

            # Cleanup
            self.lib.encoder_layer_destroy(layer)

            # Log results
            logger.info(f"Mean latency: {mean_latency:.2f} ms (target: <8ms)")
            logger.info(f"Throughput: {throughput:.1f} layers/sec (target: >125)")
            logger.info(f"Meets target: {'YES' if metrics['meets_target'] else 'NO'}")

            return metrics

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {
                'benchmark': 'single_layer',
                'status': 'FAILED',
                'error': str(e),
            }

    def benchmark_full_encoder(self, num_iterations: int = 50, warmup: int = 5) -> Dict:
        """
        Benchmark: Full 6-layer encoder
        Target: <50ms, >20 encodes/sec, >400× realtime

        Args:
            num_iterations: Number of iterations for benchmark
            warmup: Number of warmup iterations

        Returns:
            metrics: Performance metrics dictionary
        """
        logger.info("="*80)
        logger.info("Benchmark: Full 6-Layer Encoder")
        logger.info("="*80)

        # TODO: Implement when all 6 layers are working
        metrics = {
            'benchmark': 'full_encoder_6_layers',
            'status': 'SKIPPED',
            'reason': 'Requires all 6 layers with NPU integration',
            'target_latency_ms': 50.0,
            'target_throughput': 20.0,
            'target_realtime_factor': 400.0,
        }

        logger.warning(f"Benchmark skipped: {metrics['reason']}")
        return metrics

    def benchmark_batch_scaling(self, batch_sizes: List[int] = [1, 2, 4, 8]) -> Dict:
        """
        Benchmark: Batch scaling
        Measure throughput vs batch size

        Args:
            batch_sizes: List of batch sizes to test

        Returns:
            metrics: Performance metrics dictionary
        """
        logger.info("="*80)
        logger.info("Benchmark: Batch Scaling")
        logger.info("="*80)

        # TODO: Implement batch processing
        metrics = {
            'benchmark': 'batch_scaling',
            'status': 'SKIPPED',
            'reason': 'Batch processing not yet implemented',
            'batch_sizes': batch_sizes,
        }

        logger.warning(f"Benchmark skipped: {metrics['reason']}")
        return metrics

    def benchmark_warmup_effect(self, num_iterations: int = 100) -> Dict:
        """
        Benchmark: Warmup effect
        Measure cold start vs warm performance

        Args:
            num_iterations: Number of iterations

        Returns:
            metrics: Performance metrics dictionary
        """
        logger.info("="*80)
        logger.info("Benchmark: Warmup Effect")
        logger.info("="*80)

        # TODO: Implement warmup analysis
        metrics = {
            'benchmark': 'warmup_effect',
            'status': 'SKIPPED',
            'reason': 'Requires NPU integration',
        }

        logger.warning(f"Benchmark skipped: {metrics['reason']}")
        return metrics

    def run_all_benchmarks(self) -> Dict:
        """Run all performance benchmarks"""
        logger.info("="*80)
        logger.info("  NPU PERFORMANCE BENCHMARK SUITE")
        logger.info("="*80)

        all_results = {}

        # Run benchmarks
        all_results['matmul_512x512x512'] = self.benchmark_matmul_512x512x512()
        all_results['single_layer'] = self.benchmark_single_layer()
        all_results['full_encoder'] = self.benchmark_full_encoder()
        all_results['batch_scaling'] = self.benchmark_batch_scaling()
        all_results['warmup_effect'] = self.benchmark_warmup_effect()

        # Summary
        logger.info("="*80)
        logger.info("  BENCHMARK SUMMARY")
        logger.info("="*80)

        total = len(all_results)
        passed = sum(1 for r in all_results.values() if r.get('status') == 'PASS')
        failed = sum(1 for r in all_results.values() if r.get('status') == 'FAILED')
        skipped = sum(1 for r in all_results.values() if r.get('status') == 'SKIPPED')

        logger.info(f"Total benchmarks: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped: {skipped}")

        all_results['summary'] = {
            'total': total,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
        }

        return all_results


def main():
    """Run NPU performance benchmarks"""
    try:
        benchmark = NPUPerformanceBenchmark()
        results = benchmark.run_all_benchmarks()

        # Save results
        output_dir = Path("./tests/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "npu_performance_results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {output_dir / 'npu_performance_results.json'}")

        # Exit with appropriate code
        if results['summary']['failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
