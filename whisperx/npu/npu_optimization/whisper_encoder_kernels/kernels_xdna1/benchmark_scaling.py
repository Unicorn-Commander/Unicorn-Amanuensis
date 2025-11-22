#!/usr/bin/env python3
"""
NPU Kernel Scaling Benchmark - Memory Bandwidth Limit Detection
================================================================

Tests Softmax kernel at multiple sizes to find where performance degrades.
The kernel processes 1024 BF16 elements per invocation (memref<2048xi8>).
For larger sizes, we run multiple iterations and measure time per element.

Test Sizes:
- 1024 elements (baseline) - 1 iteration
- 4096 elements - 4 iterations
- 8192 elements - 8 iterations
- 16384 elements - 16 iterations
- 32768 elements - 32 iterations

AMD Phoenix NPU - XDNA1
"""

import numpy as np
import pyxrt as xrt
import struct
import time
import os

# =============================================================================
# BF16 Conversion Functions (Optimized)
# =============================================================================

def bf16_to_float(bf16_bytes):
    """Optimized vectorized BF16 to float conversion"""
    bf16_array = np.frombuffer(bf16_bytes, dtype=np.uint16)
    int32_array = bf16_array.astype(np.uint32) << 16
    return int32_array.view(np.float32)

def float_to_bf16(floats):
    """Optimized vectorized float to BF16 conversion"""
    float_array = np.asarray(floats, dtype=np.float32)
    int32_array = float_array.view(np.uint32)
    bf16_array = ((int32_array >> 16) & 0xFFFF).astype(np.uint16)
    return bf16_array.tobytes()

def softmax_ref(x):
    """Reference softmax implementation"""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

# =============================================================================
# Scaling Benchmark Class
# =============================================================================

class ScalingBenchmark:
    """Benchmark for testing kernel scaling behavior"""

    def __init__(self, device, xclbin_path, insts_path):
        self.device = device
        self.base_elements = 1024
        self.buffer_size = self.base_elements * 2  # BF16 = 2 bytes

        # Load XCLBIN
        xclbin_obj = xrt.xclbin(xclbin_path)
        uuid = xclbin_obj.get_uuid()
        device.register_xclbin(xclbin_obj)

        # Create context and kernel
        self.hw_ctx = xrt.hw_context(device, uuid)
        self.kernel = xrt.kernel(self.hw_ctx, "MLIR_AIE")

        # Load instructions
        with open(insts_path, "rb") as f:
            self.insts = f.read()

        # Allocate buffers
        self.bo_instr = xrt.bo(device, len(self.insts),
                               xrt.bo.flags.cacheable, self.kernel.group_id(1))
        self.bo_in = xrt.bo(device, self.buffer_size,
                           xrt.bo.flags.host_only, self.kernel.group_id(3))
        self.bo_out = xrt.bo(device, self.buffer_size,
                            xrt.bo.flags.host_only, self.kernel.group_id(4))

        # Write instructions
        self.bo_instr.write(self.insts, 0)
        self.bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def run_single_iteration(self, input_bf16):
        """Run single kernel invocation"""
        # Write input
        self.bo_in.write(input_bf16, 0)
        self.bo_in.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Execute kernel
        start = time.perf_counter_ns()
        run = self.kernel(3, self.bo_instr, len(self.insts), self.bo_in, self.bo_out)
        run.wait()
        elapsed = time.perf_counter_ns() - start

        # Read output
        self.bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        output = self.bo_out.read(self.buffer_size, 0).tobytes()

        return elapsed / 1e6, output  # Return time in ms

    def benchmark_scale(self, total_elements, warmup=3, iterations=10):
        """
        Benchmark processing total_elements by running multiple kernel invocations.
        Each invocation processes 1024 elements.
        """
        num_chunks = total_elements // self.base_elements

        results = {
            'total_elements': total_elements,
            'num_chunks': num_chunks,
            'times_per_run': [],
            'total_times': [],
            'times_per_chunk': [],
            'times_per_element': [],
        }

        # Generate test data for all chunks
        all_input_floats = np.random.randn(total_elements).astype(np.float32)

        # Prepare input chunks as BF16
        input_chunks = []
        for i in range(num_chunks):
            chunk_start = i * self.base_elements
            chunk_end = chunk_start + self.base_elements
            chunk = all_input_floats[chunk_start:chunk_end]
            input_chunks.append(float_to_bf16(chunk))

        # Warmup
        for _ in range(warmup):
            for bf16_chunk in input_chunks:
                self.run_single_iteration(bf16_chunk)

        # Benchmark runs
        for run_idx in range(iterations):
            chunk_times = []

            total_start = time.perf_counter_ns()

            for bf16_chunk in input_chunks:
                chunk_time, _ = self.run_single_iteration(bf16_chunk)
                chunk_times.append(chunk_time)

            total_elapsed = (time.perf_counter_ns() - total_start) / 1e6  # ms

            results['times_per_run'].append(chunk_times)
            results['total_times'].append(total_elapsed)

        # Calculate statistics
        avg_total_time = np.mean(results['total_times'])
        min_total_time = np.min(results['total_times'])

        avg_chunk_times = [np.mean([run[i] for run in results['times_per_run']])
                          for i in range(num_chunks)]

        results['avg_total_time_ms'] = avg_total_time
        results['min_total_time_ms'] = min_total_time
        results['avg_chunk_time_ms'] = np.mean(avg_chunk_times)
        results['std_chunk_time_ms'] = np.std(avg_chunk_times)
        results['avg_time_per_element_us'] = (avg_total_time * 1000) / total_elements  # us
        results['avg_time_per_1024_ms'] = avg_total_time / num_chunks

        # First vs last chunk (check for degradation)
        if num_chunks > 1:
            first_avg = np.mean([run[0] for run in results['times_per_run']])
            last_avg = np.mean([run[-1] for run in results['times_per_run']])
            results['first_chunk_ms'] = first_avg
            results['last_chunk_ms'] = last_avg
            results['degradation_ratio'] = last_avg / first_avg

        return results

# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    print("=" * 80)
    print("NPU KERNEL SCALING BENCHMARK")
    print("Finding Memory Bandwidth Limits on AMD Phoenix NPU")
    print("=" * 80)
    print()

    base_dir = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1"

    xclbin_path = f"{base_dir}/build_softmax_bf16/softmax_bf16.xclbin"
    insts_path = f"{base_dir}/build_softmax_bf16/insts.bin"

    # Verify files exist
    if not os.path.exists(xclbin_path):
        print(f"ERROR: XCLBIN not found: {xclbin_path}")
        return 1
    if not os.path.exists(insts_path):
        print(f"ERROR: Instructions not found: {insts_path}")
        return 1

    # Test sizes
    test_sizes = [
        1024,    # Baseline (1 iteration)
        4096,    # 4 iterations
        8192,    # 8 iterations
        16384,   # 16 iterations
        32768,   # 32 iterations
    ]

    warmup_iterations = 5
    benchmark_iterations = 20

    try:
        print("Initializing NPU device...")
        device = xrt.device(0)
        print("Device initialized successfully")
        print()

        # Create benchmark instance
        benchmark = ScalingBenchmark(device, xclbin_path, insts_path)

        print(f"Kernel: Softmax BF16")
        print(f"Base tile size: 1024 elements (2048 bytes)")
        print(f"Warmup iterations: {warmup_iterations}")
        print(f"Benchmark iterations: {benchmark_iterations}")
        print()

        # Run benchmarks
        results = []

        print("-" * 80)
        print("Running Scaling Tests...")
        print("-" * 80)
        print()

        for size in test_sizes:
            print(f"Testing {size} elements ({size // 1024} chunks)...", end=" ")
            result = benchmark.benchmark_scale(
                size,
                warmup=warmup_iterations,
                iterations=benchmark_iterations
            )
            results.append(result)
            print(f"Done - {result['avg_total_time_ms']:.3f} ms")

        print()

        # =================================================================
        # RESULTS TABLE
        # =================================================================
        print("=" * 80)
        print("SCALING RESULTS")
        print("=" * 80)
        print()

        # Table header
        print(f"{'Elements':<10} {'Chunks':<8} {'Total (ms)':<12} {'Per 1024 (ms)':<15} {'Per Element (us)':<18} {'Overhead Ratio'}")
        print("-" * 80)

        # Baseline for comparison
        baseline_per_1024 = results[0]['avg_time_per_1024_ms']

        for r in results:
            overhead_ratio = r['avg_time_per_1024_ms'] / baseline_per_1024
            print(f"{r['total_elements']:<10} {r['num_chunks']:<8} "
                  f"{r['avg_total_time_ms']:<12.3f} {r['avg_time_per_1024_ms']:<15.4f} "
                  f"{r['avg_time_per_element_us']:<18.4f} {overhead_ratio:.3f}x")

        print()

        # =================================================================
        # DEGRADATION ANALYSIS
        # =================================================================
        print("=" * 80)
        print("DEGRADATION ANALYSIS")
        print("=" * 80)
        print()

        print("Per-Chunk Time Degradation (First vs Last chunk):")
        print("-" * 60)
        print(f"{'Elements':<10} {'First (ms)':<12} {'Last (ms)':<12} {'Degradation'}")
        print("-" * 60)

        for r in results:
            if 'degradation_ratio' in r:
                status = ""
                if r['degradation_ratio'] > 1.1:
                    status = " [DEGRADING]"
                elif r['degradation_ratio'] < 0.9:
                    status = " [IMPROVING]"
                else:
                    status = " [STABLE]"

                print(f"{r['total_elements']:<10} {r['first_chunk_ms']:<12.4f} "
                      f"{r['last_chunk_ms']:<12.4f} {r['degradation_ratio']:.3f}x{status}")
            else:
                print(f"{r['total_elements']:<10} N/A (single chunk)")

        print()

        # =================================================================
        # SCALING CURVE VISUALIZATION (ASCII)
        # =================================================================
        print("=" * 80)
        print("SCALING CURVE (Time per 1024 elements)")
        print("=" * 80)
        print()

        # Find range for normalization
        times = [r['avg_time_per_1024_ms'] for r in results]
        min_time = min(times)
        max_time = max(times)
        time_range = max_time - min_time if max_time > min_time else 0.01

        chart_width = 50

        for r in results:
            normalized = (r['avg_time_per_1024_ms'] - min_time) / time_range
            bar_len = int(normalized * (chart_width - 10)) + 1
            bar = "#" * bar_len
            print(f"{r['total_elements']:>6}: {bar} {r['avg_time_per_1024_ms']:.4f} ms")

        print()

        # =================================================================
        # EXTRAPOLATION TO WHISPER
        # =================================================================
        print("=" * 80)
        print("EXTRAPOLATION TO WHISPER OPERATIONS")
        print("=" * 80)
        print()

        # Get scaling trend
        time_per_1024_baseline = results[0]['avg_time_per_1024_ms']
        time_per_1024_largest = results[-1]['avg_time_per_1024_ms']
        overhead_growth = time_per_1024_largest / time_per_1024_baseline

        print("Whisper Attention Dimensions:")
        print("-" * 60)

        # Typical Whisper dimensions
        whisper_sizes = [
            ("Whisper Tiny/Base attention (80x80)", 6400),
            ("Whisper Small attention (80x80)", 6400),
            ("Whisper Medium attention (80x80)", 6400),
            ("Whisper Large attention (80x80)", 6400),
            ("Whisper Base hidden (512)", 512),
            ("Whisper Medium hidden (1024)", 1024),
            ("Whisper Large hidden (1280)", 1280),
            ("Attention matrix (80x80)", 6400),
            ("Attention per head (80)", 80),
            ("Full attention flat (384x80)", 30720),
            ("Large batch (64x1024)", 65536),
        ]

        print(f"{'Operation':<40} {'Elements':<12} {'Est. Time (ms)':<15} {'Est. Chunks'}")
        print("-" * 80)

        for name, elements in whisper_sizes:
            num_chunks = max(1, elements // 1024)
            # Extrapolate based on observed scaling
            if elements <= 1024:
                est_time = time_per_1024_baseline * (elements / 1024)
            else:
                # Apply overhead growth factor
                est_time = time_per_1024_baseline * num_chunks * (1 + (overhead_growth - 1) * 0.5)

            print(f"{name:<40} {elements:<12} {est_time:<15.3f} {num_chunks}")

        print()

        # =================================================================
        # BOTTLENECK IDENTIFICATION
        # =================================================================
        print("=" * 80)
        print("BOTTLENECK IDENTIFICATION")
        print("=" * 80)
        print()

        # Calculate overhead per iteration
        if len(results) >= 2:
            size_1 = results[0]['total_elements']
            size_2 = results[-1]['total_elements']
            time_1 = results[0]['avg_time_per_1024_ms']
            time_2 = results[-1]['avg_time_per_1024_ms']

            # Linear regression to find constant overhead
            # time = base_time + overhead_per_iteration * num_iterations

            # Calculate per-iteration overhead
            overhead_increase = time_2 - time_1
            iterations_increase = (size_2 / size_1) - 1

            print("Overhead Analysis:")
            print(f"  Baseline (1024 elements):     {time_1:.4f} ms per 1024")
            print(f"  At {size_2} elements:          {time_2:.4f} ms per 1024")
            print(f"  Overhead growth:              {overhead_growth:.3f}x")
            print()

            if overhead_growth > 1.2:
                print("WARNING: Significant overhead detected with scaling!")
                print("  Likely causes:")
                print("  - Kernel launch overhead accumulating")
                print("  - DMA transfer setup time per chunk")
                print("  - Memory bandwidth saturation")
                print("  - Cache thrashing")
            elif overhead_growth > 1.05:
                print("NOTICE: Minor overhead increase with scaling")
                print("  Performance is still reasonable but not perfectly linear")
            else:
                print("EXCELLENT: Near-linear scaling observed!")
                print("  The kernel efficiently handles larger workloads")

        print()

        # =================================================================
        # THROUGHPUT ANALYSIS
        # =================================================================
        print("=" * 80)
        print("THROUGHPUT ANALYSIS")
        print("=" * 80)
        print()

        print(f"{'Elements':<10} {'Throughput (M/s)':<18} {'Bandwidth (MB/s)':<18}")
        print("-" * 60)

        for r in results:
            throughput_mps = r['total_elements'] / (r['avg_total_time_ms'] / 1000) / 1e6
            # BF16 = 2 bytes in + 2 bytes out = 4 bytes total per element
            bandwidth_mbps = throughput_mps * 4  # MB/s
            print(f"{r['total_elements']:<10} {throughput_mps:<18.2f} {bandwidth_mbps:<18.2f}")

        print()

        # Peak bandwidth check
        max_throughput = max(r['total_elements'] / (r['avg_total_time_ms'] / 1000) / 1e6
                            for r in results)

        # AMD Phoenix NPU has ~32 GB/s memory bandwidth
        npu_bandwidth_gbps = 32.0
        achieved_bandwidth_gbps = max_throughput * 4 / 1000
        bandwidth_utilization = achieved_bandwidth_gbps / npu_bandwidth_gbps * 100

        print(f"Peak throughput achieved: {max_throughput:.2f} M elements/s")
        print(f"Peak bandwidth achieved: {achieved_bandwidth_gbps:.2f} GB/s")
        print(f"NPU theoretical bandwidth: {npu_bandwidth_gbps} GB/s")
        print(f"Bandwidth utilization: {bandwidth_utilization:.1f}%")
        print()

        # =================================================================
        # RECOMMENDATIONS
        # =================================================================
        print("=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        print()

        if overhead_growth > 1.2:
            print("1. CRITICAL: Consider kernel fusion to reduce invocations")
            print("   - Combine multiple softmax operations into single kernel")
            print("   - Use persistent on-device buffers")
            print()
            print("2. IMPORTANT: Batch processing is inefficient")
            print("   - Better to process larger tiles natively")
            print("   - May need custom kernel for larger sizes")
        else:
            print("1. Scaling is acceptable for iterative processing")
            print("   - Current approach viable for real workloads")
            print()
            print("2. For Whisper attention (6400 elements):")
            est_attention_time = time_per_1024_baseline * 7 * overhead_growth
            print(f"   - Estimated time: {est_attention_time:.3f} ms")
            print(f"   - Realtime factor for 30s audio: {30000 / est_attention_time:.0f}x")

        print()

        if bandwidth_utilization < 10:
            print("3. NOTICE: Low bandwidth utilization ({:.1f}%)".format(bandwidth_utilization))
            print("   - Compute-bound, not memory-bound")
            print("   - Softmax has high arithmetic intensity")
        elif bandwidth_utilization > 80:
            print("3. WARNING: Approaching memory bandwidth limit")
            print("   - Memory-bound operation")
            print("   - Larger scales will see significant degradation")

        print()

        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        print("=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print()

        print(f"Baseline performance (1024 elements): {results[0]['avg_total_time_ms']:.3f} ms")
        print(f"Largest test ({results[-1]['total_elements']} elements): {results[-1]['avg_total_time_ms']:.3f} ms")
        print(f"Scaling efficiency: {(1/overhead_growth * 100):.1f}%")
        print()

        # Verdict
        if overhead_growth < 1.1:
            verdict = "EXCELLENT"
            desc = "Near-linear scaling, suitable for production workloads"
        elif overhead_growth < 1.3:
            verdict = "GOOD"
            desc = "Acceptable scaling with minor overhead"
        elif overhead_growth < 1.5:
            verdict = "ACCEPTABLE"
            desc = "Noticeable overhead, consider optimization"
        else:
            verdict = "POOR"
            desc = "Significant overhead, kernel fusion recommended"

        print(f"Verdict: {verdict}")
        print(f"         {desc}")
        print()

        print("=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
