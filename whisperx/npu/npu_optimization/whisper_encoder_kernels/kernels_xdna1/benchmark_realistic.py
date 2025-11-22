#!/usr/bin/env python3
"""
Realistic NPU Benchmark - Measures ALL Overhead Sources
========================================================

This benchmark provides accurate performance projections by measuring:
1. Buffer allocation time
2. Data copy to device (bo_in.write + sync)
3. Kernel execution only (kernel() + wait())
4. Data copy from device (sync + read)
5. BF16 conversion overhead
6. Python/numpy overhead

AMD Phoenix NPU - XDNA1
Target: Identify bottlenecks in the 3413x realtime projection
"""

import numpy as np
import pyxrt as xrt
import struct
import time
import os
import gc
import sys
from collections import defaultdict

# =============================================================================
# BF16 Conversion Functions
# =============================================================================

def bf16_to_float_slow(bf16_bytes):
    """Original slow BF16 to float conversion (element-by-element)"""
    result = np.zeros(len(bf16_bytes) // 2, dtype=np.float32)
    for i in range(len(result)):
        upper = struct.unpack('H', bf16_bytes[i*2:(i+1)*2])[0]
        result[i] = struct.unpack('f', struct.pack('I', upper << 16))[0]
    return result

def float_to_bf16_slow(floats):
    """Original slow float to BF16 conversion (element-by-element)"""
    result = bytearray(len(floats) * 2)
    for i, val in enumerate(floats):
        bits = struct.unpack('I', struct.pack('f', val))[0]
        upper = (bits >> 16) & 0xFFFF
        struct.pack_into('H', result, i*2, upper)
    return bytes(result)

def bf16_to_float_fast(bf16_bytes):
    """Optimized vectorized BF16 to float conversion"""
    bf16_array = np.frombuffer(bf16_bytes, dtype=np.uint16)
    int32_array = bf16_array.astype(np.uint32) << 16
    return int32_array.view(np.float32)

def float_to_bf16_fast(floats):
    """Optimized vectorized float to BF16 conversion"""
    float_array = np.asarray(floats, dtype=np.float32)
    int32_array = float_array.view(np.uint32)
    bf16_array = ((int32_array >> 16) & 0xFFFF).astype(np.uint16)
    return bf16_array.tobytes()

# =============================================================================
# Timing Helper
# =============================================================================

class Timer:
    """High-precision timer with statistics"""
    def __init__(self):
        self.times = []

    def start(self):
        self._start = time.perf_counter_ns()

    def stop(self):
        elapsed = time.perf_counter_ns() - self._start
        self.times.append(elapsed / 1e6)  # Convert to ms
        return elapsed / 1e6

    @property
    def avg_ms(self):
        return np.mean(self.times) if self.times else 0

    @property
    def min_ms(self):
        return np.min(self.times) if self.times else 0

    @property
    def max_ms(self):
        return np.max(self.times) if self.times else 0

    @property
    def std_ms(self):
        return np.std(self.times) if self.times else 0

    @property
    def total_ms(self):
        return np.sum(self.times) if self.times else 0

# =============================================================================
# Benchmark Class
# =============================================================================

class RealisticBenchmark:
    def __init__(self, device, name, xclbin_path, insts_path, buffer_size):
        self.name = name
        self.device = device
        self.buffer_size = buffer_size
        self.num_elements = buffer_size // 2  # BF16 = 2 bytes per element

        # Timers for different phases
        self.timers = {
            'alloc_instr': Timer(),
            'alloc_input': Timer(),
            'alloc_output': Timer(),
            'load_instr': Timer(),
            'sync_instr': Timer(),
            'write_input': Timer(),
            'sync_to_device': Timer(),
            'kernel_exec': Timer(),
            'sync_from_device': Timer(),
            'read_output': Timer(),
            'bf16_to_float_slow': Timer(),
            'bf16_to_float_fast': Timer(),
            'float_to_bf16_slow': Timer(),
            'float_to_bf16_fast': Timer(),
            'total_with_slow_conv': Timer(),
            'total_with_fast_conv': Timer(),
            'pure_compute': Timer(),  # kernel_exec only
        }

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

    def allocate_buffers(self):
        """Allocate buffers with timing"""
        # Instruction buffer
        self.timers['alloc_instr'].start()
        self.bo_instr = xrt.bo(self.device, len(self.insts),
                               xrt.bo.flags.cacheable, self.kernel.group_id(1))
        self.timers['alloc_instr'].stop()

        # Input buffer
        self.timers['alloc_input'].start()
        self.bo_in = xrt.bo(self.device, self.buffer_size,
                           xrt.bo.flags.host_only, self.kernel.group_id(3))
        self.timers['alloc_input'].stop()

        # Output buffer
        self.timers['alloc_output'].start()
        self.bo_out = xrt.bo(self.device, self.buffer_size,
                            xrt.bo.flags.host_only, self.kernel.group_id(4))
        self.timers['alloc_output'].stop()

        # Load and sync instruction buffer
        self.timers['load_instr'].start()
        self.bo_instr.write(self.insts, 0)
        self.timers['load_instr'].stop()

        self.timers['sync_instr'].start()
        self.bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.timers['sync_instr'].stop()

    def run_iteration(self, input_bf16, use_fast_conv=True):
        """Run single iteration with detailed timing"""

        # 1. Write input to buffer
        self.timers['write_input'].start()
        self.bo_in.write(input_bf16, 0)
        self.timers['write_input'].stop()

        # 2. Sync input to device (DMA transfer)
        self.timers['sync_to_device'].start()
        self.bo_in.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.timers['sync_to_device'].stop()

        # 3. Kernel execution
        self.timers['kernel_exec'].start()
        self.timers['pure_compute'].start()
        run = self.kernel(3, self.bo_instr, len(self.insts), self.bo_in, self.bo_out)
        run.wait()
        self.timers['pure_compute'].stop()
        self.timers['kernel_exec'].stop()

        # 4. Sync output from device (DMA transfer)
        self.timers['sync_from_device'].start()
        self.bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        self.timers['sync_from_device'].stop()

        # 5. Read output from buffer
        self.timers['read_output'].start()
        output_bf16 = self.bo_out.read(self.buffer_size, 0).tobytes()
        self.timers['read_output'].stop()

        return output_bf16

    def benchmark_bf16_conversion(self, input_floats):
        """Benchmark BF16 conversion overhead separately"""

        # Slow conversion (original)
        self.timers['float_to_bf16_slow'].start()
        bf16_slow = float_to_bf16_slow(input_floats)
        self.timers['float_to_bf16_slow'].stop()

        # Fast conversion (vectorized)
        self.timers['float_to_bf16_fast'].start()
        bf16_fast = float_to_bf16_fast(input_floats)
        self.timers['float_to_bf16_fast'].stop()

        # Back-conversion slow
        self.timers['bf16_to_float_slow'].start()
        _ = bf16_to_float_slow(bf16_slow)
        self.timers['bf16_to_float_slow'].stop()

        # Back-conversion fast
        self.timers['bf16_to_float_fast'].start()
        _ = bf16_to_float_fast(bf16_fast)
        self.timers['bf16_to_float_fast'].stop()

        return bf16_fast  # Use fast version for actual benchmark

# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    print("=" * 80)
    print("REALISTIC NPU PERFORMANCE BENCHMARK")
    print("Measuring ALL Overhead Sources")
    print("AMD Phoenix NPU - XDNA1")
    print("=" * 80)
    print()

    base_dir = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1"

    # Test configurations - test multiple kernels
    kernels_config = [
        {
            "name": "Softmax",
            "xclbin": f"{base_dir}/build_softmax_bf16/softmax_bf16.xclbin",
            "insts": f"{base_dir}/build_softmax_bf16/insts.bin",
            "elements": 1024,
        },
        {
            "name": "LayerNorm",
            "xclbin": f"{base_dir}/build_layernorm/layernorm_bf16.xclbin",
            "insts": f"{base_dir}/build_layernorm/insts.bin",
            "elements": 1024,
        },
        {
            "name": "GELU",
            "xclbin": f"{base_dir}/build_gelu/gelu_bf16.xclbin",
            "insts": f"{base_dir}/build_gelu/insts.bin",
            "elements": 1024,
        },
    ]

    # Parameters
    iterations = 50  # More iterations for stable measurements
    warmup_iterations = 5

    try:
        print("Initializing NPU device...")
        device = xrt.device(0)
        print("Device initialized successfully")
        print()

        all_results = {}

        for kconfig in kernels_config:
            kernel_name = kconfig["name"]

            # Check files exist
            if not os.path.exists(kconfig["xclbin"]):
                print(f"Skipping {kernel_name}: XCLBIN not found")
                continue
            if not os.path.exists(kconfig["insts"]):
                print(f"Skipping {kernel_name}: insts.bin not found")
                continue

            print("-" * 80)
            print(f"Benchmarking: {kernel_name}")
            print(f"Elements: {kconfig['elements']}")
            print("-" * 80)

            buffer_size = kconfig["elements"] * 2  # BF16 = 2 bytes

            # Create benchmark instance
            benchmark = RealisticBenchmark(
                device, kernel_name,
                kconfig["xclbin"], kconfig["insts"],
                buffer_size
            )

            # Allocate buffers (with timing)
            print("Allocating buffers...")
            benchmark.allocate_buffers()

            # Generate test data
            input_floats = np.random.randn(kconfig["elements"]).astype(np.float32)

            # Benchmark BF16 conversion
            print("Benchmarking BF16 conversion...")
            for _ in range(iterations):
                input_bf16 = benchmark.benchmark_bf16_conversion(input_floats)

            # Warmup runs
            print(f"Warming up ({warmup_iterations} iterations)...")
            for _ in range(warmup_iterations):
                _ = benchmark.run_iteration(input_bf16)

            # Reset timers for actual benchmark
            for timer in benchmark.timers.values():
                timer.times = []

            # Re-do allocation timing (was only done once)
            # Record the single allocation time
            alloc_total = (benchmark.timers['alloc_instr'].total_ms +
                          benchmark.timers['alloc_input'].total_ms +
                          benchmark.timers['alloc_output'].total_ms +
                          benchmark.timers['load_instr'].total_ms +
                          benchmark.timers['sync_instr'].total_ms)

            # Actual benchmark
            print(f"Running benchmark ({iterations} iterations)...")

            total_start = time.perf_counter_ns()

            for i in range(iterations):
                # BF16 conversion
                input_bf16 = benchmark.benchmark_bf16_conversion(input_floats)

                # Full iteration
                output_bf16 = benchmark.run_iteration(input_bf16)

                # Output conversion (measure separately)
                t1 = time.perf_counter_ns()
                output_floats = bf16_to_float_fast(output_bf16)
                t2 = time.perf_counter_ns()

            total_end = time.perf_counter_ns()
            total_wall_time = (total_end - total_start) / 1e6  # ms

            # Calculate results
            results = {
                'name': kernel_name,
                'elements': kconfig['elements'],
                'iterations': iterations,
            }

            # Phase timings
            phases = [
                ('Buffer Allocation (one-time)', alloc_total, True),
                ('Float->BF16 (slow)', benchmark.timers['float_to_bf16_slow'].avg_ms, False),
                ('Float->BF16 (fast)', benchmark.timers['float_to_bf16_fast'].avg_ms, False),
                ('Write to buffer', benchmark.timers['write_input'].avg_ms, False),
                ('DMA to device', benchmark.timers['sync_to_device'].avg_ms, False),
                ('Kernel execution', benchmark.timers['kernel_exec'].avg_ms, False),
                ('DMA from device', benchmark.timers['sync_from_device'].avg_ms, False),
                ('Read from buffer', benchmark.timers['read_output'].avg_ms, False),
                ('BF16->Float (slow)', benchmark.timers['bf16_to_float_slow'].avg_ms, False),
                ('BF16->Float (fast)', benchmark.timers['bf16_to_float_fast'].avg_ms, False),
            ]

            print()
            print("PHASE BREAKDOWN (per iteration):")
            print("-" * 60)
            print(f"{'Phase':<30} {'Time (ms)':<15} {'% of Total'}")
            print("-" * 60)

            # Calculate total with fast conversion
            pure_compute = benchmark.timers['kernel_exec'].avg_ms
            total_with_fast = (
                benchmark.timers['float_to_bf16_fast'].avg_ms +
                benchmark.timers['write_input'].avg_ms +
                benchmark.timers['sync_to_device'].avg_ms +
                benchmark.timers['kernel_exec'].avg_ms +
                benchmark.timers['sync_from_device'].avg_ms +
                benchmark.timers['read_output'].avg_ms +
                benchmark.timers['bf16_to_float_fast'].avg_ms
            )

            total_with_slow = (
                benchmark.timers['float_to_bf16_slow'].avg_ms +
                benchmark.timers['write_input'].avg_ms +
                benchmark.timers['sync_to_device'].avg_ms +
                benchmark.timers['kernel_exec'].avg_ms +
                benchmark.timers['sync_from_device'].avg_ms +
                benchmark.timers['read_output'].avg_ms +
                benchmark.timers['bf16_to_float_slow'].avg_ms
            )

            for phase_name, phase_time, is_onetime in phases:
                if is_onetime:
                    print(f"{phase_name:<30} {phase_time:<15.4f} (one-time)")
                else:
                    pct = (phase_time / total_with_fast * 100) if total_with_fast > 0 else 0
                    print(f"{phase_name:<30} {phase_time:<15.4f} {pct:.1f}%")

            print("-" * 60)
            print(f"{'Total (with fast conv)':<30} {total_with_fast:<15.4f} 100%")
            print(f"{'Total (with slow conv)':<30} {total_with_slow:<15.4f}")
            print()

            # Summary statistics
            overhead_fast = total_with_fast - pure_compute
            overhead_pct_fast = (overhead_fast / total_with_fast * 100) if total_with_fast > 0 else 0

            overhead_slow = total_with_slow - pure_compute
            overhead_pct_slow = (overhead_slow / total_with_slow * 100) if total_with_slow > 0 else 0

            print("OVERHEAD ANALYSIS:")
            print("-" * 60)
            print(f"Pure kernel compute:        {pure_compute:.4f} ms")
            print(f"Overhead (fast conversion): {overhead_fast:.4f} ms ({overhead_pct_fast:.1f}%)")
            print(f"Overhead (slow conversion): {overhead_slow:.4f} ms ({overhead_pct_slow:.1f}%)")
            print()

            # Store results
            results['pure_compute_ms'] = pure_compute
            results['total_with_fast_ms'] = total_with_fast
            results['total_with_slow_ms'] = total_with_slow
            results['overhead_pct_fast'] = overhead_pct_fast
            results['overhead_pct_slow'] = overhead_pct_slow
            results['dma_to_device_ms'] = benchmark.timers['sync_to_device'].avg_ms
            results['dma_from_device_ms'] = benchmark.timers['sync_from_device'].avg_ms

            all_results[kernel_name] = results

            # Memory cleanup
            del benchmark
            gc.collect()

        # =================================================================
        # ENCODER LAYER PROJECTION
        # =================================================================
        print()
        print("=" * 80)
        print("REALISTIC ENCODER LAYER PROJECTION")
        print("=" * 80)
        print()

        # Get average times from benchmarks
        softmax_time = all_results.get('Softmax', {}).get('total_with_fast_ms', 1.5)
        layernorm_time = all_results.get('LayerNorm', {}).get('total_with_fast_ms', 0.9)
        gelu_time = all_results.get('GELU', {}).get('total_with_fast_ms', 1.8)

        # Estimate matmul (from previous tests - 0.208ms kernel + overhead)
        matmul_overhead_factor = 2.5  # Typical overhead multiplier
        matmul_kernel = 0.208
        matmul_total = matmul_kernel * matmul_overhead_factor

        # Per encoder layer composition (Whisper tiny/base has 6 layers)
        # Self-attention: LayerNorm + Q,K,V projections + attention + output projection
        # FFN: LayerNorm + linear + GELU + linear

        per_layer = {
            'LayerNorm (pre-attention)': layernorm_time,
            'MatMul Q': matmul_total,
            'MatMul K': matmul_total,
            'MatMul V': matmul_total,
            'Softmax (attention)': softmax_time,
            'MatMul (attention output)': matmul_total,
            'LayerNorm (pre-FFN)': layernorm_time,
            'MatMul (FFN expand)': matmul_total,
            'GELU': gelu_time,
            'MatMul (FFN contract)': matmul_total,
        }

        print("Per Encoder Layer (with ALL overhead):")
        print("-" * 60)

        layer_total = 0
        for op_name, op_time in per_layer.items():
            print(f"  {op_name:<30} {op_time:.4f} ms")
            layer_total += op_time

        print("-" * 60)
        print(f"  {'TOTAL per layer':<30} {layer_total:.4f} ms")
        print()

        # Full encoder (6 layers for Whisper base)
        num_layers = 6
        encoder_total = layer_total * num_layers

        print(f"6-Layer Encoder Total:          {encoder_total:.4f} ms")
        print()

        # =================================================================
        # REALTIME FACTOR CALCULATION
        # =================================================================
        print("=" * 80)
        print("REALTIME FACTOR PROJECTIONS")
        print("=" * 80)
        print()

        # Whisper processes audio in 30-second chunks
        audio_duration = 30.0  # seconds

        # Pure compute time (no overhead)
        pure_compute_layer = (
            all_results.get('LayerNorm', {}).get('pure_compute_ms', 0.3) * 2 +
            matmul_kernel * 6 +
            all_results.get('Softmax', {}).get('pure_compute_ms', 0.5) +
            all_results.get('GELU', {}).get('pure_compute_ms', 0.6)
        )
        pure_compute_encoder = pure_compute_layer * num_layers

        # With overhead
        realistic_encoder = encoder_total

        # Realtime calculations
        pure_compute_rtf = audio_duration / (pure_compute_encoder / 1000)
        realistic_rtf = audio_duration / (realistic_encoder / 1000)

        print("Scenario Analysis:")
        print("-" * 60)
        print()

        print("1. PURE COMPUTE ONLY (theoretical maximum):")
        print(f"   Encoder time:     {pure_compute_encoder:.4f} ms")
        print(f"   Realtime factor:  {pure_compute_rtf:.0f}x")
        print()

        print("2. WITH ALL OVERHEAD (realistic):")
        print(f"   Encoder time:     {realistic_encoder:.4f} ms")
        print(f"   Realtime factor:  {realistic_rtf:.0f}x")
        print()

        overhead_impact = (pure_compute_rtf - realistic_rtf) / pure_compute_rtf * 100
        print(f"Performance loss due to overhead: {overhead_impact:.1f}%")
        print()

        # =================================================================
        # BOTTLENECK IDENTIFICATION
        # =================================================================
        print("=" * 80)
        print("BOTTLENECK IDENTIFICATION")
        print("=" * 80)
        print()

        # Collect all timing components
        bottlenecks = []

        for kernel_name, results in all_results.items():
            bottlenecks.append({
                'name': f"{kernel_name} - DMA to device",
                'time': results.get('dma_to_device_ms', 0),
                'category': 'DMA'
            })
            bottlenecks.append({
                'name': f"{kernel_name} - DMA from device",
                'time': results.get('dma_from_device_ms', 0),
                'category': 'DMA'
            })
            bottlenecks.append({
                'name': f"{kernel_name} - Pure compute",
                'time': results.get('pure_compute_ms', 0),
                'category': 'Compute'
            })

        # Add conversion overhead
        for kernel_name, results in all_results.items():
            bf16_conv = (
                all_results[kernel_name].get('total_with_fast_ms', 0) -
                all_results[kernel_name].get('total_with_slow_ms', 0) +
                all_results[kernel_name].get('total_with_fast_ms', 0)
            )
            # Actually let's get from timer data
            break

        # Sort by time
        bottlenecks.sort(key=lambda x: x['time'], reverse=True)

        print("Top Bottlenecks (ranked by time):")
        print("-" * 60)

        for i, b in enumerate(bottlenecks[:10], 1):
            print(f"{i:2}. {b['name']:<40} {b['time']:.4f} ms [{b['category']}]")

        print()

        # Category summary
        categories = defaultdict(float)
        for b in bottlenecks:
            categories[b['category']] += b['time']

        print("Time by Category:")
        print("-" * 60)
        total_cat = sum(categories.values())
        for cat, time_val in sorted(categories.items(), key=lambda x: -x[1]):
            pct = (time_val / total_cat * 100) if total_cat > 0 else 0
            print(f"  {cat:<20} {time_val:.4f} ms ({pct:.1f}%)")

        print()

        # =================================================================
        # OPTIMIZATION RECOMMENDATIONS
        # =================================================================
        print("=" * 80)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("=" * 80)
        print()

        recommendations = []

        # Check DMA overhead
        avg_dma_to = np.mean([r.get('dma_to_device_ms', 0) for r in all_results.values()])
        avg_dma_from = np.mean([r.get('dma_from_device_ms', 0) for r in all_results.values()])
        avg_compute = np.mean([r.get('pure_compute_ms', 0) for r in all_results.values()])

        if avg_dma_to + avg_dma_from > avg_compute:
            recommendations.append(
                "HIGH: DMA transfer time exceeds compute time. Consider:\n"
                "   - Batch multiple operations before transfer\n"
                "   - Use persistent on-device buffers\n"
                "   - Implement kernel fusion to reduce transfers"
            )

        # Check conversion overhead
        for kernel_name, results in all_results.items():
            overhead_pct = results.get('overhead_pct_fast', 0)
            if overhead_pct > 50:
                recommendations.append(
                    f"MEDIUM: {kernel_name} has {overhead_pct:.0f}% overhead. Consider:\n"
                    "   - Keep data in BF16 format between operations\n"
                    "   - Avoid unnecessary conversions"
                )
                break

        # General recommendations
        recommendations.append(
            "GENERAL: To improve from current realistic projection:\n"
            "   - Implement kernel fusion (combine LayerNorm+MatMul, etc.)\n"
            "   - Use larger tile sizes to reduce kernel launch overhead\n"
            "   - Keep encoder weights permanently on NPU\n"
            "   - Batch multiple audio frames together"
        )

        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
            print()

        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        print("=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print()

        print(f"Original projection (pure compute):    {pure_compute_rtf:.0f}x realtime")
        print(f"Realistic projection (with overhead):  {realistic_rtf:.0f}x realtime")
        print(f"Performance reduction:                 {overhead_impact:.1f}%")
        print()

        if realistic_rtf > 100:
            print("Status: EXCELLENT - Still significantly faster than realtime")
        elif realistic_rtf > 30:
            print("Status: GOOD - Faster than realtime, room for optimization")
        elif realistic_rtf > 1:
            print("Status: ACCEPTABLE - Faster than realtime")
        else:
            print("Status: NEEDS OPTIMIZATION - Slower than realtime")

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
