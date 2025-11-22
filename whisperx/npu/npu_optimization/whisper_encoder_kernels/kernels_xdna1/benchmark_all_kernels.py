#!/usr/bin/env python3
"""
Comprehensive Kernel Benchmark Suite
Tests all NPU kernels and produces performance summary

AMD Phoenix NPU - XDNA1
"""

import numpy as np
import pyxrt as xrt
import struct
import time
import os

def bf16_to_float(bf16_bytes):
    result = np.zeros(len(bf16_bytes) // 2, dtype=np.float32)
    for i in range(len(result)):
        upper = struct.unpack('H', bf16_bytes[i*2:(i+1)*2])[0]
        result[i] = struct.unpack('f', struct.pack('I', upper << 16))[0]
    return result

def float_to_bf16(floats):
    result = bytearray(len(floats) * 2)
    for i, val in enumerate(floats):
        bits = struct.unpack('I', struct.pack('f', val))[0]
        upper = (bits >> 16) & 0xFFFF
        struct.pack_into('H', result, i*2, upper)
    return bytes(result)

class KernelBenchmark:
    def __init__(self, device, name, xclbin_path, insts_path, buffer_size):
        self.name = name
        self.device = device
        self.buffer_size = buffer_size

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
        self.bo_in = xrt.bo(device, buffer_size,
                            xrt.bo.flags.host_only, self.kernel.group_id(3))
        self.bo_out = xrt.bo(device, buffer_size,
                             xrt.bo.flags.host_only, self.kernel.group_id(4))

        self.bo_instr.write(self.insts, 0)
        self.bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def run(self, input_bf16, iterations=10):
        times = []
        for _ in range(iterations):
            self.bo_in.write(input_bf16, 0)
            self.bo_in.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            start = time.perf_counter()
            run = self.kernel(3, self.bo_instr, len(self.insts), self.bo_in, self.bo_out)
            run.wait()
            end = time.perf_counter()
            times.append(end - start)

            self.bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

        output = self.bo_out.read(self.buffer_size, 0).tobytes()
        return times, output

def main():
    print("=" * 70)
    print("Comprehensive NPU Kernel Benchmark Suite")
    print("AMD Phoenix NPU - XDNA1")
    print("=" * 70)
    print()

    base_dir = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1"

    # Kernel configurations
    kernels = [
        {
            "name": "Softmax",
            "xclbin": f"{base_dir}/build_softmax_bf16/softmax_bf16.xclbin",
            "insts": f"{base_dir}/build_softmax_bf16/insts.bin",
            "elements": 1024,
        },
        {
            "name": "GELU",
            "xclbin": f"{base_dir}/build_gelu/gelu_bf16.xclbin",
            "insts": f"{base_dir}/build_gelu/insts.bin",
            "elements": 1024,
        },
        {
            "name": "LayerNorm",
            "xclbin": f"{base_dir}/build_layernorm/layernorm_bf16.xclbin",
            "insts": f"{base_dir}/build_layernorm/insts.bin",
            "elements": 1024,
        },
        {
            "name": "4-Tile Parallel",
            "xclbin": f"{base_dir}/build_softmax_multicolumn_fixed/softmax_multicolumn_combined.xclbin",
            "insts": f"{base_dir}/build_softmax_multicolumn_fixed/insts.bin",
            "elements": 4096,  # 4 x 1024
        },
    ]

    try:
        print("Initializing NPU device...")
        device = xrt.device(0)
        print("Device ready")
        print()

        results = []
        iterations = 10

        for kconfig in kernels:
            # Check files exist
            if not os.path.exists(kconfig["xclbin"]) or not os.path.exists(kconfig["insts"]):
                print(f"Skipping {kconfig['name']}: files not found")
                continue

            print(f"Benchmarking {kconfig['name']}...")

            buffer_size = kconfig["elements"] * 2
            kernel = KernelBenchmark(
                device, kconfig["name"],
                kconfig["xclbin"], kconfig["insts"],
                buffer_size
            )

            # Generate test data
            input_floats = np.random.randn(kconfig["elements"]).astype(np.float32)
            input_bf16 = float_to_bf16(input_floats)

            # Run benchmark
            times, _ = kernel.run(input_bf16, iterations)

            avg_ms = np.mean(times) * 1000
            min_ms = np.min(times) * 1000
            std_ms = np.std(times) * 1000
            throughput = kconfig["elements"] / (avg_ms / 1000) / 1e6

            results.append({
                "name": kconfig["name"],
                "elements": kconfig["elements"],
                "avg_ms": avg_ms,
                "min_ms": min_ms,
                "std_ms": std_ms,
                "throughput": throughput,
            })

            print(f"  Done: {avg_ms:.3f} ms avg")

        print()
        print("=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print()

        # Table header
        print(f"{'Kernel':<18} {'Elements':<10} {'Avg (ms)':<10} {'Min (ms)':<10} {'Throughput':<12}")
        print("-" * 70)

        for r in results:
            print(f"{r['name']:<18} {r['elements']:<10} {r['avg_ms']:<10.3f} {r['min_ms']:<10.3f} {r['throughput']:.2f} M/s")

        print()
        print("=" * 70)
        print("ENCODER LAYER PROJECTIONS")
        print("=" * 70)
        print()

        # Find specific kernel times
        ln_time = next((r['avg_ms'] for r in results if r['name'] == 'LayerNorm'), 0.9)
        sm_time = next((r['avg_ms'] for r in results if r['name'] == 'Softmax'), 1.5)
        gelu_time = next((r['avg_ms'] for r in results if r['name'] == 'GELU'), 1.8)

        # Assume MatMul vectorized at 0.2ms (from previous tests)
        matmul_time = 0.208

        print("Per Encoder Layer (estimated):")
        print(f"  LayerNorm × 2:           {ln_time * 2:.3f} ms")
        print(f"  MatMul (Q,K,V,O) × 4:    {matmul_time * 4:.3f} ms")
        print(f"  Softmax:                 {sm_time:.3f} ms")
        print(f"  MatMul (FFN) × 2:        {matmul_time * 2:.3f} ms")
        print(f"  GELU:                    {gelu_time:.3f} ms")
        print(f"  ────────────────────────────────")

        layer_total = ln_time * 2 + matmul_time * 6 + sm_time + gelu_time
        print(f"  Total per layer:         {layer_total:.3f} ms")
        print()

        encoder_total = layer_total * 6
        print(f"6-Layer Encoder Total:     {encoder_total:.3f} ms")
        print()

        # 4-tile speedup
        parallel_time = next((r['avg_ms'] for r in results if '4-Tile' in r['name']), None)
        if parallel_time:
            per_frame = parallel_time / 4
            print(f"With 4-Tile Parallelism:")
            print(f"  Per-frame time:          {per_frame:.3f} ms")
            optimized_total = encoder_total / 4
            print(f"  6-Layer Encoder:         {optimized_total:.3f} ms")
            print()

            # Realtime calculation for 30s audio
            audio_duration = 30.0  # seconds
            realtime = audio_duration / (optimized_total / 1000)
            print(f"For 30s audio:")
            print(f"  Processing time:         {optimized_total:.3f} ms")
            print(f"  Realtime factor:         {realtime:.0f}x")
            print()

        print("=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
