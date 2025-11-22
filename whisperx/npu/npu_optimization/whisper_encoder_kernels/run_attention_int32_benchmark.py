#!/usr/bin/env python3
"""INT32 Attention Benchmark - Performance + Accuracy"""
import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import time

xclbin_path = Path("build_attention_int32/attention_64x64.xclbin")
insts_path = Path("build_attention_int32/insts.bin")

print("="*70)
print("INT32 Attention - Performance Benchmark")
print("="*70)
print()

# Load NPU
npu_device = xrt.device(0)
xclbin_obj = xrt.xclbin(str(xclbin_path))
uuid = xclbin_obj.get_uuid()
npu_device.register_xclbin(xclbin_obj)
hw_ctx = xrt.hw_context(npu_device, uuid)
kernel_obj = xrt.kernel(hw_ctx, "MLIR_AIE")

with open(insts_path, "rb") as f:
    insts = f.read()
n_insts = len(insts)

print(f"✅ NPU ready: {xclbin_path.name}")
print()

# Test data
np.random.seed(42)
Q = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
K = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
V = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
QKV_combined = np.concatenate([Q.flatten(), K.flatten(), V.flatten()])

# Reference
Q_f = torch.tensor(Q, dtype=torch.float32)
K_f = torch.tensor(K, dtype=torch.float32)
V_f = torch.tensor(V, dtype=torch.float32)
scores = torch.matmul(Q_f, K_f.T) / 8.0
attention_weights = F.softmax(scores, dim=-1)
reference_output = torch.matmul(attention_weights, V_f)

# Allocate buffers
instr_bo_obj = xrt.bo(npu_device, n_insts, xrt.bo.flags.cacheable, kernel_obj.group_id(1))
instr_bo_obj.write(insts, 0)
instr_bo_obj.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

input_bo_obj = xrt.bo(npu_device, QKV_combined.nbytes, xrt.bo.flags.host_only, kernel_obj.group_id(3))
input_bo_obj.write(QKV_combined.tobytes(), 0)
input_bo_obj.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, QKV_combined.nbytes, 0)

output_bo_obj = xrt.bo(npu_device, 4096, xrt.bo.flags.host_only, kernel_obj.group_id(4))

print("Running performance benchmark...")
print("  Warmup: 10 iterations")
print("  Benchmark: 100 iterations")
print()

# Warmup
for _ in range(10):
    run_obj = kernel_obj(3, instr_bo_obj, n_insts, input_bo_obj, output_bo_obj)
    run_obj.wait(1000)

# Benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    run_obj = kernel_obj(3, instr_bo_obj, n_insts, input_bo_obj, output_bo_obj)
    run_obj.wait(1000)
    end = time.perf_counter()
    times.append((end - start) * 1000)  # ms

# Read output
output_bo_obj.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 4096, 0)
npu_output = np.frombuffer(output_bo_obj.read(4096, 0), dtype=np.int8).reshape(64, 64)

# Calculate metrics
npu_flat = npu_output.flatten().astype(np.float32)
ref_flat = reference_output.numpy().flatten()
correlation = np.corrcoef(npu_flat, ref_flat)[0, 1]
diff = npu_flat - ref_flat
mae = np.mean(np.abs(diff))
rmse = np.sqrt(np.mean(diff**2))

avg_time = np.mean(times)
std_time = np.std(times)
min_time = np.min(times)
max_time = np.max(times)
p50_time = np.percentile(times, 50)
p95_time = np.percentile(times, 95)
p99_time = np.percentile(times, 99)

print("="*70)
print("PERFORMANCE RESULTS")
print("="*70)
print()
print(f"Latency Statistics (100 runs):")
print(f"  Average:        {avg_time:.3f} ms")
print(f"  Std Dev:        {std_time:.3f} ms")
print(f"  Min:            {min_time:.3f} ms")
print(f"  Max:            {max_time:.3f} ms")
print(f"  P50 (median):   {p50_time:.3f} ms")
print(f"  P95:            {p95_time:.3f} ms")
print(f"  P99:            {p99_time:.3f} ms")
print()
print(f"Throughput:")
print(f"  {1000/avg_time:.1f} tiles/second")
print(f"  {1000/avg_time * 64*64:.0f} elements/second")
print()
print("="*70)
print("ACCURACY RESULTS")
print("="*70)
print()
print(f"Correlation:    {correlation:.4f}")
print(f"MAE:            {mae:.2f}")
print(f"RMSE:           {rmse:.2f}")
print()
print(f"Status: {'✅ PASS' if correlation >= 0.70 else '❌ FAIL'}")
print()
