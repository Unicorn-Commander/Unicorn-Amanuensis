#!/usr/bin/env python3
"""Test a single NPU matmul call to verify timing"""

import numpy as np
import time
from attention_npu import MultiHeadAttentionNPU

print("Testing single NPU kernel call timing...")

# Initialize
attn = MultiHeadAttentionNPU(n_dims=512, n_heads=8)

# Single 64x64 call
A = np.random.randn(64, 64).astype(np.float32) * 0.1
B = np.random.randn(64, 64).astype(np.float32) * 0.1

print("\nWarming up...")
_ = attn._matmul_npu_64x64(A, B)

print("Timing 10 calls...")
times = []
for i in range(10):
    start = time.perf_counter()
    _ = attn._matmul_npu_64x64(A, B)
    elapsed = time.perf_counter() - start
    times.append(elapsed * 1000)  # ms
    print(f"  Call {i+1}: {elapsed*1000:.3f} ms")

avg = np.mean(times)
print(f"\nAverage: {avg:.3f} ms per kernel call")
print(f"For 3,008 calls: {3008 * avg / 1000:.1f}s")
print(f"For 376 calls: {376 * avg / 1000:.1f}s")
