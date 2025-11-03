#!/usr/bin/env python3
"""Quick test of 16x16 matmul kernel"""

import numpy as np
import time
from npu_matmul_wrapper_batched import NPUMatmulBatched

print("Testing 16x16 kernel...")
matmul = NPUMatmulBatched(tile_size=16)
print(f"✅ Loaded: {matmul.xclbin_path}")

A = np.random.randint(-127, 128, (32, 32), dtype=np.int8)
B = np.random.randint(-127, 128, (32, 32), dtype=np.int8)

try:
    C_npu = matmul(A, B)
    print("✅ 16x16 kernel WORKS!")
except Exception as e:
    print(f"❌ 16x16 kernel FAILED: {e}")
