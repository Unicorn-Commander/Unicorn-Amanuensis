#!/usr/bin/env python3
"""Quick test of 32x32 matmul kernel using the batched wrapper"""

import numpy as np
import time
from npu_matmul_wrapper_batched import NPUMatmulBatched

print("=" * 70)
print("32x32 MATMUL KERNEL QUICK TEST")
print("=" * 70)
print()

# Initialize with 32x32 kernel
print("Initializing NPU with 32x32 kernel...")
try:
    matmul = NPUMatmulBatched(tile_size=32)
    print("✅ 32x32 kernel loaded successfully!")
    print(f"   XCLBIN: {matmul.xclbin_path}")
    print(f"   Tile size: {matmul.tile_size}x{matmul.tile_size}")
    print()
except Exception as e:
    print(f"❌ Failed to load 32x32 kernel: {e}")
    exit(1)

# Test 1: Simple 32x32 multiplication
print("Test 1: Single 32x32 tile multiplication")
print("-" * 70)
A = np.random.randint(-127, 128, (32, 32), dtype=np.int8)
B = np.random.randint(-127, 128, (32, 32), dtype=np.int8)

try:
    start = time.time()
    C_npu = matmul(A, B)
    elapsed_ms = (time.time() - start) * 1000
    
    # CPU reference
    C_ref = (A.astype(np.int32) @ B.astype(np.int32)) >> 7
    C_ref = np.clip(C_ref, -128, 127).astype(np.int8)
    
    # Compare
    diff = np.abs(C_npu.astype(np.int32) - C_ref.astype(np.int32))
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"✅ Execution time: {elapsed_ms:.2f} ms")
    print(f"   Max difference: {max_diff}")
    print(f"   Mean difference: {mean_diff:.2f}")
    print(f"   Status: {'PASS' if max_diff < 5 else 'FAIL'}")
    print()
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    print()

# Test 2: Larger matrix (64x64 = 4 tiles)
print("Test 2: 64x64 matrix (4 tiles of 32x32)")
print("-" * 70)
A = np.random.randint(-127, 128, (64, 64), dtype=np.int8)
B = np.random.randint(-127, 128, (64, 64), dtype=np.int8)

try:
    start = time.time()
    C_npu = matmul(A, B)
    elapsed_ms = (time.time() - start) * 1000
    
    # CPU reference
    C_ref = (A.astype(np.int32) @ B.astype(np.int32)) >> 7
    C_ref = np.clip(C_ref, -128, 127).astype(np.int8)
    
    # Compare
    diff = np.abs(C_npu.astype(np.int32) - C_ref.astype(np.int32))
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"✅ Execution time: {elapsed_ms:.2f} ms")
    print(f"   Tiles processed: {(64//32)**2 * 64} = {(64//32)**2} tiles × 64 k-tiles each")
    print(f"   Max difference: {max_diff}")
    print(f"   Mean difference: {mean_diff:.2f}")
    print(f"   Status: {'PASS' if max_diff < 5 else 'FAIL'}")
    print()
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    print()

# Test 3: Performance comparison size
print("Test 3: 512x512 matrix (benchmark size)")
print("-" * 70)
A = np.random.randint(-127, 128, (512, 512), dtype=np.int8)
B = np.random.randint(-127, 128, (512, 512), dtype=np.int8)

try:
    start = time.time()
    C_npu = matmul(A, B)
    elapsed_ms = (time.time() - start) * 1000
    
    n_tiles = (512 // 32) ** 2 * (512 // 32)  # (M/32)×(N/32)×(K/32)
    
    print(f"✅ Execution time: {elapsed_ms:.2f} ms")
    print(f"   Tiles processed: {n_tiles}")
    print(f"   Time per tile: {elapsed_ms/n_tiles:.4f} ms")
    print()
    
    # Simple correctness check (just a few values)
    C_ref = (A.astype(np.int32) @ B.astype(np.int32)) >> 7
    C_ref = np.clip(C_ref, -128, 127).astype(np.int8)
    
    sample_diff = np.abs(C_npu[:10, :10].astype(np.int32) - C_ref[:10, :10].astype(np.int32))
    print(f"   Sample max difference (10x10 corner): {sample_diff.max()}")
    print(f"   Sample mean difference: {sample_diff.mean():.2f}")
    print()
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    print()

print("=" * 70)
print("QUICK TEST COMPLETE")
print("=" * 70)
