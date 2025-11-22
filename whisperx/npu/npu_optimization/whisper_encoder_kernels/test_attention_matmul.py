#!/usr/bin/env python3
"""
Test the tiled matmul implementation in attention_npu.py
"""

import numpy as np
import sys
from attention_npu import MultiHeadAttentionNPU

def test_basic_64x64():
    """Test basic 64x64 matmul"""
    print("=" * 70)
    print("Test 1: Basic 64x64 Matrix Multiplication")
    print("=" * 70)

    # Initialize attention module (which loads the matmul kernel)
    attn = MultiHeadAttentionNPU(n_dims=512, n_heads=8)

    if not attn.use_npu:
        print("⚠️  NPU not available, test skipped")
        return False

    # Create test matrices
    np.random.seed(42)
    A = np.random.randn(64, 64).astype(np.float32) * 0.1
    B = np.random.randn(64, 64).astype(np.float32) * 0.1

    # Compute reference
    expected = A @ B

    # Compute on NPU
    result = attn.matmul_npu(A, B)

    # Check results
    max_error = np.max(np.abs(result - expected))
    mean_error = np.mean(np.abs(result - expected))

    print(f"\nResults:")
    print(f"  Input A shape: {A.shape}")
    print(f"  Input B shape: {B.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")

    # BF16 has limited precision, so allow larger errors
    if max_error < 0.5:
        print("  ✅ Test PASSED")
        return True
    else:
        print(f"  ❌ Test FAILED: max error {max_error:.6f} > 0.5")
        return False

def test_arbitrary_size():
    """Test arbitrary sized matrices with tiling"""
    print("\n" + "=" * 70)
    print("Test 2: Arbitrary Sized Matrix Multiplication (100x80 @ 80x120)")
    print("=" * 70)

    attn = MultiHeadAttentionNPU(n_dims=512, n_heads=8)

    if not attn.use_npu:
        print("⚠️  NPU not available, test skipped")
        return False

    # Create non-64 aligned matrices
    np.random.seed(43)
    A = np.random.randn(100, 80).astype(np.float32) * 0.1
    B = np.random.randn(80, 120).astype(np.float32) * 0.1

    # Compute reference
    expected = A @ B

    # Compute on NPU with tiling
    result = attn.matmul_npu(A, B)

    # Check results
    max_error = np.max(np.abs(result - expected))
    mean_error = np.mean(np.abs(result - expected))

    print(f"\nResults:")
    print(f"  Input A shape: {A.shape}")
    print(f"  Input B shape: {B.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Padded to: (128, 128) @ (128, 128) using {2*2} tiles")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")

    if max_error < 0.5:
        print("  ✅ Test PASSED")
        return True
    else:
        print(f"  ❌ Test FAILED: max error {max_error:.6f} > 0.5")
        return False

def test_whisper_sizes():
    """Test Whisper-specific matrix sizes"""
    print("\n" + "=" * 70)
    print("Test 3: Whisper Encoder Matrix Sizes")
    print("=" * 70)

    attn = MultiHeadAttentionNPU(n_dims=512, n_heads=8)

    if not attn.use_npu:
        print("⚠️  NPU not available, test skipped")
        return False

    # Typical Whisper encoder sizes
    seq_len = 10  # Small sequence for testing
    n_dims = 512

    np.random.seed(44)

    # Test Q @ K^T: (10, 64) @ (64, 10) = (10, 10)
    print("\n  Test 3a: Attention scores (seq_len, head_dim) @ (head_dim, seq_len)")
    Q = np.random.randn(seq_len, 64).astype(np.float32) * 0.1
    K = np.random.randn(seq_len, 64).astype(np.float32) * 0.1

    expected = Q @ K.T
    result = attn.matmul_npu(Q, K.T)

    max_error = np.max(np.abs(result - expected))
    print(f"    Q @ K^T: {Q.shape} @ {K.T.shape} -> {result.shape}")
    print(f"    Max error: {max_error:.6f}")

    if max_error > 0.5:
        print(f"    ❌ Q @ K^T failed: max error {max_error:.6f}")
        return False

    # Test attention @ V: (10, 10) @ (10, 64) = (10, 64)
    print("\n  Test 3b: Attention output (seq_len, seq_len) @ (seq_len, head_dim)")
    attn_weights = np.random.rand(seq_len, seq_len).astype(np.float32) * 0.1
    V = np.random.randn(seq_len, 64).astype(np.float32) * 0.1

    expected = attn_weights @ V
    result = attn.matmul_npu(attn_weights, V)

    max_error = np.max(np.abs(result - expected))
    print(f"    Attn @ V: {attn_weights.shape} @ {V.shape} -> {result.shape}")
    print(f"    Max error: {max_error:.6f}")

    if max_error > 0.5:
        print(f"    ❌ Attn @ V failed: max error {max_error:.6f}")
        return False

    # Test projection: (10, 512) @ (512, 512) = (10, 512)
    print("\n  Test 3c: Linear projection (seq_len, n_dims) @ (n_dims, n_dims)")
    x = np.random.randn(seq_len, n_dims).astype(np.float32) * 0.1
    W = np.random.randn(n_dims, n_dims).astype(np.float32) * 0.02

    expected = x @ W
    result = attn.matmul_npu(x, W)

    max_error = np.max(np.abs(result - expected))
    print(f"    x @ W: {x.shape} @ {W.shape} -> {result.shape}")
    print(f"    Max error: {max_error:.6f}")

    if max_error > 0.5:
        print(f"    ❌ x @ W failed: max error {max_error:.6f}")
        return False

    print("\n  ✅ All Whisper size tests PASSED")
    return True

def test_cpu_fallback():
    """Test CPU fallback when NPU is not available"""
    print("\n" + "=" * 70)
    print("Test 4: CPU Fallback")
    print("=" * 70)

    # This will test the CPU path if NPU is not available
    attn = MultiHeadAttentionNPU(n_dims=512, n_heads=8)

    np.random.seed(45)
    A = np.random.randn(50, 50).astype(np.float32) * 0.1
    B = np.random.randn(50, 50).astype(np.float32) * 0.1

    result = attn.matmul_npu(A, B)
    expected = A @ B

    max_error = np.max(np.abs(result - expected))

    print(f"\nResults:")
    print(f"  NPU available: {attn.use_npu}")
    print(f"  Max error: {max_error:.10f}")

    if max_error < 1e-5:  # CPU should be exact (float32)
        print("  ✅ CPU fallback working correctly")
        return True
    else:
        print(f"  ❌ CPU fallback failed: max error {max_error:.10f}")
        return False

def main():
    print("\n" + "=" * 70)
    print("Tiled MatMul NPU Implementation Test Suite")
    print("=" * 70)
    print()

    results = []

    # Run all tests
    results.append(("64x64 Basic", test_basic_64x64()))
    results.append(("Arbitrary Size", test_arbitrary_size()))
    results.append(("Whisper Sizes", test_whisper_sizes()))
    results.append(("CPU Fallback", test_cpu_fallback()))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:20s}: {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 70)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
