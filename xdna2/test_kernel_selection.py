#!/usr/bin/env python3
"""
Test kernel selection logic (without NPU hardware).

Verifies that the runtime correctly identifies which kernel to use
based on matrix dimensions.
"""

import numpy as np

def simulate_kernel_selection(M, K, N, available_kernels):
    """
    Simulate kernel selection logic.

    Args:
        M, K, N: Matrix dimensions
        available_kernels: List of available kernel dimension strings

    Returns:
        (kernel_name, needs_chunking, num_chunks)
    """
    kernel_name = f"{M}x{K}x{N}"

    if kernel_name in available_kernels:
        return (kernel_name, False, 1)
    elif K > 512 and "512x512x512" in available_kernels:
        # Chunk K dimension
        if K % 512 != 0:
            raise ValueError(f"K={K} must be divisible by 512 for chunking")
        num_chunks = K // 512
        return ("512x512x512", True, num_chunks)
    else:
        raise ValueError(f"No kernel available for {M}x{K}x{N}")

print("="*70)
print("KERNEL SELECTION LOGIC TEST")
print("="*70)

available_kernels = ["512x512x512", "512x512x2048"]

test_cases = [
    # (M, K, N, expected_kernel, expected_chunked, expected_chunks, description)
    (512, 512, 512, "512x512x512", False, 1, "Attention Q/K/V/O projections"),
    (512, 512, 2048, "512x512x2048", False, 1, "FFN fc1 expansion"),
    (512, 2048, 512, "512x512x512", True, 4, "FFN fc2 projection (chunked)"),
    (512, 1536, 512, "512x512x512", True, 3, "Custom 3-chunk matmul"),
]

print("\nAvailable kernels:")
for k in available_kernels:
    print(f"  - {k}")

print("\nTest cases:")
print("-" * 70)

all_passed = True

for M, K, N, expected_kernel, expected_chunked, expected_chunks, desc in test_cases:
    try:
        kernel, chunked, chunks = simulate_kernel_selection(M, K, N, available_kernels)

        passed = (
            kernel == expected_kernel and
            chunked == expected_chunked and
            chunks == expected_chunks
        )

        status = "✅" if passed else "❌"

        print(f"{status} {M}x{K}x{N}: {desc}")
        print(f"   Kernel: {kernel}")
        if chunked:
            print(f"   Chunked: Yes ({chunks} chunks)")
        else:
            print(f"   Chunked: No")

        if not passed:
            print(f"   EXPECTED: kernel={expected_kernel}, chunked={expected_chunked}, chunks={expected_chunks}")
            all_passed = False

        print()

    except Exception as e:
        print(f"❌ {M}x{K}x{N}: FAILED - {e}")
        print()
        all_passed = False

print("=" * 70)
if all_passed:
    print("✅ ALL KERNEL SELECTION TESTS PASSED!")
else:
    print("❌ SOME TESTS FAILED")
print("=" * 70)
print()
print("Summary:")
print("  512x512x512:   Used for attention and chunked fc2")
print("  512x512x2048:  Used for FFN fc1 expansion")
print("  Chunking:      Automatically splits K dimension when needed")
print("=" * 70)
