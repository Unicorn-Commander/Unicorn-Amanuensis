#!/usr/bin/env python3
"""
Test to verify weight transposition bug in C++ encoder.

This test checks if we're double-transposing weights, which could explain
the accuracy issues.
"""

import numpy as np
import sys

print("="*70)
print("  WEIGHT TRANSPOSITION BUG INVESTIGATION")
print("="*70)

# Simulate PyTorch Linear layer convention
print("\n1. PyTorch Convention:")
print("-" * 70)

# PyTorch stores weights as (out_features, in_features)
# For a Linear layer: output = input @ weight.T + bias
# Where input is (batch, in_features), weight is (out_features, in_features)

in_features = 4
out_features = 3

# PyTorch weight: (out_features, in_features) = (3, 4)
pytorch_weight = np.array([
    [1, 2, 3, 4],      # Output neuron 0
    [5, 6, 7, 8],      # Output neuron 1
    [9, 10, 11, 12]    # Output neuron 2
], dtype=np.float32)

print(f"PyTorch weight shape: {pytorch_weight.shape}")
print(f"PyTorch weight (out_features={out_features}, in_features={in_features}):")
print(pytorch_weight)

# Input: (batch=1, in_features=4)
input_vec = np.array([[1.0, 0.5, 0.25, 0.125]], dtype=np.float32)
print(f"\nInput shape: {input_vec.shape}")
print(f"Input: {input_vec}")

# PyTorch matmul: output = input @ weight.T
pytorch_output = input_vec @ pytorch_weight.T
print(f"\nPyTorch output (input @ weight.T): {pytorch_output}")
print(f"Expected: [{1*1+0.5*2+0.25*3+0.125*4}, {1*5+0.5*6+0.25*7+0.125*8}, {1*9+0.5*10+0.25*11+0.125*12}]")
print(f"         = [{3.25}, {10.25}, {17.25}]")

print("\n2. Test Script Transposition:")
print("-" * 70)

# The test script does: load_weight(...).T
transposed_weight = pytorch_weight.T
print(f"After .T in Python: {transposed_weight.shape}")
print(f"Transposed weight (in_features={in_features}, out_features={out_features}):")
print(transposed_weight)

print("\n3. C API Loading:")
print("-" * 70)

# encoder_c_api.cpp line 65-66:
# Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
#     q_weight, n_state, n_state
# )

# This maps the C array (row-major) to Eigen matrix
# For Q weight: (n_state, n_state) = (512, 512)
# But we passed it transposed, so it's actually (512, 512) in Python

print("C API maps as RowMajor:")
print(f"  Passed shape: {transposed_weight.shape} (4, 3)")
print(f"  C API interprets as: (n_state, n_state) = (4, 3)")
print(f"  So C++ receives:")
print(transposed_weight)

print("\n4. C++ Matmul (encoder_layer.cpp line 172):")
print("-" * 70)

# Line 172: const int N = weight_int8.rows();  // Weight is transposed: (N, K)
# Line 210: matmul_output_int32_ = (input_int8_ * weight_int8.transpose());

# So C++ does:
#   - input: (M, K) = (1, 4)
#   - weight received: (4, 3) after Python .T
#   - C++ interprets rows() as N, so N=4, K=3
#   - But this is WRONG! We need N=3 (output), K=4 (input)

cpp_weight = transposed_weight  # Shape (4, 3) in NumPy notation
print(f"C++ weight shape (rows, cols): {cpp_weight.shape}")
print(f"C++ code says: N = weight.rows() = {cpp_weight.shape[0]}")
print(f"C++ code says: K = weight.cols() = {cpp_weight.shape[1]}")
print(f"\nFor Linear(in=4, out=3):")
print(f"  Expected: N=3 (output), K=4 (input)")
print(f"  Got:      N=4, K=3")
print(f"\n❌ DIMENSION MISMATCH!")

# What C++ actually computes:
# input: (1, 4), weight: (4, 3)
# matmul = input @ weight = (1, 4) @ (4, 3) = (1, 3)
# But C++ does weight.transpose() first!
# So: input @ weight.T = (1, 4) @ (3, 4).T = (1, 4) @ (4, 3) = (1, 3)

# Let's see what happens without the transpose in C++
cpp_output_no_transpose = input_vec @ cpp_weight
print(f"\nIf C++ does: input @ weight (no transpose):")
print(f"  = {input_vec.shape} @ {cpp_weight.shape}")
print(f"  = {cpp_output_no_transpose}")

# And with transpose (current C++ code)
print(f"\nIf C++ does: input @ weight.T (current code):")
print(f"  = {input_vec.shape} @ {cpp_weight.T.shape}")
try:
    cpp_output_with_transpose = input_vec @ cpp_weight.T
    print(f"  = {cpp_output_with_transpose}")
except ValueError as e:
    print(f"  ERROR: {e}")
    print(f"  Cannot multiply (1, 4) @ (3, 4).T - dimension mismatch!")

print("\n5. Verification:")
print("-" * 70)

print(f"PyTorch output:      {pytorch_output}")
print(f"C++ output (no .T):  {cpp_output_no_transpose}")
print(f"Match: {np.allclose(pytorch_output, cpp_output_no_transpose)}")

if np.allclose(pytorch_output, cpp_output_no_transpose):
    print("\n✅ MATCH! The transposed weight works WITHOUT .T in C++!")
    print("\nBUT the C++ code line 210 does:")
    print("  matmul_output_int32_ = (input_int8_ * weight_int8.transpose());")
    print("\nThis means:")
    print("  1. Python: load PyTorch weight (3, 4) and transpose to (4, 3)")
    print("  2. C++: Receives (4, 3), then transposes AGAIN to (3, 4)")
    print("  3. C++ matmul: input @ weight.T.T = input @ weight_original")
    print("\n❌ DOUBLE TRANSPOSITION BUG!")

    print("\n6. Solution Check:")
    print("-" * 70)

    # What if we DON'T transpose in Python?
    print("Option A: Remove .T in Python, keep .transpose() in C++")
    correct_weight_a = pytorch_weight  # Keep original (3, 4)
    print(f"  Python passes: {correct_weight_a.shape}")
    print(f"  C++ does: input @ weight.T")
    print(f"  = {input_vec.shape} @ {correct_weight_a.shape}.T")
    correct_output_a = input_vec @ correct_weight_a.T
    print(f"  = {correct_output_a}")
    print(f"  Match PyTorch: {np.allclose(pytorch_output, correct_output_a)}")

    print("\nOption B: Keep .T in Python, remove .transpose() in C++")
    correct_weight_b = pytorch_weight.T  # Transpose (4, 3)
    print(f"  Python passes: {correct_weight_b.shape}")
    print(f"  C++ does: input @ weight (NO .T)")
    print(f"  = {input_vec.shape} @ {correct_weight_b.shape}")
    correct_output_b = input_vec @ correct_weight_b
    print(f"  = {correct_output_b}")
    print(f"  Match PyTorch: {np.allclose(pytorch_output, correct_output_b)}")

    if np.allclose(pytorch_output, correct_output_a) and np.allclose(pytorch_output, correct_output_b):
        print("\n✅ BOTH OPTIONS WORK!")
        print("\nRECOMMENDATION:")
        print("  Choose Option B: Keep .T in Python, remove .transpose() in C++")
        print("  Reason: Less disruptive, only need to change encoder_layer.cpp line 210")
else:
    print("\n⚠️ Unexpected result")

print("\n" + "="*70)
print("  INVESTIGATION COMPLETE")
print("="*70)
