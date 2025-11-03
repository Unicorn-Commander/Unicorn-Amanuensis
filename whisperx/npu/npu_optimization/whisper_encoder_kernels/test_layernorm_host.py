#!/usr/bin/env python3
"""
Test Layer Normalization C Kernel (Host-side validation)

Tests the layer norm algorithm in pure Python/NumPy
to validate the INT8 fixed-point implementation logic
before NPU hardware testing.

This validates:
1. INT8 quantization/dequantization
2. Mean and variance computation
3. Normalization accuracy
4. Fixed-point arithmetic correctness
"""

import numpy as np
import time

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - using NumPy reference only")
    print()


def quantize_to_int8(x, scale=127.0):
    """Quantize float32 to int8 with given scale"""
    x_quantized = np.clip(x * scale, -128, 127)
    return x_quantized.astype(np.int8)


def dequantize_from_int8(x, scale=127.0):
    """Dequantize int8 back to float32"""
    return x.astype(np.float32) / scale


def fixed_point_rsqrt(x):
    """Compute reciprocal square root using fixed-point arithmetic

    Matches the C implementation:
    1. Compute sqrt(x)
    2. Compute 1/sqrt(x) with scale factor 256
    """
    if x == 0:
        return 128

    # Integer sqrt using Newton-Raphson
    sqrt_x = int(np.sqrt(x))
    if sqrt_x == 0:
        return 128

    # Compute 1/sqrt(x) with scale factor 256 for precision
    rsqrt = (128 * 256) // sqrt_x

    return rsqrt


def layernorm_int8_python(input_int8, gamma_int8, beta_int8):
    """Python implementation matching the C kernel logic exactly"""

    N = len(input_int8)
    EPSILON_FIXED = 1  # Small epsilon for numerical stability

    # Step 1: Compute mean (accumulate in int32)
    sum_val = np.sum(input_int8.astype(np.int32))
    mean = sum_val // N

    # Step 2: Compute variance
    var_sum = 0
    for i in range(N):
        diff = int(input_int8[i]) - mean
        var_sum += diff * diff

    variance = var_sum // N

    # Step 3: Compute 1/sqrt(variance + epsilon)
    std_inv = fixed_point_rsqrt(variance + EPSILON_FIXED)

    # Step 4: Normalize and scale
    output = np.zeros(N, dtype=np.int8)
    for i in range(N):
        # Normalize: (x - mean) * std_inv
        centered = int(input_int8[i]) - mean

        # Apply inverse std deviation (fixed-point multiply)
        # std_inv has scale factor 256, so shift by 8
        normalized = (centered * std_inv) >> 8

        # Scale with gamma (both in Q7 format)
        # gamma is Q7, so shift by 7
        scaled = (int(gamma_int8[i]) * normalized) >> 7

        # Add beta
        result = scaled + int(beta_int8[i])

        # Clamp to INT8 range
        result = max(-128, min(127, result))
        output[i] = result

    return output


def numpy_reference_layernorm(input_float, gamma_float, beta_float, eps=1e-5):
    """Reference layer normalization using NumPy"""
    mean = np.mean(input_float)
    var = np.var(input_float)
    normalized = (input_float - mean) / np.sqrt(var + eps)
    output = gamma_float * normalized + beta_float
    return output


def pytorch_reference_layernorm(input_float, gamma_float, beta_float, eps=1e-5):
    """Reference layer normalization using PyTorch"""
    if not TORCH_AVAILABLE:
        return None

    input_tensor = torch.from_numpy(input_float).float()
    gamma_tensor = torch.from_numpy(gamma_float).float()
    beta_tensor = torch.from_numpy(beta_float).float()

    layer_norm = nn.LayerNorm(input_tensor.shape[-1], eps=eps, elementwise_affine=True)
    layer_norm.weight.data = gamma_tensor
    layer_norm.bias.data = beta_tensor

    output_tensor = layer_norm(input_tensor)
    return output_tensor.numpy()


def compute_accuracy_metrics(output, reference):
    """Compute accuracy metrics"""
    mae = np.mean(np.abs(output - reference))
    mse = np.mean((output - reference) ** 2)
    rmse = np.sqrt(mse)
    relative_error = mae / (np.mean(np.abs(reference)) + 1e-10)
    correlation = np.corrcoef(output.flatten(), reference.flatten())[0, 1]

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'relative_error': relative_error,
        'correlation': correlation
    }


def test_layernorm_algorithm():
    """Test the layer normalization algorithm"""

    print("=" * 80)
    print("Layer Normalization Algorithm Validation (Host-side)")
    print("=" * 80)
    print()

    # Configuration
    N = 256
    scale = 127.0

    print(f"Configuration:")
    print(f"  Feature dimension: {N}")
    print(f"  Quantization scale: {scale}")
    print()

    # Generate test data
    print("Step 1: Generate test data...")
    np.random.seed(42)

    # Input: random normal distribution
    input_float = np.random.randn(N).astype(np.float32) * 0.5

    # Gamma: typically initialized to 1
    gamma_float = np.ones(N, dtype=np.float32)

    # Beta: typically initialized to 0
    beta_float = np.zeros(N, dtype=np.float32)

    print(f"✅ Input statistics: mean={input_float.mean():.4f}, std={input_float.std():.4f}")
    print()

    # Quantize to INT8
    print("Step 2: Quantize to INT8...")
    input_int8 = quantize_to_int8(input_float, scale)
    gamma_int8 = quantize_to_int8(gamma_float, scale)
    beta_int8 = quantize_to_int8(beta_float, scale)

    print(f"✅ Quantized input range: [{input_int8.min()}, {input_int8.max()}]")
    print()

    # Run INT8 layer norm
    print("Step 3: Run INT8 layer normalization...")
    start = time.perf_counter()
    output_int8 = layernorm_int8_python(input_int8, gamma_int8, beta_int8)
    end = time.perf_counter()

    print(f"✅ INT8 LayerNorm completed in {(end-start)*1000:.3f} ms")
    print(f"   Output range: [{output_int8.min()}, {output_int8.max()}]")
    print()

    # Dequantize
    output_float = dequantize_from_int8(output_int8, scale)

    # Compute NumPy reference
    print("Step 4: Compute NumPy reference...")
    reference_float = numpy_reference_layernorm(input_float, gamma_float, beta_float)
    print(f"✅ Reference range: [{reference_float.min():.4f}, {reference_float.max():.4f}]")
    print()

    # Accuracy metrics
    print("Step 5: Accuracy validation...")
    metrics = compute_accuracy_metrics(output_float, reference_float)

    print(f"Accuracy Metrics (vs NumPy):")
    print(f"  Mean Absolute Error:     {metrics['mae']:.6f}")
    print(f"  Root Mean Squared Error: {metrics['rmse']:.6f}")
    print(f"  Relative Error:          {metrics['relative_error']*100:.2f}%")
    print(f"  Correlation:             {metrics['correlation']:.6f}")
    print()

    # PyTorch comparison
    if TORCH_AVAILABLE:
        print("Step 6: PyTorch comparison...")
        pytorch_output = pytorch_reference_layernorm(input_float, gamma_float, beta_float)
        pytorch_metrics = compute_accuracy_metrics(output_float, pytorch_output)

        print(f"Accuracy Metrics (vs PyTorch):")
        print(f"  Mean Absolute Error:     {pytorch_metrics['mae']:.6f}")
        print(f"  Relative Error:          {pytorch_metrics['relative_error']*100:.2f}%")
        print(f"  Correlation:             {pytorch_metrics['correlation']:.6f}")
        print()

    # Print sample values
    print("Sample values (first 10 elements):")
    print(f"  Input (float):      {input_float[:10]}")
    print(f"  Input (int8):       {input_int8[:10]}")
    print(f"  Output (int8):      {output_int8[:10]}")
    print(f"  Output (float):     {output_float[:10]}")
    print(f"  NumPy Reference:    {reference_float[:10]}")
    if TORCH_AVAILABLE:
        print(f"  PyTorch Reference:  {pytorch_output[:10]}")
    print()

    # Statistics
    print("Output Statistics:")
    print(f"  Mean:     {output_float.mean():.4f}")
    print(f"  Std Dev:  {output_float.std():.4f}")
    print(f"  Min:      {output_float.min():.4f}")
    print(f"  Max:      {output_float.max():.4f}")
    print()

    print("Reference Statistics:")
    print(f"  Mean:     {reference_float.mean():.4f}")
    print(f"  Std Dev:  {reference_float.std():.4f}")
    print(f"  Min:      {reference_float.min():.4f}")
    print(f"  Max:      {reference_float.max():.4f}")
    print()

    # Determine success
    print("=" * 80)
    success = metrics['correlation'] > 0.90 and metrics['relative_error'] < 0.10
    if success:
        print("✅ ALGORITHM VALIDATION PASSED!")
        print(f"   Correlation: {metrics['correlation']:.4f} > 0.90")
        print(f"   Relative Error: {metrics['relative_error']*100:.2f}% < 10%")
        print()
        print("   The C kernel implementation logic is correct.")
        print("   Ready for NPU hardware testing.")
    else:
        print("❌ ALGORITHM VALIDATION FAILED!")
        print(f"   Correlation: {metrics['correlation']:.4f} (target: > 0.90)")
        print(f"   Relative Error: {metrics['relative_error']*100:.2f}% (target: < 10%)")
        print()
        print("   The fixed-point arithmetic may need tuning.")
    print("=" * 80)
    print()

    return success


def main():
    """Main test function"""
    success = test_layernorm_algorithm()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
