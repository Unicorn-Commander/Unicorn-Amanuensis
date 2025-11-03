#!/usr/bin/env python3
"""
Test Layer Normalization NPU Kernel

Validates NPU layer norm implementation against PyTorch reference.
Tests with INT8 quantized inputs and compares accuracy.

Usage:
    cd build_layernorm
    python3 ../test_layernorm.py
"""

import numpy as np
import time
import sys
import os

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - will skip reference comparison")

try:
    import pyxrt as xrt
    XRT_AVAILABLE = True
except ImportError:
    XRT_AVAILABLE = False
    print("❌ PyXRT not available - cannot test on NPU")
    sys.exit(1)


def quantize_to_int8(x, scale=127.0):
    """Quantize float32 to int8 with given scale"""
    x_quantized = np.clip(x * scale, -128, 127)
    return x_quantized.astype(np.int8)


def dequantize_from_int8(x, scale=127.0):
    """Dequantize int8 back to float32"""
    return x.astype(np.float32) / scale


def pytorch_reference_layernorm(input_float, gamma_float, beta_float, eps=1e-5):
    """Reference layer normalization using PyTorch"""
    if not TORCH_AVAILABLE:
        return None, None

    # Convert to torch tensors
    input_tensor = torch.from_numpy(input_float).float()
    gamma_tensor = torch.from_numpy(gamma_float).float()
    beta_tensor = torch.from_numpy(beta_float).float()

    # Compute layer norm using PyTorch
    layer_norm = nn.LayerNorm(input_tensor.shape[-1], eps=eps, elementwise_affine=True)
    layer_norm.weight.data = gamma_tensor
    layer_norm.bias.data = beta_tensor

    output_tensor = layer_norm(input_tensor)

    return output_tensor.numpy(), layer_norm


def numpy_reference_layernorm(input_float, gamma_float, beta_float, eps=1e-5):
    """Reference layer normalization using NumPy (matches PyTorch)"""
    # Compute mean and variance
    mean = np.mean(input_float, axis=-1, keepdims=True)
    var = np.var(input_float, axis=-1, keepdims=True)

    # Normalize
    normalized = (input_float - mean) / np.sqrt(var + eps)

    # Scale and shift
    output = gamma_float * normalized + beta_float

    return output


def compute_accuracy_metrics(npu_output, reference_output):
    """Compute accuracy metrics between NPU and reference"""
    # Mean absolute error
    mae = np.mean(np.abs(npu_output - reference_output))

    # Mean squared error
    mse = np.mean((npu_output - reference_output) ** 2)

    # Root mean squared error
    rmse = np.sqrt(mse)

    # Relative error
    relative_error = mae / (np.mean(np.abs(reference_output)) + 1e-10)

    # Correlation coefficient
    correlation = np.corrcoef(npu_output.flatten(), reference_output.flatten())[0, 1]

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'relative_error': relative_error,
        'correlation': correlation
    }


def test_npu_layernorm(xclbin_path, device_id=0, num_iterations=100):
    """Test layer normalization on NPU hardware"""

    print("=" * 80)
    print("Layer Normalization NPU Test")
    print("=" * 80)
    print()

    # Configuration
    N = 256  # Feature dimension
    scale = 127.0  # Quantization scale

    print(f"Configuration:")
    print(f"  Feature dimension: {N}")
    print(f"  Quantization scale: {scale}")
    print(f"  Device ID: {device_id}")
    print(f"  XCLBIN: {xclbin_path}")
    print()

    # Generate test data
    print("Step 1: Generate test data...")
    np.random.seed(42)

    # Input: random normal distribution (typical for neural network activations)
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

    # Create combined input buffer (input + gamma + beta)
    combined_input = np.concatenate([input_int8, gamma_int8, beta_int8])
    print(f"Combined buffer size: {combined_input.shape[0]} bytes")
    print()

    # Initialize XRT device
    print("Step 3: Initialize NPU device...")
    try:
        device = xrt.device(device_id)
        xclbin_uuid = device.load_xclbin(xclbin_path)
        print(f"✅ Loaded XCLBIN: {xclbin_path}")
        print(f"   UUID: {xclbin_uuid}")
    except Exception as e:
        print(f"❌ Failed to load XCLBIN: {e}")
        return False
    print()

    # Create kernel handle
    print("Step 4: Create kernel handle...")
    try:
        kernel = xrt.kernel(device, xclbin_uuid, "layernorm_npu")
        print(f"✅ Kernel handle created")
    except Exception as e:
        print(f"❌ Failed to create kernel: {e}")
        return False
    print()

    # Allocate buffers
    print("Step 5: Allocate buffers...")
    try:
        # Input buffer (768 bytes = 256*3)
        bo_input = xrt.bo(device, combined_input.nbytes, xrt.bo.normal, kernel.group_id(0))
        bo_input.write(combined_input, 0)
        bo_input.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, combined_input.nbytes, 0)

        # Output buffer (256 bytes)
        output_int8 = np.zeros(N, dtype=np.int8)
        bo_output = xrt.bo(device, output_int8.nbytes, xrt.bo.normal, kernel.group_id(1))

        print(f"✅ Input buffer:  {combined_input.nbytes} bytes")
        print(f"✅ Output buffer: {output_int8.nbytes} bytes")
    except Exception as e:
        print(f"❌ Failed to allocate buffers: {e}")
        return False
    print()

    # Run kernel (warmup)
    print("Step 6: Warmup run...")
    try:
        run = kernel(bo_input, bo_output)
        run.wait()
        print("✅ Warmup complete")
    except Exception as e:
        print(f"❌ Kernel execution failed: {e}")
        return False
    print()

    # Benchmark performance
    print(f"Step 7: Benchmark ({num_iterations} iterations)...")
    times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        run = kernel(bo_input, bo_output)
        run.wait()
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000  # Convert to milliseconds
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000

    print(f"✅ Performance results:")
    print(f"   Average: {avg_time:.3f} ms")
    print(f"   Std Dev: {std_time:.3f} ms")
    print(f"   Min:     {min_time:.3f} ms")
    print(f"   Max:     {max_time:.3f} ms")
    print()

    # Read back results
    print("Step 8: Read back results...")
    bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, output_int8.nbytes, 0)
    bo_output.read(output_int8, 0)
    print(f"✅ Output range: [{output_int8.min()}, {output_int8.max()}]")
    print()

    # Dequantize for comparison
    output_float = dequantize_from_int8(output_int8, scale)

    # Compute reference using NumPy
    print("Step 9: Compute NumPy reference...")
    reference_float = numpy_reference_layernorm(input_float, gamma_float, beta_float)
    print(f"✅ Reference range: [{reference_float.min():.4f}, {reference_float.max():.4f}]")
    print()

    # Compute accuracy metrics
    print("Step 10: Accuracy validation...")
    metrics = compute_accuracy_metrics(output_float, reference_float)

    print(f"Accuracy Metrics:")
    print(f"  Mean Absolute Error:  {metrics['mae']:.6f}")
    print(f"  Mean Squared Error:   {metrics['mse']:.6f}")
    print(f"  Root Mean Squared Error: {metrics['rmse']:.6f}")
    print(f"  Relative Error:       {metrics['relative_error']*100:.2f}%")
    print(f"  Correlation:          {metrics['correlation']:.6f}")
    print()

    # PyTorch comparison (if available)
    if TORCH_AVAILABLE:
        print("Step 11: PyTorch reference comparison...")
        pytorch_output, _ = pytorch_reference_layernorm(input_float, gamma_float, beta_float)
        pytorch_metrics = compute_accuracy_metrics(output_float, pytorch_output)

        print(f"PyTorch Comparison:")
        print(f"  Mean Absolute Error:  {pytorch_metrics['mae']:.6f}")
        print(f"  Relative Error:       {pytorch_metrics['relative_error']*100:.2f}%")
        print(f"  Correlation:          {pytorch_metrics['correlation']:.6f}")
        print()

    # Determine success
    print("=" * 80)
    success = metrics['correlation'] > 0.95 and metrics['relative_error'] < 0.05
    if success:
        print("✅ TEST PASSED!")
        print(f"   Correlation: {metrics['correlation']:.4f} > 0.95")
        print(f"   Relative Error: {metrics['relative_error']*100:.2f}% < 5%")
        print(f"   Performance: {avg_time:.3f} ms")
    else:
        print("❌ TEST FAILED!")
        print(f"   Correlation: {metrics['correlation']:.4f} (target: > 0.95)")
        print(f"   Relative Error: {metrics['relative_error']*100:.2f}% (target: < 5%)")
    print("=" * 80)
    print()

    # Print sample values for debugging
    print("Sample values (first 10 elements):")
    print(f"  Input (float):     {input_float[:10]}")
    print(f"  Input (int8):      {input_int8[:10]}")
    print(f"  NPU Output (int8): {output_int8[:10]}")
    print(f"  NPU Output (float):{output_float[:10]}")
    print(f"  Reference:         {reference_float[:10]}")
    print()

    return success


def main():
    """Main test function"""

    # Check if XCLBIN exists
    xclbin_path = "layernorm_simple.xclbin"
    if not os.path.exists(xclbin_path):
        print(f"❌ XCLBIN not found: {xclbin_path}")
        print("   Please run compile_layernorm.sh first")
        sys.exit(1)

    # Run test
    success = test_npu_layernorm(xclbin_path, num_iterations=100)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
