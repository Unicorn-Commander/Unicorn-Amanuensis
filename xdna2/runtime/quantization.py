#!/usr/bin/env python3
"""
Quantization utilities for INT8 NPU acceleration.

Implements symmetric per-tensor quantization for weights and activations:
- FP32 → INT8 quantization
- INT32 → FP32 dequantization
- Scale computation and management

Quantization formula:
    scale = max(abs(tensor.min()), abs(tensor.max())) / 127
    quantized = (tensor / scale).round().clip(-127, 127).astype(int8)
    dequantized = quantized.astype(float32) * scale
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class QuantizationConfig:
    """Configuration for quantization parameters."""

    # Quantization range for int8
    INT8_MIN = -127
    INT8_MAX = 127

    # Symmetric quantization (no zero point)
    ZERO_POINT = 0

    # Minimum scale to avoid division by zero
    MIN_SCALE = 1e-10


def compute_quantization_scale(tensor: np.ndarray) -> float:
    """
    Compute symmetric quantization scale for a tensor.

    Args:
        tensor: Input tensor (FP32)

    Returns:
        Quantization scale (float)
    """
    # Find maximum absolute value
    max_abs = max(abs(tensor.min()), abs(tensor.max()))

    # Compute scale (avoid division by zero)
    scale = max(max_abs / 127.0, QuantizationConfig.MIN_SCALE)

    return scale


def quantize_tensor(
    tensor: np.ndarray,
    scale: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """
    Quantize FP32 tensor to INT8.

    Args:
        tensor: Input tensor (FP32)
        scale: Quantization scale (computed if None)

    Returns:
        Tuple of (quantized_tensor, scale)
    """
    # Compute scale if not provided
    if scale is None:
        scale = compute_quantization_scale(tensor)

    # Quantize: tensor / scale, round, clip to int8 range
    quantized = np.round(tensor / scale)
    quantized = np.clip(quantized, QuantizationConfig.INT8_MIN, QuantizationConfig.INT8_MAX)
    quantized = quantized.astype(np.int8)

    return quantized, scale


def dequantize_tensor(
    quantized: np.ndarray,
    scale: float
) -> np.ndarray:
    """
    Dequantize INT8/INT32 tensor to FP32.

    Args:
        quantized: Quantized tensor (INT8 or INT32)
        scale: Quantization scale

    Returns:
        Dequantized tensor (FP32)
    """
    return quantized.astype(np.float32) * scale


def quantize_matmul_inputs(
    A: np.ndarray,
    B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Quantize inputs for matmul operation.

    Args:
        A: Left matrix (FP32)
        B: Right matrix (FP32)

    Returns:
        Tuple of (A_int8, B_int8, scale_A, scale_B)
    """
    A_int8, scale_A = quantize_tensor(A)
    B_int8, scale_B = quantize_tensor(B)

    return A_int8, B_int8, scale_A, scale_B


def dequantize_matmul_output(
    C_int32: np.ndarray,
    scale_A: float,
    scale_B: float
) -> np.ndarray:
    """
    Dequantize matmul output.

    The NPU computes: C = A_int8 @ B_int8 (result is INT32)
    To get FP32: C_fp32 = C_int32 * scale_A * scale_B

    Args:
        C_int32: Matmul output from NPU (INT32)
        scale_A: Scale for matrix A
        scale_B: Scale for matrix B

    Returns:
        Dequantized output (FP32)
    """
    combined_scale = scale_A * scale_B
    return C_int32.astype(np.float32) * combined_scale


class QuantizedLinear:
    """
    Quantized linear layer for NPU execution.

    Stores quantized weights and scales for efficient NPU matmul.
    """

    def __init__(self, weight: np.ndarray, bias: Optional[np.ndarray] = None):
        """
        Initialize quantized linear layer.

        Args:
            weight: Weight matrix (out_features, in_features) FP32
            bias: Bias vector (out_features,) FP32, optional
        """
        self.weight_fp32 = weight
        self.bias = bias

        # Quantize weight
        self.weight_int8, self.weight_scale = quantize_tensor(weight)

        # Dimensions
        self.out_features, self.in_features = weight.shape

        logger.debug(f"Quantized linear: {self.in_features} → {self.out_features}")
        logger.debug(f"Weight scale: {self.weight_scale:.6f}")

    def __call__(
        self,
        x: np.ndarray,
        npu_matmul_fn,
        return_quantized: bool = False
    ) -> np.ndarray:
        """
        Execute quantized linear layer.

        Args:
            x: Input (batch_size, in_features) FP32
            npu_matmul_fn: NPU matmul function
            return_quantized: If True, return (output, x_scale, weight_scale)

        Returns:
            Output (batch_size, out_features) FP32
        """
        # Quantize input
        x_int8, x_scale = quantize_tensor(x)

        # Execute on NPU: C = x @ W^T (INT8 @ INT8 → INT32)
        # Weight needs to be transposed for matmul
        W_T = self.weight_int8.T

        M, K = x_int8.shape
        N = self.out_features

        # Run NPU matmul
        C_int32 = npu_matmul_fn(x_int8, W_T, M, K, N)

        # Dequantize output
        output = dequantize_matmul_output(C_int32, x_scale, self.weight_scale)

        # Add bias if present
        if self.bias is not None:
            output += self.bias

        if return_quantized:
            return output, x_scale, self.weight_scale
        return output


class WeightQuantizer:
    """
    Manages quantization of all model weights.

    Stores quantized weights and scales for efficient NPU execution.
    """

    def __init__(self):
        self.quantized_weights: Dict[str, Tuple[np.ndarray, float]] = {}
        self.fp32_weights: Dict[str, np.ndarray] = {}

    def quantize_weight(self, name: str, weight: np.ndarray):
        """
        Quantize and store a weight tensor.

        Args:
            name: Weight name (e.g., "encoder.layers.0.self_attn.q_proj")
            weight: Weight tensor (FP32)
        """
        weight_int8, scale = quantize_tensor(weight)
        self.quantized_weights[name] = (weight_int8, scale)
        self.fp32_weights[name] = weight

        logger.debug(f"Quantized {name}: shape={weight.shape}, scale={scale:.6f}")

    def get_quantized_weight(self, name: str) -> Tuple[np.ndarray, float]:
        """
        Get quantized weight and scale.

        Args:
            name: Weight name

        Returns:
            Tuple of (weight_int8, scale)
        """
        return self.quantized_weights[name]

    def get_fp32_weight(self, name: str) -> np.ndarray:
        """
        Get original FP32 weight.

        Args:
            name: Weight name

        Returns:
            Weight tensor (FP32)
        """
        return self.fp32_weights[name]

    def quantize_all(self, weights: Dict[str, np.ndarray]):
        """
        Quantize all weights in a dictionary.

        Args:
            weights: Dictionary of {name: weight_tensor}
        """
        for name, weight in weights.items():
            self.quantize_weight(name, weight)

        logger.info(f"Quantized {len(weights)} weight tensors")


def test_quantization():
    """Test quantization and dequantization."""

    print("Testing quantization utilities...")

    # Test 1: Single tensor quantization
    print("\n[1/4] Testing single tensor quantization...")
    x = np.random.randn(100, 200).astype(np.float32)
    x_int8, scale = quantize_tensor(x)
    x_recovered = dequantize_tensor(x_int8, scale)

    error = np.abs(x - x_recovered).mean()
    print(f"  Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  Scale: {scale:.6f}")
    print(f"  Quantized range: [{x_int8.min()}, {x_int8.max()}]")
    print(f"  Reconstruction error: {error:.6f}")

    # Test 2: Matmul quantization
    print("\n[2/4] Testing matmul quantization...")
    A = np.random.randn(64, 128).astype(np.float32)
    B = np.random.randn(128, 256).astype(np.float32)

    # CPU reference
    C_ref = A @ B

    # Quantized matmul
    A_int8, B_int8, scale_A, scale_B = quantize_matmul_inputs(A, B)
    C_int32 = A_int8.astype(np.int32) @ B_int8.astype(np.int32)
    C_recovered = dequantize_matmul_output(C_int32, scale_A, scale_B)

    error = np.abs(C_ref - C_recovered).mean()
    max_error = np.abs(C_ref - C_recovered).max()

    print(f"  Matrix sizes: ({A.shape}) @ ({B.shape}) = ({C_ref.shape})")
    print(f"  Scale A: {scale_A:.6f}, Scale B: {scale_B:.6f}")
    print(f"  Mean error: {error:.6f}")
    print(f"  Max error: {max_error:.6f}")

    # Test 3: QuantizedLinear (without NPU)
    print("\n[3/4] Testing QuantizedLinear...")
    weight = np.random.randn(512, 512).astype(np.float32)
    bias = np.random.randn(512).astype(np.float32)

    linear = QuantizedLinear(weight, bias)

    x = np.random.randn(1, 512).astype(np.float32)

    # CPU reference
    y_ref = x @ weight.T + bias

    # Quantized version (simulate NPU with CPU)
    def cpu_matmul(A, B, M, K, N):
        return (A.astype(np.int32) @ B.astype(np.int32))

    y_quant = linear(x, cpu_matmul)

    error = np.abs(y_ref - y_quant).mean()
    print(f"  Input: {x.shape}, Weight: {weight.shape}, Output: {y_quant.shape}")
    print(f"  Error: {error:.6f}")

    # Test 4: WeightQuantizer
    print("\n[4/4] Testing WeightQuantizer...")
    quantizer = WeightQuantizer()

    weights = {
        "layer1.weight": np.random.randn(512, 512).astype(np.float32),
        "layer2.weight": np.random.randn(2048, 512).astype(np.float32),
        "layer3.weight": np.random.randn(512, 2048).astype(np.float32),
    }

    quantizer.quantize_all(weights)

    print(f"  Quantized {len(weights)} weights")

    for name in weights:
        w_int8, scale = quantizer.get_quantized_weight(name)
        w_fp32 = quantizer.get_fp32_weight(name)
        print(f"    {name}: shape={w_fp32.shape}, scale={scale:.6f}")

    print("\n✅ All quantization tests passed!")


if __name__ == "__main__":
    test_quantization()
