#!/usr/bin/env python3
"""
Accuracy Validation Benchmarking

Validates NPU kernel outputs against CPU reference implementations:
- Attention accuracy vs CPU INT8 reference
- LayerNorm accuracy vs CPU reference
- GELU accuracy vs CPU reference
- MatMul accuracy vs NumPy INT8

Metrics: Correlation, MSE, Max Difference
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json


class AccuracyBenchmark:
    """Validate NPU kernel outputs against CPU reference implementations"""

    def __init__(self):
        """Initialize accuracy benchmark"""
        self.encoder = None
        self.results = {}

    def _initialize_encoder(self):
        """Lazy initialization of NPU encoder"""
        if self.encoder is None:
            parent_dir = Path(__file__).parent.parent
            sys.path.insert(0, str(parent_dir))
            from test_encoder_block import NPUEncoderBlock

            print("Initializing NPU Encoder Block...")
            self.encoder = NPUEncoderBlock()
            print("Encoder initialized!")
            print()

    def reference_attention_int8(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        CPU reference implementation of INT8 attention

        Args:
            Q, K, V: Query, Key, Value matrices (64x64 INT8)

        Returns:
            Attention output (64x64 INT8)
        """
        # Compute attention scores: Q @ K^T
        # Use int32 for intermediate results to avoid overflow
        scores = np.matmul(Q.astype(np.int32), K.T.astype(np.int32))

        # Scale (divide by sqrt(d_k) = 8 for 64-dim)
        # Use right shift for division: >> 3 is divide by 8
        scores = scores >> 3

        # Softmax (simplified for INT8)
        # In real implementation, use proper softmax with quantization
        scores_exp = np.exp(scores.astype(np.float32) / 128.0)  # Scale for INT8
        scores_softmax = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)

        # Multiply by values: softmax(scores) @ V
        output = np.matmul(scores_softmax, V.astype(np.float32))

        # Requantize to INT8
        output = np.clip(output, -128, 127).astype(np.int8)

        return output

    def reference_layernorm_int8(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        CPU reference implementation of INT8 layer normalization

        Args:
            x: Input features (256 INT8)
            gamma: Scale parameters (256 INT8)
            beta: Shift parameters (256 INT8)

        Returns:
            Normalized output (256 INT8)
        """
        # Convert to float32 for computation
        x_float = x.astype(np.float32)
        gamma_float = gamma.astype(np.float32)
        beta_float = beta.astype(np.float32)

        # Layer normalization
        mean = np.mean(x_float)
        var = np.var(x_float)
        x_norm = (x_float - mean) / np.sqrt(var + 1e-5)

        # Apply scale and shift
        output = gamma_float * x_norm + beta_float

        # Requantize to INT8
        output = np.clip(output, -128, 127).astype(np.int8)

        return output

    def reference_gelu_int8(self, x: np.ndarray) -> np.ndarray:
        """
        CPU reference implementation of INT8 GELU

        Args:
            x: Input features (512 INT8)

        Returns:
            GELU output (512 INT8)
        """
        # Convert to float32
        x_float = x.astype(np.float32) / 128.0  # Scale to [-1, 1]

        # GELU: x * Φ(x) where Φ is CDF of standard normal
        # Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        gelu_output = 0.5 * x_float * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x_float + 0.044715 * x_float ** 3)
        ))

        # Requantize to INT8
        output = np.clip(gelu_output * 128, -128, 127).astype(np.int8)

        return output

    def reference_matmul_int8(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        CPU reference implementation of INT8 matmul

        Args:
            A: Matrix A (16x16 INT8)
            B: Matrix B (16x16 INT8)

        Returns:
            Matrix C = A @ B (16x16 INT8)
        """
        # Compute in INT32 to avoid overflow
        C_int32 = np.matmul(A.astype(np.int32), B.astype(np.int32))

        # Requantize to INT8 (divide by 128 to match NPU kernel)
        C_int8 = C_int32 >> 7  # Right shift by 7 = divide by 128

        # Clamp to INT8 range
        C_int8 = np.clip(C_int8, -128, 127).astype(np.int8)

        return C_int8

    def validate_attention_accuracy(self) -> Dict:
        """
        Compare NPU attention vs CPU reference

        Returns:
            Accuracy metrics dictionary
        """
        print("Validating Attention accuracy...")
        self._initialize_encoder()

        # Generate test data
        Q = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
        K = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
        V = np.random.randint(-64, 64, (64, 64), dtype=np.int8)

        # NPU version
        npu_output = self.encoder.run_attention(Q, K, V)

        # CPU reference
        cpu_output = self.reference_attention_int8(Q, K, V)

        # Calculate metrics
        correlation = np.corrcoef(npu_output.flatten(), cpu_output.flatten())[0, 1]
        mse = np.mean((npu_output.astype(np.float32) - cpu_output.astype(np.float32)) ** 2)
        max_diff = np.max(np.abs(npu_output.astype(np.int32) - cpu_output.astype(np.int32)))
        mae = np.mean(np.abs(npu_output.astype(np.int32) - cpu_output.astype(np.int32)))

        result = {
            'kernel': 'Attention',
            'correlation': float(correlation),
            'mse': float(mse),
            'max_diff': int(max_diff),
            'mae': float(mae),
            'pass': correlation > 0.95
        }

        print(f"  Correlation: {result['correlation']:.4f}")
        print(f"  MSE:         {result['mse']:.4f}")
        print(f"  Max Diff:    {result['max_diff']}")
        print(f"  MAE:         {result['mae']:.2f}")
        print(f"  Status:      {'PASS' if result['pass'] else 'FAIL'}")
        print()

        self.results['attention'] = result
        return result

    def validate_layernorm_accuracy(self) -> Dict:
        """
        Compare NPU LayerNorm vs CPU reference

        Returns:
            Accuracy metrics dictionary
        """
        print("Validating LayerNorm accuracy...")
        self._initialize_encoder()

        # Generate test data
        input_256 = np.random.randint(-64, 64, 256, dtype=np.int8)
        gamma = np.ones(256, dtype=np.int8)
        beta = np.zeros(256, dtype=np.int8)

        # NPU version
        npu_output = self.encoder.run_layernorm(input_256, gamma, beta)

        # CPU reference
        cpu_output = self.reference_layernorm_int8(input_256, gamma, beta)

        # Calculate metrics
        correlation = np.corrcoef(npu_output.flatten(), cpu_output.flatten())[0, 1]
        mse = np.mean((npu_output.astype(np.float32) - cpu_output.astype(np.float32)) ** 2)
        max_diff = np.max(np.abs(npu_output.astype(np.int32) - cpu_output.astype(np.int32)))
        mae = np.mean(np.abs(npu_output.astype(np.int32) - cpu_output.astype(np.int32)))

        result = {
            'kernel': 'LayerNorm',
            'correlation': float(correlation),
            'mse': float(mse),
            'max_diff': int(max_diff),
            'mae': float(mae),
            'pass': correlation > 0.95
        }

        print(f"  Correlation: {result['correlation']:.4f}")
        print(f"  MSE:         {result['mse']:.4f}")
        print(f"  Max Diff:    {result['max_diff']}")
        print(f"  MAE:         {result['mae']:.2f}")
        print(f"  Status:      {'PASS' if result['pass'] else 'FAIL'}")
        print()

        self.results['layernorm'] = result
        return result

    def validate_gelu_accuracy(self) -> Dict:
        """
        Compare NPU GELU vs CPU reference

        Returns:
            Accuracy metrics dictionary
        """
        print("Validating GELU accuracy...")
        self._initialize_encoder()

        # Generate test data
        input_512 = np.random.randint(-64, 64, 512, dtype=np.int8)

        # NPU version
        npu_output = self.encoder.run_gelu(input_512)

        # CPU reference
        cpu_output = self.reference_gelu_int8(input_512)

        # Calculate metrics
        correlation = np.corrcoef(npu_output.flatten(), cpu_output.flatten())[0, 1]
        mse = np.mean((npu_output.astype(np.float32) - cpu_output.astype(np.float32)) ** 2)
        max_diff = np.max(np.abs(npu_output.astype(np.int32) - cpu_output.astype(np.int32)))
        mae = np.mean(np.abs(npu_output.astype(np.int32) - cpu_output.astype(np.int32)))

        result = {
            'kernel': 'GELU',
            'correlation': float(correlation),
            'mse': float(mse),
            'max_diff': int(max_diff),
            'mae': float(mae),
            'pass': correlation > 0.90  # Lower threshold for GELU
        }

        print(f"  Correlation: {result['correlation']:.4f}")
        print(f"  MSE:         {result['mse']:.4f}")
        print(f"  Max Diff:    {result['max_diff']}")
        print(f"  MAE:         {result['mae']:.2f}")
        print(f"  Status:      {'PASS' if result['pass'] else 'FAIL'}")
        print()

        self.results['gelu'] = result
        return result

    def validate_matmul_accuracy(self) -> Dict:
        """
        Compare NPU MatMul vs NumPy INT8 reference

        Returns:
            Accuracy metrics dictionary
        """
        print("Validating MatMul accuracy...")
        self._initialize_encoder()

        # Generate test data
        A = np.random.randint(-64, 64, (16, 16), dtype=np.int8)
        B = np.random.randint(-64, 64, (16, 16), dtype=np.int8)

        # NPU version
        npu_output = self.encoder.run_matmul(A, B)

        # CPU reference
        cpu_output = self.reference_matmul_int8(A, B)

        # Calculate metrics
        correlation = np.corrcoef(npu_output.flatten(), cpu_output.flatten())[0, 1]
        mse = np.mean((npu_output.astype(np.float32) - cpu_output.astype(np.float32)) ** 2)
        max_diff = np.max(np.abs(npu_output.astype(np.int32) - cpu_output.astype(np.int32)))
        mae = np.mean(np.abs(npu_output.astype(np.int32) - cpu_output.astype(np.int32)))

        # Check for perfect match
        exact_match = np.array_equal(npu_output, cpu_output)

        result = {
            'kernel': 'MatMul',
            'correlation': float(correlation),
            'mse': float(mse),
            'max_diff': int(max_diff),
            'mae': float(mae),
            'exact_match': exact_match,
            'pass': correlation > 0.99
        }

        print(f"  Correlation:  {result['correlation']:.6f}")
        print(f"  MSE:          {result['mse']:.4f}")
        print(f"  Max Diff:     {result['max_diff']}")
        print(f"  MAE:          {result['mae']:.2f}")
        print(f"  Exact Match:  {exact_match}")
        print(f"  Status:       {'PASS' if result['pass'] else 'FAIL'}")
        print()

        self.results['matmul'] = result
        return result

    def validate_all_kernels(self) -> Dict:
        """
        Validate all kernel outputs

        Returns:
            Dictionary containing results for all kernels
        """
        print("=" * 70)
        print("ACCURACY VALIDATION - ALL KERNELS")
        print("=" * 70)
        print()

        try:
            self.validate_attention_accuracy()
            self.validate_layernorm_accuracy()
            self.validate_gelu_accuracy()
            self.validate_matmul_accuracy()

        except Exception as e:
            print(f"Error during validation: {e}")
            import traceback
            traceback.print_exc()

        # Summary
        print("=" * 70)
        print("ACCURACY VALIDATION SUMMARY")
        print("=" * 70)
        print()
        print(f"{'Kernel':<15} {'Correlation':<15} {'MSE':<12} {'Max Diff':<12} {'Status':<10}")
        print("-" * 70)

        for kernel_name, result in self.results.items():
            status = "PASS" if result['pass'] else "FAIL"
            print(f"{result['kernel']:<15} {result['correlation']:>12.4f}   "
                  f"{result['mse']:>10.4f}   {result['max_diff']:>10}   {status:<10}")

        print()

        return self.results

    def save_results(self, output_file: str):
        """Save validation results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Run standalone validation
    benchmark = AccuracyBenchmark()
    results = benchmark.validate_all_kernels()
    benchmark.save_results("accuracy_validation_results.json")
