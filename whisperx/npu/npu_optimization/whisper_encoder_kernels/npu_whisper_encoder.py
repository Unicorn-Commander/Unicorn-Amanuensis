#!/usr/bin/env python3
"""
NPU Whisper Encoder - Combines weight loader with NPU kernels

Loads Whisper encoder weights from ONNX and executes encoder layers
using AMD Phoenix NPU kernels (LayerNorm, Softmax, GELU, MatMul).

This is an initial implementation focusing on data flow correctness.
Actual dimensions may need adjustment for production use.

AMD Phoenix NPU - XDNA1
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional
from pathlib import Path
import sys

# Import our components
from whisper_weight_loader import WhisperWeightLoader
sys.path.insert(0, str(Path(__file__).parent / 'kernels_xdna1'))
from npu_encoder import NPUEncoder


class NPUWhisperEncoder:
    """
    NPU-accelerated Whisper Encoder using actual model weights.

    Combines:
    - WhisperWeightLoader: Loads weights from ONNX model
    - NPUEncoder: Executes NPU kernels (LayerNorm, Softmax, GELU, MatMul)

    Whisper-base architecture:
    - 6 encoder layers
    - Hidden dim: 512
    - FFN dim: 2048
    - Attention heads: 8
    - Head dim: 64
    """

    # Whisper-base configuration
    HIDDEN_DIM = 512
    FFN_DIM = 2048
    NUM_HEADS = 8
    HEAD_DIM = 64  # 512 / 8
    NUM_LAYERS = 6

    def __init__(self, onnx_path: str, kernel_dir: str):
        """
        Initialize NPU Whisper Encoder.

        Args:
            onnx_path: Path to ONNX encoder model (e.g., encoder_model.onnx)
            kernel_dir: Path to directory containing NPU kernel builds
        """
        print("=" * 70)
        print("NPU Whisper Encoder Initialization")
        print("=" * 70)

        # Store paths
        self.onnx_path = Path(onnx_path)
        self.kernel_dir = Path(kernel_dir)

        # Verify paths
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        if not self.kernel_dir.exists():
            raise FileNotFoundError(f"Kernel directory not found: {kernel_dir}")

        # Load weights
        print("\n[1/3] Loading Whisper weights from ONNX...")
        self.weight_loader = WhisperWeightLoader(str(onnx_path))

        # Verify layer count
        if self.weight_loader.num_layers != self.NUM_LAYERS:
            print(f"  WARNING: Expected {self.NUM_LAYERS} layers, found {self.weight_loader.num_layers}")

        # Initialize NPU kernels
        print("\n[2/3] Initializing NPU kernels...")
        # Use 512 elements for element-wise ops (matches hidden dim)
        self.npu_encoder = NPUEncoder(str(kernel_dir), num_elements=self.HIDDEN_DIM)

        # Pre-load and cache weights for all layers
        print("\n[3/3] Caching layer weights...")
        self._cache_weights()

        # Performance tracking
        self._layer_times: Dict[str, list] = {
            'layernorm_pre_attn': [],
            'qkv_projection': [],
            'attention': [],
            'output_projection': [],
            'residual_1': [],
            'layernorm_pre_ffn': [],
            'ffn_fc1': [],
            'gelu': [],
            'ffn_fc2': [],
            'residual_2': [],
        }

        print("\n" + "=" * 70)
        print("NPU Whisper Encoder initialized successfully!")
        print(f"  Layers: {self.weight_loader.num_layers}")
        print(f"  Hidden dim: {self.HIDDEN_DIM}")
        print(f"  FFN dim: {self.FFN_DIM}")
        print(f"  Attention heads: {self.NUM_HEADS}")
        print("=" * 70)

    def _cache_weights(self):
        """Pre-load and convert all layer weights to BF16 format."""
        self._cached_weights = {}

        for layer_idx in range(self.weight_loader.num_layers):
            layer_weights = self.weight_loader.get_layer_weights(layer_idx)

            # Convert all weights to BF16 bytes for NPU
            self._cached_weights[layer_idx] = {}
            for key, weight in layer_weights.items():
                # Convert to BF16 bytes
                bf16_bytes = self.npu_encoder.float_to_bf16(weight.flatten())
                self._cached_weights[layer_idx][key] = {
                    'bf16': bf16_bytes,
                    'shape': weight.shape,
                    'dtype': weight.dtype,
                }

            print(f"  Cached layer {layer_idx}: {len(layer_weights)} weights")

    def _get_weight_bf16(self, layer_idx: int, name: str) -> bytes:
        """Get cached BF16 weight bytes."""
        return self._cached_weights[layer_idx][name]['bf16']

    def _get_weight_shape(self, layer_idx: int, name: str) -> Tuple:
        """Get weight shape."""
        return self._cached_weights[layer_idx][name]['shape']

    def _layernorm_on_npu(self, x: np.ndarray, weight: bytes, bias: bytes) -> np.ndarray:
        """
        Apply LayerNorm using NPU kernel.

        Note: Current NPU kernel does basic normalization. For production,
        would need to incorporate weight (gamma) and bias (beta) scaling.

        Args:
            x: Input array (batch, seq, hidden)
            weight: LayerNorm gamma in BF16
            bias: LayerNorm beta in BF16

        Returns:
            Normalized output
        """
        # For now, process each position through NPU LayerNorm
        # This is a simplified version - production would batch this
        original_shape = x.shape
        x_flat = x.flatten()

        # Process in chunks of hidden_dim through NPU
        result = np.zeros_like(x_flat)
        chunk_size = self.HIDDEN_DIM

        for i in range(0, len(x_flat), chunk_size):
            chunk = x_flat[i:i+chunk_size]
            if len(chunk) < chunk_size:
                # Pad if needed
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            # Convert to BF16 and run through NPU
            chunk_bf16 = self.npu_encoder.float_to_bf16(chunk)
            norm_bf16 = self.npu_encoder.layernorm(chunk_bf16)
            norm_float = self.npu_encoder.bf16_to_float(norm_bf16)

            # Store result
            end = min(i + chunk_size, len(x_flat))
            result[i:end] = norm_float[:end-i]

        return result.reshape(original_shape)

    def _matmul_tiled(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Perform matrix multiplication using tiled NPU MatMul kernel.

        The NPU kernel supports 64x64 matrices, so we tile larger matrices.

        Args:
            A: Input matrix (M, K)
            B: Weight matrix (K, N)

        Returns:
            Result matrix (M, N)
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Dimension mismatch: {K} != {K2}"

        # For this initial version, we'll use CPU for non-64x64 matrices
        # and NPU for 64x64 tiles where possible
        if M == 64 and K == 64 and N == 64:
            # Direct NPU execution
            A_bf16 = self.npu_encoder.float_to_bf16(A.flatten())
            B_bf16 = self.npu_encoder.float_to_bf16(B.flatten())
            C_bf16 = self.npu_encoder.matmul(A_bf16, B_bf16)
            return self.npu_encoder.bf16_to_float(C_bf16).reshape(M, N)
        else:
            # CPU fallback for non-matching dimensions
            # In production, implement proper tiling
            return np.matmul(A, B)

    def _softmax_on_npu(self, x: np.ndarray) -> np.ndarray:
        """
        Apply softmax using NPU kernel.

        Args:
            x: Input array

        Returns:
            Softmax output (sums to 1.0 along last axis)
        """
        original_shape = x.shape

        # For attention scores, need softmax along last dimension
        if len(original_shape) > 1:
            # Reshape to process each row through NPU
            flat = x.reshape(-1, x.shape[-1])
            result = np.zeros_like(flat)

            for i in range(flat.shape[0]):
                row = flat[i]
                if len(row) <= self.HIDDEN_DIM:
                    # Pad to hidden_dim if needed
                    padded = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
                    padded[:len(row)] = row

                    row_bf16 = self.npu_encoder.float_to_bf16(padded)
                    sm_bf16 = self.npu_encoder.softmax(row_bf16)
                    sm_float = self.npu_encoder.bf16_to_float(sm_bf16)
                    result[i] = sm_float[:len(row)]
                else:
                    # CPU fallback for longer sequences
                    exp_x = np.exp(row - np.max(row))
                    result[i] = exp_x / exp_x.sum()

            return result.reshape(original_shape)
        else:
            # Single vector
            if len(x) <= self.HIDDEN_DIM:
                padded = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
                padded[:len(x)] = x
                x_bf16 = self.npu_encoder.float_to_bf16(padded)
                sm_bf16 = self.npu_encoder.softmax(x_bf16)
                return self.npu_encoder.bf16_to_float(sm_bf16)[:len(x)]
            else:
                exp_x = np.exp(x - np.max(x))
                return exp_x / exp_x.sum()

    def _gelu_on_npu(self, x: np.ndarray) -> np.ndarray:
        """
        Apply GELU activation using NPU kernel.

        Args:
            x: Input array

        Returns:
            GELU-activated output
        """
        original_shape = x.shape
        x_flat = x.flatten()

        # Process in chunks
        result = np.zeros_like(x_flat)
        chunk_size = self.HIDDEN_DIM

        for i in range(0, len(x_flat), chunk_size):
            chunk = x_flat[i:i+chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            chunk_bf16 = self.npu_encoder.float_to_bf16(chunk)
            gelu_bf16 = self.npu_encoder.gelu(chunk_bf16)
            gelu_float = self.npu_encoder.bf16_to_float(gelu_bf16)

            end = min(i + chunk_size, len(x_flat))
            result[i:end] = gelu_float[:end-i]

        return result.reshape(original_shape)

    def encode_layer(self, hidden_states: np.ndarray, layer_idx: int) -> np.ndarray:
        """
        Run one encoder layer with actual weights on NPU.

        Transformer encoder layer structure:
        1. LayerNorm (pre-attention)
        2. Self-attention: Q, K, V projections -> attention scores -> output
        3. Residual connection
        4. LayerNorm (pre-FFN)
        5. FFN: FC1 -> GELU -> FC2
        6. Residual connection

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)
            layer_idx: Which encoder layer to run (0 to num_layers-1)

        Returns:
            Output tensor with same shape as input
        """
        if layer_idx < 0 or layer_idx >= self.weight_loader.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {self.weight_loader.num_layers})")

        # Ensure input is float32
        hidden_states = np.asarray(hidden_states, dtype=np.float32)
        original_shape = hidden_states.shape

        # Handle 2D or 3D input
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states[np.newaxis, :]  # Add batch dimension

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Verify dimensions
        if hidden_dim != self.HIDDEN_DIM:
            print(f"  WARNING: Input hidden_dim {hidden_dim} != expected {self.HIDDEN_DIM}")

        # Get cached weights
        weights = self._cached_weights[layer_idx]

        # ============================================
        # 1. Pre-attention LayerNorm
        # ============================================
        t0 = time.perf_counter()

        # Get LayerNorm weights
        ln1_weight = self._get_weight_bf16(layer_idx, 'self_attn_layer_norm_weight')
        ln1_bias = self._get_weight_bf16(layer_idx, 'self_attn_layer_norm_bias')

        # Apply LayerNorm
        normed = self._layernorm_on_npu(hidden_states, ln1_weight, ln1_bias)

        self._layer_times['layernorm_pre_attn'].append(time.perf_counter() - t0)

        # ============================================
        # 2. Self-Attention
        # ============================================

        # 2a. QKV Projections
        t0 = time.perf_counter()

        # Get projection weights
        # Note: Using CPU matmul for now due to dimension mismatch
        # In production, would tile these operations

        # For simplicity in this initial version, reshape for matrix ops
        normed_2d = normed.reshape(-1, hidden_dim)  # (batch*seq, hidden)

        # Q, K, V projections
        # Weight shapes: (hidden_dim, hidden_dim) = (512, 512)
        q_weight = self.weight_loader.get_layer_weights(layer_idx).get('q_proj_weight')
        k_weight = self.weight_loader.get_layer_weights(layer_idx).get('k_proj_weight')
        v_weight = self.weight_loader.get_layer_weights(layer_idx).get('v_proj_weight')

        if q_weight is not None and k_weight is not None and v_weight is not None:
            # Compute Q, K, V
            Q = self._matmul_tiled(normed_2d, q_weight)
            K = self._matmul_tiled(normed_2d, k_weight)
            V = self._matmul_tiled(normed_2d, v_weight)

            # Add biases
            q_bias = self.weight_loader.get_layer_weights(layer_idx).get('q_proj_bias')
            k_bias = self.weight_loader.get_layer_weights(layer_idx).get('k_proj_bias')
            v_bias = self.weight_loader.get_layer_weights(layer_idx).get('v_proj_bias')

            if q_bias is not None:
                Q = Q + q_bias
            if k_bias is not None:
                K = K + k_bias
            if v_bias is not None:
                V = V + v_bias
        else:
            # Fallback: identity projection
            print(f"  WARNING: Layer {layer_idx} missing Q/K/V weights, using identity")
            Q = K = V = normed_2d

        self._layer_times['qkv_projection'].append(time.perf_counter() - t0)

        # 2b. Compute attention scores
        t0 = time.perf_counter()

        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.NUM_HEADS, self.HEAD_DIM)
        K = K.reshape(batch_size, seq_len, self.NUM_HEADS, self.HEAD_DIM)
        V = V.reshape(batch_size, seq_len, self.NUM_HEADS, self.HEAD_DIM)

        # Transpose to (batch, heads, seq, head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scale = 1.0 / np.sqrt(self.HEAD_DIM)

        # Attention scores: (batch, heads, seq, seq)
        attn_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale

        # Apply softmax (per head, per batch)
        attn_probs = np.zeros_like(attn_scores)
        for b in range(batch_size):
            for h in range(self.NUM_HEADS):
                attn_probs[b, h] = self._softmax_on_npu(attn_scores[b, h])

        # Apply attention to values
        # (batch, heads, seq, seq) @ (batch, heads, seq, head_dim) -> (batch, heads, seq, head_dim)
        attn_output = np.matmul(attn_probs, V)

        # Reshape back: (batch, seq, hidden_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)

        self._layer_times['attention'].append(time.perf_counter() - t0)

        # 2c. Output projection
        t0 = time.perf_counter()

        o_weight = self.weight_loader.get_layer_weights(layer_idx).get('o_proj_weight')
        o_bias = self.weight_loader.get_layer_weights(layer_idx).get('o_proj_bias')

        if o_weight is not None:
            attn_output_2d = attn_output.reshape(-1, hidden_dim)
            projected = self._matmul_tiled(attn_output_2d, o_weight)
            if o_bias is not None:
                projected = projected + o_bias
            attn_output = projected.reshape(batch_size, seq_len, hidden_dim)

        self._layer_times['output_projection'].append(time.perf_counter() - t0)

        # ============================================
        # 3. First Residual Connection
        # ============================================
        t0 = time.perf_counter()
        hidden_states = hidden_states + attn_output
        self._layer_times['residual_1'].append(time.perf_counter() - t0)

        # ============================================
        # 4. Pre-FFN LayerNorm
        # ============================================
        t0 = time.perf_counter()

        ln2_weight = self._get_weight_bf16(layer_idx, 'final_layer_norm_weight')
        ln2_bias = self._get_weight_bf16(layer_idx, 'final_layer_norm_bias')

        normed = self._layernorm_on_npu(hidden_states, ln2_weight, ln2_bias)

        self._layer_times['layernorm_pre_ffn'].append(time.perf_counter() - t0)

        # ============================================
        # 5. Feed-Forward Network
        # ============================================

        # 5a. FC1: hidden_dim -> ffn_dim
        t0 = time.perf_counter()

        fc1_weight = self.weight_loader.get_layer_weights(layer_idx).get('fc1_weight')
        fc1_bias = self.weight_loader.get_layer_weights(layer_idx).get('fc1_bias')

        if fc1_weight is not None:
            normed_2d = normed.reshape(-1, hidden_dim)
            ffn_hidden = self._matmul_tiled(normed_2d, fc1_weight)
            if fc1_bias is not None:
                ffn_hidden = ffn_hidden + fc1_bias
        else:
            print(f"  WARNING: Layer {layer_idx} missing FC1 weight")
            ffn_hidden = normed.reshape(-1, hidden_dim)

        self._layer_times['ffn_fc1'].append(time.perf_counter() - t0)

        # 5b. GELU activation
        t0 = time.perf_counter()
        ffn_hidden = self._gelu_on_npu(ffn_hidden)
        self._layer_times['gelu'].append(time.perf_counter() - t0)

        # 5c. FC2: ffn_dim -> hidden_dim
        t0 = time.perf_counter()

        fc2_weight = self.weight_loader.get_layer_weights(layer_idx).get('fc2_weight')
        fc2_bias = self.weight_loader.get_layer_weights(layer_idx).get('fc2_bias')

        if fc2_weight is not None:
            ffn_output = self._matmul_tiled(ffn_hidden, fc2_weight)
            if fc2_bias is not None:
                ffn_output = ffn_output + fc2_bias
            ffn_output = ffn_output.reshape(batch_size, seq_len, hidden_dim)
        else:
            print(f"  WARNING: Layer {layer_idx} missing FC2 weight")
            ffn_output = ffn_hidden.reshape(batch_size, seq_len, hidden_dim)

        self._layer_times['ffn_fc2'].append(time.perf_counter() - t0)

        # ============================================
        # 6. Second Residual Connection
        # ============================================
        t0 = time.perf_counter()
        hidden_states = hidden_states + ffn_output
        self._layer_times['residual_2'].append(time.perf_counter() - t0)

        # Return with original shape
        if len(original_shape) == 2:
            return hidden_states[0]  # Remove batch dimension
        return hidden_states

    def encode(self, mel_features: np.ndarray) -> np.ndarray:
        """
        Run all encoder layers on mel spectrogram features.

        Args:
            mel_features: Input mel features (batch, time, features) or (time, features)

        Returns:
            Encoded output with same shape as input
        """
        hidden_states = np.asarray(mel_features, dtype=np.float32)

        print(f"\nEncoding input shape: {hidden_states.shape}")

        for layer_idx in range(self.weight_loader.num_layers):
            hidden_states = self.encode_layer(hidden_states, layer_idx)
            print(f"  Layer {layer_idx}: output shape = {hidden_states.shape}")

        return hidden_states

    def benchmark_layer(self, layer_idx: int = 0, seq_len: int = 100,
                        num_iterations: int = 10) -> dict:
        """
        Benchmark one encoder layer with timing breakdown.

        Args:
            layer_idx: Which layer to benchmark
            seq_len: Sequence length for dummy input
            num_iterations: Number of iterations to run

        Returns:
            Dictionary with timing statistics for each operation
        """
        print(f"\nBenchmarking layer {layer_idx} with seq_len={seq_len}...")

        # Reset timing stats
        for key in self._layer_times:
            self._layer_times[key] = []

        # Create dummy input
        hidden_states = np.random.randn(1, seq_len, self.HIDDEN_DIM).astype(np.float32)

        # Warmup
        _ = self.encode_layer(hidden_states, layer_idx)

        # Benchmark
        total_times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            _ = self.encode_layer(hidden_states, layer_idx)
            total_times.append(time.perf_counter() - start)

        # Compile results
        results = {
            'total_ms': np.mean(total_times) * 1000,
            'total_min_ms': np.min(total_times) * 1000,
            'total_max_ms': np.max(total_times) * 1000,
            'breakdown': {}
        }

        for key, times in self._layer_times.items():
            if times:
                results['breakdown'][key] = {
                    'avg_ms': np.mean(times) * 1000,
                    'pct': np.sum(times) / np.sum(total_times) * 100,
                }

        return results

    def print_benchmark_results(self, results: dict):
        """Print formatted benchmark results."""
        print("\n" + "=" * 70)
        print("ENCODER LAYER BENCHMARK RESULTS")
        print("=" * 70)

        print(f"\nTotal time: {results['total_ms']:.3f} ms "
              f"(min: {results['total_min_ms']:.3f}, max: {results['total_max_ms']:.3f})")

        print("\nBreakdown by operation:")
        print("-" * 50)

        # Sort by percentage
        sorted_breakdown = sorted(
            results['breakdown'].items(),
            key=lambda x: x[1]['pct'],
            reverse=True
        )

        for name, stats in sorted_breakdown:
            bar_len = int(stats['pct'] / 2)
            bar = '#' * bar_len
            print(f"  {name:25s}: {stats['avg_ms']:7.3f} ms ({stats['pct']:5.1f}%) {bar}")

        print("=" * 70)


def main():
    """Test the NPU Whisper Encoder."""
    print("=" * 70)
    print("NPU Whisper Encoder Test")
    print("=" * 70)

    # Paths
    onnx_path = (
        "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/"
        "whisper_onnx_cache/models--onnx-community--whisper-base/"
        "onnx/encoder_model.onnx"
    )
    kernel_dir = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1"

    try:
        # Initialize encoder
        encoder = NPUWhisperEncoder(onnx_path, kernel_dir)

        # Test 1: Single layer with dummy data
        print("\n" + "=" * 70)
        print("TEST 1: Single encoder layer")
        print("=" * 70)

        # Create dummy input (similar to actual Whisper input)
        # Typical: (batch=1, seq_len=1500, hidden=512)
        seq_len = 100  # Reduced for faster testing
        dummy_input = np.random.randn(seq_len, encoder.HIDDEN_DIM).astype(np.float32)

        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Input stats: mean={np.mean(dummy_input):.4f}, std={np.std(dummy_input):.4f}")

        # Run layer 0
        start = time.perf_counter()
        output = encoder.encode_layer(dummy_input, layer_idx=0)
        elapsed = time.perf_counter() - start

        print(f"\nOutput shape: {output.shape}")
        print(f"Output stats: mean={np.mean(output):.4f}, std={np.std(output):.4f}")
        print(f"Time: {elapsed * 1000:.2f} ms")

        # Test 2: Benchmark single layer
        print("\n" + "=" * 70)
        print("TEST 2: Benchmark single layer")
        print("=" * 70)

        results = encoder.benchmark_layer(layer_idx=0, seq_len=100, num_iterations=5)
        encoder.print_benchmark_results(results)

        # Test 3: Full encoder (all layers)
        print("\n" + "=" * 70)
        print("TEST 3: Full encoder (all 6 layers)")
        print("=" * 70)

        # Smaller sequence for full encoder test
        full_input = np.random.randn(50, encoder.HIDDEN_DIM).astype(np.float32)

        start = time.perf_counter()
        full_output = encoder.encode(full_input)
        full_elapsed = time.perf_counter() - start

        print(f"\nFull encoder output shape: {full_output.shape}")
        print(f"Full encoder time: {full_elapsed * 1000:.2f} ms")
        print(f"Time per layer: {full_elapsed * 1000 / encoder.weight_loader.num_layers:.2f} ms")

        # Report dimension mismatches found
        print("\n" + "=" * 70)
        print("DIMENSION ANALYSIS")
        print("=" * 70)

        print("\nWeight shapes for layer 0:")
        layer_weights = encoder.weight_loader.get_layer_weights(0)
        for name, weight in sorted(layer_weights.items()):
            print(f"  {name}: {weight.shape}")

        print("\nExpected dimensions:")
        print(f"  Hidden dim: {encoder.HIDDEN_DIM}")
        print(f"  FFN dim: {encoder.FFN_DIM}")
        print(f"  MatMul tile size: 64x64")
        print(f"  Element-wise buffer: {encoder.npu_encoder.num_elements} elements")

        # Check for potential issues
        issues = []

        if 'q_proj_weight' in layer_weights:
            q_shape = layer_weights['q_proj_weight'].shape
            if q_shape != (encoder.HIDDEN_DIM, encoder.HIDDEN_DIM):
                issues.append(f"Q projection shape {q_shape} != expected ({encoder.HIDDEN_DIM}, {encoder.HIDDEN_DIM})")

        if 'fc1_weight' in layer_weights:
            fc1_shape = layer_weights['fc1_weight'].shape
            if fc1_shape != (encoder.HIDDEN_DIM, encoder.FFN_DIM):
                issues.append(f"FC1 shape {fc1_shape} != expected ({encoder.HIDDEN_DIM}, {encoder.FFN_DIM})")

        if 'fc2_weight' in layer_weights:
            fc2_shape = layer_weights['fc2_weight'].shape
            if fc2_shape != (encoder.FFN_DIM, encoder.HIDDEN_DIM):
                issues.append(f"FC2 shape {fc2_shape} != expected ({encoder.FFN_DIM}, {encoder.HIDDEN_DIM})")

        if issues:
            print("\nDimension issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\nNo dimension issues found!")

        print("\n" + "=" * 70)
        print("NPU WHISPER ENCODER TEST COMPLETE!")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
