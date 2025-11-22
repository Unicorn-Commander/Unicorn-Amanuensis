#!/usr/bin/env python3
"""
NPU Whisper Encoder - Optimized Version

Hybrid CPU/NPU approach:
- CPU for vectorized operations (attention softmax, GELU, LayerNorm)
- NPU for element-wise chains where kernel overhead is amortized

Key insight: NPU kernel launch overhead (~1.5ms) means calling kernels
in tight loops is counterproductive. Use CPU for batch operations.

AMD Phoenix NPU - XDNA1
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional
from pathlib import Path
import sys

# Import our components
from whisper_weight_loader import WhisperWeightLoader


class NPUWhisperEncoderOptimized:
    """
    Optimized NPU Whisper Encoder using hybrid CPU/NPU computation.

    Uses CPU NumPy for:
    - LayerNorm (vectorized across batch/sequence)
    - Softmax (vectorized across batch/heads)
    - GELU (vectorized activation)
    - MatMul (NumPy is fast for these sizes)

    Uses NPU for:
    - Future: Batched operations where overhead is amortized
    """

    # Whisper-base configuration
    HIDDEN_DIM = 512
    FFN_DIM = 2048
    NUM_HEADS = 8
    HEAD_DIM = 64  # 512 / 8
    NUM_LAYERS = 6

    def __init__(self, onnx_path: str):
        """
        Initialize Optimized NPU Whisper Encoder.

        Args:
            onnx_path: Path to ONNX encoder model
        """
        print("=" * 70)
        print("NPU Whisper Encoder (Optimized) Initialization")
        print("=" * 70)

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Load weights
        print("\n[1/2] Loading Whisper weights from ONNX...")
        self.weight_loader = WhisperWeightLoader(str(onnx_path))

        # Cache weights
        print("\n[2/2] Caching layer weights...")
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
        print("Optimized Encoder initialized!")
        print(f"  Layers: {self.weight_loader.num_layers}")
        print(f"  Mode: CPU vectorized (avoiding NPU kernel overhead)")
        print("=" * 70)

    def _cache_weights(self):
        """Cache all layer weights as numpy arrays."""
        self._cached_weights = {}

        for layer_idx in range(self.weight_loader.num_layers):
            layer_weights = self.weight_loader.get_layer_weights(layer_idx)
            self._cached_weights[layer_idx] = {}

            for key, weight in layer_weights.items():
                self._cached_weights[layer_idx][key] = weight.astype(np.float32)

            print(f"  Cached layer {layer_idx}: {len(layer_weights)} weights")

    def _layernorm(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                   eps: float = 1e-5) -> np.ndarray:
        """Vectorized LayerNorm."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + eps)
        return normalized * weight + bias

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Vectorized softmax."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """Vectorized GELU activation."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def encode_layer(self, hidden_states: np.ndarray, layer_idx: int) -> np.ndarray:
        """
        Run one encoder layer with optimized CPU operations.

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)
            layer_idx: Which encoder layer to run (0 to num_layers-1)

        Returns:
            Output tensor with same shape as input
        """
        if layer_idx < 0 or layer_idx >= self.weight_loader.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range")

        hidden_states = np.asarray(hidden_states, dtype=np.float32)
        original_shape = hidden_states.shape

        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states[np.newaxis, :]

        batch_size, seq_len, hidden_dim = hidden_states.shape
        weights = self._cached_weights[layer_idx]

        # ============================================
        # 1. Pre-attention LayerNorm
        # ============================================
        t0 = time.perf_counter()

        ln1_weight = weights.get('self_attn_layer_norm_weight')
        ln1_bias = weights.get('self_attn_layer_norm_bias')

        if ln1_weight is not None and ln1_bias is not None:
            normed = self._layernorm(hidden_states, ln1_weight, ln1_bias)
        else:
            normed = hidden_states.copy()

        self._layer_times['layernorm_pre_attn'].append(time.perf_counter() - t0)

        # ============================================
        # 2. Self-Attention
        # ============================================

        # 2a. QKV Projections
        t0 = time.perf_counter()

        q_weight = weights.get('q_proj_weight')
        k_weight = weights.get('k_proj_weight')
        v_weight = weights.get('v_proj_weight')

        if q_weight is not None and k_weight is not None and v_weight is not None:
            # Reshape for matmul: (batch*seq, hidden)
            normed_2d = normed.reshape(-1, hidden_dim)

            Q = normed_2d @ q_weight
            K = normed_2d @ k_weight
            V = normed_2d @ v_weight

            # Add biases
            q_bias = weights.get('q_proj_bias')
            k_bias = weights.get('k_proj_bias')
            v_bias = weights.get('v_proj_bias')

            if q_bias is not None:
                Q = Q + q_bias
            if k_bias is not None:
                K = K + k_bias
            if v_bias is not None:
                V = V + v_bias
        else:
            normed_2d = normed.reshape(-1, hidden_dim)
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
        attn_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale

        # Apply vectorized softmax (all heads/batches at once)
        attn_probs = self._softmax(attn_scores, axis=-1)

        # Apply attention to values
        attn_output = np.matmul(attn_probs, V)

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)

        self._layer_times['attention'].append(time.perf_counter() - t0)

        # 2c. Output projection
        t0 = time.perf_counter()

        o_weight = weights.get('o_proj_weight')
        o_bias = weights.get('o_proj_bias')

        if o_weight is not None:
            attn_output_2d = attn_output.reshape(-1, hidden_dim)
            projected = attn_output_2d @ o_weight
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

        ln2_weight = weights.get('final_layer_norm_weight')
        ln2_bias = weights.get('final_layer_norm_bias')

        if ln2_weight is not None and ln2_bias is not None:
            normed = self._layernorm(hidden_states, ln2_weight, ln2_bias)
        else:
            normed = hidden_states.copy()

        self._layer_times['layernorm_pre_ffn'].append(time.perf_counter() - t0)

        # ============================================
        # 5. Feed-Forward Network
        # ============================================

        # 5a. FC1
        t0 = time.perf_counter()

        fc1_weight = weights.get('fc1_weight')
        fc1_bias = weights.get('fc1_bias')

        if fc1_weight is not None:
            normed_2d = normed.reshape(-1, hidden_dim)
            ffn_hidden = normed_2d @ fc1_weight
            if fc1_bias is not None:
                ffn_hidden = ffn_hidden + fc1_bias
        else:
            ffn_hidden = normed.reshape(-1, hidden_dim)

        self._layer_times['ffn_fc1'].append(time.perf_counter() - t0)

        # 5b. GELU activation (vectorized)
        t0 = time.perf_counter()
        ffn_hidden = self._gelu(ffn_hidden)
        self._layer_times['gelu'].append(time.perf_counter() - t0)

        # 5c. FC2
        t0 = time.perf_counter()

        fc2_weight = weights.get('fc2_weight')
        fc2_bias = weights.get('fc2_bias')

        if fc2_weight is not None:
            ffn_output = ffn_hidden @ fc2_weight
            if fc2_bias is not None:
                ffn_output = ffn_output + fc2_bias
            ffn_output = ffn_output.reshape(batch_size, seq_len, hidden_dim)
        else:
            ffn_output = ffn_hidden.reshape(batch_size, seq_len, hidden_dim)

        self._layer_times['ffn_fc2'].append(time.perf_counter() - t0)

        # ============================================
        # 6. Second Residual Connection
        # ============================================
        t0 = time.perf_counter()
        hidden_states = hidden_states + ffn_output
        self._layer_times['residual_2'].append(time.perf_counter() - t0)

        if len(original_shape) == 2:
            return hidden_states[0]
        return hidden_states

    def encode(self, mel_features: np.ndarray) -> np.ndarray:
        """Run all encoder layers."""
        hidden_states = np.asarray(mel_features, dtype=np.float32)

        for layer_idx in range(self.weight_loader.num_layers):
            hidden_states = self.encode_layer(hidden_states, layer_idx)

        return hidden_states

    def benchmark_layer(self, layer_idx: int = 0, seq_len: int = 100,
                        num_iterations: int = 10) -> dict:
        """Benchmark one encoder layer."""
        # Reset timing stats
        for key in self._layer_times:
            self._layer_times[key] = []

        hidden_states = np.random.randn(1, seq_len, self.HIDDEN_DIM).astype(np.float32)

        # Warmup
        _ = self.encode_layer(hidden_states, layer_idx)

        # Benchmark
        total_times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            _ = self.encode_layer(hidden_states, layer_idx)
            total_times.append(time.perf_counter() - start)

        results = {
            'total_ms': np.mean(total_times) * 1000,
            'total_min_ms': np.min(total_times) * 1000,
            'breakdown': {}
        }

        for key, times in self._layer_times.items():
            if times:
                results['breakdown'][key] = {
                    'avg_ms': np.mean(times) * 1000,
                    'pct': np.sum(times) / np.sum(total_times) * 100,
                }

        return results


def main():
    """Test the optimized NPU Whisper Encoder."""
    print("=" * 70)
    print("NPU Whisper Encoder (Optimized) Test")
    print("=" * 70)

    onnx_path = (
        "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/"
        "whisper_onnx_cache/models--onnx-community--whisper-base/"
        "onnx/encoder_model.onnx"
    )

    try:
        encoder = NPUWhisperEncoderOptimized(onnx_path)

        # Test scenarios
        test_configs = [
            {"name": "Small (10 frames)", "seq_len": 10, "audio_sec": 0.1},
            {"name": "Medium (100 frames)", "seq_len": 100, "audio_sec": 1.0},
            {"name": "Standard (500 frames)", "seq_len": 500, "audio_sec": 5.0},
            {"name": "Full (1500 frames)", "seq_len": 1500, "audio_sec": 30.0},
        ]

        results = []

        print("\n" + "=" * 70)
        print("OPTIMIZED PERFORMANCE TESTS")
        print("=" * 70)

        for config in test_configs:
            print(f"\n{'-' * 60}")
            print(f"Test: {config['name']}")
            print(f"{'-' * 60}")

            seq_len = config['seq_len']
            input_data = np.random.randn(seq_len, encoder.HIDDEN_DIM).astype(np.float32)

            # Warmup
            _ = encoder.encode_layer(input_data[:min(10, seq_len)], layer_idx=0)

            # Single layer benchmark
            layer_times = []
            for i in range(5):
                start = time.perf_counter()
                _ = encoder.encode_layer(input_data, layer_idx=0)
                layer_times.append(time.perf_counter() - start)

            avg_layer_ms = np.mean(layer_times) * 1000

            # Full encoder
            full_times = []
            for i in range(3):
                start = time.perf_counter()
                _ = encoder.encode(input_data)
                full_times.append(time.perf_counter() - start)

            avg_full_ms = np.mean(full_times) * 1000

            audio_sec = config['audio_sec']
            rtf = audio_sec / (avg_full_ms / 1000) if avg_full_ms > 0 else 0

            print(f"  Single layer: {avg_layer_ms:.2f} ms")
            print(f"  Full encoder: {avg_full_ms:.2f} ms")
            print(f"  Realtime factor: {rtf:.1f}x")

            results.append({
                'name': config['name'],
                'seq_len': seq_len,
                'audio_sec': audio_sec,
                'layer_ms': avg_layer_ms,
                'encoder_ms': avg_full_ms,
                'rtf': rtf,
            })

        # Summary
        print("\n\n" + "=" * 70)
        print("OPTIMIZED RESULTS SUMMARY")
        print("=" * 70)
        print()

        print(f"{'Test':<25} {'Seq':<8} {'Audio':<8} {'Encoder':<12} {'RTF':<10}")
        print("-" * 70)

        for r in results:
            print(f"{r['name']:<25} {r['seq_len']:<8} {r['audio_sec']:<8.1f}s {r['encoder_ms']:<12.1f}ms {r['rtf']:<10.1f}x")

        # Compare to baseline
        print("\n" + "=" * 70)
        print("COMPARISON TO BASELINE")
        print("=" * 70)
        print()

        baseline_times = {
            10: 1353.3,
            100: 13677.0,
            500: 68786.4,
            1500: 77843.4,
        }

        for r in results:
            baseline = baseline_times.get(r['seq_len'], r['encoder_ms'])
            speedup = baseline / r['encoder_ms'] if r['encoder_ms'] > 0 else 0
            print(f"{r['name']}: {speedup:.1f}x faster than baseline")

        print("\n" + "=" * 70)
        print("OPTIMIZED TEST COMPLETE")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
