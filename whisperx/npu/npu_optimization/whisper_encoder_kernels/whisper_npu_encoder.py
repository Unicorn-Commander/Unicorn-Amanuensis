#!/usr/bin/env python3
"""
Whisper NPU Encoder - Complete NPU-accelerated encoder implementation
Uses working NPU kernels for maximum performance

Target: 60-80x realtime for Whisper Base (30 seconds audio)
Components: Attention + MatMul + LayerNorm + GELU (all on NPU)

Status: Integration in progress
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
import time
from pathlib import Path
from typing import Optional, Dict, List
import threading

# Import NPU kernel wrappers
from npu_attention_wrapper import NPUAttention
from npu_matmul_wrapper import NPUMatmul


class WhisperNPUEncoderLayer:
    """
    Single Whisper encoder layer on NPU

    Architecture:
      1. LayerNorm
      2. Multi-head self-attention (NPU)
      3. Residual connection
      4. LayerNorm
      5. FFN: Linear → GELU → Linear (NPU matmul + GELU)
      6. Residual connection
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        attention: NPUAttention = None,
        matmul: NPUMatmul = None
    ):
        """
        Initialize encoder layer with shared NPU kernels

        Args:
            d_model: Model dimension (512 for Whisper Base)
            num_heads: Number of attention heads (8 for Whisper Base)
            d_ff: Feed-forward dimension (2048 for Whisper Base)
            attention: Shared NPU attention kernel
            matmul: Shared NPU matmul kernel
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Use shared NPU kernels (cannot load multiple XCLBINs)
        self.attention = attention
        self.matmul = matmul

        # TODO: Initialize LayerNorm and GELU kernels when wrappers ready
        self.layernorm_kernel = None
        self.gelu_kernel = None

        # Statistics
        self.forward_calls = 0
        self.total_time_ms = 0.0

    def forward(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass through encoder layer

        Args:
            hidden_states: Input features (seq_len, d_model) INT8
            attention_mask: Optional attention mask

        Returns:
            Output features (seq_len, d_model) INT8
        """
        start = time.perf_counter()

        # Save for residual
        residual = hidden_states.copy()

        # 1. LayerNorm (TODO: use NPU kernel)
        # For now, skip LayerNorm to focus on attention + FFN
        normed = hidden_states

        # 2. Multi-head self-attention (NPU)
        attn_output = self.attention.multi_head_attention(
            normed, normed, normed,
            num_heads=self.num_heads,
            mask=attention_mask,
            quantize=False  # Already INT8
        )

        # 3. Residual connection
        hidden_states = residual + attn_output
        hidden_states = np.clip(hidden_states, -128, 127).astype(np.int8)

        # Save for residual
        residual = hidden_states.copy()

        # 4. LayerNorm (TODO: use NPU kernel)
        # For now, skip LayerNorm
        normed = hidden_states

        # 5. FFN: Linear → GELU → Linear
        # FFN is d_model → d_ff → d_model
        # For Whisper Base: 512 → 2048 → 512

        # For now, skip FFN to measure attention-only performance
        # TODO: Add NPU FFN when matmul wrapper integrated
        ffn_output = normed

        # 6. Residual connection
        hidden_states = residual + ffn_output
        hidden_states = np.clip(hidden_states, -128, 127).astype(np.int8)

        # Update statistics
        elapsed = (time.perf_counter() - start) * 1000
        self.forward_calls += 1
        self.total_time_ms += elapsed

        return hidden_states

    def get_stats(self) -> Dict[str, float]:
        """Get layer statistics"""
        avg_time = self.total_time_ms / self.forward_calls if self.forward_calls > 0 else 0
        return {
            'forward_calls': self.forward_calls,
            'total_time_ms': self.total_time_ms,
            'avg_time_ms': avg_time
        }


class WhisperNPUEncoder:
    """
    Complete Whisper encoder on NPU (6 layers for Whisper Base)

    Target performance: 60-80x realtime for 30s audio
    Current status: Attention-only (10-12x realtime), need FFN + LayerNorm
    """

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        device_id: int = 0
    ):
        """
        Initialize Whisper encoder

        Args:
            num_layers: Number of encoder layers (6 for Whisper Base)
            d_model: Model dimension (512 for Whisper Base)
            num_heads: Number of attention heads (8 for Whisper Base)
            d_ff: Feed-forward dimension (2048 for Whisper Base)
            device_id: NPU device ID
        """
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Initialize SHARED NPU kernels (can only load one XCLBIN at a time)
        print(f"Initializing {num_layers}-layer Whisper encoder on NPU...")
        print("  Loading NPU attention kernel...")
        self.attention = NPUAttention(device_id=device_id)
        print("  ✅ Attention kernel loaded")

        # Note: Cannot load matmul kernel at same time as attention
        # Will add later when we implement kernel switching
        self.matmul = None
        # print("  Loading NPU matmul kernel...")
        # self.matmul = NPUMatmul(device_id=device_id)
        # print("  ✅ MatMul kernel loaded")

        # Initialize encoder layers (all share same NPU kernels)
        self.layers = []
        for i in range(num_layers):
            layer = WhisperNPUEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                attention=self.attention,
                matmul=self.matmul
            )
            self.layers.append(layer)
        print(f"✅ Encoder initialized ({num_layers} layers, shared kernels)")

        # Statistics
        self.encode_calls = 0
        self.total_time_ms = 0.0

    def forward(
        self,
        mel_features: np.ndarray,
        attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Encode mel spectrogram features

        Args:
            mel_features: Mel features (seq_len, d_model) FP32 or INT8
            attention_mask: Optional attention mask

        Returns:
            Encoded features (seq_len, d_model) INT8
        """
        start = time.perf_counter()

        # Quantize if needed
        if mel_features.dtype == np.float32:
            # Quantize to INT8
            max_val = np.abs(mel_features).max()
            if max_val > 0:
                scale = 127.0 / max_val
                hidden_states = np.round(mel_features * scale).astype(np.int8)
            else:
                hidden_states = mel_features.astype(np.int8)
        else:
            hidden_states = mel_features

        # Pass through encoder layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer.forward(hidden_states, attention_mask)

        # Update statistics
        elapsed = (time.perf_counter() - start) * 1000
        self.encode_calls += 1
        self.total_time_ms += elapsed

        return hidden_states

    def __call__(self, mel_features: np.ndarray) -> np.ndarray:
        """Convenience method for forward pass"""
        return self.forward(mel_features)

    def get_stats(self) -> Dict[str, float]:
        """Get encoder statistics"""
        avg_time = self.total_time_ms / self.encode_calls if self.encode_calls > 0 else 0

        # Get per-layer statistics
        layer_stats = []
        for i, layer in enumerate(self.layers):
            stats = layer.get_stats()
            layer_stats.append(stats)

        return {
            'encode_calls': self.encode_calls,
            'total_time_ms': self.total_time_ms,
            'avg_time_ms': avg_time,
            'layer_stats': layer_stats
        }

    def print_stats(self):
        """Print encoder statistics"""
        stats = self.get_stats()

        print("\n" + "=" * 70)
        print("WHISPER NPU ENCODER STATISTICS")
        print("=" * 70)
        print(f"Encode calls: {stats['encode_calls']}")
        print(f"Total time: {stats['total_time_ms']:.2f}ms")
        print(f"Average time: {stats['avg_time_ms']:.2f}ms")
        print()

        # Calculate realtime factor
        audio_duration = 30.0  # seconds
        if stats['avg_time_ms'] > 0:
            rtf = (audio_duration * 1000) / stats['avg_time_ms']
            print(f"Realtime factor: {rtf:.1f}x (for 30s audio)")
        print()

        # Per-layer statistics
        print("Per-layer statistics:")
        for i, layer_stat in enumerate(stats['layer_stats']):
            if layer_stat['forward_calls'] > 0:
                print(f"  Layer {i+1}: {layer_stat['avg_time_ms']:.2f}ms average")

        print("=" * 70)


def benchmark_encoder():
    """Benchmark Whisper NPU encoder"""
    print("=" * 70)
    print("WHISPER NPU ENCODER BENCHMARK")
    print("=" * 70)
    print()

    # Initialize encoder
    encoder = WhisperNPUEncoder(
        num_layers=6,
        d_model=512,
        num_heads=8,
        d_ff=2048
    )
    print()

    # Test configurations
    configs = [
        {"seq_len": 150, "name": "10 seconds audio"},
        {"seq_len": 750, "name": "30 seconds audio (partial)"},
        {"seq_len": 1500, "name": "30 seconds audio (full)"},
    ]

    for config in configs:
        print(f"Test: {config['name']} ({config['seq_len']} frames)")

        # Generate test data
        mel_features = np.random.randint(-32, 32, (config['seq_len'], 512), dtype=np.int8)

        # Warm-up
        _ = encoder(mel_features)

        # Benchmark (3 iterations)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            output = encoder(mel_features)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)

        # Calculate realtime factor for this sequence length
        # Assume 50 frames per second
        audio_duration = config['seq_len'] / 50.0
        rtf = (audio_duration * 1000) / avg_time

        print(f"  Time: {avg_time:.2f}ms")
        print(f"  Realtime factor: {rtf:.1f}x")
        print()

    # Print summary statistics
    encoder.print_stats()

    return encoder


def estimate_full_pipeline_performance():
    """Estimate full Whisper pipeline performance with NPU"""
    print("=" * 70)
    print("FULL PIPELINE PERFORMANCE ESTIMATE")
    print("=" * 70)
    print()

    # Current measurements (attention-only encoder)
    attention_only_rtf = 10.6  # From test_npu_attention_simple.py

    # Component breakdown for Whisper Base encoder
    print("Current Status (Attention-only on NPU):")
    print(f"  Attention RTF: {attention_only_rtf:.1f}x")
    print(f"  Attention is ~65% of encoder compute")
    print(f"  Estimated full encoder (with CPU FFN): ~{attention_only_rtf * 0.65:.1f}x RTF")
    print()

    # With ALL components on NPU
    print("Target (All components on NPU):")
    print()

    # Attention: Already measured
    attention_time_per_layer = 470  # ms for 1500-frame sequence
    print(f"  Attention: {attention_time_per_layer:.0f}ms per layer (measured)")

    # FFN: 2 matmuls + GELU
    # MatMul 1: (1500, 512) @ (512, 2048) = (1500, 2048)
    # GELU: (1500, 2048)
    # MatMul 2: (1500, 2048) @ (2048, 512) = (1500, 512)

    matmul_time_per_tile = 0.484  # ms per 16x16 tile

    # MatMul 1: tiles needed
    M1, K1, N1 = 1500, 512, 2048
    tiles1 = (M1 // 16) * (K1 // 16) * (N1 // 16)
    matmul1_time = tiles1 * matmul_time_per_tile

    # MatMul 2: tiles needed
    M2, K2, N2 = 1500, 2048, 512
    tiles2 = (M2 // 16) * (K2 // 16) * (N2 // 16)
    matmul2_time = tiles2 * matmul_time_per_tile

    # GELU: assume ~0.2ms for 1500×2048 (from test_gelu.py)
    gelu_time = 5.0  # ms (conservative estimate)

    # LayerNorm: assume ~0.1ms for 1500×512 (from test_layernorm.py)
    layernorm_time = 2.0  # ms per instance, 2 per layer

    ffn_time_per_layer = matmul1_time + gelu_time + matmul2_time
    total_time_per_layer = attention_time_per_layer + ffn_time_per_layer + layernorm_time

    print(f"  FFN MatMul 1: ~{matmul1_time:.0f}ms per layer ({tiles1} tiles)")
    print(f"  FFN GELU: ~{gelu_time:.0f}ms per layer")
    print(f"  FFN MatMul 2: ~{matmul2_time:.0f}ms per layer ({tiles2} tiles)")
    print(f"  LayerNorm (2x): ~{layernorm_time:.0f}ms per layer")
    print()
    print(f"Total per layer: ~{total_time_per_layer:.0f}ms")
    print()

    # Full encoder (6 layers)
    num_layers = 6
    total_encoder_time = total_time_per_layer * num_layers

    print(f"Full encoder ({num_layers} layers): ~{total_encoder_time:.0f}ms = {total_encoder_time/1000:.2f}s")
    print()

    # Realtime factor
    audio_duration = 30.0  # seconds
    rtf = (audio_duration * 1000) / total_encoder_time

    print(f"Estimated RTF: {rtf:.1f}x")
    print()

    # Evaluate against target
    if rtf >= 60:
        print(f"✅ TARGET ACHIEVED! {rtf:.1f}x >= 60x realtime")
        print(f"   This meets the 60-80x performance goal!")
    elif rtf >= 50:
        print(f"✅ CLOSE TO TARGET! {rtf:.1f}x realtime")
        print(f"   Within range of 60-80x with optimizations")
    elif rtf >= 40:
        print(f"⚠️ MODERATE: {rtf:.1f}x realtime")
        print(f"   Need further optimization for 60-80x target")
    else:
        print(f"⚠️ BELOW TARGET: {rtf:.1f}x realtime")
        print(f"   Significant optimization needed")

    print()
    print("Next Steps:")
    print("  1. Integrate matmul wrapper for FFN layers")
    print("  2. Add GELU kernel wrapper")
    print("  3. Add LayerNorm kernel wrapper")
    print("  4. Optimize DMA transfers and buffer reuse")
    print("  5. Test with real audio and measure WER")

    print("=" * 70)
    print()

    return rtf


if __name__ == "__main__":
    # Run benchmarks
    print("\n")

    # Test encoder
    encoder = benchmark_encoder()

    # Estimate full pipeline
    estimated_rtf = estimate_full_pipeline_performance()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Current Implementation:")
    print("  ✅ NPU Attention kernel: Working (2.44ms per tile)")
    print("  ✅ NPU MatMul kernel: Working (0.484ms per tile)")
    print("  ✅ Wrapper classes: Functional")
    print("  ⚠️ FFN integration: TODO")
    print("  ⚠️ LayerNorm: TODO")
    print("  ⚠️ GELU: TODO")
    print()
    print(f"Estimated Performance (all NPU): ~{estimated_rtf:.1f}x realtime")
    print()

    if estimated_rtf >= 60:
        print("🎉 TARGET ACHIEVABLE!")
        print("   Complete integration to reach 60-80x realtime")
    else:
        print("⚠️ Additional optimization needed to reach 60-80x")

    print("=" * 70)
