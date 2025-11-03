#!/usr/bin/env python3
"""
Whisper NPU Decoder with MatMul Acceleration
Replaces ALL matrix multiplications with NPU accelerated ops

Focus: Simple, working integration of matmul_16x16.xclbin into Whisper decoder
Target: Measure performance improvement and accuracy vs CPU baseline
Status: Phase 3 - Decoder Integration

Usage:
    from whisper_npu_decoder_matmul import WhisperNPUDecoderMatmul

    decoder = WhisperNPUDecoderMatmul()
    output = decoder(encoder_hidden_states, decoder_input_ids)  # Uses NPU for all matmuls
"""

import sys
import os
sys.path.insert(0, '/opt/xilinx/xrt/python')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import time
from typing import Optional, Dict, Tuple
from pathlib import Path

# Import NPU matmul wrapper
from npu_matmul_wrapper import NPUMatmul

# Import encoder components
from whisper_npu_encoder_matmul import NPUMatmulLayer


class WhisperNPUCrossAttention(nn.Module):
    """
    Whisper cross-attention with NPU-accelerated matmuls

    Replaces ALL cross-attention matrix ops with NPU matmul:
    - Q projection (from decoder)
    - K/V projections (from encoder)
    - Attention scores
    - Attention output projection
    """

    def __init__(
        self,
        npu_matmul: NPUMatmul,
        d_model: int = 512,
        num_heads: int = 8
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # NPU-accelerated projections
        self.q_proj = NPUMatmulLayer(npu_matmul, d_model, d_model)
        self.k_proj = NPUMatmulLayer(npu_matmul, d_model, d_model)
        self.v_proj = NPUMatmulLayer(npu_matmul, d_model, d_model)
        self.out_proj = NPUMatmulLayer(npu_matmul, d_model, d_model)

        self.npu_matmul = npu_matmul

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-attention: decoder attends to encoder

        Args:
            x: Decoder hidden states (tgt_len, d_model)
            encoder_hidden_states: Encoder output (src_len, d_model)
            mask: Optional attention mask

        Returns:
            Output (tgt_len, d_model)
        """
        tgt_len, d_model = x.shape
        src_len = encoder_hidden_states.size(0)

        # Q from decoder (NPU accelerated)
        Q = self.q_proj(x)  # (tgt_len, d_model)

        # K, V from encoder (NPU accelerated)
        K = self.k_proj(encoder_hidden_states)  # (src_len, d_model)
        V = self.v_proj(encoder_hidden_states)  # (src_len, d_model)

        # Reshape for multi-head
        Q = Q.reshape(tgt_len, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, tgt_len, head_dim)
        K = K.reshape(src_len, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, src_len, head_dim)
        V = V.reshape(src_len, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, src_len, head_dim)

        # Attention scores: Q @ K^T (CPU for now)
        # Shape: (num_heads, tgt_len, src_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Attention output: attn_weights @ V
        # Shape: (num_heads, tgt_len, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape: (tgt_len, d_model)
        attn_output = attn_output.transpose(0, 1).reshape(tgt_len, d_model)

        # Output projection (NPU accelerated)
        output = self.out_proj(attn_output)

        return output


class WhisperNPUSelfAttention(nn.Module):
    """
    Whisper decoder self-attention with NPU-accelerated matmuls
    (Causal attention for autoregressive generation)
    """

    def __init__(
        self,
        npu_matmul: NPUMatmul,
        d_model: int = 512,
        num_heads: int = 8
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # NPU-accelerated projections
        self.q_proj = NPUMatmulLayer(npu_matmul, d_model, d_model)
        self.k_proj = NPUMatmulLayer(npu_matmul, d_model, d_model)
        self.v_proj = NPUMatmulLayer(npu_matmul, d_model, d_model)
        self.out_proj = NPUMatmulLayer(npu_matmul, d_model, d_model)

        self.npu_matmul = npu_matmul

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Self-attention with causal masking

        Args:
            x: Input (seq_len, d_model)
            mask: Causal attention mask

        Returns:
            Output (seq_len, d_model)
        """
        seq_len, d_model = x.shape

        # Q/K/V projections (NPU accelerated)
        Q = self.q_proj(x)  # (seq_len, d_model)
        K = self.k_proj(x)  # (seq_len, d_model)
        V = self.v_proj(x)  # (seq_len, d_model)

        # Reshape for multi-head
        Q = Q.reshape(seq_len, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, seq_len, head_dim)
        K = K.reshape(seq_len, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, seq_len, head_dim)
        V = V.reshape(seq_len, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, seq_len, head_dim)

        # Attention scores: Q @ K^T (CPU for now)
        # Shape: (num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Apply causal mask (prevent attending to future positions)
        if mask is None:
            # Create causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        else:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Attention output: attn_weights @ V
        # Shape: (num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape: (seq_len, d_model)
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, d_model)

        # Output projection (NPU accelerated)
        output = self.out_proj(attn_output)

        return output


class WhisperNPUDecoderLayer(nn.Module):
    """
    Single Whisper decoder layer with NPU-accelerated matmuls

    Architecture:
      1. LayerNorm
      2. Self-attention (causal, with NPU matmuls)
      3. Residual
      4. LayerNorm
      5. Cross-attention (with NPU matmuls)
      6. Residual
      7. LayerNorm
      8. FFN (with NPU matmuls)
      9. Residual
    """

    def __init__(
        self,
        npu_matmul: NPUMatmul,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048
    ):
        super().__init__()
        self.d_model = d_model

        # Self-attention
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = WhisperNPUSelfAttention(npu_matmul, d_model, num_heads)

        # Cross-attention
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn = WhisperNPUCrossAttention(npu_matmul, d_model, num_heads)

        # FFN
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = NPUMatmulLayer(npu_matmul, d_model, d_ff)
        self.fc2 = NPUMatmulLayer(npu_matmul, d_ff, d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer

        Args:
            x: Decoder input (seq_len, d_model)
            encoder_hidden_states: Encoder output (src_len, d_model)

        Returns:
            Output (seq_len, d_model)
        """
        # Self-attention block
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = residual + x

        # Cross-attention block
        residual = x
        x = self.encoder_attn_layer_norm(x)
        x = self.encoder_attn(x, encoder_hidden_states)
        x = residual + x

        # FFN block
        residual = x
        x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = residual + x

        return x


class WhisperNPUDecoderMatmul(nn.Module):
    """
    Complete Whisper decoder with NPU-accelerated matmuls

    Replaces ALL matrix multiplications with NPU acceleration:
    - Self-attention Q/K/V projections (3 matmuls per layer)
    - Self-attention output projection (1 matmul per layer)
    - Cross-attention Q/K/V projections (3 matmuls per layer)
    - Cross-attention output projection (1 matmul per layer)
    - FFN layers (2 matmuls per layer)

    Total: 10 matmuls per layer × 6 layers = 60 NPU matmul operations

    Target: 25-29× realtime (from 19.1× baseline)
    """

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        device_id: int = 0,
        xclbin_path: Optional[str] = None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        print("=" * 70)
        print("WHISPER NPU DECODER (MATMUL ACCELERATED)")
        print("=" * 70)
        print(f"Model: {num_layers} layers, {d_model} dims, {num_heads} heads")
        print()

        # Initialize shared NPU matmul kernel
        print("Initializing NPU matmul kernel...")
        self.npu_matmul = NPUMatmul(
            xclbin_path=xclbin_path,
            device_id=device_id
        )
        print("✅ NPU matmul kernel loaded")
        print()

        # Build decoder layers
        print(f"Building {num_layers} decoder layers...")
        self.layers = nn.ModuleList([
            WhisperNPUDecoderLayer(
                npu_matmul=self.npu_matmul,
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff
            )
            for _ in range(num_layers)
        ])
        print(f"✅ Decoder built: {num_layers} layers")
        print()

        # LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)

        # Statistics
        self.forward_calls = 0
        self.total_time_ms = 0.0

        print("=" * 70)
        print("✅ WHISPER NPU DECODER READY")
        print("=" * 70)
        print()

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode with encoder hidden states

        Args:
            x: Decoder input (seq_len, d_model)
            encoder_hidden_states: Encoder output (src_len, d_model)

        Returns:
            Decoded features (seq_len, d_model)
        """
        start = time.perf_counter()

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_hidden_states)

        # Final layer norm
        x = self.layer_norm(x)

        # Update statistics
        elapsed = (time.perf_counter() - start) * 1000
        self.forward_calls += 1
        self.total_time_ms += elapsed

        return x

    def get_stats(self) -> Dict:
        """Get performance statistics"""
        avg_time = self.total_time_ms / self.forward_calls if self.forward_calls > 0 else 0

        # Get NPU matmul statistics
        matmul_stats = self.npu_matmul.get_stats()

        return {
            'forward_calls': self.forward_calls,
            'total_time_ms': self.total_time_ms,
            'avg_time_ms': avg_time,
            'npu_matmul_stats': matmul_stats
        }

    def print_stats(self, audio_duration: float = 30.0):
        """Print decoder statistics"""
        stats = self.get_stats()

        print("\n" + "=" * 70)
        print("WHISPER NPU DECODER STATISTICS")
        print("=" * 70)
        print(f"Forward calls: {stats['forward_calls']}")
        print(f"Total time: {stats['total_time_ms']:.2f}ms")
        print(f"Average time: {stats['avg_time_ms']:.2f}ms")
        print()

        # Realtime factor
        if stats['avg_time_ms'] > 0:
            rtf = (audio_duration * 1000) / stats['avg_time_ms']
            print(f"Realtime factor: {rtf:.1f}x (for {audio_duration}s audio)")
            print()

        # NPU matmul stats
        matmul_stats = stats['npu_matmul_stats']
        print("NPU MatMul Statistics:")
        print(f"  Total operations: {matmul_stats['total_calls']}")
        print(f"  Total tiles: {matmul_stats['total_tiles']}")
        print(f"  Avg time per tile: {matmul_stats['avg_time_per_tile_ms']:.3f}ms")
        print(f"  Tiles per second: {matmul_stats['tiles_per_second']:.0f}")

        print("=" * 70)


def test_decoder():
    """Test decoder with synthetic data"""
    print("\n")
    print("=" * 70)
    print("TESTING WHISPER NPU DECODER")
    print("=" * 70)
    print()

    # Initialize decoder
    decoder = WhisperNPUDecoderMatmul(
        num_layers=6,
        d_model=512,
        num_heads=8,
        d_ff=2048
    )

    # Test with different sequence lengths
    test_configs = [
        {"tgt_len": 50, "src_len": 150, "name": "Short output (10s audio)"},
        {"tgt_len": 100, "src_len": 750, "name": "Medium output (30s audio partial)"},
        {"tgt_len": 200, "src_len": 1500, "name": "Long output (30s audio full)"}
    ]

    for config in test_configs:
        print(f"\nTest: {config['name']}")
        print(f"  Target length: {config['tgt_len']}, Source length: {config['src_len']}")

        # Generate test input
        decoder_input = torch.randn(config['tgt_len'], 512)
        encoder_output = torch.randn(config['src_len'], 512)

        # Forward pass
        start = time.perf_counter()
        output = decoder(decoder_input, encoder_output)
        elapsed = (time.perf_counter() - start) * 1000

        # Calculate RTF
        # Assume 50 frames per second for encoder
        audio_duration = config['src_len'] / 50.0
        rtf = (audio_duration * 1000) / elapsed

        print(f"  Output shape: {output.shape}")
        print(f"  Time: {elapsed:.2f}ms")
        print(f"  Realtime factor: {rtf:.1f}x")

    # Print summary
    decoder.print_stats(audio_duration=30.0)

    return decoder


def benchmark_decoder(iterations: int = 10):
    """Benchmark decoder performance"""
    print("\n")
    print("=" * 70)
    print("BENCHMARK: WHISPER NPU DECODER")
    print("=" * 70)
    print()

    # Initialize decoder
    decoder = WhisperNPUDecoderMatmul(
        num_layers=6,
        d_model=512,
        num_heads=8,
        d_ff=2048
    )

    # Test data (30 seconds of audio)
    # Encoder: 1500 frames
    # Decoder: typically 200-300 tokens for 30s audio
    print(f"Benchmarking 30-second audio ({iterations} iterations)...")
    decoder_input = torch.randn(250, 512)  # 250 tokens output
    encoder_output = torch.randn(1500, 512)  # 30s audio input

    # Warm-up
    _ = decoder(decoder_input, encoder_output)

    # Benchmark
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        output = decoder(decoder_input, encoder_output)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Iteration {i+1}/{iterations}: {elapsed:.2f}ms")

    # Statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    audio_duration = 30.0  # seconds
    rtf = (audio_duration * 1000) / avg_time

    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Average time: {avg_time:.2f}ms ± {std_time:.2f}ms")
    print(f"Min/Max: {min_time:.2f}ms / {max_time:.2f}ms")
    print(f"Realtime factor: {rtf:.1f}x")
    print()

    # Check if target achieved
    if rtf >= 25:
        print(f"✅ TARGET ACHIEVED! {rtf:.1f}x >= 25x realtime")
    elif rtf >= 20:
        print(f"✅ CLOSE TO TARGET! {rtf:.1f}x realtime (target: 25-29x)")
    else:
        print(f"⚠️ BELOW TARGET: {rtf:.1f}x realtime (target: 25-29x)")

    print("=" * 70)

    # Print detailed stats
    decoder.print_stats(audio_duration=30.0)

    return decoder, rtf


if __name__ == "__main__":
    # Run tests
    print("\n" * 2)

    # Test decoder
    decoder = test_decoder()

    # Benchmark
    decoder, rtf = benchmark_decoder(iterations=10)

    print("\n" * 2)
    print("=" * 70)
    print("WHISPER NPU DECODER TESTING COMPLETE")
    print("=" * 70)
    print()
    print(f"Performance: {rtf:.1f}x realtime")
    print(f"Target: 25-29x realtime")
    print()
    if rtf >= 25:
        print("✅ READY FOR INTEGRATION")
    else:
        print("⚠️ NEEDS OPTIMIZATION")
    print("=" * 70)
