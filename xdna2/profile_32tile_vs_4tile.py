#!/usr/bin/env python3
"""Profile 32-tile vs 4-tile to understand performance gap."""

import numpy as np
import time
import sys
import logging
sys.path.insert(0, "xdna2")
sys.path.insert(0, "/opt/xilinx/xrt/python")

from runtime.whisper_xdna2_runtime import create_runtime

# Enable debug logging to see matmul timings
logging.basicConfig(level=logging.DEBUG)

print("="*70)
print("  32-TILE VS 4-TILE PROFILING")
print("="*70)

def profile_single_layer(use_4tile):
    """Profile a single encoder layer."""
    tile_name = "4-tile" if use_4tile else "32-tile"
    print(f"\n{'='*70}")
    print(f"  {tile_name.upper()} PROFILING")
    print(f"{'='*70}")
    
    runtime = create_runtime(model_size="base", use_4tile=use_4tile)
    runtime._load_encoder_weights()
    
    np.random.seed(42)
    hidden_states = np.random.randn(512, 512).astype(np.float32)
    
    # Warmup
    _ = runtime._run_encoder_layer(hidden_states, layer_idx=0)
    
    # Timed run with detailed breakdown
    print(f"\nTiming {tile_name} single layer...")
    
    # Test just attention
    start = time.perf_counter()
    x_norm = runtime._layer_norm(
        hidden_states,
        runtime.encoder_weights[f"layers.0.self_attn_layer_norm.weight"],
        runtime.encoder_weights[f"layers.0.self_attn_layer_norm.bias"]
    )
    attn_out = runtime._run_attention_layer(x_norm, layer_idx=0)
    attn_time = (time.perf_counter() - start) * 1000
    
    # Test just FFN
    start = time.perf_counter()
    x_norm = runtime._layer_norm(
        hidden_states,
        runtime.encoder_weights[f"layers.0.final_layer_norm.weight"],
        runtime.encoder_weights[f"layers.0.final_layer_norm.bias"]
    )
    ffn_out = runtime._run_ffn_layer(x_norm, layer_idx=0)
    ffn_time = (time.perf_counter() - start) * 1000
    
    total_time = attn_time + ffn_time
    
    print(f"\n{tile_name} Results:")
    print(f"  Attention: {attn_time:.2f} ms ({attn_time/total_time*100:.1f}%)")
    print(f"  FFN: {ffn_time:.2f} ms ({ffn_time/total_time*100:.1f}%)")
    print(f"  Total: {total_time:.2f} ms")
    
    return {
        'attention_ms': attn_time,
        'ffn_ms': ffn_time,
        'total_ms': total_time
    }

# Profile both
results_4tile = profile_single_layer(use_4tile=True)
results_32tile = profile_single_layer(use_4tile=False)

# Comparison
print("\n" + "="*70)
print("  COMPARISON")
print("="*70)

print(f"\nAttention:")
print(f"  4-tile:  {results_4tile['attention_ms']:.2f} ms")
print(f"  32-tile: {results_32tile['attention_ms']:.2f} ms")
print(f"  Speedup: {results_4tile['attention_ms'] / results_32tile['attention_ms']:.2f}x")

print(f"\nFFN:")
print(f"  4-tile:  {results_4tile['ffn_ms']:.2f} ms")
print(f"  32-tile: {results_32tile['ffn_ms']:.2f} ms")
print(f"  Speedup: {results_4tile['ffn_ms'] / results_32tile['ffn_ms']:.2f}x")

print(f"\nTotal:")
print(f"  4-tile:  {results_4tile['total_ms']:.2f} ms")
print(f"  32-tile: {results_32tile['total_ms']:.2f} ms")
print(f"  Speedup: {results_4tile['total_ms'] / results_32tile['total_ms']:.2f}x")

print("\n" + "="*70)
