#!/usr/bin/env python3
"""
Quick test for single encoder components.

Tests individual layers without full test harness overhead.
"""

import sys
sys.path.insert(0, 'xdna2')

import numpy as np
import time
from runtime.whisper_xdna2_runtime import create_runtime

print("="*70)
print("SINGLE LAYER TEST")
print("="*70)

# Create runtime (use 4-tile for testing - more flexible dimensions)
print("\n[1/5] Creating runtime with NPU (4-tile kernel)...")
runtime = create_runtime(model_size="base", use_4tile=True)
print("✅ Runtime created")

# Load weights
print("\n[2/5] Loading Whisper Base weights...")
runtime._load_encoder_weights()
print(f"✅ Loaded {len(runtime.encoder_weights)} tensors")
print(f"✅ Quantized {len(runtime.quantizer.quantized_weights)} matrices")

# Create test input (use realistic Whisper dimensions)
# For 10 seconds of audio: ~500 frames after conv downsampling
# For 30 seconds: ~1500 frames
# Use 512 (multiple of tile size) for testing
print("\n[3/5] Creating test input...")
seq_len = 512  # Use tile-friendly dimension
hidden_dim = 512
x = np.random.randn(seq_len, hidden_dim).astype(np.float32)
print(f"Input shape: {x.shape} (realistic for ~15s audio)")

# Test attention layer
print("\n[4/5] Testing attention layer (layer 0)...")
try:
    start = time.perf_counter()
    attn_out = runtime._run_attention_layer(x, layer_idx=0)
    elapsed = time.perf_counter() - start

    print(f"✅ Attention output shape: {attn_out.shape}")
    print(f"   Time: {elapsed*1000:.2f}ms")

    # Check output is reasonable
    assert attn_out.shape == (seq_len, hidden_dim), "Wrong output shape"
    assert not np.isnan(attn_out).any(), "NaN in output"
    assert not np.isinf(attn_out).any(), "Inf in output"

    print("✅ Attention layer PASS")
except Exception as e:
    print(f"❌ Attention layer FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test FFN layer
print("\n[5/5] Testing FFN layer (layer 0)...")
try:
    start = time.perf_counter()
    ffn_out = runtime._run_ffn_layer(x, layer_idx=0)
    elapsed = time.perf_counter() - start

    print(f"✅ FFN output shape: {ffn_out.shape}")
    print(f"   Time: {elapsed*1000:.2f}ms")

    # Check output is reasonable
    assert ffn_out.shape == (seq_len, hidden_dim), "Wrong output shape"
    assert not np.isnan(ffn_out).any(), "NaN in output"
    assert not np.isinf(ffn_out).any(), "Inf in output"

    print("✅ FFN layer PASS")
except Exception as e:
    print(f"❌ FFN layer FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL SINGLE LAYER TESTS PASSED!")
print("="*70)
