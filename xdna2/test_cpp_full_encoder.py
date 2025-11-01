#!/usr/bin/env python3
"""Test full 6-layer C++ encoder with CPU fallback."""

import numpy as np
import ctypes
from ctypes import c_void_p, c_float, c_int, c_size_t, POINTER, c_char_p
import time
import sys

print("="*70)
print("  FULL 6-LAYER C++ ENCODER TEST (CPU)")
print("="*70)

# Load C++ encoder library
lib_path = "/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build/libwhisper_encoder_cpp.so"
lib = ctypes.CDLL(lib_path)
print(f"✅ Loaded: {lib_path}")

# Define C API
lib.encoder_layer_create.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t]
lib.encoder_layer_create.restype = c_void_p

lib.encoder_layer_destroy.argtypes = [c_void_p]

lib.encoder_layer_load_weights.argtypes = [
    c_void_p,
    POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float),
    c_size_t, c_size_t
]
lib.encoder_layer_load_weights.restype = c_int

lib.encoder_layer_forward.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float), c_size_t, c_size_t]
lib.encoder_layer_forward.restype = c_int

# Configuration
n_heads, n_state, ffn_dim, seq_len = 8, 512, 2048, 512

print(f"\nModel: Whisper Base")
print(f"  Layers: 6")
print(f"  n_state: {n_state}")
print(f"  seq_len: {seq_len}")

# Create 6 layers
print(f"\n{'='*70}")
print("  CREATING 6 ENCODER LAYERS")
print(f"{'='*70}")

layers = []
for i in range(6):
    h = lib.encoder_layer_create(i, n_heads, n_state, ffn_dim)
    if not h:
        print(f"❌ Failed to create layer {i}")
        sys.exit(1)
    layers.append(h)
    print(f"✅ Layer {i} created (0x{h:x})")

# Create random weights (Xavier initialization)
print(f"\n{'='*70}")
print("  INITIALIZING WEIGHTS")
print(f"{'='*70}")

np.random.seed(42)

def xavier_weight(shape):
    w = np.random.randn(*shape).astype(np.float32)
    w *= np.sqrt(2.0 / shape[0])
    return w

# Load weights into all layers
for layer_idx, handle in enumerate(layers):
    q_w = xavier_weight((n_state, n_state))
    k_w = xavier_weight((n_state, n_state))
    v_w = xavier_weight((n_state, n_state))
    out_w = xavier_weight((n_state, n_state))

    fc1_w = xavier_weight((ffn_dim, n_state))
    fc2_w = xavier_weight((n_state, ffn_dim))

    q_b = np.zeros(n_state, dtype=np.float32)
    k_b = np.zeros(n_state, dtype=np.float32)
    v_b = np.zeros(n_state, dtype=np.float32)
    out_b = np.zeros(n_state, dtype=np.float32)
    fc1_b = np.zeros(ffn_dim, dtype=np.float32)
    fc2_b = np.zeros(n_state, dtype=np.float32)

    attn_ln_w = np.ones(n_state, dtype=np.float32)
    attn_ln_b = np.zeros(n_state, dtype=np.float32)
    ffn_ln_w = np.ones(n_state, dtype=np.float32)
    ffn_ln_b = np.zeros(n_state, dtype=np.float32)

    result = lib.encoder_layer_load_weights(
        handle,
        q_w.ctypes.data_as(POINTER(c_float)),
        k_w.ctypes.data_as(POINTER(c_float)),
        v_w.ctypes.data_as(POINTER(c_float)),
        out_w.ctypes.data_as(POINTER(c_float)),
        q_b.ctypes.data_as(POINTER(c_float)),
        k_b.ctypes.data_as(POINTER(c_float)),
        v_b.ctypes.data_as(POINTER(c_float)),
        out_b.ctypes.data_as(POINTER(c_float)),
        fc1_w.ctypes.data_as(POINTER(c_float)),
        fc2_w.ctypes.data_as(POINTER(c_float)),
        fc1_b.ctypes.data_as(POINTER(c_float)),
        fc2_b.ctypes.data_as(POINTER(c_float)),
        attn_ln_w.ctypes.data_as(POINTER(c_float)),
        attn_ln_b.ctypes.data_as(POINTER(c_float)),
        ffn_ln_w.ctypes.data_as(POINTER(c_float)),
        ffn_ln_b.ctypes.data_as(POINTER(c_float)),
        n_state,
        ffn_dim
    )

    if result != 0:
        print(f"❌ Failed to load weights for layer {layer_idx}")
        sys.exit(1)

    print(f"✅ Layer {layer_idx} weights loaded")

# Prepare input
print(f"\n{'='*70}")
print("  BENCHMARK")
print(f"{'='*70}")

input_data = np.random.randn(seq_len, n_state).astype(np.float32)
output_data = np.zeros((seq_len, n_state), dtype=np.float32)
temp_buffer = np.zeros((seq_len, n_state), dtype=np.float32)

print(f"Input: {input_data.shape}")

# Warmup
current = input_data.copy()
for i, h in enumerate(layers):
    lib.encoder_layer_forward(
        h,
        current.ctypes.data_as(POINTER(c_float)),
        temp_buffer.ctypes.data_as(POINTER(c_float)),
        seq_len,
        n_state
    )
    current = temp_buffer.copy()

print(f"✅ Warmup complete\n")

# Benchmark
times = []
for run in range(10):
    current = input_data.copy()

    start = time.perf_counter()

    for layer_idx, handle in enumerate(layers):
        lib.encoder_layer_forward(
            handle,
            current.ctypes.data_as(POINTER(c_float)),
            temp_buffer.ctypes.data_as(POINTER(c_float)),
            seq_len,
            n_state
        )
        current = temp_buffer.copy()

    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    print(f"  Run {run+1:2d}: {elapsed:7.2f} ms")

output_data = current

# Results
avg = np.mean(times)
min_t = np.min(times)
max_t = np.max(times)

audio_duration = 10.24
realtime = (audio_duration * 1000) / avg

print(f"\n{'='*70}")
print("  RESULTS")
print(f"{'='*70}")

print(f"\nC++ Encoder (6 layers, CPU fallback):")
print(f"  Average: {avg:.2f} ms")
print(f"  Min:     {min_t:.2f} ms")
print(f"  Max:     {max_t:.2f} ms")

print(f"\nPerformance:")
print(f"  Audio:     {audio_duration:.2f} seconds")
print(f"  Realtime:  {realtime:.2f}x")

print(f"\nComparison:")
print(f"  Python:    5.59x realtime (1,831 ms)")
print(f"  C++ CPU:   {realtime:.2f}x realtime ({avg:.0f} ms)")
print(f"  Speedup:   {1831/avg:.2f}x")

if realtime >= 17:
    print(f"\n🎉 TARGET ACHIEVED: {realtime:.2f}x >= 17x!")
elif realtime >= 7:
    print(f"\n✅ GOOD PROGRESS: {realtime:.2f}x (NPU will push to 17-28x)")
else:
    print(f"\n⏳ Below target: {realtime:.2f}x")

print(f"\nOutput:")
print(f"  Mean: {output_data.mean():.4f}")
print(f"  Std:  {output_data.std():.4f}")
print(f"  {'✅ Valid' if not (np.isnan(output_data).any() or np.isinf(output_data).any()) else '❌ Invalid'}")

# Cleanup
for h in layers:
    lib.encoder_layer_destroy(h)

print(f"\n{'='*70}")
