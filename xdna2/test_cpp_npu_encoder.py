#!/usr/bin/env python3
"""Test C++ encoder with NPU integration."""

import numpy as np
import ctypes
from ctypes import c_void_p, c_float, c_int, c_size_t, POINTER, c_char_p, CFUNCTYPE
import time
import sys
import os

# Add XRT to path
sys.path.insert(0, "/opt/xilinx/xrt/python")
os.environ['PYTHONPATH'] = f"/opt/xilinx/xrt/python:{os.environ.get('PYTHONPATH', '')}"

print("="*70)
print("  C++ ENCODER + NPU INTEGRATION TEST")
print("="*70)

# Load C++ encoder library
lib_path = "/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build/libwhisper_encoder_cpp.so"
try:
    lib = ctypes.CDLL(lib_path)
    print(f"✅ Loaded C++ library: {lib_path}")
except OSError as e:
    print(f"❌ Failed to load C++ library: {e}")
    sys.exit(1)

# Load Python NPU runtime
sys.path.insert(0, ".")
try:
    from runtime.whisper_xdna2_runtime import create_runtime
    print(f"✅ Loaded Python NPU runtime")
except Exception as e:
    print(f"❌ Failed to load Python runtime: {e}")
    print(f"   Error: {type(e).__name__}: {e}")
    sys.exit(1)

# Define C API function signatures
lib.encoder_get_version.restype = c_char_p
lib.encoder_check_config.restype = c_int

lib.encoder_layer_create.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t]
lib.encoder_layer_create.restype = c_void_p

lib.encoder_layer_destroy.argtypes = [c_void_p]
lib.encoder_layer_destroy.restype = None

lib.encoder_layer_load_weights.argtypes = [
    c_void_p,  # handle
    POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float),
    c_size_t, c_size_t
]
lib.encoder_layer_load_weights.restype = c_int

lib.encoder_layer_forward.argtypes = [
    c_void_p,
    POINTER(c_float),
    POINTER(c_float),
    c_size_t,
    c_size_t
]
lib.encoder_layer_forward.restype = c_int

# Check version
version = lib.encoder_get_version().decode('utf-8')
print(f"   C++ encoder version: {version}")

# Whisper Base configuration
n_heads = 8
n_state = 512
ffn_dim = 2048
seq_len = 512

print(f"\n{'='*70}")
print("  INITIALIZING NPU RUNTIME")
print(f"{'='*70}")

try:
    npu_runtime = create_runtime(model_size="base", use_4tile=False)
    print(f"✅ NPU runtime created (32-tile)")

    npu_runtime._load_encoder_weights()
    print(f"✅ Whisper weights loaded")
except Exception as e:
    print(f"❌ Failed to initialize NPU: {e}")
    print(f"\nFalling back to CPU-only test...")
    npu_runtime = None

print(f"\n{'='*70}")
print("  CREATING C++ ENCODER (6 LAYERS)")
print(f"{'='*70}")

# Create 6 encoder layers
layers = []
for layer_idx in range(6):
    handle = lib.encoder_layer_create(layer_idx, n_heads, n_state, ffn_dim)
    if not handle:
        print(f"❌ Failed to create layer {layer_idx}")
        sys.exit(1)
    layers.append(handle)
    print(f"✅ Layer {layer_idx} created")

# Load real Whisper weights into each layer
if npu_runtime:
    print(f"\n{'='*70}")
    print("  LOADING REAL WHISPER WEIGHTS")
    print(f"{'='*70}")

    for layer_idx, handle in enumerate(layers):
        weights = npu_runtime.encoder_weights

        # Extract weights for this layer
        q_w = weights[f"layers.{layer_idx}.self_attn.q_proj.weight"].astype(np.float32)
        k_w = weights[f"layers.{layer_idx}.self_attn.k_proj.weight"].astype(np.float32)
        v_w = weights[f"layers.{layer_idx}.self_attn.v_proj.weight"].astype(np.float32)
        out_w = weights[f"layers.{layer_idx}.self_attn.out_proj.weight"].astype(np.float32)

        q_b = weights[f"layers.{layer_idx}.self_attn.q_proj.bias"].astype(np.float32)
        k_b = weights[f"layers.{layer_idx}.self_attn.k_proj.bias"].astype(np.float32)
        v_b = weights[f"layers.{layer_idx}.self_attn.v_proj.bias"].astype(np.float32)
        out_b = weights[f"layers.{layer_idx}.self_attn.out_proj.bias"].astype(np.float32)

        fc1_w = weights[f"layers.{layer_idx}.fc1.weight"].astype(np.float32)
        fc2_w = weights[f"layers.{layer_idx}.fc2.weight"].astype(np.float32)
        fc1_b = weights[f"layers.{layer_idx}.fc1.bias"].astype(np.float32)
        fc2_b = weights[f"layers.{layer_idx}.fc2.bias"].astype(np.float32)

        attn_ln_w = weights[f"layers.{layer_idx}.self_attn_layer_norm.weight"].astype(np.float32)
        attn_ln_b = weights[f"layers.{layer_idx}.self_attn_layer_norm.bias"].astype(np.float32)
        ffn_ln_w = weights[f"layers.{layer_idx}.final_layer_norm.weight"].astype(np.float32)
        ffn_ln_b = weights[f"layers.{layer_idx}.final_layer_norm.bias"].astype(np.float32)

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

# Prepare test input
print(f"\n{'='*70}")
print("  PREPARING TEST INPUT")
print(f"{'='*70}")

np.random.seed(42)
input_data = np.random.randn(seq_len, n_state).astype(np.float32)
output_data = np.zeros((seq_len, n_state), dtype=np.float32)

print(f"  Input: {input_data.shape}")

# Warmup
print(f"\n{'='*70}")
print("  WARMUP RUN")
print(f"{'='*70}")

current = input_data.copy()
for layer_idx, handle in enumerate(layers):
    result = lib.encoder_layer_forward(
        handle,
        current.ctypes.data_as(POINTER(c_float)),
        output_data.ctypes.data_as(POINTER(c_float)),
        seq_len,
        n_state
    )
    if result != 0:
        print(f"❌ Layer {layer_idx} forward failed")
        break
    current = output_data.copy()

print(f"✅ Warmup complete")

# Benchmark
print(f"\n{'='*70}")
print("  BENCHMARK (10 runs)")
print(f"{'='*70}")

num_runs = 10
times = []

for run in range(num_runs):
    current = input_data.copy()

    start = time.perf_counter()

    for layer_idx, handle in enumerate(layers):
        result = lib.encoder_layer_forward(
            handle,
            current.ctypes.data_as(POINTER(c_float)),
            output_data.ctypes.data_as(POINTER(c_float)),
            seq_len,
            n_state
        )
        if result != 0:
            print(f"❌ Run {run+1}, Layer {layer_idx} failed")
            break
        current = output_data.copy()

    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    print(f"  Run {run+1:2d}: {elapsed:7.2f} ms")

avg_time = np.mean(times)
min_time = np.min(times)
max_time = np.max(times)
std_time = np.std(times)

# Calculate realtime factor
audio_duration = 10.24  # seconds (3000 frames * 10ms + 2 frames)
realtime_factor = (audio_duration * 1000) / avg_time

print(f"\n{'='*70}")
print("  RESULTS")
print(f"{'='*70}")
print(f"\nC++ Encoder (CPU fallback):")
print(f"  Average: {avg_time:.2f} ms")
print(f"  Min:     {min_time:.2f} ms")
print(f"  Max:     {max_time:.2f} ms")
print(f"  Std Dev: {std_time:.2f} ms")
print(f"\nPerformance:")
print(f"  Audio duration: {audio_duration:.2f} seconds")
print(f"  Realtime factor: {realtime_factor:.2f}x")

print(f"\nComparison:")
print(f"  Python (Phase 4):  5.59x realtime (1,831 ms)")
print(f"  C++ (CPU):         {realtime_factor:.2f}x realtime ({avg_time:.0f} ms)")
print(f"  Speedup:           {1831 / avg_time:.2f}x")

if realtime_factor >= 17:
    print(f"\n🎉 TARGET ACHIEVED: {realtime_factor:.2f}x >= 17x!")
else:
    print(f"\n⏳ Below target: {realtime_factor:.2f}x < 17x")
    print(f"   (NPU integration needed for full speedup)")

print(f"\nOutput validation:")
print(f"  Shape: {output_data.shape}")
print(f"  Mean:  {output_data.mean():.4f}")
print(f"  Std:   {output_data.std():.4f}")
print(f"  Min:   {output_data.min():.4f}")
print(f"  Max:   {output_data.max():.4f}")

if np.isnan(output_data).any():
    print(f"  ❌ Contains NaN")
elif np.isinf(output_data).any():
    print(f"  ❌ Contains Inf")
else:
    print(f"  ✅ Valid output")

# Cleanup
for handle in layers:
    lib.encoder_layer_destroy(handle)

print(f"\n{'='*70}")
print("  ENCODER TEST COMPLETE")
print(f"{'='*70}")
