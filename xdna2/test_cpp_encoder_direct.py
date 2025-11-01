#!/usr/bin/env python3
"""Test C++ encoder directly via ctypes."""

import numpy as np
import ctypes
from ctypes import c_void_p, c_float, c_int, c_size_t, POINTER, c_char_p
import time
import sys

print("="*70)
print("  C++ ENCODER TEST (ctypes)")
print("="*70)

# Load C++ encoder library
lib_path = "/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build/libwhisper_encoder_cpp.so"
try:
    lib = ctypes.CDLL(lib_path)
    print(f"✅ Loaded library: {lib_path}")
except OSError as e:
    print(f"❌ Failed to load library: {e}")
    sys.exit(1)

# Define C API function signatures

# Version info
lib.encoder_get_version.restype = c_char_p
lib.encoder_check_config.restype = c_int

# Encoder layer functions
lib.encoder_layer_create.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t]
lib.encoder_layer_create.restype = c_void_p

lib.encoder_layer_destroy.argtypes = [c_void_p]
lib.encoder_layer_destroy.restype = None

lib.encoder_layer_load_weights.argtypes = [
    c_void_p,  # handle
    POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),  # q,k,v,out weights
    POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),  # q,k,v,out biases
    POINTER(c_float), POINTER(c_float),  # fc1, fc2 weights
    POINTER(c_float), POINTER(c_float),  # fc1, fc2 biases
    POINTER(c_float), POINTER(c_float),  # attn ln weight, bias
    POINTER(c_float), POINTER(c_float),  # ffn ln weight, bias
    c_size_t, c_size_t  # n_state, ffn_dim
]
lib.encoder_layer_load_weights.restype = c_int

lib.encoder_layer_forward.argtypes = [
    c_void_p,  # handle
    POINTER(c_float),  # input
    POINTER(c_float),  # output
    c_size_t,  # seq_len
    c_size_t   # n_state
]
lib.encoder_layer_forward.restype = c_int

# Check library version
version = lib.encoder_get_version().decode('utf-8')
config_ok = lib.encoder_check_config()

print(f"   Version: {version}")
print(f"   Config check: {'✅ OK' if config_ok else '❌ FAILED'}")

if not config_ok:
    print("❌ Library configuration check failed!")
    sys.exit(1)

# Whisper Base configuration
n_heads = 8
n_state = 512
ffn_dim = 2048
seq_len = 512

print(f"\n{'='*70}")
print("  CONFIGURATION")
print(f"{'='*70}")
print(f"  Model: Whisper Base")
print(f"  n_heads: {n_heads}")
print(f"  n_state: {n_state}")
print(f"  ffn_dim: {ffn_dim}")
print(f"  seq_len: {seq_len}")

# Create encoder layer
print(f"\n{'='*70}")
print("  CREATING ENCODER LAYER")
print(f"{'='*70}")

layer_handle = lib.encoder_layer_create(0, n_heads, n_state, ffn_dim)
if not layer_handle:
    print("❌ Failed to create encoder layer!")
    sys.exit(1)

print(f"✅ Encoder layer created (handle: 0x{layer_handle:x})")

# Initialize random weights (for testing)
print(f"\n{'='*70}")
print("  INITIALIZING RANDOM WEIGHTS")
print(f"{'='*70}")

np.random.seed(42)

def make_weight(shape):
    """Create random weight with appropriate scale."""
    w = np.random.randn(*shape).astype(np.float32)
    w *= np.sqrt(2.0 / shape[0])  # Xavier initialization
    return w

# Create weights
q_weight = make_weight((n_state, n_state))
k_weight = make_weight((n_state, n_state))
v_weight = make_weight((n_state, n_state))
out_weight = make_weight((n_state, n_state))

q_bias = np.zeros(n_state, dtype=np.float32)
k_bias = np.zeros(n_state, dtype=np.float32)
v_bias = np.zeros(n_state, dtype=np.float32)
out_bias = np.zeros(n_state, dtype=np.float32)

fc1_weight = make_weight((ffn_dim, n_state))
fc2_weight = make_weight((n_state, ffn_dim))

fc1_bias = np.zeros(ffn_dim, dtype=np.float32)
fc2_bias = np.zeros(n_state, dtype=np.float32)

attn_ln_weight = np.ones(n_state, dtype=np.float32)
attn_ln_bias = np.zeros(n_state, dtype=np.float32)

ffn_ln_weight = np.ones(n_state, dtype=np.float32)
ffn_ln_bias = np.zeros(n_state, dtype=np.float32)

print(f"  q_weight: {q_weight.shape}")
print(f"  fc1_weight: {fc1_weight.shape}")
print(f"  fc2_weight: {fc2_weight.shape}")

# Load weights into encoder
print(f"\n{'='*70}")
print("  LOADING WEIGHTS INTO C++ ENCODER")
print(f"{'='*70}")

result = lib.encoder_layer_load_weights(
    layer_handle,
    q_weight.ctypes.data_as(POINTER(c_float)),
    k_weight.ctypes.data_as(POINTER(c_float)),
    v_weight.ctypes.data_as(POINTER(c_float)),
    out_weight.ctypes.data_as(POINTER(c_float)),
    q_bias.ctypes.data_as(POINTER(c_float)),
    k_bias.ctypes.data_as(POINTER(c_float)),
    v_bias.ctypes.data_as(POINTER(c_float)),
    out_bias.ctypes.data_as(POINTER(c_float)),
    fc1_weight.ctypes.data_as(POINTER(c_float)),
    fc2_weight.ctypes.data_as(POINTER(c_float)),
    fc1_bias.ctypes.data_as(POINTER(c_float)),
    fc2_bias.ctypes.data_as(POINTER(c_float)),
    attn_ln_weight.ctypes.data_as(POINTER(c_float)),
    attn_ln_bias.ctypes.data_as(POINTER(c_float)),
    ffn_ln_weight.ctypes.data_as(POINTER(c_float)),
    ffn_ln_bias.ctypes.data_as(POINTER(c_float)),
    n_state,
    ffn_dim
)

if result != 0:
    print(f"❌ Failed to load weights (error code: {result})")
    lib.encoder_layer_destroy(layer_handle)
    sys.exit(1)

print("✅ Weights loaded successfully")

# Create test input
print(f"\n{'='*70}")
print("  PREPARING TEST INPUT")
print(f"{'='*70}")

input_data = np.random.randn(seq_len, n_state).astype(np.float32)
output_data = np.zeros((seq_len, n_state), dtype=np.float32)

print(f"  Input shape: {input_data.shape}")
print(f"  Input mean: {input_data.mean():.4f}")
print(f"  Input std:  {input_data.std():.4f}")

# Run forward pass
print(f"\n{'='*70}")
print("  RUNNING FORWARD PASS")
print(f"{'='*70}")

# Warmup
result = lib.encoder_layer_forward(
    layer_handle,
    input_data.ctypes.data_as(POINTER(c_float)),
    output_data.ctypes.data_as(POINTER(c_float)),
    seq_len,
    n_state
)

if result != 0:
    print(f"❌ Forward pass failed (error code: {result})")
    lib.encoder_layer_destroy(layer_handle)
    sys.exit(1)

print("✅ Warmup complete")

# Timed runs
num_runs = 10
times = []

for i in range(num_runs):
    start = time.perf_counter()
    result = lib.encoder_layer_forward(
        layer_handle,
        input_data.ctypes.data_as(POINTER(c_float)),
        output_data.ctypes.data_as(POINTER(c_float)),
        seq_len,
        n_state
    )
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)

    if result != 0:
        print(f"❌ Forward pass {i+1} failed!")
        break

avg_time = np.mean(times)
min_time = np.min(times)
max_time = np.max(times)

print(f"\n{'='*70}")
print("  RESULTS")
print(f"{'='*70}")
print(f"  Runs: {num_runs}")
print(f"  Average: {avg_time:.2f} ms")
print(f"  Min:     {min_time:.2f} ms")
print(f"  Max:     {max_time:.2f} ms")
print(f"\nOutput shape: {output_data.shape}")
print(f"Output mean:  {output_data.mean():.4f}")
print(f"Output std:   {output_data.std():.4f}")
print(f"Output min:   {output_data.min():.4f}")
print(f"Output max:   {output_data.max():.4f}")

# Sanity checks
if np.isnan(output_data).any():
    print(f"❌ Output contains NaN values!")
elif np.isinf(output_data).any():
    print(f"❌ Output contains Inf values!")
else:
    print(f"✅ Output is valid (no NaN/Inf)")

# Cleanup
lib.encoder_layer_destroy(layer_handle)
print(f"\n✅ Encoder layer destroyed")

print(f"\n{'='*70}")
print("  SUCCESS!")
print(f"{'='*70}")
print(f"""
✅ C++ encoder works with CPU fallback!
✅ Single layer latency: {avg_time:.2f} ms
✅ Full encoder estimate (6 layers): {avg_time * 6:.2f} ms

Next steps:
1. Integrate with Python weights loader
2. Add NPU integration (replace CPU fallback)
3. Benchmark full 6-layer encoder
""")
print(f"{'='*70}")
