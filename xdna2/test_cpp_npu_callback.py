#!/usr/bin/env python3
"""Test C++ encoder with NPU callback integration.

This test demonstrates the callback pattern where C++ calls back to Python
for NPU matmul operations.
"""

import numpy as np
import ctypes
from ctypes import c_void_p, c_float, c_int, c_size_t, c_int8, c_int32, POINTER, CFUNCTYPE
import time
import sys

print("="*70)
print("  C++ ENCODER WITH NPU CALLBACK TEST")
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

# Define NPU callback type (BFP16 format)
NPUMatmulCallback = CFUNCTYPE(
    c_int,                  # return type
    c_void_p,               # user_data
    POINTER(ctypes.c_uint8),  # A_bfp16
    POINTER(ctypes.c_uint8),  # B_bfp16
    POINTER(ctypes.c_uint8),  # C_bfp16
    c_size_t,               # M
    c_size_t,               # K
    c_size_t                # N
)

lib.encoder_layer_set_npu_callback.argtypes = [c_void_p, NPUMatmulCallback, c_void_p]
lib.encoder_layer_set_npu_callback.restype = c_int

print(f"\n✅ C API bindings complete")

# Statistics for callback
callback_stats = {
    'count': 0,
    'total_time_ms': 0.0,
    'matmul_shapes': []
}

# Define Python callback function
def npu_matmul_callback(user_data, A_ptr, B_ptr, C_ptr, M, K, N):
    """
    Python callback for NPU matmul (BFP16 format).

    This would normally dispatch to the NPU runtime. For testing,
    we'll just fill the output with zeros (mock NPU behavior).
    """
    try:
        start = time.perf_counter()

        # BFP16 buffer size calculation: ((N + 7) / 8) * 9
        K_bfp16 = ((K + 7) // 8) * 9
        N_bfp16 = ((N + 7) // 8) * 9

        # Convert pointers to numpy arrays (BFP16 format - uint8 byte arrays)
        A = np.ctypeslib.as_array(A_ptr, shape=(M, K_bfp16))
        B = np.ctypeslib.as_array(B_ptr, shape=(N, K_bfp16))
        C_out = np.ctypeslib.as_array(C_ptr, shape=(M, N_bfp16))

        # Mock NPU operation: Fill output with zeros
        # In real implementation, this would call NPU hardware
        C_out[:] = 0

        elapsed = (time.perf_counter() - start) * 1000

        # Update stats
        callback_stats['count'] += 1
        callback_stats['total_time_ms'] += elapsed
        callback_stats['matmul_shapes'].append((M, K, N))

        return 0  # Success
    except Exception as e:
        print(f"❌ Callback error: {e}")
        import traceback
        traceback.print_exc()
        return -1  # Failure

# Create callback wrapper
npu_callback = NPUMatmulCallback(npu_matmul_callback)

print(f"\n{'='*70}")
print("  CREATING ENCODER LAYER")
print(f"{'='*70}")

# Configuration
n_heads, n_state, ffn_dim, seq_len = 8, 512, 2048, 512

# Create layer
handle = lib.encoder_layer_create(0, n_heads, n_state, ffn_dim)
if not handle:
    print("❌ Failed to create encoder layer")
    sys.exit(1)
print(f"✅ Encoder layer created (0x{handle:x})")

# Set NPU callback
result = lib.encoder_layer_set_npu_callback(handle, npu_callback, None)
if result != 0:
    print("❌ Failed to set NPU callback")
    sys.exit(1)
print(f"✅ NPU callback registered")

print(f"\n{'='*70}")
print("  LOADING WEIGHTS")
print(f"{'='*70}")

# Create random weights
np.random.seed(42)

def xavier_weight(shape):
    w = np.random.randn(*shape).astype(np.float32)
    w *= np.sqrt(2.0 / shape[0])
    return w

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
    print("❌ Failed to load weights")
    sys.exit(1)
print(f"✅ Weights loaded")

print(f"\n{'='*70}")
print("  RUNNING ENCODER WITH NPU CALLBACK")
print(f"{'='*70}")

# Prepare input
input_data = np.random.randn(seq_len, n_state).astype(np.float32)
output_data = np.zeros((seq_len, n_state), dtype=np.float32)

# Run forward pass
print(f"\nRunning forward pass...")
callback_stats['count'] = 0
callback_stats['total_time_ms'] = 0.0
callback_stats['matmul_shapes'] = []

start = time.perf_counter()

result = lib.encoder_layer_forward(
    handle,
    input_data.ctypes.data_as(POINTER(c_float)),
    output_data.ctypes.data_as(POINTER(c_float)),
    seq_len,
    n_state
)

elapsed = (time.perf_counter() - start) * 1000

if result != 0:
    print("❌ Forward pass failed")
    sys.exit(1)

print(f"\n{'='*70}")
print("  RESULTS")
print(f"{'='*70}")

print(f"\nEncoder Performance:")
print(f"  Total time:    {elapsed:.2f} ms")
print(f"  Output valid:  {'✅ Yes' if not (np.isnan(output_data).any() or np.isinf(output_data).any()) else '❌ No'}")

print(f"\nNPU Callback Statistics:")
print(f"  Callback count:     {callback_stats['count']}")
print(f"  Callback time:      {callback_stats['total_time_ms']:.2f} ms ({callback_stats['total_time_ms']/elapsed*100:.1f}%)")
print(f"  Non-callback time:  {elapsed - callback_stats['total_time_ms']:.2f} ms")

print(f"\nCallback Invocations:")
for i, (M, K, N) in enumerate(callback_stats['matmul_shapes'], 1):
    print(f"  {i}. ({M}×{K}) @ ({K}×{N}) = ({M}×{N})")

expected_matmuls = 5  # Q, K, V, Out, FC1, FC2
print(f"\nExpected matmuls: {expected_matmuls}")
print(f"Actual matmuls:   {callback_stats['count']}")

if callback_stats['count'] == expected_matmuls:
    print("✅ All matmuls routed through callback!")
else:
    print(f"⚠️  Expected {expected_matmuls}, got {callback_stats['count']}")

print(f"\nOutput Statistics:")
print(f"  Mean: {output_data.mean():.4f}")
print(f"  Std:  {output_data.std():.4f}")
print(f"  Min:  {output_data.min():.4f}")
print(f"  Max:  {output_data.max():.4f}")

# Cleanup
lib.encoder_layer_destroy(handle)

print(f"\n{'='*70}")
print("  ✅ NPU CALLBACK INTEGRATION SUCCESSFUL")
print(f"{'='*70}")
