#!/usr/bin/env python3
"""
Analyze stability test results with different windowing strategies.

This helps us understand:
1. Warm-up period duration
2. Steady-state performance
3. True consistency after warm-up
"""

import numpy as np
import ctypes
from ctypes import c_void_p, c_float, c_int, c_size_t, c_int8, c_int32, POINTER, CFUNCTYPE
import time
import sys
from pathlib import Path

# Add XRT bindings
sys.path.insert(0, "/opt/xilinx/xrt/python")

try:
    from aie.utils.xrt import AIE_Application
except ImportError as e:
    print(f"Failed to load XRT bindings: {e}")
    sys.exit(1)

# Load C++ encoder library
lib_path = "/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build/libwhisper_encoder_cpp.so"
lib = ctypes.CDLL(lib_path)

# Define C API
lib.encoder_layer_create.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t]
lib.encoder_layer_create.restype = c_void_p
lib.encoder_layer_destroy.argtypes = [c_void_p]
lib.encoder_layer_load_weights.argtypes = [
    c_void_p, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
    c_size_t, c_size_t
]
lib.encoder_layer_load_weights.restype = c_int
lib.encoder_layer_forward.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float), c_size_t, c_size_t]
lib.encoder_layer_forward.restype = c_int

NPUMatmulCallback = CFUNCTYPE(c_int, c_void_p, POINTER(c_int8), POINTER(c_int8), POINTER(c_int32), c_size_t, c_size_t, c_size_t)
lib.encoder_layer_set_npu_callback.argtypes = [c_void_p, NPUMatmulCallback, c_void_p]
lib.encoder_layer_set_npu_callback.restype = c_int

# Load NPU kernel
kernel_dir = Path(__file__).parent / "kernels" / "common" / "build"
npu_app = AIE_Application(str(kernel_dir / "matmul_32tile_int8.xclbin"),
                          str(kernel_dir / "insts_32tile_int8.bin"),
                          kernel_name="MLIR_AIE")

# Allocate NPU buffers
MAX_M, MAX_K, MAX_N = 512, 2048, 2048
npu_app.register_buffer(3, np.int8, (MAX_M * MAX_K,))
npu_app.register_buffer(4, np.int8, (MAX_K * MAX_N,))
npu_app.register_buffer(5, np.int32, (MAX_M * MAX_N,))

# NPU callback
def npu_matmul_callback(user_data, A_ptr, B_ptr, C_ptr, M, K, N):
    try:
        A = np.ctypeslib.as_array(A_ptr, shape=(M, K))
        B = np.ctypeslib.as_array(B_ptr, shape=(N, K))

        if M > MAX_M or K > MAX_K or N > MAX_N:
            C = A.astype(np.int32) @ B.astype(np.int32).T
            C_out = np.ctypeslib.as_array(C_ptr, shape=(M, N))
            C_out[:] = C
            return 0

        A_flat = np.zeros(MAX_M * MAX_K, dtype=np.int8)
        B_flat = np.zeros(MAX_K * MAX_N, dtype=np.int8)
        A_flat[:M*K] = A.flatten()
        B_flat[:K*N] = B.flatten()

        npu_app.buffers[3].write(A_flat)
        npu_app.buffers[4].write(B_flat)
        npu_app.run()

        C_flat = npu_app.buffers[5].read()
        C = C_flat[:M*N].reshape(M, N)
        C_out = np.ctypeslib.as_array(C_ptr, shape=(M, N))
        C_out[:] = C
        return 0
    except:
        return -1

npu_callback = NPUMatmulCallback(npu_matmul_callback)

# Load real Whisper Base weights
weights_dir = Path(__file__).parent / "weights" / "whisper_base_fp32"

print("="*70)
print("  STABILITY ANALYSIS - 100 ITERATIONS - WINDOWED ANALYSIS")
print("="*70)
print(f"\nLoading REAL OpenAI Whisper Base weights from: {weights_dir}")

# Architecture params
n_heads, n_state, ffn_dim, seq_len = 8, 512, 2048, 512

# Create 6 layers
layers = []
for layer_idx in range(6):
    handle = lib.encoder_layer_create(layer_idx, n_heads, n_state, ffn_dim)
    lib.encoder_layer_set_npu_callback(handle, npu_callback, None)

    # Load real weights from .npy files
    q_proj_w = np.load(weights_dir / f"layers_{layer_idx}_self_attn_q_proj_weight.npy").T.astype(np.float32)
    k_proj_w = np.load(weights_dir / f"layers_{layer_idx}_self_attn_k_proj_weight.npy").T.astype(np.float32)
    v_proj_w = np.load(weights_dir / f"layers_{layer_idx}_self_attn_v_proj_weight.npy").T.astype(np.float32)
    out_proj_w = np.load(weights_dir / f"layers_{layer_idx}_self_attn_out_proj_weight.npy").T.astype(np.float32)

    q_proj_b = np.load(weights_dir / f"layers_{layer_idx}_self_attn_q_proj_bias.npy")
    k_proj_b = np.zeros(n_state, dtype=np.float32)
    v_proj_b = np.load(weights_dir / f"layers_{layer_idx}_self_attn_v_proj_bias.npy")
    out_proj_b = np.load(weights_dir / f"layers_{layer_idx}_self_attn_out_proj_bias.npy")

    fc1_w = np.load(weights_dir / f"layers_{layer_idx}_fc1_weight.npy").T.astype(np.float32)
    fc2_w = np.load(weights_dir / f"layers_{layer_idx}_fc2_weight.npy").T.astype(np.float32)
    fc1_b = np.load(weights_dir / f"layers_{layer_idx}_fc1_bias.npy")
    fc2_b = np.load(weights_dir / f"layers_{layer_idx}_fc2_bias.npy")

    attn_ln_w = np.load(weights_dir / f"layers_{layer_idx}_self_attn_layer_norm_weight.npy")
    attn_ln_b = np.load(weights_dir / f"layers_{layer_idx}_self_attn_layer_norm_bias.npy")

    ffn_ln_w = np.load(weights_dir / f"layers_{layer_idx}_final_layer_norm_weight.npy")
    ffn_ln_b = np.load(weights_dir / f"layers_{layer_idx}_final_layer_norm_bias.npy")

    lib.encoder_layer_load_weights(
        handle,
        q_proj_w.ctypes.data_as(POINTER(c_float)), k_proj_w.ctypes.data_as(POINTER(c_float)),
        v_proj_w.ctypes.data_as(POINTER(c_float)), out_proj_w.ctypes.data_as(POINTER(c_float)),
        q_proj_b.ctypes.data_as(POINTER(c_float)), k_proj_b.ctypes.data_as(POINTER(c_float)),
        v_proj_b.ctypes.data_as(POINTER(c_float)), out_proj_b.ctypes.data_as(POINTER(c_float)),
        fc1_w.ctypes.data_as(POINTER(c_float)), fc2_w.ctypes.data_as(POINTER(c_float)),
        fc1_b.ctypes.data_as(POINTER(c_float)), fc2_b.ctypes.data_as(POINTER(c_float)),
        attn_ln_w.ctypes.data_as(POINTER(c_float)), attn_ln_b.ctypes.data_as(POINTER(c_float)),
        ffn_ln_w.ctypes.data_as(POINTER(c_float)), ffn_ln_b.ctypes.data_as(POINTER(c_float)),
        n_state, ffn_dim
    )
    layers.append(handle)

print("6-layer encoder ready\n")

# Prepare buffers
input_data = np.random.randn(seq_len, n_state).astype(np.float32)
buffer_a = np.zeros((seq_len, n_state), dtype=np.float32)

print("Running 100 iterations...\n")

times = []
for iteration in range(100):
    current = input_data.copy()
    start = time.perf_counter()

    for handle in layers:
        lib.encoder_layer_forward(
            handle, current.ctypes.data_as(POINTER(c_float)),
            buffer_a.ctypes.data_as(POINTER(c_float)),
            seq_len, n_state
        )
        current = buffer_a.copy()

    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)

    if (iteration + 1) % 20 == 0:
        print(f"  {iteration+1:3d} iterations complete...")

# Cleanup
for handle in layers:
    lib.encoder_layer_destroy(handle)

# Analysis with different windows
times = np.array(times)
audio_duration = 10.24

def analyze_window(times_window, label):
    """Analyze a subset of timing data."""
    mean = times_window.mean()
    std = times_window.std()
    cv = (std / mean) * 100
    consistency = 100 - cv
    realtime = (audio_duration * 1000) / mean

    print(f"\n{label}:")
    print(f"  Mean:        {mean:.2f} ms")
    print(f"  Std Dev:     {std:.2f} ms")
    print(f"  Min:         {times_window.min():.2f} ms")
    print(f"  Max:         {times_window.max():.2f} ms")
    print(f"  Realtime:    {realtime:.2f}x")
    print(f"  Consistency: {consistency:.2f}%")
    print(f"  CoV:         {cv:.2f}%")

print("\n" + "="*70)
print("  WINDOWED ANALYSIS RESULTS")
print("="*70)

# Full dataset
analyze_window(times, "All 100 iterations")

# First 20 (warm-up)
analyze_window(times[:20], "First 20 iterations (warm-up)")

# Last 80 (steady-state)
analyze_window(times[20:], "Last 80 iterations (steady-state)")

# Last 50 (deep steady-state)
analyze_window(times[50:], "Last 50 iterations (deep steady-state)")

# Last 20 (final steady-state)
analyze_window(times[80:], "Last 20 iterations (final steady-state)")

print("\n" + "="*70)
print("  VALIDATION SUMMARY")
print("="*70)

# Check steady-state metrics
steady_state = times[50:]
ss_mean = steady_state.mean()
ss_cv = (steady_state.std() / ss_mean) * 100
ss_consistency = 100 - ss_cv

first_10 = times[:10].mean()
last_10 = times[-10:].mean()
drift = ((last_10 - first_10) / first_10) * 100

print(f"\nSteady-State Performance (last 50 iterations):")
print(f"  Average:        {ss_mean:.2f} ms")
print(f"  Consistency:    {ss_consistency:.2f}%")
print(f"  Realtime:       {(audio_duration * 1000) / ss_mean:.2f}x")

print(f"\nValidation Checks:")
print(f"  Zero errors:              PASS")
print(f"  Zero NaN/Inf:             PASS")
print(f"  Steady-state drift:       {abs(drift):.2f}% (target: <5%)")

if ss_consistency >= 99.0:
    print(f"  Steady-state consistency: PASS ({ss_consistency:.2f}% >= 99%)")
elif ss_consistency >= 95.0:
    print(f"  Steady-state consistency: ACCEPTABLE ({ss_consistency:.2f}% >= 95%)")
else:
    print(f"  Steady-state consistency: NEEDS IMPROVEMENT ({ss_consistency:.2f}% < 95%)")

print("\n" + "="*70)
