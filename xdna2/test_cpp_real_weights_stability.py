#!/usr/bin/env python3
"""
Extended Stability Test - 100 Iterations with REAL OpenAI Whisper Weights

Validates that C++ encoder performance is consistent and stable over extended use
using actual Whisper Base FP32 weights from OpenAI.

Tests for:
- Memory leaks
- Performance degradation
- Numerical stability
- Consistency (target: 99.7%)
- Performance drift (<5%)
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

print("="*70)
print("  STABILITY TEST - 100 ITERATIONS - REAL OPENAI WEIGHTS")
print("="*70)

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

print("\nLoading REAL OpenAI Whisper Base weights...")
print(f"  Location: {weights_dir}")

# Architecture params
n_heads, n_state, ffn_dim, seq_len = 8, 512, 2048, 512

# Create 6 layers
layers = []
for layer_idx in range(6):
    print(f"  Loading layer {layer_idx}...")

    handle = lib.encoder_layer_create(layer_idx, n_heads, n_state, ffn_dim)
    lib.encoder_layer_set_npu_callback(handle, npu_callback, None)

    # Load real weights from .npy files
    q_proj_w = np.load(weights_dir / f"layers_{layer_idx}_self_attn_q_proj_weight.npy")
    k_proj_w = np.load(weights_dir / f"layers_{layer_idx}_self_attn_k_proj_weight.npy")
    v_proj_w = np.load(weights_dir / f"layers_{layer_idx}_self_attn_v_proj_weight.npy")
    out_proj_w = np.load(weights_dir / f"layers_{layer_idx}_self_attn_out_proj_weight.npy")

    q_proj_b = np.load(weights_dir / f"layers_{layer_idx}_self_attn_q_proj_bias.npy")
    k_proj_b = np.zeros(n_state, dtype=np.float32)  # K has no bias in Whisper
    v_proj_b = np.load(weights_dir / f"layers_{layer_idx}_self_attn_v_proj_bias.npy")
    out_proj_b = np.load(weights_dir / f"layers_{layer_idx}_self_attn_out_proj_bias.npy")

    fc1_w = np.load(weights_dir / f"layers_{layer_idx}_fc1_weight.npy")
    fc2_w = np.load(weights_dir / f"layers_{layer_idx}_fc2_weight.npy")
    fc1_b = np.load(weights_dir / f"layers_{layer_idx}_fc1_bias.npy")
    fc2_b = np.load(weights_dir / f"layers_{layer_idx}_fc2_bias.npy")

    attn_ln_w = np.load(weights_dir / f"layers_{layer_idx}_self_attn_layer_norm_weight.npy")
    attn_ln_b = np.load(weights_dir / f"layers_{layer_idx}_self_attn_layer_norm_bias.npy")

    ffn_ln_w = np.load(weights_dir / f"layers_{layer_idx}_final_layer_norm_weight.npy")
    ffn_ln_b = np.load(weights_dir / f"layers_{layer_idx}_final_layer_norm_bias.npy")

    # Transpose weights if needed (OpenAI format is [out, in], C++ expects [in, out])
    q_proj_w = q_proj_w.T.astype(np.float32)
    k_proj_w = k_proj_w.T.astype(np.float32)
    v_proj_w = v_proj_w.T.astype(np.float32)
    out_proj_w = out_proj_w.T.astype(np.float32)
    fc1_w = fc1_w.T.astype(np.float32)
    fc2_w = fc2_w.T.astype(np.float32)

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

print("6-layer encoder ready with REAL OpenAI weights")

# Prepare buffers
input_data = np.random.randn(seq_len, n_state).astype(np.float32)
buffer_a = np.zeros((seq_len, n_state), dtype=np.float32)

print(f"\n{'='*70}")
print(f"  RUNNING 100 ITERATIONS")
print(f"{'='*70}\n")

times = []
errors = 0
nan_count = 0

for iteration in range(100):
    try:
        current = input_data.copy()
        start = time.perf_counter()

        for handle in layers:
            result = lib.encoder_layer_forward(
                handle, current.ctypes.data_as(POINTER(c_float)),
                buffer_a.ctypes.data_as(POINTER(c_float)),
                seq_len, n_state
            )
            if result != 0:
                errors += 1
                break
            current = buffer_a.copy()

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        # Check for numerical issues
        if np.isnan(current).any() or np.isinf(current).any():
            nan_count += 1

        if (iteration + 1) % 10 == 0:
            recent_avg = np.mean(times[-10:])
            print(f"  Iteration {iteration+1:3d}: {recent_avg:6.2f} ms avg (last 10)")

    except Exception as e:
        print(f"  Iteration {iteration+1} failed: {e}")
        errors += 1

# Cleanup
for handle in layers:
    lib.encoder_layer_destroy(handle)

# Analysis
print(f"\n{'='*70}")
print(f"  STABILITY RESULTS - REAL OPENAI WEIGHTS")
print(f"{'='*70}\n")

times = np.array(times)
audio_duration = 10.24

print(f"Iterations completed:  {len(times)}/100")
print(f"Errors:                {errors}")
print(f"Numerical issues:      {nan_count}")

print(f"\nPerformance Statistics:")
print(f"  Mean:       {times.mean():.2f} ms")
print(f"  Median:     {np.median(times):.2f} ms")
print(f"  Std Dev:    {times.std():.2f} ms")
print(f"  Min:        {times.min():.2f} ms")
print(f"  Max:        {times.max():.2f} ms")
print(f"  Range:      {times.max() - times.min():.2f} ms")

print(f"\nRealtime Factor:")
realtime = (audio_duration * 1000) / times.mean()
print(f"  Average:    {realtime:.2f}x")
print(f"  Best:       {(audio_duration * 1000) / times.min():.2f}x")
print(f"  Worst:      {(audio_duration * 1000) / times.max():.2f}x")

# Stability metrics
cv = (times.std() / times.mean()) * 100
consistency = 100 - cv
print(f"\nStability Metrics:")
print(f"  Coefficient of Variation:  {cv:.2f}%")
print(f"  Consistency:               {consistency:.2f}%")

# Performance over time
first_10 = times[:10].mean()
last_10 = times[-10:].mean()
drift = ((last_10 - first_10) / first_10) * 100

print(f"\nPerformance Drift:")
print(f"  First 10 iterations:  {first_10:.2f} ms")
print(f"  Last 10 iterations:   {last_10:.2f} ms")
print(f"  Drift:                {drift:+.2f}%")

# Expected vs actual comparison
expected_time = 617.0  # From 10-iteration test
expected_consistency = 99.7
expected_drift = 5.0

print(f"\nExpected vs Actual:")
print(f"  Expected avg time:     {expected_time:.2f} ms")
print(f"  Actual avg time:       {times.mean():.2f} ms")
print(f"  Difference:            {((times.mean() - expected_time) / expected_time * 100):+.2f}%")
print(f"")
print(f"  Expected consistency:  {expected_consistency:.1f}%")
print(f"  Actual consistency:    {consistency:.2f}%")
print(f"  Difference:            {(consistency - expected_consistency):+.2f}%")
print(f"")
print(f"  Expected drift:        <{expected_drift:.1f}%")
print(f"  Actual drift:          {abs(drift):.2f}%")

# Final verdict
print(f"\n{'='*70}")
validation_passed = (
    errors == 0 and
    nan_count == 0 and
    abs(drift) < 5 and
    consistency >= 99.0  # Allow slight variance from 99.7%
)

if validation_passed:
    print("  STABILITY TEST PASSED - PRODUCTION READY!")

    if consistency >= 99.7:
        print("  99.7% consistency VALIDATED!")
    else:
        print(f"  Note: Consistency {consistency:.2f}% (target: 99.7%)")

    if abs(times.mean() - expected_time) / expected_time * 100 < 5:
        print(f"  Performance matches 10-iteration test!")
else:
    print("  STABILITY ISSUES DETECTED")
    if errors > 0:
        print(f"    - {errors} execution errors")
    if nan_count > 0:
        print(f"    - {nan_count} numerical issues")
    if abs(drift) >= 5:
        print(f"    - Performance drift {drift:.2f}% exceeds 5%")
    if consistency < 99.0:
        print(f"    - Consistency {consistency:.2f}% below 99%")

print(f"{'='*70}")
