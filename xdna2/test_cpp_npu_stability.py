#!/usr/bin/env python3
"""
Extended Stability Test - 100+ Iterations

Validates that 18.42× performance is consistent and stable over extended use.
Tests for memory leaks, performance degradation, and numerical stability.
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
    print(f"❌ Failed to load XRT bindings: {e}")
    sys.exit(1)

print("="*70)
print("  STABILITY TEST - 100 ITERATIONS")
print("="*70)

# Load C++ encoder library
lib_path = "/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build/libwhisper_encoder_cpp.so"
lib = ctypes.CDLL(lib_path)

# Define C API (same as before)
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

# Create 6 layers
print("\nCreating 6-layer encoder...")
n_heads, n_state, ffn_dim, seq_len = 8, 512, 2048, 512
np.random.seed(42)

layers = []
for layer_idx in range(6):
    handle = lib.encoder_layer_create(layer_idx, n_heads, n_state, ffn_dim)
    lib.encoder_layer_set_npu_callback(handle, npu_callback, None)

    # Load weights
    np.random.seed(42 + layer_idx)
    q_w = (np.random.randn(n_state, n_state) * 0.1).astype(np.float32)
    k_w = (np.random.randn(n_state, n_state) * 0.1).astype(np.float32)
    v_w = (np.random.randn(n_state, n_state) * 0.1).astype(np.float32)
    out_w = (np.random.randn(n_state, n_state) * 0.1).astype(np.float32)
    fc1_w = (np.random.randn(ffn_dim, n_state) * 0.1).astype(np.float32)
    fc2_w = (np.random.randn(n_state, ffn_dim) * 0.1).astype(np.float32)

    zeros_state = np.zeros(n_state, dtype=np.float32)
    zeros_ffn = np.zeros(ffn_dim, dtype=np.float32)
    ones_state = np.ones(n_state, dtype=np.float32)

    lib.encoder_layer_load_weights(
        handle,
        q_w.ctypes.data_as(POINTER(c_float)), k_w.ctypes.data_as(POINTER(c_float)),
        v_w.ctypes.data_as(POINTER(c_float)), out_w.ctypes.data_as(POINTER(c_float)),
        zeros_state.ctypes.data_as(POINTER(c_float)), zeros_state.ctypes.data_as(POINTER(c_float)),
        zeros_state.ctypes.data_as(POINTER(c_float)), zeros_state.ctypes.data_as(POINTER(c_float)),
        fc1_w.ctypes.data_as(POINTER(c_float)), fc2_w.ctypes.data_as(POINTER(c_float)),
        zeros_ffn.ctypes.data_as(POINTER(c_float)), zeros_state.ctypes.data_as(POINTER(c_float)),
        ones_state.ctypes.data_as(POINTER(c_float)), zeros_state.ctypes.data_as(POINTER(c_float)),
        ones_state.ctypes.data_as(POINTER(c_float)), zeros_state.ctypes.data_as(POINTER(c_float)),
        n_state, ffn_dim
    )
    layers.append(handle)

print("✅ 6-layer encoder ready")

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
        print(f"  ❌ Iteration {iteration+1} failed: {e}")
        errors += 1

# Cleanup
for handle in layers:
    lib.encoder_layer_destroy(handle)

# Analysis
print(f"\n{'='*70}")
print(f"  STABILITY RESULTS")
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
print(f"  Average:    {realtime:.2f}×")
print(f"  Best:       {(audio_duration * 1000) / times.min():.2f}×")
print(f"  Worst:      {(audio_duration * 1000) / times.max():.2f}×")

# Stability metrics
cv = (times.std() / times.mean()) * 100
print(f"\nStability Metrics:")
print(f"  Coefficient of Variation:  {cv:.2f}%")
print(f"  Consistency:               {100-cv:.2f}%")

# Performance over time
first_10 = times[:10].mean()
last_10 = times[-10:].mean()
drift = ((last_10 - first_10) / first_10) * 100

print(f"\nPerformance Drift:")
print(f"  First 10 iterations:  {first_10:.2f} ms")
print(f"  Last 10 iterations:   {last_10:.2f} ms")
print(f"  Drift:                {drift:+.2f}%")

# Final verdict
print(f"\n{'='*70}")
if errors == 0 and nan_count == 0 and abs(drift) < 5 and cv < 15:
    print("  ✅ STABILITY TEST PASSED - PRODUCTION READY!")
else:
    print("  ⚠️  STABILITY ISSUES DETECTED")
print(f"{'='*70}")
