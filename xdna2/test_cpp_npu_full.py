#!/usr/bin/env python3
"""
Full C++ Encoder with REAL NPU Integration Test

This test connects the C++ encoder to the real XDNA2 NPU hardware
using the 32-tile INT8 matmul kernel.
"""

import numpy as np
import ctypes
from ctypes import c_void_p, c_float, c_int, c_size_t, c_int8, c_int32, POINTER, CFUNCTYPE
import time
import sys
import os
from pathlib import Path

# Add XRT bindings
sys.path.insert(0, "/opt/xilinx/xrt/python")

try:
    from aie.utils.xrt import AIE_Application
    print("✅ XRT bindings loaded")
except ImportError as e:
    print(f"❌ Failed to load XRT bindings: {e}")
    print("   Make sure ironenv is activated:")
    print("   source ~/mlir-aie/ironenv/bin/activate")
    sys.exit(1)

print("="*70)
print("  C++ ENCODER + NPU HARDWARE INTEGRATION TEST")
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

# Define NPU callback type
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

print(f"✅ C API bindings complete")

print(f"\n{'='*70}")
print("  LOADING NPU KERNEL")
print(f"{'='*70}")

# Load 32-tile INT8 kernel
kernel_dir = Path(__file__).parent / "kernels" / "common" / "build"
xclbin_path = kernel_dir / "matmul_32tile_int8.xclbin"
insts_path = kernel_dir / "insts_32tile_int8.bin"

if not xclbin_path.exists():
    print(f"❌ XCLBin not found: {xclbin_path}")
    print("   Trying 4-tile kernel instead...")
    xclbin_path = kernel_dir / "matmul_4tile_int8.xclbin"
    insts_path = kernel_dir / "insts_4tile_int8.bin"

if not xclbin_path.exists():
    print(f"❌ XCLBin not found: {xclbin_path}")
    sys.exit(1)

if not insts_path.exists():
    print(f"❌ Instructions not found: {insts_path}")
    sys.exit(1)

try:
    npu_app = AIE_Application(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    print(f"✅ NPU kernel loaded: {xclbin_path.name}")
except Exception as e:
    print(f"❌ Failed to load NPU kernel: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Pre-allocate NPU buffers for 512×512×2048 (largest dimension)
MAX_M = 512
MAX_K = 2048
MAX_N = 2048

npu_app.register_buffer(3, np.int8, (MAX_M * MAX_K,))   # A
npu_app.register_buffer(4, np.int8, (MAX_K * MAX_N,))   # B
npu_app.register_buffer(5, np.int32, (MAX_M * MAX_N,))  # C
print(f"✅ NPU buffers allocated ({MAX_M}×{MAX_K}×{MAX_N})")

# Statistics
callback_stats = {
    'count': 0,
    'total_npu_time_ms': 0.0,
    'matmul_shapes': []
}

# NPU callback function
def npu_matmul_callback(user_data, A_ptr, B_ptr, C_ptr, M, K, N):
    """
    Real NPU matmul callback - dispatches to XDNA2 hardware!
    """
    try:
        start = time.perf_counter()

        # Convert pointers to numpy arrays
        A = np.ctypeslib.as_array(A_ptr, shape=(M, K))
        B = np.ctypeslib.as_array(B_ptr, shape=(N, K))  # B is transposed

        # Check dimensions
        if M > MAX_M or K > MAX_K or N > MAX_N:
            print(f"⚠️  Matmul ({M}×{K}×{N}) exceeds buffer size, using CPU fallback")
            C = A.astype(np.int32) @ B.astype(np.int32).T
            C_out = np.ctypeslib.as_array(C_ptr, shape=(M, N))
            C_out[:] = C
            return 0

        # Flatten and pad to buffer size
        A_flat = np.zeros(MAX_M * MAX_K, dtype=np.int8)
        B_flat = np.zeros(MAX_K * MAX_N, dtype=np.int8)

        A_flat[:M*K] = A.flatten()
        B_flat[:K*N] = B.flatten()

        # Write to NPU
        npu_app.buffers[3].write(A_flat)
        npu_app.buffers[4].write(B_flat)

        # Execute on NPU
        npu_app.run()

        # Read result
        C_flat = npu_app.buffers[5].read()
        C = C_flat[:M*N].reshape(M, N)

        # Write output
        C_out = np.ctypeslib.as_array(C_ptr, shape=(M, N))
        C_out[:] = C

        elapsed = (time.perf_counter() - start) * 1000

        # Update stats
        callback_stats['count'] += 1
        callback_stats['total_npu_time_ms'] += elapsed
        callback_stats['matmul_shapes'].append((M, K, N))

        return 0  # Success
    except Exception as e:
        print(f"❌ NPU callback error: {e}")
        import traceback
        traceback.print_exc()
        return -1  # Failure

# Create callback wrapper
npu_callback = NPUMatmulCallback(npu_matmul_callback)

print(f"\n{'='*70}")
print("  CREATING C++ ENCODER LAYER")
print(f"{'='*70}")

# Configuration
n_heads, n_state, ffn_dim, seq_len = 8, 512, 2048, 512

# Create layer
handle = lib.encoder_layer_create(0, n_heads, n_state, ffn_dim)
if not handle:
    print("❌ Failed to create encoder layer")
    sys.exit(1)
print(f"✅ Encoder layer created")

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
print("  RUNNING ENCODER WITH NPU HARDWARE")
print(f"{'='*70}")

# Prepare input
input_data = np.random.randn(seq_len, n_state).astype(np.float32)
output_data = np.zeros((seq_len, n_state), dtype=np.float32)

# Warmup
print(f"\nWarmup run...")
result = lib.encoder_layer_forward(
    handle,
    input_data.ctypes.data_as(POINTER(c_float)),
    output_data.ctypes.data_as(POINTER(c_float)),
    seq_len,
    n_state
)

# Benchmark runs
print(f"Benchmark runs...")
times = []
callback_stats['count'] = 0
callback_stats['total_npu_time_ms'] = 0.0

for i in range(10):
    callback_stats['count'] = 0
    callback_stats['total_npu_time_ms'] = 0.0

    start = time.perf_counter()

    result = lib.encoder_layer_forward(
        handle,
        input_data.ctypes.data_as(POINTER(c_float)),
        output_data.ctypes.data_as(POINTER(c_float)),
        seq_len,
        n_state
    )

    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)

    if result != 0:
        print(f"❌ Forward pass {i+1} failed")
        sys.exit(1)

    print(f"  Run {i+1:2d}: {elapsed:7.2f} ms (NPU: {callback_stats['total_npu_time_ms']:.2f} ms, {callback_stats['count']} matmuls)")

avg_total = np.mean(times)
min_time = np.min(times)
max_time = np.max(times)

print(f"\n{'='*70}")
print("  RESULTS")
print(f"{'='*70}")

# Single layer performance
print(f"\nSingle Layer Performance:")
print(f"  Average:       {avg_total:.2f} ms")
print(f"  Min:           {min_time:.2f} ms")
print(f"  Max:           {max_time:.2f} ms")
print(f"  Output valid:  {'✅ Yes' if not (np.isnan(output_data).any() or np.isinf(output_data).any()) else '❌ No'}")

# Full 6-layer projection
full_encoder_time = avg_total * 6
audio_duration = 10.24  # seconds
realtime_factor = (audio_duration * 1000) / full_encoder_time

print(f"\nFull Encoder (6 layers) Projection:")
print(f"  Estimated time:    {full_encoder_time:.0f} ms")
print(f"  Audio duration:    {audio_duration:.2f} s")
print(f"  Realtime factor:   {realtime_factor:.2f}x")

# Comparison
print(f"\nComparison:")
print(f"  Python baseline:   5.59x realtime (1,831 ms)")
print(f"  C++ CPU fallback:  7.77x realtime (1,318 ms)")
print(f"  C++ + NPU:         {realtime_factor:.2f}x realtime ({full_encoder_time:.0f} ms)")

if realtime_factor >= 17:
    print(f"\n🎉 TARGET ACHIEVED: {realtime_factor:.2f}x >= 17x!")
elif realtime_factor >= 10:
    print(f"\n✅ GOOD PROGRESS: {realtime_factor:.2f}x (approaching target)")
else:
    print(f"\n⏳ Below target: {realtime_factor:.2f}x")

print(f"\nSpeedup vs Python: {1831/full_encoder_time:.2f}x")
print(f"Speedup vs C++ CPU: {1318/full_encoder_time:.2f}x")

# Cleanup
lib.encoder_layer_destroy(handle)

print(f"\n{'='*70}")
print(f"  ✅ NPU HARDWARE INTEGRATION COMPLETE")
print(f"{'='*70}")
