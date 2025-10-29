#!/usr/bin/env python3
"""
Quick test: What correlation do we ACTUALLY get with current XCLBINs?
Then we know if we need to rebuild or if there's another issue.
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
import librosa

print("="*70)
print("QUICK CORRELATION TEST")
print("="*70)

# Test with NEW fixed XCLBIN (with FFT scaling + HTK mel filters)
xclbin_path = "build_fixed_v3/mel_fixed_v3.xclbin"
print(f"\nTesting: {xclbin_path}")
print("This XCLBIN includes:")
print("  - FFT scaling fix (21:06 UTC)")
print("  - HTK mel filterbank (21:23 UTC)")
print("  - Expected correlation: >0.95")

# Load NPU (using custom runtime pattern)
device = xrt.device(0)
xclbin = xrt.xclbin(xclbin_path)  # Load XCLBIN from file
device.register_xclbin(xclbin)  # Register with device
uuid = xclbin.get_uuid()  # Get UUID
hw_ctx = xrt.hw_context(device, uuid)  # Create hardware context
kernel = xrt.kernel(hw_ctx, "MLIR_AIE")  # Get kernel

# Test audio: 1000 Hz sine
sr = 16000
audio = np.sin(2 * np.pi * 1000 * np.linspace(0, 1, sr, endpoint=False)).astype(np.float32)

# Get NPU output (1 frame = 400 samples)
audio_frame = audio[:400]
audio_int16 = (audio_frame * 32767).astype(np.int16)

# Load instruction binary
insts_bin = open("build_fixed_v3/insts_v3.bin", "rb").read()
n_insts = len(insts_bin)

# Create buffers (using correct API pattern)
input_size = 800  # 800 bytes (400 INT16 samples)
output_size = 80   # 80 INT8 bytes (80 mel bins)

instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
input_bo = xrt.bo(device, input_size, xrt.bo.flags.host_only, kernel.group_id(3))
output_bo = xrt.bo(device, output_size, xrt.bo.flags.host_only, kernel.group_id(4))

# Write instructions and input
instr_bo.write(insts_bin, 0)
instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

input_data = audio_int16.tobytes()
input_bo.write(input_data, 0)
input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, input_size, 0)

# Run kernel (with opcode)
opcode = 3
run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
state = run.wait(10000)  # 10 second timeout

if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
    print(f"❌ Kernel failed with state: {state}")
    import sys
    sys.exit(1)

# Read output
output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, output_size, 0)
npu_output = np.frombuffer(output_bo.read(output_size, 0), dtype=np.int8)

print(f"NPU output range: [{npu_output.min()}, {npu_output.max()}]")
print(f"NPU output mean: {npu_output.mean():.2f}")
print(f"Non-zero bins: {np.count_nonzero(npu_output)}/80")

# Get librosa reference
mel_ref = librosa.feature.melspectrogram(
    y=audio, sr=sr, n_fft=512, hop_length=160, n_mels=80,
    htk=True, power=2.0, fmin=0, fmax=8000
)
mel_ref_db = librosa.power_to_db(mel_ref, ref=np.max)
mel_ref_frame = mel_ref_db[:, 0]  # First frame

# Normalize both for comparison
def normalize(x):
    x = x.astype(np.float32)
    # Add small epsilon to avoid division by zero
    x_range = x.max() - x.min()
    if x_range < 1e-6:
        return np.zeros_like(x)
    return (x - x.min()) / x_range

# Also try converting NPU output to dB-like scale
npu_float = npu_output.astype(np.float32) + 1.0  # Add 1 to avoid log(0)
npu_db = 20 * np.log10(npu_float / npu_float.max())  # Convert to dB scale

npu_norm = normalize(npu_output.astype(np.float32))
npu_db_norm = normalize(npu_db)
ref_norm = normalize(mel_ref_frame)

# Calculate correlations
correlation_linear = np.corrcoef(npu_norm, ref_norm)[0, 1]
correlation_db = np.corrcoef(npu_db_norm, ref_norm)[0, 1]

print(f"\nLibrosa output range: [{mel_ref_frame.min():.2f}, {mel_ref_frame.max():.2f}]")
print(f"NPU output (dB scale): [{npu_db.min():.2f}, {npu_db.max():.2f}]")
print(f"\nCorrelation (linear): {correlation_linear:.4f}")
print(f"Correlation (dB scale): {correlation_db:.4f}")

# Use the better correlation
correlation = max(correlation_linear, correlation_db)

print("\n" + "="*70)
if correlation > 0.95:
    print("✅ EXCELLENT! Already working!")
elif correlation > 0.8:
    print("⚠️  GOOD but could be better")
elif correlation > 0.5:
    print("⚠️  MODERATE - needs investigation")
else:
    print("❌ LOW - kernel needs fixes compiled in")

print(f"\nConclusion: {'Fixes already in XCLBIN' if correlation > 0.9 else 'Need to recompile with fixes'}")
print("="*70)
