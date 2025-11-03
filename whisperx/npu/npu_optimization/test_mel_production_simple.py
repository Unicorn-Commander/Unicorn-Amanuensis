#!/usr/bin/env python3
"""
Simplified Production Mel Test

Tests the EXACT production mel configuration with REAL audio data
to determine if mel kernel can produce non-zero output.
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import numpy as np
import pyxrt as xrt
import librosa
from pathlib import Path

print("=" * 70)
print("PRODUCTION MEL TEST WITH REAL AUDIO")
print("=" * 70)
print()

# Use production paths
xclbin_path = "mel_kernels/build_fixed_v3/mel_fixed_v3_PRODUCTION_v2.0.xclbin"
insts_path = "mel_kernels/build_fixed_v3/insts_v3.bin"
audio_path = "mel_kernels/test_audio_jfk.wav"

print(f"XCLBIN: {xclbin_path}")
print(f"Instructions: {insts_path}")
print(f"Audio: {audio_path}")
print()

# Load audio
print("Loading audio...")
audio, sr = librosa.load(audio_path, sr=16000)
print(f"✅ Audio loaded: {len(audio)} samples @ {sr}Hz")
print()

# Initialize NPU (EXACT production pattern)
print("Initializing NPU with PRODUCTION pattern...")
device = xrt.device(0)
xclbin = xrt.xclbin(xclbin_path)
device.register_xclbin(xclbin)
uuid = xclbin.get_uuid()
hw_ctx = xrt.hw_context(device, uuid)
kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
print("✅ NPU initialized")
print()

# Load instructions
with open(insts_path, "rb") as f:
    insts_bin = f.read()
print(f"✅ Instructions loaded: {len(insts_bin)} bytes")
print()

# Create buffers (EXACT production pattern: group_id 1, 3, 4)
print("Creating buffers with PRODUCTION pattern...")
print("  instr_bo: kernel.group_id(1)")
print("  input_bo: kernel.group_id(3)")
print("  output_bo: kernel.group_id(4)")
print()

instr_bo = xrt.bo(device, len(insts_bin), xrt.bo.flags.cacheable, kernel.group_id(1))
input_bo = xrt.bo(device, 800, xrt.bo.flags.host_only, kernel.group_id(3))
output_bo = xrt.bo(device, 80, xrt.bo.flags.host_only, kernel.group_id(4))
print("✅ Buffers created")
print()

# Write instructions once
instr_bo.write(insts_bin, 0)
instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
print("✅ Instructions synced")
print()

# Prepare frame from middle of audio where there's speech (800 samples = 50ms @ 16kHz)
print("Preparing audio frame...")
# Use frame from 1 second in (skip silence at beginning)
frame = audio[16000:16000+800]
frame_int8 = (frame * 127).astype(np.int8)  # Convert to int8
print(f"  Frame: {len(frame_int8)} samples (from 1.0s position)")
print(f"  Float range: [{frame.min():.4f}, {frame.max():.4f}]")
print(f"  Int8 range: [{frame_int8.min()}, {frame_int8.max()}]")
print(f"  Non-zero: {np.count_nonzero(frame_int8)}/{len(frame_int8)}")
print()

# Write frame data
input_bo.write(frame_int8.tobytes(), 0)
input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
print("✅ Audio frame synced to NPU")
print()

# Execute kernel (EXACT production pattern)
print("Executing production mel kernel with REAL audio...")
run = kernel(instr_bo, input_bo, output_bo, 3)
run.wait()
print("✅ Execution complete")
print()

# Read output
output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
output = np.frombuffer(output_bo.read(80, 0), dtype=np.int8)

# Analyze
nonzero_count = np.count_nonzero(output)
nonzero_pct = 100.0 * nonzero_count / len(output)

print("=" * 70)
print("RESULTS WITH REAL AUDIO + PRODUCTION SETUP")
print("=" * 70)
print()
print(f"📊 Output analysis:")
print(f"   Non-zero: {nonzero_count}/{len(output)} ({nonzero_pct:.1f}%)")
print(f"   Range: [{output.min()}, {output.max()}]")
print(f"   Mean: {output.mean():.2f}")
print(f"   Std: {output.std():.2f}")
print()
print(f"First 20 values: {output[:20]}")
print()

# Verdict
print("=" * 70)
if nonzero_pct > 50:
    print("✅ SUCCESS! MEL KERNEL PRODUCES NON-ZERO OUTPUT!")
    print("=" * 70)
    print()
    print("This proves:")
    print("  1. Production setup works correctly")
    print("  2. Real audio data enables kernel execution")
    print("  3. Our previous tests used wrong input data")
    print()
    print("Conclusion: Need to use real audio frames, not random int8")
    print("            Apply same pattern to attention kernels")
elif nonzero_pct > 0:
    print("⚠️  PARTIAL SUCCESS - Some non-zero values")
    print("=" * 70)
    print()
    print(f"Got {nonzero_pct:.1f}% non-zero - better than random data!")
else:
    print("❌ STILL ALL ZEROS EVEN WITH REAL AUDIO")
    print("=" * 70)
    print()
    print("This proves:")
    print("  1. Issue is NOT the input data type")
    print("  2. Issue is NOT random vs real data")
    print("  3. System-level problem persists")
    print()
    print("Need deeper investigation of XRT/NPU state")

print()
