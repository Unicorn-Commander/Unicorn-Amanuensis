#!/usr/bin/env python3
"""
Quick smoke test for mel_fixed_new.xclbin kernel

Tests basic NPU functionality with a simple 1000 Hz sine wave and
compares output against librosa reference to validate the FFT scaling
fix and HTK mel filterbank fix.

Expected correlation: >0.95 (from 4.68% before fixes)

Author: NPU Testing & Validation Team Lead
Date: October 28, 2025
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import numpy as np
import pyxrt as xrt
import librosa
from pathlib import Path

# Audio configuration
SAMPLE_RATE = 16000
AUDIO_DURATION = 1.0  # 1 second
N_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)
FRAME_SIZE = 400  # 25ms frame for Whisper

# NPU configuration
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 80
FMIN = 0
FMAX = 8000


def generate_test_tone(frequency=1000, amplitude=0.8):
    """Generate a simple test sine wave"""
    t = np.linspace(0, AUDIO_DURATION, N_SAMPLES, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal.astype(np.float32)


def compute_librosa_mel(audio):
    """Compute mel spectrogram using librosa (reference)"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        htk=True,  # HTK formula
        power=2.0
    )

    # Convert to dB scale
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_db


def run_npu_kernel(xclbin_path, insts_path, audio_int16):
    """Run single frame through NPU kernel

    Args:
        xclbin_path: Path to XCLBIN file
        insts_path: Path to instruction binary
        audio_int16: INT16 audio samples (400 samples)

    Returns:
        mel_bins: INT8 mel spectrogram (80 bins)
    """
    # Initialize NPU
    device = xrt.device(0)
    xclbin = xrt.xclbin(str(xclbin_path))
    device.register_xclbin(xclbin)

    uuid = xclbin.get_uuid()
    hw_ctx = xrt.hw_context(device, uuid)
    kernel = xrt.kernel(hw_ctx, "MLIR_AIE")

    # Read instruction binary
    with open(insts_path, 'rb') as f:
        insts_bin = f.read()
    n_insts = len(insts_bin)

    # Allocate buffers
    instr_bo = xrt.bo(device, n_insts,
                      xrt.bo.flags.cacheable, kernel.group_id(1))
    input_bo = xrt.bo(device, 800,  # 400 INT16 = 800 bytes
                      xrt.bo.flags.host_only, kernel.group_id(3))
    output_bo = xrt.bo(device, 80,  # 80 INT8 = 80 bytes
                       xrt.bo.flags.host_only, kernel.group_id(4))

    # Write instructions
    instr_bo.write(insts_bin, 0)
    instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                  n_insts, 0)

    # Convert audio to bytes (little-endian INT16)
    input_data = audio_int16.astype(np.int16).tobytes()

    # Write input to NPU
    input_bo.write(input_data, 0)
    input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 800, 0)

    # Execute kernel
    opcode = 3
    run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
    state = run.wait(5000)

    if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        raise RuntimeError(f"NPU kernel failed with state: {state}")

    # Read output
    output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 80, 0)
    mel_bins = np.frombuffer(output_bo.read(80, 0), dtype=np.int8)

    return mel_bins


def main():
    """Run quick smoke test"""
    print("=" * 70)
    print("NPU MEL KERNEL QUICK SMOKE TEST")
    print("=" * 70)
    print()

    # Paths
    base_dir = Path(__file__).parent
    xclbin_path = base_dir / "build_fixed" / "mel_fixed_new.xclbin"
    insts_path = base_dir / "build_fixed" / "insts_new.bin"

    if not xclbin_path.exists():
        print(f"ERROR: XCLBIN not found: {xclbin_path}")
        print("Build Team has not yet delivered mel_fixed_new.xclbin")
        sys.exit(1)

    if not insts_path.exists():
        print(f"ERROR: Instructions not found: {insts_path}")
        sys.exit(1)

    print(f"XCLBIN: {xclbin_path}")
    print(f"Instructions: {insts_path}")
    print()

    # Generate test audio
    print("Generating 1000 Hz test tone...")
    audio_float = generate_test_tone(frequency=1000, amplitude=0.8)

    # Convert to INT16 for NPU
    audio_int16 = (audio_float * 32767).astype(np.int16)

    # Take first frame (400 samples)
    audio_frame = audio_int16[:FRAME_SIZE]

    print(f"Audio: {len(audio_float)} samples @ {SAMPLE_RATE} Hz")
    print(f"Frame: {len(audio_frame)} samples (25ms)")
    print()

    # Run NPU kernel
    print("Running NPU kernel...")
    try:
        npu_mel = run_npu_kernel(xclbin_path, insts_path, audio_frame)
        print(f"SUCCESS: NPU returned {len(npu_mel)} mel bins")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()

    # Compute librosa reference
    print("Computing librosa reference...")
    mel_ref = compute_librosa_mel(audio_float)

    # Take first frame from librosa output
    mel_ref_frame = mel_ref[:, 0]  # 80 mel bins

    print(f"Reference: {len(mel_ref_frame)} mel bins")
    print()

    # Convert NPU output to float for comparison
    npu_mel_float = npu_mel.astype(np.float32)

    # Compute correlation
    print("Computing correlation...")

    # Pearson correlation coefficient
    mean_npu = np.mean(npu_mel_float)
    mean_ref = np.mean(mel_ref_frame)

    npu_centered = npu_mel_float - mean_npu
    ref_centered = mel_ref_frame - mean_ref

    numerator = np.sum(npu_centered * ref_centered)
    denominator = np.sqrt(np.sum(npu_centered**2) * np.sum(ref_centered**2))

    if denominator > 0:
        correlation = numerator / denominator
    else:
        correlation = 0.0

    # Compute MSE
    mse = np.mean((npu_mel_float - mel_ref_frame) ** 2)

    # Display results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Correlation:     {correlation:.6f}  ({correlation*100:.2f}%)")
    print(f"MSE:             {mse:.4f}")
    print()

    # Verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    if correlation > 0.95:
        print("SUCCESS: Correlation >0.95 (EXCELLENT)")
        print("FFT scaling fix and HTK mel filterbank fix VALIDATED!")
        print()
        print("Status: READY FOR FULL VALIDATION")
        return 0
    elif correlation > 0.90:
        print("GOOD: Correlation >0.90 (ACCEPTABLE)")
        print("Improvements confirmed but not optimal")
        print()
        print("Status: Proceed with caution")
        return 0
    elif correlation > 0.50:
        print("PARTIAL: Correlation >0.50 (PARTIAL FIX)")
        print("Some improvements but issues remain")
        print()
        print("Status: Need debugging")
        return 1
    else:
        print(f"FAILED: Correlation {correlation*100:.2f}% (BROKEN)")
        print("Fixes did not work - kernel still produces uncorrelated output")
        print()
        print("Status: BLOCKER - do not proceed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
