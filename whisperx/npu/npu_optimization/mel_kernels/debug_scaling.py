#!/usr/bin/env python3
"""
Debug the scaling issue in mel kernel.
Compares NPU output with expected librosa output to find scaling bug.
"""

import numpy as np
import librosa
import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
from pathlib import Path

def compute_librosa_mel(audio_float, n_mels=80):
    """Compute mel spectrogram using librosa (reference)"""
    mel = librosa.feature.melspectrogram(
        y=audio_float,
        sr=16000,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=n_mels,
        fmin=0,
        fmax=8000
    )
    # Convert to log scale
    mel_log = librosa.power_to_db(mel, ref=np.max)
    # Normalize to 0-127
    mel_norm = ((mel_log - mel_log.min()) / (mel_log.max() - mel_log.min()) * 127).astype(np.int8)
    return mel_norm[:, 0]  # First frame only

def test_npu_vs_librosa():
    """Compare NPU output with librosa to identify scaling bug"""
    print("="*70)
    print("NPU vs LIBROSA SCALING DEBUG")
    print("="*70)

    # Generate strong test signal (1 kHz sine, max amplitude)
    sample_rate = 16000
    duration = 0.025  # 400 samples
    t = np.linspace(0, duration, 400)
    audio_float = 0.95 * np.sin(2 * np.pi * 1000 * t)

    print(f"\nTest signal: 1000 Hz sine wave")
    print(f"  Amplitude: 0.95 (near maximum)")
    print(f"  Samples: 400")
    print(f"  Float range: [{audio_float.min():.3f}, {audio_float.max():.3f}]")

    # Compute librosa reference
    mel_librosa = compute_librosa_mel(audio_float)
    print(f"\nLibrosa MEL (INT8):")
    print(f"  Range: [{mel_librosa.min()}, {mel_librosa.max()}]")
    print(f"  Mean: {mel_librosa.mean():.2f}")
    print(f"  First 10: {mel_librosa[:10]}")
    print(f"  Non-zero: {np.count_nonzero(mel_librosa)}/80")

    # Compute NPU output
    xclbin_path = Path(__file__).parent / "build_fixed_v3" / "mel_fixed_v3.xclbin"
    insts_path = Path(__file__).parent / "build_fixed_v3" / "insts_v3.bin"

    device = xrt.device(0)
    xclbin = xrt.xclbin(str(xclbin_path))
    device.register_xclbin(xclbin)
    uuid = xclbin.get_uuid()
    hw_ctx = xrt.hw_context(device, uuid)
    kernel = xrt.kernel(hw_ctx, "MLIR_AIE")

    # Convert to NPU input format
    audio_int16 = (audio_float * 32767).astype(np.int16)
    input_data = audio_int16.tobytes()

    # Read instructions
    insts_bin = open(insts_path, "rb").read()
    n_insts = len(insts_bin)

    # Allocate buffers
    input_size = 800
    output_size = 80

    instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
    input_bo = xrt.bo(device, input_size, xrt.bo.flags.host_only, kernel.group_id(3))
    output_bo = xrt.bo(device, output_size, xrt.bo.flags.host_only, kernel.group_id(4))

    # Write data
    instr_bo.write(insts_bin, 0)
    instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

    input_bo.write(input_data, 0)
    input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, input_size, 0)

    # Execute
    opcode = 3
    run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
    state = run.wait(1000)

    if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        print(f"❌ Kernel execution failed: {state}")
        return

    # Read output
    output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, output_size, 0)
    mel_npu = np.frombuffer(output_bo.read(output_size, 0), dtype=np.int8)

    print(f"\nNPU MEL (INT8):")
    print(f"  Range: [{mel_npu.min()}, {mel_npu.max()}]")
    print(f"  Mean: {mel_npu.mean():.2f}")
    print(f"  First 10: {mel_npu[:10]}")
    print(f"  Non-zero: {np.count_nonzero(mel_npu)}/80")

    # Compare
    print(f"\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    scaling_factor = mel_librosa.max() / max(mel_npu.max(), 1)
    print(f"Librosa max: {mel_librosa.max()}")
    print(f"NPU max:     {mel_npu.max()}")
    print(f"Scaling factor needed: {scaling_factor:.1f}x")

    if scaling_factor > 10:
        print(f"\n❌ SCALING BUG CONFIRMED!")
        print(f"NPU output is {scaling_factor:.1f}x too small!")
        print(f"\nPossible causes:")
        print(f"  1. FFT output not scaled correctly after butterfly operations")
        print(f"  2. Magnitude computation losing precision")
        print(f"  3. Mel filterbank weights too small")
        print(f"  4. Final INT8 conversion factor wrong")
    elif scaling_factor < 0.5:
        print(f"\n⚠️ NPU output is {1/scaling_factor:.1f}x too large!")
    else:
        print(f"\n✅ Scaling is reasonable (within 2x)")

    # Detailed breakdown
    print(f"\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70)

    # Simulate each stage
    print("\nStage 1: Input (INT16)")
    print(f"  Range: [{audio_int16.min()}, {audio_int16.max()}]")
    print(f"  Expected: close to ±31129 for amplitude 0.95")

    print("\nStage 2: After Hann window (Q15)")
    # Hann window peaks at 1.0 in center, so max value should be preserved
    print(f"  Expected: similar to input (Hann window preserves center values)")

    print("\nStage 3: After FFT (Q15 complex)")
    # With per-stage /2 scaling, 512-point FFT divides by 256
    print(f"  Expected: reduced by ~256x due to FFT scaling")
    print(f"  Input ±31129 → FFT ±122 (approx)")

    print("\nStage 4: After magnitude (Q15)")
    print(f"  Expected: |FFT|² then >>15")
    print(f"  122² = 14884, >>15 = 0 (PROBLEM!)")
    print(f"  This is likely where precision is lost!")

    print("\nStage 5: After mel filters (Q15)")
    print(f"  Expected: sum of weighted magnitudes")
    print(f"  If magnitudes are ~0, sum will be ~0")

    print("\nStage 6: After INT8 conversion")
    print(f"  Formula: (mel_energy * 127) / 32767")
    print(f"  If mel_energy < 258, result will be 0")

    print(f"\n" + "="*70)
    print("ROOT CAUSE ANALYSIS")
    print("="*70)
    print("The FFT per-stage scaling (>>1) is correct to prevent overflow,")
    print("but it causes the magnitude computation to underflow!")
    print("")
    print("After 9 stages of /2, the FFT output is ~512x smaller.")
    print("When we compute magnitude² and >>15, we lose all precision.")
    print("")
    print("SOLUTION:")
    print("  Option 1: Don't shift magnitude result (keep Q30 instead of Q15)")
    print("  Option 2: Use less aggressive FFT scaling")
    print("  Option 3: Scale up before magnitude computation")
    print("  Option 4: Adjust mel_energy to INT8 scaling factor")

if __name__ == "__main__":
    test_npu_vs_librosa()
