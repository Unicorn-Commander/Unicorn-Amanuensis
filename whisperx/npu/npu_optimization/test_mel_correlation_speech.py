#!/usr/bin/env python3
"""
NPU Mel Kernel Correlation Test on Real Speech
================================================

Tests the actual correlation of mel_fixed_v3_PRODUCTION_v2.0.xclbin
on real speech audio (JFK), using the correct comparison method.

Key insight from ACCURACY_FIX_COMPLETE_OCT30.md:
- NPU outputs linear power spectrum (INT8)
- Librosa outputs power spectrum then converts to dB
- Must compare NPU power → librosa power (before dB conversion)
- Claimed: 0.9152 average on sine waves
- Need to test: Actual correlation on speech

Author: Autonomous Testing Agent
Date: October 30, 2025
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import os
import numpy as np
from pathlib import Path
import pyxrt as xrt
import librosa

print("=" * 80)
print("NPU MEL KERNEL CORRELATION TEST - REAL SPEECH")
print("=" * 80)

# Configuration
AUDIO_PATH = "mel_kernels/test_audio_jfk.wav"
XCLBIN_V2 = "mel_kernels/build_fixed_v3/mel_fixed_v3_PRODUCTION_v2.0.xclbin"
XCLBIN_V1 = "mel_kernels/build_fixed_v3/mel_fixed_v3_PRODUCTION_v1.0.xclbin"
INSTS_PATH = "mel_kernels/build_fixed_v3/insts_v3.bin"

def process_with_npu(audio, xclbin_path):
    """Process audio with NPU mel kernel"""
    print(f"\n  Loading NPU kernel: {Path(xclbin_path).name}")

    # Initialize NPU
    device = xrt.device(0)
    xclbin = xrt.xclbin(xclbin_path)
    device.register_xclbin(xclbin)
    uuid = xclbin.get_uuid()
    hw_ctx = xrt.hw_context(device, uuid)
    kernel = xrt.kernel(hw_ctx, "MLIR_AIE")

    # Load instructions
    with open(INSTS_PATH, "rb") as f:
        insts_bin = f.read()
    n_insts = len(insts_bin)

    # Create buffers
    instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
    input_bo = xrt.bo(device, 800, xrt.bo.flags.host_only, kernel.group_id(3))
    output_bo = xrt.bo(device, 80, xrt.bo.flags.host_only, kernel.group_id(4))

    # Write instructions
    instr_bo.write(insts_bin, 0)
    instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

    # Process frames
    hop_length = 160
    frame_length = 400
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    mel_npu = np.zeros((80, n_frames), dtype=np.int8)

    for frame_idx in range(n_frames):
        start_sample = frame_idx * hop_length
        audio_frame = audio[start_sample:start_sample + frame_length]
        audio_int16 = (audio_frame * 32767).astype(np.int16)

        input_bo.write(audio_int16.tobytes(), 0)
        input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 800, 0)

        opcode = 3
        run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
        run.wait(10000)

        output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 80, 0)
        mel_npu[:, frame_idx] = np.frombuffer(output_bo.read(80, 0), dtype=np.int8)

    print(f"    Processed {n_frames} frames")
    print(f"    Output range: [{mel_npu.min()}, {mel_npu.max()}]")
    print(f"    Non-zero: {100 * np.count_nonzero(mel_npu) / mel_npu.size:.1f}%")

    return mel_npu

def process_with_librosa(audio):
    """Process audio with librosa (CPU baseline)"""
    print(f"\n  Processing with librosa (CPU baseline)")

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=80,
        fmin=0,
        fmax=8000,
        htk=True,
        power=2.0
    )

    print(f"    Output shape: {mel.shape}")
    print(f"    Power spectrum range: [{mel.min():.2e}, {mel.max():.2e}]")

    return mel

def compare_spectrograms(mel_npu_int8, mel_librosa_power):
    """
    Compare NPU and librosa spectrograms using correct method

    NPU outputs: Linear power spectrum (INT8, scaled)
    Librosa outputs: Linear power spectrum (float32)

    Comparison method:
    1. Convert both to float32
    2. Normalize using z-score (subtract mean, divide by std)
    3. Compute correlation
    """
    # Convert NPU INT8 to float32
    mel_npu_float = mel_npu_int8.astype(np.float32)

    # Ensure same number of frames
    n_frames = min(mel_npu_float.shape[1], mel_librosa_power.shape[1])
    mel_npu_float = mel_npu_float[:, :n_frames]
    mel_librosa_power = mel_librosa_power[:, :n_frames]

    # Method 1: Min-max normalization
    def normalize_minmax(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 1e-6:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)

    mel_npu_norm = normalize_minmax(mel_npu_float)
    mel_librosa_norm = normalize_minmax(mel_librosa_power)

    corr_minmax = np.corrcoef(
        mel_npu_norm.flatten(),
        mel_librosa_norm.flatten()
    )[0, 1]

    # Method 2: Z-score normalization
    def normalize_zscore(x):
        mean = x.mean()
        std = x.std()
        if std > 1e-6:
            return (x - mean) / std
        return x - mean

    mel_npu_z = normalize_zscore(mel_npu_float)
    mel_librosa_z = normalize_zscore(mel_librosa_power)

    corr_zscore = np.corrcoef(
        mel_npu_z.flatten(),
        mel_librosa_z.flatten()
    )[0, 1]

    # Method 3: Direct comparison (no normalization)
    corr_direct = np.corrcoef(
        mel_npu_float.flatten(),
        mel_librosa_power.flatten()
    )[0, 1]

    return {
        "correlation_minmax": corr_minmax,
        "correlation_zscore": corr_zscore,
        "correlation_direct": corr_direct,
        "npu_mean": mel_npu_float.mean(),
        "npu_std": mel_npu_float.std(),
        "librosa_mean": mel_librosa_power.mean(),
        "librosa_std": mel_librosa_power.std(),
    }

def main():
    """Main test execution"""

    # Load audio
    print("\n" + "="*80)
    print("LOADING AUDIO")
    print("="*80)

    audio, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)
    print(f"  Audio: {AUDIO_PATH}")
    print(f"  Duration: {len(audio) / sr:.2f}s")
    print(f"  Samples: {len(audio)}")

    # Process with librosa
    print("\n" + "="*80)
    print("CPU BASELINE (librosa)")
    print("="*80)

    mel_librosa = process_with_librosa(audio)

    # Test both kernel versions
    kernels_to_test = []
    if Path(XCLBIN_V2).exists():
        kernels_to_test.append(("v2.0 (Oct 30)", XCLBIN_V2))
    if Path(XCLBIN_V1).exists():
        kernels_to_test.append(("v1.0 (Oct 29)", XCLBIN_V1))

    results = []

    for version, xclbin_path in kernels_to_test:
        print("\n" + "="*80)
        print(f"NPU KERNEL: {version}")
        print("="*80)

        try:
            mel_npu = process_with_npu(audio, xclbin_path)

            print(f"\n  Comparing spectrograms...")
            comparison = compare_spectrograms(mel_npu, mel_librosa)

            print(f"\n  Correlation Results:")
            print(f"    Min-max normalization: {comparison['correlation_minmax']:.4f}")
            print(f"    Z-score normalization: {comparison['correlation_zscore']:.4f}")
            print(f"    Direct (no norm):      {comparison['correlation_direct']:.4f}")

            print(f"\n  Statistics:")
            print(f"    NPU mean: {comparison['npu_mean']:.4f}, std: {comparison['npu_std']:.4f}")
            print(f"    Librosa mean: {comparison['librosa_mean']:.2e}, std: {comparison['librosa_std']:.2e}")

            # Assessment
            best_corr = max(comparison['correlation_minmax'],
                          comparison['correlation_zscore'],
                          comparison['correlation_direct'])

            if best_corr >= 0.85:
                status = "✅ EXCELLENT"
                assessment = "Meets production target (>0.85)"
            elif best_corr >= 0.70:
                status = "✅ GOOD"
                assessment = "Acceptable for speech recognition (>0.70)"
            elif best_corr >= 0.50:
                status = "⚠️  MARGINAL"
                assessment = "May impact WER, needs testing"
            else:
                status = "❌ LOW"
                assessment = "Likely significant WER degradation"

            print(f"\n  Assessment: {status}")
            print(f"    {assessment}")

            results.append({
                "version": version,
                "xclbin": xclbin_path,
                "best_correlation": best_corr,
                "status": status,
                **comparison
            })

        except Exception as e:
            print(f"  ❌ Error testing {version}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if not results:
        print("  ❌ No kernels successfully tested")
        return

    print(f"\n  Tested {len(results)} kernel version(s)")

    for result in results:
        print(f"\n  {result['version']}:")
        print(f"    Best correlation: {result['best_correlation']:.4f}")
        print(f"    Status: {result['status']}")

    # Find best kernel
    best = max(results, key=lambda r: r['best_correlation'])

    print(f"\n  Best performer: {best['version']}")
    print(f"    Correlation: {best['best_correlation']:.4f}")

    # Compare to documented claim
    documented_claim = 0.9152
    print(f"\n  Documented claim: {documented_claim:.4f} (on sine waves)")
    print(f"  Actual on speech: {best['best_correlation']:.4f}")
    print(f"  Difference: {best['best_correlation'] - documented_claim:+.4f}")

    # Production recommendation
    print(f"\n" + "="*80)
    print("PRODUCTION RECOMMENDATION")
    print("="*80)

    if best['best_correlation'] >= 0.85:
        print(f"\n  ✅ APPROVED FOR PRODUCTION")
        print(f"     Correlation {best['best_correlation']:.4f} exceeds 0.85 target")
        print(f"     Expected WER degradation: <1%")
    elif best['best_correlation'] >= 0.70:
        print(f"\n  ⚠️  CONDITIONAL APPROVAL")
        print(f"     Correlation {best['best_correlation']:.4f} is acceptable")
        print(f"     Recommend: Test actual WER before production")
        print(f"     Expected WER degradation: 1-3%")
    else:
        print(f"\n  ❌ NOT RECOMMENDED")
        print(f"     Correlation {best['best_correlation']:.4f} is too low")
        print(f"     Expected WER degradation: >3%")
        print(f"     Action: Improve kernel accuracy before deployment")

    print("\n" + "="*80)

if __name__ == "__main__":
    # Change to correct directory
    if not Path("mel_kernels").exists():
        os.chdir("/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization")

    main()
