#!/usr/bin/env python3
"""
NPU Mel Preprocessing Integration Test
Tests mel spectrogram computation with both simple and optimized kernels

Team 2: WhisperX Integration Lead
Date: October 28, 2025

This test focuses on the mel preprocessing component which is the first step
in the WhisperX pipeline and demonstrates NPU acceleration capabilities.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

# Add to path for npu_mel_preprocessing import
# File is in: whisperx/npu/npu_optimization/mel_kernels/
# NPU module is in: whisperx/npu/
npu_dir = Path(__file__).parent.parent.parent  # Go up to whisperx/npu/
sys.path.insert(0, str(npu_dir))

# Import NPU module at module level
from npu_mel_preprocessing import NPUMelPreprocessor

def load_audio(audio_path):
    """Load audio file using librosa"""
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    return audio, sr


def compute_mel_cpu(audio, sr=16000):
    """Compute mel spectrogram using CPU (librosa baseline)"""
    import librosa

    start = time.time()
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=400,
        hop_length=160,
        win_length=400,
        n_mels=80,
        fmin=0,
        fmax=sr//2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    elapsed = time.time() - start

    return mel_db, elapsed


def compute_mel_npu(audio, xclbin_path, kernel_name):
    """Compute mel spectrogram using NPU"""
    start_init = time.time()
    preprocessor = NPUMelPreprocessor(
        xclbin_path=xclbin_path,
        sample_rate=16000,
        n_mels=80,
        frame_size=400,
        hop_length=160,
        fallback_to_cpu=False  # Force NPU-only for this test
    )
    init_time = time.time() - start_init

    if not preprocessor.npu_available:
        raise RuntimeError(f"NPU not available for {kernel_name} kernel")

    start_process = time.time()
    mel_features = preprocessor.process_audio(audio)
    process_time = time.time() - start_process

    metrics = preprocessor.get_performance_metrics()

    preprocessor.close()

    return mel_features, {
        "init_time": init_time,
        "process_time": process_time,
        "metrics": metrics
    }


def calculate_mel_similarity(mel1, mel2):
    """Calculate similarity between two mel spectrograms"""
    # Ensure same shape
    min_frames = min(mel1.shape[1] if len(mel1.shape) > 1 else mel1.shape[0],
                     mel2.shape[1] if len(mel2.shape) > 1 else mel2.shape[0])

    if len(mel1.shape) > 1:
        mel1 = mel1[:, :min_frames]
    else:
        mel1 = mel1[:min_frames]

    if len(mel2.shape) > 1:
        mel2 = mel2[:, :min_frames]
    else:
        mel2 = mel2[:min_frames]

    # Calculate correlation coefficient
    correlation = np.corrcoef(mel1.flatten(), mel2.flatten())[0, 1]

    # Calculate MSE
    mse = np.mean((mel1 - mel2) ** 2)

    # Calculate PSNR
    max_val = max(np.max(np.abs(mel1)), np.max(np.abs(mel2)))
    if max_val > 0:
        psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float('inf')
    else:
        psnr = float('inf')

    return {
        "correlation": correlation,
        "mse": mse,
        "psnr": psnr
    }


def test_kernel(kernel_name, xclbin_path, audio_path):
    """Test mel preprocessing with specific NPU kernel"""
    print(f"\n{'='*80}")
    print(f"TESTING: {kernel_name.upper()} KERNEL")
    print(f"{'='*80}")
    print(f"XCLBIN: {xclbin_path}")
    print(f"Audio: {audio_path}")

    # Check files exist
    if not os.path.exists(xclbin_path):
        print(f"ERROR: XCLBIN not found: {xclbin_path}")
        return None
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio not found: {audio_path}")
        return None

    try:
        # Load audio
        print("\nLoading audio...")
        audio, sr = load_audio(audio_path)
        audio_duration = len(audio) / sr
        print(f"  Duration: {audio_duration:.2f}s ({len(audio)} samples @ {sr}Hz)")

        # Compute CPU baseline
        print("\nComputing CPU baseline (librosa)...")
        mel_cpu, cpu_time = compute_mel_cpu(audio, sr)
        print(f"  CPU time: {cpu_time:.4f}s")
        print(f"  Mel shape: {mel_cpu.shape}")

        # Compute NPU version
        print(f"\nComputing with {kernel_name} NPU kernel...")
        mel_npu, npu_stats = compute_mel_npu(audio, xclbin_path, kernel_name)
        print(f"  Init time: {npu_stats['init_time']:.4f}s")
        print(f"  Process time: {npu_stats['process_time']:.4f}s")
        print(f"  Mel shape: {mel_npu.shape}")

        # Calculate similarity
        print("\nComparing CPU vs NPU mel spectrograms...")
        similarity = calculate_mel_similarity(mel_cpu, mel_npu)
        print(f"  Correlation: {similarity['correlation']:.6f}")
        print(f"  MSE: {similarity['mse']:.6f}")
        print(f"  PSNR: {similarity['psnr']:.2f} dB")

        # Calculate speedup
        speedup = cpu_time / npu_stats['process_time'] if npu_stats['process_time'] > 0 else 0
        rtf_cpu = audio_duration / cpu_time if cpu_time > 0 else 0
        rtf_npu = audio_duration / npu_stats['process_time'] if npu_stats['process_time'] > 0 else 0

        # Print results
        print(f"\n{'-'*80}")
        print(f"RESULTS: {kernel_name.upper()}")
        print(f"{'-'*80}")
        print(f"Audio duration:        {audio_duration:.2f}s")
        print(f"CPU processing time:   {cpu_time:.4f}s ({rtf_cpu:.2f}x realtime)")
        print(f"NPU processing time:   {npu_stats['process_time']:.4f}s ({rtf_npu:.2f}x realtime)")
        print(f"NPU speedup:           {speedup:.2f}x")
        print(f"Mel correlation:       {similarity['correlation']:.6f}")
        print(f"Mel PSNR:              {similarity['psnr']:.2f} dB")
        print(f"{'-'*80}")

        # NPU metrics
        if 'metrics' in npu_stats:
            metrics = npu_stats['metrics']
            print(f"\nNPU Metrics:")
            print(f"  Total frames: {metrics.get('total_frames', 0)}")
            print(f"  NPU time per frame: {metrics.get('npu_time_per_frame_ms', 0):.2f}ms")
            if metrics.get('speedup', 0) > 0:
                print(f"  NPU vs CPU speedup: {metrics.get('speedup', 0):.2f}x")

        return {
            "kernel": kernel_name,
            "xclbin": xclbin_path,
            "audio_duration": audio_duration,
            "cpu_time": cpu_time,
            "npu_time": npu_stats['process_time'],
            "npu_init_time": npu_stats['init_time'],
            "speedup": speedup,
            "rtf_cpu": rtf_cpu,
            "rtf_npu": rtf_npu,
            "correlation": similarity['correlation'],
            "mse": similarity['mse'],
            "psnr": similarity['psnr'],
            "npu_metrics": npu_stats.get('metrics', {}),
            "success": True,
            "error": None
        }

    except Exception as e:
        print(f"\nERROR during {kernel_name} kernel test:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            "kernel": kernel_name,
            "xclbin": xclbin_path,
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def main():
    """Main test runner"""

    print("="*80)
    print("NPU MEL PREPROCESSING INTEGRATION TEST")
    print("Testing mel spectrogram computation with Simple and Optimized kernels")
    print("="*80)

    # Test configuration
    base_dir = Path(__file__).parent
    audio_path = base_dir / "test_audio_jfk.wav"

    # Test cases
    tests = [
        {
            "name": "simple",
            "xclbin": base_dir / "build_fixed" / "mel_fixed_new.xclbin"
        },
        {
            "name": "optimized",
            "xclbin": base_dir / "build_optimized" / "mel_optimized_new.xclbin"
        }
    ]

    # Run tests
    results = []
    for test_config in tests:
        result = test_kernel(
            kernel_name=test_config["name"],
            xclbin_path=str(test_config["xclbin"]),
            audio_path=str(audio_path)
        )

        if result:
            results.append(result)

    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    if len(results) == 2:
        simple = results[0]
        optimized = results[1]

        print(f"\n{'Metric':<30} {'Simple':<20} {'Optimized':<20} {'Difference':<15}")
        print(f"{'-'*85}")

        if simple["success"] and optimized["success"]:
            # Speed comparison
            speed_diff = ((simple["npu_time"] - optimized["npu_time"]) / simple["npu_time"] * 100) if simple["npu_time"] > 0 else 0
            print(f"{'NPU Processing Time':<30} {simple['npu_time']:>18.4f}s {optimized['npu_time']:>18.4f}s {speed_diff:>13.1f}%")

            speedup_diff = optimized["speedup"] - simple["speedup"]
            print(f"{'NPU Speedup vs CPU':<30} {simple['speedup']:>18.2f}x {optimized['speedup']:>18.2f}x {speedup_diff:>13.2f}x")

            rtf_diff = optimized["rtf_npu"] - simple["rtf_npu"]
            print(f"{'Realtime Factor (NPU)':<30} {simple['rtf_npu']:>18.2f}x {optimized['rtf_npu']:>18.2f}x {rtf_diff:>13.2f}x")

            # Quality comparison
            corr_diff = (optimized["correlation"] - simple["correlation"])
            print(f"{'Mel Correlation':<30} {simple['correlation']:>18.6f} {optimized['correlation']:>18.6f} {corr_diff:>+13.6f}")

            psnr_diff = optimized["psnr"] - simple["psnr"]
            print(f"{'Mel PSNR (dB)':<30} {simple['psnr']:>18.2f} {optimized['psnr']:>18.2f} {psnr_diff:>+13.2f}")

            # Success criteria
            print(f"\n{'='*80}")
            print("SUCCESS CRITERIA")
            print(f"{'='*80}")

            criteria = {
                "Both kernels execute successfully": simple["success"] and optimized["success"],
                "Optimized kernel faster (any improvement)": optimized["npu_time"] < simple["npu_time"],
                "High correlation with CPU baseline (>0.9)": simple["correlation"] > 0.9 and optimized["correlation"] > 0.9,
                "Good PSNR (>30dB indicates quality)": simple["psnr"] > 30 and optimized["psnr"] > 30,
                "NPU provides speedup vs CPU": simple["speedup"] > 1.0 and optimized["speedup"] > 1.0,
                "No crashes or errors": simple["error"] is None and optimized["error"] is None
            }

            all_passed = True
            for criterion, passed in criteria.items():
                status = "PASS" if passed else "FAIL"
                if not passed:
                    all_passed = False
                print(f"[{status}] {criterion}")

            print(f"\n{'='*80}")
            if all_passed:
                print("ALL CRITERIA PASSED - NPU MEL PREPROCESSING INTEGRATION SUCCESSFUL!")
            else:
                print("SOME CRITERIA FAILED - SEE DETAILS ABOVE")
            print(f"{'='*80}")

        else:
            if not simple["success"]:
                print(f"Simple kernel FAILED: {simple.get('error', 'Unknown error')}")
            if not optimized["success"]:
                print(f"Optimized kernel FAILED: {optimized.get('error', 'Unknown error')}")

    # Save results to JSON
    results_path = base_dir / "mel_preprocessing_test_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        json.dump(convert_to_json_serializable(results), f, indent=2)

    print(f"\nResults saved to: {results_path}")

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    results = main()
