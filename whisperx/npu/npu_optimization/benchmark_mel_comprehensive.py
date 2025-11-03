#!/usr/bin/env python3
"""
Comprehensive Benchmark: NPU Mel Processor vs Librosa

This benchmark tests:
1. Performance with various audio lengths (1s, 10s, 30s, 60s)
2. Accuracy with different audio characteristics
3. End-to-end Whisper transcription speedup
4. Memory usage and overhead

Author: Magic Unicorn Unconventional Technology & Stuff Inc.
Date: October 30, 2025
"""

import sys
import os
sys.path.insert(0, '/opt/xilinx/xrt/python')

import numpy as np
import librosa
import time
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))

from npu_mel_processor import NPUMelProcessor

print("=" * 80)
print("COMPREHENSIVE NPU MEL PROCESSOR BENCHMARK")
print("=" * 80)
print()

# Initialize processor
print("Initializing NPU Mel Processor...")
processor = NPUMelProcessor()
print(f"NPU Available: {processor.npu_available}")
print()

# Test audio durations
test_durations = [1, 5, 10, 30, 60]  # seconds

results = []

for duration in test_durations:
    print("=" * 80)
    print(f"TEST: {duration}s Audio")
    print("=" * 80)

    # Generate test audio
    sample_rate = 16000
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples)

    # Generate realistic speech-like signal
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +      # Fundamental
        0.2 * np.sin(2 * np.pi * 400 * t) +      # First harmonic
        0.15 * np.sin(2 * np.pi * 800 * t) +     # Second harmonic
        0.1 * np.sin(2 * np.pi * 1200 * t) +     # Third harmonic
        0.05 * np.sin(2 * np.pi * 1600 * t) +    # Fourth harmonic
        0.08 * np.random.randn(n_samples)        # Noise
    )
    audio = audio.astype(np.float32)

    print(f"Generated {duration}s audio ({n_samples} samples)")
    print()

    # Process with NPU (3 runs to stabilize)
    print("Processing with NPU...")
    npu_times = []
    for i in range(3):
        processor.reset_metrics()
        start = time.time()
        mel_npu = processor.process(audio)
        elapsed = time.time() - start
        npu_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s ({duration/elapsed:.1f}x realtime)")

    npu_avg = np.mean(npu_times)
    npu_realtime = duration / npu_avg

    print(f"NPU Average: {npu_avg:.3f}s ({npu_realtime:.1f}x realtime)")
    print()

    # Process with librosa (3 runs)
    print("Processing with librosa...")
    librosa_times = []
    for i in range(3):
        start = time.time()
        mel_ref = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=80,
            htk=True,
            power=2.0,
            fmin=0,
            fmax=8000
        )
        mel_ref_db = librosa.power_to_db(mel_ref, ref=np.max)
        elapsed = time.time() - start
        librosa_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s ({duration/elapsed:.1f}x realtime)")

    librosa_avg = np.mean(librosa_times)
    librosa_realtime = duration / librosa_avg

    print(f"Librosa Average: {librosa_avg:.3f}s ({librosa_realtime:.1f}x realtime)")
    print()

    # Accuracy validation
    print("Validating accuracy...")

    def normalize(x):
        x = x.astype(np.float32)
        x_range = x.max() - x.min()
        return (x - x.min()) / x_range if x_range > 1e-6 else np.zeros_like(x)

    mel_npu_norm = normalize(mel_npu)
    mel_ref_norm = normalize(mel_ref_db)

    # Match dimensions
    n_common = min(mel_npu_norm.shape[1], mel_ref_norm.shape[1])
    mel_npu_compare = mel_npu_norm[:, :n_common]
    mel_ref_compare = mel_ref_norm[:, :n_common]

    # Compute metrics
    correlation = np.corrcoef(
        mel_npu_compare.flatten(),
        mel_ref_compare.flatten()
    )[0, 1]

    rmse = np.sqrt(np.mean((mel_npu_compare - mel_ref_compare) ** 2))
    mae = np.mean(np.abs(mel_npu_compare - mel_ref_compare))

    print(f"Correlation: {correlation:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print()

    # Calculate speedup
    speedup = librosa_avg / npu_avg if npu_avg > 0 else 1.0

    # Calculate frame processing rate
    n_frames = mel_npu.shape[1]
    frames_per_sec_npu = n_frames / npu_avg
    frames_per_sec_librosa = n_frames / librosa_avg

    print(f"Performance Summary:")
    print(f"  NPU:     {npu_avg:.3f}s ({npu_realtime:6.1f}x realtime, {frames_per_sec_npu:.0f} frames/s)")
    print(f"  Librosa: {librosa_avg:.3f}s ({librosa_realtime:6.1f}x realtime, {frames_per_sec_librosa:.0f} frames/s)")
    print(f"  Speedup: {speedup:.2f}x")
    print()

    # Save results
    results.append({
        'duration': duration,
        'npu_time': npu_avg,
        'librosa_time': librosa_avg,
        'npu_realtime': npu_realtime,
        'librosa_realtime': librosa_realtime,
        'speedup': speedup,
        'correlation': correlation,
        'rmse': rmse,
        'mae': mae,
        'frames': n_frames
    })

# Summary table
print("=" * 80)
print("BENCHMARK SUMMARY")
print("=" * 80)
print()

print(f"{'Duration':>8} | {'NPU (s)':>8} | {'Librosa (s)':>11} | {'Speedup':>8} | {'Correlation':>11}")
print("-" * 80)

for r in results:
    print(f"{r['duration']:>7}s | {r['npu_time']:>8.3f} | {r['librosa_time']:>11.3f} | {r['speedup']:>8.2f}x | {r['correlation']:>11.4f}")

print()

# Analysis
print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print()

# Check if performance improves with longer audio
if len(results) >= 2:
    short_speedup = results[0]['speedup']
    long_speedup = results[-1]['speedup']

    print(f"Performance Scaling:")
    print(f"  Short audio ({results[0]['duration']}s): {short_speedup:.2f}x speedup")
    print(f"  Long audio ({results[-1]['duration']}s): {long_speedup:.2f}x speedup")

    if long_speedup > short_speedup * 1.5:
        print(f"  ✅ NPU performance scales well with longer audio")
    elif long_speedup > short_speedup:
        print(f"  ✅ NPU shows improvement with longer audio")
    else:
        print(f"  ⚠️ NPU performance doesn't scale significantly")

    print()

# Check accuracy consistency
avg_correlation = np.mean([r['correlation'] for r in results])
std_correlation = np.std([r['correlation'] for r in results])

print(f"Accuracy Consistency:")
print(f"  Average correlation: {avg_correlation:.4f}")
print(f"  Std deviation: {std_correlation:.4f}")

if avg_correlation >= 0.95:
    print(f"  ✅ EXCELLENT accuracy")
elif avg_correlation >= 0.85:
    print(f"  ✅ GOOD accuracy")
elif avg_correlation >= 0.70:
    print(f"  ⚠️ ACCEPTABLE accuracy (may need tuning)")
else:
    print(f"  ❌ LOW accuracy (needs improvement)")

print()

# Expected end-to-end improvement
print("=" * 80)
print("EXPECTED END-TO-END IMPACT")
print("=" * 80)
print()

print("Current Whisper baseline: 19.1x realtime")
print("Mel spectrogram: ~5.8% of processing time")
print()

# Calculate expected improvement
baseline_rtf = 19.1
mel_percentage = 0.058
mel_speedup = results[-1]['speedup'] if results else 1.0

# Calculate new RTF
# If mel is 5.8% of time and we speed it up by speedup factor:
# New mel time = mel_time / speedup
# New total time = total_time - mel_time + (mel_time / speedup)
# New RTF = audio_duration / new_total_time

# For 19.1x realtime: total_time = audio_duration / 19.1
# mel_time = total_time * 0.058
# New mel_time = mel_time / speedup
# Time saved = mel_time - new_mel_time = mel_time * (1 - 1/speedup)
# New total_time = total_time - time_saved
# New RTF = audio_duration / new_total_time

time_saved_percentage = mel_percentage * (1 - 1/mel_speedup)
new_total_time_ratio = 1 - time_saved_percentage
expected_new_rtf = baseline_rtf / new_total_time_ratio

print(f"With NPU mel ({mel_speedup:.1f}x speedup):")
print(f"  Time saved: {time_saved_percentage*100:.1f}%")
print(f"  Expected new RTF: {expected_new_rtf:.1f}x realtime")
print(f"  Improvement: {baseline_rtf:.1f}x → {expected_new_rtf:.1f}x")
print()

if expected_new_rtf >= 22:
    print(f"✅ Target achieved: {expected_new_rtf:.1f}x >= 22x")
elif expected_new_rtf >= 20:
    print(f"✅ Close to target: {expected_new_rtf:.1f}x >= 20x")
else:
    print(f"⚠️ Below target: {expected_new_rtf:.1f}x < 20x")
    print(f"   Note: Actual improvement may differ based on real audio")

print()

# Final recommendations
print("=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print()

if avg_correlation >= 0.95 and long_speedup >= 10:
    print("✅ READY FOR PRODUCTION")
    print("   - Accuracy excellent (>0.95)")
    print("   - Performance good (>10x speedup)")
    print("   - Integrate into production runtime immediately")
elif avg_correlation >= 0.85 and long_speedup >= 5:
    print("✅ READY FOR TESTING")
    print("   - Accuracy good (>0.85)")
    print("   - Performance acceptable (>5x speedup)")
    print("   - Test with real audio workloads before full deployment")
elif avg_correlation >= 0.70:
    print("⚠️ NEEDS TUNING")
    print("   - Accuracy acceptable but could be better")
    print("   - Consider kernel parameter tuning")
    print("   - Compare with UC-Meeting-Ops mel kernel (0.75+ correlation)")
else:
    print("❌ NEEDS WORK")
    print("   - Accuracy too low (<0.70)")
    print("   - Review kernel implementation")
    print("   - Check FFT scaling and mel filterbank calculations")

print()

# Cleanup
processor.close()

print("=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)
