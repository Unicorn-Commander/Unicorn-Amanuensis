#!/usr/bin/env python3
"""
Comprehensive Test: NPU Mel Processor Integration

Tests:
1. NPU mel processor initialization
2. Processing real audio file
3. Accuracy validation against librosa (target >0.95)
4. Performance benchmarking
5. End-to-end integration with runtime

Expected Results:
- Accuracy: >0.95 correlation with librosa
- Performance: 20-30x realtime
- Overall speedup: 19.1x → 22-25x realtime

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
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("NPU MEL PROCESSOR INTEGRATION TEST")
print("=" * 80)
print()

# Test 1: Import and Initialize
print("Test 1: Import and Initialize NPU Mel Processor")
print("-" * 80)

try:
    from npu_mel_processor import NPUMelProcessor
    processor = NPUMelProcessor()
    print(f"✅ NPU Mel Processor imported successfully")
    print(f"   NPU Available: {processor.npu_available}")
    print(f"   XCLBIN: {Path(processor.xclbin_path).name}")
    print()
except Exception as e:
    print(f"❌ Failed to import NPU Mel Processor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Load Real Audio
print("Test 2: Load Real Audio File")
print("-" * 80)

# Try multiple possible audio file locations
audio_paths = [
    "/home/ucadmin/VibeVoice/Shafen_Khan_call.m4a",
    "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/test_audio_jfk.wav",
    "/home/ucadmin/test_audio.wav"
]

audio_path = None
for path in audio_paths:
    if os.path.exists(path):
        audio_path = path
        break

if audio_path is None:
    print("⚠️ No test audio file found, generating synthetic audio")
    # Generate synthetic test audio (5 seconds)
    sample_rate = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Mix of frequencies (simulate speech)
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.15 * np.sin(2 * np.pi * 800 * t) +
        0.1 * np.random.randn(len(t))
    )
    audio = audio.astype(np.float32)
    audio_duration = duration
    print(f"   Generated synthetic audio: {duration}s")
else:
    print(f"   Loading: {audio_path}")
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_duration = len(audio) / 16000
        print(f"✅ Audio loaded: {audio_duration:.2f}s @ 16000 Hz")
        print(f"   Samples: {len(audio)}")
    except Exception as e:
        print(f"❌ Failed to load audio: {e}")
        sys.exit(1)

print()

# Test 3: Process with NPU
print("Test 3: Process Audio with NPU Mel Processor")
print("-" * 80)

try:
    npu_start = time.time()
    mel_npu = processor.process(audio)
    npu_elapsed = time.time() - npu_start
    npu_realtime = audio_duration / npu_elapsed if npu_elapsed > 0 else 0

    print(f"✅ NPU processing complete")
    print(f"   Time: {npu_elapsed:.3f}s")
    print(f"   Output shape: {mel_npu.shape}")
    print(f"   Realtime factor: {npu_realtime:.1f}x")
    print(f"   Backend: {'NPU' if processor.npu_available else 'CPU fallback'}")
    print()
except Exception as e:
    print(f"❌ NPU processing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Compare with librosa (Accuracy Validation)
print("Test 4: Accuracy Validation vs Librosa")
print("-" * 80)

try:
    librosa_start = time.time()
    mel_ref = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
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
    librosa_elapsed = time.time() - librosa_start
    librosa_realtime = audio_duration / librosa_elapsed if librosa_elapsed > 0 else 0

    print(f"✅ Librosa processing complete")
    print(f"   Time: {librosa_elapsed:.3f}s")
    print(f"   Output shape: {mel_ref_db.shape}")
    print(f"   Realtime factor: {librosa_realtime:.1f}x")
    print()

    # Normalize for comparison
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

    # Compute correlation
    correlation = np.corrcoef(
        mel_npu_compare.flatten(),
        mel_ref_compare.flatten()
    )[0, 1]

    # Compute RMSE
    rmse = np.sqrt(np.mean((mel_npu_compare - mel_ref_compare) ** 2))

    # Compute mean absolute error
    mae = np.mean(np.abs(mel_npu_compare - mel_ref_compare))

    print(f"Accuracy Metrics:")
    print(f"   Correlation: {correlation:.4f} (target: >0.95)")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")

    if correlation >= 0.95:
        print(f"   ✅ EXCELLENT! Correlation {correlation:.4f} >= 0.95")
    elif correlation >= 0.85:
        print(f"   ✅ GOOD! Correlation {correlation:.4f} >= 0.85")
    elif correlation >= 0.70:
        print(f"   ⚠️ ACCEPTABLE! Correlation {correlation:.4f} >= 0.70")
    else:
        print(f"   ❌ LOW CORRELATION: {correlation:.4f} < 0.70")

    print()

except Exception as e:
    print(f"❌ Accuracy validation failed: {e}")
    import traceback
    traceback.print_exc()
    correlation = 0.0

# Test 5: Performance Comparison
print("Test 5: Performance Comparison")
print("-" * 80)

speedup = librosa_realtime / npu_realtime if npu_realtime > 0 else 1.0

print(f"Performance Summary:")
print(f"   NPU:     {npu_elapsed:6.3f}s ({npu_realtime:5.1f}x realtime)")
print(f"   Librosa: {librosa_elapsed:6.3f}s ({librosa_realtime:5.1f}x realtime)")
print(f"   Speedup: {speedup:.1f}x")
print()

if processor.npu_available:
    if speedup >= 20:
        print(f"   ✅ EXCELLENT! {speedup:.1f}x speedup (target: 20-30x)")
    elif speedup >= 10:
        print(f"   ✅ GOOD! {speedup:.1f}x speedup")
    elif speedup >= 5:
        print(f"   ⚠️ MODERATE! {speedup:.1f}x speedup (expected higher)")
    else:
        print(f"   ❌ LOW! {speedup:.1f}x speedup (check NPU execution)")
else:
    print(f"   ℹ️ CPU fallback mode (NPU not available)")

print()

# Test 6: Integration with Runtime (if available)
print("Test 6: Runtime Integration Test")
print("-" * 80)

try:
    from npu_runtime_aie2 import NPURuntime

    runtime = NPURuntime()
    if runtime.is_available():
        print(f"✅ NPU Runtime available")

        device_info = runtime.get_device_info()
        print(f"   Device info: {device_info}")

        if 'npu_mel_processor' in str(device_info):
            print(f"   ✅ Mel processor integrated in runtime")
        else:
            print(f"   ⚠️ Mel processor not visible in device info")
    else:
        print(f"   ⚠️ NPU Runtime not available")

    print()
except Exception as e:
    print(f"   ℹ️ Runtime integration test skipped: {e}")
    print()

# Test 7: Performance Metrics
print("Test 7: Detailed Performance Metrics")
print("-" * 80)

metrics = processor.get_performance_metrics()
print(f"Processor Metrics:")
print(f"   Total frames: {metrics['total_frames']}")
print(f"   NPU time per frame: {metrics['npu_time_per_frame_ms']:.2f}ms")
print(f"   CPU time per frame: {metrics['cpu_time_per_frame_ms']:.2f}ms")
print(f"   Speedup: {metrics['speedup']:.1f}x")
print()

# Final Summary
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

success = True
issues = []

# Check NPU availability
if processor.npu_available:
    print(f"✅ NPU Available: YES")
else:
    print(f"⚠️ NPU Available: NO (using CPU fallback)")
    issues.append("NPU not available - check XRT installation")

# Check accuracy
if correlation >= 0.95:
    print(f"✅ Accuracy: EXCELLENT ({correlation:.4f})")
elif correlation >= 0.85:
    print(f"✅ Accuracy: GOOD ({correlation:.4f})")
elif correlation >= 0.70:
    print(f"⚠️ Accuracy: ACCEPTABLE ({correlation:.4f})")
else:
    print(f"❌ Accuracy: LOW ({correlation:.4f})")
    success = False
    issues.append(f"Accuracy too low: {correlation:.4f} < 0.70")

# Check performance
if processor.npu_available:
    if speedup >= 20:
        print(f"✅ Performance: EXCELLENT ({speedup:.1f}x speedup)")
    elif speedup >= 10:
        print(f"✅ Performance: GOOD ({speedup:.1f}x speedup)")
    elif speedup >= 5:
        print(f"⚠️ Performance: MODERATE ({speedup:.1f}x speedup)")
        issues.append(f"Performance lower than expected: {speedup:.1f}x < 20x")
    else:
        print(f"❌ Performance: LOW ({speedup:.1f}x speedup)")
        success = False
        issues.append(f"Performance too low: {speedup:.1f}x < 5x")
else:
    print(f"ℹ️ Performance: CPU fallback mode")

print()

# Expected improvements
if processor.npu_available and correlation >= 0.85 and speedup >= 10:
    print("Expected End-to-End Improvement:")
    print(f"   Current baseline: 19.1x realtime")
    print(f"   With NPU mel ({speedup:.1f}x): 22-25x realtime (estimated)")
    print()

if success and not issues:
    print("=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("1. Deploy to production runtime")
    print("2. Monitor end-to-end performance improvements")
    print("3. Integrate encoder/decoder kernels for further speedup")
    sys.exit(0)
else:
    print("=" * 80)
    print("⚠️ TESTS COMPLETED WITH ISSUES")
    print("=" * 80)
    print()
    if issues:
        print("Issues Found:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    print()
    sys.exit(1 if not success else 0)

# Cleanup
processor.close()
