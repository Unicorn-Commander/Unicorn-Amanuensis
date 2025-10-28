#!/usr/bin/env python3
"""
Simple NPU test with synthetic audio (no dependencies needed)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import time
from npu_mel_preprocessing import NPUMelPreprocessor

print("=" * 70)
print("SIMPLE NPU MEL PREPROCESSING TEST")
print("=" * 70)

# Generate synthetic audio: 5-second sine wave sweep (1 kHz to 4 kHz)
print("\nGenerating test audio...")
sample_rate = 16000
duration = 5.0  # 5 seconds
t = np.linspace(0, duration, int(sample_rate * duration))

# Frequency sweep from 1 kHz to 4 kHz
f0 = 1000  # Start frequency
f1 = 4000  # End frequency
audio = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t).astype(np.float32)

print(f"  Duration: {duration}s")
print(f"  Sample rate: {sample_rate}Hz")
print(f"  Samples: {len(audio)}")
print(f"  Frequency sweep: {f0}Hz → {f1}Hz")

# Initialize NPU preprocessor
print("\nInitializing NPU preprocessor...")
preprocessor = NPUMelPreprocessor(fallback_to_cpu=True)

if preprocessor.npu_available:
    print("  ✅ NPU mode enabled")
    print("  XCLBIN loaded successfully")
else:
    print("  ⚠️  NPU not available - using CPU fallback")

# Process audio
print("\nProcessing audio on NPU...")
start = time.time()
mel_features = preprocessor.process_audio(audio)
elapsed = time.time() - start

rtf = duration / elapsed if elapsed > 0 else 0

# Get metrics
metrics = preprocessor.get_performance_metrics()

# Display results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Audio duration:    {duration:.2f}s")
print(f"Processing time:   {elapsed:.4f}s")
print(f"Real-time factor:  {rtf:.2f}x")
print(f"Output shape:      {mel_features.shape} (mels, frames)")
print(f"Backend:           {'NPU' if metrics['npu_available'] else 'CPU'}")
print(f"\nFrame statistics:")
print(f"  Total frames:    {metrics['total_frames']}")
print(f"  Avg time/frame:  {metrics['npu_time_per_frame_ms']:.2f}ms")

if metrics['npu_available']:
    print(f"\n✅ NPU acceleration working!")
    print(f"   Expected speedup: ~6x vs CPU librosa")
else:
    print(f"\n⚠️  Running in CPU fallback mode")
    print(f"   Install XRT and XCLBIN for NPU acceleration")

# Show mel spectrogram statistics
print(f"\nMel Spectrogram Statistics:")
print(f"  Min value:  {np.min(mel_features):.6f}")
print(f"  Max value:  {np.max(mel_features):.6f}")
print(f"  Mean value: {np.mean(mel_features):.6f}")
print(f"  Std dev:    {np.std(mel_features):.6f}")

# Show first few values
print(f"\nFirst frame (first 8 mel bins):")
print(mel_features[:8, 0])

# Cleanup
preprocessor.close()

print("\n" + "=" * 70)
print("TEST COMPLETE!")
print("=" * 70)
print()
