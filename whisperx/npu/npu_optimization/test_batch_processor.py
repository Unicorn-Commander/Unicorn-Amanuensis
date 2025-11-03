#!/usr/bin/env python3
"""
Test script for NPU Mel Processor Batch v2.0

Tests the batch processing functionality with various scenarios:
- Small batches (< 100 frames)
- Exact batch size (100 frames)
- Large batches (> 100 frames) requiring multiple kernel calls
- Empty audio
- Edge cases

Author: Magic Unicorn Unconventional Technology & Stuff Inc.
Date: November 1, 2025
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from npu_mel_processor_batch import NPUMelProcessorBatch
    print("✅ Successfully imported NPUMelProcessorBatch")
except ImportError as e:
    print(f"❌ Failed to import NPUMelProcessorBatch: {e}")
    sys.exit(1)


def generate_test_audio(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Generate test audio signal (sine wave sweep).

    Args:
        duration_sec: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        audio: Numpy array of audio samples (float32)
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, dtype=np.float32)

    # Frequency sweep from 100Hz to 8000Hz
    freq_start = 100
    freq_end = 8000
    freq = freq_start + (freq_end - freq_start) * (t / duration_sec)

    # Generate sine wave with varying frequency
    audio = 0.5 * np.sin(2 * np.pi * freq * t)

    return audio.astype(np.float32)


def test_batch_size(processor, batch_size: int, audio_duration: float = 1.0):
    """
    Test specific batch size.

    Args:
        processor: NPUMelProcessorBatch instance
        batch_size: Batch size to test
        audio_duration: Audio duration in seconds
    """
    print(f"\n{'='*70}")
    print(f"Test: Batch size {batch_size} frames")
    print(f"{'='*70}")

    # Generate test audio
    audio = generate_test_audio(audio_duration)
    n_samples = len(audio)

    # Expected number of frames
    hop_length = processor.hop_length
    frame_size = processor.frame_size
    expected_frames = (n_samples - frame_size) // hop_length + 1

    print(f"Audio duration: {audio_duration}s ({n_samples} samples)")
    print(f"Expected frames: {expected_frames}")
    print(f"Batch size: {batch_size}")
    print(f"Expected batches: {(expected_frames + batch_size - 1) // batch_size}")

    # Process
    try:
        start_time = time.time()
        mel_features = processor.process(audio)
        elapsed = time.time() - start_time

        # Validate output shape
        assert mel_features.shape[0] == processor.n_mels, f"Expected {processor.n_mels} mel bins, got {mel_features.shape[0]}"
        assert mel_features.shape[1] == expected_frames, f"Expected {expected_frames} frames, got {mel_features.shape[1]}"

        # Calculate performance
        rtf = (audio_duration / elapsed) if elapsed > 0 else 0

        print(f"\n✅ SUCCESS!")
        print(f"  Output shape: {mel_features.shape} (mels, frames)")
        print(f"  Processing time: {elapsed:.4f}s")
        print(f"  Realtime factor: {rtf:.2f}x")
        print(f"  Backend: {'NPU' if processor.npu_available else 'CPU'}")
        print(f"  Batches processed: {processor.batch_calls}")

        # Check for reasonable mel values
        mel_min = mel_features.min()
        mel_max = mel_features.max()
        mel_mean = mel_features.mean()

        print(f"  Mel stats: min={mel_min:.4f}, max={mel_max:.4f}, mean={mel_mean:.4f}")

        if mel_min == mel_max:
            print(f"  ⚠️ WARNING: All mel values are identical ({mel_min})")

        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases(processor):
    """Test edge cases."""
    print(f"\n{'='*70}")
    print("Test: Edge Cases")
    print(f"{'='*70}")

    # Test 1: Very short audio (< 1 frame)
    print("\n1. Very short audio (100 samples)")
    audio_short = generate_test_audio(0.00625)  # 100 samples
    try:
        mel = processor.process(audio_short)
        print(f"   ✅ Short audio: {mel.shape}")
    except Exception as e:
        print(f"   ⚠️ Short audio failed: {e}")

    # Test 2: Exact frame size
    print("\n2. Exact frame size (400 samples)")
    audio_exact = generate_test_audio(0.025)  # 400 samples
    try:
        mel = processor.process(audio_exact)
        print(f"   ✅ Exact frame: {mel.shape}")
    except Exception as e:
        print(f"   ⚠️ Exact frame failed: {e}")

    # Test 3: Exact batch size
    print("\n3. Exact batch size (100 frames)")
    # 100 frames = (100-1) * 160 + 400 = 16240 samples
    samples_needed = (processor.batch_size - 1) * processor.hop_length + processor.frame_size
    audio_batch = np.random.randn(samples_needed).astype(np.float32) * 0.5
    try:
        mel = processor.process(audio_batch)
        expected = processor.batch_size
        print(f"   ✅ Exact batch: {mel.shape} (expected {expected} frames)")
        assert mel.shape[1] == expected, f"Expected {expected} frames, got {mel.shape[1]}"
    except Exception as e:
        print(f"   ⚠️ Exact batch failed: {e}")

    # Test 4: Partial batch (batch_size + 50)
    print("\n4. Partial batch (150 frames)")
    frames_needed = processor.batch_size + 50
    samples_needed = (frames_needed - 1) * processor.hop_length + processor.frame_size
    audio_partial = np.random.randn(samples_needed).astype(np.float32) * 0.5
    try:
        mel = processor.process(audio_partial)
        expected = frames_needed
        print(f"   ✅ Partial batch: {mel.shape} (expected {expected} frames)")
        assert mel.shape[1] == expected, f"Expected {expected} frames, got {mel.shape[1]}"
    except Exception as e:
        print(f"   ⚠️ Partial batch failed: {e}")

    # Test 5: Multiple batches
    print("\n5. Multiple batches (250 frames)")
    frames_needed = 250
    samples_needed = (frames_needed - 1) * processor.hop_length + processor.frame_size
    audio_multi = np.random.randn(samples_needed).astype(np.float32) * 0.5
    try:
        mel = processor.process(audio_multi)
        expected = frames_needed
        print(f"   ✅ Multiple batches: {mel.shape} (expected {expected} frames)")
        assert mel.shape[1] == expected, f"Expected {expected} frames, got {mel.shape[1]}"
    except Exception as e:
        print(f"   ⚠️ Multiple batches failed: {e}")


def test_performance_comparison(processor_batch, processor_single=None):
    """Compare batch vs single-frame performance."""
    print(f"\n{'='*70}")
    print("Test: Performance Comparison")
    print(f"{'='*70}")

    # Generate longer audio for performance test
    audio = generate_test_audio(5.0)  # 5 seconds
    n_samples = len(audio)

    print(f"Audio: 5.0s ({n_samples} samples)")

    # Test batch processor
    print("\n1. Batch Processor:")
    start = time.time()
    mel_batch = processor_batch.process(audio)
    time_batch = time.time() - start
    rtf_batch = (5.0 / time_batch) if time_batch > 0 else 0

    print(f"   Time: {time_batch:.4f}s")
    print(f"   RTF: {rtf_batch:.2f}x")
    print(f"   Batches: {processor_batch.batch_calls}")

    # If single-frame processor available, compare
    if processor_single:
        print("\n2. Single-Frame Processor:")
        try:
            start = time.time()
            mel_single = processor_single.process(audio)
            time_single = time.time() - start
            rtf_single = (5.0 / time_single) if time_single > 0 else 0

            print(f"   Time: {time_single:.4f}s")
            print(f"   RTF: {rtf_single:.2f}x")

            # Calculate speedup
            speedup = time_single / time_batch if time_batch > 0 else 0
            print(f"\n   📊 Speedup: {speedup:.2f}x faster with batch processing")

            # Check if outputs match
            if mel_batch.shape == mel_single.shape:
                diff = np.abs(mel_batch - mel_single)
                max_diff = diff.max()
                mean_diff = diff.mean()
                print(f"   Difference: max={max_diff:.6f}, mean={mean_diff:.6f}")

                if max_diff < 0.01:
                    print(f"   ✅ Outputs match within tolerance")
                else:
                    print(f"   ⚠️ Outputs differ significantly")
        except Exception as e:
            print(f"   ⚠️ Single-frame test failed: {e}")


def main():
    """Main test function."""
    print("="*70)
    print("NPU Mel Processor Batch - Test Suite")
    print("="*70)

    # Test different batch sizes
    batch_sizes = [10, 50, 100, 200, 500, 1000]

    for batch_size in batch_sizes:
        print(f"\n{'#'*70}")
        print(f"Testing batch_size = {batch_size}")
        print(f"{'#'*70}")

        try:
            # Create batch processor
            processor = NPUMelProcessorBatch(
                batch_size=batch_size,
                fallback_to_cpu=True  # Allow CPU fallback for testing
            )

            # Test various scenarios
            test_batch_size(processor, batch_size, audio_duration=1.0)

            # Only run edge cases for default batch size
            if batch_size == 100:
                test_edge_cases(processor)

            # Cleanup
            processor.close()

        except Exception as e:
            print(f"\n❌ Failed to create processor with batch_size={batch_size}: {e}")
            import traceback
            traceback.print_exc()

    # Performance comparison test (batch_size=100 vs single-frame)
    print(f"\n{'#'*70}")
    print("Performance Comparison Test")
    print(f"{'#'*70}")

    try:
        processor_batch = NPUMelProcessorBatch(batch_size=100, fallback_to_cpu=True)

        # Try to import single-frame processor for comparison
        try:
            from npu_mel_processor import NPUMelProcessor
            processor_single = NPUMelProcessor(fallback_to_cpu=True)
            test_performance_comparison(processor_batch, processor_single)
            processor_single.close()
        except ImportError:
            print("⚠️ Single-frame processor not available for comparison")
            test_performance_comparison(processor_batch)

        processor_batch.close()

    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print("Test Suite Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
