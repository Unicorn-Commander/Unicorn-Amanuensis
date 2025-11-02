#!/usr/bin/env python3
"""
Test Buffer Pool Configuration

Quick validation test for Week 18 buffer pool configuration changes.
Tests that buffer sizes are calculated correctly based on MAX_AUDIO_DURATION.

Author: CC-1L Buffer Management Team
Date: November 2, 2025
"""

import os
import sys
from pathlib import Path

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from buffer_pool import GlobalBufferManager


def test_buffer_config(max_audio_duration):
    """
    Test buffer configuration for given max audio duration.

    Args:
        max_audio_duration: Maximum audio duration in seconds
    """
    print(f"\nTesting buffer config for MAX_AUDIO_DURATION={max_audio_duration}s")
    print("-" * 70)

    # Calculate expected sizes
    SAMPLE_RATE = 16000
    MAX_AUDIO_SAMPLES = max_audio_duration * SAMPLE_RATE
    MAX_MEL_FRAMES = (MAX_AUDIO_SAMPLES // 160) * 2
    MAX_ENCODER_FRAMES = MAX_MEL_FRAMES

    print(f"Expected buffer sizes:")
    print(f"  Audio: {MAX_AUDIO_SAMPLES:,} samples ({MAX_AUDIO_SAMPLES*4/1024/1024:.1f} MB)")
    print(f"  Mel: {MAX_MEL_FRAMES:,} frames ({MAX_MEL_FRAMES*80*4/1024/1024:.1f} MB)")
    print(f"  Encoder: {MAX_ENCODER_FRAMES:,} frames ({MAX_ENCODER_FRAMES*512*4/1024/1024:.1f} MB)")

    # Reset singleton (for testing multiple configurations)
    GlobalBufferManager._instance = None

    # Create buffer manager
    manager = GlobalBufferManager.instance()

    # Configure with calculated sizes
    manager.configure({
        'mel': {
            'size': MAX_MEL_FRAMES * 80 * 4,
            'count': 3,
            'max_count': 5,
            'dtype': np.float32,
            'shape': (MAX_MEL_FRAMES, 80),
            'zero_on_release': False
        },
        'audio': {
            'size': MAX_AUDIO_SAMPLES * 4,
            'count': 2,
            'max_count': 5,
            'dtype': np.float32,
            'shape': (MAX_AUDIO_SAMPLES,),  # CRITICAL: Must specify shape!
            'zero_on_release': False
        },
        'encoder_output': {
            'size': MAX_ENCODER_FRAMES * 512 * 4,
            'count': 2,
            'max_count': 5,
            'dtype': np.float32,
            'shape': (MAX_ENCODER_FRAMES, 512),
            'zero_on_release': False
        }
    })

    print(f"\nBuffer pools created:")

    # Test acquiring buffers
    print(f"\nTesting buffer acquisition:")
    audio_buf = manager.acquire('audio')
    mel_buf = manager.acquire('mel')
    encoder_buf = manager.acquire('encoder_output')

    print(f"  Audio buffer shape: {audio_buf.shape} (expected: ({MAX_AUDIO_SAMPLES},))")
    print(f"  Mel buffer shape: {mel_buf.shape} (expected: ({MAX_MEL_FRAMES}, 80))")
    print(f"  Encoder buffer shape: {encoder_buf.shape} (expected: ({MAX_ENCODER_FRAMES}, 512))")

    # Verify shapes
    assert audio_buf.shape == (MAX_AUDIO_SAMPLES,), f"Audio buffer shape mismatch"
    assert mel_buf.shape == (MAX_MEL_FRAMES, 80), f"Mel buffer shape mismatch"
    assert encoder_buf.shape == (MAX_ENCODER_FRAMES, 512), f"Encoder buffer shape mismatch"

    print(f"\n  ✅ All buffer shapes correct!")

    # Test variable-sized data (e.g., 15s audio in 30s buffer)
    test_duration = min(15, max_audio_duration)
    test_samples = test_duration * SAMPLE_RATE
    test_audio = np.random.randn(test_samples).astype(np.float32)

    print(f"\nTesting variable-sized data ({test_duration}s audio in {max_audio_duration}s buffer):")
    print(f"  Test audio: {test_samples:,} samples")

    # This should work (data smaller than buffer)
    np.copyto(audio_buf[:len(test_audio)], test_audio)
    print(f"  ✅ Successfully copied {test_duration}s audio into {max_audio_duration}s buffer")

    # Release buffers
    manager.release('audio', audio_buf)
    manager.release('mel', mel_buf)
    manager.release('encoder_output', encoder_buf)

    print(f"\n  ✅ Buffers released successfully")

    # Get stats
    stats = manager.get_stats()
    print(f"\nBuffer pool statistics:")
    for pool_name, pool_stats in stats.items():
        print(f"  {pool_name}:")
        print(f"    Total buffers: {pool_stats['total_buffers']}")
        print(f"    Available: {pool_stats['buffers_available']}")
        print(f"    In use: {pool_stats['buffers_in_use']}")
        print(f"    Hit rate: {pool_stats['hit_rate']*100:.0f}%")

    # Cleanup
    manager.clear_all()

    print(f"\n✅ Test passed for MAX_AUDIO_DURATION={max_audio_duration}s\n")


def main():
    """Run tests for different durations"""
    print("="*70)
    print("  Buffer Configuration Validation Tests")
    print("="*70)

    test_durations = [10, 30, 60, 120]

    for duration in test_durations:
        try:
            test_buffer_config(duration)
        except Exception as e:
            print(f"\n❌ Test failed for {duration}s: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print("="*70)
    print("  All buffer configuration tests passed!")
    print("="*70)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
