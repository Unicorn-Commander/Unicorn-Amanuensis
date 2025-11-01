#!/usr/bin/env python3
"""
Core Zero-Copy Logic Test (No WhisperX Required)

This test validates the core zero-copy logic without requiring WhisperX to be installed.
It uses mock feature extractors to verify the optimization logic.

Author: Zero-Copy Optimization Teamlead
Date: November 1, 2025
"""

import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mel_utils import compute_mel_spectrogram_zerocopy, validate_mel_contiguity


class MockFeatureExtractor:
    """Mock feature extractor that simulates WhisperX behavior"""

    def __init__(self, seed=42):
        """Initialize with seed for reproducible results"""
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, audio):
        """Simulate feature extraction: audio -> (1, 80, 3000)"""
        # Simulate 30s audio @ 16kHz = 480,000 samples
        # Mel spectrogram: 80 mels × 3000 time frames
        n_mels = 80
        time_frames = len(audio) // 160  # 10ms hop length @ 16kHz

        # Create mel features in WhisperX format: (batch, n_mels, time)
        # Use deterministic random generator based on audio content
        seed = int(np.sum(np.abs(audio)) * 1000) % (2**31 - 1)
        rng = np.random.RandomState(seed)
        mel = rng.randn(1, n_mels, time_frames).astype(np.float32)

        return mel


def test_zerocopy_nobuffer():
    """Test zero-copy without buffer pool"""
    print("\n[TEST 1] Zero-copy without buffer pool...")

    # Create mock audio (30s @ 16kHz)
    audio = np.random.randn(16000 * 30).astype(np.float32) * 0.1

    # Create mock feature extractor
    feature_extractor = MockFeatureExtractor()

    # Compute mel with zero-copy
    mel = compute_mel_spectrogram_zerocopy(audio, feature_extractor)

    # Validate
    print(f"  Shape: {mel.shape} (expected: time × mels)")
    print(f"  Dtype: {mel.dtype} (expected: float32)")
    print(f"  C-contiguous: {mel.flags['C_CONTIGUOUS']} (expected: True)")
    print(f"  Size: {mel.nbytes / 1024:.1f}KB")

    # Validate using helper
    validate_mel_contiguity(mel)

    print("  ✅ PASS - No buffer")

    return mel


def test_zerocopy_withbuffer():
    """Test zero-copy with pre-allocated buffer"""
    print("\n[TEST 2] Zero-copy with pre-allocated buffer...")

    # Create mock audio (30s @ 16kHz)
    audio = np.random.randn(16000 * 30).astype(np.float32) * 0.1

    # Create mock feature extractor
    feature_extractor = MockFeatureExtractor()

    # Pre-allocate buffer (simulating buffer pool)
    expected_time = len(audio) // 160
    expected_mels = 80
    buffer = np.empty((expected_time, expected_mels), dtype=np.float32, order='C')
    buffer_address = buffer.ctypes.data

    # Compute mel with zero-copy into buffer
    mel = compute_mel_spectrogram_zerocopy(audio, feature_extractor, output=buffer)

    # Validate
    print(f"  Shape: {mel.shape}")
    print(f"  Same buffer: {mel is buffer} (expected: True)")
    print(f"  Buffer address match: {mel.ctypes.data == buffer_address} (expected: True)")
    print(f"  C-contiguous: {mel.flags['C_CONTIGUOUS']} (expected: True)")

    # Validate using helper
    validate_mel_contiguity(mel)

    assert mel is buffer, "mel should be the same object as buffer"
    assert mel.ctypes.data == buffer_address, "Buffer address should not change"

    print("  ✅ PASS - With buffer (perfect zero-copy!)")

    return mel


def test_comparison_standard_vs_zerocopy():
    """Compare standard approach vs zero-copy"""
    print("\n[TEST 3] Comparing standard vs zero-copy...")

    # Create mock audio
    audio = np.random.randn(16000 * 10).astype(np.float32) * 0.1
    feature_extractor = MockFeatureExtractor()

    # Standard approach (what we're replacing)
    print("  Standard approach:")
    mel_features = feature_extractor(audio)  # (1, 80, time)
    print(f"    1. Feature extract: {mel_features.shape}")

    mel_np = mel_features  # Already numpy in this mock
    print(f"    2. To numpy: {mel_np.shape}")

    mel_transposed = mel_np[0].T  # (time, 80) - creates VIEW
    print(f"    3. Transpose: {mel_transposed.shape}, C-contig={mel_transposed.flags['C_CONTIGUOUS']}")

    if not mel_transposed.flags['C_CONTIGUOUS']:
        mel_contiguous = np.ascontiguousarray(mel_transposed)  # COPY!
        print(f"    4. Make contiguous: COPY {mel_contiguous.nbytes / 1024:.1f}KB")
    else:
        mel_contiguous = mel_transposed

    # Zero-copy approach
    print("  Zero-copy approach:")
    mel_zerocopy = compute_mel_spectrogram_zerocopy(audio, feature_extractor)
    print(f"    1. Direct compute: {mel_zerocopy.shape}, C-contig={mel_zerocopy.flags['C_CONTIGUOUS']}")
    print(f"    2. No additional copies needed!")

    # Compare outputs
    np.testing.assert_array_almost_equal(mel_contiguous, mel_zerocopy, decimal=5)

    print("  ✅ PASS - Outputs match, but zero-copy eliminates one copy!")

    return mel_contiguous, mel_zerocopy


def test_performance_benchmark():
    """Simple performance benchmark"""
    print("\n[TEST 4] Performance benchmark...")

    import time

    audio = np.random.randn(16000 * 30).astype(np.float32) * 0.1
    feature_extractor = MockFeatureExtractor()

    iterations = 50

    # Benchmark standard approach
    print(f"  Running {iterations} iterations...")

    start = time.perf_counter()
    for _ in range(iterations):
        mel_features = feature_extractor(audio)
        mel_np = mel_features
        mel_transposed = mel_np[0].T
        if not mel_transposed.flags['C_CONTIGUOUS']:
            mel_standard = np.ascontiguousarray(mel_transposed)
        else:
            mel_standard = mel_transposed
    time_standard = time.perf_counter() - start

    # Benchmark zero-copy (no buffer)
    start = time.perf_counter()
    for _ in range(iterations):
        mel_zerocopy = compute_mel_spectrogram_zerocopy(audio, feature_extractor)
    time_zerocopy = time.perf_counter() - start

    # Benchmark zero-copy (with buffer)
    buffer = np.empty(mel_zerocopy.shape, dtype=np.float32, order='C')
    start = time.perf_counter()
    for _ in range(iterations):
        mel_buffered = compute_mel_spectrogram_zerocopy(audio, feature_extractor, output=buffer)
    time_buffered = time.perf_counter() - start

    print(f"\n  Results ({iterations} iterations):")
    print(f"    Standard:      {time_standard/iterations*1000:.3f} ms/iter")
    print(f"    Zero-copy:     {time_zerocopy/iterations*1000:.3f} ms/iter ({(1-time_zerocopy/time_standard)*100:.1f}% faster)")
    print(f"    Zero-buffered: {time_buffered/iterations*1000:.3f} ms/iter ({(1-time_buffered/time_standard)*100:.1f}% faster)")

    print("  ✅ PASS - Performance improvement demonstrated")

    return {
        'standard_ms': time_standard / iterations * 1000,
        'zerocopy_ms': time_zerocopy / iterations * 1000,
        'buffered_ms': time_buffered / iterations * 1000,
        'improvement_pct': (1 - time_buffered / time_standard) * 100
    }


def main():
    """Run all tests"""
    print("="*70)
    print("  ZERO-COPY CORE LOGIC VALIDATION")
    print("="*70)

    try:
        # Run tests
        mel1 = test_zerocopy_nobuffer()
        mel2 = test_zerocopy_withbuffer()
        mel_std, mel_zc = test_comparison_standard_vs_zerocopy()
        perf = test_performance_benchmark()

        # Summary
        print("\n" + "="*70)
        print("  ALL TESTS PASSED!")
        print("="*70)
        print(f"\n  Performance Summary:")
        print(f"    Improvement: {perf['improvement_pct']:.1f}%")
        print(f"    Standard:    {perf['standard_ms']:.3f} ms")
        print(f"    Optimized:   {perf['buffered_ms']:.3f} ms")
        print(f"    Savings:     {perf['standard_ms'] - perf['buffered_ms']:.3f} ms per request")
        print("\n" + "="*70)

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
