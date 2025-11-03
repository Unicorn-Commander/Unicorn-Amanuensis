#!/usr/bin/env python3
"""
Test NPU Mel Preprocessing Runtime
Tests if the NPU mel preprocessing can load and execute with existing XCLBINs
"""

import sys
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Add NPU module to path
sys.path.insert(0, str(Path(__file__).parent / 'npu'))

def test_npu_import():
    """Test if NPUMelPreprocessor can be imported"""
    logger.info("=" * 70)
    logger.info("TEST 1: Import NPUMelPreprocessor")
    logger.info("=" * 70)

    try:
        from npu_mel_preprocessing import NPUMelPreprocessor
        logger.info("✅ SUCCESS: NPUMelPreprocessor imported")
        return True
    except Exception as e:
        logger.error(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_npu_initialization():
    """Test if NPU can initialize with existing XCLBINs"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Initialize NPU with mel_fft.xclbin")
    logger.info("=" * 70)

    try:
        from npu_mel_preprocessing import NPUMelPreprocessor

        # Try different XCLBIN files
        xclbin_candidates = [
            "npu/npu_optimization/mel_kernels/build/mel_fft.xclbin",
            "npu/npu_optimization/mel_kernels/build/mel_int8_final.xclbin",
            "npu/npu_optimization/mel_kernels/build/mel_int8_optimized.xclbin",
        ]

        for xclbin_path in xclbin_candidates:
            full_path = Path(__file__).parent / xclbin_path
            logger.info(f"\nTrying: {xclbin_path}")
            logger.info(f"  Full path: {full_path}")
            logger.info(f"  Exists: {full_path.exists()}")

            if not full_path.exists():
                logger.warning(f"  ⚠️  File not found, skipping")
                continue

            try:
                preprocessor = NPUMelPreprocessor(
                    xclbin_path=str(full_path),
                    fallback_to_cpu=True
                )

                if preprocessor.npu_available:
                    logger.info(f"  ✅ SUCCESS: NPU initialized with {xclbin_path}")
                    return preprocessor, str(full_path)
                else:
                    logger.warning(f"  ⚠️  NPU not available (fallback to CPU)")

            except Exception as e:
                logger.error(f"  ❌ FAILED: {e}")

        logger.warning("⚠️  No XCLBINs could initialize NPU - will use CPU fallback")
        # Still return CPU fallback preprocessor for testing
        preprocessor = NPUMelPreprocessor(fallback_to_cpu=True)
        return preprocessor, None

    except Exception as e:
        logger.error(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_mel_processing(preprocessor):
    """Test mel preprocessing with test audio"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Process test audio")
    logger.info("=" * 70)

    try:
        # Generate 1 second test audio (1000 Hz sine wave)
        sample_rate = 16000
        duration = 1.0
        freq = 1000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.5

        logger.info(f"Test audio: {len(audio)} samples ({duration}s) @ {freq} Hz")

        # Process
        mel_features = preprocessor.process_audio(audio)

        logger.info(f"✅ SUCCESS: Mel features shape: {mel_features.shape}")
        logger.info(f"  Expected: (80, ~100) for 1s audio")
        logger.info(f"  Min value: {mel_features.min():.4f}")
        logger.info(f"  Max value: {mel_features.max():.4f}")
        logger.info(f"  Mean value: {mel_features.mean():.4f}")

        # Get performance metrics
        metrics = preprocessor.get_performance_metrics()
        logger.info(f"\nPerformance Metrics:")
        logger.info(f"  Total frames: {metrics['total_frames']}")
        logger.info(f"  NPU available: {metrics['npu_available']}")
        if metrics['npu_available']:
            logger.info(f"  NPU time/frame: {metrics['npu_time_per_frame_ms']:.2f}ms")
            logger.info(f"  Expected speedup: {metrics['speedup']:.2f}x vs CPU")
        else:
            logger.info(f"  CPU time/frame: {metrics['cpu_time_per_frame_ms']:.2f}ms")

        return True

    except Exception as e:
        logger.error(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_librosa_comparison(preprocessor):
    """Compare NPU/CPU output with librosa (gold standard)"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Compare with librosa (accuracy validation)")
    logger.info("=" * 70)

    try:
        import librosa

        # Generate test audio
        sample_rate = 16000
        duration = 1.0
        freq = 1000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.5

        # Get NPU/CPU output
        mel_npu = preprocessor.process_audio(audio)

        # Get librosa output (gold standard)
        mel_librosa = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=80,
            fmin=0,
            fmax=sample_rate // 2
        )

        # Convert to log scale
        mel_librosa = np.log10(mel_librosa + 1e-10)

        logger.info(f"NPU/CPU output shape: {mel_npu.shape}")
        logger.info(f"Librosa output shape: {mel_librosa.shape}")

        # Compute correlation
        if mel_npu.shape == mel_librosa.shape:
            correlation = np.corrcoef(mel_npu.flatten(), mel_librosa.flatten())[0, 1]
            logger.info(f"\n📊 Correlation with librosa: {correlation:.4f}")

            if correlation > 0.95:
                logger.info(f"  ✅ EXCELLENT: >0.95 (production ready)")
            elif correlation > 0.80:
                logger.info(f"  ⚠️  GOOD: 0.80-0.95 (acceptable)")
            elif correlation > 0.50:
                logger.info(f"  ⚠️  FAIR: 0.50-0.80 (needs improvement)")
            else:
                logger.info(f"  ❌ POOR: <0.50 (needs recompilation)")
                logger.info(f"  → XCLBINs likely DO NOT include Oct 28 fixes")
                logger.info(f"  → Need to recompile with fixed C code")

            return correlation
        else:
            logger.warning(f"⚠️  Shape mismatch - cannot compute correlation")
            return None

    except Exception as e:
        logger.error(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all tests"""
    logger.info("\n" + "=" * 70)
    logger.info("NPU MEL PREPROCESSING RUNTIME TEST SUITE")
    logger.info("=" * 70)

    # Test 1: Import
    if not test_npu_import():
        logger.error("\n❌ CRITICAL: Cannot import NPUMelPreprocessor")
        return 1

    # Test 2: Initialization
    preprocessor, xclbin_path = test_npu_initialization()
    if preprocessor is None:
        logger.error("\n❌ CRITICAL: Cannot initialize preprocessor")
        return 1

    # Test 3: Processing
    if not test_mel_processing(preprocessor):
        logger.error("\n❌ CRITICAL: Cannot process audio")
        preprocessor.close()
        return 1

    # Test 4: Accuracy
    correlation = test_librosa_comparison(preprocessor)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"NPU Available: {preprocessor.npu_available}")
    if xclbin_path:
        logger.info(f"XCLBIN Used: {Path(xclbin_path).name}")
    else:
        logger.info(f"XCLBIN Used: None (CPU fallback)")

    if correlation is not None:
        logger.info(f"Librosa Correlation: {correlation:.4f}")
        if correlation > 0.95:
            logger.info("\n✅ RECOMMENDATION: Can enable NPU preprocessing NOW")
            logger.info("   Accuracy is excellent (>0.95 correlation)")
        elif correlation > 0.80:
            logger.info("\n⚠️  RECOMMENDATION: Can enable but may want to recompile")
            logger.info("   Accuracy is acceptable (0.80-0.95 correlation)")
        else:
            logger.info("\n❌ RECOMMENDATION: MUST recompile XCLBINs first")
            logger.info("   Accuracy is poor (<0.80 correlation)")
            logger.info("   Current XCLBINs do NOT include Oct 28 fixes")
    else:
        logger.info("\n⚠️  RECOMMENDATION: Test with real audio to verify accuracy")

    preprocessor.close()
    logger.info("\n" + "=" * 70)
    return 0

if __name__ == "__main__":
    exit(main())
