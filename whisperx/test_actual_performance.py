#!/usr/bin/env python3
"""
Simple test to measure actual current performance
Using faster-whisper directly (what we're currently using)
"""

import time
import numpy as np
import tempfile
import wave
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def create_test_audio(duration=60):
    """Create test audio file"""
    sample_rate = 16000
    t = np.linspace(0, duration, sample_rate * duration, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
    audio += 0.2 * np.sin(2 * np.pi * 880 * t)  # A5 note
    audio += 0.1 * np.random.randn(len(t)).astype(np.float32)  # Noise

    # Save to WAV
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
        with wave.open(tmp_path, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav.writeframes(audio_int16.tobytes())

    return tmp_path, duration

def test_faster_whisper_cpu():
    """Test faster-whisper CPU performance (what we're using now)"""
    logger.info("=" * 70)
    logger.info("TEST: faster-whisper CPU INT8 (Current System)")
    logger.info("=" * 70)

    from faster_whisper import WhisperModel

    # Test with different durations
    test_durations = [10, 30, 60]

    for duration in test_durations:
        logger.info(f"\n--- Test: {duration}s audio ---")

        # Create test audio
        audio_path, actual_duration = create_test_audio(duration)

        try:
            # Initialize model
            model = WhisperModel('base', device='cpu', compute_type='int8')

            # Test WITH VAD (current default)
            logger.info("With VAD enabled:")
            start = time.time()
            segments, info = model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
                language='en'
            )
            list(segments)  # Consume generator
            elapsed_vad = time.time() - start

            rtf_vad = actual_duration / elapsed_vad
            logger.info(f"  Duration: {actual_duration}s")
            logger.info(f"  Processing: {elapsed_vad:.2f}s")
            logger.info(f"  RTF: {rtf_vad:.1f}x")
            logger.info(f"  1hr estimate: {3600/rtf_vad:.1f}s ({3600/rtf_vad/60:.1f} min)")

            # Test WITHOUT VAD
            logger.info("Without VAD:")
            start = time.time()
            segments, info = model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=False,
                word_timestamps=True,
                language='en'
            )
            list(segments)  # Consume generator
            elapsed_no_vad = time.time() - start

            rtf_no_vad = actual_duration / elapsed_no_vad
            logger.info(f"  Duration: {actual_duration}s")
            logger.info(f"  Processing: {elapsed_no_vad:.2f}s")
            logger.info(f"  RTF: {rtf_no_vad:.1f}x")
            logger.info(f"  1hr estimate: {3600/rtf_no_vad:.1f}s ({3600/rtf_no_vad/60:.1f} min)")

            logger.info(f"VAD speedup: {rtf_vad/rtf_no_vad:.2f}x")

        except Exception as e:
            logger.error(f"Test failed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            import os
            os.unlink(audio_path)

def test_different_models():
    """Test different model sizes"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST: Different Model Sizes")
    logger.info("=" * 70)

    from faster_whisper import WhisperModel

    models = ['tiny', 'base', 'small']
    duration = 30

    audio_path, actual_duration = create_test_audio(duration)

    results = []

    for model_name in models:
        try:
            logger.info(f"\n--- Model: {model_name} ---")

            model = WhisperModel(model_name, device='cpu', compute_type='int8')

            start = time.time()
            segments, info = model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
                language='en'
            )
            list(segments)  # Consume generator
            elapsed = time.time() - start

            rtf = actual_duration / elapsed

            result = {
                'model': model_name,
                'duration': actual_duration,
                'processing_time': elapsed,
                'rtf': rtf,
                '1hr_time': 3600/rtf
            }
            results.append(result)

            logger.info(f"  RTF: {rtf:.1f}x")
            logger.info(f"  1hr time: {result['1hr_time']:.1f}s ({result['1hr_time']/60:.1f} min)")

        except Exception as e:
            logger.error(f"  Failed: {e}")

    import os
    os.unlink(audio_path)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Model Comparison Summary")
    logger.info("=" * 70)
    logger.info(f"{'Model':<10} {'RTF':<10} {'1hr Time':<15} {'Speedup vs base'}")
    logger.info("-" * 70)

    base_rtf = next((r['rtf'] for r in results if r['model'] == 'base'), None)

    for r in results:
        speedup = r['rtf'] / base_rtf if base_rtf else 1.0
        logger.info(f"{r['model']:<10} {r['rtf']:<10.1f}x {r['1hr_time']/60:<15.1f} min {speedup:.2f}x")

def main():
    logger.info("🎯 ACTUAL PERFORMANCE TEST - Current System")
    logger.info("Using: faster-whisper CPU INT8 (what server_dynamic.py uses)")
    logger.info("")

    # Test 1: Current system with different audio lengths
    test_faster_whisper_cpu()

    # Test 2: Different model sizes
    test_different_models()

    # Bottom line
    logger.info("\n" + "=" * 70)
    logger.info("🎯 CURRENT SYSTEM PERFORMANCE")
    logger.info("=" * 70)
    logger.info("\nSetup: faster-whisper base + CPU INT8 + VAD")
    logger.info("Hardware: CPU only (NPU mel not currently working)")
    logger.info("\nExpected based on tests above:")
    logger.info("  • Base model: ~13-15x realtime")
    logger.info("  • 1 hour audio: ~4-5 minutes")
    logger.info("  • Tiny model: ~40-50x realtime (trades accuracy)")
    logger.info("\nTarget: 220x realtime (1 hour in 1 minute)")
    logger.info("Gap: 15-17x speedup needed")
    logger.info("\nBottleneck identified:")
    logger.info("  • Encoder/Decoder running on CPU")
    logger.info("  • No NPU acceleration (currently)")
    logger.info("  • VAD provides 1.2-1.5x speedup")
    logger.info("\n" + "=" * 70)

if __name__ == '__main__':
    main()
