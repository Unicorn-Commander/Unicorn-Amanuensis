#!/usr/bin/env python3
"""
Test with real speech audio to get accurate performance metrics
"""

import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_with_real_audio():
    """Test with real speech audio"""
    from faster_whisper import WhisperModel

    # Find JFK audio
    audio_path = Path(__file__).parent / "npu/npu_optimization/mel_kernels/test_audio_jfk.wav"

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return

    logger.info("=" * 70)
    logger.info("REAL AUDIO PERFORMANCE TEST")
    logger.info("=" * 70)
    logger.info(f"Audio file: {audio_path.name}")

    # Get audio duration
    import wave
    with wave.open(str(audio_path), 'rb') as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        duration = frames / float(rate)

    logger.info(f"Duration: {duration:.1f}s")

    # Test different configurations
    configs = [
        {'name': 'Base + VAD', 'model': 'base', 'vad': True},
        {'name': 'Base (no VAD)', 'model': 'base', 'vad': False},
        {'name': 'Tiny + VAD', 'model': 'tiny', 'vad': True},
        {'name': 'Small + VAD', 'model': 'small', 'vad': True},
    ]

    results = []

    for config in configs:
        logger.info(f"\n--- {config['name']} ---")

        try:
            model = WhisperModel(config['model'], device='cpu', compute_type='int8')

            start = time.time()
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=5,
                vad_filter=config['vad'],
                word_timestamps=True,
                language='en'
            )
            segment_list = list(segments)  # Consume generator
            elapsed = time.time() - start

            rtf = duration / elapsed
            extrapolated_1hr = 3600 / rtf

            result = {
                'config': config['name'],
                'duration': duration,
                'processing_time': elapsed,
                'rtf': rtf,
                '1hr_time': extrapolated_1hr,
                'num_segments': len(segment_list)
            }
            results.append(result)

            logger.info(f"  Processing time: {elapsed:.2f}s")
            logger.info(f"  RTF: {rtf:.1f}x")
            logger.info(f"  1 hour would take: {extrapolated_1hr:.1f}s ({extrapolated_1hr/60:.1f} min)")
            logger.info(f"  Segments: {len(segment_list)}")

            # Show first segment as verification
            if segment_list:
                logger.info(f"  First segment: \"{segment_list[0].text.strip()}\"")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Configuration':<20} {'RTF':<12} {'1hr Time':<20} {'vs Base'}")
    logger.info("-" * 70)

    base_rtf = next((r['rtf'] for r in results if 'Base + VAD' in r['config']), None)

    for r in results:
        speedup = r['rtf'] / base_rtf if base_rtf else 1.0
        time_str = f"{r['1hr_time']:.1f}s ({r['1hr_time']/60:.1f} min)"
        logger.info(f"{r['config']:<20} {r['rtf']:<12.1f}x {time_str:<20} {speedup:.2f}x")

    return results

def main():
    logger.info("🎯 REAL AUDIO BENCHMARK - Current System Performance\n")

    results = test_with_real_audio()

    if results:
        base_result = next((r for r in results if 'Base + VAD' in r['config']), None)

        if base_result:
            logger.info("\n" + "=" * 70)
            logger.info("🎯 CURRENT SYSTEM STATUS")
            logger.info("=" * 70)
            logger.info(f"\nActual Performance: {base_result['rtf']:.1f}x realtime")
            logger.info(f"1 hour audio: {base_result['1hr_time']/60:.1f} minutes")
            logger.info(f"\nTarget Performance: 220x realtime")
            logger.info(f"1 hour audio: 1 minute (16.4 seconds)")
            logger.info(f"\nGap: {220/base_result['rtf']:.1f}x speedup needed")
            logger.info("\n" + "=" * 70)

if __name__ == '__main__':
    main()
