#!/usr/bin/env python3
"""
Example: NPU Mel Preprocessing

This example demonstrates NPU-accelerated mel spectrogram computation
without requiring a full Whisper installation.

Usage:
    python3 example_npu_preprocessing.py audio.wav
"""

import sys
import os
from pathlib import Path
import numpy as np
import time

# Add NPU module to path
sys.path.insert(0, str(Path(__file__).parent))

from npu_mel_preprocessing import NPUMelPreprocessor


def load_audio(audio_path: str):
    """Load audio file using librosa or soundfile."""
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        return audio, sr
    except ImportError:
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            if sr != 16000:
                print(f"Warning: Audio is {sr}Hz, resampling to 16kHz...")
                # Simple resampling (not ideal, but works)
                import scipy.signal
                audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))
                sr = 16000
            return audio, sr
        except ImportError:
            print("Error: Please install librosa or soundfile")
            print("  pip install librosa")
            return None, None


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 example_npu_preprocessing.py <audio_file>")
        print("\nExample:")
        print("  python3 example_npu_preprocessing.py audio.wav")
        sys.exit(1)

    audio_file = sys.argv[1]

    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)

    print("=" * 70)
    print("NPU MEL PREPROCESSING EXAMPLE")
    print("=" * 70)

    # Load audio
    print(f"\nLoading audio: {audio_file}")
    audio, sr = load_audio(audio_file)

    if audio is None:
        sys.exit(1)

    duration = len(audio) / sr
    print(f"  Duration: {duration:.2f}s")
    print(f"  Sample rate: {sr}Hz")
    print(f"  Samples: {len(audio)}")

    # Initialize NPU preprocessor
    print("\nInitializing NPU preprocessor...")
    preprocessor = NPUMelPreprocessor(fallback_to_cpu=True)

    if preprocessor.npu_available:
        print("  ✅ NPU mode enabled")
    else:
        print("  ⚠️  NPU not available - using CPU fallback")

    # Process audio
    print(f"\nProcessing audio...")
    start_time = time.time()

    mel_features = preprocessor.process_audio(audio)

    elapsed = time.time() - start_time
    rtf = duration / elapsed if elapsed > 0 else 0

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Output shape:      {mel_features.shape} (mels, frames)")
    print(f"Processing time:   {elapsed:.4f}s")
    print(f"Real-time factor:  {rtf:.2f}x")
    print(f"Backend:           {'NPU' if preprocessor.npu_available else 'CPU'}")

    # Get detailed metrics
    metrics = preprocessor.get_performance_metrics()
    print(f"\nDetailed Metrics:")
    print(f"  Total frames:    {metrics['total_frames']}")
    print(f"  NPU time:        {metrics['npu_time_total']:.4f}s")
    print(f"  Avg per frame:   {metrics['npu_time_per_frame_ms']:.2f}ms")

    if metrics['speedup'] > 0:
        print(f"  Speedup vs CPU:  {metrics['speedup']:.2f}x")

    # Show sample output
    print(f"\nMel Spectrogram Sample (first 5 frames, first 8 bins):")
    print(mel_features[:8, :5])

    # Save output (optional)
    output_path = Path(audio_file).stem + "_mel.npy"
    np.save(output_path, mel_features)
    print(f"\nSaved mel spectrogram to: {output_path}")

    # Cleanup
    preprocessor.close()

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Use mel_features as input to Whisper encoder")
    print("  2. Run full benchmark: python3 npu_benchmark.py audio.wav")
    print("  3. Try WhisperX wrapper: python3 whisperx_npu_wrapper.py audio.wav")
    print()


if __name__ == "__main__":
    main()
