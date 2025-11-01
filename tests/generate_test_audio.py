#!/usr/bin/env python3
"""
Generate Test Audio Files

Creates synthetic audio files for pipeline integration and load testing:
- Multiple durations (1s, 5s, 10s, 30s)
- Different sample rates (16kHz for Whisper)
- Various content types (silence, sine wave, speech-like)
- WAV format (uncompressed)

Usage:
    # Generate all test audio files
    python generate_test_audio.py

    # Generate specific duration
    python generate_test_audio.py --duration 10

    # Custom output directory
    python generate_test_audio.py --output /path/to/audio

Author: CC-1L Multi-Stream Integration Team
Date: November 1, 2025
"""

import numpy as np
import wave
import argparse
from pathlib import Path
from typing import Tuple


def generate_silence(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Generate silence audio.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Audio samples as float32 array
    """
    num_samples = int(duration * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)


def generate_sine_wave(duration: float, frequency: float = 440.0, sample_rate: int = 16000) -> np.ndarray:
    """
    Generate sine wave audio.

    Args:
        duration: Duration in seconds
        frequency: Frequency in Hz
        sample_rate: Sample rate in Hz

    Returns:
        Audio samples as float32 array
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio * 0.5  # Reduce amplitude to avoid clipping


def generate_speech_like(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Generate speech-like audio (multiple sine waves + noise).

    This creates a more realistic audio signal that resembles human speech
    by combining multiple frequency components and some noise.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Audio samples as float32 array
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, dtype=np.float32)

    # Fundamental frequency (varies over time like speech)
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.5 * t)  # 120-150 Hz (male voice range)

    # Generate harmonics
    audio = np.zeros(num_samples, dtype=np.float32)
    for harmonic in range(1, 6):  # First 5 harmonics
        amplitude = 1.0 / harmonic  # Decreasing amplitude
        audio += amplitude * np.sin(2 * np.pi * harmonic * f0 * t)

    # Add some noise (simulates breath/consonants)
    noise = np.random.normal(0, 0.05, num_samples).astype(np.float32)
    audio += noise

    # Add amplitude modulation (simulates syllables)
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz modulation
    audio *= modulation

    # Normalize to prevent clipping
    audio = audio / np.max(np.abs(audio)) * 0.7

    return audio


def save_wav(audio: np.ndarray, output_path: Path, sample_rate: int = 16000):
    """
    Save audio to WAV file.

    Args:
        audio: Audio samples as float32 array (range: -1.0 to 1.0)
        output_path: Output file path
        sample_rate: Sample rate in Hz
    """
    # Convert float32 to int16 for WAV file
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(str(output_path), 'w') as wav_file:
        # Set parameters: 1 channel (mono), 2 bytes per sample (int16), sample_rate
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        # Write audio data
        wav_file.writeframes(audio_int16.tobytes())

    print(f"  Saved: {output_path} ({len(audio)/sample_rate:.1f}s, {len(audio_int16)*2} bytes)")


def generate_test_suite(output_dir: Path):
    """
    Generate comprehensive test audio suite.

    Creates:
    - test_audio.wav: 10s speech-like (default test audio)
    - test_1s.wav: 1 second speech-like
    - test_5s.wav: 5 seconds speech-like
    - test_30s.wav: 30 seconds speech-like
    - test_silence.wav: 5 seconds silence
    - test_tone.wav: 5 seconds 440Hz tone

    Args:
        output_dir: Directory to save audio files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Generating Test Audio Suite")
    print(f"{'='*70}")
    print(f"  Output directory: {output_dir}")
    print(f"  Sample rate: 16000 Hz (Whisper standard)")
    print(f"  Format: WAV (16-bit PCM mono)")
    print(f"{'='*70}\n")

    # Default test audio (10s speech-like)
    print("Generating test_audio.wav (default, 10s speech-like)...")
    audio = generate_speech_like(10.0)
    save_wav(audio, output_dir / "test_audio.wav")

    # 1 second
    print("\nGenerating test_1s.wav...")
    audio = generate_speech_like(1.0)
    save_wav(audio, output_dir / "test_1s.wav")

    # 5 seconds
    print("\nGenerating test_5s.wav...")
    audio = generate_speech_like(5.0)
    save_wav(audio, output_dir / "test_5s.wav")

    # 30 seconds
    print("\nGenerating test_30s.wav...")
    audio = generate_speech_like(30.0)
    save_wav(audio, output_dir / "test_30s.wav")

    # Silence
    print("\nGenerating test_silence.wav (5s)...")
    audio = generate_silence(5.0)
    save_wav(audio, output_dir / "test_silence.wav")

    # Tone
    print("\nGenerating test_tone.wav (440Hz, 5s)...")
    audio = generate_sine_wave(5.0, frequency=440.0)
    save_wav(audio, output_dir / "test_tone.wav")

    print(f"\n{'='*70}")
    print(f"  ✅ Test audio suite generated successfully!")
    print(f"{'='*70}")
    print(f"  Files created:")
    print(f"    - test_audio.wav (default, 10s)")
    print(f"    - test_1s.wav")
    print(f"    - test_5s.wav")
    print(f"    - test_30s.wav")
    print(f"    - test_silence.wav")
    print(f"    - test_tone.wav")
    print(f"{'='*70}\n")

    print(f"  Usage:")
    print(f"    # Run integration tests")
    print(f"    pytest test_pipeline_integration.py -v")
    print(f"    ")
    print(f"    # Run load tests")
    print(f"    python load_test_pipeline.py")
    print(f"    ")
    print(f"    # Test with specific audio")
    print(f"    python load_test_pipeline.py --audio {output_dir / 'test_5s.wav'}")
    print(f"{'='*70}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate test audio files")
    parser.add_argument("--output", type=Path, help="Output directory (default: tests/audio/)")
    parser.add_argument("--duration", type=float, help="Generate single file with specific duration")

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        # Default to tests/audio/
        output_dir = Path(__file__).parent / "audio"

    if args.duration:
        # Generate single file
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating {args.duration}s speech-like audio...")
        audio = generate_speech_like(args.duration)
        output_path = output_dir / f"test_{args.duration}s.wav"
        save_wav(audio, output_path)
        print(f"\n✅ Generated: {output_path}\n")
    else:
        # Generate full suite
        generate_test_suite(output_dir)


if __name__ == "__main__":
    main()
