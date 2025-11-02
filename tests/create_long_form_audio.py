#!/usr/bin/env python3
"""
Create Long-Form Test Audio Files

Generates synthetic speech-like audio for testing long-form transcription:
- test_30s.wav: 30 seconds
- test_60s.wav: 60 seconds
- test_120s.wav: 120 seconds

Uses speech-like frequency patterns to simulate realistic audio without
actual speech content.

Author: CC-1L Buffer Management Team
Date: November 2, 2025
Status: Week 18 Long-Form Audio Testing
"""

import numpy as np
import wave
import os
from pathlib import Path

# Constants
SAMPLE_RATE = 16000  # WhisperX uses 16kHz
CHANNELS = 1  # Mono
SAMPLE_WIDTH = 2  # 16-bit PCM

# Output directory
OUTPUT_DIR = Path(__file__).parent / "audio"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_speech_like_audio(duration_s: int, seed: int = 42) -> np.ndarray:
    """
    Generate speech-like synthetic audio.

    Uses overlapping sine waves with speech-typical frequencies (85-255 Hz)
    and amplitude modulation to simulate speech patterns.

    Args:
        duration_s: Audio duration in seconds
        seed: Random seed for reproducibility

    Returns:
        NumPy array of audio samples (int16)
    """
    np.random.seed(seed)

    samples = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, samples)

    # Speech fundamental frequency range: 85-255 Hz (male to female range)
    # We'll create patterns that sound like different speakers
    frequencies = [
        (85, 110),    # Low male voice
        (110, 135),   # Medium male voice
        (135, 165),   # High male voice / low female voice
        (165, 200),   # Medium female voice
        (200, 255),   # High female voice
    ]

    # Generate audio in segments (simulate different speakers/words)
    segment_duration = 0.5  # 500ms segments (typical word duration)
    segment_samples = int(segment_duration * SAMPLE_RATE)
    num_segments = int(np.ceil(samples / segment_samples))

    audio = np.zeros(samples, dtype=np.float32)

    for seg_idx in range(num_segments):
        seg_start = seg_idx * segment_samples
        seg_end = min(seg_start + segment_samples, samples)
        seg_len = seg_end - seg_start

        # Random frequency range for this segment
        freq_range = frequencies[seg_idx % len(frequencies)]
        f0 = np.random.uniform(*freq_range)  # Fundamental frequency

        # Generate segment with harmonics (like speech formants)
        seg_audio = np.zeros(seg_len, dtype=np.float32)
        seg_t = t[seg_start:seg_end] - t[seg_start]

        # Add fundamental + harmonics (simulate formants)
        for harmonic in range(1, 4):  # 3 harmonics
            freq = f0 * harmonic
            amplitude = 1.0 / harmonic  # Harmonics decay in amplitude
            seg_audio += amplitude * np.sin(2 * np.pi * freq * seg_t)

        # Amplitude modulation (simulate pitch/loudness variation)
        mod_freq = np.random.uniform(2, 5)  # 2-5 Hz modulation
        envelope = 0.7 + 0.3 * np.sin(2 * np.pi * mod_freq * seg_t)
        seg_audio *= envelope

        # Add some noise (simulate breathiness)
        noise = np.random.normal(0, 0.05, seg_len)
        seg_audio += noise

        # Apply segment to audio
        audio[seg_start:seg_end] = seg_audio

        # Add silence between some segments (simulate pauses)
        if seg_idx < num_segments - 1 and np.random.random() < 0.3:
            # 30% chance of short pause
            pause_samples = int(0.1 * SAMPLE_RATE)  # 100ms pause
            next_seg_start = seg_end
            next_seg_end = min(next_seg_start + pause_samples, samples)
            audio[next_seg_start:next_seg_end] = 0

    # Normalize to prevent clipping
    audio = audio / np.max(np.abs(audio)) * 0.9

    # Convert to int16 PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    return audio_int16


def write_wav(filename: str, audio: np.ndarray):
    """
    Write audio to WAV file.

    Args:
        filename: Output filename (in OUTPUT_DIR)
        audio: Audio samples (int16)
    """
    filepath = OUTPUT_DIR / filename

    with wave.open(str(filepath), 'wb') as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio.tobytes())

    file_size = os.path.getsize(filepath)
    duration = len(audio) / SAMPLE_RATE

    print(f"✅ Created: {filepath}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Size: {file_size / 1024:.1f} KB")
    print(f"   Samples: {len(audio):,}")
    print()


def main():
    """Generate all test audio files"""
    print("="*70)
    print("  Long-Form Test Audio Generator")
    print("="*70)
    print()

    # Generate test audio files
    durations = [
        (30, "test_30s.wav"),
        (60, "test_60s.wav"),
        (120, "test_120s.wav"),
    ]

    for duration, filename in durations:
        print(f"Generating {filename} ({duration}s)...")
        audio = create_speech_like_audio(duration)
        write_wav(filename, audio)

    print("="*70)
    print("  All test audio files created successfully!")
    print("="*70)
    print()
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Print summary
    print("Summary:")
    for duration, filename in durations:
        filepath = OUTPUT_DIR / filename
        if filepath.exists():
            size = os.path.getsize(filepath) / 1024
            print(f"  {filename}: {duration}s, {size:.1f} KB")

    print()
    print("You can now test with:")
    print(f"  python tests/integration_test_week15.py")
    print(f"  MAX_AUDIO_DURATION={durations[-1][0]} python -m uvicorn xdna2.server:app --port 9000")
    print()


if __name__ == "__main__":
    main()
