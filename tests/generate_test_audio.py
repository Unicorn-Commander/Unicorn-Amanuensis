#!/usr/bin/env python3
"""
Generate test audio files for performance benchmarking

Creates:
- test_1s.wav: 1 second of speech-like audio
- test_5s.wav: 5 seconds of speech-like audio  
- test_30s.wav: 30 seconds of speech-like audio
- test_silence.wav: 5 seconds of silence
"""

import numpy as np
from scipy.io import wavfile
from pathlib import Path

def generate_speech_like_audio(duration_s, sample_rate=16000):
    """Generate speech-like audio (mixed frequencies)"""
    t = np.linspace(0, duration_s, int(duration_s * sample_rate))

    # Mix of frequencies typical in speech (100-8000 Hz)
    audio = np.zeros_like(t)

    # Fundamental frequency (pitch)
    audio += 0.3 * np.sin(2 * np.pi * 150 * t)

    # Formants (vowel-like resonances)
    audio += 0.2 * np.sin(2 * np.pi * 500 * t)
    audio += 0.15 * np.sin(2 * np.pi * 1500 * t)
    audio += 0.1 * np.sin(2 * np.pi * 2500 * t)

    # Add some noise (consonants)
    audio += 0.05 * np.random.randn(len(t))

    # Amplitude modulation (speech rhythm)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    audio *= envelope

    # Normalize to int16 range
    audio = audio / np.max(np.abs(audio))
    audio = (audio * 32767 * 0.8).astype(np.int16)

    return audio

def main():
    """Generate all test audio files"""
    output_dir = Path(__file__).parent
    sample_rate = 16000

    print("Generating test audio files...")
    print(f"Output directory: {output_dir}")
    print()

    # 1 second audio
    print("Generating test_1s.wav...")
    audio_1s = generate_speech_like_audio(1.0, sample_rate)
    wavfile.write(output_dir / "test_1s.wav", sample_rate, audio_1s)
    print(f"  Created: {len(audio_1s) / sample_rate:.1f}s @ {sample_rate}Hz")

    # 5 second audio
    print("Generating test_5s.wav...")
    audio_5s = generate_speech_like_audio(5.0, sample_rate)
    wavfile.write(output_dir / "test_5s.wav", sample_rate, audio_5s)
    print(f"  Created: {len(audio_5s) / sample_rate:.1f}s @ {sample_rate}Hz")

    # 30 second audio
    print("Generating test_30s.wav...")
    audio_30s = generate_speech_like_audio(30.0, sample_rate)
    wavfile.write(output_dir / "test_30s.wav", sample_rate, audio_30s)
    print(f"  Created: {len(audio_30s) / sample_rate:.1f}s @ {sample_rate}Hz")

    # Silence
    print("Generating test_silence.wav...")
    silence = np.zeros(5 * sample_rate, dtype=np.int16)
    wavfile.write(output_dir / "test_silence.wav", sample_rate, silence)
    print(f"  Created: 5.0s @ {sample_rate}Hz (silence)")

    print()
    print("✅ All test audio files generated successfully!")
    print()
    print("Files:")
    for wav_file in sorted(output_dir.glob("test_*.wav")):
        size_kb = wav_file.stat().st_size / 1024
        print(f"  {wav_file.name}: {size_kb:.1f} KB")

if __name__ == "__main__":
    main()
