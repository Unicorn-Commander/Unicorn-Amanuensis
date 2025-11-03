#!/usr/bin/env python3
"""
Generate speech-like audio for testing
Creates audio with formant-like structure (vowel sounds)
"""

import numpy as np
import soundfile as sf

def generate_speech_like_audio(duration=5.0, sample_rate=16000):
    """Generate audio that sounds more like speech than a pure sine wave"""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create speech-like formants (vowel "ah" sound)
    # Multiple frequencies to simulate formants
    f0 = 120  # Fundamental frequency (pitch)
    f1 = 700  # First formant
    f2 = 1220 # Second formant
    f3 = 2600 # Third formant

    # Generate harmonics
    audio = np.zeros_like(t)
    audio += 0.4 * np.sin(2 * np.pi * f0 * t)      # Fundamental
    audio += 0.3 * np.sin(2 * np.pi * f1 * t)      # Formant 1
    audio += 0.2 * np.sin(2 * np.pi * f2 * t)      # Formant 2
    audio += 0.1 * np.sin(2 * np.pi * f3 * t)      # Formant 3

    # Add amplitude modulation to simulate syllables
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz (syllable rate)
    audio = audio * modulation

    # Add some noise for realism
    noise = np.random.randn(len(t)) * 0.02
    audio = audio + noise

    # Normalize
    audio = audio / np.abs(audio).max() * 0.8

    return audio.astype(np.float32), sample_rate

if __name__ == "__main__":
    print("Generating speech-like audio...")
    audio, sr = generate_speech_like_audio(duration=5.0)

    output_file = "/tmp/test_speech_like.wav"
    sf.write(output_file, audio, sr)

    print(f"✅ Created: {output_file}")
    print(f"   Duration: {len(audio) / sr:.1f}s")
    print(f"   Sample rate: {sr} Hz")
    print("\nNow run:")
    print(f"  python3 test_kv_cache_fix.py")
