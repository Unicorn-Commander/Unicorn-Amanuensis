#!/usr/bin/env python3
"""
Generate synthetic test signals for NPU mel spectrogram validation

Creates comprehensive test suite including:
- Pure tones at various frequencies
- Frequency sweeps (chirps)
- Various noise types (white, pink, brown)
- Impulses and step functions
- Complex audio combinations

Author: Magic Unicorn Inc.
Date: October 28, 2025
"""

import numpy as np
import os
from pathlib import Path

# Audio parameters (match Whisper configuration)
SAMPLE_RATE = 16000
DURATION = 0.025  # 25ms (400 samples)
N_SAMPLES = 400


def generate_pure_tone(frequency, amplitude=0.8, phase=0):
    """Generate pure sine wave at specified frequency"""
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return signal


def generate_chirp(f0=100, f1=4000, amplitude=0.8):
    """Generate frequency sweep (linear chirp)"""
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    # Linear chirp: f(t) = f0 + (f1-f0) * t/duration
    chirp_rate = (f1 - f0) / DURATION
    phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t**2)
    signal = amplitude * np.sin(phase)
    return signal


def generate_white_noise(amplitude=0.5):
    """Generate white noise (uniform power across frequencies)"""
    noise = np.random.uniform(-1, 1, N_SAMPLES)
    return amplitude * noise / np.max(np.abs(noise))


def generate_pink_noise(amplitude=0.5):
    """Generate pink noise (1/f power spectrum)"""
    # Generate white noise
    white = np.random.randn(N_SAMPLES)

    # Apply 1/f filter in frequency domain
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(N_SAMPLES, 1/SAMPLE_RATE)
    freqs[0] = 1  # Avoid division by zero

    # Apply 1/sqrt(f) filter (pink noise)
    fft_filtered = fft / np.sqrt(freqs)

    # Convert back to time domain
    pink = np.fft.irfft(fft_filtered, N_SAMPLES)

    # Normalize
    return amplitude * pink / np.max(np.abs(pink))


def generate_brown_noise(amplitude=0.5):
    """Generate brown noise (1/f^2 power spectrum)"""
    # Generate white noise
    white = np.random.randn(N_SAMPLES)

    # Apply 1/f^2 filter in frequency domain
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(N_SAMPLES, 1/SAMPLE_RATE)
    freqs[0] = 1  # Avoid division by zero

    # Apply 1/f filter (brown noise)
    fft_filtered = fft / freqs

    # Convert back to time domain
    brown = np.fft.irfft(fft_filtered, N_SAMPLES)

    # Normalize
    return amplitude * brown / np.max(np.abs(brown))


def generate_impulse(position=200, amplitude=1.0):
    """Generate impulse (single spike)"""
    signal = np.zeros(N_SAMPLES)
    signal[position] = amplitude
    return signal


def generate_step(step_position=200, amplitude=0.8):
    """Generate step function"""
    signal = np.zeros(N_SAMPLES)
    signal[step_position:] = amplitude
    return signal


def generate_multi_tone(frequencies, amplitudes=None):
    """Generate combination of multiple tones"""
    if amplitudes is None:
        amplitudes = [1.0 / len(frequencies)] * len(frequencies)

    signal = np.zeros(N_SAMPLES)
    for freq, amp in zip(frequencies, amplitudes):
        signal += generate_pure_tone(freq, amp)

    # Normalize to prevent clipping
    return signal / np.max(np.abs(signal)) * 0.8


def generate_silence():
    """Generate complete silence"""
    return np.zeros(N_SAMPLES)


def generate_dc_offset(offset=0.5):
    """Generate DC offset (constant value)"""
    return np.ones(N_SAMPLES) * offset


def save_signal(signal, filename, output_dir="test_audio"):
    """Save signal as raw INT16 audio (800 bytes, little-endian)"""
    # Convert to INT16
    signal_int16 = (signal * 16000).astype(np.int16)

    # Save as raw bytes (little-endian)
    Path(output_dir).mkdir(exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'wb') as f:
        signal_int16.tobytes()  # Little-endian by default on x86
        f.write(signal_int16.tobytes())

    print(f"✅ Generated: {output_path} ({len(signal_int16)*2} bytes)")

    return signal_int16


def generate_test_suite(output_dir="test_audio"):
    """Generate complete test suite"""
    print("=" * 70)
    print("Generating NPU Mel Spectrogram Test Suite")
    print("=" * 70)
    print()

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    test_signals = {}

    # 1. Pure tones at key frequencies
    print("1. Pure Tones:")
    print("-" * 70)
    test_freqs = [100, 250, 500, 1000, 2000, 3000, 4000, 6000]
    for freq in test_freqs:
        signal = generate_pure_tone(freq)
        signal_int16 = save_signal(signal, f"tone_{freq}hz.raw", output_dir)
        test_signals[f"tone_{freq}hz"] = signal_int16
    print()

    # 2. Chirps (frequency sweeps)
    print("2. Chirps (Frequency Sweeps):")
    print("-" * 70)
    chirp_configs = [
        (100, 1000, "chirp_100_1000hz.raw"),
        (100, 4000, "chirp_100_4000hz.raw"),
        (1000, 4000, "chirp_1000_4000hz.raw"),
    ]
    for f0, f1, filename in chirp_configs:
        signal = generate_chirp(f0, f1)
        signal_int16 = save_signal(signal, filename, output_dir)
        test_signals[filename.replace('.raw', '')] = signal_int16
    print()

    # 3. Noise types
    print("3. Noise Types:")
    print("-" * 70)
    noise_types = [
        (generate_white_noise(), "white_noise.raw"),
        (generate_pink_noise(), "pink_noise.raw"),
        (generate_brown_noise(), "brown_noise.raw"),
    ]
    for signal, filename in noise_types:
        signal_int16 = save_signal(signal, filename, output_dir)
        test_signals[filename.replace('.raw', '')] = signal_int16
    print()

    # 4. Edge cases
    print("4. Edge Cases:")
    print("-" * 70)
    edge_cases = [
        (generate_silence(), "silence.raw"),
        (generate_dc_offset(0.5), "dc_offset.raw"),
        (generate_impulse(), "impulse.raw"),
        (generate_step(), "step.raw"),
    ]
    for signal, filename in edge_cases:
        signal_int16 = save_signal(signal, filename, output_dir)
        test_signals[filename.replace('.raw', '')] = signal_int16
    print()

    # 5. Multi-tone combinations
    print("5. Multi-Tone Combinations:")
    print("-" * 70)
    multi_tone_configs = [
        ([440, 880], [0.5, 0.5], "two_tones_440_880hz.raw"),  # Octave
        ([100, 200, 300], None, "harmonics_100_200_300hz.raw"),  # Harmonics
        ([1000, 1100], [0.5, 0.5], "beating_1000_1100hz.raw"),  # Beating
    ]
    for freqs, amps, filename in multi_tone_configs:
        signal = generate_multi_tone(freqs, amps)
        signal_int16 = save_signal(signal, filename, output_dir)
        test_signals[filename.replace('.raw', '')] = signal_int16
    print()

    # 6. Clipping test
    print("6. Clipping Test:")
    print("-" * 70)
    clipping_signal = generate_pure_tone(1000, amplitude=1.2)  # Intentional clipping
    clipping_signal = np.clip(clipping_signal, -1.0, 1.0)
    signal_int16 = save_signal(clipping_signal, "clipping_1000hz.raw", output_dir)
    test_signals["clipping_1000hz"] = signal_int16
    print()

    # 7. Very quiet signal
    print("7. Very Quiet Signal:")
    print("-" * 70)
    quiet_signal = generate_pure_tone(1000, amplitude=0.01)
    signal_int16 = save_signal(quiet_signal, "quiet_1000hz.raw", output_dir)
    test_signals["quiet_1000hz"] = signal_int16
    print()

    # Summary
    print("=" * 70)
    print(f"Test Suite Generation Complete!")
    print(f"Total test files: {len(test_signals)}")
    print(f"Output directory: {output_dir}")
    print()
    print("File format:")
    print("  - 800 bytes per file (400 INT16 samples)")
    print("  - Little-endian encoding")
    print("  - 16 kHz sample rate")
    print("  - 25ms duration")
    print()
    print("Next steps:")
    print("  1. Run: python3 benchmark_accuracy.py")
    print("  2. Compare NPU vs CPU mel spectrograms")
    print("  3. Review accuracy_report.md")
    print("=" * 70)

    return test_signals


if __name__ == '__main__':
    import sys

    # Parse command line arguments
    output_dir = "test_audio"
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]

    # Generate test suite
    test_signals = generate_test_suite(output_dir)

    # Success
    print()
    print("✨ Test signal generation complete!")
