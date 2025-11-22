#!/usr/bin/env python3
"""
Test Mel Spectrogram Generator for Whisper Encoder Testing

Generates synthetic and real mel spectrograms for testing Whisper encoder kernels.
Whisper base encoder expects:
  - Input shape: (1, 80, 3000) or (80, 3000)
  - 80 mel frequency bins
  - 3000 time frames (30 seconds of audio at 100 Hz)
"""

import numpy as np
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available, will skip real audio processing")

try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, will use basic filtering")


def generate_synthetic_mel_spectrogram(
    n_mels: int = 80,
    n_frames: int = 3000,
    frequency_structure: bool = True,
    temporal_patterns: bool = True,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate realistic synthetic mel spectrogram data.

    Args:
        n_mels: Number of mel frequency bins (default: 80)
        n_frames: Number of time frames (default: 3000)
        frequency_structure: Add frequency structure to the data
        temporal_patterns: Add temporal patterns
        seed: Random seed for reproducibility

    Returns:
        Mel spectrogram of shape (n_mels, n_frames)
    """
    if seed is not None:
        np.random.seed(seed)

    # Start with base noise
    mel_spec = np.random.randn(n_mels, n_frames) * 0.1

    # Add frequency structure (formant-like bands)
    if frequency_structure:
        # Create frequency bands that resemble speech formants
        formant_frequencies = [0.1, 0.25, 0.5, 0.7]  # Relative positions in mel bins
        formant_widths = [8, 6, 5, 4]  # Width of each band

        for formant_freq, width in zip(formant_frequencies, formant_widths):
            center_bin = int(formant_freq * n_mels)
            for offset in range(-width, width + 1):
                bin_idx = center_bin + offset
                if 0 <= bin_idx < n_mels:
                    # Create Gaussian envelope
                    envelope = np.exp(-(offset ** 2) / (2 * (width / 3) ** 2))
                    # Add spectral energy at this frequency
                    mel_spec[bin_idx, :] += envelope * (0.5 + 0.3 * np.random.random())

    # Add temporal patterns (speech-like dynamics)
    if temporal_patterns:
        # Create amplitude modulation patterns
        time_axis = np.arange(n_frames)

        # Speech-like envelope (attacks and decays)
        envelope = np.ones(n_frames)

        # Add syllable-like patterns
        syllable_spacing = 150  # Frames between syllables (~1.5 seconds)
        for i in range(0, n_frames, syllable_spacing):
            attack = min(50, n_frames - i)  # Attack time
            decay = min(100, n_frames - i)  # Decay time

            attack_env = np.linspace(0, 1, attack)
            decay_env = np.linspace(1, 0.1, decay)

            if i + attack <= n_frames:
                envelope[i:i+attack] *= attack_env
            if i + attack + decay <= n_frames:
                envelope[i+attack:i+attack+decay] *= decay_env

        # Apply envelope to entire spectrogram
        mel_spec *= envelope[np.newaxis, :]

        # Add frequency modulation (pitch variation)
        pitch_mod = 0.15 * np.sin(2 * np.pi * time_axis / n_frames * 3)
        for f in range(n_mels):
            shift = int(pitch_mod[0] * 10)  # Subtle shift
            mel_spec[f, :] += 0.05 * np.roll(mel_spec[f, :], shift)

    # Convert to log scale (typical for mel spectrograms)
    mel_spec = np.log(np.maximum(mel_spec, 1e-9) + 1)

    # Normalize to reasonable range
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
    mel_spec = np.clip(mel_spec, -10, 10)

    return mel_spec


def load_and_compute_mel_spectrogram(
    audio_path: str,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    target_frames: int = 3000
) -> Tuple[np.ndarray, Dict]:
    """
    Load audio file and compute mel spectrogram using librosa.

    Args:
        audio_path: Path to audio file
        sr: Sample rate (default: 16000 Hz for Whisper)
        n_mels: Number of mel bins (default: 80)
        n_fft: FFT window size (default: 400)
        hop_length: Hop length (default: 160)
        target_frames: Target number of frames (default: 3000)

    Returns:
        Tuple of (mel_spectrogram, metadata_dict)
    """
    if not LIBROSA_AVAILABLE:
        raise RuntimeError("librosa is required to load audio files")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio
    y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0.0,
        fmax=sr // 2
    )

    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad or truncate to target frames
    n_frames = mel_spec_db.shape[1]
    if n_frames < target_frames:
        # Pad with silence (minimum value)
        pad_amount = target_frames - n_frames
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_amount)),
                             mode='constant', constant_values=np.min(mel_spec_db))
    elif n_frames > target_frames:
        # Truncate
        mel_spec_db = mel_spec_db[:, :target_frames]

    # Metadata
    metadata = {
        'original_shape': mel_spec.shape,
        'final_shape': mel_spec_db.shape,
        'sample_rate': sr,
        'n_mels': n_mels,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'audio_duration_sec': len(y) / sr,
        'n_frames': mel_spec_db.shape[1],
        'audio_path': audio_path,
        'audio_name': os.path.basename(audio_path)
    }

    return mel_spec_db, metadata


def find_test_audio_files(
    search_root: str = "/home/ucadmin/UC-1/Unicorn-Amanuensis",
    exclude_venv: bool = True
) -> list:
    """
    Find test audio files in the project.

    Args:
        search_root: Root directory to search
        exclude_venv: Exclude virtual environment directories

    Returns:
        List of audio file paths
    """
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = []

    for root, dirs, files in os.walk(search_root):
        # Skip virtual environments
        if exclude_venv:
            dirs[:] = [d for d in dirs if 'venv' not in d and '.venv' not in d]

        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                full_path = os.path.join(root, file)
                audio_files.append(full_path)

    return sorted(audio_files)


def save_mel_spectrogram(mel_spec: np.ndarray, output_path: str, metadata: Dict = None) -> None:
    """
    Save mel spectrogram and metadata to numpy file.

    Args:
        mel_spec: Mel spectrogram array
        output_path: Path to save numpy file
        metadata: Optional metadata dictionary
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if metadata:
        # Save with metadata as npz format
        np.savez_compressed(
            output_path.replace('.npy', '.npz'),
            mel_spec=mel_spec,
            metadata=np.array(metadata, dtype=object)
        )
    else:
        # Save as simple npy file
        np.save(output_path, mel_spec)


def print_mel_statistics(mel_spec: np.ndarray, name: str = "Mel Spectrogram") -> None:
    """
    Print statistics about mel spectrogram.

    Args:
        mel_spec: Mel spectrogram array
        name: Name for display
    """
    print(f"\n{'='*60}")
    print(f"Mel Spectrogram Statistics: {name}")
    print(f"{'='*60}")
    print(f"Shape:              {mel_spec.shape}")
    print(f"Data type:          {mel_spec.dtype}")
    print(f"Min value:          {mel_spec.min():.6f}")
    print(f"Max value:          {mel_spec.max():.6f}")
    print(f"Mean value:         {mel_spec.mean():.6f}")
    print(f"Std deviation:      {mel_spec.std():.6f}")
    print(f"Median value:       {np.median(mel_spec):.6f}")
    print(f"25th percentile:    {np.percentile(mel_spec, 25):.6f}")
    print(f"75th percentile:    {np.percentile(mel_spec, 75):.6f}")
    print(f"Memory size:        {mel_spec.nbytes / (1024*1024):.2f} MB")
    print(f"{'='*60}")


def main():
    """Main execution function."""

    # Configuration
    n_mels = 80
    n_frames = 3000
    output_dir = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/test_data"

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Whisper Encoder Test Mel Spectrogram Generator")
    print("="*60)

    # Generate synthetic mel spectrogram
    print("\n[1] Generating synthetic mel spectrogram...")
    synthetic_mel = generate_synthetic_mel_spectrogram(
        n_mels=n_mels,
        n_frames=n_frames,
        frequency_structure=True,
        temporal_patterns=True,
        seed=42
    )

    print_mel_statistics(synthetic_mel, "Synthetic Mel Spectrogram")

    # Save synthetic data
    synthetic_path = os.path.join(output_dir, "test_mel_synthetic.npy")
    np.save(synthetic_path, synthetic_mel)
    print(f"Saved synthetic mel spectrogram to: {synthetic_path}")

    # Find test audio files
    print("\n[2] Searching for test audio files...")
    audio_files = find_test_audio_files()

    # Filter to keep only practical test files (exclude venv scipy tests)
    practical_audio_files = [
        f for f in audio_files
        if 'venv' not in f and 'scipy' not in f
    ]

    if practical_audio_files:
        print(f"Found {len(practical_audio_files)} test audio files:")
        for audio_path in practical_audio_files[:10]:  # Show first 10
            print(f"  - {audio_path}")
    else:
        print("No test audio files found in project")

    # Process real audio files if librosa is available
    if LIBROSA_AVAILABLE and practical_audio_files:
        print("\n[3] Processing real audio files with librosa...")

        for audio_path in practical_audio_files[:3]:  # Process first 3 files
            try:
                print(f"\nProcessing: {os.path.basename(audio_path)}")
                mel_spec, metadata = load_and_compute_mel_spectrogram(
                    audio_path,
                    sr=16000,
                    n_mels=n_mels,
                    target_frames=n_frames
                )

                print_mel_statistics(mel_spec, f"Real Audio: {metadata['audio_name']}")

                # Save real mel spectrogram
                base_name = Path(audio_path).stem
                output_path = os.path.join(output_dir, f"test_mel_{base_name}.npz")
                save_mel_spectrogram(mel_spec, output_path, metadata)
                print(f"Saved real mel spectrogram to: {output_path}")

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
    elif not LIBROSA_AVAILABLE:
        print("\n[3] Skipping real audio processing (librosa not available)")
        print("    Install with: pip install librosa")

    # Create test batch with batch dimension
    print("\n[4] Creating batched test data...")
    synthetic_mel_batched = synthetic_mel[np.newaxis, :, :]  # Add batch dimension

    batched_path = os.path.join(output_dir, "test_mel_batched.npy")
    np.save(batched_path, synthetic_mel_batched)
    print(f"Batched shape: {synthetic_mel_batched.shape}")
    print(f"Saved batched mel spectrogram to: {batched_path}")

    # Create summary report
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Synthetic mel shape:     {synthetic_mel.shape}")
    print(f"Batched mel shape:       {synthetic_mel_batched.shape}")
    print(f"Expected Whisper shape:  (1, {n_mels}, {n_frames}) or ({n_mels}, {n_frames})")
    print(f"Output directory:        {output_dir}")
    print(f"Files created:")
    print(f"  - {os.path.basename(synthetic_path)}")
    print(f"  - {os.path.basename(batched_path)}")
    if LIBROSA_AVAILABLE and practical_audio_files:
        print(f"  - Real audio mel spectrograms (.npz files)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
