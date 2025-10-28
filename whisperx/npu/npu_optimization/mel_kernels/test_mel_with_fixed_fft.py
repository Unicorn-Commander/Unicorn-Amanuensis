#!/usr/bin/env python3
"""
Test complete mel spectrogram pipeline with FIXED FFT
This proves the scaling fix resolves the accuracy issues
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
from test_fft_cpu import fft_radix2_512_q15_python, to_q15

def compute_mel_with_fixed_fft(audio, sr=16000):
    """
    Compute mel spectrogram using FIXED Q15 FFT (with scaling)
    This simulates the corrected NPU kernel
    """
    n_fft = 512
    hop_length = 160
    n_mels = 80

    # Hann window
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / (n_fft - 1))).astype(np.float32)

    # Pad audio to match librosa's framing
    audio_padded = np.pad(audio, (0, n_fft), mode='constant')

    # Compute mel spectrogram frame by frame
    n_frames = 1 + (len(audio_padded) - n_fft) // hop_length
    mel_output = np.zeros((n_mels, n_frames), dtype=np.float32)

    for frame_idx in range(n_frames):
        start = frame_idx * hop_length
        frame = audio_padded[start:start + n_fft]

        # Apply window
        windowed = frame * window

        # Convert to Q15
        windowed_q15 = to_q15(windowed)

        # FFT with FIXED scaling
        real_q15, imag_q15 = fft_radix2_512_q15_python(windowed_q15)

        # Compute power spectrum
        power = (real_q15.astype(np.float32)**2 + imag_q15.astype(np.float32)**2)

        # Only first half (due to symmetry)
        power_first_half = power[:257]

        # Simple mel binning (linear mapping 257 bins → 80 mels)
        for mel_bin in range(n_mels):
            start_bin = (mel_bin * 257) // n_mels
            end_bin = ((mel_bin + 1) * 257) // n_mels
            mel_output[mel_bin, frame_idx] = power_first_half[start_bin:end_bin].mean()

    # Convert to log scale (dB)
    mel_db = librosa.power_to_db(mel_output, ref=np.max)

    return mel_db

def test_mel_accuracy():
    """Compare FIXED Q15 mel with librosa reference"""

    print("="*70)
    print("MEL SPECTROGRAM WITH FIXED FFT TEST")
    print("="*70)

    # Generate test audio (1000 Hz sine wave)
    sr = 16000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 1000
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

    print(f"\nTest audio: {len(audio)} samples, {freq} Hz sine wave")

    # Reference: librosa mel spectrogram
    print("\nComputing reference mel (librosa)...")
    mel_ref = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=512, hop_length=160, n_mels=80,
        power=2.0, htk=True, norm='slaney'
    )
    mel_ref_db = librosa.power_to_db(mel_ref, ref=np.max)

    # FIXED Q15 mel
    print("Computing FIXED Q15 mel...")
    mel_q15_db = compute_mel_with_fixed_fft(audio, sr)

    # Compare
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    # Normalize for comparison
    mel_ref_norm = (mel_ref_db - mel_ref_db.min()) / (mel_ref_db.max() - mel_ref_db.min())
    mel_q15_norm = (mel_q15_db - mel_q15_db.min()) / (mel_q15_db.max() - mel_q15_db.min())

    # Flatten for correlation
    correlation = np.corrcoef(mel_ref_norm.flatten(), mel_q15_norm.flatten())[0, 1]
    mse = np.mean((mel_ref_norm - mel_q15_norm)**2)

    print(f"\nCorrelation: {correlation:.4f}")
    print(f"MSE: {mse:.6f}")
    print(f"Shape match: {mel_ref_db.shape} == {mel_q15_db.shape}")

    # Find peak mel bin
    mel_ref_avg = mel_ref_db.mean(axis=1)
    mel_q15_avg = mel_q15_db.mean(axis=1)

    peak_ref = np.argmax(mel_ref_avg)
    peak_q15 = np.argmax(mel_q15_avg)

    print(f"\nPeak mel bin (librosa): {peak_ref}")
    print(f"Peak mel bin (Q15 FIXED): {peak_q15}")
    print(f"Peak bin match: {peak_ref == peak_q15}")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Librosa mel
    im1 = axes[0, 0].imshow(mel_ref_db, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title('Librosa Mel Spectrogram (Reference)')
    axes[0, 0].set_xlabel('Time Frame')
    axes[0, 0].set_ylabel('Mel Bin')
    plt.colorbar(im1, ax=axes[0, 0])

    # FIXED Q15 mel
    im2 = axes[0, 1].imshow(mel_q15_db, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 1].set_title('FIXED Q15 Mel Spectrogram')
    axes[0, 1].set_xlabel('Time Frame')
    axes[0, 1].set_ylabel('Mel Bin')
    plt.colorbar(im2, ax=axes[0, 1])

    # Difference
    diff = np.abs(mel_ref_norm - mel_q15_norm)
    im3 = axes[1, 0].imshow(diff, aspect='auto', origin='lower', cmap='hot')
    axes[1, 0].set_title(f'Absolute Difference (MSE: {mse:.6f})')
    axes[1, 0].set_xlabel('Time Frame')
    axes[1, 0].set_ylabel('Mel Bin')
    plt.colorbar(im3, ax=axes[1, 0])

    # Average mel energy per bin
    axes[1, 1].plot(mel_ref_avg, label='Librosa', alpha=0.7, linewidth=2)
    axes[1, 1].plot(mel_q15_avg, label='FIXED Q15', alpha=0.7, linewidth=2)
    axes[1, 1].set_xlabel('Mel Bin')
    axes[1, 1].set_ylabel('Average Energy (dB)')
    axes[1, 1].set_title(f'Mel Energy Profile (Corr: {correlation:.4f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mel_fixed_fft_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: mel_fixed_fft_comparison.png")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if correlation > 0.95:
        print(f"✅ SUCCESS! FFT fix works! Correlation: {correlation:.4f}")
        print(f"   The scaling fix resolved the overflow issues.")
        print(f"   Ready to recompile NPU kernel with this fix.")
    elif correlation > 0.8:
        print(f"⚠️  IMPROVED but not perfect. Correlation: {correlation:.4f}")
        print(f"   FFT scaling helps but mel binning may need refinement.")
    else:
        print(f"❌ Still broken. Correlation: {correlation:.4f}")
        print(f"   Additional fixes needed beyond FFT scaling.")

    print("="*70)

    return correlation

if __name__ == "__main__":
    correlation = test_mel_accuracy()
    sys.exit(0 if correlation > 0.95 else 1)
