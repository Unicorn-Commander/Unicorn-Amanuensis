#!/usr/bin/env python3
"""
Mel Filterbank Coefficient Generator for Whisper NPU Kernels

Generates Q15 fixed-point mel filterbank coefficients using HTK formula.
Compatible with Whisper's mel spectrogram preprocessing requirements.

Technical Specifications:
- Sample rate: 16000 Hz
- FFT size: 512 (257 bins for one-sided spectrum)
- Number of mel bins: 80
- Frequency range: 0-8000 Hz (Nyquist)
- Mel scale: HTK formula (used by Whisper)
- Output format: Q15 fixed-point (int16_t)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import os

# Whisper configuration constants
SAMPLE_RATE = 16000
FFT_SIZE = 512
N_FFT_BINS = FFT_SIZE // 2 + 1  # 257 bins (one-sided spectrum)
N_MELS = 80
F_MIN = 0.0
F_MAX = SAMPLE_RATE / 2.0  # 8000 Hz (Nyquist frequency)

# Q15 fixed-point constants
Q15_SCALE = 32767  # Maximum positive value for int16_t
Q15_MIN = -32768
Q15_MAX = 32767


def hz_to_mel_htk(hz: float) -> float:
    """
    Convert Hz to mel scale using HTK formula.

    This is the formula used by Whisper (NOT the Slaney formula).
    HTK formula: mel = 2595 * log10(1 + f/700)

    Args:
        hz: Frequency in Hz

    Returns:
        Frequency in mel scale
    """
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz_htk(mel: float) -> float:
    """
    Convert mel scale to Hz using HTK formula.

    Inverse of HTK formula: f = 700 * (10^(mel/2595) - 1)

    Args:
        mel: Frequency in mel scale

    Returns:
        Frequency in Hz
    """
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create mel filterbank using HTK formula.

    Returns triangular filters in the mel frequency domain.
    Each filter is a triangle with:
    - Rising edge from lower frequency to center
    - Falling edge from center to upper frequency
    - Normalized so sum of weights = 1.0

    Returns:
        Tuple of (filterbank, center_frequencies)
        - filterbank: Shape (N_MELS, N_FFT_BINS) with float weights
        - center_frequencies: Shape (N_MELS,) with center freq in Hz
    """
    # Create mel-spaced frequencies
    mel_min = hz_to_mel_htk(F_MIN)
    mel_max = hz_to_mel_htk(F_MAX)
    mel_points = np.linspace(mel_min, mel_max, N_MELS + 2)

    # Convert back to Hz
    hz_points = mel_to_hz_htk(mel_points)

    # Convert Hz to FFT bin indices
    bin_points = np.floor((FFT_SIZE + 1) * hz_points / SAMPLE_RATE).astype(int)

    # Ensure bins are within valid range
    bin_points = np.clip(bin_points, 0, N_FFT_BINS - 1)

    # Create filterbank matrix
    filterbank = np.zeros((N_MELS, N_FFT_BINS), dtype=np.float64)

    # Build triangular filters
    for i in range(N_MELS):
        left_bin = bin_points[i]
        center_bin = bin_points[i + 1]
        right_bin = bin_points[i + 2]

        # Rising edge: left to center
        for j in range(left_bin, center_bin):
            if center_bin > left_bin:
                filterbank[i, j] = (j - left_bin) / (center_bin - left_bin)

        # Falling edge: center to right
        for j in range(center_bin, right_bin):
            if right_bin > center_bin:
                filterbank[i, j] = (right_bin - j) / (right_bin - center_bin)

        # Normalize so sum of weights = 1.0 for each filter
        filter_sum = np.sum(filterbank[i, :])
        if filter_sum > 0:
            filterbank[i, :] /= filter_sum

    # Get center frequencies for each mel bin
    center_frequencies = hz_points[1:-1]

    return filterbank, center_frequencies


def quantize_to_q15(value: float) -> int:
    """
    Convert floating-point value to Q15 fixed-point.

    Q15 format: 1 sign bit + 15 fractional bits
    Range: -1.0 to +0.999969482421875

    Args:
        value: Float value to quantize (should be in range [-1, 1])

    Returns:
        Q15 integer in range [Q15_MIN, Q15_MAX]
    """
    # Scale and round
    q15_value = int(np.round(value * Q15_SCALE))

    # Clamp to valid range
    q15_value = max(Q15_MIN, min(Q15_MAX, q15_value))

    return q15_value


def generate_c_header(filterbank: np.ndarray, center_frequencies: np.ndarray):
    """
    Generate C header file with Q15 fixed-point coefficients.

    Creates sparse representation storing only non-zero coefficients
    with start/end bin indices for efficiency.

    Args:
        filterbank: Float filterbank matrix (N_MELS, N_FFT_BINS)
        center_frequencies: Center frequency for each mel bin
    """
    output_path = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/mel_coeffs_fixed.h"

    with open(output_path, 'w') as f:
        # Header comment
        f.write("/**\n")
        f.write(" * Mel Filterbank Coefficients - Q15 Fixed-Point Format\n")
        f.write(" * \n")
        f.write(" * Generated for Whisper NPU Mel Spectrogram Kernels\n")
        f.write(" * \n")
        f.write(" * Configuration:\n")
        f.write(f" *   - Sample rate: {SAMPLE_RATE} Hz\n")
        f.write(f" *   - FFT size: {FFT_SIZE}\n")
        f.write(f" *   - FFT bins: {N_FFT_BINS} (one-sided spectrum)\n")
        f.write(f" *   - Mel bins: {N_MELS}\n")
        f.write(f" *   - Frequency range: {F_MIN}-{F_MAX} Hz\n")
        f.write(" *   - Mel scale: HTK formula (Whisper-compatible)\n")
        f.write(" *   - Format: Q15 (1 sign + 15 fractional bits)\n")
        f.write(" * \n")
        f.write(" * Q15 Format Details:\n")
        f.write(" *   - Type: int16_t\n")
        f.write(f" *   - Range: [{Q15_MIN}, {Q15_MAX}]\n")
        f.write(" *   - Scale factor: 32767 (0x7FFF)\n")
        f.write(" *   - Precision: ~0.003%\n")
        f.write(" * \n")
        f.write(" * Filter Structure:\n")
        f.write(" *   - Triangular filters in mel-frequency domain\n")
        f.write(" *   - Rising edge: lower → center (weight 0→1)\n")
        f.write(" *   - Falling edge: center → upper (weight 1→0)\n")
        f.write(" *   - Normalized: sum of weights = 1.0 per filter\n")
        f.write(" */\n\n")

        f.write("#ifndef MEL_COEFFS_FIXED_H\n")
        f.write("#define MEL_COEFFS_FIXED_H\n\n")

        f.write("#include <stdint.h>\n\n")

        # Configuration constants
        f.write("// Configuration constants\n")
        f.write(f"#define MEL_SAMPLE_RATE {SAMPLE_RATE}\n")
        f.write(f"#define MEL_FFT_SIZE {FFT_SIZE}\n")
        f.write(f"#define MEL_N_FFT_BINS {N_FFT_BINS}\n")
        f.write(f"#define MEL_N_MELS {N_MELS}\n")
        f.write(f"#define MEL_Q15_SCALE {Q15_SCALE}\n\n")

        # Structure definition
        f.write("// Sparse mel filter structure\n")
        f.write("typedef struct {\n")
        f.write("    int16_t start_bin;              // First non-zero FFT bin\n")
        f.write("    int16_t end_bin;                // Last non-zero FFT bin (exclusive)\n")
        f.write(f"    int16_t weights[{N_FFT_BINS}];    // Q15 filter weights (0 for unused bins)\n")
        f.write("} mel_filter_q15_t;\n\n")

        # Generate filter data
        f.write(f"// Mel filterbank coefficients ({N_MELS} filters)\n")
        f.write(f"const mel_filter_q15_t mel_filters_q15[{N_MELS}] = {{\n")

        stats = {
            'total_nonzero': 0,
            'max_width': 0,
            'min_width': N_FFT_BINS,
            'q15_max': Q15_MIN,
            'q15_min': Q15_MAX,
        }

        for i in range(N_MELS):
            # Find non-zero range
            nonzero_indices = np.where(filterbank[i, :] > 0)[0]

            if len(nonzero_indices) == 0:
                start_bin = 0
                end_bin = 0
            else:
                start_bin = int(nonzero_indices[0])
                end_bin = int(nonzero_indices[-1]) + 1

            # Convert to Q15
            q15_weights = np.zeros(N_FFT_BINS, dtype=np.int16)
            for j in range(start_bin, end_bin):
                q15_weights[j] = quantize_to_q15(filterbank[i, j])

            # Update statistics
            width = end_bin - start_bin
            stats['total_nonzero'] += width
            stats['max_width'] = max(stats['max_width'], width)
            stats['min_width'] = min(stats['min_width'], width) if width > 0 else stats['min_width']
            stats['q15_max'] = max(stats['q15_max'], np.max(q15_weights))
            if width > 0:
                stats['q15_min'] = min(stats['q15_min'], np.min(q15_weights[start_bin:end_bin]))

            # Write filter entry
            center_freq = center_frequencies[i]
            mel_freq = hz_to_mel_htk(center_freq)

            f.write(f"    // Filter {i}: {center_freq:.1f} Hz ({mel_freq:.1f} mel), bins {start_bin}-{end_bin-1}\n")
            f.write(f"    {{\n")
            f.write(f"        .start_bin = {start_bin},\n")
            f.write(f"        .end_bin = {end_bin},\n")
            f.write(f"        .weights = {{\n")

            # Write weights in rows of 8
            for row_start in range(0, N_FFT_BINS, 8):
                row_end = min(row_start + 8, N_FFT_BINS)
                weights_str = ", ".join(f"{q15_weights[j]:6d}" for j in range(row_start, row_end))

                # Add comment for non-zero region
                if row_start < end_bin and row_end > start_bin:
                    f.write(f"            {weights_str},  // bins {row_start}-{row_end-1}\n")
                else:
                    f.write(f"            {weights_str},\n")

            f.write(f"        }}\n")
            f.write(f"    }}")

            if i < N_MELS - 1:
                f.write(",\n")
            else:
                f.write("\n")

        f.write("};\n\n")

        # Add helper function declarations
        f.write("// Helper function to apply mel filterbank to power spectrum\n")
        f.write("// Input:  power_spectrum[257] - Q15 FFT power values\n")
        f.write("// Output: mel_spectrum[80]    - Q15 mel-filtered values\n")
        f.write("static inline void apply_mel_filterbank_q15(\n")
        f.write("    const int16_t* power_spectrum,\n")
        f.write("    int16_t* mel_spectrum\n")
        f.write(") {\n")
        f.write(f"    for (int i = 0; i < {N_MELS}; i++) {{\n")
        f.write("        const mel_filter_q15_t* filter = &mel_filters_q15[i];\n")
        f.write("        int32_t sum = 0;\n")
        f.write("        \n")
        f.write("        // Accumulate weighted sum (Q15 * Q15 = Q30)\n")
        f.write("        for (int j = filter->start_bin; j < filter->end_bin; j++) {\n")
        f.write("            sum += (int32_t)power_spectrum[j] * (int32_t)filter->weights[j];\n")
        f.write("        }\n")
        f.write("        \n")
        f.write("        // Convert Q30 back to Q15 (divide by 32768)\n")
        f.write("        mel_spectrum[i] = (int16_t)(sum >> 15);\n")
        f.write("    }\n")
        f.write("}\n\n")

        f.write("#endif // MEL_COEFFS_FIXED_H\n")

    print(f"\nGenerated C header: {output_path}")

    # Print statistics
    print("\n=== Mel Filterbank Statistics ===")
    print(f"Total filters: {N_MELS}")
    print(f"Total non-zero coefficients: {stats['total_nonzero']}")
    print(f"Average filter width: {stats['total_nonzero'] / N_MELS:.1f} bins")
    print(f"Filter width range: {stats['min_width']}-{stats['max_width']} bins")
    print(f"Q15 value range: [{stats['q15_min']}, {stats['q15_max']}]")
    print(f"Q15 range used: {(stats['q15_max'] - stats['q15_min']) / Q15_SCALE * 100:.1f}% of full scale")

    return stats


def validate_against_librosa(filterbank: np.ndarray):
    """
    Compare generated filterbank with librosa reference.

    Args:
        filterbank: Generated filterbank to validate
    """
    try:
        import librosa

        # Generate librosa filterbank with HTK=True
        librosa_fb = librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=FFT_SIZE,
            n_mels=N_MELS,
            fmin=F_MIN,
            fmax=F_MAX,
            htk=True  # Use HTK formula
        )

        # Compare
        diff = np.abs(filterbank - librosa_fb)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print("\n=== Validation Against Librosa ===")
        print(f"Maximum difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        print(f"Match quality: {'EXCELLENT' if max_diff < 0.01 else 'GOOD' if max_diff < 0.05 else 'FAIR'}")

        return max_diff < 0.05

    except ImportError:
        print("\n=== Validation Skipped ===")
        print("librosa not available for validation")
        return True


def plot_filterbank(filterbank: np.ndarray, center_frequencies: np.ndarray):
    """
    Create visualization of mel filterbank.

    Args:
        filterbank: Float filterbank matrix
        center_frequencies: Center frequency for each mel bin
    """
    output_path = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/mel_filterbank_visualization.png"

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: All filters
    ax = axes[0]
    freq_axis = np.linspace(0, F_MAX, N_FFT_BINS)

    # Plot every 5th filter for clarity
    for i in range(0, N_MELS, 5):
        ax.plot(freq_axis, filterbank[i, :], label=f'Filter {i}', alpha=0.7)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Filter Weight')
    ax.set_title('Mel Filterbank - Triangular Filters (HTK Formula)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    # Plot 2: Heatmap of all filters
    ax = axes[1]
    im = ax.imshow(filterbank, aspect='auto', origin='lower', cmap='viridis',
                   extent=[0, F_MAX, 0, N_MELS])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Mel Bin')
    ax.set_title('Mel Filterbank Heatmap')
    plt.colorbar(im, ax=ax, label='Filter Weight')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGenerated visualization: {output_path}")


def test_q15_quantization(filterbank: np.ndarray):
    """
    Test Q15 quantization error.

    Args:
        filterbank: Original float filterbank

    Returns:
        Dictionary with quantization error statistics
    """
    # Quantize and dequantize
    q15_filterbank = np.zeros_like(filterbank)

    for i in range(N_MELS):
        for j in range(N_FFT_BINS):
            q15_value = quantize_to_q15(filterbank[i, j])
            q15_filterbank[i, j] = q15_value / Q15_SCALE

    # Calculate errors
    abs_error = np.abs(filterbank - q15_filterbank)
    rel_error = np.zeros_like(abs_error)

    # Calculate relative error only for non-zero coefficients
    nonzero_mask = filterbank > 0
    rel_error[nonzero_mask] = abs_error[nonzero_mask] / filterbank[nonzero_mask] * 100

    stats = {
        'max_abs_error': np.max(abs_error),
        'mean_abs_error': np.mean(abs_error[nonzero_mask]),
        'max_rel_error': np.max(rel_error[nonzero_mask]),
        'mean_rel_error': np.mean(rel_error[nonzero_mask]),
    }

    print("\n=== Q15 Quantization Error ===")
    print(f"Maximum absolute error: {stats['max_abs_error']:.6f}")
    print(f"Mean absolute error: {stats['mean_abs_error']:.6f}")
    print(f"Maximum relative error: {stats['max_rel_error']:.3f}%")
    print(f"Mean relative error: {stats['mean_rel_error']:.3f}%")
    print(f"Quality: {'EXCELLENT' if stats['max_rel_error'] < 0.1 else 'GOOD' if stats['max_rel_error'] < 1.0 else 'ACCEPTABLE'}")

    return stats


def main():
    """Main generation pipeline."""
    print("=" * 60)
    print("Mel Filterbank Coefficient Generator")
    print("HTK Formula for Whisper NPU Kernels")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  FFT size: {FFT_SIZE}")
    print(f"  FFT bins: {N_FFT_BINS}")
    print(f"  Mel bins: {N_MELS}")
    print(f"  Frequency range: {F_MIN}-{F_MAX} Hz")
    print(f"  Mel scale: HTK formula")
    print(f"  Output format: Q15 fixed-point")

    # Generate filterbank
    print("\n[1/5] Generating mel filterbank...")
    filterbank, center_frequencies = create_mel_filterbank()
    print(f"  Created {N_MELS} triangular filters")

    # Validate against librosa
    print("\n[2/5] Validating against librosa...")
    validate_against_librosa(filterbank)

    # Test Q15 quantization
    print("\n[3/5] Testing Q15 quantization...")
    q15_stats = test_q15_quantization(filterbank)

    # Generate C header
    print("\n[4/5] Generating C header file...")
    header_stats = generate_c_header(filterbank, center_frequencies)

    # Create visualization
    print("\n[5/5] Creating visualization...")
    plot_filterbank(filterbank, center_frequencies)

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  1. mel_coeffs_fixed.h - Q15 coefficient table")
    print("  2. mel_filterbank_visualization.png - Filter visualization")
    print("\nNext steps:")
    print("  1. Include mel_coeffs_fixed.h in your NPU kernel")
    print("  2. Use apply_mel_filterbank_q15() function")
    print("  3. Run apply_mel_filters_reference.py for testing")


if __name__ == "__main__":
    main()
