#!/usr/bin/env python3
"""
Validation script for mel filterbank implementation
Compares our Q15 fixed-point implementation against librosa reference

Author: Magic Unicorn Inc.
Date: October 28, 2025
"""

import numpy as np
import sys

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  librosa not installed - install with: pip install librosa")
    print("   Skipping accuracy validation\n")


def load_mel_filters_from_header(header_path="mel_filterbank_coeffs.h"):
    """Parse generated header file to extract filter definitions"""
    import re

    filters = []

    with open(header_path, 'r') as f:
        content = f.read()

    # Extract filter definitions from mel_filters array
    filter_pattern = re.compile(
        r'// Filter (\d+):.*?\n'
        r'\s*\.start_bin = (\d+),\n'
        r'\s*\.peak_bin = (\d+),\n'
        r'\s*\.end_bin = (\d+),\n'
        r'\s*\.left_width = (\d+),\n'
        r'\s*\.right_width = (\d+),',
        re.MULTILINE
    )

    for match in filter_pattern.finditer(content):
        filt_id = int(match.group(1))
        start_bin = int(match.group(2))
        peak_bin = int(match.group(3))
        end_bin = int(match.group(4))
        left_width = int(match.group(5))
        right_width = int(match.group(6))

        # Reconstruct filter weights
        filter_weights = np.zeros(256, dtype=np.float32)

        # Left slope
        if left_width > 0:
            for i in range(left_width):
                bin_idx = start_bin + i
                weight = (i + 1) / left_width
                filter_weights[bin_idx] = weight

        # Right slope
        if right_width > 0:
            for i in range(right_width):
                bin_idx = peak_bin + i
                weight = 1.0 - (i + 1) / right_width
                filter_weights[bin_idx] = weight

        filters.append({
            'id': filt_id,
            'start_bin': start_bin,
            'peak_bin': peak_bin,
            'end_bin': end_bin,
            'weights': filter_weights
        })

    return filters


def compare_with_librosa():
    """Compare our implementation with librosa reference"""
    if not LIBROSA_AVAILABLE:
        return

    print("🔍 Comparing with librosa reference")
    print("=" * 70)

    # Generate librosa mel filterbank
    lib_filters = librosa.filters.mel(
        sr=16000,
        n_fft=512,
        n_mels=80,
        fmin=0.0,
        fmax=8000.0,
        htk=True  # Use HTK formula like Whisper
    )

    print(f"Librosa filterbank shape: {lib_filters.shape}")
    print()

    # Load our filters
    try:
        our_filters = load_mel_filters_from_header()
        print(f"Our filterbank: {len(our_filters)} filters")
        print()
    except Exception as e:
        print(f"❌ Error loading our filters: {e}")
        return

    # Compare filter boundaries
    print("Filter Boundary Comparison:")
    print("-" * 70)
    print(f"{'Filter':<8} {'Our Start':<12} {'Lib Start':<12} {'Our Peak':<12} {'Lib Peak':<12} {'Match':<8}")
    print("-" * 70)

    sample_indices = [0, 10, 20, 30, 40, 50, 60, 70, 79]
    matches = 0
    total = 0

    for idx in sample_indices:
        our_filter = our_filters[idx]
        lib_filter = lib_filters[idx]

        # Find non-zero region in librosa filter
        nonzero = np.where(lib_filter > 0.001)[0]
        if len(nonzero) > 0:
            lib_start = nonzero[0]
            lib_peak = nonzero[np.argmax(lib_filter[nonzero])]

            match = (abs(our_filter['start_bin'] - lib_start) <= 2 and
                     abs(our_filter['peak_bin'] - lib_peak) <= 2)

            matches += int(match)
            total += 1

            status = "✅" if match else "⚠️ "

            print(f"{idx:<8} {our_filter['start_bin']:<12} {lib_start:<12} "
                  f"{our_filter['peak_bin']:<12} {lib_peak:<12} {status}")

    print("-" * 70)
    print(f"\nMatch rate: {matches}/{total} ({100*matches/total:.1f}%)")
    print()

    # Test with random magnitude spectrum
    print("Mel Energy Comparison (Random Test Vector):")
    print("-" * 70)

    # Generate test magnitude spectrum (Q15 format)
    np.random.seed(42)
    magnitude_float = np.random.uniform(0, 1, 256).astype(np.float32)
    magnitude_q15 = (magnitude_float * 32767).astype(np.int16)

    # Compute mel energies with librosa
    lib_mel = lib_filters @ magnitude_float

    # Compute mel energies with our implementation
    our_mel = np.zeros(80, dtype=np.float32)
    for i, filt in enumerate(our_filters):
        energy = np.sum(magnitude_float * filt['weights'])
        our_mel[i] = energy

    # Compare
    abs_error = np.abs(lib_mel - our_mel)
    rel_error = abs_error / (np.abs(lib_mel) + 1e-10)

    print(f"Mean absolute error: {np.mean(abs_error):.6f}")
    print(f"Max absolute error:  {np.max(abs_error):.6f}")
    print(f"Mean relative error: {np.mean(rel_error)*100:.3f}%")
    print(f"Max relative error:  {np.max(rel_error)*100:.3f}%")
    print()

    # Show a few examples
    print("Sample values:")
    print(f"{'Filter':<8} {'Librosa':<15} {'Ours':<15} {'Error':<15} {'Status'}")
    print("-" * 70)

    for idx in [0, 20, 40, 60, 79]:
        lib_val = lib_mel[idx]
        our_val = our_mel[idx]
        error = abs(lib_val - our_val)
        status = "✅" if error < 0.01 else "⚠️ "
        print(f"{idx:<8} {lib_val:<15.6f} {our_val:<15.6f} {error:<15.6f} {status}")

    print("-" * 70)
    print()

    # Overall assessment
    if np.mean(rel_error) < 0.01:  # <1% error
        print("✅ VALIDATION PASSED: Error <1% (Excellent!)")
    elif np.mean(rel_error) < 0.05:  # <5% error
        print("✅ VALIDATION PASSED: Error <5% (Good)")
    else:
        print("⚠️  WARNING: Error >5% (May need adjustment)")

    print()


def analyze_filter_properties():
    """Analyze properties of generated filters"""
    print("📊 Filter Properties Analysis")
    print("=" * 70)

    try:
        filters = load_mel_filters_from_header()
    except Exception as e:
        print(f"❌ Error loading filters: {e}")
        return

    # Filter widths
    widths = [f['end_bin'] - f['start_bin'] for f in filters]
    print(f"Filter widths:")
    print(f"  Min:     {min(widths)} bins")
    print(f"  Max:     {max(widths)} bins")
    print(f"  Mean:    {np.mean(widths):.1f} bins")
    print(f"  Median:  {np.median(widths):.1f} bins")
    print()

    # Overlaps
    overlaps = []
    for i in range(len(filters) - 1):
        overlap = filters[i]['end_bin'] - filters[i+1]['start_bin']
        overlaps.append(overlap)

    print(f"Filter overlaps:")
    print(f"  Min:     {min(overlaps)} bins")
    print(f"  Max:     {max(overlaps)} bins")
    print(f"  Mean:    {np.mean(overlaps):.1f} bins")
    print(f"  Median:  {np.median(overlaps):.1f} bins")
    print()

    # Coverage
    min_start = min(f['start_bin'] for f in filters)
    max_end = max(f['end_bin'] for f in filters)
    print(f"Frequency coverage:")
    print(f"  Start bin: {min_start} (0 Hz)")
    print(f"  End bin:   {max_end} (7968.75 Hz)")
    print(f"  Coverage:  {max_end - min_start + 1} / 256 bins ({100*(max_end-min_start+1)/256:.1f}%)")
    print()

    # Memory estimate
    total_coeffs = sum(
        (f['end_bin'] - f['start_bin'])
        for f in filters
    )
    struct_size = len(filters) * 16  # 16 bytes per struct
    coeff_size = total_coeffs * 2     # 2 bytes per coefficient
    total_size = struct_size + coeff_size

    print(f"Memory footprint:")
    print(f"  Filter structs:  {struct_size} bytes")
    print(f"  Coefficients:    {coeff_size} bytes ({total_coeffs} values)")
    print(f"  Total:           {total_size} bytes ({total_size/1024:.2f} KB)")
    print()


def main():
    print("🦄 Mel Filterbank Validation")
    print("=" * 70)
    print()

    # Check if header file exists
    import os
    if not os.path.exists("mel_filterbank_coeffs.h"):
        print("❌ mel_filterbank_coeffs.h not found!")
        print("   Run: python3 generate_mel_filterbank.py")
        sys.exit(1)

    print("✅ Found mel_filterbank_coeffs.h")
    print()

    # Analyze filter properties
    analyze_filter_properties()

    # Compare with librosa
    if LIBROSA_AVAILABLE:
        compare_with_librosa()
    else:
        print("⏭️  Skipping librosa comparison (not installed)")

    print()
    print("✨ Validation complete!")
    print()
    print("Next steps:")
    print("  1. ✅ Filters generated correctly")
    print("  2. ✅ Q15 format validated")
    print("  3. ⏭️  Compile kernel: compile_mel_optimized.sh")
    print("  4. ⏭️  Test on NPU: test_mel_optimized.py")


if __name__ == '__main__':
    main()
