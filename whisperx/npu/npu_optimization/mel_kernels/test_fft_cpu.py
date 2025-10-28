#!/usr/bin/env python3
"""
Test FFT implementation on CPU to identify errors before fixing NPU kernel
This simulates the exact Q15 fixed-point arithmetic used in the NPU kernel
"""

import numpy as np
import matplotlib.pyplot as plt

# Q15 fixed-point conversion
Q15_SCALE = 32767  # 2^15 - 1

def to_q15(value):
    """Convert float to Q15 fixed-point"""
    return np.clip(np.round(value * Q15_SCALE), -32768, 32767).astype(np.int16)

def from_q15(value):
    """Convert Q15 fixed-point to float"""
    return value.astype(np.float32) / Q15_SCALE

def mul_q15(a, b):
    """Q15 multiplication with proper scaling"""
    product = a.astype(np.int32) * b.astype(np.int32)
    # Round and scale back to Q15
    return np.clip((product + (1 << 14)) >> 15, -32768, 32767).astype(np.int16)

def bit_reverse_512(x):
    """Bit-reverse for 512-point FFT (9 bits)"""
    reversed_x = 0
    for i in range(9):
        reversed_x = (reversed_x << 1) | (x & 1)
        x >>= 1
    return reversed_x

def fft_radix2_512_q15_python(input_samples):
    """
    Python implementation of the exact FFT in fft_fixed_point.c
    This simulates Q15 arithmetic to find bugs before fixing C code
    """
    FFT_SIZE = 512
    LOG2_SIZE = 9

    # Generate twiddle factors (same as fft_coeffs_fixed.h)
    twiddle_cos_q15 = np.zeros(256, dtype=np.int16)
    twiddle_sin_q15 = np.zeros(256, dtype=np.int16)

    for k in range(256):
        angle = -2 * np.pi * k / FFT_SIZE
        twiddle_cos_q15[k] = to_q15(np.cos(angle))
        twiddle_sin_q15[k] = to_q15(np.sin(angle))

    # Initialize output with bit-reversed input
    output_real = np.zeros(FFT_SIZE, dtype=np.int16)
    output_imag = np.zeros(FFT_SIZE, dtype=np.int16)

    for i in range(FFT_SIZE):
        rev = bit_reverse_512(i)
        output_real[rev] = input_samples[i]
        output_imag[rev] = 0

    # FFT butterfly stages
    for stage in range(LOG2_SIZE):
        m = 1 << (stage + 1)
        half_m = m >> 1
        twid_step = FFT_SIZE // m

        for k in range(0, FFT_SIZE, m):
            for j in range(half_m):
                idx_even = k + j
                idx_odd = k + j + half_m

                # Get twiddle factor
                twid_idx = j * twid_step
                W_real = twiddle_cos_q15[twid_idx]
                W_imag = twiddle_sin_q15[twid_idx]

                # Get even and odd samples
                even_real = output_real[idx_even]
                even_imag = output_imag[idx_even]
                odd_real = output_real[idx_odd]
                odd_imag = output_imag[idx_odd]

                # Complex multiplication: t = W * odd
                # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
                ac = odd_real.astype(np.int32) * W_real.astype(np.int32)
                bd = odd_imag.astype(np.int32) * W_imag.astype(np.int32)
                ad = odd_real.astype(np.int32) * W_imag.astype(np.int32)
                bc = odd_imag.astype(np.int32) * W_real.astype(np.int32)

                t_real = np.clip((ac - bd + (1 << 14)) >> 15, -32768, 32767).astype(np.int16)
                t_imag = np.clip((ad + bc + (1 << 14)) >> 15, -32768, 32767).astype(np.int16)

                # Butterfly: even ± t
                # CRITICAL FIX: Scale down by 2 at each stage to prevent overflow
                sum_real = even_real.astype(np.int32) + t_real.astype(np.int32)
                sum_imag = even_imag.astype(np.int32) + t_imag.astype(np.int32)
                diff_real = even_real.astype(np.int32) - t_real.astype(np.int32)
                diff_imag = even_imag.astype(np.int32) - t_imag.astype(np.int32)

                # Scale down by 2 (with rounding) to prevent overflow
                output_real[idx_even] = ((sum_real + 1) >> 1).astype(np.int16)
                output_imag[idx_even] = ((sum_imag + 1) >> 1).astype(np.int16)
                output_real[idx_odd] = ((diff_real + 1) >> 1).astype(np.int16)
                output_imag[idx_odd] = ((diff_imag + 1) >> 1).astype(np.int16)

    return output_real, output_imag

def test_fft_q15():
    """Test Q15 FFT implementation against numpy"""

    print("="*70)
    print("FFT Q15 IMPLEMENTATION TEST")
    print("="*70)

    # Test 1: DC signal (constant value)
    print("\nTest 1: DC Signal (all samples = 0.5)")
    samples_dc = np.ones(512, dtype=np.float32) * 0.5
    samples_dc_q15 = to_q15(samples_dc)

    # Reference: numpy FFT
    fft_ref = np.fft.rfft(samples_dc, n=512)
    magnitude_ref = np.abs(fft_ref)

    # Q15 FFT
    real_q15, imag_q15 = fft_radix2_512_q15_python(samples_dc_q15)
    magnitude_q15 = np.sqrt(real_q15.astype(np.float32)**2 + imag_q15.astype(np.float32)**2)

    # DC should have all energy in bin 0
    print(f"  NumPy FFT bin 0: {magnitude_ref[0]:.2f}")
    print(f"  Q15 FFT bin 0:   {magnitude_q15[0]:.2f}")
    print(f"  Ratio: {magnitude_q15[0] / magnitude_ref[0]:.4f}")

    # Test 2: 1000 Hz sine wave
    print("\nTest 2: 1000 Hz Sine Wave (16 kHz sample rate)")
    sr = 16000
    duration = 512 / sr
    t = np.linspace(0, duration, 512, endpoint=False)
    freq = 1000
    samples_sine = np.sin(2 * np.pi * freq * t).astype(np.float32)
    samples_sine_q15 = to_q15(samples_sine)

    # Reference
    fft_ref = np.fft.rfft(samples_sine, n=512)
    magnitude_ref = np.abs(fft_ref)
    peak_bin_ref = np.argmax(magnitude_ref)
    expected_bin = int(freq * 512 / sr)  # 32

    # Q15 FFT
    real_q15, imag_q15 = fft_radix2_512_q15_python(samples_sine_q15)
    magnitude_q15 = np.sqrt(real_q15.astype(np.float32)**2 + imag_q15.astype(np.float32)**2)

    # Only look at first half (0-256) for real FFT
    magnitude_q15_first_half = magnitude_q15[:257]
    peak_bin_q15 = np.argmax(magnitude_q15_first_half)

    print(f"  Expected peak bin: {expected_bin}")
    print(f"  NumPy peak bin: {peak_bin_ref}")
    print(f"  Q15 peak bin: {peak_bin_q15}")
    print(f"  NumPy peak magnitude: {magnitude_ref[peak_bin_ref]:.2f}")
    print(f"  Q15 peak magnitude: {magnitude_q15_first_half[peak_bin_q15]:.2f}")

    # Correlation between Q15 and numpy
    # Normalize both for comparison
    magnitude_ref_norm = magnitude_ref / magnitude_ref.max()
    magnitude_q15_norm = magnitude_q15_first_half / magnitude_q15_first_half.max()
    correlation = np.corrcoef(magnitude_ref_norm, magnitude_q15_norm)[0, 1]
    print(f"  Correlation with numpy: {correlation:.4f}")

    # Plot comparison
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(magnitude_ref, label='NumPy FFT', alpha=0.7)
    plt.plot(magnitude_q15[:257], label='Q15 FFT', alpha=0.7)
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    plt.title('FFT Comparison (1000 Hz Sine)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(magnitude_ref[:100], label='NumPy FFT', alpha=0.7)
    plt.plot(magnitude_q15[:100], label='Q15 FFT', alpha=0.7)
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    plt.title('FFT Comparison (Zoomed to 0-100 bins)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fft_q15_test.png', dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: fft_q15_test.png")

    # Test 3: Impulse (delta function)
    print("\nTest 3: Impulse (delta function)")
    samples_impulse = np.zeros(512, dtype=np.float32)
    samples_impulse[0] = 1.0
    samples_impulse_q15 = to_q15(samples_impulse)

    # Reference
    fft_ref = np.fft.rfft(samples_impulse, n=512)
    magnitude_ref = np.abs(fft_ref)

    # Q15 FFT
    real_q15, imag_q15 = fft_radix2_512_q15_python(samples_impulse_q15)
    magnitude_q15 = np.sqrt(real_q15.astype(np.float32)**2 + imag_q15.astype(np.float32)**2)

    # Impulse should have flat spectrum
    correlation = np.corrcoef(magnitude_ref, magnitude_q15[:257])[0, 1]
    print(f"  Correlation with numpy: {correlation:.4f}")
    print(f"  NumPy mean magnitude: {magnitude_ref.mean():.2f}")
    print(f"  Q15 mean magnitude: {magnitude_q15[:257].mean():.2f}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Q15 FFT implementation: {'✅ WORKING' if correlation > 0.9 else '❌ BROKEN'}")
    print(f"Target correlation: >0.99")
    print(f"Actual correlation: {correlation:.4f}")

    if correlation < 0.9:
        print("\n⚠️  FFT implementation has errors!")
        print("Possible issues:")
        print("  - Bit-reversal errors")
        print("  - Twiddle factor sign errors")
        print("  - Q15 overflow during butterfly operations")
        print("  - Incorrect complex multiplication")

    return correlation

if __name__ == "__main__":
    correlation = test_fft_q15()
