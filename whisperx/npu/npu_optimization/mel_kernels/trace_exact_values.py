#!/usr/bin/env python3
"""
Trace exact values through each stage to find where precision is lost.
"""

import numpy as np

print("="*70)
print("EXACT VALUE TRACE")
print("="*70)

# Input: 0.95 amplitude sine wave at 1000 Hz
audio_float = 0.95
audio_int16 = int(audio_float * 32767)
print(f"\n1. Input INT16: {audio_int16}")

# After FFT scaling (9 stages of >>1)
fft_q15 = audio_int16
for stage in range(9):
    fft_q15 = (fft_q15 + 1) >> 1
print(f"2. After FFT (9x >>1): {fft_q15}")

# Magnitude² in Q30
mag_sq_q30 = fft_q15 * fft_q15
print(f"3. Magnitude² (Q30, no shift): {mag_sq_q30}")

# Mel filter weight (assume max weight ~0.5 in Q15)
weight_q15 = int(0.5 * 32767)
print(f"4. Mel filter weight (Q15): {weight_q15}")

# Weighted magnitude (Q30 × Q15 = Q45, then >>15 → Q30)
weighted_q45 = mag_sq_q30 * weight_q15
weighted_q30 = weighted_q45 >> 15
print(f"5. Weighted (Q30 × Q15 >> 15): {weighted_q30}")

# Accumulate across ~10 bins in a mel filter
mel_energy_q30 = weighted_q30 * 10
print(f"6. Mel energy (sum 10 bins, Q30): {mel_energy_q30}")

# Scale to INT8 (Q30 >> 30) × 127
scaled_int8 = (mel_energy_q30 * 127) >> 30
print(f"7. Scaled to INT8 ((energy × 127) >> 30): {scaled_int8}")

print(f"\n" + "="*70)
print("PROBLEM:")
print("="*70)
print(f"Expected INT8 value: 50-100 (for strong signal)")
print(f"Actual INT8 value: {scaled_int8}")
print(f"\nRatio: {scaled_int8} / 80 = {scaled_int8/80:.4f}x (should be ~1.0x)")

print(f"\n" + "="*70)
print("ROOT CAUSE:")
print("="*70)
print(f"The FFT scaling (/512) reduces values so much that:")
print(f"  Input: {audio_int16}")
print(f"  FFT:   {fft_q15}  (512x smaller)")
print(f"  Mag²:  {mag_sq_q30}  (very small even without shifting!)")
print(f"")
print(f"Even keeping full Q30 precision, the values are ~{mag_sq_q30} which is tiny.")
print(f"After mel filtering and >>30 scaling, we get {scaled_int8} instead of ~80.")

print(f"\n" + "="*70)
print("SOLUTION:")
print("="*70)
print(f"The final scaling is wrong for FFT-scaled values!")
print(f"")
print(f"Current: (mel_energy * 127) >> 30")
print(f"Problem: This assumes mel_energy uses full Q30 range [0, 2^30-1]")
print(f"Reality: After FFT /512 scaling, mel_energy is ~512²x = 262144x smaller!")
print(f"")
print(f"FIX: Compensate for FFT scaling in final conversion:")
print(f"  FFT divides by 512, so magnitude² is divided by 262144")
print(f"  Need to multiply mel_energy by scaling factor BEFORE >>30")
print(f"  Or equivalently: shift by LESS than 30")
print(f"")

# Calculate correct shift
fft_scale_factor = 512  # 9 stages of /2
mag_sq_reduction = fft_scale_factor ** 2  # 262144
print(f"FFT scaling reduces magnitude² by: {mag_sq_reduction}x")
print(f"Log2 of reduction: {np.log2(mag_sq_reduction):.1f} bits")
print(f"")
print(f"So instead of >>30, we should use >>{30 - int(np.log2(mag_sq_reduction))}")
print(f"That's >>12 instead of >>30!")

# Test the fix
corrected_scaled = (mel_energy_q30 * 127) >> 12
print(f"\nWith corrected scaling ((energy × 127) >> 12): {corrected_scaled}")
print(f"This is {corrected_scaled/80:.2f}x the expected value (much better!)")
