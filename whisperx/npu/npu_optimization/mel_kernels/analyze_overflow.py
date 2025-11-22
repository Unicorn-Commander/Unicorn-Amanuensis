#!/usr/bin/env python3
"""
Analyze the overflow problem more carefully.
"""

import numpy as np

print("="*70)
print("OVERFLOW ANALYSIS")
print("="*70)

# Simulate the pipeline
print("\n1. Input: INT16 audio (amplitude 0.95)")
audio_int16 = int(0.95 * 32767)
print(f"   Value: {audio_int16} (±31129)")

print("\n2. After FFT (9 stages, each >>1)")
fft_scaled = audio_int16
for stage in range(9):
    fft_scaled = (fft_scaled + 1) >> 1
print(f"   After 9 stages of >>1: {fft_scaled}")
print(f"   Scaling factor: {audio_int16 / max(fft_scaled, 1):.1f}x reduction")

print("\n3. Magnitude² computation")
mag_sq_q30 = fft_scaled * fft_scaled
print(f"   FFT² (Q30): {mag_sq_q30}")
print(f"   Bits needed: {mag_sq_q30.bit_length()}")

print("\n4. Different shift strategies:")
for shift in [7, 10, 12, 15]:
    result = mag_sq_q30 >> shift
    print(f"   >>{ shift}: {result:10d} (Q{30-shift})")

print("\n5. Problem with storing in int16_t:")
print(f"   int16_t max: 32767")
print(f"   Magnitude² >>7 (Q23): {mag_sq_q30 >> 7} - {'OVERFLOWS!' if (mag_sq_q30 >> 7) > 32767 else 'OK'}")
print(f"   Magnitude² >>15 (Q15): {mag_sq_q30 >> 15} - {'too small' if (mag_sq_q30 >> 15) < 100 else 'OK'}")

print("\n" + "="*70)
print("SOLUTION:")
print("="*70)
print("The int16_t storage is the constraint!")
print("")
print("Option A: Use >>12 to get Q18 (fits in int16_t)")
mag_q18 = mag_sq_q30 >> 12
print(f"  Magnitude >>12 (Q18): {mag_q18}")
print(f"  Fits in int16_t: {mag_q18 <= 32767}")
print(f"  Then scale mel_energy from Q18 to INT8: (energy * 127) >> 18")

print("\nOption B: Use >>10 to get Q20 (may overflow int16_t)")
mag_q20 = mag_sq_q30 >> 10
print(f"  Magnitude >>10 (Q20): {mag_q20}")
print(f"  Fits in int16_t: {mag_q20 <= 32767}")

print("\nOption C: Don't shift, clamp, then compensate in mel_energy")
mag_unshifted = min(mag_sq_q30, 32767)
print(f"  Magnitude (clamped to 32767): {mag_unshifted}")
print(f"  Lost precision: {mag_sq_q30 / max(mag_unshifted, 1):.1f}x")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)
print("Use >>12 shift (Q18 format) as compromise:")
print("  - Preserves 3 more bits than Q15 (8x more precision)")
print("  - Fits in int16_t without overflow")
print("  - Adjust final scaling: (mel_energy * 127) >> 18")
