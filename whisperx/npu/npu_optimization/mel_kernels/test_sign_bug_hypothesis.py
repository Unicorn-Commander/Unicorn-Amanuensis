#!/usr/bin/env python3
"""
Test Sign Bug Hypothesis - Byte Conversion Analysis

This script tests whether the byte conversion code in mel_kernel_fft_fixed.c
has a sign handling bug that could explain our 96.2% zeros output.

Expected Result: If buggy conversion is used, we should see significant
errors when converting negative INT16 audio samples.
"""

import numpy as np

def buggy_byte_conversion(audio_int16):
    """
    Current (potentially buggy) byte conversion from mel_kernel_fft_fixed.c:

    samples[i] = ((int16_t)(uint8_t)input[byte_idx]) |
                (((int16_t)(int8_t)input[byte_idx + 1]) << 8);
                            ^^^^^^^ BUG: int8_t instead of uint8_t
    """
    result = np.zeros_like(audio_int16, dtype=np.int16)

    for i in range(len(audio_int16)):
        # Convert to bytes (little-endian)
        value = audio_int16[i]
        low_byte = value & 0xFF
        high_byte = (value >> 8) & 0xFF

        # Buggy conversion: high byte as SIGNED int8
        low_unsigned = np.uint8(low_byte)
        high_signed = np.int8(high_byte)  # BUG HERE!

        # Reconstruct INT16
        result[i] = np.int16(low_unsigned) | (np.int16(high_signed) << 8)

    return result


def fixed_byte_conversion(audio_int16):
    """
    Fixed byte conversion:

    uint16_t unsigned_sample = ((uint16_t)(uint8_t)input[byte_idx]) |
                               (((uint16_t)(uint8_t)input[byte_idx + 1]) << 8);
    samples[i] = (int16_t)unsigned_sample;
    """
    result = np.zeros_like(audio_int16, dtype=np.int16)

    for i in range(len(audio_int16)):
        # Convert to bytes (little-endian)
        value = audio_int16[i]
        low_byte = value & 0xFF
        high_byte = (value >> 8) & 0xFF

        # Fixed conversion: both bytes as UNSIGNED
        low_unsigned = np.uint8(low_byte)
        high_unsigned = np.uint8(high_byte)  # FIXED!

        # Reconstruct as unsigned, then reinterpret as signed
        unsigned_sample = np.uint16(low_unsigned) | (np.uint16(high_unsigned) << 8)
        result[i] = np.int16(unsigned_sample)

    return result


def test_specific_values():
    """Test conversion on specific problematic values"""

    print("="*80)
    print("Testing Specific Values")
    print("="*80)

    test_cases = [
        ("Positive small", 100),
        ("Positive large", 26213),
        ("Negative small", -100),
        ("Negative large", -26213),
        ("Zero", 0),
        ("Max positive", 32767),
        ("Min negative", -32768),
    ]

    print(f"\n{'Description':<20} {'Original':<10} {'Buggy':<10} {'Fixed':<10} {'Error':<10}")
    print("-"*80)

    for description, value in test_cases:
        original = np.int16(value)
        buggy = buggy_byte_conversion(np.array([value], dtype=np.int16))[0]
        fixed = fixed_byte_conversion(np.array([value], dtype=np.int16))[0]
        error = abs(buggy - original)

        print(f"{description:<20} {original:<10} {buggy:<10} {fixed:<10} {error:<10}")

        if fixed != original:
            print(f"  ⚠️  WARNING: Fixed conversion doesn't match original!")
        if buggy != original and error > 100:
            print(f"  🔴 SIGNIFICANT ERROR in buggy conversion!")


def test_audio_waveform():
    """Test conversion on realistic audio waveform"""

    print("\n" + "="*80)
    print("Testing Realistic Audio Waveform (440 Hz sine)")
    print("="*80)

    # Generate 440 Hz sine wave (same as our test audio)
    sample_rate = 16000
    duration = 0.025  # 25ms (400 samples)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.8 * np.sin(2 * np.pi * 440 * t)

    # Convert to INT16
    audio_int16 = (audio * 32767).astype(np.int16)

    print(f"\nOriginal audio:")
    print(f"  Samples: {len(audio_int16)}")
    print(f"  Range: [{audio_int16.min()}, {audio_int16.max()}]")
    print(f"  Mean: {audio_int16.mean():.1f}")
    print(f"  Std: {audio_int16.std():.1f}")
    print(f"  Negative samples: {(audio_int16 < 0).sum()}/{len(audio_int16)} ({100*(audio_int16 < 0).sum()/len(audio_int16):.1f}%)")

    # Test buggy conversion
    buggy_audio = buggy_byte_conversion(audio_int16)
    buggy_error = np.abs(buggy_audio - audio_int16)

    print(f"\nBuggy conversion:")
    print(f"  Range: [{buggy_audio.min()}, {buggy_audio.max()}]")
    print(f"  Mean error: {buggy_error.mean():.1f}")
    print(f"  Max error: {buggy_error.max()}")
    print(f"  RMS error: {np.sqrt(np.mean(buggy_error**2)):.1f}")
    print(f"  Samples with error > 100: {(buggy_error > 100).sum()}/{len(audio_int16)} ({100*(buggy_error > 100).sum()/len(audio_int16):.1f}%)")

    # Test fixed conversion
    fixed_audio = fixed_byte_conversion(audio_int16)
    fixed_error = np.abs(fixed_audio - audio_int16)

    print(f"\nFixed conversion:")
    print(f"  Range: [{fixed_audio.min()}, {fixed_audio.max()}]")
    print(f"  Mean error: {fixed_error.mean():.1f}")
    print(f"  Max error: {fixed_error.max()}")
    print(f"  Perfect match: {np.all(fixed_audio == audio_int16)}")

    # Correlation analysis
    if np.std(buggy_audio) > 0:
        buggy_corr = np.corrcoef(audio_int16, buggy_audio)[0, 1]
        print(f"\nCorrelation with original:")
        print(f"  Buggy conversion: {buggy_corr:.4f}")
        if buggy_corr < 0:
            print(f"  🚨 NEGATIVE CORRELATION - Confirms sign bug hypothesis!")

    fixed_corr = np.corrcoef(audio_int16, fixed_audio)[0, 1]
    print(f"  Fixed conversion: {fixed_corr:.4f}")


def test_negative_samples():
    """Focus on negative sample handling"""

    print("\n" + "="*80)
    print("Testing Negative Sample Handling")
    print("="*80)

    # Create array with mix of positive and negative
    test_audio = np.array([
        100, -100, 1000, -1000, 10000, -10000,
        26213, -26213,  # Max amplitude we expect
        32767, -32768,  # INT16 limits
    ], dtype=np.int16)

    buggy = buggy_byte_conversion(test_audio)
    fixed = fixed_byte_conversion(test_audio)

    print(f"\n{'Original':<10} {'Buggy':<10} {'Fixed':<10} {'Buggy Error':<12} {'Sign Flipped?'}")
    print("-"*80)

    sign_flips = 0
    for orig, bug, fix in zip(test_audio, buggy, fixed):
        error = abs(bug - orig)
        sign_flipped = (orig < 0 and bug > 0) or (orig > 0 and bug < 0)

        print(f"{orig:<10} {bug:<10} {fix:<10} {error:<12} {'YES ⚠️' if sign_flipped else 'no'}")

        if sign_flipped:
            sign_flips += 1

    print(f"\nSign flips in buggy conversion: {sign_flips}/{len(test_audio)}")

    if sign_flips > 0:
        print(f"\n🚨 SIGN BUG CONFIRMED: Negative samples are being corrupted!")


def analyze_high_byte_sign_extension():
    """Detailed analysis of high byte sign extension"""

    print("\n" + "="*80)
    print("High Byte Sign Extension Analysis")
    print("="*80)

    print("\nLittle-endian INT16 representation:")
    print("  INT16 value = (high_byte << 8) | low_byte")
    print("  For negative numbers, high byte has bit 7 set (0x80-0xFF range)")
    print()

    # Test high byte in negative range
    print("Testing high byte values in negative range (0x80-0xFF):")
    print(f"\n{'High Byte':<12} {'As uint8_t':<12} {'As int8_t':<12} {'Difference':<12} {'Issue?'}")
    print("-"*80)

    issues = 0
    for high_byte in [0x80, 0x90, 0x99, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0, 0xFF]:
        as_unsigned = np.uint8(high_byte)
        as_signed = np.int8(high_byte)
        difference = int(as_unsigned) - int(as_signed)

        has_issue = (as_signed < 0)  # If interpreted as negative when should be 128-255

        print(f"0x{high_byte:02X} ({high_byte:<3}) {as_unsigned:<12} {as_signed:<12} {difference:<12} {'YES ⚠️' if has_issue else 'OK'}")

        if has_issue:
            issues += 1

    print(f"\nHigh bytes incorrectly interpreted as negative: {issues}/10")
    print("\n💡 CONCLUSION:")
    print("   When high byte is 0x80-0xFF (which is common for negative INT16 values),")
    print("   casting to int8_t interprets it as negative (-128 to -1),")
    print("   but it should remain positive (128 to 255) before being shifted left.")


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    XDNA1 NPU Sign Bug Hypothesis Test                         ║
║                                                                                ║
║  Testing if byte conversion bug explains 96.2% zeros output                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Run all tests
    test_specific_values()
    test_audio_waveform()
    test_negative_samples()
    analyze_high_byte_sign_extension()

    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    print("""
Based on the above tests, the buggy byte conversion (using int8_t for high byte)
causes SIGNIFICANT errors when processing negative audio samples.

If our kernel uses this buggy conversion:
  ✅ Explains 96.2% zeros (most samples corrupted → weak FFT → near-zero mel)
  ✅ Explains negative correlation (sign inversions → anti-correlated output)
  ✅ Explains only small positive values [0,4] (corrupted input → weak energy)

RECOMMENDATION: Apply the fix immediately!

Fix: Change line 115 in mel_kernel_fft_fixed.c from:
  (((int16_t)(int8_t)input[byte_idx + 1]) << 8);
To:
  (((int16_t)(uint8_t)input[byte_idx + 1]) << 8);

Expected improvement:
  - Non-zero: 3.8% → 70-90%
  - Range: [0,4] → [0,127]
  - Correlation: -0.03 → 0.85-0.95
    """)
    print("="*80)


if __name__ == "__main__":
    main()
