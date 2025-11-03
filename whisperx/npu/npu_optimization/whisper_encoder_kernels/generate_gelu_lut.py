#!/usr/bin/env python3
"""
Generate GELU Lookup Table for INT8 Implementation

Creates a precomputed lookup table for GELU activation function
optimized for AMD Phoenix NPU INT8 operations.

GELU(x) = x * Φ(x) where Φ(x) is Gaussian CDF
Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

For INT8 range [-128, 127] normalized to [-1, 1]
"""

import numpy as np
import math

def gelu_numpy(x):
    """
    GELU activation using numpy
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))

def generate_gelu_lut():
    """Generate GELU lookup table for INT8 range"""

    # INT8 input range: [-128, 127]
    x_int8 = np.arange(-128, 128, dtype=np.int32)

    # Normalize to approximate floating point range
    # Typical normalization: x_float = x_int8 / 127.0
    x_float = x_int8.astype(np.float32) / 127.0

    # Compute GELU using numpy approximation
    gelu_float = gelu_numpy(x_float)

    # Requantize back to INT8
    # GELU output range is roughly same as input for this scale
    gelu_int8 = np.clip(np.round(gelu_float * 127.0), -128, 127).astype(np.int8)

    return gelu_int8, x_int8, x_float, gelu_float

def print_c_array(gelu_int8):
    """Print as C array for inclusion in kernel"""
    print("/**")
    print(" * GELU Lookup Table for INT8")
    print(" * Precomputed for range [-128, 127]")
    print(" * Index = input_value + 128 (to map to [0, 255])")
    print(" * Generated using: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))")
    print(" */")
    print("static const int8_t gelu_lut[256] = {")

    # Print in rows of 16 for readability
    for i in range(0, 256, 16):
        row = gelu_int8[i:i+16]
        row_str = ", ".join(f"{val:4d}" for val in row)
        if i + 16 < 256:
            print(f"    {row_str},")
        else:
            print(f"    {row_str}")

    print("};")

def print_statistics(x_int8, x_float, gelu_int8, gelu_float):
    """Print statistics for validation"""
    print("\n" + "="*70)
    print("GELU LUT Statistics")
    print("="*70)

    # Compute statistics
    gelu_float_requantized = gelu_int8.astype(np.float32) / 127.0

    print(f"\nInput Range (INT8):  [{x_int8.min()}, {x_int8.max()}]")
    print(f"Input Range (Float): [{x_float.min():.4f}, {x_float.max():.4f}]")
    print(f"\nOutput Range (INT8):  [{gelu_int8.min()}, {gelu_int8.max()}]")
    print(f"Output Range (Float): [{gelu_float.min():.4f}, {gelu_float.max():.4f}]")

    # Error analysis
    abs_error = np.abs(gelu_float_requantized - gelu_float)
    print(f"\nQuantization Error (Float scale):")
    print(f"  Mean Absolute Error: {abs_error.mean():.6f}")
    print(f"  Max Absolute Error:  {abs_error.max():.6f}")
    print(f"  RMS Error:           {np.sqrt((abs_error**2).mean()):.6f}")

    # INT8 error
    int8_error = np.abs(gelu_int8.astype(np.float32) - gelu_float * 127.0)
    print(f"\nQuantization Error (INT8 scale):")
    print(f"  Mean Absolute Error: {int8_error.mean():.2f} units")
    print(f"  Max Absolute Error:  {int8_error.max():.2f} units")
    print(f"  RMS Error:           {np.sqrt((int8_error**2).mean()):.2f} units")

    # Special values
    print(f"\nSpecial Values:")
    print(f"  GELU(-128) = {gelu_int8[0]:4d}  (x = {x_float[0]:.4f}, expected ≈ -128)")
    print(f"  GELU(   0) = {gelu_int8[128]:4d}  (x = {x_float[128]:.4f}, expected ≈ 0)")
    print(f"  GELU( 127) = {gelu_int8[255]:4d}  (x = {x_float[255]:.4f}, expected ≈ 127)")

    # Monotonicity check
    is_monotonic = np.all(np.diff(gelu_int8) >= 0)
    print(f"\nMonotonicity: {'✅ PASS' if is_monotonic else '❌ FAIL'}")

    print("="*70)

def save_binary_lut(gelu_int8, filename="gelu_lut.bin"):
    """Save LUT as binary file for potential runtime loading"""
    gelu_int8.tofile(filename)
    print(f"\n✅ Binary LUT saved to: {filename} ({len(gelu_int8)} bytes)")

def main():
    print("="*70)
    print("GELU Lookup Table Generator for AMD Phoenix NPU")
    print("="*70)
    print()

    # Generate LUT
    gelu_int8, x_int8, x_float, gelu_float = generate_gelu_lut()

    # Print C array
    print_c_array(gelu_int8)

    # Print statistics
    print_statistics(x_int8, x_float, gelu_int8, gelu_float)

    # Save binary version
    save_binary_lut(gelu_int8)

    print("\n✅ GELU LUT generation complete!")
    print("\nUsage in C kernel:")
    print("  uint8_t idx = (uint8_t)(input[i] + 128);")
    print("  output[i] = gelu_lut[idx];")
    print()

if __name__ == "__main__":
    main()
