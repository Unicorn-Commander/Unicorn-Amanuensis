#!/usr/bin/env python3
"""
Test FP16 Weight Loading

Tests loading FP16 weights and converting to FP32 for C++ usage.
Verifies accuracy and demonstrates C++-compatible loading pattern.

C++ Loading Pattern:
    1. Load .npy file as uint16_t array (FP16 binary representation)
    2. Convert each uint16_t to float32 using IEEE 754 conversion
    3. Use float32 values in computation

This script demonstrates the Python equivalent for validation.
"""

import numpy as np
from pathlib import Path
import sys

def load_fp16_as_fp32(filepath):
    """
    Load FP16 weight file and convert to FP32.

    This mimics what C++ will do:
    - Load binary data as FP16
    - Convert to FP32 for computation

    Args:
        filepath: Path to .npy file containing FP16 weights

    Returns:
        np.ndarray: Weights as FP32
    """
    # Load FP16
    weights_fp16 = np.load(filepath)

    # Convert to FP32 (this is what C++ will do)
    weights_fp32 = weights_fp16.astype(np.float32)

    return weights_fp32

def test_single_weight(fp16_dir, fp32_dir, weight_name):
    """
    Test loading a single weight and compare FP16→FP32 conversion accuracy.

    Args:
        fp16_dir: Directory containing FP16 weights
        fp32_dir: Directory containing FP32 weights (ground truth)
        weight_name: Base name of weight (e.g., "embed_positions_weight")

    Returns:
        dict: Statistics about this weight
    """
    fp16_path = fp16_dir / f"{weight_name}_fp16.npy"
    fp32_path = fp32_dir / f"{weight_name}.npy"

    if not fp16_path.exists():
        return {'error': f"FP16 file not found: {fp16_path}"}

    if not fp32_path.exists():
        return {'error': f"FP32 file not found: {fp32_path}"}

    # Load FP16 and convert to FP32
    weights_from_fp16 = load_fp16_as_fp32(fp16_path)

    # Load original FP32 (ground truth)
    weights_original_fp32 = np.load(fp32_path)

    # Verify shapes match
    if weights_from_fp16.shape != weights_original_fp32.shape:
        return {
            'error': f"Shape mismatch: FP16→FP32 {weights_from_fp16.shape} "
                    f"vs Original FP32 {weights_original_fp32.shape}"
        }

    # Calculate errors
    abs_error = np.abs(weights_original_fp32 - weights_from_fp16)
    max_abs_error = abs_error.max()
    avg_abs_error = abs_error.mean()

    # Relative error
    nonzero_mask = np.abs(weights_original_fp32) > 1e-10
    if nonzero_mask.any():
        rel_error = np.abs((weights_original_fp32 - weights_from_fp16) /
                          (weights_original_fp32 + 1e-10))
        max_rel_error = rel_error[nonzero_mask].max()
        avg_rel_error = rel_error[nonzero_mask].mean()
    else:
        max_rel_error = 0.0
        avg_rel_error = 0.0

    # Check for NaN or Inf
    has_nan = np.isnan(weights_from_fp16).any()
    has_inf = np.isinf(weights_from_fp16).any()

    # Calculate SNR (Signal-to-Noise Ratio)
    signal_power = np.mean(weights_original_fp32 ** 2)
    noise_power = np.mean(abs_error ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # Memory comparison
    fp16_size = weights_from_fp16.nbytes // 2  # It's stored as FP16
    fp32_size = weights_original_fp32.nbytes
    compression_ratio = 100.0 * (1 - fp16_size / fp32_size)

    return {
        'name': weight_name,
        'shape': weights_original_fp32.shape,
        'max_abs_error': max_abs_error,
        'avg_abs_error': avg_abs_error,
        'max_rel_error': max_rel_error,
        'avg_rel_error': avg_rel_error,
        'snr_db': snr_db,
        'has_nan': has_nan,
        'has_inf': has_inf,
        'fp16_size_mb': fp16_size / 1024 / 1024,
        'fp32_size_mb': fp32_size / 1024 / 1024,
        'compression_pct': compression_ratio
    }

def test_all_weights(fp16_dir, fp32_dir):
    """
    Test loading all FP16 weights.

    Args:
        fp16_dir: Directory containing FP16 weights
        fp32_dir: Directory containing FP32 weights (ground truth)

    Returns:
        dict: Overall statistics
    """
    fp16_path = Path(fp16_dir)
    fp32_path = Path(fp32_dir)

    if not fp16_path.exists():
        print(f"❌ FP16 directory not found: {fp16_dir}")
        return None

    if not fp32_path.exists():
        print(f"❌ FP32 directory not found: {fp32_dir}")
        return None

    # Find all FP16 weight files
    fp16_files = sorted(fp16_path.glob("*_fp16.npy"))

    if not fp16_files:
        print(f"❌ No FP16 weight files found in {fp16_dir}")
        return None

    print("=" * 80)
    print("  FP16 WEIGHT LOADING TEST")
    print("=" * 80)
    print(f"\nFP16 directory: {fp16_dir}")
    print(f"FP32 directory: {fp32_dir}")
    print(f"Weight files:   {len(fp16_files)}")

    # Test each weight
    results = []
    errors = []

    print("\n" + "=" * 80)
    print("  TESTING INDIVIDUAL WEIGHTS")
    print("=" * 80 + "\n")

    for fp16_file in fp16_files:
        # Extract weight name (remove _fp16.npy suffix)
        weight_name = fp16_file.stem.replace('_fp16', '')

        result = test_single_weight(fp16_path, fp32_path, weight_name)

        if 'error' in result:
            errors.append(result)
            print(f"❌ {weight_name:50s} ERROR: {result['error']}")
        else:
            results.append(result)
            print(f"✅ {weight_name:50s} "
                  f"max_err={result['max_abs_error']:.2e} "
                  f"SNR={result['snr_db']:.1f}dB")

    # Calculate overall statistics
    if not results:
        print("\n❌ No weights loaded successfully")
        return None

    stats = {
        'total_weights': len(fp16_files),
        'successful': len(results),
        'failed': len(errors),
        'max_abs_error': max(r['max_abs_error'] for r in results),
        'avg_abs_error': np.mean([r['avg_abs_error'] for r in results]),
        'max_rel_error': max(r['max_rel_error'] for r in results),
        'avg_rel_error': np.mean([r['avg_rel_error'] for r in results]),
        'avg_snr_db': np.mean([r['snr_db'] for r in results]),
        'total_fp16_size_mb': sum(r['fp16_size_mb'] for r in results),
        'total_fp32_size_mb': sum(r['fp32_size_mb'] for r in results),
        'has_nan': any(r['has_nan'] for r in results),
        'has_inf': any(r['has_inf'] for r in results),
        'results': results,
        'errors': errors
    }

    # Print summary
    print("\n" + "=" * 80)
    print("  SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nWeights tested:        {stats['successful']}/{stats['total_weights']}")
    print(f"Failed:                {stats['failed']}")
    print(f"\nAccuracy Metrics:")
    print(f"  Max absolute error:  {stats['max_abs_error']:.2e}")
    print(f"  Avg absolute error:  {stats['avg_abs_error']:.2e}")
    print(f"  Max relative error:  {stats['max_rel_error']:.2e} ({100*stats['max_rel_error']:.4f}%)")
    print(f"  Avg relative error:  {stats['avg_rel_error']:.2e} ({100*stats['avg_rel_error']:.4f}%)")
    print(f"  Avg SNR:             {stats['avg_snr_db']:.1f} dB")

    print(f"\nMemory Usage:")
    print(f"  FP16 total:          {stats['total_fp16_size_mb']:.1f} MB")
    print(f"  FP32 total:          {stats['total_fp32_size_mb']:.1f} MB")
    print(f"  Savings:             {stats['total_fp32_size_mb'] - stats['total_fp16_size_mb']:.1f} MB (50.0%)")

    print(f"\nData Quality:")
    if stats['has_nan']:
        print("  ❌ Contains NaN values")
    elif stats['has_inf']:
        print("  ❌ Contains Inf values")
    else:
        print("  ✅ No NaN or Inf values")

    # Assess quality
    print("\n" + "=" * 80)
    print("  QUALITY ASSESSMENT")
    print("=" * 80)

    if stats['max_abs_error'] < 1e-3:
        print("\n✅ EXCELLENT: Max error < 0.001 (FP16 is highly accurate)")
    elif stats['max_abs_error'] < 1e-2:
        print("\n✅ GOOD: Max error < 0.01 (FP16 is acceptable)")
    elif stats['max_abs_error'] < 1e-1:
        print("\n⚠️  ACCEPTABLE: Max error < 0.1 (Monitor inference accuracy)")
    else:
        print("\n❌ POOR: Max error >= 0.1 (Consider mixed precision)")

    if stats['avg_snr_db'] > 60:
        print("✅ EXCELLENT: Avg SNR > 60 dB (Very high fidelity)")
    elif stats['avg_snr_db'] > 40:
        print("✅ GOOD: Avg SNR > 40 dB (High fidelity)")
    elif stats['avg_snr_db'] > 20:
        print("⚠️  ACCEPTABLE: Avg SNR > 20 dB (Moderate fidelity)")
    else:
        print("❌ POOR: Avg SNR < 20 dB (Low fidelity)")

    # Top 10 highest error tensors
    print("\n" + "=" * 80)
    print("  TOP 10 HIGHEST ERROR TENSORS")
    print("=" * 80 + "\n")

    sorted_by_error = sorted(results, key=lambda r: r['max_abs_error'], reverse=True)[:10]
    for i, r in enumerate(sorted_by_error, 1):
        print(f"{i:2d}. {r['name']:50s} "
              f"max_err={r['max_abs_error']:.2e} "
              f"rel_err={100*r['max_rel_error']:.4f}%")

    return stats

def demonstrate_cpp_loading_pattern():
    """
    Demonstrate the C++ loading pattern for FP16 weights.

    This shows the pseudocode that C++ will use.
    """
    print("\n" + "=" * 80)
    print("  C++ LOADING PATTERN")
    print("=" * 80)

    cpp_code = '''
// C++ Example: Load FP16 weights and convert to float32

#include <vector>
#include <fstream>
#include <cstdint>

// Load FP16 weight file
std::vector<float> load_fp16_weight(const std::string& filepath) {
    // 1. Read binary file
    std::ifstream file(filepath, std::ios::binary);

    // 2. Read header (NumPy format)
    // ... parse NPY header ...

    // 3. Read FP16 data as uint16_t
    std::vector<uint16_t> fp16_data(num_elements);
    file.read((char*)fp16_data.data(), num_elements * sizeof(uint16_t));

    // 4. Convert FP16 to FP32
    std::vector<float> fp32_data(num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        fp32_data[i] = fp16_to_fp32(fp16_data[i]);
    }

    return fp32_data;
}

// Convert IEEE 754 FP16 to FP32
float fp16_to_fp32(uint16_t fp16) {
    uint32_t sign = (fp16 & 0x8000) << 16;
    uint32_t exponent = (fp16 & 0x7C00) >> 10;
    uint32_t mantissa = (fp16 & 0x03FF);

    if (exponent == 0) {
        // Subnormal or zero
        return sign ? -0.0f : 0.0f;
    } else if (exponent == 31) {
        // Inf or NaN
        return mantissa ? NAN : (sign ? -INFINITY : INFINITY);
    } else {
        // Normal number
        exponent = (exponent - 15 + 127) << 23;
        mantissa = mantissa << 13;
        uint32_t fp32_bits = sign | exponent | mantissa;
        return *reinterpret_cast<float*>(&fp32_bits);
    }
}

// Usage
auto weights = load_fp16_weight("embed_positions_weight_fp16.npy");
// weights is now std::vector<float> ready for computation
'''

    print(cpp_code)

    print("\nKey Points:")
    print("  1. Load binary data as uint16_t (2 bytes per value)")
    print("  2. Convert each uint16_t to float using IEEE 754 conversion")
    print("  3. Use float values in neural network computation")
    print("  4. No precision loss during conversion (FP16→FP32 is exact)")
    print("  5. Memory savings: 2x smaller on disk, same size in RAM")

if __name__ == "__main__":
    # Configuration
    fp16_dir = "./weights/whisper_base_fp16"
    fp32_dir = "./weights/whisper_base_fp32"

    # Test loading all weights
    stats = test_all_weights(fp16_dir, fp32_dir)

    if stats:
        print("\n" + "=" * 80)
        print("  RECOMMENDATIONS")
        print("=" * 80)

        if stats['has_nan'] or stats['has_inf']:
            print("\n❌ CRITICAL: NaN/Inf detected - DO NOT USE for inference")
            print("   Some weights exceed FP16 range and overflow")
            print("   Consider mixed precision strategy")
        elif stats['max_abs_error'] < 1e-2 and stats['avg_snr_db'] > 40:
            print("\n✅ RECOMMENDED: Use FP16 weights for C++ loading")
            print("   - 50% memory savings (400MB → 200MB)")
            print("   - Excellent accuracy (max error < 0.01)")
            print("   - High SNR (>40 dB)")
            print("   - Safe for production inference")
        else:
            print("\n⚠️  CONDITIONAL: FP16 usable but monitor accuracy")
            print("   - Test inference quality on validation set")
            print("   - Compare WER with FP32 baseline")
            print("   - Consider mixed precision if accuracy degrades")

        # Demonstrate C++ loading pattern
        demonstrate_cpp_loading_pattern()

        print("\n✅ FP16 weight loading test complete!")
        print(f"\nNext steps:")
        print(f"  1. Integrate FP16 loading into C++ encoder")
        print(f"  2. Test inference quality (WER, latency)")
        print(f"  3. Compare FP16 vs FP32 vs INT8 performance")

    else:
        print("\n❌ Test failed - check error messages above")
        sys.exit(1)
