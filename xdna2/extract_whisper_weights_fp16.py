#!/usr/bin/env python3
"""
Extract Whisper Base Encoder Weights in FP16 Format

Converts PyTorch Whisper Base encoder weights to NumPy FP16 format for C++ loading.
FP16 (float16) provides 2x memory savings with minimal precision loss for neural networks.

FP16 Characteristics:
- Range: ±65,504 (vs FP32: ±3.4×10^38)
- Precision: ~3-4 decimal digits (vs FP32: 6-7 digits)
- Memory: 2 bytes per value (vs FP32: 4 bytes)
- Total Size: ~200MB (vs FP32: ~400MB, INT8: ~100MB)
"""

import torch
import numpy as np
from pathlib import Path
import sys

def check_fp16_safety(weights):
    """
    Check if weights are safe for FP16 conversion.

    FP16 safe range: [-65,504, +65,504]
    Values outside this range will overflow to infinity.

    Returns:
        dict: Statistics about FP16 safety
    """
    stats = {
        'total_tensors': len(weights),
        'total_values': 0,
        'max_abs_value': 0.0,
        'min_value': float('inf'),
        'max_value': float('-inf'),
        'out_of_range_tensors': [],
        'overflow_count': 0,
        'safe_for_fp16': True
    }

    FP16_MAX = 65504.0

    for key, value in weights.items():
        stats['total_values'] += value.size

        min_val = value.min()
        max_val = value.max()
        max_abs = np.abs(value).max()

        stats['min_value'] = min(stats['min_value'], min_val)
        stats['max_value'] = max(stats['max_value'], max_val)
        stats['max_abs_value'] = max(stats['max_abs_value'], max_abs)

        # Check for overflow
        if max_abs > FP16_MAX:
            stats['safe_for_fp16'] = False
            overflow_values = np.sum(np.abs(value) > FP16_MAX)
            stats['overflow_count'] += overflow_values
            stats['out_of_range_tensors'].append({
                'name': key,
                'shape': value.shape,
                'max_abs': max_abs,
                'overflow_count': overflow_values,
                'overflow_percent': 100.0 * overflow_values / value.size
            })

    return stats

def save_fp16_weights(weights, output_dir):
    """
    Save weights as FP16 NumPy arrays.

    C++ will load these as uint16_t arrays and convert to float32 for computation.
    """
    for key, value in weights.items():
        # Convert to FP16
        value_fp16 = value.astype(np.float16)

        # Save as .npy
        filename = output_dir / f"{key.replace('.', '_')}_fp16.npy"
        np.save(filename, value_fp16)

        # Calculate compression stats
        fp32_size = value.nbytes
        fp16_size = value_fp16.nbytes
        compression = 100.0 * (1 - fp16_size / fp32_size)

        print(f"  ✅ {filename.name:60s} {str(value.shape):20s} ({compression:.1f}% smaller)")

def verify_fp16_accuracy(weights_fp32, output_dir):
    """
    Verify FP16 conversion accuracy by loading and comparing.

    Returns:
        dict: Accuracy statistics
    """
    stats = {
        'tensors_checked': 0,
        'max_abs_error': 0.0,
        'max_rel_error': 0.0,
        'avg_abs_error': 0.0,
        'avg_rel_error': 0.0,
        'errors_by_tensor': []
    }

    total_abs_error = 0.0
    total_rel_error = 0.0
    total_values = 0

    for key, value_fp32 in weights_fp32.items():
        # Load FP16 version
        filename = output_dir / f"{key.replace('.', '_')}_fp16.npy"
        value_fp16 = np.load(filename)

        # Convert back to FP32 for comparison
        value_fp16_as_fp32 = value_fp16.astype(np.float32)

        # Calculate errors
        abs_error = np.abs(value_fp32 - value_fp16_as_fp32)
        max_abs_error = abs_error.max()
        avg_abs_error = abs_error.mean()

        # Relative error (avoid division by zero)
        nonzero_mask = np.abs(value_fp32) > 1e-10
        if nonzero_mask.any():
            rel_error = np.abs((value_fp32 - value_fp16_as_fp32) / (value_fp32 + 1e-10))
            max_rel_error = rel_error[nonzero_mask].max()
            avg_rel_error = rel_error[nonzero_mask].mean()
        else:
            max_rel_error = 0.0
            avg_rel_error = 0.0

        # Check for NaN or Inf
        has_nan = np.isnan(value_fp16_as_fp32).any()
        has_inf = np.isinf(value_fp16_as_fp32).any()

        stats['tensors_checked'] += 1
        stats['max_abs_error'] = max(stats['max_abs_error'], max_abs_error)
        stats['max_rel_error'] = max(stats['max_rel_error'], max_rel_error)

        total_abs_error += avg_abs_error * value_fp32.size
        total_rel_error += avg_rel_error * nonzero_mask.sum()
        total_values += value_fp32.size

        stats['errors_by_tensor'].append({
            'name': key,
            'shape': value_fp32.shape,
            'max_abs_error': max_abs_error,
            'avg_abs_error': avg_abs_error,
            'max_rel_error': max_rel_error,
            'avg_rel_error': avg_rel_error,
            'has_nan': has_nan,
            'has_inf': has_inf
        })

    stats['avg_abs_error'] = total_abs_error / total_values
    stats['avg_rel_error'] = total_rel_error / (total_values + 1e-10)

    return stats

def extract_encoder_weights_fp16(model_path, output_dir):
    """
    Extract Whisper encoder weights and save as FP16.

    Args:
        model_path: Path to pytorch_model.bin
        output_dir: Directory to save FP16 weights
    """
    print("=" * 80)
    print("  WHISPER BASE ENCODER WEIGHT EXTRACTION - FP16 FORMAT")
    print("=" * 80)

    # Load PyTorch model
    print(f"\n📥 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    # Filter encoder weights
    encoder_weights = {}
    for key, value in checkpoint.items():
        if key.startswith('model.encoder.'):
            # Remove 'model.encoder.' prefix
            clean_key = key.replace('model.encoder.', '')
            encoder_weights[clean_key] = value.numpy()

    print(f"\n✅ Extracted {len(encoder_weights)} encoder weight tensors")

    # Check FP16 safety
    print("\n" + "=" * 80)
    print("  FP16 SAFETY ANALYSIS")
    print("=" * 80)

    safety_stats = check_fp16_safety(encoder_weights)

    print(f"\nTotal tensors:     {safety_stats['total_tensors']}")
    print(f"Total values:      {safety_stats['total_values']:,}")
    print(f"Value range:       [{safety_stats['min_value']:.6f}, {safety_stats['max_value']:.6f}]")
    print(f"Max absolute:      {safety_stats['max_abs_value']:.6f}")
    print(f"FP16 max:          ±65,504")

    if safety_stats['safe_for_fp16']:
        print("\n✅ ALL WEIGHTS SAFE FOR FP16 CONVERSION")
    else:
        print(f"\n⚠️  WARNING: {len(safety_stats['out_of_range_tensors'])} tensors exceed FP16 range")
        print(f"   {safety_stats['overflow_count']} values will overflow to infinity")
        print("\nOut-of-range tensors:")
        for tensor in safety_stats['out_of_range_tensors']:
            print(f"  - {tensor['name']:50s} max_abs={tensor['max_abs']:.2f} "
                  f"({tensor['overflow_percent']:.2f}% overflow)")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save weights as FP16
    print("\n" + "=" * 80)
    print("  SAVING FP16 WEIGHTS")
    print("=" * 80)
    print(f"\n💾 Saving to: {output_dir}\n")

    save_fp16_weights(encoder_weights, output_path)

    # Verify accuracy
    print("\n" + "=" * 80)
    print("  FP16 ACCURACY VERIFICATION")
    print("=" * 80)

    accuracy_stats = verify_fp16_accuracy(encoder_weights, output_path)

    print(f"\nTensors verified:   {accuracy_stats['tensors_checked']}")
    print(f"Max absolute error: {accuracy_stats['max_abs_error']:.2e}")
    print(f"Avg absolute error: {accuracy_stats['avg_abs_error']:.2e}")
    print(f"Max relative error: {accuracy_stats['max_rel_error']:.2e} ({100*accuracy_stats['max_rel_error']:.4f}%)")
    print(f"Avg relative error: {accuracy_stats['avg_rel_error']:.2e} ({100*accuracy_stats['avg_rel_error']:.4f}%)")

    # Check for NaN/Inf
    nan_tensors = [t for t in accuracy_stats['errors_by_tensor'] if t['has_nan']]
    inf_tensors = [t for t in accuracy_stats['errors_by_tensor'] if t['has_inf']]

    if nan_tensors:
        print(f"\n❌ ERROR: {len(nan_tensors)} tensors contain NaN after conversion")
        for t in nan_tensors:
            print(f"  - {t['name']}")
    elif inf_tensors:
        print(f"\n❌ ERROR: {len(inf_tensors)} tensors contain Inf after conversion")
        for t in inf_tensors:
            print(f"  - {t['name']}")
    else:
        print("\n✅ NO NaN OR Inf DETECTED")

    # Memory savings
    print("\n" + "=" * 80)
    print("  MEMORY SAVINGS")
    print("=" * 80)

    fp32_size = sum(w.nbytes for w in encoder_weights.values())
    fp16_size = fp32_size // 2

    print(f"\nFP32 size:   {fp32_size / 1024 / 1024:.1f} MB")
    print(f"FP16 size:   {fp16_size / 1024 / 1024:.1f} MB")
    print(f"Savings:     {(fp32_size - fp16_size) / 1024 / 1024:.1f} MB (50.0%)")

    print("\n✅ FP16 weight extraction complete!")

    return encoder_weights, safety_stats, accuracy_stats

if __name__ == "__main__":
    # Configuration
    model_cache = Path("./whisper_weights")

    # Find downloaded model
    snapshot_dirs = list(model_cache.glob("models--openai--whisper-base/snapshots/*"))
    if not snapshot_dirs:
        print("❌ Whisper Base model not found. Please download first.")
        sys.exit(1)

    model_path = snapshot_dirs[0] / "pytorch_model.bin"

    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    # Extract FP16 weights
    output_dir = "./weights/whisper_base_fp16"
    weights, safety_stats, accuracy_stats = extract_encoder_weights_fp16(
        model_path, output_dir
    )

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"\nOutput directory:  {output_dir}")
    print(f"Tensors saved:     {len(weights)}")
    print(f"FP16 safe:         {'✅ Yes' if safety_stats['safe_for_fp16'] else '❌ No'}")
    print(f"Max error:         {accuracy_stats['max_abs_error']:.2e}")
    print(f"\nReady to load into C++ encoder!")
    print("\nNext steps:")
    print("  1. Test loading with test_fp16_weight_loading.py")
    print("  2. Compare with FP32 and INT8 formats")
    print("  3. Integrate into C++ encoder (convert FP16→FP32 on load)")
