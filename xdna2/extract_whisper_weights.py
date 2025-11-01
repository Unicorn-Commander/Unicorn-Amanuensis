#!/usr/bin/env python3
"""
Extract Whisper Base Encoder Weights from HuggingFace Model

Converts PyTorch Whisper Base encoder weights to NumPy format for C++ loading.
Supports both FP32 and INT8 quantization.
"""

import torch
import numpy as np
from pathlib import Path
import sys

def extract_encoder_weights(model_path, output_dir, quantize_int8=False):
    """
    Extract Whisper encoder weights from PyTorch checkpoint.

    Args:
        model_path: Path to pytorch_model.bin
        output_dir: Directory to save weights
        quantize_int8: If True, quantize weights to INT8
    """
    print("=" * 70)
    print("  WHISPER BASE ENCODER WEIGHT EXTRACTION")
    print("=" * 70)

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
            print(f"  ✅ {clean_key}: {value.shape}")

    print(f"\n✅ Extracted {len(encoder_weights)} encoder weight tensors")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save weights
    print(f"\n💾 Saving weights to: {output_dir}")

    if quantize_int8:
        print("  🔧 Quantizing to INT8...")
        save_quantized_weights(encoder_weights, output_path)
    else:
        print("  📦 Saving as FP32...")
        save_fp32_weights(encoder_weights, output_path)

    print("\n✅ Weight extraction complete!")
    return encoder_weights

def save_fp32_weights(weights, output_dir):
    """Save weights as FP32 NumPy arrays."""
    for key, value in weights.items():
        filename = output_dir / f"{key.replace('.', '_')}.npy"
        np.save(filename, value)
        print(f"  ✅ Saved: {filename.name}")

def save_quantized_weights(weights, output_dir):
    """Save weights as INT8 with scales."""
    for key, value in weights.items():
        # Quantize to INT8 (symmetric per-tensor)
        max_abs = np.abs(value).max()
        scale = max_abs / 127.0

        quantized = np.round(value / scale).astype(np.int8)

        # Save quantized weights and scale
        base_name = key.replace('.', '_')
        np.save(output_dir / f"{base_name}_int8.npy", quantized)
        np.save(output_dir / f"{base_name}_scale.npy", np.array([scale], dtype=np.float32))

        print(f"  ✅ Quantized: {base_name} (scale={scale:.6f})")

def print_encoder_architecture(weights):
    """Print Whisper encoder architecture from weights."""
    print("\n" + "=" * 70)
    print("  WHISPER BASE ENCODER ARCHITECTURE")
    print("=" * 70)

    # Count layers
    layer_indices = set()
    for key in weights.keys():
        if 'layers.' in key:
            layer_idx = int(key.split('layers.')[1].split('.')[0])
            layer_indices.add(layer_idx)

    n_layers = len(layer_indices)

    # Get dimensions from weights
    embed_dim = weights['embed_positions.weight'].shape[1] if 'embed_positions.weight' in weights else None

    print(f"\nConfiguration:")
    print(f"  Layers:           {n_layers}")
    print(f"  Embedding Dim:    {embed_dim}")

    # Layer structure
    print(f"\nLayer 0 weights:")
    for key in sorted(weights.keys()):
        if 'layers.0.' in key:
            shape = weights[key].shape
            print(f"  {key:60s} {str(shape):20s}")

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

    # Extract FP32 weights
    output_dir_fp32 = "./weights/whisper_base_fp32"
    weights = extract_encoder_weights(model_path, output_dir_fp32, quantize_int8=False)

    # Extract INT8 quantized weights
    output_dir_int8 = "./weights/whisper_base_int8"
    extract_encoder_weights(model_path, output_dir_int8, quantize_int8=True)

    # Print architecture
    print_encoder_architecture(weights)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\nFP32 weights:  {output_dir_fp32}")
    print(f"INT8 weights:  {output_dir_int8}")
    print(f"\nReady to load into C++ encoder!")
