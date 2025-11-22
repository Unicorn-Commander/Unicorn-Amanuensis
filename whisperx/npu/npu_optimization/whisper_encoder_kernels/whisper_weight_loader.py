#!/usr/bin/env python3
"""
Whisper Weight Loader for AMD Phoenix NPU

Loads Whisper encoder weights from ONNX model and prepares them for NPU execution.
Converts to BF16 format and organizes by layer for efficient kernel binding.

Usage:
    loader = WhisperWeightLoader(onnx_path)
    weights = loader.get_layer_weights(0)  # Get layer 0 weights
    bf16_data = loader.to_bf16()  # Convert all to BF16 bytes
    loader.summary()  # Print weight shapes and sizes
"""

import onnx
import numpy as np
import struct
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class LayerWeightInfo:
    """Container for per-layer weight information"""
    layer_idx: int
    attention_params: int
    ffn_params: int
    layernorm_params: int
    total_params: int
    fp32_size_bytes: int
    bf16_size_bytes: int


class WhisperWeightLoader:
    """Load Whisper encoder weights from ONNX for NPU execution"""

    def __init__(self, onnx_path: str):
        """
        Initialize the weight loader.

        Args:
            onnx_path: Path to the ONNX encoder model
        """
        self.onnx_path = Path(onnx_path)
        self.model = None
        self.weights: Dict[str, np.ndarray] = {}
        self.bf16_weights: Dict[str, bytes] = {}
        self._loaded = False
        self._converted = False
        self.num_layers = 0

        # Load the model immediately
        self._load_model()

    def _load_model(self) -> None:
        """Load and parse ONNX model"""
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        print(f"Loading ONNX model from: {self.onnx_path}")
        self.model = onnx.load(str(self.onnx_path))

        # Extract all weights from initializers
        for init in self.model.graph.initializer:
            name = init.name

            # Determine dtype based on ONNX data type
            if init.data_type == onnx.TensorProto.FLOAT:
                dtype = np.float32
            elif init.data_type == onnx.TensorProto.FLOAT16:
                dtype = np.float16
            elif init.data_type == onnx.TensorProto.INT8:
                dtype = np.int8
            elif init.data_type == onnx.TensorProto.INT64:
                dtype = np.int64
            else:
                # Skip unsupported types
                continue

            # Load raw data and reshape
            if len(init.raw_data) > 0:
                data = np.frombuffer(init.raw_data, dtype=dtype)
            else:
                # Handle alternative storage format (for small tensors)
                if dtype == np.float32:
                    data = np.array(init.float_data, dtype=dtype)
                elif dtype == np.int64:
                    data = np.array(init.int64_data, dtype=dtype)
                else:
                    continue

            shape = tuple(init.dims)
            if len(shape) > 0:
                self.weights[name] = data.reshape(shape)
            else:
                # Scalar value
                self.weights[name] = data

        # Determine number of encoder layers
        self.num_layers = self._detect_num_layers()
        self._loaded = True

        print(f"Loaded {len(self.weights)} weight tensors")
        print(f"Detected {self.num_layers} encoder layers")

    def _detect_num_layers(self) -> int:
        """Detect number of encoder layers from weight names"""
        max_layer = -1
        for name in self.weights.keys():
            # Handle both naming conventions:
            # 1. /encoder/layers.0/... (from research docs)
            # 2. layers.0.self_attn... (actual ONNX export)
            if 'layers.' in name:
                try:
                    # Extract layer index
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part == 'layers' and i + 1 < len(parts):
                            # Get the layer number (could be followed by / or another .)
                            layer_str = parts[i + 1].split('/')[0]
                            layer_idx = int(layer_str)
                            max_layer = max(max_layer, layer_idx)
                            break
                except (ValueError, IndexError):
                    continue
        return max_layer + 1 if max_layer >= 0 else 0

    def get_layer_weights(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """
        Get organized weights for a specific encoder layer.

        Args:
            layer_idx: Layer index (0 to num_layers-1)

        Returns:
            Dictionary with keys: q_proj, k_proj, v_proj, o_proj, fc1, fc2,
            self_attn_layer_norm, final_layer_norm (each with weight and bias)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")

        layer_weights = {}

        # Handle both naming conventions
        # Pattern 1: /encoder/layers.0/self_attn/q_proj/weight
        # Pattern 2: layers.0.self_attn.q_proj.bias (for biases)
        # Pattern 3: onnx::MatMul_XXX (for weights - need to map by position)

        # First, collect biases using the layers.X pattern
        prefix_dot = f'layers.{layer_idx}.'

        for name, weight in self.weights.items():
            if prefix_dot not in name:
                continue

            # Extract the component name after the layer prefix
            # e.g., "layers.0.self_attn.v_proj.bias" -> "self_attn.v_proj.bias"
            suffix = name[name.index(prefix_dot) + len(prefix_dot):]

            # Map attention biases
            if 'self_attn.q_proj.bias' in suffix:
                layer_weights['q_proj_bias'] = weight
            elif 'self_attn.k_proj.bias' in suffix:
                layer_weights['k_proj_bias'] = weight
            elif 'self_attn.v_proj.bias' in suffix:
                layer_weights['v_proj_bias'] = weight
            elif 'self_attn.out_proj.bias' in suffix:
                layer_weights['o_proj_bias'] = weight
            # Map FFN biases
            elif 'fc1.bias' in suffix:
                layer_weights['fc1_bias'] = weight
            elif 'fc2.bias' in suffix:
                layer_weights['fc2_bias'] = weight
            # Map LayerNorm
            elif 'self_attn_layer_norm.weight' in suffix:
                layer_weights['self_attn_layer_norm_weight'] = weight
            elif 'self_attn_layer_norm.bias' in suffix:
                layer_weights['self_attn_layer_norm_bias'] = weight
            elif 'final_layer_norm.weight' in suffix:
                layer_weights['final_layer_norm_weight'] = weight
            elif 'final_layer_norm.bias' in suffix:
                layer_weights['final_layer_norm_bias'] = weight

        # For ONNX Community models, weights are stored as onnx::MatMul_XXX
        # We need to map them based on their position and shape
        # Analysis of shapes in the model:
        # - Attention weights (q, k, v, out): [512, 512]
        # - FC1: [512, 2048]
        # - FC2: [2048, 512]

        matmul_weights = []
        for name in sorted(self.weights.keys()):
            if name.startswith('onnx::MatMul_'):
                matmul_weights.append((name, self.weights[name]))

        # Determine number of MatMuls per layer by analyzing pattern
        # Expected: 6 per layer (q, k, v, out, fc1, fc2)
        total_matmuls = len(matmul_weights)
        matmuls_per_layer = total_matmuls // self.num_layers if self.num_layers > 0 else 6

        if matmul_weights:
            layer_start = layer_idx * matmuls_per_layer
            layer_end = layer_start + matmuls_per_layer

            if layer_end <= len(matmul_weights):
                layer_matmuls = matmul_weights[layer_start:layer_end]

                # Separate by shape for correct mapping
                small_weights = []  # [512, 512] - attention projections
                fc1_weight = None   # [512, 2048]
                fc2_weight = None   # [2048, 512]

                for name, weight in layer_matmuls:
                    shape = weight.shape
                    if len(shape) == 2:
                        if shape == (512, 512):
                            small_weights.append(weight)
                        elif shape == (512, 2048):
                            fc1_weight = weight
                        elif shape == (2048, 512):
                            fc2_weight = weight

                # Map attention weights in order (q, k, v, out or similar pattern)
                if len(small_weights) >= 4:
                    layer_weights['q_proj_weight'] = small_weights[0]
                    layer_weights['k_proj_weight'] = small_weights[1]
                    layer_weights['v_proj_weight'] = small_weights[2]
                    layer_weights['o_proj_weight'] = small_weights[3]
                elif len(small_weights) == 3:
                    # Some models fuse Q and K
                    layer_weights['q_proj_weight'] = small_weights[0]
                    layer_weights['v_proj_weight'] = small_weights[1]
                    layer_weights['o_proj_weight'] = small_weights[2]

                # Map FFN weights
                if fc1_weight is not None:
                    layer_weights['fc1_weight'] = fc1_weight
                if fc2_weight is not None:
                    layer_weights['fc2_weight'] = fc2_weight

        return layer_weights

    def _fp32_to_bf16_bytes(self, fp32_array: np.ndarray) -> bytes:
        """
        Convert FP32 array to BF16 bytes.

        BF16 format keeps the same exponent as FP32 but truncates mantissa
        from 23 bits to 7 bits. This is done by taking the upper 16 bits
        of each 32-bit float.

        Args:
            fp32_array: Input FP32 numpy array

        Returns:
            bytes containing BF16 data
        """
        # Ensure input is float32
        if fp32_array.dtype != np.float32:
            fp32_array = fp32_array.astype(np.float32)

        # Get raw bytes and view as uint32
        flat = fp32_array.flatten()
        fp32_bytes = flat.tobytes()
        uint32_view = np.frombuffer(fp32_bytes, dtype=np.uint32)

        # Right shift by 16 to get BF16 (upper 16 bits of FP32)
        bf16_uint = (uint32_view >> 16).astype(np.uint16)

        return bf16_uint.tobytes()

    def to_bf16(self) -> Dict[str, bytes]:
        """
        Convert all weights to BF16 format bytes.

        Returns:
            Dictionary mapping weight names to BF16 byte data
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        print("Converting weights to BF16 format...")

        self.bf16_weights = {}
        for name, weight in self.weights.items():
            # Skip non-float weights (like positions, etc.)
            if weight.dtype not in [np.float32, np.float16]:
                continue

            # Convert float16 to float32 first if needed
            if weight.dtype == np.float16:
                weight = weight.astype(np.float32)

            self.bf16_weights[name] = self._fp32_to_bf16_bytes(weight)

        self._converted = True
        print(f"Converted {len(self.bf16_weights)} weight tensors to BF16")

        return self.bf16_weights

    def get_bf16_layer_weights(self, layer_idx: int) -> Dict[str, bytes]:
        """
        Get BF16 weights for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Dictionary with BF16 byte data for each weight
        """
        if not self._converted:
            self.to_bf16()

        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range")

        # Get the original layer weights first
        layer_weights = self.get_layer_weights(layer_idx)

        # Convert to BF16
        layer_bf16 = {}
        for key, weight in layer_weights.items():
            layer_bf16[key] = self._fp32_to_bf16_bytes(weight)

        return layer_bf16

    def get_global_weights(self) -> Dict[str, np.ndarray]:
        """
        Get global (non-layer-specific) weights.

        Returns:
            Dictionary with conv layers, positional embeddings, etc.
        """
        global_weights = {}

        for name, weight in self.weights.items():
            # Skip layer-specific weights
            if '/encoder/layers.' in name:
                continue

            global_weights[name] = weight

        return global_weights

    def summary(self) -> None:
        """Print weight shapes and total size information"""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        print("\n" + "=" * 70)
        print("WHISPER ENCODER WEIGHT SUMMARY")
        print("=" * 70)

        print(f"\nModel: {self.onnx_path.name}")
        print(f"Total weight tensors: {len(self.weights)}")
        print(f"Encoder layers: {self.num_layers}")

        # Calculate total sizes
        total_params = 0
        total_fp32_bytes = 0

        # Global weights
        print("\n" + "-" * 70)
        print("GLOBAL WEIGHTS (Conv, Embeddings, etc.)")
        print("-" * 70)

        global_params = 0
        for name, weight in self.weights.items():
            if '/encoder/layers.' not in name:
                params = weight.size
                global_params += params
                total_params += params
                size_kb = weight.nbytes / 1024
                total_fp32_bytes += weight.nbytes
                print(f"  {name}")
                print(f"    Shape: {weight.shape}, Dtype: {weight.dtype}, Size: {size_kb:.1f} KB")

        print(f"\n  Global total: {global_params:,} params ({global_params * 4 / 1024 / 1024:.2f} MB FP32)")

        # Per-layer weights
        print("\n" + "-" * 70)
        print("PER-LAYER WEIGHTS")
        print("-" * 70)

        layer_info = []
        for layer_idx in range(self.num_layers):
            layer_weights = self.get_layer_weights(layer_idx)

            attn_params = 0
            ffn_params = 0
            ln_params = 0

            print(f"\nLayer {layer_idx}:")

            for name, weight in layer_weights.items():
                params = weight.size
                total_params += params
                total_fp32_bytes += weight.nbytes

                if any(p in name for p in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                    attn_params += params
                elif 'fc' in name:
                    ffn_params += params
                elif 'layer_norm' in name:
                    ln_params += params

                print(f"  {name}: {weight.shape} ({params:,} params)")

            layer_total = attn_params + ffn_params + ln_params
            layer_info.append(LayerWeightInfo(
                layer_idx=layer_idx,
                attention_params=attn_params,
                ffn_params=ffn_params,
                layernorm_params=ln_params,
                total_params=layer_total,
                fp32_size_bytes=layer_total * 4,
                bf16_size_bytes=layer_total * 2
            ))

            print(f"  Layer {layer_idx} subtotal: {layer_total:,} params")
            print(f"    Attention: {attn_params:,}, FFN: {ffn_params:,}, LayerNorm: {ln_params:,}")

        # Summary statistics
        print("\n" + "-" * 70)
        print("TOTAL SIZES")
        print("-" * 70)

        fp32_mb = total_fp32_bytes / 1024 / 1024
        bf16_mb = total_params * 2 / 1024 / 1024
        int8_mb = total_params / 1024 / 1024

        print(f"\nTotal parameters: {total_params:,}")
        print(f"\nFormat sizes:")
        print(f"  FP32: {fp32_mb:.2f} MB")
        print(f"  BF16: {bf16_mb:.2f} MB (2x compression)")
        print(f"  INT8: {int8_mb:.2f} MB (4x compression)")

        # Per-layer breakdown
        if layer_info:
            avg_layer = sum(l.total_params for l in layer_info) / len(layer_info)
            print(f"\nPer-layer average: {avg_layer:,.0f} params ({avg_layer * 2 / 1024 / 1024:.2f} MB BF16)")

        print("\n" + "=" * 70)

    def get_weight_shapes(self) -> Dict[str, Tuple]:
        """
        Get shapes of all weight tensors.

        Returns:
            Dictionary mapping weight names to their shapes
        """
        return {name: weight.shape for name, weight in self.weights.items()}

    def get_total_size(self, format: str = 'bf16') -> int:
        """
        Get total size of weights in specified format.

        Args:
            format: 'fp32', 'bf16', or 'int8'

        Returns:
            Total size in bytes
        """
        total_params = sum(w.size for w in self.weights.values())

        if format == 'fp32':
            return total_params * 4
        elif format == 'bf16':
            return total_params * 2
        elif format == 'int8':
            return total_params
        else:
            raise ValueError(f"Unknown format: {format}")


def main():
    """Run the weight loader on the actual Whisper base model"""

    # Path to the ONNX encoder model
    onnx_path = (
        "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/"
        "whisper_onnx_cache/models--onnx-community--whisper-base/"
        "onnx/encoder_model.onnx"
    )

    print("\n" + "=" * 70)
    print("WHISPER WEIGHT LOADER - NPU Optimization")
    print("=" * 70)

    # Check if model exists
    if not Path(onnx_path).exists():
        print(f"\nERROR: ONNX model not found at: {onnx_path}")
        print("Please ensure the Whisper ONNX model is downloaded.")
        return

    # Initialize loader
    try:
        loader = WhisperWeightLoader(onnx_path)
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        return

    # Print summary
    loader.summary()

    # Test getting layer weights
    print("\n" + "=" * 70)
    print("LAYER WEIGHT DETAILS")
    print("=" * 70)

    for layer_idx in range(min(2, loader.num_layers)):  # Show first 2 layers
        print(f"\nLayer {layer_idx} weights:")
        layer_weights = loader.get_layer_weights(layer_idx)
        for name, weight in sorted(layer_weights.items()):
            print(f"  {name}: shape={weight.shape}, dtype={weight.dtype}")

    # Convert to BF16
    print("\n" + "=" * 70)
    print("BF16 CONVERSION")
    print("=" * 70)

    bf16_weights = loader.to_bf16()

    # Report BF16 sizes
    total_bf16_bytes = sum(len(data) for data in bf16_weights.values())
    print(f"\nTotal BF16 size: {total_bf16_bytes / 1024 / 1024:.2f} MB")

    # Show sample BF16 data for first layer
    print("\nBF16 weight sizes for layer 0:")
    layer0_bf16 = loader.get_bf16_layer_weights(0)
    for name, data in sorted(layer0_bf16.items()):
        print(f"  {name}: {len(data)} bytes ({len(data) / 1024:.1f} KB)")

    # Global weights
    print("\n" + "=" * 70)
    print("GLOBAL WEIGHTS")
    print("=" * 70)

    global_weights = loader.get_global_weights()
    print(f"\nFound {len(global_weights)} global weight tensors:")
    for name, weight in sorted(global_weights.items()):
        print(f"  {name}: shape={weight.shape}, dtype={weight.dtype}")

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    print(f"\nNumber of layers found: {loader.num_layers}")
    print(f"Total weight tensors: {len(loader.weights)}")
    print(f"Total FP32 size: {loader.get_total_size('fp32') / 1024 / 1024:.2f} MB")
    print(f"Total BF16 size: {loader.get_total_size('bf16') / 1024 / 1024:.2f} MB")
    print(f"Total INT8 size: {loader.get_total_size('int8') / 1024 / 1024:.2f} MB")

    # Check for any issues
    print("\n" + "=" * 70)
    print("ISSUES / NOTES")
    print("=" * 70)

    issues = []

    # Check for expected layer count
    if loader.num_layers != 6:
        issues.append(f"Expected 6 encoder layers, found {loader.num_layers}")

    # Check for missing weights in each layer
    expected_weights = [
        'q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'o_proj_weight',
        'q_proj_bias', 'k_proj_bias', 'v_proj_bias', 'o_proj_bias',
        'fc1_weight', 'fc2_weight', 'fc1_bias', 'fc2_bias',
        'self_attn_layer_norm_weight', 'self_attn_layer_norm_bias',
        'final_layer_norm_weight', 'final_layer_norm_bias'
    ]

    for layer_idx in range(loader.num_layers):
        layer_weights = loader.get_layer_weights(layer_idx)
        for expected in expected_weights:
            if expected not in layer_weights:
                issues.append(f"Layer {layer_idx} missing: {expected}")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo issues found. All expected weights present.")

    print("\n" + "=" * 70)
    print("WEIGHT LOADING COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
