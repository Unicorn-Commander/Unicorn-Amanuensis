#!/usr/bin/env python3
"""
PyTorch Reference Implementation for Whisper Encoder

This module provides a golden reference implementation using the official
PyTorch Whisper model for validation of the C++ NPU implementation.

Features:
- Load official Whisper models from HuggingFace
- Extract weights for C++ comparison
- Full encoder forward pass
- Layer-by-layer forward pass (for debugging)
- Export reference outputs as .npy files

Usage:
    from pytorch_reference import WhisperEncoderReference

    # Create reference
    ref = WhisperEncoderReference("openai/whisper-base")

    # Extract weights
    weights = ref.extract_weights()

    # Encode mel spectrogram
    mel = np.random.randn(1, 80, 3000).astype(np.float32)
    embeddings = ref.encode(mel)

    # Layer-by-layer encoding
    layer_output = ref.encode_layer(0, input_embeddings)
"""

import numpy as np
import torch
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WhisperEncoderReference:
    """
    PyTorch reference implementation of Whisper encoder

    This class provides a golden reference for validating the C++ NPU implementation.
    It uses the official HuggingFace Transformers library.
    """

    def __init__(self, model_name: str = "openai/whisper-base"):
        """
        Initialize Whisper encoder reference

        Args:
            model_name: HuggingFace model name (default: openai/whisper-base)
        """
        logger.info(f"Loading Whisper model: {model_name}")

        try:
            from transformers import WhisperModel
        except ImportError:
            logger.error("Transformers library not found. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
            from transformers import WhisperModel

        # Load model
        self.model = WhisperModel.from_pretrained(model_name)
        self.model.eval()  # Inference mode
        self.encoder = self.model.encoder

        # Get configuration
        self.config = self.model.config
        self.n_layers = self.config.encoder_layers
        self.n_heads = self.config.encoder_attention_heads
        self.n_state = self.config.d_model
        self.ffn_dim = self.config.encoder_ffn_dim

        logger.info(f"Model loaded successfully:")
        logger.info(f"  Layers: {self.n_layers}")
        logger.info(f"  Attention Heads: {self.n_heads}")
        logger.info(f"  Hidden Size: {self.n_state}")
        logger.info(f"  FFN Dim: {self.ffn_dim}")

    def extract_weights(self) -> Dict[str, np.ndarray]:
        """
        Extract all encoder weights for C++ comparison

        Returns:
            dict: Dictionary of weight tensors (layer_name -> numpy array)
        """
        logger.info("Extracting encoder weights...")

        weights = {}
        for name, param in self.encoder.named_parameters():
            weights[name] = param.detach().cpu().numpy()

        logger.info(f"Extracted {len(weights)} weight tensors")
        return weights

    def save_weights(self, output_dir: str = "./weights/whisper_base_fp32") -> None:
        """
        Save weights to .npy files for C++ loading

        Args:
            output_dir: Directory to save weights
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving weights to: {output_path}")

        weights = self.extract_weights()
        for name, tensor in weights.items():
            # Replace '.' with '_' for filename
            filename = name.replace('.', '_') + '.npy'
            filepath = output_path / filename
            np.save(filepath, tensor)
            logger.debug(f"  Saved: {filename} ({tensor.shape})")

        logger.info(f"Saved {len(weights)} weight files")

    def encode(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """
        Encode mel spectrogram to embeddings (full encoder)

        Args:
            mel_spectrogram: Input mel spectrogram
                Shape: (batch, 80, 3000) - Whisper mel input
                   or: (batch, 1500, 512) - Post-convolution embeddings

        Returns:
            embeddings: Encoder output embeddings
                Shape: (batch, 1500, 512)
        """
        with torch.no_grad():
            mel_tensor = torch.from_numpy(mel_spectrogram).float()

            # Determine input shape
            if mel_tensor.shape[1] == 80:
                # Raw mel spectrogram (batch, 80, 3000)
                # Use full encoder (with conv layers)
                output = self.encoder(mel_tensor)
                embeddings = output.last_hidden_state.numpy()
            else:
                # Post-convolution embeddings (batch, 1500, 512)
                # Bypass conv layers, run through transformer layers only
                embeddings = self._encode_transformer_only(mel_tensor).numpy()

            return embeddings

    def _encode_transformer_only(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode using transformer layers only (bypass conv layers)

        This is used for comparing C++ implementation which only implements
        the transformer layers (not the conv1d layers).

        Args:
            input_embeddings: Input embeddings (batch, seq_len, hidden_dim)

        Returns:
            output_embeddings: Output embeddings (batch, seq_len, hidden_dim)
        """
        hidden_states = input_embeddings

        # Apply layer norm (if exists)
        if hasattr(self.encoder, 'layer_norm') and self.encoder.layer_norm is not None:
            hidden_states = self.encoder.layer_norm(hidden_states)

        # Run through all encoder layers
        for layer in self.encoder.layers:
            layer_output = layer(
                hidden_states,
                attention_mask=None,
                layer_head_mask=None,
                output_attentions=False
            )
            hidden_states = layer_output[0]  # Extract hidden states

        return hidden_states

    def encode_layer(self, layer_idx: int, input_embeddings: np.ndarray) -> np.ndarray:
        """
        Run single encoder layer (for detailed comparison)

        Args:
            layer_idx: Layer index (0-5 for Whisper Base)
            input_embeddings: Input to this layer
                Shape: (batch, seq_len, hidden_dim)

        Returns:
            output_embeddings: Output of this layer
                Shape: (batch, seq_len, hidden_dim)
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {self.n_layers})")

        layer = self.encoder.layers[layer_idx]

        with torch.no_grad():
            input_tensor = torch.from_numpy(input_embeddings).float()
            output = layer(input_tensor, attention_mask=None, layer_head_mask=None, output_attentions=False)
            return output[0].numpy()

    def encode_all_layers_separately(
        self,
        input_embeddings: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Run all encoder layers separately and return intermediate outputs

        This is useful for debugging layer-by-layer accuracy.

        Args:
            input_embeddings: Initial input (post-convolution)
                Shape: (batch, seq_len, hidden_dim)

        Returns:
            outputs: Dictionary of layer outputs (layer_idx -> output)
        """
        outputs = {}
        current = input_embeddings

        for i in range(self.n_layers):
            output = self.encode_layer(i, current)
            outputs[i] = output
            current = output  # Feed to next layer

        return outputs

    def compute_accuracy_metrics(
        self,
        reference: np.ndarray,
        candidate: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute accuracy metrics between reference and candidate

        Args:
            reference: Reference output (ground truth)
            candidate: Candidate output (to validate)

        Returns:
            metrics: Dictionary of accuracy metrics
        """
        # Flatten arrays
        ref_flat = reference.flatten()
        cand_flat = candidate.flatten()

        # Cosine similarity
        dot_product = np.dot(ref_flat, cand_flat)
        norm_ref = np.linalg.norm(ref_flat)
        norm_cand = np.linalg.norm(cand_flat)
        cosine_sim = dot_product / (norm_ref * norm_cand) if norm_ref > 0 and norm_cand > 0 else 1.0

        # Absolute errors
        abs_diff = np.abs(reference - candidate)
        mae = abs_diff.mean()
        max_abs_error = abs_diff.max()

        # Relative errors
        epsilon = 1e-8
        rel_error = abs_diff / (np.abs(reference) + epsilon)
        mean_rel_error = rel_error.mean()
        max_rel_error = rel_error.max()

        # Element-wise accuracy (< 1% error)
        accurate_elements = np.sum(rel_error < 0.01)
        total_elements = rel_error.size
        accuracy_pct = (accurate_elements / total_elements) * 100

        return {
            'cosine_similarity': float(cosine_sim),
            'mae': float(mae),
            'max_abs_error': float(max_abs_error),
            'mean_rel_error': float(mean_rel_error),
            'max_rel_error': float(max_rel_error),
            'accuracy_pct': float(accuracy_pct),
        }

    def export_test_vectors(
        self,
        output_dir: str = "./tests/regression_database/test_vectors",
        num_vectors: int = 10,
        seed: int = 42
    ) -> None:
        """
        Export test vectors for regression testing

        Args:
            output_dir: Directory to save test vectors
            num_vectors: Number of test vectors to generate
            seed: Random seed for reproducibility
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        np.random.seed(seed)

        logger.info(f"Generating {num_vectors} test vectors...")

        for i in range(num_vectors):
            # Generate random input (batch=1, seq_len=1500, hidden_dim=512)
            # Note: Use 1500 (actual Whisper sequence length after conv)
            input_emb = np.random.randn(1, 1500, self.n_state).astype(np.float32)

            # Encode with PyTorch
            output_emb = self.encode(input_emb)

            # Save input and output
            np.save(output_path / f"input_{i:03d}.npy", input_emb)
            np.save(output_path / f"output_pytorch_{i:03d}.npy", output_emb)

            logger.debug(f"  Saved test vector {i:03d}")

        logger.info(f"Test vectors saved to: {output_path}")


def main():
    """Demo usage of WhisperEncoderReference"""
    print("="*80)
    print("  WHISPER ENCODER PYTORCH REFERENCE")
    print("="*80)

    # Create reference
    ref = WhisperEncoderReference("openai/whisper-base")

    # Generate test input (post-convolution shape)
    print("\n[1] Generating test input...")
    np.random.seed(42)
    input_emb = np.random.randn(1, 1500, 512).astype(np.float32)
    print(f"  Input shape: {input_emb.shape}")

    # Full encoder
    print("\n[2] Running full encoder (6 layers)...")
    output = ref.encode(input_emb)
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean():.4f}")
    print(f"  Output std: {output.std():.4f}")

    # Layer-by-layer
    print("\n[3] Running layer-by-layer...")
    layer_outputs = ref.encode_all_layers_separately(input_emb)
    for i, layer_out in layer_outputs.items():
        print(f"  Layer {i}: mean={layer_out.mean():.4f}, std={layer_out.std():.4f}")

    # Save weights
    print("\n[4] Saving weights...")
    ref.save_weights("./weights/whisper_base_fp32")
    print("  Weights saved to: ./weights/whisper_base_fp32/")

    # Export test vectors
    print("\n[5] Exporting test vectors...")
    ref.export_test_vectors(num_vectors=5)
    print("  Test vectors saved to: ./tests/regression_database/test_vectors/")

    print("\n" + "="*80)
    print("  PYTORCH REFERENCE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
