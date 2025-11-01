"""
Whisper Conv1d Embedding Layer

Implements the conv1d preprocessing that converts mel spectrograms (80-dim)
to encoder embeddings (512-dim) before passing to the encoder layers.

This is the missing layer that caused Bug #5: dimension mismatch (80 vs 512).

Architecture:
    Mel Spectrogram (n_frames, 80)
    → Conv1d (80→512, kernel=3, stride=1, padding=1) + GELU
    → Conv2d (512→512, kernel=3, stride=2, padding=1) + GELU
    → Embeddings (n_frames//2, 512)

Author: Week 13 Encoder Conv1d Fix Team
Date: November 1, 2025
Status: Bug #5 Fix
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union


def apply_whisper_conv1d(
    mel_spectrogram: Union[np.ndarray, torch.Tensor],
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    conv2_bias: torch.Tensor
) -> np.ndarray:
    """
    Apply Whisper's conv1d embedding layers to mel spectrogram.

    This converts the 80-dimensional mel spectrogram to 512-dimensional
    embeddings that the encoder expects.

    Args:
        mel_spectrogram: Mel spectrogram with shape (n_frames, 80) or (80, n_frames)
                        Can be numpy array or torch tensor
        conv1_weight: First conv layer weight (512, 80, 3)
        conv1_bias: First conv layer bias (512,)
        conv2_weight: Second conv layer weight (512, 512, 3)
        conv2_bias: Second conv layer bias (512,)

    Returns:
        Embeddings with shape (n_frames//2, 512) as numpy array

    Notes:
        - Input mel can be (n_frames, 80) or (80, n_frames), will auto-detect
        - Output is always (n_frames//2, 512) due to stride=2 in conv2
        - Result is returned on CPU as numpy array for C++ encoder

    Example:
        >>> mel = np.random.randn(100, 80)  # 100 frames
        >>> embeddings = apply_whisper_conv1d(mel, conv1_w, conv1_b, conv2_w, conv2_b)
        >>> embeddings.shape
        (50, 512)  # 50 frames due to stride=2
    """
    # Convert to torch tensor if needed
    if isinstance(mel_spectrogram, np.ndarray):
        x = torch.from_numpy(mel_spectrogram).float()
    else:
        x = mel_spectrogram.float()

    # Detect and fix shape: we need (n_frames, 80)
    # If shape is (80, n_frames), transpose
    if x.shape[-1] != 80:
        if x.shape[0] == 80:
            x = x.t()
        else:
            raise ValueError(
                f"Invalid mel spectrogram shape: {x.shape}. "
                f"Expected (n_frames, 80) or (80, n_frames)"
            )

    # Now x is (n_frames, 80)
    # Transpose for conv1d: (batch=1, channels=80, time=n_frames)
    x = x.t().unsqueeze(0)  # (1, 80, n_frames)

    # First conv layer: 80 → 512
    x = F.conv1d(x, conv1_weight, conv1_bias, stride=1, padding=1)
    x = F.gelu(x)

    # Second conv layer: 512 → 512, stride=2
    x = F.conv1d(x, conv2_weight, conv2_bias, stride=2, padding=1)
    x = F.gelu(x)

    # Transpose back: (1, 512, n_frames//2) → (n_frames//2, 512)
    x = x.squeeze(0).t()

    # Convert to numpy for C++ encoder
    result = x.detach().cpu().numpy()

    # Pad to multiple of 8 (required by BFP16 encoder)
    n_frames = result.shape[0]
    if n_frames % 8 != 0:
        pad_frames = 8 - (n_frames % 8)
        result = np.pad(result, ((0, pad_frames), (0, 0)), mode='constant', constant_values=0)

    return result


class WhisperConv1dPreprocessor:
    """
    Stateful wrapper for Whisper conv1d preprocessing.

    Loads the conv1d weights from a Whisper model once at initialization,
    then applies them efficiently to mel spectrograms.

    Usage:
        # Initialize once
        from transformers import WhisperModel
        model = WhisperModel.from_pretrained('openai/whisper-base')
        preprocessor = WhisperConv1dPreprocessor(model)

        # Use many times
        for mel in mel_spectrograms:
            embeddings = preprocessor.process(mel)
            encoder_output = encoder.forward(embeddings)
    """

    def __init__(self, whisper_model):
        """
        Initialize preprocessor from Whisper model.

        Args:
            whisper_model: Whisper model from transformers or whisperx
                          (must have encoder.conv1 and encoder.conv2)
        """
        # Extract conv weights
        self.conv1_weight = whisper_model.encoder.conv1.weight.data
        self.conv1_bias = whisper_model.encoder.conv1.bias.data
        self.conv2_weight = whisper_model.encoder.conv2.weight.data
        self.conv2_bias = whisper_model.encoder.conv2.bias.data

        # Validate shapes
        assert self.conv1_weight.shape == (512, 80, 3), \
            f"Conv1 weight shape mismatch: {self.conv1_weight.shape}"
        assert self.conv2_weight.shape == (512, 512, 3), \
            f"Conv2 weight shape mismatch: {self.conv2_weight.shape}"

    def process(self, mel_spectrogram: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Apply conv1d preprocessing to mel spectrogram.

        Args:
            mel_spectrogram: Mel spectrogram (n_frames, 80) or (80, n_frames)

        Returns:
            Embeddings (n_frames//2, 512) as numpy array
        """
        return apply_whisper_conv1d(
            mel_spectrogram,
            self.conv1_weight,
            self.conv1_bias,
            self.conv2_weight,
            self.conv2_bias
        )


def test_conv1d_preprocessing():
    """Test conv1d preprocessing with a real Whisper model"""
    print("\n=== Testing Whisper Conv1d Preprocessing ===\n")

    # Load Whisper model
    print("Loading Whisper model...")
    from transformers import WhisperModel
    model = WhisperModel.from_pretrained('openai/whisper-base')
    print("✓ Model loaded\n")

    # Create preprocessor
    preprocessor = WhisperConv1dPreprocessor(model)
    print("✓ Preprocessor initialized\n")

    # Test with random mel spectrogram
    print("Testing with random mel spectrogram...")
    mel = np.random.randn(100, 80).astype(np.float32)  # 100 frames
    print(f"  Input shape: {mel.shape}")

    embeddings = preprocessor.process(mel)
    print(f"  Output shape: {embeddings.shape}")
    print(f"  Expected shape: (50, 512) due to stride=2")

    assert embeddings.shape == (50, 512), f"Shape mismatch: {embeddings.shape}"
    assert embeddings.dtype == np.float32, f"Dtype mismatch: {embeddings.dtype}"
    print("✓ Test passed\n")

    # Test with transposed input
    print("Testing with transposed input (80, n_frames)...")
    mel_t = mel.T  # (80, 100)
    embeddings_t = preprocessor.process(mel_t)
    print(f"  Input shape: {mel_t.shape}")
    print(f"  Output shape: {embeddings_t.shape}")
    assert embeddings_t.shape == (50, 512), f"Shape mismatch: {embeddings_t.shape}"
    print("✓ Transpose handling works\n")

    # Test with torch tensor
    print("Testing with torch tensor...")
    mel_torch = torch.from_numpy(mel)
    embeddings_torch = preprocessor.process(mel_torch)
    assert embeddings_torch.shape == (50, 512), f"Shape mismatch: {embeddings_torch.shape}"
    print("✓ Torch tensor handling works\n")

    print("=== All Tests Passed ===\n")


if __name__ == "__main__":
    test_conv1d_preprocessing()
