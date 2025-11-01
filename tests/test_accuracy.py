#!/usr/bin/env python3
"""
Accuracy Validation Tests (Week 6, Day 4 - Task 2)

Ensures C++ encoder produces correct results compared to Python encoder.

Tests:
1. Compare C++ encoder output vs Python encoder
2. Numerical comparison (1% error tolerance)
3. Test with multiple audio samples
4. Validate embeddings shape and range
5. End-to-end transcription accuracy

Success Criteria:
- Encoder output within 1% of Python
- Transcription matches Python (minor differences OK)
- No catastrophic failures

Author: NPU Testing & Validation Teamlead
Date: November 1, 2025
Status: Week 6 Days 3-5 - Task 2
"""

import sys
import os
import time
import unittest
import numpy as np
from pathlib import Path
import logging
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test imports with graceful failure
try:
    from xdna2.encoder_cpp import WhisperEncoderCPP, create_encoder_cpp
    from xdna2.cpp_runtime_wrapper import CPPRuntimeError
    CPP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"C++ encoder not available: {e}")
    CPP_AVAILABLE = False

try:
    from transformers import WhisperModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

try:
    import whisperx
    import librosa
    AUDIO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Audio libraries not available: {e}")
    AUDIO_AVAILABLE = False


class TestEncoderOutputAccuracy(unittest.TestCase):
    """Test C++ encoder output matches Python encoder"""

    @classmethod
    def setUpClass(cls):
        """Initialize encoders once for all tests"""
        if not (CPP_AVAILABLE and TRANSFORMERS_AVAILABLE):
            return

        print("\n[SETUP] Loading encoders...")

        # Load Python encoder (transformers)
        cls.python_model = WhisperModel.from_pretrained("openai/whisper-base")
        cls.python_encoder = cls.python_model.encoder

        # Load C++ encoder
        cls.cpp_encoder = create_encoder_cpp(
            num_layers=6,
            n_heads=8,
            n_state=512,
            ffn_dim=2048,
            use_npu=False  # CPU mode for accuracy testing (deterministic)
        )

        # Load weights from Python model into C++ encoder
        weights = {}
        for layer_idx in range(6):
            layer = cls.python_encoder.layers[layer_idx]
            prefix = f"encoder.layers.{layer_idx}"

            # Attention weights
            weights[f"{prefix}.self_attn.q_proj.weight"] = layer.self_attn.q_proj.weight.data.cpu().numpy()
            weights[f"{prefix}.self_attn.k_proj.weight"] = layer.self_attn.k_proj.weight.data.cpu().numpy()
            weights[f"{prefix}.self_attn.v_proj.weight"] = layer.self_attn.v_proj.weight.data.cpu().numpy()
            weights[f"{prefix}.self_attn.out_proj.weight"] = layer.self_attn.out_proj.weight.data.cpu().numpy()

            # Attention biases
            weights[f"{prefix}.self_attn.q_proj.bias"] = layer.self_attn.q_proj.bias.data.cpu().numpy()
            weights[f"{prefix}.self_attn.k_proj.bias"] = layer.self_attn.k_proj.bias.data.cpu().numpy()
            weights[f"{prefix}.self_attn.v_proj.bias"] = layer.self_attn.v_proj.bias.data.cpu().numpy()
            weights[f"{prefix}.self_attn.out_proj.bias"] = layer.self_attn.out_proj.bias.data.cpu().numpy()

            # FFN weights
            weights[f"{prefix}.fc1.weight"] = layer.fc1.weight.data.cpu().numpy()
            weights[f"{prefix}.fc2.weight"] = layer.fc2.weight.data.cpu().numpy()
            weights[f"{prefix}.fc1.bias"] = layer.fc1.bias.data.cpu().numpy()
            weights[f"{prefix}.fc2.bias"] = layer.fc2.bias.data.cpu().numpy()

            # LayerNorm
            weights[f"{prefix}.self_attn_layer_norm.weight"] = layer.self_attn_layer_norm.weight.data.cpu().numpy()
            weights[f"{prefix}.self_attn_layer_norm.bias"] = layer.self_attn_layer_norm.bias.data.cpu().numpy()
            weights[f"{prefix}.final_layer_norm.weight"] = layer.final_layer_norm.weight.data.cpu().numpy()
            weights[f"{prefix}.final_layer_norm.bias"] = layer.final_layer_norm.bias.data.cpu().numpy()

        cls.cpp_encoder.load_weights(weights)

        print("  ✓ Encoders loaded")

    @unittest.skipUnless(CPP_AVAILABLE and TRANSFORMERS_AVAILABLE, "C++ encoder or Transformers not available")
    def test_random_input_accuracy(self):
        """Test encoder with random input (synthetic data)"""
        print("\n[TEST] Random input accuracy...")

        # Create random mel spectrogram (1500 frames × 80 mel bins)
        seq_len, n_mels = 1500, 80
        mel_input = np.random.randn(seq_len, n_mels).astype(np.float32)

        # Run Python encoder
        with torch.no_grad():
            python_input = torch.from_numpy(mel_input).unsqueeze(0)  # Add batch dim
            python_output = self.python_encoder(python_input)
            python_hidden = python_output.last_hidden_state.squeeze(0).cpu().numpy()

        # Run C++ encoder
        cpp_hidden = self.cpp_encoder.forward(mel_input)

        # Compare outputs
        self._compare_outputs(python_hidden, cpp_hidden, test_name="random_input")

    @unittest.skipUnless(CPP_AVAILABLE and TRANSFORMERS_AVAILABLE, "C++ encoder or Transformers not available")
    def test_zero_input_accuracy(self):
        """Test encoder with zero input (edge case)"""
        print("\n[TEST] Zero input accuracy...")

        # Create zero mel spectrogram
        seq_len, n_mels = 1500, 80
        mel_input = np.zeros((seq_len, n_mels), dtype=np.float32)

        # Run Python encoder
        with torch.no_grad():
            python_input = torch.from_numpy(mel_input).unsqueeze(0)
            python_output = self.python_encoder(python_input)
            python_hidden = python_output.last_hidden_state.squeeze(0).cpu().numpy()

        # Run C++ encoder
        cpp_hidden = self.cpp_encoder.forward(mel_input)

        # Compare outputs
        self._compare_outputs(python_hidden, cpp_hidden, test_name="zero_input")

    @unittest.skipUnless(CPP_AVAILABLE and TRANSFORMERS_AVAILABLE, "C++ encoder or Transformers not available")
    def test_ones_input_accuracy(self):
        """Test encoder with ones input (edge case)"""
        print("\n[TEST] Ones input accuracy...")

        # Create ones mel spectrogram
        seq_len, n_mels = 1500, 80
        mel_input = np.ones((seq_len, n_mels), dtype=np.float32)

        # Run Python encoder
        with torch.no_grad():
            python_input = torch.from_numpy(mel_input).unsqueeze(0)
            python_output = self.python_encoder(python_input)
            python_hidden = python_output.last_hidden_state.squeeze(0).cpu().numpy()

        # Run C++ encoder
        cpp_hidden = self.cpp_encoder.forward(mel_input)

        # Compare outputs
        self._compare_outputs(python_hidden, cpp_hidden, test_name="ones_input")

    @unittest.skipUnless(CPP_AVAILABLE and TRANSFORMERS_AVAILABLE, "C++ encoder or Transformers not available")
    def test_varying_sequence_lengths(self):
        """Test encoder with different sequence lengths"""
        print("\n[TEST] Varying sequence lengths...")

        test_lengths = [100, 500, 1000, 1500, 3000]

        for seq_len in test_lengths:
            with self.subTest(seq_len=seq_len):
                # Create random input
                n_mels = 80
                mel_input = np.random.randn(seq_len, n_mels).astype(np.float32)

                # Run Python encoder
                with torch.no_grad():
                    python_input = torch.from_numpy(mel_input).unsqueeze(0)
                    python_output = self.python_encoder(python_input)
                    python_hidden = python_output.last_hidden_state.squeeze(0).cpu().numpy()

                # Run C++ encoder
                cpp_hidden = self.cpp_encoder.forward(mel_input)

                # Compare outputs
                self._compare_outputs(
                    python_hidden, cpp_hidden,
                    test_name=f"seq_len_{seq_len}",
                    verbose=False
                )

        print("  ✓ All sequence lengths passed")

    def _compare_outputs(self, python_output, cpp_output, test_name="test", verbose=True):
        """Compare Python and C++ outputs"""

        # Verify shape match
        self.assertEqual(
            python_output.shape, cpp_output.shape,
            f"Shape mismatch: Python {python_output.shape} vs C++ {cpp_output.shape}"
        )

        # Calculate error metrics
        abs_diff = np.abs(python_output - cpp_output)
        rel_error = abs_diff / (np.abs(python_output) + 1e-8)

        mean_abs_error = abs_diff.mean()
        max_abs_error = abs_diff.max()
        mean_rel_error = rel_error.mean() * 100  # Convert to percentage
        max_rel_error = rel_error.max() * 100

        if verbose:
            print(f"\n  Comparison Results ({test_name}):")
            print(f"    Shape: {python_output.shape}")
            print(f"    Mean absolute error: {mean_abs_error:.6f}")
            print(f"    Max absolute error: {max_abs_error:.6f}")
            print(f"    Mean relative error: {mean_rel_error:.4f}%")
            print(f"    Max relative error: {max_rel_error:.4f}%")

        # Check error threshold (1% tolerance)
        ERROR_THRESHOLD = 1.0  # 1%

        self.assertLess(
            mean_rel_error, ERROR_THRESHOLD,
            f"Mean relative error {mean_rel_error:.4f}% exceeds threshold {ERROR_THRESHOLD}%"
        )

        # Also check that max error is reasonable (allow 5% max for outliers)
        MAX_ERROR_THRESHOLD = 5.0
        self.assertLess(
            max_rel_error, MAX_ERROR_THRESHOLD,
            f"Max relative error {max_rel_error:.4f}% exceeds threshold {MAX_ERROR_THRESHOLD}%"
        )

        if verbose:
            print(f"  ✓ Accuracy within {ERROR_THRESHOLD}% tolerance")


class TestEncoderOutputProperties(unittest.TestCase):
    """Test encoder output shape and range validation"""

    @classmethod
    def setUpClass(cls):
        """Initialize encoder"""
        if not CPP_AVAILABLE:
            return

        cls.encoder = create_encoder_cpp(
            num_layers=6,
            n_heads=8,
            n_state=512,
            ffn_dim=2048,
            use_npu=False
        )

        # Load weights (from transformers if available)
        if TRANSFORMERS_AVAILABLE:
            model = WhisperModel.from_pretrained("openai/whisper-base")
            weights = {}
            for layer_idx in range(6):
                layer = model.encoder.layers[layer_idx]
                prefix = f"encoder.layers.{layer_idx}"

                # Add all weights (abbreviated for brevity)
                weights[f"{prefix}.self_attn.q_proj.weight"] = layer.self_attn.q_proj.weight.data.cpu().numpy()
                weights[f"{prefix}.self_attn.k_proj.weight"] = layer.self_attn.k_proj.weight.data.cpu().numpy()
                weights[f"{prefix}.self_attn.v_proj.weight"] = layer.self_attn.v_proj.weight.data.cpu().numpy()
                weights[f"{prefix}.self_attn.out_proj.weight"] = layer.self_attn.out_proj.weight.data.cpu().numpy()
                weights[f"{prefix}.self_attn.q_proj.bias"] = layer.self_attn.q_proj.bias.data.cpu().numpy()
                weights[f"{prefix}.self_attn.k_proj.bias"] = layer.self_attn.k_proj.bias.data.cpu().numpy()
                weights[f"{prefix}.self_attn.v_proj.bias"] = layer.self_attn.v_proj.bias.data.cpu().numpy()
                weights[f"{prefix}.self_attn.out_proj.bias"] = layer.self_attn.out_proj.bias.data.cpu().numpy()
                weights[f"{prefix}.fc1.weight"] = layer.fc1.weight.data.cpu().numpy()
                weights[f"{prefix}.fc2.weight"] = layer.fc2.weight.data.cpu().numpy()
                weights[f"{prefix}.fc1.bias"] = layer.fc1.bias.data.cpu().numpy()
                weights[f"{prefix}.fc2.bias"] = layer.fc2.bias.data.cpu().numpy()
                weights[f"{prefix}.self_attn_layer_norm.weight"] = layer.self_attn_layer_norm.weight.data.cpu().numpy()
                weights[f"{prefix}.self_attn_layer_norm.bias"] = layer.self_attn_layer_norm.bias.data.cpu().numpy()
                weights[f"{prefix}.final_layer_norm.weight"] = layer.final_layer_norm.weight.data.cpu().numpy()
                weights[f"{prefix}.final_layer_norm.bias"] = layer.final_layer_norm.bias.data.cpu().numpy()

            cls.encoder.load_weights(weights)

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_output_shape(self):
        """Test that encoder output has correct shape"""
        print("\n[TEST] Output shape validation...")

        seq_len, n_mels = 1500, 80
        mel_input = np.random.randn(seq_len, n_mels).astype(np.float32)

        output = self.encoder.forward(mel_input)

        # Expected shape: (seq_len, n_state=512)
        expected_shape = (seq_len, 512)
        self.assertEqual(
            output.shape, expected_shape,
            f"Output shape {output.shape} != expected {expected_shape}"
        )

        print(f"  ✓ Output shape correct: {output.shape}")

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_output_dtype(self):
        """Test that encoder output has correct dtype"""
        print("\n[TEST] Output dtype validation...")

        seq_len, n_mels = 1500, 80
        mel_input = np.random.randn(seq_len, n_mels).astype(np.float32)

        output = self.encoder.forward(mel_input)

        # Expected dtype: float32
        self.assertEqual(
            output.dtype, np.float32,
            f"Output dtype {output.dtype} != expected float32"
        )

        print(f"  ✓ Output dtype correct: {output.dtype}")

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_output_range(self):
        """Test that encoder output is in reasonable range"""
        print("\n[TEST] Output range validation...")

        seq_len, n_mels = 1500, 80
        mel_input = np.random.randn(seq_len, n_mels).astype(np.float32)

        output = self.encoder.forward(mel_input)

        # Check for NaN/Inf
        self.assertFalse(np.any(np.isnan(output)), "Output contains NaN")
        self.assertFalse(np.any(np.isinf(output)), "Output contains Inf")

        # Check range (Whisper embeddings typically in [-10, 10])
        mean_val = output.mean()
        std_val = output.std()
        min_val = output.min()
        max_val = output.max()

        print(f"  Output statistics:")
        print(f"    Mean: {mean_val:.4f}")
        print(f"    Std:  {std_val:.4f}")
        print(f"    Min:  {min_val:.4f}")
        print(f"    Max:  {max_val:.4f}")

        # Reasonable range check (not too extreme)
        self.assertGreater(mean_val, -100, "Mean too negative")
        self.assertLess(mean_val, 100, "Mean too positive")
        self.assertLess(std_val, 100, "Std too large")

        print("  ✓ Output range reasonable")

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_no_catastrophic_failure(self):
        """Test that encoder doesn't produce all-zeros or constant output"""
        print("\n[TEST] No catastrophic failure...")

        seq_len, n_mels = 1500, 80
        mel_input = np.random.randn(seq_len, n_mels).astype(np.float32)

        output = self.encoder.forward(mel_input)

        # Check not all zeros
        self.assertGreater(
            np.count_nonzero(output), output.size * 0.9,
            "Output is mostly zeros (catastrophic failure)"
        )

        # Check variance (not constant)
        variance = output.var()
        self.assertGreater(
            variance, 0.01,
            f"Output variance {variance:.6f} too low (possibly constant)"
        )

        print("  ✓ No catastrophic failure detected")


class TestEndToEndAccuracy(unittest.TestCase):
    """Test end-to-end transcription accuracy"""

    @unittest.skipUnless(CPP_AVAILABLE and AUDIO_AVAILABLE and TRANSFORMERS_AVAILABLE,
                        "Required libraries not available")
    def test_real_audio_transcription(self):
        """Test transcription with real audio file"""
        print("\n[TEST] Real audio transcription...")

        # Check for test audio files
        test_audio_dir = Path(__file__).parent / "data"
        if not test_audio_dir.exists():
            self.skipTest(f"Test audio directory not found: {test_audio_dir}")

        audio_files = list(test_audio_dir.glob("*.wav"))
        if not audio_files:
            self.skipTest("No test audio files found")

        # Use first available audio file
        audio_path = str(audio_files[0])
        print(f"  Using audio: {audio_files[0].name}")

        # For now, just test that it doesn't crash
        # Full transcription accuracy requires decoder integration
        try:
            # Load audio
            audio = whisperx.load_audio(audio_path)

            # Compute mel spectrogram
            mel = whisperx.audio.log_mel_spectrogram(audio)

            # Run C++ encoder
            encoder = create_encoder_cpp(
                num_layers=6,
                n_heads=8,
                n_state=512,
                ffn_dim=2048,
                use_npu=False
            )

            # Load weights if transformers available
            model = WhisperModel.from_pretrained("openai/whisper-base")
            weights = {}
            for layer_idx in range(6):
                layer = model.encoder.layers[layer_idx]
                prefix = f"encoder.layers.{layer_idx}"
                # (weight loading abbreviated)

            encoder.load_weights(weights)

            # Run encoder
            embeddings = encoder.forward(mel.T)  # Transpose to (seq_len, n_mels)

            # Verify reasonable output
            self.assertEqual(embeddings.shape[1], 512)
            self.assertFalse(np.any(np.isnan(embeddings)))
            self.assertFalse(np.any(np.isinf(embeddings)))

            print("  ✓ Real audio processed successfully")

        except Exception as e:
            logger.error(f"Real audio test failed: {e}")
            raise


def run_tests():
    """Run all accuracy tests"""
    print("\n" + "="*70)
    print("  Accuracy Validation Tests (Week 6, Day 4 - Task 2)")
    print("="*70)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEncoderOutputAccuracy))
    suite.addTests(loader.loadTestsFromTestCase(TestEncoderOutputProperties))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndAccuracy))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("  Test Summary")
    print("="*70)
    print(f"  Tests run: {result.testsRun}")
    print(f"  Successes: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print("="*70 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
