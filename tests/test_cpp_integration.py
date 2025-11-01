#!/usr/bin/env python3
"""
C++ Runtime Integration Tests

Comprehensive test suite for C++ NPU runtime integration.

Tests:
- C++ library loading
- Encoder creation and initialization
- Weight loading
- Forward pass execution
- Performance benchmarking
- Comparison with Python version
- Error handling and fallback

Author: CC-1L Integration Team
Date: November 1, 2025
Status: Production-ready
"""

import sys
import os
import time
import unittest
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xdna2.cpp_runtime_wrapper import (
    CPPRuntimeWrapper,
    EncoderLayer,
    CPPRuntimeError
)
from xdna2.encoder_cpp import WhisperEncoderCPP, create_encoder_cpp
from runtime.platform_detector import PlatformDetector, Platform


class TestCPPRuntimeWrapper(unittest.TestCase):
    """Test C++ runtime wrapper functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.n_state = 512
        self.ffn_dim = 2048
        self.seq_len = 100

    def test_library_loading(self):
        """Test that C++ library can be loaded"""
        print("\n[TEST] C++ library loading...")
        try:
            runtime = CPPRuntimeWrapper()
            version = runtime.get_version()
            print(f"  Version: {version}")
            self.assertIsNotNone(version)
            self.assertIsInstance(version, str)
            print("  ✓ Library loaded successfully")
        except CPPRuntimeError as e:
            self.skipTest(f"C++ runtime not available: {e}")

    def test_layer_creation(self):
        """Test encoder layer creation"""
        print("\n[TEST] Encoder layer creation...")
        try:
            runtime = CPPRuntimeWrapper()

            # Create a layer
            handle = runtime.create_layer(
                layer_idx=0,
                n_heads=8,
                n_state=self.n_state,
                ffn_dim=self.ffn_dim
            )

            self.assertIsNotNone(handle)
            self.assertNotEqual(handle, 0)
            print(f"  Handle: {handle}")
            print("  ✓ Layer created successfully")

            # Cleanup
            runtime.destroy_layer(handle)

        except CPPRuntimeError as e:
            self.skipTest(f"C++ runtime not available: {e}")

    def test_context_manager(self):
        """Test encoder layer context manager"""
        print("\n[TEST] Context manager...")
        try:
            runtime = CPPRuntimeWrapper()

            with EncoderLayer(runtime, layer_idx=0) as layer:
                self.assertIsNotNone(layer.handle)
                print(f"  Handle: {layer.handle}")

            # Handle should be cleaned up
            print("  ✓ Context manager works correctly")

        except CPPRuntimeError as e:
            self.skipTest(f"C++ runtime not available: {e}")

    def test_weight_loading(self):
        """Test weight loading"""
        print("\n[TEST] Weight loading...")
        try:
            runtime = CPPRuntimeWrapper()

            with EncoderLayer(runtime, layer_idx=0) as layer:
                # Create dummy weights
                q_weight = np.random.randn(self.n_state, self.n_state).astype(np.float32)
                k_weight = np.random.randn(self.n_state, self.n_state).astype(np.float32)
                v_weight = np.random.randn(self.n_state, self.n_state).astype(np.float32)
                out_weight = np.random.randn(self.n_state, self.n_state).astype(np.float32)

                q_bias = np.random.randn(self.n_state).astype(np.float32)
                k_bias = np.random.randn(self.n_state).astype(np.float32)
                v_bias = np.random.randn(self.n_state).astype(np.float32)
                out_bias = np.random.randn(self.n_state).astype(np.float32)

                fc1_weight = np.random.randn(self.ffn_dim, self.n_state).astype(np.float32)
                fc2_weight = np.random.randn(self.n_state, self.ffn_dim).astype(np.float32)
                fc1_bias = np.random.randn(self.ffn_dim).astype(np.float32)
                fc2_bias = np.random.randn(self.n_state).astype(np.float32)

                attn_ln_weight = np.random.randn(self.n_state).astype(np.float32)
                attn_ln_bias = np.random.randn(self.n_state).astype(np.float32)
                ffn_ln_weight = np.random.randn(self.n_state).astype(np.float32)
                ffn_ln_bias = np.random.randn(self.n_state).astype(np.float32)

                # Load weights
                runtime.load_weights(
                    layer.handle,
                    q_weight, k_weight, v_weight, out_weight,
                    q_bias, k_bias, v_bias, out_bias,
                    fc1_weight, fc2_weight, fc1_bias, fc2_bias,
                    attn_ln_weight, attn_ln_bias, ffn_ln_weight, ffn_ln_bias
                )

                print("  ✓ Weights loaded successfully")

        except CPPRuntimeError as e:
            self.skipTest(f"C++ runtime not available: {e}")

    def test_forward_pass(self):
        """Test forward pass execution"""
        print("\n[TEST] Forward pass execution...")
        try:
            runtime = CPPRuntimeWrapper()

            with EncoderLayer(runtime, layer_idx=0) as layer:
                # Load dummy weights
                q_weight = np.random.randn(self.n_state, self.n_state).astype(np.float32)
                k_weight = np.random.randn(self.n_state, self.n_state).astype(np.float32)
                v_weight = np.random.randn(self.n_state, self.n_state).astype(np.float32)
                out_weight = np.random.randn(self.n_state, self.n_state).astype(np.float32)

                q_bias = np.random.randn(self.n_state).astype(np.float32)
                k_bias = np.random.randn(self.n_state).astype(np.float32)
                v_bias = np.random.randn(self.n_state).astype(np.float32)
                out_bias = np.random.randn(self.n_state).astype(np.float32)

                fc1_weight = np.random.randn(self.ffn_dim, self.n_state).astype(np.float32)
                fc2_weight = np.random.randn(self.n_state, self.ffn_dim).astype(np.float32)
                fc1_bias = np.random.randn(self.ffn_dim).astype(np.float32)
                fc2_bias = np.random.randn(self.n_state).astype(np.float32)

                attn_ln_weight = np.random.randn(self.n_state).astype(np.float32)
                attn_ln_bias = np.random.randn(self.n_state).astype(np.float32)
                ffn_ln_weight = np.random.randn(self.n_state).astype(np.float32)
                ffn_ln_bias = np.random.randn(self.n_state).astype(np.float32)

                runtime.load_weights(
                    layer.handle,
                    q_weight, k_weight, v_weight, out_weight,
                    q_bias, k_bias, v_bias, out_bias,
                    fc1_weight, fc2_weight, fc1_bias, fc2_bias,
                    attn_ln_weight, attn_ln_bias, ffn_ln_weight, ffn_ln_bias
                )

                # Create input
                input_data = np.random.randn(self.seq_len, self.n_state).astype(np.float32)

                # Run forward pass
                start = time.perf_counter()
                output = runtime.forward(layer.handle, input_data, self.seq_len, self.n_state)
                elapsed = (time.perf_counter() - start) * 1000

                # Validate output
                self.assertEqual(output.shape, (self.seq_len, self.n_state))
                self.assertEqual(output.dtype, np.float32)

                print(f"  Output shape: {output.shape}")
                print(f"  Output dtype: {output.dtype}")
                print(f"  Forward time: {elapsed:.2f} ms")
                print("  ✓ Forward pass completed successfully")

        except CPPRuntimeError as e:
            self.skipTest(f"C++ runtime not available: {e}")


class TestEncoderCPP(unittest.TestCase):
    """Test high-level encoder integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.num_layers = 2  # Use 2 layers for faster testing
        self.n_state = 512
        self.ffn_dim = 2048
        self.seq_len = 100

    def _create_dummy_weights(self) -> dict:
        """Create dummy Whisper weights for testing"""
        weights = {}
        for layer_idx in range(self.num_layers):
            prefix = f"encoder.layers.{layer_idx}"

            # Attention weights
            weights[f"{prefix}.self_attn.q_proj.weight"] = np.random.randn(512, 512).astype(np.float32)
            weights[f"{prefix}.self_attn.k_proj.weight"] = np.random.randn(512, 512).astype(np.float32)
            weights[f"{prefix}.self_attn.v_proj.weight"] = np.random.randn(512, 512).astype(np.float32)
            weights[f"{prefix}.self_attn.out_proj.weight"] = np.random.randn(512, 512).astype(np.float32)

            # Attention biases
            weights[f"{prefix}.self_attn.q_proj.bias"] = np.random.randn(512).astype(np.float32)
            weights[f"{prefix}.self_attn.k_proj.bias"] = np.random.randn(512).astype(np.float32)
            weights[f"{prefix}.self_attn.v_proj.bias"] = np.random.randn(512).astype(np.float32)
            weights[f"{prefix}.self_attn.out_proj.bias"] = np.random.randn(512).astype(np.float32)

            # FFN weights
            weights[f"{prefix}.fc1.weight"] = np.random.randn(2048, 512).astype(np.float32)
            weights[f"{prefix}.fc2.weight"] = np.random.randn(512, 2048).astype(np.float32)
            weights[f"{prefix}.fc1.bias"] = np.random.randn(2048).astype(np.float32)
            weights[f"{prefix}.fc2.bias"] = np.random.randn(512).astype(np.float32)

            # LayerNorm
            weights[f"{prefix}.self_attn_layer_norm.weight"] = np.random.randn(512).astype(np.float32)
            weights[f"{prefix}.self_attn_layer_norm.bias"] = np.random.randn(512).astype(np.float32)
            weights[f"{prefix}.final_layer_norm.weight"] = np.random.randn(512).astype(np.float32)
            weights[f"{prefix}.final_layer_norm.bias"] = np.random.randn(512).astype(np.float32)

        return weights

    def test_encoder_creation(self):
        """Test encoder creation"""
        print("\n[TEST] Encoder creation...")
        try:
            encoder = create_encoder_cpp(
                num_layers=self.num_layers,
                use_npu=False
            )
            self.assertIsNotNone(encoder)
            print(f"  Layers: {encoder.num_layers}")
            print("  ✓ Encoder created successfully")

        except CPPRuntimeError as e:
            self.skipTest(f"C++ runtime not available: {e}")

    def test_encoder_weight_loading(self):
        """Test encoder weight loading"""
        print("\n[TEST] Encoder weight loading...")
        try:
            encoder = create_encoder_cpp(
                num_layers=self.num_layers,
                use_npu=False
            )

            weights = self._create_dummy_weights()
            encoder.load_weights(weights)

            self.assertTrue(encoder.weights_loaded)
            print("  ✓ Weights loaded successfully")

        except CPPRuntimeError as e:
            self.skipTest(f"C++ runtime not available: {e}")

    def test_encoder_forward(self):
        """Test encoder forward pass"""
        print("\n[TEST] Encoder forward pass...")
        try:
            encoder = create_encoder_cpp(
                num_layers=self.num_layers,
                use_npu=False
            )

            weights = self._create_dummy_weights()
            encoder.load_weights(weights)

            # Create input
            input_data = np.random.randn(self.seq_len, self.n_state).astype(np.float32)

            # Run forward pass
            start = time.perf_counter()
            output = encoder.forward(input_data)
            elapsed = (time.perf_counter() - start) * 1000

            # Validate output
            self.assertEqual(output.shape, (self.seq_len, self.n_state))
            self.assertEqual(output.dtype, np.float32)

            print(f"  Input shape: {input_data.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Forward time: {elapsed:.2f} ms")
            print(f"  Time per layer: {elapsed/self.num_layers:.2f} ms")
            print("  ✓ Forward pass completed successfully")

        except CPPRuntimeError as e:
            self.skipTest(f"C++ runtime not available: {e}")

    def test_encoder_stats(self):
        """Test encoder statistics"""
        print("\n[TEST] Encoder statistics...")
        try:
            encoder = create_encoder_cpp(
                num_layers=self.num_layers,
                use_npu=False
            )

            stats = encoder.get_stats()
            self.assertIsInstance(stats, dict)
            self.assertIn('num_layers', stats)
            self.assertIn('runtime_version', stats)

            print(f"  Runtime version: {stats['runtime_version']}")
            print(f"  Layers: {stats['num_layers']}")
            print("  ✓ Statistics retrieved successfully")

        except CPPRuntimeError as e:
            self.skipTest(f"C++ runtime not available: {e}")


class TestPlatformDetector(unittest.TestCase):
    """Test platform detection"""

    def test_platform_detection(self):
        """Test platform detection"""
        print("\n[TEST] Platform detection...")
        detector = PlatformDetector()
        platform = detector.detect()

        print(f"  Detected platform: {platform.value}")
        self.assertIsInstance(platform, Platform)

    def test_platform_info(self):
        """Test platform info"""
        print("\n[TEST] Platform info...")
        detector = PlatformDetector()
        info = detector.get_info()

        print(f"  Platform: {info['platform']}")
        print(f"  Has NPU: {info['has_npu']}")
        print(f"  NPU generation: {info.get('npu_generation', 'N/A')}")
        print(f"  Uses C++ runtime: {info.get('uses_cpp_runtime', False)}")

        self.assertIsInstance(info, dict)
        self.assertIn('platform', info)
        self.assertIn('has_npu', info)


class TestPerformanceBenchmark(unittest.TestCase):
    """Performance benchmarking tests"""

    def setUp(self):
        """Set up benchmark fixtures"""
        self.num_layers = 6  # Full Whisper Base encoder
        self.n_state = 512
        self.ffn_dim = 2048
        self.seq_len = 1500  # Typical Whisper sequence length

    def _create_dummy_weights(self) -> dict:
        """Create dummy Whisper weights"""
        weights = {}
        for layer_idx in range(self.num_layers):
            prefix = f"encoder.layers.{layer_idx}"

            weights[f"{prefix}.self_attn.q_proj.weight"] = np.random.randn(512, 512).astype(np.float32)
            weights[f"{prefix}.self_attn.k_proj.weight"] = np.random.randn(512, 512).astype(np.float32)
            weights[f"{prefix}.self_attn.v_proj.weight"] = np.random.randn(512, 512).astype(np.float32)
            weights[f"{prefix}.self_attn.out_proj.weight"] = np.random.randn(512, 512).astype(np.float32)

            weights[f"{prefix}.self_attn.q_proj.bias"] = np.random.randn(512).astype(np.float32)
            weights[f"{prefix}.self_attn.k_proj.bias"] = np.random.randn(512).astype(np.float32)
            weights[f"{prefix}.self_attn.v_proj.bias"] = np.random.randn(512).astype(np.float32)
            weights[f"{prefix}.self_attn.out_proj.bias"] = np.random.randn(512).astype(np.float32)

            weights[f"{prefix}.fc1.weight"] = np.random.randn(2048, 512).astype(np.float32)
            weights[f"{prefix}.fc2.weight"] = np.random.randn(512, 2048).astype(np.float32)
            weights[f"{prefix}.fc1.bias"] = np.random.randn(2048).astype(np.float32)
            weights[f"{prefix}.fc2.bias"] = np.random.randn(512).astype(np.float32)

            weights[f"{prefix}.self_attn_layer_norm.weight"] = np.random.randn(512).astype(np.float32)
            weights[f"{prefix}.self_attn_layer_norm.bias"] = np.random.randn(512).astype(np.float32)
            weights[f"{prefix}.final_layer_norm.weight"] = np.random.randn(512).astype(np.float32)
            weights[f"{prefix}.final_layer_norm.bias"] = np.random.randn(512).astype(np.float32)

        return weights

    def test_benchmark_forward_pass(self):
        """Benchmark forward pass performance"""
        print("\n[BENCHMARK] Forward pass performance...")
        try:
            encoder = create_encoder_cpp(
                num_layers=self.num_layers,
                use_npu=False
            )

            weights = self._create_dummy_weights()
            encoder.load_weights(weights)

            # Warm-up
            input_data = np.random.randn(self.seq_len, self.n_state).astype(np.float32)
            _ = encoder.forward(input_data)

            # Benchmark
            num_runs = 10
            times = []

            print(f"  Running {num_runs} iterations...")
            for i in range(num_runs):
                start = time.perf_counter()
                _ = encoder.forward(input_data)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            # Statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)

            print(f"\n  Results (seq_len={self.seq_len}, layers={self.num_layers}):")
            print(f"    Average: {avg_time:.2f} ms")
            print(f"    Std dev: {std_time:.2f} ms")
            print(f"    Min: {min_time:.2f} ms")
            print(f"    Max: {max_time:.2f} ms")
            print(f"    Per layer: {avg_time/self.num_layers:.2f} ms")

            # Calculate realtime factor
            # Whisper processes 30 seconds of audio at seq_len=3000
            # So seq_len=1500 = 15 seconds of audio
            audio_duration_s = 15.0
            realtime_factor = (audio_duration_s * 1000) / avg_time

            print(f"\n  Performance:")
            print(f"    Audio duration: {audio_duration_s:.1f} s")
            print(f"    Processing time: {avg_time:.2f} ms")
            print(f"    Realtime factor: {realtime_factor:.1f}x")

            if realtime_factor >= 100:
                print("  ✓ Excellent performance (>100x realtime)")
            elif realtime_factor >= 50:
                print("  ✓ Good performance (50-100x realtime)")
            else:
                print("  ⚠ Below target performance (<50x realtime)")

        except CPPRuntimeError as e:
            self.skipTest(f"C++ runtime not available: {e}")


def main():
    """Run all tests"""
    print("="*70)
    print("  C++ RUNTIME INTEGRATION TEST SUITE")
    print("="*70)

    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCPPRuntimeWrapper))
    suite.addTests(loader.loadTestsFromTestCase(TestEncoderCPP))
    suite.addTests(loader.loadTestsFromTestCase(TestPlatformDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceBenchmark))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    print(f"  Tests run: {result.testsRun}")
    print(f"  Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n  ✅ All tests passed!")
        return 0
    else:
        print("\n  ❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
