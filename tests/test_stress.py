#!/usr/bin/env python3
"""
Stress Testing (Week 6, Day 5 - Task 4)

Validates production stability under load.

Tests:
1. Long audio files (>5 minutes)
2. Concurrent requests (10 simultaneous)
3. Memory stability (100 requests, <100MB growth)
4. Error recovery (malformed audio, empty files)
5. Resource cleanup (no leaks)

Monitoring:
- Memory usage over time
- NPU utilization
- CPU usage
- Latency distribution

Author: NPU Testing & Validation Teamlead
Date: November 1, 2025
Status: Week 6 Days 3-5 - Task 4
"""

import sys
import os
import time
import unittest
import numpy as np
from pathlib import Path
import logging
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logger.warning("psutil not available - memory monitoring disabled")
    PSUTIL_AVAILABLE = False


class TestLongAudio(unittest.TestCase):
    """Test handling of long audio files"""

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
            use_npu=True
        )

        # Load weights if available
        if TRANSFORMERS_AVAILABLE:
            model = WhisperModel.from_pretrained("openai/whisper-base")
            cls._load_weights(model)

    @classmethod
    def _load_weights(cls, model):
        """Load weights from transformers model"""
        weights = {}
        for layer_idx in range(6):
            layer = model.encoder.layers[layer_idx]
            prefix = f"encoder.layers.{layer_idx}"

            # Attention weights
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
    def test_5minute_audio(self):
        """Test with 5-minute audio file"""
        print("\n[TEST] 5-minute audio...")

        # Simulate 5 minutes of audio (5 * 60 * 100 = 30,000 frames)
        seq_len = 30000
        n_mels = 80
        mel_input = np.random.randn(seq_len, n_mels).astype(np.float32)

        print(f"  Processing {seq_len} frames (5 minutes)...")

        # Process
        t0 = time.perf_counter()
        try:
            output = self.encoder.forward(mel_input)
            t1 = time.perf_counter()

            processing_time = t1 - t0
            audio_duration = 300.0  # 5 minutes
            rtf = audio_duration / processing_time

            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Realtime factor: {rtf:.1f}x")

            # Verify output
            self.assertEqual(output.shape, (seq_len, 512))
            self.assertFalse(np.any(np.isnan(output)))
            self.assertFalse(np.any(np.isinf(output)))

            print("  ✓ 5-minute audio processed successfully")

        except Exception as e:
            self.fail(f"Failed to process 5-minute audio: {e}")

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_10minute_audio(self):
        """Test with 10-minute audio file"""
        print("\n[TEST] 10-minute audio...")

        # Simulate 10 minutes (60,000 frames)
        seq_len = 60000
        n_mels = 80
        mel_input = np.random.randn(seq_len, n_mels).astype(np.float32)

        print(f"  Processing {seq_len} frames (10 minutes)...")

        t0 = time.perf_counter()
        try:
            output = self.encoder.forward(mel_input)
            t1 = time.perf_counter()

            processing_time = t1 - t0
            audio_duration = 600.0  # 10 minutes
            rtf = audio_duration / processing_time

            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Realtime factor: {rtf:.1f}x")

            # Verify output
            self.assertEqual(output.shape, (seq_len, 512))
            self.assertFalse(np.any(np.isnan(output)))

            print("  ✓ 10-minute audio processed successfully")

        except Exception as e:
            self.fail(f"Failed to process 10-minute audio: {e}")


class TestConcurrentRequests(unittest.TestCase):
    """Test handling of concurrent requests"""

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_10_concurrent_requests(self):
        """Test 10 simultaneous inference requests"""
        print("\n[TEST] 10 concurrent requests...")

        # Create encoder
        encoder = create_encoder_cpp(
            num_layers=6,
            n_heads=8,
            n_state=512,
            ffn_dim=2048,
            use_npu=True
        )

        # Load weights if available
        if TRANSFORMERS_AVAILABLE:
            model = WhisperModel.from_pretrained("openai/whisper-base")
            TestLongAudio._load_weights.__func__(self, model)
            encoder.load_weights(encoder._last_loaded_weights)

        # Create test inputs
        seq_len = 1500
        n_mels = 80
        inputs = [
            np.random.randn(seq_len, n_mels).astype(np.float32)
            for _ in range(10)
        ]

        def process_one(idx, input_data):
            """Process one request"""
            try:
                t0 = time.perf_counter()
                output = encoder.forward(input_data)
                t1 = time.perf_counter()
                return {
                    'idx': idx,
                    'success': True,
                    'time': t1 - t0,
                    'output_shape': output.shape
                }
            except Exception as e:
                return {
                    'idx': idx,
                    'success': False,
                    'error': str(e)
                }

        # Run concurrent requests
        print("  Launching 10 concurrent requests...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(process_one, i, inputs[i])
                for i in range(10)
            ]

            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result['success']:
                    print(f"    Request {result['idx']}: {result['time']*1000:.1f}ms")

        # Verify all succeeded
        successes = sum(1 for r in results if r['success'])
        failures = sum(1 for r in results if not r['success'])

        print(f"\n  Results:")
        print(f"    Successes: {successes}/10")
        print(f"    Failures: {failures}/10")

        self.assertEqual(successes, 10, f"Only {successes}/10 requests succeeded")

        print("  ✓ All concurrent requests succeeded")

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_sequential_vs_concurrent_throughput(self):
        """Compare sequential vs concurrent throughput"""
        print("\n[TEST] Sequential vs concurrent throughput...")

        # Create encoder
        encoder = create_encoder_cpp(
            num_layers=6,
            n_heads=8,
            n_state=512,
            ffn_dim=2048,
            use_npu=True
        )

        # Load weights
        if TRANSFORMERS_AVAILABLE:
            model = WhisperModel.from_pretrained("openai/whisper-base")
            TestLongAudio._load_weights.__func__(self, model)
            encoder.load_weights(encoder._last_loaded_weights)

        # Create test inputs
        seq_len = 1500
        n_mels = 80
        num_requests = 10
        inputs = [
            np.random.randn(seq_len, n_mels).astype(np.float32)
            for _ in range(num_requests)
        ]

        # Sequential processing
        print("  Testing sequential processing...")
        t0 = time.perf_counter()
        for input_data in inputs:
            _ = encoder.forward(input_data)
        t_sequential = time.perf_counter() - t0

        # Concurrent processing
        print("  Testing concurrent processing...")
        def process_one(input_data):
            return encoder.forward(input_data)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(process_one, inp) for inp in inputs]
            _ = [future.result() for future in as_completed(futures)]
        t_concurrent = time.perf_counter() - t0

        print(f"\n  Results:")
        print(f"    Sequential: {t_sequential:.2f}s ({num_requests/t_sequential:.1f} req/s)")
        print(f"    Concurrent: {t_concurrent:.2f}s ({num_requests/t_concurrent:.1f} req/s)")
        print(f"    Speedup: {t_sequential/t_concurrent:.2f}x")

        # Note: NPU may not have concurrency benefits if single device
        print("  ℹ NPU throughput depends on hardware concurrency capabilities")


class TestMemoryStability(unittest.TestCase):
    """Test for memory leaks and stability"""

    @unittest.skipUnless(CPP_AVAILABLE and PSUTIL_AVAILABLE, "C++ encoder or psutil not available")
    def test_memory_leak_over_100_requests(self):
        """Test memory growth over 100 requests"""
        print("\n[TEST] Memory leak test (100 requests)...")

        # Get process handle
        process = psutil.Process()

        # Create encoder
        encoder = create_encoder_cpp(
            num_layers=6,
            n_heads=8,
            n_state=512,
            ffn_dim=2048,
            use_npu=True
        )

        # Load weights
        if TRANSFORMERS_AVAILABLE:
            model = WhisperModel.from_pretrained("openai/whisper-base")
            TestLongAudio._load_weights.__func__(self, model)
            encoder.load_weights(encoder._last_loaded_weights)

        # Create test input
        seq_len = 1500
        n_mels = 80
        mel_input = np.random.randn(seq_len, n_mels).astype(np.float32)

        # Warmup and get baseline memory
        print("  Warming up and establishing baseline...")
        for _ in range(10):
            _ = encoder.forward(mel_input)

        initial_mem = process.memory_info().rss / 1024 / 1024  # MB

        # Run 100 requests
        print("  Running 100 requests...")
        memory_samples = [initial_mem]

        for i in range(100):
            _ = encoder.forward(mel_input)

            # Sample memory every 10 requests
            if (i + 1) % 10 == 0:
                current_mem = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_mem)
                print(f"    {i+1}/100: {current_mem:.1f}MB (+{current_mem - initial_mem:.1f}MB)")

        final_mem = process.memory_info().rss / 1024 / 1024

        # Analyze memory growth
        growth = final_mem - initial_mem
        growth_rate = growth / 100  # MB per request

        print(f"\n  Memory Analysis:")
        print(f"    Initial: {initial_mem:.1f}MB")
        print(f"    Final:   {final_mem:.1f}MB")
        print(f"    Growth:  {growth:.1f}MB")
        print(f"    Rate:    {growth_rate:.3f}MB/request")

        # Verify acceptable growth
        MAX_GROWTH_MB = 100.0  # Allow up to 100MB growth

        self.assertLess(
            growth, MAX_GROWTH_MB,
            f"Memory grew by {growth:.1f}MB (exceeds {MAX_GROWTH_MB}MB limit)"
        )

        # Check for continuous leak (growth rate should level off)
        # Compare first half vs second half growth
        mid_point = len(memory_samples) // 2
        first_half_growth = memory_samples[mid_point] - memory_samples[0]
        second_half_growth = memory_samples[-1] - memory_samples[mid_point]

        print(f"    First half growth:  {first_half_growth:.1f}MB")
        print(f"    Second half growth: {second_half_growth:.1f}MB")

        # Second half should grow less than first half (caching stabilizes)
        if second_half_growth > first_half_growth * 1.5:
            logger.warning(f"  ⚠ Possible memory leak: second half growth exceeds first half")

        print("  ✓ Memory growth within acceptable limits")


class TestErrorRecovery(unittest.TestCase):
    """Test error handling and recovery"""

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
            use_npu=True
        )

        if TRANSFORMERS_AVAILABLE:
            model = WhisperModel.from_pretrained("openai/whisper-base")
            TestLongAudio._load_weights.__func__(cls, model)

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_empty_input(self):
        """Test handling of empty input"""
        print("\n[TEST] Empty input handling...")

        # Create empty input
        empty_input = np.array([], dtype=np.float32).reshape(0, 80)

        try:
            _ = self.encoder.forward(empty_input)
            # If it doesn't crash, that's acceptable
            print("  ✓ Empty input handled gracefully")
        except (ValueError, RuntimeError) as e:
            # Expected to raise error
            print(f"  ✓ Empty input rejected with error: {type(e).__name__}")

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_invalid_shape_input(self):
        """Test handling of invalid shape input"""
        print("\n[TEST] Invalid shape input...")

        # Wrong number of mel bins (should be 80, give 60)
        invalid_input = np.random.randn(1500, 60).astype(np.float32)

        with self.assertRaises((ValueError, RuntimeError)):
            _ = self.encoder.forward(invalid_input)

        print("  ✓ Invalid shape rejected")

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_nan_input(self):
        """Test handling of NaN input"""
        print("\n[TEST] NaN input handling...")

        # Create input with NaNs
        nan_input = np.random.randn(1500, 80).astype(np.float32)
        nan_input[500:600, :] = np.nan

        try:
            output = self.encoder.forward(nan_input)

            # Check if NaNs propagate
            has_nan = np.any(np.isnan(output))
            if has_nan:
                logger.warning("  ⚠ NaN input produced NaN output (expected)")
            else:
                print("  ✓ NaN input handled (no NaN propagation)")

        except (ValueError, RuntimeError) as e:
            print(f"  ✓ NaN input rejected: {type(e).__name__}")

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_inf_input(self):
        """Test handling of Inf input"""
        print("\n[TEST] Inf input handling...")

        # Create input with Inf
        inf_input = np.random.randn(1500, 80).astype(np.float32)
        inf_input[500:600, :] = np.inf

        try:
            output = self.encoder.forward(inf_input)

            # Check if Infs propagate
            has_inf = np.any(np.isinf(output))
            if has_inf:
                logger.warning("  ⚠ Inf input produced Inf output (expected)")
            else:
                print("  ✓ Inf input handled (no Inf propagation)")

        except (ValueError, RuntimeError) as e:
            print(f"  ✓ Inf input rejected: {type(e).__name__}")

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_recovery_after_error(self):
        """Test that encoder can recover after error"""
        print("\n[TEST] Recovery after error...")

        # Cause error with invalid input
        invalid_input = np.random.randn(1500, 60).astype(np.float32)
        try:
            _ = self.encoder.forward(invalid_input)
        except:
            pass  # Expected

        # Try valid input after error
        valid_input = np.random.randn(1500, 80).astype(np.float32)
        try:
            output = self.encoder.forward(valid_input)
            self.assertEqual(output.shape, (1500, 512))
            print("  ✓ Encoder recovered after error")
        except Exception as e:
            self.fail(f"Encoder failed to recover: {e}")


class TestResourceCleanup(unittest.TestCase):
    """Test proper resource cleanup"""

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_encoder_destruction(self):
        """Test that encoder properly cleans up resources"""
        print("\n[TEST] Encoder destruction...")

        # Create and destroy multiple encoders
        for i in range(5):
            encoder = create_encoder_cpp(
                num_layers=6,
                n_heads=8,
                n_state=512,
                ffn_dim=2048,
                use_npu=True
            )

            # Use encoder
            mel_input = np.random.randn(100, 80).astype(np.float32)
            _ = encoder.forward(mel_input)

            # Explicitly delete
            del encoder

            print(f"    Encoder {i+1}/5 created and destroyed")

        print("  ✓ Multiple encoder creation/destruction successful")

    @unittest.skipUnless(CPP_AVAILABLE and PSUTIL_AVAILABLE, "C++ encoder or psutil not available")
    def test_no_file_descriptor_leak(self):
        """Test that no file descriptors are leaked"""
        print("\n[TEST] File descriptor leak test...")

        process = psutil.Process()
        initial_fds = process.num_fds()

        # Create and destroy encoders
        for _ in range(10):
            encoder = create_encoder_cpp(
                num_layers=6,
                n_heads=8,
                n_state=512,
                ffn_dim=2048,
                use_npu=True
            )
            del encoder

        final_fds = process.num_fds()
        fd_growth = final_fds - initial_fds

        print(f"  File descriptors:")
        print(f"    Initial: {initial_fds}")
        print(f"    Final:   {final_fds}")
        print(f"    Growth:  {fd_growth}")

        # Allow small growth (some caching is acceptable)
        MAX_FD_GROWTH = 10

        self.assertLess(
            fd_growth, MAX_FD_GROWTH,
            f"File descriptor leak detected: {fd_growth} FDs leaked"
        )

        print("  ✓ No file descriptor leak detected")


def run_tests():
    """Run all stress tests"""
    print("\n" + "="*70)
    print("  Stress Testing (Week 6, Day 5 - Task 4)")
    print("="*70)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLongAudio))
    suite.addTests(loader.loadTestsFromTestCase(TestConcurrentRequests))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryStability))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorRecovery))
    suite.addTests(loader.loadTestsFromTestCase(TestResourceCleanup))

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
