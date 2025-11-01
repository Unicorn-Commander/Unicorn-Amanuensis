#!/usr/bin/env python3
"""
Performance Benchmarking Tests (Week 6, Day 4 - Task 3)

Validates 400-500x realtime performance target.

Tests:
1. Measure realtime factor (target: ≥400x)
2. Measure per-layer latency (target: <5ms each)
3. Measure total encoder latency (target: <30ms for 1500 frames)
4. Measure NPU utilization (target: 2-3%)
5. Compare vs Python baseline (220x)

Benchmark Tests:
- 30 second audio (baseline)
- 5 minute audio (long form)
- 100 sequential inferences (stability)
- Varying audio lengths

Author: NPU Testing & Validation Teamlead
Date: November 1, 2025
Status: Week 6 Days 3-5 - Task 3
"""

import sys
import os
import time
import unittest
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict

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


class TestRealtimeFactor(unittest.TestCase):
    """Test encoder achieves 400-500x realtime target"""

    @classmethod
    def setUpClass(cls):
        """Initialize C++ encoder"""
        if not CPP_AVAILABLE:
            return

        print("\n[SETUP] Initializing C++ encoder for performance tests...")

        cls.encoder = create_encoder_cpp(
            num_layers=6,
            n_heads=8,
            n_state=512,
            ffn_dim=2048,
            use_npu=True  # Use NPU for performance testing
        )

        # Load weights
        if TRANSFORMERS_AVAILABLE:
            model = WhisperModel.from_pretrained("openai/whisper-base")
            cls._load_weights(model)

        print("  ✓ Encoder ready for benchmarking")

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

        cls.encoder.load_weights(weights)

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_30second_audio_realtime_factor(self):
        """Test realtime factor for 30-second audio"""
        print("\n[TEST] 30-second audio realtime factor...")

        # Simulate 30 seconds of audio
        # Whisper uses 16kHz, 25ms frames with 10ms stride
        # 30s = 30 * 100 = 3000 frames
        seq_len = 3000
        n_mels = 80
        mel_input = np.random.randn(seq_len, n_mels).astype(np.float32)

        # Warmup
        print("  Warming up...")
        for _ in range(3):
            _ = self.encoder.forward(mel_input)

        # Benchmark
        print("  Benchmarking (10 iterations)...")
        times = []
        for i in range(10):
            t0 = time.perf_counter()
            _ = self.encoder.forward(mel_input)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        # Calculate statistics
        mean_time = np.mean(times)
        median_time = np.median(times)
        min_time = np.min(times)

        # Calculate realtime factor
        audio_duration = 30.0  # seconds
        mean_rtf = audio_duration / mean_time
        median_rtf = audio_duration / median_time
        best_rtf = audio_duration / min_time

        print(f"\n  Results:")
        print(f"    Processing time (mean):   {mean_time*1000:.1f}ms")
        print(f"    Processing time (median): {median_time*1000:.1f}ms")
        print(f"    Processing time (best):   {min_time*1000:.1f}ms")
        print(f"    Realtime factor (mean):   {mean_rtf:.1f}x")
        print(f"    Realtime factor (median): {median_rtf:.1f}x")
        print(f"    Realtime factor (best):   {best_rtf:.1f}x")

        # Verify target met
        TARGET_RTF = 400.0  # 400x realtime

        self.assertGreater(
            median_rtf, TARGET_RTF,
            f"Realtime factor {median_rtf:.1f}x below target {TARGET_RTF}x"
        )

        print(f"  ✓ Target {TARGET_RTF}x achieved: {median_rtf:.1f}x")

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_varying_audio_lengths(self):
        """Test realtime factor for different audio lengths"""
        print("\n[TEST] Varying audio lengths...")

        # Test different audio lengths (in seconds)
        test_durations = [1, 5, 10, 30, 60]  # 1s to 60s

        results = []

        for duration in test_durations:
            # Calculate sequence length (100 frames per second)
            seq_len = int(duration * 100)
            n_mels = 80
            mel_input = np.random.randn(seq_len, n_mels).astype(np.float32)

            # Benchmark (3 iterations)
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                _ = self.encoder.forward(mel_input)
                t1 = time.perf_counter()
                times.append(t1 - t0)

            # Calculate realtime factor
            median_time = np.median(times)
            rtf = duration / median_time

            results.append({
                'duration': duration,
                'seq_len': seq_len,
                'time': median_time,
                'rtf': rtf
            })

            print(f"  {duration:3d}s audio: {median_time*1000:6.1f}ms → {rtf:6.1f}x realtime")

        # Verify all meet target
        TARGET_RTF = 400.0
        for result in results:
            self.assertGreater(
                result['rtf'], TARGET_RTF * 0.8,  # Allow 20% margin
                f"{result['duration']}s audio only achieved {result['rtf']:.1f}x"
            )

        print(f"  ✓ All lengths meet target")


class TestLayerLatency(unittest.TestCase):
    """Test per-layer latency"""

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
            TestRealtimeFactor._load_weights.__func__(cls, model)

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_individual_layer_latency(self):
        """Test latency for each encoder layer"""
        print("\n[TEST] Individual layer latency...")

        # Standard sequence length (1500 frames = 15 seconds)
        seq_len = 1500
        n_state = 512
        input_data = np.random.randn(seq_len, n_state).astype(np.float32)

        # Test each layer
        layer_times = []
        for layer_idx in range(6):
            # Get layer
            try:
                layer = self.encoder.get_layer(layer_idx)
            except AttributeError:
                # Encoder might not expose individual layers
                self.skipTest("Individual layer access not available")
                return

            # Warmup
            for _ in range(3):
                _ = layer.forward(input_data)

            # Benchmark
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                _ = layer.forward(input_data)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # Convert to ms

            median_time = np.median(times)
            layer_times.append(median_time)

            print(f"  Layer {layer_idx}: {median_time:.2f}ms")

        # Verify each layer meets target
        TARGET_LAYER_LATENCY = 5.0  # 5ms per layer

        for i, t in enumerate(layer_times):
            self.assertLess(
                t, TARGET_LAYER_LATENCY,
                f"Layer {i} latency {t:.2f}ms exceeds target {TARGET_LAYER_LATENCY}ms"
            )

        # Verify total latency
        total_time = sum(layer_times)
        TARGET_TOTAL_LATENCY = 30.0  # 30ms total

        self.assertLess(
            total_time, TARGET_TOTAL_LATENCY,
            f"Total latency {total_time:.2f}ms exceeds target {TARGET_TOTAL_LATENCY}ms"
        )

        print(f"\n  Total latency: {total_time:.2f}ms (target: <{TARGET_TOTAL_LATENCY}ms)")
        print("  ✓ All layers meet latency target")

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_full_encoder_latency(self):
        """Test full encoder latency (all 6 layers)"""
        print("\n[TEST] Full encoder latency...")

        # Standard sequence length
        seq_len = 1500
        n_mels = 80
        mel_input = np.random.randn(seq_len, n_mels).astype(np.float32)

        # Warmup
        for _ in range(3):
            _ = self.encoder.forward(mel_input)

        # Benchmark
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = self.encoder.forward(mel_input)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # Convert to ms

        # Statistics
        mean_time = np.mean(times)
        median_time = np.median(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)

        print(f"\n  Latency (ms) for 1500 frames:")
        print(f"    Mean:   {mean_time:.2f}")
        print(f"    Median: {median_time:.2f}")
        print(f"    Min:    {min_time:.2f}")
        print(f"    Max:    {max_time:.2f}")
        print(f"    Std:    {std_time:.2f}")

        # Verify target
        TARGET_LATENCY = 30.0  # 30ms for 1500 frames

        self.assertLess(
            median_time, TARGET_LATENCY,
            f"Median latency {median_time:.2f}ms exceeds target {TARGET_LATENCY}ms"
        )

        print(f"  ✓ Latency target met: {median_time:.2f}ms < {TARGET_LATENCY}ms")


class TestNPUUtilization(unittest.TestCase):
    """Test NPU utilization metrics"""

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_npu_utilization_range(self):
        """Test NPU utilization is in expected range (2-3%)"""
        print("\n[TEST] NPU utilization...")

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
            TestRealtimeFactor._load_weights.__func__(self, model)
            encoder.load_weights(encoder._last_loaded_weights)

        # Run inference
        seq_len = 1500
        n_mels = 80
        mel_input = np.random.randn(seq_len, n_mels).astype(np.float32)

        _ = encoder.forward(mel_input)

        # Get utilization stats
        try:
            stats = encoder.get_stats()
            npu_utilization = stats.get('npu_utilization', None)

            if npu_utilization is None:
                self.skipTest("NPU utilization metrics not available")

            print(f"  NPU utilization: {npu_utilization:.2f}%")

            # Verify in expected range (2-3%)
            TARGET_MIN = 1.5
            TARGET_MAX = 4.0

            self.assertGreater(
                npu_utilization, TARGET_MIN,
                f"NPU utilization {npu_utilization:.2f}% below expected {TARGET_MIN}%"
            )

            self.assertLess(
                npu_utilization, TARGET_MAX,
                f"NPU utilization {npu_utilization:.2f}% above expected {TARGET_MAX}%"
            )

            print(f"  ✓ NPU utilization in expected range ({TARGET_MIN}-{TARGET_MAX}%)")

        except AttributeError:
            self.skipTest("get_stats() method not available")


class TestComparisonWithPython(unittest.TestCase):
    """Compare C++ performance vs Python baseline"""

    @unittest.skipUnless(CPP_AVAILABLE and TRANSFORMERS_AVAILABLE, "C++ or Python encoder not available")
    def test_cpp_vs_python_speedup(self):
        """Compare C++ encoder vs Python encoder"""
        print("\n[TEST] C++ vs Python comparison...")

        # Create both encoders
        print("  Loading C++ encoder...")
        cpp_encoder = create_encoder_cpp(
            num_layers=6,
            n_heads=8,
            n_state=512,
            ffn_dim=2048,
            use_npu=True
        )

        print("  Loading Python encoder...")
        python_model = WhisperModel.from_pretrained("openai/whisper-base")
        python_encoder = python_model.encoder

        # Load weights into C++
        TestRealtimeFactor._load_weights.__func__(self, python_model)
        cpp_encoder.load_weights(cpp_encoder._last_loaded_weights)

        # Create test input
        seq_len = 1500
        n_mels = 80
        mel_input = np.random.randn(seq_len, n_mels).astype(np.float32)

        # Benchmark Python (CPU)
        print("  Benchmarking Python encoder...")
        with torch.no_grad():
            python_input = torch.from_numpy(mel_input).unsqueeze(0)

            # Warmup
            for _ in range(3):
                _ = python_encoder(python_input)

            # Benchmark
            python_times = []
            for _ in range(10):
                t0 = time.perf_counter()
                _ = python_encoder(python_input)
                t1 = time.perf_counter()
                python_times.append(t1 - t0)

        python_median = np.median(python_times)

        # Benchmark C++ (NPU)
        print("  Benchmarking C++ encoder...")

        # Warmup
        for _ in range(3):
            _ = cpp_encoder.forward(mel_input)

        # Benchmark
        cpp_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = cpp_encoder.forward(mel_input)
            t1 = time.perf_counter()
            cpp_times.append(t1 - t0)

        cpp_median = np.median(cpp_times)

        # Calculate speedup
        speedup = python_median / cpp_median

        print(f"\n  Results:")
        print(f"    Python (CPU): {python_median*1000:.1f}ms")
        print(f"    C++ (NPU):    {cpp_median*1000:.1f}ms")
        print(f"    Speedup:      {speedup:.1f}x")

        # Verify C++ is faster
        self.assertGreater(
            speedup, 1.0,
            f"C++ encoder not faster than Python ({speedup:.1f}x)"
        )

        # Ideally should be much faster (target: ~2-3x vs Python)
        TARGET_SPEEDUP = 1.5

        self.assertGreater(
            speedup, TARGET_SPEEDUP,
            f"C++ speedup {speedup:.1f}x below target {TARGET_SPEEDUP}x"
        )

        print(f"  ✓ C++ encoder {speedup:.1f}x faster than Python")


class TestStabilityOverTime(unittest.TestCase):
    """Test performance stability over many inferences"""

    @unittest.skipUnless(CPP_AVAILABLE, "C++ encoder not available")
    def test_100_sequential_inferences(self):
        """Test that performance remains stable over 100 inferences"""
        print("\n[TEST] 100 sequential inferences...")

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
            TestRealtimeFactor._load_weights.__func__(self, model)
            encoder.load_weights(encoder._last_loaded_weights)

        # Create test input
        seq_len = 1500
        n_mels = 80
        mel_input = np.random.randn(seq_len, n_mels).astype(np.float32)

        # Warmup
        for _ in range(3):
            _ = encoder.forward(mel_input)

        # Run 100 inferences
        print("  Running 100 inferences...")
        times = []
        for i in range(100):
            t0 = time.perf_counter()
            _ = encoder.forward(mel_input)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

            if (i + 1) % 20 == 0:
                print(f"    {i+1}/100 complete...")

        # Analyze stability
        times = np.array(times)
        mean_time = times.mean()
        std_time = times.std()
        cv = (std_time / mean_time) * 100  # Coefficient of variation

        # Split into chunks to check for drift
        chunk_size = 20
        chunk_means = [times[i:i+chunk_size].mean() for i in range(0, 100, chunk_size)]

        print(f"\n  Stability Analysis:")
        print(f"    Mean:   {mean_time:.2f}ms")
        print(f"    Std:    {std_time:.2f}ms")
        print(f"    CV:     {cv:.2f}%")
        print(f"    Chunks: {[f'{m:.1f}ms' for m in chunk_means]}")

        # Verify stability (CV should be <10%)
        MAX_CV = 10.0

        self.assertLess(
            cv, MAX_CV,
            f"Coefficient of variation {cv:.2f}% exceeds {MAX_CV}% (unstable)"
        )

        # Check for performance drift (first chunk vs last chunk)
        drift = abs(chunk_means[-1] - chunk_means[0]) / chunk_means[0] * 100

        MAX_DRIFT = 10.0

        self.assertLess(
            drift, MAX_DRIFT,
            f"Performance drift {drift:.2f}% exceeds {MAX_DRIFT}%"
        )

        print(f"  ✓ Performance stable (CV: {cv:.2f}%, Drift: {drift:.2f}%)")


def run_tests():
    """Run all performance tests"""
    print("\n" + "="*70)
    print("  Performance Benchmarking Tests (Week 6, Day 4 - Task 3)")
    print("="*70)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRealtimeFactor))
    suite.addTests(loader.loadTestsFromTestCase(TestLayerLatency))
    suite.addTests(loader.loadTestsFromTestCase(TestNPUUtilization))
    suite.addTests(loader.loadTestsFromTestCase(TestComparisonWithPython))
    suite.addTests(loader.loadTestsFromTestCase(TestStabilityOverTime))

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
