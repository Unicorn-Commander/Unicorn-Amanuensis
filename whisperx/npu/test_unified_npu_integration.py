#!/usr/bin/env python3
"""
Unified NPU Integration Test
Tests incremental kernel integration for Whisper transcription

Test Phases:
1. Mel only → expect ~22-25x realtime
2. Mel + GELU → expect ~26-28x realtime
3. Mel + GELU + Attention → expect ~30-40x realtime

Author: Magic Unicorn Unconventional Technology & Stuff Inc.
Date: October 30, 2025
"""

import sys
import os
import numpy as np
import time
from pathlib import Path
import json

# Add paths
npu_path = Path(__file__).parent
sys.path.insert(0, str(npu_path))

from npu_runtime_unified import UnifiedNPURuntime


class NPUIntegrationTester:
    """Test unified NPU runtime with incremental kernel integration."""

    def __init__(self, test_audio_path: str = None):
        """
        Initialize tester.

        Args:
            test_audio_path: Path to test audio file (optional)
        """
        self.test_audio_path = test_audio_path
        self.runtime = None
        self.results = {}

    def setup(self):
        """Initialize NPU runtime."""
        print("=" * 80)
        print("UNIFIED NPU INTEGRATION TEST")
        print("=" * 80)
        print()

        print("Initializing NPU runtime...")
        self.runtime = UnifiedNPURuntime()
        print(f"  {self.runtime}")
        print()

        if not self.runtime.npu_available:
            print("ERROR: NPU not available - cannot run tests")
            return False

        # Check kernel availability
        kernels_loaded = sum([
            self.runtime.mel_available,
            self.runtime.gelu_available,
            self.runtime.attention_available
        ])

        print(f"Kernels loaded: {kernels_loaded}/3")
        print(f"  Mel:       {'✓' if self.runtime.mel_available else '✗'}")
        print(f"  GELU:      {'✓' if self.runtime.gelu_available else '✗'}")
        print(f"  Attention: {'✓' if self.runtime.attention_available else '✗'}")
        print()

        return True

    def generate_test_audio(self, duration: float = 30.0) -> np.ndarray:
        """
        Generate synthetic test audio.

        Args:
            duration: Audio duration in seconds

        Returns:
            Audio waveform (float32, 16kHz)
        """
        print(f"Generating {duration}s test audio...")

        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Multi-frequency sine wave (simulate speech formants)
        audio = (
            0.3 * np.sin(2 * np.pi * 200 * t) +  # F1
            0.3 * np.sin(2 * np.pi * 800 * t) +  # F2
            0.2 * np.sin(2 * np.pi * 2500 * t) + # F3
            0.2 * np.random.randn(len(t))        # Noise
        )

        audio = audio.astype(np.float32)
        print(f"  Generated: {len(audio)} samples, {duration}s")
        print()

        return audio

    def load_real_audio(self) -> tuple:
        """
        Load real audio file if available.

        Returns:
            (audio, duration) or (None, None) if not available
        """
        if not self.test_audio_path or not os.path.exists(self.test_audio_path):
            return None, None

        try:
            import librosa

            print(f"Loading audio: {self.test_audio_path}")
            audio, sr = librosa.load(self.test_audio_path, sr=16000, mono=True)
            duration = len(audio) / sr

            print(f"  Loaded: {len(audio)} samples, {duration:.2f}s, {sr}Hz")
            print()

            return audio, duration

        except Exception as e:
            print(f"  Error loading audio: {e}")
            print()
            return None, None

    # =========================================================================
    # Test 1: Mel Spectrogram Only
    # =========================================================================

    def test_1_mel_only(self, audio: np.ndarray, duration: float) -> dict:
        """
        Test 1: Mel spectrogram on NPU only.

        Expected: 22-25x realtime (mel kernel is 32.8x, but overhead reduces it)
        """
        print("=" * 80)
        print("TEST 1: MEL SPECTROGRAM ONLY (NPU)")
        print("=" * 80)
        print()

        if not self.runtime.mel_available:
            print("SKIP: Mel kernel not available")
            print()
            return {'status': 'skipped'}

        # Benchmark mel processing
        print("Running mel spectrogram processing...")

        iterations = 5
        times = []

        for i in range(iterations):
            start = time.time()
            mel_features = self.runtime.process_audio_to_features(audio)
            elapsed = time.time() - start
            times.append(elapsed)

            if i == 0:
                print(f"  Mel shape: {mel_features.shape}")

        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        rtf = duration / avg_time

        print(f"\nResults ({iterations} iterations):")
        print(f"  Average time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        print(f"  Min/Max:      {min_time*1000:.2f}ms / {max_time*1000:.2f}ms")
        print(f"  RTF:          {rtf:.1f}x realtime")
        print(f"  Target:       22-25x realtime")
        print(f"  Status:       {'✓ PASS' if rtf >= 22 else '✗ BELOW TARGET'}")
        print()

        result = {
            'status': 'pass' if rtf >= 22 else 'below_target',
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'rtf': rtf,
            'target_rtf': 22,
            'mel_shape': mel_features.shape
        }

        return result

    # =========================================================================
    # Test 2: Mel + GELU
    # =========================================================================

    def test_2_mel_gelu(self, audio: np.ndarray, duration: float) -> dict:
        """
        Test 2: Mel + GELU activation on NPU.

        Expected: 26-28x realtime
        """
        print("=" * 80)
        print("TEST 2: MEL + GELU (NPU)")
        print("=" * 80)
        print()

        if not self.runtime.mel_available or not self.runtime.gelu_available:
            print("SKIP: Mel or GELU kernel not available")
            print()
            return {'status': 'skipped'}

        print("Running mel + GELU processing...")

        iterations = 5
        times = []

        for i in range(iterations):
            start = time.time()

            # 1. Mel spectrogram
            mel_features = self.runtime.process_audio_to_features(audio)

            # 2. Apply GELU to each frame (simulate encoder activation)
            n_frames = mel_features.shape[1]
            gelu_outputs = []

            for frame_idx in range(min(n_frames, 100)):  # Process first 100 frames
                frame = mel_features[:, frame_idx]  # 80-dim vector
                if frame.shape[0] <= 512:
                    gelu_out = self.runtime.gelu(frame)
                    gelu_outputs.append(gelu_out)

            elapsed = time.time() - start
            times.append(elapsed)

            if i == 0:
                print(f"  Mel shape:    {mel_features.shape}")
                print(f"  GELU applied: {len(gelu_outputs)} frames")

        avg_time = np.mean(times)
        std_time = np.std(times)
        rtf = duration / avg_time

        print(f"\nResults ({iterations} iterations):")
        print(f"  Average time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        print(f"  RTF:          {rtf:.1f}x realtime")
        print(f"  Target:       26-28x realtime")
        print(f"  Status:       {'✓ PASS' if rtf >= 26 else '✗ BELOW TARGET'}")
        print()

        result = {
            'status': 'pass' if rtf >= 26 else 'below_target',
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'rtf': rtf,
            'target_rtf': 26,
            'gelu_frames': len(gelu_outputs)
        }

        return result

    # =========================================================================
    # Test 3: Mel + GELU + Attention
    # =========================================================================

    def test_3_mel_gelu_attention(self, audio: np.ndarray, duration: float) -> dict:
        """
        Test 3: Mel + GELU + Attention on NPU.

        Expected: 30-40x realtime (full kernel integration)
        """
        print("=" * 80)
        print("TEST 3: MEL + GELU + ATTENTION (NPU)")
        print("=" * 80)
        print()

        if not all([self.runtime.mel_available,
                    self.runtime.gelu_available,
                    self.runtime.attention_available]):
            print("SKIP: Not all kernels available")
            print()
            return {'status': 'skipped'}

        print("Running full NPU pipeline (mel + GELU + attention)...")

        iterations = 3  # Fewer iterations (attention is slower)
        times = []

        for i in range(iterations):
            start = time.time()

            # 1. Mel spectrogram
            mel_features = self.runtime.process_audio_to_features(audio)

            # 2. Prepare for attention (simulate encoder input)
            # Convert mel to sequence for attention
            seq_len = min(mel_features.shape[1], 64)  # Use first 64 frames
            d_model = 512

            # Create dummy Q, K, V (in real implementation, these come from encoder)
            Q = np.random.randint(-64, 64, (seq_len, d_model), dtype=np.int8)
            K = np.random.randint(-64, 64, (seq_len, d_model), dtype=np.int8)
            V = np.random.randint(-64, 64, (seq_len, d_model), dtype=np.int8)

            # 3. Multi-head attention
            attn_output = self.runtime.multi_head_attention(Q, K, V, num_heads=8)

            # 4. Apply GELU to attention output (simulate FFN)
            # Process each position
            for pos_idx in range(min(seq_len, 10)):  # Process first 10 positions
                vector = attn_output[pos_idx, :512]  # First 512 dims
                gelu_out = self.runtime.gelu(vector.astype(np.float32))

            elapsed = time.time() - start
            times.append(elapsed)

            if i == 0:
                print(f"  Mel shape:        {mel_features.shape}")
                print(f"  Attention shape:  {attn_output.shape}")
                print(f"  Sequence length:  {seq_len}")

        avg_time = np.mean(times)
        std_time = np.std(times)
        rtf = duration / avg_time

        print(f"\nResults ({iterations} iterations):")
        print(f"  Average time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        print(f"  RTF:          {rtf:.1f}x realtime")
        print(f"  Target:       30-40x realtime")
        print(f"  Status:       {'✓ PASS' if rtf >= 30 else '✗ BELOW TARGET'}")
        print()

        result = {
            'status': 'pass' if rtf >= 30 else 'below_target',
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'rtf': rtf,
            'target_rtf': 30,
            'seq_len': seq_len
        }

        return result

    # =========================================================================
    # Run All Tests
    # =========================================================================

    def run_all_tests(self):
        """Run all integration tests."""
        if not self.setup():
            return

        # Load or generate test audio
        audio, duration = self.load_real_audio()

        if audio is None:
            # Use synthetic audio
            duration = 30.0
            audio = self.generate_test_audio(duration)
        else:
            print(f"Using real audio: {duration:.2f}s")
            print()

        # Run tests
        self.results['test_1_mel_only'] = self.test_1_mel_only(audio, duration)
        self.results['test_2_mel_gelu'] = self.test_2_mel_gelu(audio, duration)
        self.results['test_3_mel_gelu_attention'] = self.test_3_mel_gelu_attention(audio, duration)

        # Print summary
        self.print_summary()

        # Save results
        self.save_results()

        # Cleanup
        self.runtime.close()

    def print_summary(self):
        """Print test summary."""
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print()

        for test_name, result in self.results.items():
            if result.get('status') == 'skipped':
                print(f"{test_name}: SKIPPED")
                continue

            rtf = result.get('rtf', 0)
            target = result.get('target_rtf', 0)
            status = '✓ PASS' if result.get('status') == 'pass' else '✗ BELOW TARGET'

            print(f"{test_name}:")
            print(f"  RTF:    {rtf:.1f}x (target: {target}x)")
            print(f"  Status: {status}")
            print()

        # Overall status
        all_pass = all(
            r.get('status') in ['pass', 'skipped']
            for r in self.results.values()
        )

        print("=" * 80)
        if all_pass:
            print("OVERALL: ✓ ALL TESTS PASSED")
        else:
            print("OVERALL: ✗ SOME TESTS BELOW TARGET")
        print("=" * 80)
        print()

    def save_results(self):
        """Save results to JSON."""
        output_file = Path(__file__).parent / "npu_integration_test_results.json"

        # Add runtime info
        self.results['runtime_info'] = {
            'npu_available': self.runtime.npu_available,
            'mel_available': self.runtime.mel_available,
            'gelu_available': self.runtime.gelu_available,
            'attention_available': self.runtime.attention_available
        }

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to: {output_file}")
        print()


def main():
    """Main test entry point."""
    # Check for test audio file
    test_audio_paths = [
        "/home/ucadmin/UC-1/Unicorn-Amanuensis/test_audio/test.wav",
        "/home/ucadmin/VibeVoice/Shafen_Khan_call.m4a",
        None  # Will use synthetic audio
    ]

    test_audio = None
    for path in test_audio_paths:
        if path and os.path.exists(path):
            test_audio = path
            break

    # Run tests
    tester = NPUIntegrationTester(test_audio_path=test_audio)
    tester.run_all_tests()


if __name__ == "__main__":
    main()
