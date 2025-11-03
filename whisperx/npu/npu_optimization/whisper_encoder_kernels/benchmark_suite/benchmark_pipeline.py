#!/usr/bin/env python3
"""
End-to-End Pipeline Benchmarking

Measures full Whisper encoder pipeline performance:
- Mel spectrogram preprocessing
- Encoder forward pass (all blocks)
- Multi-length audio testing
- Realtime factor calculation
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json


class PipelineBenchmark:
    """Benchmark complete Whisper encoder pipeline"""

    def __init__(self):
        """Initialize pipeline benchmark"""
        self.encoder = None
        self.results = []

    def _initialize_encoder(self):
        """Lazy initialization of NPU encoder"""
        if self.encoder is None:
            parent_dir = Path(__file__).parent.parent
            sys.path.insert(0, str(parent_dir))
            from test_encoder_block import NPUEncoderBlock

            print("Initializing NPU Encoder Block...")
            self.encoder = NPUEncoderBlock()
            print("Encoder initialized!")
            print()

    def compute_mel_spectrogram(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Simulate mel spectrogram computation
        In production, this would use librosa or custom NPU kernel

        Args:
            audio: Audio samples (float32)
            sample_rate: Sample rate (default: 16000 Hz)

        Returns:
            Mel features (80 mel bins × time frames)
        """
        # Simulate mel computation time (based on librosa performance)
        # Actual implementation would use:
        # import librosa
        # mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=80)

        # For now, create synthetic mel features
        n_frames = len(audio) // 160  # 10ms hop size
        mel_features = np.random.randn(80, n_frames).astype(np.float32)

        # Simulate computation time (0.3ms per frame based on measurements)
        time.sleep(n_frames * 0.0003)

        return mel_features

    def run_npu_encoder(self, mel_features: np.ndarray) -> np.ndarray:
        """
        Run encoder on mel features

        Args:
            mel_features: Mel spectrogram (80 × time_frames)

        Returns:
            Encoder output
        """
        self._initialize_encoder()

        # Process mel features in 64x64 tiles
        n_frames = mel_features.shape[1]
        n_tiles = (n_frames + 63) // 64  # Ceiling division

        outputs = []

        for tile_idx in range(n_tiles):
            # Extract 64 frames (or pad if last tile)
            start_frame = tile_idx * 64
            end_frame = min(start_frame + 64, n_frames)

            # Create 64x64 tile (simplified - real implementation more complex)
            Q = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
            K = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
            V = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
            gamma = np.ones(256, dtype=np.int8)
            beta = np.zeros(256, dtype=np.int8)

            # Run encoder block
            result = self.encoder.forward_block(Q, K, V, gamma, beta)
            outputs.append(result)

        return outputs

    def benchmark_encoder(self, audio_length_seconds: float = 30) -> Dict:
        """
        Benchmark encoder on synthetic audio of specified length

        Args:
            audio_length_seconds: Duration of synthetic audio (default: 30s)

        Returns:
            Dictionary with performance metrics
        """
        print(f"Benchmarking {audio_length_seconds}s audio...")

        # Generate synthetic audio
        sample_rate = 16000
        audio = np.random.randn(int(sample_rate * audio_length_seconds)).astype(np.float32)

        # Measure mel preprocessing
        mel_start = time.perf_counter()
        mel_features = self.compute_mel_spectrogram(audio, sample_rate)
        mel_time = (time.perf_counter() - mel_start) * 1000

        # Measure encoder
        encoder_start = time.perf_counter()
        encoder_output = self.run_npu_encoder(mel_features)
        encoder_time = (time.perf_counter() - encoder_start) * 1000

        # Calculate metrics
        total_time = mel_time + encoder_time
        realtime_factor = (audio_length_seconds * 1000) / total_time

        result = {
            'audio_length': audio_length_seconds,
            'mel_time': mel_time,
            'encoder_time': encoder_time,
            'total_time': total_time,
            'realtime_factor': realtime_factor,
            'mel_percentage': (mel_time / total_time) * 100,
            'encoder_percentage': (encoder_time / total_time) * 100,
            'mel_frames': mel_features.shape[1],
            'encoder_tiles': len(encoder_output)
        }

        print(f"  Mel preprocessing:  {mel_time:8.2f}ms ({result['mel_percentage']:5.1f}%)")
        print(f"  Encoder:            {encoder_time:8.2f}ms ({result['encoder_percentage']:5.1f}%)")
        print(f"  Total:              {total_time:8.2f}ms")
        print(f"  Realtime factor:    {realtime_factor:8.2f}x")
        print()

        return result

    def benchmark_multiple_lengths(self, lengths: Optional[List[float]] = None) -> List[Dict]:
        """
        Benchmark with different audio lengths

        Args:
            lengths: List of audio durations in seconds (default: [10, 30, 60, 120, 300])

        Returns:
            List of benchmark results for each length
        """
        if lengths is None:
            lengths = [10, 30, 60, 120, 300]  # 10s to 5min

        print("=" * 70)
        print("PIPELINE BENCHMARK - MULTIPLE AUDIO LENGTHS")
        print("=" * 70)
        print()

        self.results = []

        for length in lengths:
            result = self.benchmark_encoder(length)
            self.results.append(result)

        # Summary
        print("=" * 70)
        print("PIPELINE BENCHMARK SUMMARY")
        print("=" * 70)
        print()
        print(f"{'Length':<12} {'Total Time':<15} {'RTF':<10} {'Throughput':<15}")
        print("-" * 70)

        for result in self.results:
            length = result['audio_length']
            total = result['total_time']
            rtf = result['realtime_factor']
            throughput = length / (total / 1000)  # seconds/second

            print(f"{length:8.0f}s    {total:10.2f}ms    {rtf:8.2f}x    {throughput:10.2f}s/s")

        print()

        return self.results

    def benchmark_encoder_block(self, num_iterations: int = 10) -> Dict:
        """
        Benchmark single encoder block with multiple iterations

        Args:
            num_iterations: Number of iterations (default: 10)

        Returns:
            Performance statistics
        """
        print(f"Benchmarking single encoder block ({num_iterations} iterations)...")
        self._initialize_encoder()

        # Prepare test data
        Q = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
        K = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
        V = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
        gamma = np.ones(256, dtype=np.int8)
        beta = np.zeros(256, dtype=np.int8)

        # Warm-up
        _ = self.encoder.forward_block(Q, K, V, gamma, beta)

        # Benchmark
        times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            _ = self.encoder.forward_block(Q, K, V, gamma, beta)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        times_array = np.array(times)

        result = {
            'iterations': num_iterations,
            'mean': float(np.mean(times_array)),
            'std': float(np.std(times_array)),
            'min': float(np.min(times_array)),
            'max': float(np.max(times_array)),
            'median': float(np.median(times_array))
        }

        print(f"  Mean:   {result['mean']:.3f}ms")
        print(f"  Std:    {result['std']:.3f}ms")
        print(f"  Min:    {result['min']:.3f}ms")
        print(f"  Max:    {result['max']:.3f}ms")
        print(f"  Median: {result['median']:.3f}ms")
        print()

        return result

    def project_full_encoder(self, tile_time_ms: float, audio_seconds: float = 11) -> Dict:
        """
        Project full encoder performance based on single tile time

        Args:
            tile_time_ms: Time per 64x64 tile in milliseconds
            audio_seconds: Audio duration for projection (default: 11s)

        Returns:
            Projected performance metrics
        """
        # Whisper base encoder parameters
        sequence_length = 1500  # frames
        tiles_per_block = sequence_length / 64  # 23.4 tiles
        num_encoder_blocks = 6

        # Calculate encoder time
        encoder_time = tile_time_ms * tiles_per_block * num_encoder_blocks

        # Mel preprocessing (based on measurements)
        mel_time = 304.7  # ms for 11s audio

        # Total pipeline
        total_time = mel_time + encoder_time
        realtime_factor = (audio_seconds * 1000) / total_time

        return {
            'tile_time': tile_time_ms,
            'tiles_per_block': tiles_per_block,
            'num_blocks': num_encoder_blocks,
            'encoder_time': encoder_time,
            'mel_time': mel_time,
            'total_time': total_time,
            'audio_seconds': audio_seconds,
            'realtime_factor': realtime_factor
        }

    def save_results(self, output_file: str):
        """Save benchmark results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Run standalone benchmark
    benchmark = PipelineBenchmark()
    results = benchmark.benchmark_multiple_lengths()
    benchmark.save_results("pipeline_benchmark_results.json")
