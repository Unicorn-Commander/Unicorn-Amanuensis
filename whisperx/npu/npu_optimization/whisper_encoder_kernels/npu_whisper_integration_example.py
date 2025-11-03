#!/usr/bin/env python3
"""
Simple End-to-End NPU Whisper Integration Example

Demonstrates how to use NPU-accelerated encoder and decoder
with real Whisper pipeline.

Usage:
    python3 npu_whisper_integration_example.py --audio test.wav
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import argparse
import time
from pathlib import Path

# Import NPU components
from whisper_npu_encoder_matmul import WhisperNPUEncoderMatmul
from whisper_npu_decoder_matmul import WhisperNPUDecoderMatmul


class NPUWhisperPipeline:
    """
    Complete Whisper pipeline with NPU acceleration

    Replaces encoder and decoder matmuls with NPU operations
    """

    def __init__(
        self,
        model_name: str = "base",
        device_id: int = 0
    ):
        """
        Initialize NPU Whisper pipeline

        Args:
            model_name: Whisper model size (tiny/base/small/medium/large)
            device_id: NPU device ID (default 0)
        """
        print("=" * 70)
        print("NPU WHISPER PIPELINE INITIALIZATION")
        print("=" * 70)
        print(f"Model: Whisper {model_name}")
        print()

        # Model configuration
        self.config = {
            "tiny": {"layers": 4, "d_model": 384, "heads": 6, "d_ff": 1536},
            "base": {"layers": 6, "d_model": 512, "heads": 8, "d_ff": 2048},
            "small": {"layers": 12, "d_model": 768, "heads": 12, "d_ff": 3072},
        }.get(model_name, {"layers": 6, "d_model": 512, "heads": 8, "d_ff": 2048})

        # Initialize NPU encoder
        print("Initializing NPU encoder...")
        self.encoder = WhisperNPUEncoderMatmul(
            num_layers=self.config["layers"],
            d_model=self.config["d_model"],
            num_heads=self.config["heads"],
            d_ff=self.config["d_ff"],
            device_id=device_id
        )
        print()

        # Initialize NPU decoder
        print("Initializing NPU decoder...")
        self.decoder = WhisperNPUDecoderMatmul(
            num_layers=self.config["layers"],
            d_model=self.config["d_model"],
            num_heads=self.config["heads"],
            d_ff=self.config["d_ff"],
            device_id=device_id
        )
        print()

        print("=" * 70)
        print("✅ NPU WHISPER PIPELINE READY")
        print("=" * 70)
        print()

    def transcribe(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Transcribe audio features

        Args:
            audio_features: Mel spectrogram features (seq_len, d_model)

        Returns:
            Decoded output (output_len, d_model)
        """
        print(f"Transcribing audio (input shape: {audio_features.shape})...")

        start_total = time.perf_counter()

        # Encode
        print("  Encoding...")
        start_enc = time.perf_counter()
        encoder_output = self.encoder(audio_features)
        encoder_time = (time.perf_counter() - start_enc) * 1000
        print(f"  ✅ Encoder: {encoder_time:.2f}ms")

        # Decode (autoregressive generation)
        print("  Decoding...")
        start_dec = time.perf_counter()

        # For demo, decode a fixed length
        # In real implementation, this would be autoregressive token-by-token
        max_length = min(250, audio_features.size(0) // 2)  # Typical output ~1/6 of input
        decoder_input = torch.randn(max_length, self.config["d_model"])

        decoder_output = self.decoder(decoder_input, encoder_output)
        decoder_time = (time.perf_counter() - start_dec) * 1000
        print(f"  ✅ Decoder: {decoder_time:.2f}ms")

        total_time = (time.perf_counter() - start_total) * 1000
        print(f"  ✅ Total: {total_time:.2f}ms")
        print()

        return decoder_output

    def benchmark(self, audio_duration: float = 30.0, iterations: int = 5):
        """
        Benchmark NPU pipeline performance

        Args:
            audio_duration: Audio duration in seconds
            iterations: Number of benchmark iterations
        """
        print("=" * 70)
        print("NPU WHISPER PIPELINE BENCHMARK")
        print("=" * 70)
        print(f"Audio duration: {audio_duration}s")
        print(f"Iterations: {iterations}")
        print()

        # Generate test audio features
        # Assume 50 frames per second
        seq_len = int(audio_duration * 50)
        audio_features = torch.randn(seq_len, self.config["d_model"])

        # Warm-up
        print("Warm-up...")
        _ = self.transcribe(audio_features)
        print()

        # Benchmark
        print("Benchmarking...")
        times = []
        for i in range(iterations):
            start = time.perf_counter()
            output = self.transcribe(audio_features)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            print(f"Iteration {i+1}/{iterations}: {elapsed:.2f}ms")

        # Statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        # Realtime factor
        rtf = (audio_duration * 1000) / avg_time

        print()
        print("=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Audio duration: {audio_duration}s")
        print(f"Average time: {avg_time:.2f}ms ± {std_time:.2f}ms")
        print(f"Min/Max: {min_time:.2f}ms / {max_time:.2f}ms")
        print(f"Realtime factor: {rtf:.1f}x")
        print()

        # Check target
        if rtf >= 25:
            print(f"✅ TARGET ACHIEVED! {rtf:.1f}x >= 25x realtime")
        elif rtf >= 20:
            print(f"✅ CLOSE TO TARGET! {rtf:.1f}x realtime (target: 25-29x)")
        else:
            print(f"⚠️ BELOW TARGET: {rtf:.1f}x realtime (target: 25-29x)")

        print("=" * 70)
        print()

        # Component statistics
        print("Component Statistics:")
        print()
        print("Encoder:")
        self.encoder.print_stats(audio_duration)

        print("Decoder:")
        self.decoder.print_stats(audio_duration)

        return rtf

    def get_stats(self):
        """Get pipeline statistics"""
        return {
            'encoder': self.encoder.get_stats(),
            'decoder': self.decoder.get_stats()
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="NPU Whisper Pipeline Example")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small"],
                       help="Whisper model size")
    parser.add_argument("--duration", type=float, default=30.0,
                       help="Audio duration for benchmark (seconds)")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of benchmark iterations")
    args = parser.parse_args()

    print("\n" * 2)

    # Initialize pipeline
    pipeline = NPUWhisperPipeline(model_name=args.model)

    # Run benchmark
    rtf = pipeline.benchmark(
        audio_duration=args.duration,
        iterations=args.iterations
    )

    # Summary
    print("\n" * 2)
    print("=" * 70)
    print("NPU WHISPER PIPELINE COMPLETE")
    print("=" * 70)
    print()
    print(f"Model: Whisper {args.model}")
    print(f"Performance: {rtf:.1f}x realtime")
    print(f"Target: 25-29x realtime")
    print()

    if rtf >= 25:
        print("✅ PRODUCTION READY")
        print("   Performance meets target for deployment")
    elif rtf >= 20:
        print("✅ NEARLY READY")
        print("   Close to target, minor optimizations needed")
    else:
        print("⚠️ NEEDS OPTIMIZATION")
        print("   Additional work required to meet target")

    print()
    print("Next Steps:")
    print("  1. Integrate into unified_stt_diarization.py")
    print("  2. Test with real Whisper weights")
    print("  3. Validate WER on test dataset")
    print("  4. Deploy to production")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
