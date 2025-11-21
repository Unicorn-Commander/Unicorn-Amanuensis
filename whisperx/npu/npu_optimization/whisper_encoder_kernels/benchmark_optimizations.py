#!/usr/bin/env python3
"""
Benchmark Different Optimization Strategies
Compare v1 (sequential) vs v2 (chunked) vs CPU-only
"""

import numpy as np
import time
from whisper_encoder_optimized import WhisperEncoderOptimized
from whisper_encoder_optimized_v2 import WhisperEncoderOptimizedV2

def benchmark_encoder(encoder, mel_features, name):
    """Benchmark encoder performance"""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {name}")
    print(f"{'='*70}")

    # Warm-up run
    print("  Warming up...")
    _ = encoder.forward(mel_features, verbose=False)

    # Timed run
    print("  Running timed benchmark...")
    start = time.time()
    output = encoder.forward(mel_features, verbose=True)
    elapsed = time.time() - start

    print(f"\n📊 Results for {name}:")
    print(f"   Total time: {elapsed:.3f}s ({elapsed*1000:.1f}ms)")
    print(f"   Output shape: {output.shape}")
    print(f"   Output stats: mean={np.mean(output):.6f}, std={np.std(output):.6f}")

    return elapsed, output

def main():
    print("="*70)
    print("Optimization Benchmark - Phoenix NPU")
    print("="*70)

    # Test configurations
    test_sizes = [
        (100, "Small - 100 frames"),
        (500, "Medium - 500 frames"),
        (1000, "Large - 1000 frames"),
        (3001, "Full - 3001 frames (30s audio)"),
    ]

    results = {}

    for seq_len, description in test_sizes:
        print(f"\n\n{'#'*70}")
        print(f"Test Case: {description}")
        print(f"{'#'*70}")

        # Create test input
        mel_features = np.random.randn(seq_len, 80).astype(np.float32) * 0.1

        # Test v1 (sequential NPU calls - only for smaller sizes)
        if seq_len <= 500:
            try:
                print(f"\n--- Testing v1 (Sequential NPU) ---")
                encoder_v1 = WhisperEncoderOptimized(model_size="base", device_id=0)
                time_v1, output_v1 = benchmark_encoder(encoder_v1, mel_features, "v1-Sequential")
                results[(seq_len, "v1")] = time_v1
            except Exception as e:
                print(f"   ⚠️  v1 failed: {e}")
                results[(seq_len, "v1")] = None
        else:
            print(f"\n--- Skipping v1 for {seq_len} frames (too slow) ---")
            results[(seq_len, "v1")] = None

        # Test v2 (chunked with NPU)
        try:
            print(f"\n--- Testing v2 (Chunked NPU) ---")
            encoder_v2_npu = WhisperEncoderOptimizedV2(model_size="base", device_id=0)
            time_v2_npu, output_v2_npu = benchmark_encoder(encoder_v2_npu, mel_features, "v2-Chunked-NPU")
            results[(seq_len, "v2-npu")] = time_v2_npu
        except Exception as e:
            print(f"   ⚠️  v2 NPU failed: {e}")
            results[(seq_len, "v2-npu")] = None

        # Test v2 (chunked with CPU fallback - disable NPU)
        try:
            print(f"\n--- Testing v2 (Chunked CPU) ---")
            encoder_v2_cpu = WhisperEncoderOptimizedV2(model_size="base", device_id=0)
            encoder_v2_cpu.use_ln_npu = False  # Force CPU fallback
            time_v2_cpu, output_v2_cpu = benchmark_encoder(encoder_v2_cpu, mel_features, "v2-Chunked-CPU")
            results[(seq_len, "v2-cpu")] = time_v2_cpu
        except Exception as e:
            print(f"   ⚠️  v2 CPU failed: {e}")
            results[(seq_len, "v2-cpu")] = None

    # Print summary
    print(f"\n\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Frames':<15} {'v1 (seq)':<15} {'v2 (NPU)':<15} {'v2 (CPU)':<15} {'Best':<15}")
    print("-"*75)

    for seq_len, description in test_sizes:
        v1_time = results.get((seq_len, "v1"))
        v2_npu_time = results.get((seq_len, "v2-npu"))
        v2_cpu_time = results.get((seq_len, "v2-cpu"))

        v1_str = f"{v1_time:.3f}s" if v1_time else "N/A"
        v2_npu_str = f"{v2_npu_time:.3f}s" if v2_npu_time else "N/A"
        v2_cpu_str = f"{v2_cpu_time:.3f}s" if v2_cpu_time else "N/A"

        # Find best time
        times = [t for t in [v1_time, v2_npu_time, v2_cpu_time] if t is not None]
        if times:
            best_time = min(times)
            if best_time == v1_time:
                best_str = "v1 (seq)"
            elif best_time == v2_npu_time:
                best_str = "v2 (NPU)"
            else:
                best_str = "v2 (CPU)"
        else:
            best_str = "N/A"

        print(f"{seq_len:<15} {v1_str:<15} {v2_npu_str:<15} {v2_cpu_str:<15} {best_str:<15}")

    # Calculate speedups
    print(f"\n{'='*70}")
    print("SPEEDUP ANALYSIS")
    print(f"{'='*70}\n")

    for seq_len, description in test_sizes:
        v1_time = results.get((seq_len, "v1"))
        v2_npu_time = results.get((seq_len, "v2-npu"))
        v2_cpu_time = results.get((seq_len, "v2-cpu"))

        print(f"\n{description}:")

        if v1_time and v2_npu_time:
            speedup_npu = v1_time / v2_npu_time
            print(f"   v2-NPU vs v1:  {speedup_npu:.2f}x speedup")

        if v1_time and v2_cpu_time:
            speedup_cpu = v1_time / v2_cpu_time
            print(f"   v2-CPU vs v1:  {speedup_cpu:.2f}x speedup")

        if v2_npu_time and v2_cpu_time:
            ratio = v2_npu_time / v2_cpu_time
            if ratio > 1:
                print(f"   CPU is {ratio:.2f}x faster than NPU! ⚠️")
            else:
                print(f"   NPU is {1/ratio:.2f}x faster than CPU")

    # Realtime factor analysis (for 3001 frames = 30s audio)
    if results.get((3001, "v2-npu")) or results.get((3001, "v2-cpu")):
        print(f"\n{'='*70}")
        print("REALTIME FACTOR (30 seconds of audio)")
        print(f"{'='*70}\n")

        audio_duration = 5.0  # seconds (3001 frames = 5 seconds for our test)

        for variant in ["v1", "v2-npu", "v2-cpu"]:
            time_val = results.get((3001, variant))
            if time_val:
                rtf = audio_duration / time_val
                print(f"   {variant:<15}: {rtf:.2f}x realtime")

    print("\n" + "="*70)
    print("✅ Benchmark complete!")
    print("="*70)

if __name__ == "__main__":
    main()
