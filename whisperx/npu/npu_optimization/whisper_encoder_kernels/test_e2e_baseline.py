#!/usr/bin/env python3
"""
End-to-End Baseline Test for NPU Whisper Encoder

Establishes baseline performance metrics for the current implementation.
Tests with synthetic mel spectrograms of various sizes.

AMD Phoenix NPU - XDNA1
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=" * 70)
    print("NPU Whisper Encoder - End-to-End Baseline Test")
    print("AMD Phoenix NPU - XDNA1")
    print("=" * 70)
    print()

    # Paths
    onnx_path = (
        "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/"
        "whisper_onnx_cache/models--onnx-community--whisper-base/"
        "onnx/encoder_model.onnx"
    )
    kernel_dir = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1"
    test_data_dir = Path(__file__).parent / "test_data"

    try:
        # Import encoder
        from npu_whisper_encoder import NPUWhisperEncoder

        print("Initializing NPU Whisper Encoder...")
        encoder = NPUWhisperEncoder(onnx_path, kernel_dir)
        print()

        # Test scenarios
        test_configs = [
            {"name": "Small (10 frames)", "seq_len": 10, "audio_sec": 0.1},
            {"name": "Medium (100 frames)", "seq_len": 100, "audio_sec": 1.0},
            {"name": "Standard (500 frames)", "seq_len": 500, "audio_sec": 5.0},
            {"name": "Full (1500 frames)", "seq_len": 1500, "audio_sec": 30.0},
        ]

        results = []

        print("=" * 70)
        print("BASELINE PERFORMANCE TESTS")
        print("=" * 70)
        print()

        for config in test_configs:
            print(f"\n{'-' * 60}")
            print(f"Test: {config['name']}")
            print(f"Sequence length: {config['seq_len']}")
            print(f"Equivalent audio: {config['audio_sec']}s")
            print(f"{'-' * 60}")

            # Create test input (simulated mel features projected to hidden dim)
            seq_len = config['seq_len']
            input_data = np.random.randn(seq_len, encoder.HIDDEN_DIM).astype(np.float32)

            # Warmup
            print("Warmup run...")
            _ = encoder.encode_layer(input_data[:min(10, seq_len)], layer_idx=0)

            # Single layer benchmark
            print("Benchmarking single layer...")
            layer_times = []
            num_runs = 3

            for i in range(num_runs):
                start = time.perf_counter()
                output = encoder.encode_layer(input_data, layer_idx=0)
                elapsed = time.perf_counter() - start
                layer_times.append(elapsed * 1000)  # ms

            avg_layer_ms = np.mean(layer_times)
            min_layer_ms = np.min(layer_times)

            # Full encoder benchmark (only for smaller sizes)
            if seq_len <= 500:
                print("Benchmarking full encoder (6 layers)...")
                full_times = []
                num_full_runs = 2

                for i in range(num_full_runs):
                    start = time.perf_counter()
                    full_output = encoder.encode(input_data)
                    elapsed = time.perf_counter() - start
                    full_times.append(elapsed * 1000)

                avg_full_ms = np.mean(full_times)
                min_full_ms = np.min(full_times)
            else:
                print("Skipping full encoder benchmark (too slow)")
                avg_full_ms = avg_layer_ms * 6  # Estimate
                min_full_ms = min_layer_ms * 6

            # Calculate metrics
            audio_sec = config['audio_sec']
            encoder_time_sec = avg_full_ms / 1000

            if encoder_time_sec > 0:
                rtf = audio_sec / encoder_time_sec
            else:
                rtf = float('inf')

            # Output info
            print(f"\nOutput shape: {output.shape if 'output' in dir() else '?'}")
            print(f"Output stats: mean={np.mean(output):.4f}, std={np.std(output):.4f}")

            # Results
            print(f"\nPerformance:")
            print(f"  Single layer: {avg_layer_ms:.2f} ms (min: {min_layer_ms:.2f})")
            print(f"  Full encoder: {avg_full_ms:.2f} ms (min: {min_full_ms:.2f})")
            print(f"  Realtime factor: {rtf:.1f}x")

            results.append({
                'name': config['name'],
                'seq_len': seq_len,
                'audio_sec': audio_sec,
                'layer_ms': avg_layer_ms,
                'encoder_ms': avg_full_ms,
                'rtf': rtf,
            })

        # Summary
        print("\n")
        print("=" * 70)
        print("BASELINE RESULTS SUMMARY")
        print("=" * 70)
        print()

        print(f"{'Test':<25} {'Seq':<8} {'Audio':<8} {'Encoder':<12} {'RTF':<10}")
        print("-" * 70)

        for r in results:
            print(f"{r['name']:<25} {r['seq_len']:<8} {r['audio_sec']:<8.1f}s {r['encoder_ms']:<12.1f}ms {r['rtf']:<10.1f}x")

        print()

        # Bottleneck analysis
        print("=" * 70)
        print("BOTTLENECK ANALYSIS")
        print("=" * 70)
        print()

        # Get timing breakdown from last run
        if hasattr(encoder, '_layer_times'):
            breakdown = encoder._layer_times
            total_time = sum(sum(times) for times in breakdown.values())

            print("Operation breakdown (single layer):")
            print("-" * 50)

            sorted_ops = sorted(
                [(k, sum(v)) for k, v in breakdown.items() if v],
                key=lambda x: x[1],
                reverse=True
            )

            for op, time_sum in sorted_ops:
                pct = (time_sum / total_time * 100) if total_time > 0 else 0
                avg_ms = (time_sum / len(breakdown[op]) * 1000) if breakdown[op] else 0
                bar = '#' * int(pct / 2)
                print(f"  {op:<25}: {avg_ms:7.2f} ms ({pct:5.1f}%) {bar}")

        print()

        # Recommendations
        print("=" * 70)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("=" * 70)
        print()

        # Check if we have significant bottlenecks
        if hasattr(encoder, '_layer_times') and encoder._layer_times:
            attention_time = sum(encoder._layer_times.get('attention', [0]))
            total = sum(sum(times) for times in encoder._layer_times.values())
            attn_pct = (attention_time / total * 100) if total > 0 else 0

            if attn_pct > 50:
                print(f"CRITICAL: Attention is {attn_pct:.1f}% of layer time")
                print("  - Per-row softmax calls are the bottleneck")
                print("  - Need batched softmax kernel for entire attention matrix")
                print("  - Also need CPU-to-NPU matmul for Q@K^T and attention@V")
                print()

        # Check for CPU fallback
        print("Current CPU fallbacks (need NPU kernels):")
        print("  - MatMul: All dimensions except 64x64 use CPU numpy")
        print("  - Q/K/V projections: (512, 512) use CPU")
        print("  - FFN: (512, 2048), (2048, 512) use CPU")
        print()

        # Target vs achieved
        target_rtf = 220.0
        best_rtf = max(r['rtf'] for r in results)
        gap = target_rtf / best_rtf

        print(f"Performance gap to 220x target: {gap:.1f}x slower")
        print()

        # Key optimizations needed
        print("Key optimizations to close gap:")
        print("  1. Batched softmax kernel (eliminate per-row calls)")
        print("  2. Larger tiled matmul (512x512 support)")
        print("  3. Fused LayerNorm with scale/bias")
        print("  4. DMA pipelining for continuous operation")
        print()

        print("=" * 70)
        print("END-TO-END BASELINE TEST COMPLETE")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
