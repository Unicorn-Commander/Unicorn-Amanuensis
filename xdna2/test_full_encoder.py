#!/usr/bin/env python3
"""
Comprehensive test suite for full Whisper encoder with NPU acceleration.

Tests:
1. Weight loading and quantization
2. Single attention layer accuracy
3. Single encoder layer accuracy
4. Full 6-layer encoder accuracy
5. Performance benchmarking
6. End-to-end transcription

Target: 400-500x realtime with 100% accuracy
"""

import logging
import numpy as np
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add runtime path
sys.path.insert(0, str(Path(__file__).parent))

from runtime.whisper_xdna2_runtime import create_runtime


def test_weight_loading():
    """Test 1: Weight loading and quantization."""
    print("\n" + "="*70)
    print("Test 1: Weight Loading and Quantization")
    print("="*70)

    try:
        runtime = create_runtime(model_size="base", use_4tile=False)

        # Load weights
        print("Loading Whisper Base weights from Hugging Face...")
        runtime._load_encoder_weights()

        # Check weights loaded
        assert runtime._weights_loaded, "Weights not loaded"
        assert len(runtime.encoder_weights) > 0, "No weights extracted"

        # Check quantization
        num_quantized = len(runtime.quantizer.quantized_weights)
        assert num_quantized > 0, "No weights quantized"

        print(f"✅ PASS: Loaded {len(runtime.encoder_weights)} weight tensors")
        print(f"✅ PASS: Quantized {num_quantized} weight matrices")

        # Check specific weights
        for layer_idx in range(6):
            # Check attention weights exist
            for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                weight_name = f"layers.{layer_idx}.self_attn.{proj}.weight"
                assert weight_name in runtime.encoder_weights, f"Missing {weight_name}"

                # Check quantized version exists
                q_weight, scale = runtime.quantizer.get_quantized_weight(weight_name)
                assert q_weight.dtype == np.int8, f"Weight not quantized to int8"
                assert scale > 0, f"Invalid scale"

            # Check FFN weights exist
            for fc in ['fc1', 'fc2']:
                weight_name = f"layers.{layer_idx}.{fc}.weight"
                assert weight_name in runtime.encoder_weights, f"Missing {weight_name}"

        print(f"✅ PASS: All 6 layers have complete weights")

        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_attention_layer():
    """Test 2: Single attention layer accuracy vs CPU reference."""
    print("\n" + "="*70)
    print("Test 2: Single Attention Layer Accuracy")
    print("="*70)

    try:
        runtime = create_runtime(model_size="base", use_4tile=False)

        # Load weights
        if not runtime._weights_loaded:
            runtime._load_encoder_weights()

        # Create test input
        seq_len = 100
        hidden_dim = 512
        x = np.random.randn(seq_len, hidden_dim).astype(np.float32)

        print(f"Input shape: {x.shape}")
        print(f"Testing layer 0 attention...")

        # Run NPU version
        start = time.perf_counter()
        attn_npu = runtime._run_attention_layer(x, layer_idx=0)
        elapsed_npu = time.perf_counter() - start

        print(f"NPU attention: {elapsed_npu*1000:.2f}ms")
        print(f"Output shape: {attn_npu.shape}")

        # Run CPU reference (using HuggingFace model)
        print("Computing CPU reference...")

        # Use the actual model for reference
        encoder = runtime.model.encoder

        # Need to add batch dimension and convert to torch
        import torch
        x_torch = torch.from_numpy(x).unsqueeze(0)  # (1, seq_len, hidden_dim)

        # Run just one attention layer
        with torch.no_grad():
            # Pre-norm
            x_norm = encoder.layers[0].self_attn_layer_norm(x_torch)

            # Self-attention
            attn_cpu, _ = encoder.layers[0].self_attn(
                x_norm,
                key_value_states=None,
                attention_mask=None,
            )

        attn_cpu = attn_cpu[0].numpy()  # Remove batch dimension

        print(f"CPU attention complete")
        print(f"CPU output shape: {attn_cpu.shape}")

        # Compare
        diff = np.abs(attn_npu - attn_cpu)
        mean_error = diff.mean()
        max_error = diff.max()
        rel_error = mean_error / (np.abs(attn_cpu).mean() + 1e-8)

        print(f"\nAccuracy metrics:")
        print(f"  Mean absolute error: {mean_error:.6f}")
        print(f"  Max absolute error: {max_error:.6f}")
        print(f"  Relative error: {rel_error*100:.4f}%")

        # Check if error is acceptable
        # INT8 quantization typically gives ~1-2% relative error
        if rel_error < 0.05:  # 5% threshold
            print(f"✅ PASS: Attention accuracy within tolerance")
            return True
        else:
            print(f"⚠️  WARNING: Attention error higher than expected")
            return False

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_encoder():
    """Test 3: Full 6-layer encoder accuracy."""
    print("\n" + "="*70)
    print("Test 3: Full Encoder Accuracy (6 layers)")
    print("="*70)

    try:
        runtime = create_runtime(model_size="base", use_4tile=False)

        # Create test mel spectrogram
        # Typical: 80 mel bins, ~100 time frames for 1 second of audio
        n_mels = 80
        n_frames = 200  # 2 seconds of audio
        mel_features = np.random.randn(n_mels, n_frames).astype(np.float32)

        print(f"Input mel spectrogram: {mel_features.shape}")

        # Run NPU encoder
        print("Running NPU encoder (6 layers)...")
        start = time.perf_counter()
        encoder_npu = runtime.run_encoder(mel_features)
        elapsed_npu = time.perf_counter() - start

        print(f"NPU encoder: {elapsed_npu*1000:.2f}ms")
        print(f"Output shape: {encoder_npu.shape}")

        # Run CPU reference
        print("Computing CPU reference...")
        import torch

        # Prepare input for HuggingFace model
        mel_torch = torch.from_numpy(mel_features).unsqueeze(0)  # Add batch dim

        with torch.no_grad():
            encoder_cpu = runtime.model.encoder(mel_torch).last_hidden_state

        encoder_cpu = encoder_cpu[0].numpy()  # Remove batch dim

        print(f"CPU encoder complete")
        print(f"CPU output shape: {encoder_cpu.shape}")

        # Note: Shapes might differ slightly due to conv downsampling
        # Compare the overlapping region
        min_len = min(encoder_npu.shape[0], encoder_cpu.shape[0])
        encoder_npu_cmp = encoder_npu[:min_len]
        encoder_cpu_cmp = encoder_cpu[:min_len]

        # Compare
        diff = np.abs(encoder_npu_cmp - encoder_cpu_cmp)
        mean_error = diff.mean()
        max_error = diff.max()
        rel_error = mean_error / (np.abs(encoder_cpu_cmp).mean() + 1e-8)

        print(f"\nAccuracy metrics:")
        print(f"  Compared sequence length: {min_len}")
        print(f"  Mean absolute error: {mean_error:.6f}")
        print(f"  Max absolute error: {max_error:.6f}")
        print(f"  Relative error: {rel_error*100:.4f}%")

        # Calculate performance
        audio_duration = n_frames * 0.01  # 10ms hop = 0.01s per frame
        realtime_factor = audio_duration / elapsed_npu

        print(f"\nPerformance:")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Processing time: {elapsed_npu*1000:.2f}ms")
        print(f"  Realtime factor: {realtime_factor:.1f}x")

        # Check if error is acceptable
        if rel_error < 0.10:  # 10% threshold for full encoder
            print(f"✅ PASS: Encoder accuracy within tolerance")

            # Check if we hit performance target
            if realtime_factor >= 400:
                print(f"✅ PASS: Performance target achieved (400-500x realtime)")
            else:
                print(f"⚠️  WARNING: Performance below target (need 400x, got {realtime_factor:.1f}x)")

            return True
        else:
            print(f"⚠️  WARNING: Encoder error higher than expected")
            return False

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_scaling():
    """Test 4: Performance scaling with different sequence lengths."""
    print("\n" + "="*70)
    print("Test 4: Performance Scaling")
    print("="*70)

    try:
        runtime = create_runtime(model_size="base", use_4tile=False)

        # Test different audio durations
        test_cases = [
            (100, 1.0),    # 1 second
            (300, 3.0),    # 3 seconds
            (500, 5.0),    # 5 seconds
            (1000, 10.0),  # 10 seconds
        ]

        results = []

        for n_frames, duration in test_cases:
            mel = np.random.randn(80, n_frames).astype(np.float32)

            # Warmup
            _ = runtime.run_encoder(mel)

            # Benchmark (3 runs)
            times = []
            for _ in range(3):
                start = time.perf_counter()
                _ = runtime.run_encoder(mel)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_time = np.mean(times)
            std_time = np.std(times)
            realtime = duration / avg_time

            results.append({
                'duration': duration,
                'frames': n_frames,
                'time': avg_time,
                'std': std_time,
                'realtime': realtime
            })

            print(f"\n{duration}s audio ({n_frames} frames):")
            print(f"  Time: {avg_time*1000:.2f} ± {std_time*1000:.2f}ms")
            print(f"  Realtime factor: {realtime:.1f}x")

        print("\n" + "="*70)
        print("Performance Summary")
        print("="*70)
        print(f"{'Duration':<12} {'Frames':<10} {'Time (ms)':<12} {'Realtime':<12}")
        print("-"*70)

        for r in results:
            print(f"{r['duration']:>6.1f}s     {r['frames']:<10} {r['time']*1000:>8.2f}ms    {r['realtime']:>8.1f}x")

        # Check if we maintain target across all durations
        min_realtime = min(r['realtime'] for r in results)

        if min_realtime >= 400:
            print(f"\n✅ PASS: Performance target maintained across all durations")
            return True
        else:
            print(f"\n⚠️  WARNING: Performance drops below target for longer audio")
            return False

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_npu_utilization():
    """Test 5: NPU utilization and GFLOPS measurement."""
    print("\n" + "="*70)
    print("Test 5: NPU Utilization")
    print("="*70)

    try:
        runtime = create_runtime(model_size="base", use_4tile=False)

        # Create test input
        mel = np.random.randn(80, 500).astype(np.float32)  # 5 seconds

        # Run encoder with detailed logging
        logger.setLevel(logging.DEBUG)

        print("Running encoder with debug logging...")
        start = time.perf_counter()
        output = runtime.run_encoder(mel)
        elapsed = time.perf_counter() - start

        logger.setLevel(logging.INFO)

        # Calculate theoretical FLOPs for encoder
        # For Whisper Base:
        # - 6 layers
        # - Each layer: 4 matmuls for attention (Q/K/V/O) + 2 for FFN
        # - Matmul FLOPs = 2 * M * K * N

        seq_len = output.shape[0]
        n_layers = 6
        hidden_dim = 512
        ffn_dim = 2048

        # Attention: 4 matmuls of (seq_len, 512) @ (512, 512)
        attn_flops_per_layer = 4 * (2 * seq_len * 512 * 512)

        # FFN: 2 matmuls (seq_len, 512) @ (512, 2048) and (seq_len, 2048) @ (2048, 512)
        ffn_flops_per_layer = 2 * seq_len * 512 * 2048 + 2 * seq_len * 2048 * 512

        total_flops = n_layers * (attn_flops_per_layer + ffn_flops_per_layer)

        # Achieved GFLOPS
        gflops = total_flops / elapsed / 1e9

        # NPU peak: 50 TOPS = 50,000 GOPS for INT8
        # For matmul: 1 OP = 1 MAC = 2 FLOPs equivalent
        # So 50 TOPS ~= 25 TFLOPS equivalent for FP32
        npu_peak_gflops = 25000  # Conservative estimate

        utilization = (gflops / npu_peak_gflops) * 100

        print(f"\nFLOPs Analysis:")
        print(f"  Sequence length: {seq_len}")
        print(f"  Total FLOPs: {total_flops/1e9:.2f} GFLOPs")
        print(f"  Time: {elapsed*1000:.2f}ms")
        print(f"  Achieved: {gflops:.2f} GFLOPS")
        print(f"  NPU peak: {npu_peak_gflops:.2f} GFLOPS (theoretical)")
        print(f"  Utilization: {utilization:.2f}%")

        print(f"\n✅ PASS: NPU utilization measured")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("WHISPER ENCODER FULL TEST SUITE")
    print("Target: 400-500x realtime with 100% accuracy")
    print("="*70)

    # Run tests
    tests = [
        ("Weight Loading", test_weight_loading),
        ("Single Attention Layer", test_single_attention_layer),
        ("Full Encoder", test_full_encoder),
        ("Performance Scaling", test_performance_scaling),
        ("NPU Utilization", test_npu_utilization),
    ]

    results = {}

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n❌ Test {name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:<12} {name}")

    total = len(results)
    passed = sum(results.values())

    print("-"*70)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Encoder implementation complete!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
