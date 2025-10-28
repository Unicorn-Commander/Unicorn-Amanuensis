#!/usr/bin/env python3
"""
ONNX Runtime Execution Provider Performance Test
Tests different execution providers with Whisper ONNX models
"""

import onnxruntime as ort
import numpy as np
import time
from pathlib import Path

# Model paths
MODEL_DIR = Path("/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/onnx/")
ENCODER_INT8 = MODEL_DIR / "encoder_model_int8.onnx"
ENCODER_FP16 = MODEL_DIR / "encoder_model_fp16.onnx"
ENCODER_FP32 = MODEL_DIR / "encoder_model.onnx"
DECODER_INT8 = MODEL_DIR / "decoder_model_merged_int8.onnx"

# Available providers
AVAILABLE_PROVIDERS = ort.get_available_providers()

print("=" * 80)
print("ONNX Runtime Execution Provider Performance Test")
print("=" * 80)
print(f"ONNX Runtime Version: {ort.__version__}")
print(f"Available Providers: {AVAILABLE_PROVIDERS}")
print("=" * 80)

def create_encoder_input():
    """Create dummy mel spectrogram input for encoder"""
    # Whisper base encoder expects: (batch_size, n_mels=80, n_frames=3000)
    batch_size = 1
    n_mels = 80
    n_frames = 3000
    mel_spectrogram = np.random.randn(batch_size, n_mels, n_frames).astype(np.float32)
    return mel_spectrogram

def create_decoder_input():
    """Create dummy input for decoder"""
    batch_size = 1
    seq_len = 1
    # decoder_input_ids: (batch_size, seq_len)
    decoder_input_ids = np.array([[50258]], dtype=np.int64)  # Start token
    return decoder_input_ids

def test_model_loading(model_path, providers):
    """Test loading a model with specific providers"""
    print(f"\n{'='*80}")
    print(f"Testing: {model_path.name}")
    print(f"Provider: {providers[0]}")
    print(f"{'='*80}")

    try:
        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load model
        start_load = time.time()
        session = ort.InferenceSession(
            str(model_path),
            providers=providers,
            sess_options=sess_options
        )
        load_time = time.time() - start_load

        print(f"✓ Model loaded successfully in {load_time:.4f}s")
        print(f"  Provider used: {session.get_providers()}")

        # Get input/output info
        print(f"\n  Input Nodes:")
        for inp in session.get_inputs():
            print(f"    - {inp.name}: {inp.shape} ({inp.type})")

        print(f"\n  Output Nodes:")
        for out in session.get_outputs():
            print(f"    - {out.name}: {out.shape} ({out.type})")

        return session, load_time

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None, None

def benchmark_encoder(session, num_runs=10, warmup_runs=3):
    """Benchmark encoder inference"""
    if session is None:
        return None

    print(f"\n  Benchmarking inference (warmup={warmup_runs}, runs={num_runs})...")

    try:
        # Create input
        mel_input = create_encoder_input()
        input_name = session.get_inputs()[0].name

        # Warmup runs
        for _ in range(warmup_runs):
            _ = session.run(None, {input_name: mel_input})

        # Timed runs
        times = []
        for _ in range(num_runs):
            start = time.time()
            outputs = session.run(None, {input_name: mel_input})
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        print(f"  ✓ Inference Results:")
        print(f"    - Average: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        print(f"    - Min: {min_time*1000:.2f}ms")
        print(f"    - Max: {max_time*1000:.2f}ms")
        print(f"    - Output shape: {outputs[0].shape}")

        return {
            'avg_ms': avg_time * 1000,
            'std_ms': std_time * 1000,
            'min_ms': min_time * 1000,
            'max_ms': max_time * 1000,
            'output_shape': outputs[0].shape
        }

    except Exception as e:
        print(f"  ✗ Benchmark failed: {e}")
        return None

def benchmark_decoder(session, num_runs=10, warmup_runs=3):
    """Benchmark decoder inference"""
    if session is None:
        return None

    print(f"\n  Benchmarking inference (warmup={warmup_runs}, runs={num_runs})...")

    try:
        # Create inputs
        decoder_input_ids = create_decoder_input()

        # Get encoder hidden states (dummy)
        batch_size = 1
        seq_len = 1500  # 30s audio at 50Hz
        hidden_size = 512  # Whisper base
        encoder_hidden_states = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)

        inputs = {
            'input_ids': decoder_input_ids,
            'encoder_hidden_states': encoder_hidden_states
        }

        # Warmup runs
        for _ in range(warmup_runs):
            _ = session.run(None, inputs)

        # Timed runs
        times = []
        for _ in range(num_runs):
            start = time.time()
            outputs = session.run(None, inputs)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        print(f"  ✓ Inference Results:")
        print(f"    - Average: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        print(f"    - Min: {min_time*1000:.2f}ms")
        print(f"    - Max: {max_time*1000:.2f}ms")
        print(f"    - Output shape: {outputs[0].shape}")

        return {
            'avg_ms': avg_time * 1000,
            'std_ms': std_time * 1000,
            'min_ms': min_time * 1000,
            'max_ms': max_time * 1000,
            'output_shape': outputs[0].shape
        }

    except Exception as e:
        print(f"  ✗ Benchmark failed: {e}")
        return None

def main():
    """Main test execution"""
    results = {}

    # Test configurations
    test_configs = [
        # Encoder tests
        ("encoder_int8", ENCODER_INT8, "encoder"),
        ("encoder_fp16", ENCODER_FP16, "encoder"),
        ("encoder_fp32", ENCODER_FP32, "encoder"),
        # Decoder test
        ("decoder_int8", DECODER_INT8, "decoder"),
    ]

    # Test each provider
    for provider in AVAILABLE_PROVIDERS:
        provider_results = {}

        print(f"\n\n{'#'*80}")
        print(f"# TESTING PROVIDER: {provider}")
        print(f"{'#'*80}")

        for config_name, model_path, model_type in test_configs:
            if not model_path.exists():
                print(f"\n✗ Model not found: {model_path}")
                continue

            # Load model
            session, load_time = test_model_loading(model_path, [provider])

            if session is not None:
                # Benchmark
                if model_type == "encoder":
                    bench_results = benchmark_encoder(session, num_runs=20, warmup_runs=5)
                else:
                    bench_results = benchmark_decoder(session, num_runs=20, warmup_runs=5)

                if bench_results:
                    provider_results[config_name] = {
                        'load_time_s': load_time,
                        **bench_results
                    }

        results[provider] = provider_results

    # Summary
    print(f"\n\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")

    for provider, provider_results in results.items():
        print(f"\n{provider}:")
        for config_name, metrics in provider_results.items():
            print(f"  {config_name}:")
            print(f"    Load: {metrics['load_time_s']:.4f}s")
            print(f"    Inference: {metrics['avg_ms']:.2f}ms ± {metrics['std_ms']:.2f}ms")

    # Find best performer for encoder
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    best_encoder = None
    best_encoder_time = float('inf')

    for provider, provider_results in results.items():
        for config_name, metrics in provider_results.items():
            if 'encoder' in config_name:
                if metrics['avg_ms'] < best_encoder_time:
                    best_encoder_time = metrics['avg_ms']
                    best_encoder = (provider, config_name, metrics)

    if best_encoder:
        provider, config_name, metrics = best_encoder
        print(f"\n✓ Best Encoder Configuration:")
        print(f"  Provider: {provider}")
        print(f"  Model: {config_name}")
        print(f"  Performance: {metrics['avg_ms']:.2f}ms ± {metrics['std_ms']:.2f}ms")

        speedup_vs_cpu = None
        if 'CPUExecutionProvider' in results and config_name in results['CPUExecutionProvider']:
            cpu_time = results['CPUExecutionProvider'][config_name]['avg_ms']
            speedup_vs_cpu = cpu_time / metrics['avg_ms']
            print(f"  Speedup vs CPU: {speedup_vs_cpu:.2f}x")

    return results

if __name__ == "__main__":
    results = main()
