#!/usr/bin/env python3
"""
Hardware validation of full 6-layer Whisper encoder on XDNA2 NPU.

Tests the complete encoder implementation on actual NPU hardware and measures:
- Single encoder layer latency
- Full 6-layer encoder latency
- Layer-by-layer performance breakdown
- Accuracy vs CPU reference
- Realtime factor (audio_duration / latency)

Target: 400-500x realtime (vs 220x on XDNA1 baseline)
"""

import numpy as np
import time
import sys
import logging
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/opt/xilinx/xrt/python")

from runtime.whisper_xdna2_runtime import create_runtime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_single_layer():
    """
    Test single encoder layer on NPU.

    Returns:
        Latency in milliseconds
    """
    print_section_header("TEST 1: Single Encoder Layer")

    print("\nInitializing runtime...")
    runtime = create_runtime(model_size="base", use_4tile=True)

    print("Loading encoder weights...")
    runtime._load_encoder_weights()

    # Test input: [seq_len=512, d_model=512]
    # Note: Real Whisper has variable seq_len, we use 512 for max case
    np.random.seed(42)
    hidden_states = np.random.randn(512, 512).astype(np.float32)

    print(f"Input shape: {hidden_states.shape}")
    print(f"Input range: [{hidden_states.min():.3f}, {hidden_states.max():.3f}]")

    # Warmup run
    print("\nWarmup run...")
    _ = runtime._run_encoder_layer(hidden_states, layer_idx=0)

    # Timed runs (average of 3)
    print("Timed runs (3 iterations)...")
    latencies = []
    for i in range(3):
        start = time.perf_counter()
        output = runtime._run_encoder_layer(hidden_states, layer_idx=0)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        print(f"  Run {i+1}: {latency:.2f} ms")

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    print(f"\nResults:")
    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  Std deviation: {std_latency:.2f} ms")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    return avg_latency


def test_full_encoder():
    """
    Test full 6-layer encoder on NPU.

    Returns:
        Tuple of (latency_ms, realtime_factor)
    """
    print_section_header("TEST 2: Full 6-Layer Encoder")

    print("\nInitializing runtime...")
    runtime = create_runtime(model_size="base", use_4tile=True)

    print("Loading encoder weights...")
    runtime._load_encoder_weights()

    # Test input: [seq_len=512, d_model=512]
    np.random.seed(42)
    hidden_states = np.random.randn(512, 512).astype(np.float32)

    print(f"Input shape: {hidden_states.shape}")

    # Warmup run
    print("\nWarmup run...")
    _ = runtime._run_encoder(hidden_states)

    # Timed runs (average of 3)
    print("Timed runs (3 iterations)...")
    latencies = []
    for i in range(3):
        start = time.perf_counter()
        output = runtime._run_encoder(hidden_states)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        print(f"  Run {i+1}: {latency:.2f} ms")

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    # Calculate realtime factor
    # 512 frames @ 20ms/frame = 10.24s of audio
    # (Whisper uses 25ms window + 10ms hop = ~20ms/frame effective)
    audio_duration = 10.24
    realtime_factor = (audio_duration * 1000) / avg_latency

    print(f"\nResults:")
    print(f"  Average latency: {avg_latency:.2f} ms ({avg_latency/1000:.3f} seconds)")
    print(f"  Std deviation: {std_latency:.2f} ms")
    print(f"  Audio duration: {audio_duration:.2f} seconds")
    print(f"  Realtime factor: {realtime_factor:.2f}x")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    return avg_latency, realtime_factor


def test_layer_by_layer():
    """
    Profile each encoder layer individually.

    Returns:
        List of latencies for each layer
    """
    print_section_header("TEST 3: Layer-by-Layer Profiling")

    print("\nInitializing runtime...")
    runtime = create_runtime(model_size="base", use_4tile=True)

    print("Loading encoder weights...")
    runtime._load_encoder_weights()

    # Test input
    np.random.seed(42)
    hidden_states = np.random.randn(512, 512).astype(np.float32)

    layer_latencies = []

    print("\nProfiling each layer (2 runs each, averaged):")
    for layer_idx in range(6):
        # Warmup
        _ = runtime._run_encoder_layer(hidden_states, layer_idx=layer_idx)

        # Timed runs
        runs = []
        for _ in range(2):
            start = time.perf_counter()
            output = runtime._run_encoder_layer(hidden_states, layer_idx=layer_idx)
            latency = (time.perf_counter() - start) * 1000
            runs.append(latency)

        avg = np.mean(runs)
        layer_latencies.append(avg)
        print(f"  Layer {layer_idx}: {avg:.2f} ms (runs: {runs[0]:.2f}, {runs[1]:.2f})")

        # Use output as input for next layer (simulates actual flow)
        hidden_states = output

    total = sum(layer_latencies)
    avg_per_layer = total / 6

    print(f"\nSummary:")
    print(f"  Total: {total:.2f} ms")
    print(f"  Average per layer: {avg_per_layer:.2f} ms")
    print(f"  Min layer: {min(layer_latencies):.2f} ms")
    print(f"  Max layer: {max(layer_latencies):.2f} ms")

    return layer_latencies


def _run_encoder_cpu(runtime, hidden_states: np.ndarray) -> np.ndarray:
    """
    Run encoder on CPU for accuracy comparison.

    Uses FP32 operations for reference implementation.

    Args:
        runtime: WhisperXDNA2Runtime instance
        hidden_states: Input (seq_len, d_model)

    Returns:
        Encoder output (seq_len, d_model)
    """
    dims = runtime.model_dims[runtime.model_size]
    n_state = dims["n_state"]
    n_head = dims["n_head"]
    head_dim = n_state // n_head

    x = hidden_states.copy()

    # Run 6 layers
    for layer_idx in range(6):
        # Get FP32 weights
        q_weight = runtime.encoder_weights[f"layers.{layer_idx}.self_attn.q_proj.weight"]
        k_weight = runtime.encoder_weights[f"layers.{layer_idx}.self_attn.k_proj.weight"]
        v_weight = runtime.encoder_weights[f"layers.{layer_idx}.self_attn.v_proj.weight"]
        out_weight = runtime.encoder_weights[f"layers.{layer_idx}.self_attn.out_proj.weight"]

        q_bias = runtime.encoder_weights[f"layers.{layer_idx}.self_attn.q_proj.bias"]
        k_bias = runtime.encoder_weights[f"layers.{layer_idx}.self_attn.k_proj.bias"]
        v_bias = runtime.encoder_weights[f"layers.{layer_idx}.self_attn.v_proj.bias"]
        out_bias = runtime.encoder_weights[f"layers.{layer_idx}.self_attn.out_proj.bias"]

        fc1_weight = runtime.encoder_weights[f"layers.{layer_idx}.fc1.weight"]
        fc1_bias = runtime.encoder_weights[f"layers.{layer_idx}.fc1.bias"]
        fc2_weight = runtime.encoder_weights[f"layers.{layer_idx}.fc2.weight"]
        fc2_bias = runtime.encoder_weights[f"layers.{layer_idx}.fc2.bias"]

        attn_ln_weight = runtime.encoder_weights[f"layers.{layer_idx}.self_attn_layer_norm.weight"]
        attn_ln_bias = runtime.encoder_weights[f"layers.{layer_idx}.self_attn_layer_norm.bias"]
        ffn_ln_weight = runtime.encoder_weights[f"layers.{layer_idx}.final_layer_norm.weight"]
        ffn_ln_bias = runtime.encoder_weights[f"layers.{layer_idx}.final_layer_norm.bias"]

        # 1. Self-attention block
        x_norm = runtime._layer_norm(x, attn_ln_weight, attn_ln_bias)

        # Q/K/V projections (using transposed weights)
        Q = x_norm @ q_weight.T + q_bias
        K = x_norm @ k_weight.T + k_bias
        V = x_norm @ v_weight.T + v_bias

        # Reshape for multi-head
        seq_len = x.shape[0]
        Q = Q.reshape(seq_len, n_head, head_dim).transpose(1, 0, 2)
        K = K.reshape(seq_len, n_head, head_dim).transpose(1, 0, 2)
        V = V.reshape(seq_len, n_head, head_dim).transpose(1, 0, 2)

        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(head_dim)
        attn_outputs = []
        for h in range(n_head):
            scores = (Q[h] @ K[h].T) * scale
            attn_weights = runtime._softmax(scores, axis=-1)
            head_out = attn_weights @ V[h]
            attn_outputs.append(head_out)

        # Concatenate heads
        attn_output = np.stack(attn_outputs, axis=0).transpose(1, 0, 2)
        attn_output = attn_output.reshape(seq_len, n_state)

        # Output projection
        attn_output = attn_output @ out_weight.T + out_bias

        # Residual
        x = x + attn_output

        # 2. Feed-forward block
        x_norm = runtime._layer_norm(x, ffn_ln_weight, ffn_ln_bias)

        # FFN
        ffn_out = x_norm @ fc1_weight.T + fc1_bias
        ffn_out = runtime._gelu(ffn_out)
        ffn_out = ffn_out @ fc2_weight.T + fc2_bias

        # Residual
        x = x + ffn_out

    # Final layer norm
    final_ln_weight = runtime.encoder_weights["layer_norm.weight"]
    final_ln_bias = runtime.encoder_weights["layer_norm.bias"]
    x = runtime._layer_norm(x, final_ln_weight, final_ln_bias)

    return x


def test_accuracy():
    """
    Validate accuracy vs CPU reference.

    Returns:
        True if accuracy is within tolerance, False otherwise
    """
    print_section_header("TEST 4: Accuracy Validation")

    print("\nInitializing runtime...")
    runtime = create_runtime(model_size="base", use_4tile=True)

    print("Loading encoder weights...")
    runtime._load_encoder_weights()

    # Use same random seed for reproducibility
    np.random.seed(42)
    hidden_states = np.random.randn(512, 512).astype(np.float32)

    print("Running NPU encoder...")
    output_npu = runtime._run_encoder(hidden_states)

    print("Running CPU reference...")
    output_cpu = _run_encoder_cpu(runtime, hidden_states)

    # Compare
    diff = output_npu - output_cpu
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    max_diff = np.max(np.abs(diff))

    # Relative error
    rel_error = mae / (np.abs(output_cpu).mean() + 1e-8)

    print(f"\nAccuracy Metrics:")
    print(f"  MSE (Mean Squared Error): {mse:.6f}")
    print(f"  MAE (Mean Absolute Error): {mae:.6f}")
    print(f"  Max absolute diff: {max_diff:.6f}")
    print(f"  Relative error: {rel_error*100:.3f}%")

    print(f"\nOutput Statistics:")
    print(f"  NPU range: [{output_npu.min():.3f}, {output_npu.max():.3f}]")
    print(f"  CPU range: [{output_cpu.min():.3f}, {output_cpu.max():.3f}]")
    print(f"  NPU mean/std: {output_npu.mean():.3f} / {output_npu.std():.3f}")
    print(f"  CPU mean/std: {output_cpu.mean():.3f} / {output_cpu.std():.3f}")

    # Check tolerance
    # For INT8 quantization, <2% error is excellent
    tolerance = 0.02
    passed = rel_error < tolerance

    if passed:
        print(f"\n  PASS: Relative error {rel_error*100:.3f}% < {tolerance*100}%")
    else:
        print(f"\n  FAIL: Relative error {rel_error*100:.3f}% >= {tolerance*100}%")

    return passed


def test_operation_breakdown():
    """
    Break down performance by operation type.

    Returns:
        Dictionary with timing breakdown
    """
    print_section_header("TEST 5: Operation Breakdown")

    print("\nInitializing runtime...")
    runtime = create_runtime(model_size="base", use_4tile=True)

    print("Loading encoder weights...")
    runtime._load_encoder_weights()

    np.random.seed(42)
    hidden_states = np.random.randn(512, 512).astype(np.float32)

    # Time components of a single layer
    layer_idx = 0

    # Warmup
    _ = runtime._run_encoder_layer(hidden_states, layer_idx=layer_idx)

    print("\nTiming individual operations (layer 0, averaged over 3 runs):")

    # 1. Attention
    attn_times = []
    for _ in range(3):
        x_norm = runtime._layer_norm(
            hidden_states,
            runtime.encoder_weights[f"layers.{layer_idx}.self_attn_layer_norm.weight"],
            runtime.encoder_weights[f"layers.{layer_idx}.self_attn_layer_norm.bias"]
        )
        start = time.perf_counter()
        _ = runtime._run_attention_layer(x_norm, layer_idx)
        attn_times.append((time.perf_counter() - start) * 1000)

    # 2. FFN
    ffn_times = []
    for _ in range(3):
        x_norm = runtime._layer_norm(
            hidden_states,
            runtime.encoder_weights[f"layers.{layer_idx}.final_layer_norm.weight"],
            runtime.encoder_weights[f"layers.{layer_idx}.final_layer_norm.bias"]
        )
        start = time.perf_counter()
        _ = runtime._run_ffn_layer(x_norm, layer_idx)
        ffn_times.append((time.perf_counter() - start) * 1000)

    attn_avg = np.mean(attn_times)
    ffn_avg = np.mean(ffn_times)
    total = attn_avg + ffn_avg

    print(f"  Attention: {attn_avg:.2f} ms ({attn_avg/total*100:.1f}%)")
    print(f"  FFN: {ffn_avg:.2f} ms ({ffn_avg/total*100:.1f}%)")
    print(f"  Total: {total:.2f} ms")

    return {
        'attention_ms': attn_avg,
        'ffn_ms': ffn_avg,
        'total_ms': total
    }


def main():
    """Main test harness."""
    print("=" * 70)
    print("  WHISPER ENCODER HARDWARE VALIDATION")
    print("  XDNA2 NPU - Full 6-Layer Test")
    print("=" * 70)
    print(f"\nTest Configuration:")
    print(f"  Model: Whisper Base (6 layers)")
    print(f"  Sequence length: 512 tokens")
    print(f"  Hidden dimension: 512")
    print(f"  Kernel: 4-tile INT8 matmul")
    print(f"  Quantization: Symmetric INT8")

    try:
        # Test 1: Single layer
        single_latency = test_single_layer()

        # Test 2: Full encoder
        full_latency, realtime_factor = test_full_encoder()

        # Test 3: Layer profiling
        layer_latencies = test_layer_by_layer()

        # Test 4: Accuracy
        accuracy_pass = test_accuracy()

        # Test 5: Operation breakdown
        op_breakdown = test_operation_breakdown()

        # Summary
        print_section_header("VALIDATION SUMMARY")

        print(f"\nPerformance Results:")
        print(f"  Single layer: {single_latency:.2f} ms")
        print(f"  Full encoder (6 layers): {full_latency:.2f} ms")
        print(f"  Realtime factor: {realtime_factor:.2f}x")
        print(f"  Per-layer average: {full_latency/6:.2f} ms")

        print(f"\nAccuracy Results:")
        if accuracy_pass:
            print(f"  Status: PASS")
        else:
            print(f"  Status: FAIL")

        print(f"\nOperation Breakdown:")
        print(f"  Attention: {op_breakdown['attention_ms']:.2f} ms")
        print(f"  FFN: {op_breakdown['ffn_ms']:.2f} ms")

        print_section_header("COMPARISON TO BASELINE")

        baseline_rtf = 220.0
        print(f"\nXDNA1 baseline (Unicorn-Amanuensis): {baseline_rtf:.0f}x realtime")
        print(f"Current result (XDNA2, this test): {realtime_factor:.2f}x realtime")

        if realtime_factor >= baseline_rtf:
            improvement = realtime_factor - baseline_rtf
            print(f"\nEXCEEDS baseline by {improvement:.2f}x!")
        else:
            gap = baseline_rtf - realtime_factor
            pct_of_baseline = (realtime_factor / baseline_rtf) * 100
            print(f"\nBelow baseline by {gap:.2f}x ({pct_of_baseline:.1f}% of baseline)")
            print(f"(Expected - Phase 4-5 optimizations will close gap)")

        target_rtf = 450.0
        if realtime_factor >= target_rtf:
            print(f"\nTARGET ACHIEVED: {realtime_factor:.2f}x >= {target_rtf:.0f}x!")
        else:
            gap_to_target = target_rtf - realtime_factor
            speedup_needed = target_rtf / realtime_factor
            print(f"\nTarget: {target_rtf:.0f}x realtime")
            print(f"Gap: {gap_to_target:.2f}x ({speedup_needed:.2f}x speedup needed)")

        print("\n" + "=" * 70)

        return {
            'single_latency_ms': single_latency,
            'full_latency_ms': full_latency,
            'realtime_factor': realtime_factor,
            'layer_latencies_ms': layer_latencies,
            'accuracy_pass': accuracy_pass,
            'op_breakdown': op_breakdown
        }

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    results = main()

    # Exit code: 0 if accuracy passed, 1 otherwise
    sys.exit(0 if results['accuracy_pass'] else 1)
