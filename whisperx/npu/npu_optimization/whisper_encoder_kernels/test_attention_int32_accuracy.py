#!/usr/bin/env python3
"""
INT32 Attention Accuracy Test
Measures correlation improvement from INT32 score precision fix
Target: ≥0.70 (from 0.123 baseline)
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

try:
    import xrt
except ImportError:
    print("❌ XRT not available - install XRT 2.20.0")
    sys.exit(1)

def test_int32_attention_accuracy():
    print("="*70)
    print("INT32 Attention Accuracy Test")
    print("="*70)
    print()

    # Configuration
    xclbin_path = Path("build_attention_int32/attention_int32.xclbin")

    if not xclbin_path.exists():
        print(f"❌ XCLBIN not found: {xclbin_path}")
        print("   Run XCLBIN generation first (Step 2)")
        return None

    # Load NPU device
    print("Loading NPU device...")
    try:
        device = xrt.xrt_device(0)
        device.load_xclbin(str(xclbin_path))
        print(f"✅ XCLBIN loaded: {xclbin_path}")
    except Exception as e:
        print(f"❌ Failed to load XCLBIN: {e}")
        return None

    print()

    # Generate test data
    print("Generating test data...")
    np.random.seed(42)
    Q = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
    K = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
    V = np.random.randint(-64, 64, (64, 64), dtype=np.int8)

    # Combine Q, K, V for NPU input (12288 bytes)
    QKV_combined = np.concatenate([Q.flatten(), K.flatten(), V.flatten()])

    print(f"  Q shape: {Q.shape}, range: [{Q.min()}, {Q.max()}]")
    print(f"  K shape: {K.shape}, range: [{K.min()}, {K.max()}]")
    print(f"  V shape: {V.shape}, range: [{V.min()}, {V.max()}]")
    print()

    # Reference computation (PyTorch FP32)
    print("Computing reference (PyTorch FP32)...")
    Q_f = torch.tensor(Q, dtype=torch.float32)
    K_f = torch.tensor(K, dtype=torch.float32)
    V_f = torch.tensor(V, dtype=torch.float32)

    # Attention: softmax(Q @ K^T / sqrt(64)) @ V
    scores = torch.matmul(Q_f, K_f.T) / 8.0  # sqrt(64) = 8
    attention_weights = F.softmax(scores, dim=-1)
    reference_output = torch.matmul(attention_weights, V_f)

    print(f"  Scores range: [{scores.min():.2f}, {scores.max():.2f}]")
    print(f"  Attention weights: [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")
    print(f"  Reference output: [{reference_output.min():.2f}, {reference_output.max():.2f}]")
    print()

    # NPU execution
    print("Executing on NPU...")
    try:
        # Allocate buffers
        input_bo = xrt.bo(device, QKV_combined.nbytes, xrt.bo.normal, 0)
        output_bo = xrt.bo(device, 4096, xrt.bo.normal, 0)  # 64×64 INT8

        # Copy input
        input_bo.write(QKV_combined, 0)
        input_bo.sync(xrt.xclDirection.HOST_TO_DEVICE, QKV_combined.nbytes, 0)

        # Get kernel
        kernel = xrt.kernel(device, device.get_xclbin_uuid(), "MLIR_AIE")

        # Execute
        run = kernel(input_bo, output_bo, 3)  # scale_shift = 3 (divide by 8)
        run.wait()

        # Read output
        output_bo.sync(xrt.xclDirection.DEVICE_TO_HOST, 4096, 0)
        npu_output = np.zeros(4096, dtype=np.int8)
        output_bo.read(npu_output, 0)
        npu_output = npu_output.reshape(64, 64)

        print(f"✅ NPU execution complete")
        print(f"  NPU output range: [{npu_output.min()}, {npu_output.max()}]")

    except Exception as e:
        print(f"❌ NPU execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    print()

    # Accuracy analysis
    print("="*70)
    print("ACCURACY RESULTS")
    print("="*70)
    print()

    # Correlation
    npu_flat = npu_output.flatten().astype(np.float32)
    ref_flat = reference_output.numpy().flatten()
    correlation = np.corrcoef(npu_flat, ref_flat)[0, 1]

    # Error metrics
    diff = npu_flat - ref_flat
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    max_error = np.max(np.abs(diff))

    # Statistics
    within_5 = np.sum(np.abs(diff) <= 5) / len(diff) * 100
    within_10 = np.sum(np.abs(diff) <= 10) / len(diff) * 100

    # Display results
    print(f"Correlation:        {correlation:.4f}")
    print(f"Target:             ≥0.70")
    print(f"Baseline (INT8):    0.123")
    print(f"Improvement:        {correlation / 0.123:.1f}× over baseline")
    print()
    print(f"MAE:                {mae:.2f}")
    print(f"RMSE:               {rmse:.2f}")
    print(f"Max Error:          {max_error:.2f}")
    print(f"Within ±5:          {within_5:.1f}%")
    print(f"Within ±10:         {within_10:.1f}%")
    print()

    # Pass/Fail
    if correlation >= 0.70:
        print("✅ PASS - Target correlation achieved!")
        print(f"   {correlation:.4f} ≥ 0.70 (target)")
    elif correlation >= 0.50:
        print("⚠️  PARTIAL - Significant improvement but below target")
        print(f"   {correlation:.4f} < 0.70 (target)")
    else:
        print("❌ FAIL - Correlation below target")
        print(f"   {correlation:.4f} < 0.70 (target)")

    print()
    print("="*70)

    return correlation

if __name__ == "__main__":
    try:
        correlation = test_int32_attention_accuracy()
        sys.exit(0 if correlation and correlation >= 0.70 else 1)
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
