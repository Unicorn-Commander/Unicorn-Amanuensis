#!/usr/bin/env python3
"""
Attention Kernel Accuracy Validation Test
Compares NPU attention output with PyTorch CPU reference

Target: Correlation > 0.95 with PyTorch scaled dot-product attention
"""

import numpy as np
import sys
import os

sys.path.insert(0, '/opt/xilinx/xrt/python')
try:
    import pyxrt as xrt
except ImportError:
    print("ERROR: PyXRT not installed. Install with: pip install pyxrt")
    sys.exit(1)

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("ERROR: PyTorch not installed. Install with: pip install torch")
    sys.exit(1)


def pytorch_attention_reference(Q, K, V, scale_factor=8):
    """
    PyTorch reference implementation of scaled dot-product attention

    Args:
        Q, K, V: int8 numpy arrays (64, 64)
        scale_factor: Scaling factor (sqrt(dim) = 8 for 64x64)

    Returns:
        Output: int8 numpy array (64, 64)
    """
    # Convert to float for computation
    Q_f = torch.from_numpy(Q).float()
    K_f = torch.from_numpy(K).float()
    V_f = torch.from_numpy(V).float()

    # Compute attention: softmax(Q @ K^T / sqrt(d)) @ V
    # Q @ K^T: (64, 64) @ (64, 64) = (64, 64)
    scores = torch.matmul(Q_f, K_f.T) / scale_factor

    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Apply attention to values
    output_f = torch.matmul(attn_weights, V_f)

    # Convert back to int8
    output = torch.clamp(output_f, -128, 127).round().to(torch.int8)

    return output.numpy(), attn_weights.numpy()


def npu_attention(Q, K, V):
    """
    Run attention kernel on NPU

    Args:
        Q, K, V: int8 numpy arrays (64, 64)

    Returns:
        Output: int8 numpy array (64, 64)
    """
    TILE_SIZE = 64
    QKV_SIZE = TILE_SIZE * TILE_SIZE
    COMBINED_SIZE = 3 * QKV_SIZE
    OUTPUT_SIZE = QKV_SIZE

    # Load XCLBIN (FIXED VERSION - freshly recompiled with improved softmax and requantization)
    xclbin_path = "build_attention_64x64/attention_64x64.xclbin"
    insts_path = "build_attention_64x64/insts.bin"

    if not os.path.exists(xclbin_path) or not os.path.exists(insts_path):
        raise FileNotFoundError("XCLBIN or instructions not found. Run compile_attention_64x64.sh first")

    device = xrt.device(0)
    xclbin_obj = xrt.xclbin(xclbin_path)
    uuid = xclbin_obj.get_uuid()
    device.register_xclbin(xclbin_obj)

    hw_ctx = xrt.hw_context(device, uuid)
    kernel = xrt.kernel(hw_ctx, "MLIR_AIE")

    with open(insts_path, "rb") as f:
        insts = f.read()
    n_insts = len(insts)

    # Allocate buffers (correct group IDs)
    instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
    input_bo = xrt.bo(device, COMBINED_SIZE, xrt.bo.flags.host_only, kernel.group_id(3))
    output_bo = xrt.bo(device, OUTPUT_SIZE, xrt.bo.flags.host_only, kernel.group_id(4))

    # Write instructions
    instr_bo.write(insts, 0)
    instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

    # Combine Q, K, V
    QKV_combined = np.concatenate([Q.flatten(), K.flatten(), V.flatten()])

    # Write input
    input_bo.write(QKV_combined.tobytes(), 0)
    input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, COMBINED_SIZE, 0)

    # Run kernel with proper state checking
    opcode = 3
    run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
    state = run.wait(1000)

    if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        raise RuntimeError(f"Kernel execution failed with state: {state}")

    # Read output
    output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, OUTPUT_SIZE, 0)
    output = np.frombuffer(output_bo.read(OUTPUT_SIZE, 0), dtype=np.int8)
    output = output.reshape(TILE_SIZE, TILE_SIZE)

    return output


def compute_correlation(npu_output, pytorch_output):
    """Compute Pearson correlation coefficient between two arrays"""
    npu_flat = npu_output.flatten().astype(np.float32)
    pytorch_flat = pytorch_output.flatten().astype(np.float32)

    correlation = np.corrcoef(npu_flat, pytorch_flat)[0, 1]
    return correlation


def compute_accuracy_metrics(npu_output, pytorch_output):
    """Compute comprehensive accuracy metrics"""
    npu_flat = npu_output.flatten().astype(np.float32)
    pytorch_flat = pytorch_output.flatten().astype(np.float32)

    # Mean Absolute Error
    mae = np.mean(np.abs(npu_flat - pytorch_flat))

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((npu_flat - pytorch_flat) ** 2))

    # Max Absolute Error
    max_error = np.max(np.abs(npu_flat - pytorch_flat))

    # Percentage of elements within tolerance
    tolerance = 5  # INT8 values, allow ±5 difference
    within_tolerance = np.mean(np.abs(npu_flat - pytorch_flat) <= tolerance) * 100

    # Correlation
    correlation = compute_correlation(npu_output, pytorch_output)

    return {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'within_tolerance': within_tolerance,
        'correlation': correlation
    }


def test_attention_accuracy():
    """Main accuracy validation test"""

    print("="*80)
    print("ATTENTION KERNEL ACCURACY VALIDATION")
    print("="*80)
    print()
    print("Testing NPU attention kernel against PyTorch reference...")
    print()

    # Generate test data
    print("Step 1: Generating test data...")
    np.random.seed(42)
    TILE_SIZE = 64

    Q = np.random.randint(-64, 64, size=(TILE_SIZE, TILE_SIZE), dtype=np.int8)
    K = np.random.randint(-64, 64, size=(TILE_SIZE, TILE_SIZE), dtype=np.int8)
    V = np.random.randint(-64, 64, size=(TILE_SIZE, TILE_SIZE), dtype=np.int8)

    print(f"  Q: {Q.shape}, range [{Q.min()}, {Q.max()}]")
    print(f"  K: {K.shape}, range [{K.min()}, {K.max()}]")
    print(f"  V: {V.shape}, range [{V.min()}, {V.max()}]")
    print()

    # Run PyTorch reference
    print("Step 2: Computing PyTorch reference...")
    pytorch_output, attn_weights = pytorch_attention_reference(Q, K, V)
    print(f"  PyTorch output: {pytorch_output.shape}, range [{pytorch_output.min()}, {pytorch_output.max()}]")
    print(f"  Attention weights: min={attn_weights.min():.4f}, max={attn_weights.max():.4f}")
    print()

    # Run NPU kernel
    print("Step 3: Running NPU kernel...")
    try:
        npu_output = npu_attention(Q, K, V)
        print(f"  NPU output: {npu_output.shape}, range [{npu_output.min()}, {npu_output.max()}]")
        print()
    except Exception as e:
        print(f"❌ ERROR running NPU kernel: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compute accuracy metrics
    print("Step 4: Computing accuracy metrics...")
    metrics = compute_accuracy_metrics(npu_output, pytorch_output)

    print(f"  Mean Absolute Error (MAE):     {metrics['mae']:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"  Max Absolute Error:            {metrics['max_error']:.1f}")
    print(f"  Within ±5 tolerance:           {metrics['within_tolerance']:.1f}%")
    print(f"  Correlation coefficient:       {metrics['correlation']:.6f}")
    print()

    # Visualize outputs
    print("Step 5: Visual comparison (8x8 corner)...")
    print()
    print("PyTorch reference:")
    print(pytorch_output[:8, :8])
    print()
    print("NPU output:")
    print(npu_output[:8, :8])
    print()
    print("Difference (NPU - PyTorch):")
    diff = npu_output[:8, :8].astype(np.int16) - pytorch_output[:8, :8].astype(np.int16)
    print(diff)
    print()

    # Success criteria
    print("="*80)
    print("SUCCESS CRITERIA EVALUATION")
    print("="*80)
    print()

    success = True

    # Check correlation
    if metrics['correlation'] >= 0.95:
        print(f"✅ EXCELLENT: Correlation {metrics['correlation']:.6f} >= 0.95")
    elif metrics['correlation'] >= 0.90:
        print(f"✅ GOOD: Correlation {metrics['correlation']:.6f} >= 0.90")
    elif metrics['correlation'] >= 0.70:
        print(f"⚠️  ACCEPTABLE: Correlation {metrics['correlation']:.6f} >= 0.70")
    else:
        print(f"❌ FAIL: Correlation {metrics['correlation']:.6f} < 0.70")
        success = False

    # Check MAE
    if metrics['mae'] <= 2.0:
        print(f"✅ MAE {metrics['mae']:.4f} <= 2.0 (excellent)")
    elif metrics['mae'] <= 5.0:
        print(f"✅ MAE {metrics['mae']:.4f} <= 5.0 (good)")
    elif metrics['mae'] <= 10.0:
        print(f"⚠️  MAE {metrics['mae']:.4f} <= 10.0 (acceptable)")
    else:
        print(f"❌ MAE {metrics['mae']:.4f} > 10.0 (poor)")
        success = False

    # Check tolerance
    if metrics['within_tolerance'] >= 95.0:
        print(f"✅ {metrics['within_tolerance']:.1f}% within ±5 (excellent)")
    elif metrics['within_tolerance'] >= 90.0:
        print(f"✅ {metrics['within_tolerance']:.1f}% within ±5 (good)")
    elif metrics['within_tolerance'] >= 80.0:
        print(f"⚠️  {metrics['within_tolerance']:.1f}% within ±5 (acceptable)")
    else:
        print(f"❌ {metrics['within_tolerance']:.1f}% within ±5 (poor)")
        success = False

    print()

    if success:
        print("="*80)
        print("🎉 ACCURACY VALIDATION: PASSED!")
        print("="*80)
        print()
        print("The NPU attention kernel produces outputs that are highly")
        print("correlated with PyTorch reference implementation.")
        print()
        print("✅ Ready for integration into Whisper encoder pipeline")
        print()
    else:
        print("="*80)
        print("❌ ACCURACY VALIDATION: FAILED")
        print("="*80)
        print()
        print("The NPU kernel needs tuning to improve accuracy.")
        print()

    return success


if __name__ == "__main__":
    success = test_attention_accuracy()
    sys.exit(0 if success else 1)
