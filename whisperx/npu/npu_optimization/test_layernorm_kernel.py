#!/usr/bin/env python3
"""
LayerNorm Kernel NPU Test and Validation
Tests layernorm_simple.xclbin (256 elements) with combined buffer

Based on working test_attention_64x64.py pattern with instruction buffer loading.

Validates:
- NPU execution correctness
- Accuracy vs PyTorch LayerNorm (target: >0.99 correlation)
- Mean ≈ 0, Std ≈ 1 after normalization
- Performance metrics

Usage:
    cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels
    python3 ../test_layernorm_kernel.py
"""

import numpy as np
import time
import sys
import os
from pathlib import Path

# Add XRT path
sys.path.insert(0, '/opt/xilinx/xrt/python')

try:
    import pyxrt as xrt
except ImportError:
    print("ERROR: PyXRT not installed. Install with: pip install pyxrt")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available - will use NumPy reference only")


def layernorm_reference_numpy(input_float, gamma_float, beta_float, eps=1e-5):
    """NumPy reference layer normalization (matches PyTorch)"""
    # Compute mean and variance
    mean = np.mean(input_float, axis=-1, keepdims=True)
    var = np.var(input_float, axis=-1, keepdims=True)

    # Normalize
    normalized = (input_float - mean) / np.sqrt(var + eps)

    # Scale and shift
    output = gamma_float * normalized + beta_float

    return output, mean, var


def layernorm_reference_torch(input_float, gamma_float, beta_float, eps=1e-5):
    """PyTorch reference layer normalization"""
    if not TORCH_AVAILABLE:
        return None

    # Convert to torch tensors
    input_tensor = torch.from_numpy(input_float).float()
    gamma_tensor = torch.from_numpy(gamma_float).float()
    beta_tensor = torch.from_numpy(beta_float).float()

    # Compute layer norm using PyTorch
    layer_norm = nn.LayerNorm(input_tensor.shape[-1], eps=eps, elementwise_affine=True)
    layer_norm.weight.data = gamma_tensor
    layer_norm.bias.data = beta_tensor

    output_tensor = layer_norm(input_tensor)

    return output_tensor.numpy()


def test_layernorm_kernel(size=256, num_iterations=100):
    """
    Test LayerNorm kernel on NPU hardware

    Args:
        size: Number of features (256)
        num_iterations: Number of benchmark iterations
    """

    print("="*70)
    print(f"LayerNorm Kernel NPU Test ({size} features)")
    print("="*70)
    print()

    # Paths
    base_dir = Path("/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels")

    xclbin_path = base_dir / "build_layernorm" / "layernorm_simple.xclbin"
    insts_path = base_dir / "build_layernorm" / "insts.bin"

    # Verify files exist
    if not xclbin_path.exists():
        print(f"ERROR: XCLBIN not found: {xclbin_path}")
        print("Run: bash compile_layernorm.sh")
        return None

    if not insts_path.exists():
        print(f"ERROR: Instructions not found: {insts_path}")
        print("Run: bash compile_layernorm.sh")
        return None

    print(f"Configuration:")
    print(f"  XCLBIN: {xclbin_path.name}")
    print(f"  Instructions: {insts_path.name}")
    print(f"  Features: {size}")
    print()

    # Step 1: Load XCLBIN
    print(f"Step 1: Loading XCLBIN from {xclbin_path.name}...")
    try:
        device = xrt.device(0)
        xclbin_obj = xrt.xclbin(str(xclbin_path))
        uuid = xclbin_obj.get_uuid()
        device.register_xclbin(xclbin_obj)
        print(f"✅ XCLBIN loaded successfully")
        print(f"   UUID: {uuid}")

        # Create hardware context
        hw_ctx = xrt.hw_context(device, uuid)
        print("✅ Hardware context created")

        # Get kernel
        kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
        print("✅ Kernel found: MLIR_AIE")

        # Load instruction sequence
        with open(insts_path, "rb") as f:
            insts = f.read()
        n_insts = len(insts)
        print(f"✅ Instructions loaded: {n_insts} bytes")
    except Exception as e:
        print(f"❌ ERROR loading XCLBIN: {e}")
        import traceback
        traceback.print_exc()
        return None
    print()

    # Step 2: Generate test data
    print(f"Step 2: Generating test data ({size} features)...")
    np.random.seed(42)

    # Input: random normal distribution (typical for neural network activations)
    input_float = np.random.randn(size).astype(np.float32) * 0.5

    # Gamma: typically initialized to 1
    gamma_float = np.ones(size, dtype=np.float32)

    # Beta: typically initialized to 0
    beta_float = np.zeros(size, dtype=np.float32)

    print(f"  Input stats: mean={input_float.mean():.4f}, std={input_float.std():.4f}")
    print()

    # Quantize to INT8
    scale = 127.0
    input_int8 = np.clip(np.round(input_float * scale), -128, 127).astype(np.int8)
    gamma_int8 = np.clip(np.round(gamma_float * scale), -128, 127).astype(np.int8)
    beta_int8 = np.clip(np.round(beta_float * scale), -128, 127).astype(np.int8)

    # Create combined input buffer (input + gamma + beta = 256 + 256 + 256 = 768 bytes)
    combined_input = np.concatenate([input_int8, gamma_int8, beta_int8])

    print(f"  Quantized input range: [{input_int8.min()}, {input_int8.max()}]")
    print(f"  Combined buffer size: {combined_input.shape[0]} bytes")
    print()

    # Step 3: Allocate NPU buffers
    print(f"Step 3: Allocating NPU buffers...")
    try:
        # Instruction buffer (group_id 1)
        instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
        instr_bo.write(insts, 0)
        instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

        # Combined input buffer (group_id 3): input + gamma + beta
        input_size = combined_input.nbytes
        input_bo = xrt.bo(device, input_size, xrt.bo.flags.host_only, kernel.group_id(3))
        input_bo.write(combined_input.tobytes(), 0)
        input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, input_size, 0)

        # Output buffer (group_id 4): normalized output
        output_size = size  # 256 bytes
        output_bo = xrt.bo(device, output_size, xrt.bo.flags.host_only, kernel.group_id(4))

        print(f"✅ Allocated instruction buffer: {n_insts} bytes")
        print(f"✅ Allocated input buffer: {input_size} bytes")
        print(f"✅ Allocated output buffer: {output_size} bytes")
    except Exception as e:
        print(f"❌ ERROR allocating buffers: {e}")
        import traceback
        traceback.print_exc()
        return None
    print()

    # Step 4: Warmup runs
    print(f"Step 4: Warmup (3 iterations)...")
    try:
        for i in range(3):
            opcode = 3  # Standard opcode for NPU kernels
            run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
            state = run.wait(1000)  # 1 second timeout
            if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                print(f"❌ ERROR in warmup iteration {i+1}: kernel state {state}")
                return None
        print("✅ Warmup complete")
    except Exception as e:
        print(f"❌ ERROR in warmup: {e}")
        import traceback
        traceback.print_exc()
        return None
    print()

    # Step 5: Benchmark performance
    print(f"Step 5: Running benchmark ({num_iterations} iterations)...")
    times = []
    try:
        for i in range(num_iterations):
            start = time.perf_counter()
            opcode = 3
            run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
            state = run.wait(1000)
            if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                print(f"❌ ERROR in iteration {i+1}: kernel state {state}")
                return None
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    except Exception as e:
        print(f"❌ ERROR in benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"✅ Kernel execution complete")
    print(f"   Average: {avg_time:.3f} ms")
    print(f"   Std Dev: {std_time:.3f} ms")
    print(f"   Min:     {min_time:.3f} ms")
    print(f"   Max:     {max_time:.3f} ms")
    print()

    # Step 6: Read output
    print(f"Step 6: Reading output from NPU...")
    try:
        output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, output_size, 0)
        output_int8 = np.frombuffer(output_bo.read(output_size, 0), dtype=np.int8)
        print(f"✅ Output retrieved: {output_int8.shape}")
        print(f"   Output range: [{output_int8.min()}, {output_int8.max()}]")
        print(f"   Output mean:  {output_int8.mean():.2f}")
    except Exception as e:
        print(f"❌ ERROR reading output: {e}")
        import traceback
        traceback.print_exc()
        return None
    print()

    # Step 7: Compute reference
    print(f"Step 7: Computing reference LayerNorm...")

    # Dequantize output for comparison
    output_float = output_int8.astype(np.float32) / scale

    # NumPy reference
    ref_numpy, mean_np, var_np = layernorm_reference_numpy(input_float, gamma_float, beta_float)
    ref_int8_numpy = np.clip(np.round(ref_numpy * scale), -128, 127).astype(np.int8)

    # PyTorch reference (if available)
    if TORCH_AVAILABLE:
        ref_torch = layernorm_reference_torch(input_float, gamma_float, beta_float)
        ref_int8_torch = np.clip(np.round(ref_torch * scale), -128, 127).astype(np.int8)
        print(f"✅ PyTorch reference computed")
        ref_int8 = ref_int8_torch
        ref_float = ref_torch
    else:
        ref_int8 = ref_int8_numpy
        ref_float = ref_numpy
        print(f"✅ NumPy reference computed")

    print(f"   Reference range: [{ref_int8.min()}, {ref_int8.max()}]")
    print(f"   Reference mean:  {ref_int8.mean():.2f}")
    print(f"   Reference (float) mean: {ref_float.mean():.6f}")
    print(f"   Reference (float) std:  {ref_float.std():.6f}")
    print()

    # Step 8: Compute accuracy metrics
    print(f"Step 8: Computing accuracy metrics...")

    # INT8 errors
    int8_error = np.abs(output_int8.astype(np.int32) - ref_int8.astype(np.int32))
    mae_int8 = int8_error.mean()
    max_error_int8 = int8_error.max()

    # Correlation
    correlation = np.corrcoef(output_int8.flatten(), ref_int8.flatten())[0, 1]

    # Float-space comparison
    mae_float = np.abs(output_float - ref_float).mean()

    # Check normalized statistics
    output_mean = output_float.mean()
    output_std = output_float.std()

    print(f"Accuracy Metrics:")
    print(f"  Mean Absolute Error (INT8): {mae_int8:.2f} units")
    print(f"  Max Absolute Error (INT8):  {max_error_int8} units")
    print(f"  Mean Absolute Error (Float): {mae_float:.6f}")
    print(f"  Correlation:                 {correlation:.6f}")
    print()

    print(f"Normalized Statistics (Float):")
    print(f"  Output mean: {output_mean:.6f} (target: ~0.0)")
    print(f"  Output std:  {output_std:.6f} (target: ~1.0)")
    print()

    # Performance metrics
    throughput = size / (avg_time / 1000) / 1e6  # Million features/sec

    # For Whisper: estimate realtime factor
    # Whisper base: 12 encoder blocks, each with 2 LayerNorm (512 features)
    # Total: 24 LayerNorm operations per 30-second chunk
    # Current kernel is 256 features, so 512 features = 2x this kernel
    operations_per_chunk = 24 * (512 / size)  # Number of 256-feature LayerNorm ops
    time_per_chunk = operations_per_chunk * avg_time / 1000  # seconds
    realtime_factor_ln = 30.0 / time_per_chunk if time_per_chunk > 0 else 0

    print(f"Performance Metrics:")
    print(f"  Throughput: {throughput:.2f} M features/sec")
    print(f"  Features per ms: {size / avg_time:.0f}")
    print()

    print(f"Whisper Base Encoder Estimate:")
    print(f"  LayerNorm ops per 30s: {int(operations_per_chunk)} × {size} features")
    print(f"  Time for all LayerNorm: {time_per_chunk*1000:.2f} ms")
    print(f"  LayerNorm-only realtime: {realtime_factor_ln:.0f}x")
    print()

    # Success criteria
    print("="*70)
    print("SUCCESS CRITERIA")
    print("="*70)
    print()

    success = True

    # Check 1: Accuracy
    correlation_target = 0.99
    if correlation >= correlation_target:
        print(f"✅ Correlation: PASSED ({correlation:.4f} >= {correlation_target})")
    else:
        print(f"❌ Correlation: FAILED ({correlation:.4f} < {correlation_target})")
        success = False

    # Check 2: MAE
    mae_target = 2.0  # INT8 units
    if mae_int8 <= mae_target:
        print(f"✅ Mean Error: PASSED ({mae_int8:.2f} <= {mae_target})")
    else:
        print(f"⚠️  Mean Error: WARNING ({mae_int8:.2f} > {mae_target})")

    # Check 3: Normalized mean ≈ 0
    mean_target = 0.1  # Allow small deviation
    if abs(output_mean) <= mean_target:
        print(f"✅ Normalized Mean: PASSED (|{output_mean:.4f}| <= {mean_target})")
    else:
        print(f"⚠️  Normalized Mean: WARNING (|{output_mean:.4f}| > {mean_target})")

    # Check 4: Normalized std ≈ 1
    std_target_low = 0.9
    std_target_high = 1.1
    if std_target_low <= output_std <= std_target_high:
        print(f"✅ Normalized Std: PASSED ({output_std:.4f} in [{std_target_low}, {std_target_high}])")
    else:
        print(f"⚠️  Normalized Std: WARNING ({output_std:.4f} not in [{std_target_low}, {std_target_high}])")

    # Check 5: Performance
    perf_target = 1.0  # ms
    if avg_time <= perf_target:
        print(f"✅ Performance: PASSED ({avg_time:.3f} ms <= {perf_target} ms)")
    else:
        print(f"⚠️  Performance: ACCEPTABLE ({avg_time:.3f} ms > {perf_target} ms target)")

    # Check 6: Non-zero output
    if np.count_nonzero(output_int8) > 0:
        print(f"✅ Non-zero output: PASSED ({np.count_nonzero(output_int8)}/{size} elements)")
    else:
        print(f"❌ Non-zero output: FAILED (all zeros)")
        success = False

    print()

    if success:
        print("="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
    else:
        print("="*70)
        print("⚠️  SOME TESTS FAILED")
        print("="*70)

    print()

    # Sample values for debugging
    print("Sample values (first 10 elements):")
    print(f"  Input (float):     {input_float[:10]}")
    print(f"  Input (int8):      {input_int8[:10]}")
    print(f"  NPU Output (int8): {output_int8[:10]}")
    print(f"  NPU Output (float):{output_float[:10]}")
    print(f"  Reference (float): {ref_float[:10]}")
    print()

    # Return results
    return {
        'size': size,
        'success': success,
        'correlation': correlation,
        'mae_int8': mae_int8,
        'max_error_int8': max_error_int8,
        'mae_float': mae_float,
        'output_mean': output_mean,
        'output_std': output_std,
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'throughput_mfps': throughput,
        'realtime_factor': realtime_factor_ln,
        'input': input_int8,
        'output_npu': output_int8,
        'output_ref': ref_int8
    }


def main():
    print("="*70)
    print("LayerNorm Kernel Test Suite")
    print("AMD Phoenix NPU (AIE2)")
    print("="*70)
    print()

    # Test 256-feature kernel (layernorm_simple)
    result = test_layernorm_kernel(size=256, num_iterations=100)

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print()

    if result:
        print(f"LayerNorm Simple (256):")
        print(f"  Correlation:     {result['correlation']:.6f}")
        print(f"  MAE (INT8):      {result['mae_int8']:.2f} units")
        print(f"  Output mean:     {result['output_mean']:.6f} (target: ~0.0)")
        print(f"  Output std:      {result['output_std']:.6f} (target: ~1.0)")
        print(f"  Performance:     {result['avg_time_ms']:.3f} ms")
        print(f"  Throughput:      {result['throughput_mfps']:.2f} M features/s")
        print(f"  Realtime factor: {result['realtime_factor']:.0f}x")
        print(f"  Status:          {'✅ PASSED' if result['success'] else '❌ FAILED'}")
        print()

    if result and result['success']:
        print("="*70)
        print("🎉 LAYERNORM KERNEL VALIDATED SUCCESSFULLY!")
        print("="*70)
        print()
        print("Next steps:")
        print("  1. Create npu_layernorm_wrapper.py for production use")
        print("  2. Scale to 512 features (Whisper base full size)")
        print("  3. Integrate with Whisper encoder")
        print("  4. Measure contribution to full pipeline speedup")
        return 0
    else:
        print("="*70)
        print("⚠️  LAYERNORM TEST FAILED")
        print("="*70)
        print()
        print("Review output and investigate:")
        print("  1. Kernel implementation correctness")
        print("  2. Fixed-point arithmetic precision")
        print("  3. DMA transfer integrity")
        return 1


if __name__ == "__main__":
    sys.exit(main())
