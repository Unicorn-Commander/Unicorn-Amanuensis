#!/usr/bin/env python3
"""
GELU Kernel NPU Test and Validation
Tests both gelu_simple.xclbin (512 elements) and gelu_2048.xclbin (2048 elements)

Based on working test_attention_64x64.py pattern with instruction buffer loading.

Validates:
- NPU execution correctness
- Accuracy vs PyTorch GELU (target: >0.99 correlation)
- Performance (<0.5ms target)
- Different input sizes (512, 2048)

Usage:
    cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels
    python3 ../test_gelu_kernel.py
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


def gelu_reference_numpy(x):
    """NumPy reference GELU implementation"""
    import math
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))


def gelu_reference_torch(x_tensor):
    """PyTorch reference GELU implementation"""
    if not TORCH_AVAILABLE:
        return None
    gelu = nn.GELU()
    return gelu(x_tensor)


def test_gelu_kernel(size=512, num_iterations=100):
    """
    Test GELU kernel on NPU hardware

    Args:
        size: Number of elements (512 or 2048)
        num_iterations: Number of benchmark iterations
    """

    print("="*70)
    print(f"GELU Kernel NPU Test ({size} elements)")
    print("="*70)
    print()

    # Determine which XCLBIN to use
    base_dir = Path("/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels")

    if size == 512:
        xclbin_path = base_dir / "build_gelu" / "gelu_simple.xclbin"
        insts_path = base_dir / "build_gelu" / "insts_512.bin"
    elif size == 2048:
        xclbin_path = base_dir / "build_gelu" / "gelu_2048.xclbin"
        insts_path = base_dir / "build_gelu" / "insts_2048.bin"
    else:
        print(f"ERROR: Unsupported size {size}. Use 512 or 2048.")
        return None

    # Verify files exist
    if not xclbin_path.exists():
        print(f"ERROR: XCLBIN not found: {xclbin_path}")
        print("Run: bash compile_gelu.sh")
        return None

    if not insts_path.exists():
        print(f"ERROR: Instructions not found: {insts_path}")
        print("Run: bash compile_gelu.sh")
        return None

    print(f"Configuration:")
    print(f"  XCLBIN: {xclbin_path.name}")
    print(f"  Instructions: {insts_path.name}")
    print(f"  Size: {size} elements")
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
    print(f"Step 2: Generating random INT8 test data ({size} elements)...")
    np.random.seed(42)

    # Generate input in range suitable for GELU (roughly [-2, 2] in float)
    # For INT8, this is [-64, 64] with scale factor of 32
    x_int8 = np.random.randint(-64, 64, size=size, dtype=np.int8)

    print(f"  Input range: [{x_int8.min()}, {x_int8.max()}]")
    print(f"  Input mean:  {x_int8.mean():.2f}")
    print(f"  Input std:   {x_int8.std():.2f}")
    print()

    # Step 3: Allocate NPU buffers
    print(f"Step 3: Allocating NPU buffers...")
    try:
        # Instruction buffer (group_id 1)
        instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
        instr_bo.write(insts, 0)
        instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

        # Input buffer (group_id 3)
        input_bo = xrt.bo(device, size, xrt.bo.flags.host_only, kernel.group_id(3))
        input_bo.write(x_int8.tobytes(), 0)
        input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, size, 0)

        # Output buffer (group_id 4)
        output_bo = xrt.bo(device, size, xrt.bo.flags.host_only, kernel.group_id(4))

        print(f"✅ Allocated instruction buffer: {n_insts} bytes")
        print(f"✅ Allocated input buffer: {size} bytes")
        print(f"✅ Allocated output buffer: {size} bytes")
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
        output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, size, 0)
        y_npu = np.frombuffer(output_bo.read(size, 0), dtype=np.int8)
        print(f"✅ Output retrieved: {y_npu.shape}")
        print(f"   Output range: [{y_npu.min()}, {y_npu.max()}]")
        print(f"   Output mean:  {y_npu.mean():.2f}")
    except Exception as e:
        print(f"❌ ERROR reading output: {e}")
        import traceback
        traceback.print_exc()
        return None
    print()

    # Step 7: Compute reference
    print(f"Step 7: Computing reference GELU...")

    # Dequantize input to float32
    x_float = x_int8.astype(np.float32) / 127.0

    # NumPy reference
    y_ref_numpy = gelu_reference_numpy(x_float)
    y_ref_int8_numpy = np.clip(np.round(y_ref_numpy * 127), -128, 127).astype(np.int8)

    # PyTorch reference (if available)
    if TORCH_AVAILABLE:
        x_torch = torch.from_numpy(x_float)
        y_torch = gelu_reference_torch(x_torch).numpy()
        y_ref_int8_torch = np.clip(np.round(y_torch * 127), -128, 127).astype(np.int8)
        print(f"✅ PyTorch reference computed")
    else:
        y_ref_int8_torch = y_ref_int8_numpy
        print(f"✅ NumPy reference computed")

    print(f"   Reference range: [{y_ref_int8_torch.min()}, {y_ref_int8_torch.max()}]")
    print(f"   Reference mean:  {y_ref_int8_torch.mean():.2f}")
    print()

    # Step 8: Compute accuracy metrics
    print(f"Step 8: Computing accuracy metrics...")

    # INT8 errors
    int8_error = np.abs(y_npu.astype(np.int32) - y_ref_int8_torch.astype(np.int32))
    mae_int8 = int8_error.mean()
    max_error_int8 = int8_error.max()

    # Correlation
    correlation = np.corrcoef(y_npu.flatten(), y_ref_int8_torch.flatten())[0, 1]

    # Float-space comparison
    y_npu_float = y_npu.astype(np.float32) / 127.0
    y_ref_float = y_ref_int8_torch.astype(np.float32) / 127.0
    mae_float = np.abs(y_npu_float - y_ref_float).mean()

    print(f"Accuracy Metrics:")
    print(f"  Mean Absolute Error (INT8): {mae_int8:.2f} units")
    print(f"  Max Absolute Error (INT8):  {max_error_int8} units")
    print(f"  Mean Absolute Error (Float): {mae_float:.6f}")
    print(f"  Correlation:                 {correlation:.6f}")
    print()

    # Performance metrics
    throughput = size / (avg_time / 1000) / 1e6  # Million elements/sec

    # For Whisper: estimate realtime factor
    # Whisper base: 12 encoder blocks, each with 1 GELU (2048 elements)
    # Total: 12 * 2048 = 24,576 GELU operations per 30-second chunk
    # At avg_time per operation
    operations_per_chunk = 12 * 2048 / size  # Number of this-sized operations
    time_per_chunk = operations_per_chunk * avg_time / 1000  # seconds
    realtime_factor_gelu = 30.0 / time_per_chunk if time_per_chunk > 0 else 0

    print(f"Performance Metrics:")
    print(f"  Throughput: {throughput:.2f} M elements/sec")
    print(f"  Elements per ms: {size / avg_time:.0f}")
    print()

    print(f"Whisper Base Encoder Estimate:")
    print(f"  GELU operations per 30s: {int(operations_per_chunk * size)} elements")
    print(f"  Time for all GELU: {time_per_chunk*1000:.2f} ms")
    print(f"  GELU-only realtime: {realtime_factor_gelu:.0f}x")
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

    # Check 3: Performance
    perf_target = 0.5  # ms
    if avg_time <= perf_target:
        print(f"✅ Performance: PASSED ({avg_time:.3f} ms <= {perf_target} ms)")
    else:
        print(f"⚠️  Performance: ACCEPTABLE ({avg_time:.3f} ms > {perf_target} ms target)")

    # Check 4: Non-zero output
    if np.count_nonzero(y_npu) > 0:
        print(f"✅ Non-zero output: PASSED ({np.count_nonzero(y_npu)}/{size} elements)")
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

    # Return results
    return {
        'size': size,
        'success': success,
        'correlation': correlation,
        'mae_int8': mae_int8,
        'max_error_int8': max_error_int8,
        'mae_float': mae_float,
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'throughput_meps': throughput,
        'realtime_factor': realtime_factor_gelu,
        'input': x_int8,
        'output_npu': y_npu,
        'output_ref': y_ref_int8_torch
    }


def test_edge_cases():
    """Test GELU with edge case inputs"""
    print("="*70)
    print("Edge Case Tests")
    print("="*70)
    print()

    # Load LUT for comparison (if exists)
    lut_file = Path("/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/gelu_lut.bin")
    if lut_file.exists():
        gelu_lut = np.fromfile(lut_file, dtype=np.int8)
        print("✅ LUT loaded for edge case validation")
    else:
        print("⚠️  LUT file not found - skipping LUT comparison")
        gelu_lut = None

    test_cases = [
        ("Zeros", np.zeros(10, dtype=np.int8)),
        ("Min value", np.full(10, -128, dtype=np.int8)),
        ("Max value", np.full(10, 127, dtype=np.int8)),
        ("Small positive", np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50], dtype=np.int8)),
        ("Small negative", np.array([-1, -2, -3, -4, -5, -10, -20, -30, -40, -50], dtype=np.int8)),
        ("Random range", np.random.randint(-64, 64, 10, dtype=np.int8)),
    ]

    for name, x_int8 in test_cases:
        # Reference
        x_float = x_int8.astype(np.float32) / 127.0
        y_float_ref = gelu_reference_numpy(x_float)
        y_ref = np.clip(np.round(y_float_ref * 127), -128, 127).astype(np.int8)

        # LUT lookup (if available)
        if gelu_lut is not None:
            idx = (x_int8.astype(np.uint8) + 128).astype(np.uint8)
            y_lut = gelu_lut[idx]
            lut_error = np.abs(y_lut - y_ref).max()
            lut_str = f"LUT: {y_lut[:5]}, Error: {lut_error}"
        else:
            lut_str = "LUT: N/A"

        print(f"{name}:")
        print(f"  Input: {x_int8[:5]}")
        print(f"  Ref:   {y_ref[:5]}")
        print(f"  {lut_str}")
        print()


def main():
    print("="*70)
    print("GELU Activation Kernel Test Suite")
    print("AMD Phoenix NPU (AIE2)")
    print("="*70)
    print()

    # Test edge cases first
    test_edge_cases()

    # Test 512-element kernel (gelu_simple)
    print("\n" + "="*70)
    print("TEST 1: GELU Simple (512 elements)")
    print("="*70)
    print()
    result_512 = test_gelu_kernel(size=512, num_iterations=100)

    # Test 2048-element kernel (gelu_2048)
    print("\n" + "="*70)
    print("TEST 2: GELU 2048 (2048 elements)")
    print("="*70)
    print()
    result_2048 = test_gelu_kernel(size=2048, num_iterations=100)

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print()

    if result_512:
        print(f"GELU Simple (512):")
        print(f"  Correlation: {result_512['correlation']:.6f}")
        print(f"  MAE (INT8):  {result_512['mae_int8']:.2f} units")
        print(f"  Performance: {result_512['avg_time_ms']:.3f} ms")
        print(f"  Throughput:  {result_512['throughput_meps']:.2f} M elem/s")
        print(f"  Status:      {'✅ PASSED' if result_512['success'] else '❌ FAILED'}")
        print()

    if result_2048:
        print(f"GELU 2048 (2048):")
        print(f"  Correlation: {result_2048['correlation']:.6f}")
        print(f"  MAE (INT8):  {result_2048['mae_int8']:.2f} units")
        print(f"  Performance: {result_2048['avg_time_ms']:.3f} ms")
        print(f"  Throughput:  {result_2048['throughput_meps']:.2f} M elem/s")
        print(f"  Status:      {'✅ PASSED' if result_2048['success'] else '❌ FAILED'}")
        print()

    # Overall success
    overall_success = (
        result_512 and result_512['success'] and
        result_2048 and result_2048['success']
    )

    if overall_success:
        print("="*70)
        print("🎉 ALL GELU KERNELS VALIDATED SUCCESSFULLY!")
        print("="*70)
        print()
        print("Next steps:")
        print("  1. Create npu_gelu_wrapper.py for production use")
        print("  2. Integrate with Whisper encoder")
        print("  3. Measure contribution to full pipeline speedup")
        return 0
    else:
        print("="*70)
        print("⚠️  SOME GELU TESTS FAILED")
        print("="*70)
        print()
        print("Review output and investigate:")
        print("  1. Kernel implementation correctness")
        print("  2. LUT accuracy")
        print("  3. DMA transfer integrity")
        return 1


if __name__ == "__main__":
    sys.exit(main())
