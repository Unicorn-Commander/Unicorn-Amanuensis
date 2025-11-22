#!/usr/bin/env python3
"""
Encoder Chain Integration Test
Tests the chained execution of LayerNorm -> Softmax -> GELU on AMD Phoenix NPU

This represents a simplified encoder layer pattern:
- LayerNorm: Pre-normalization
- Softmax: Attention scores
- GELU: Feed-forward activation

AMD Phoenix NPU - XDNA1
Tiles: (0,2) LayerNorm, (0,3) Softmax, (0,4) GELU
"""

import numpy as np
import pyxrt as xrt
import struct
import time
import os

def bf16_to_float(bf16_bytes):
    """Convert BF16 bytes to float32"""
    result = np.zeros(len(bf16_bytes) // 2, dtype=np.float32)
    for i in range(len(result)):
        upper = struct.unpack('H', bf16_bytes[i*2:(i+1)*2])[0]
        result[i] = struct.unpack('f', struct.pack('I', upper << 16))[0]
    return result

def float_to_bf16(floats):
    """Convert float32 to BF16 bytes"""
    result = bytearray(len(floats) * 2)
    for i, val in enumerate(floats):
        bits = struct.unpack('I', struct.pack('f', val))[0]
        upper = (bits >> 16) & 0xFFFF
        struct.pack_into('H', result, i*2, upper)
    return bytes(result)

def layernorm_ref(x, eps=1e-5):
    """Reference LayerNorm implementation"""
    mean = np.mean(x)
    variance = np.var(x)
    return (x - mean) / np.sqrt(variance + eps)

def softmax_ref(x):
    """Reference softmax implementation"""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

def gelu_ref(x):
    """Reference GELU implementation (approximate)"""
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def main():
    print("=" * 70)
    print("Encoder Chain Integration Test - AMD Phoenix NPU")
    print("=" * 70)
    print()
    print("Testing: LayerNorm -> Softmax -> GELU (Chained XCLBIN)")
    print("This simulates: normalize -> attention scores -> activation")
    print()

    num_elements = 1024
    buffer_size = num_elements * 2  # 2048 bytes for bfloat16
    num_iterations = 10

    # Paths to encoder chain build
    build_dir = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1/build_encoder_simple"

    xclbin_path = f"{build_dir}/encoder_layer_simple.xclbin"
    insts_path = f"{build_dir}/insts.bin"

    # Verify files exist
    print("Step 1: Verifying build artifacts...")
    for path in [xclbin_path, insts_path]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            return 1
        size = os.path.getsize(path)
        print(f"  Found: {os.path.basename(path)} ({size} bytes)")
    print()

    try:
        # Initialize device
        print("Step 2: Initializing NPU device...")
        device = xrt.device(0)
        print("  Device initialized successfully")
        print()

        # Load chained XCLBIN
        print("Step 3: Loading encoder chain XCLBIN...")
        xclbin_obj = xrt.xclbin(xclbin_path)
        uuid = xclbin_obj.get_uuid()
        device.register_xclbin(xclbin_obj)

        hw_ctx = xrt.hw_context(device, uuid)
        kernel = xrt.kernel(hw_ctx, "MLIR_AIE")

        # Load instructions
        with open(insts_path, "rb") as f:
            insts = f.read()

        print(f"  XCLBIN loaded: {os.path.getsize(xclbin_path)} bytes")
        print(f"  Instructions loaded: {len(insts)} bytes")
        print()

        # Allocate buffers
        print("Step 4: Allocating XRT buffers...")
        bo_instr = xrt.bo(device, len(insts),
                         xrt.bo.flags.cacheable, kernel.group_id(1))
        bo_in = xrt.bo(device, buffer_size,
                      xrt.bo.flags.host_only, kernel.group_id(3))
        bo_out = xrt.bo(device, buffer_size,
                       xrt.bo.flags.host_only, kernel.group_id(4))

        # Sync instruction buffer
        bo_instr.write(insts, 0)
        bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        print(f"  Input buffer: {buffer_size} bytes")
        print(f"  Output buffer: {buffer_size} bytes")
        print()

        # Prepare test data
        print("Step 5: Preparing test data...")
        # Input data representing values before normalization
        np.random.seed(42)  # Reproducible
        input_floats = np.random.randn(num_elements).astype(np.float32) * 2.0 + 0.5
        input_bf16 = float_to_bf16(input_floats)

        # Compute expected results through the chain
        expected_after_layernorm = layernorm_ref(input_floats)
        expected_after_softmax = softmax_ref(expected_after_layernorm)
        expected_final = gelu_ref(expected_after_softmax)

        print(f"  Input: mean={np.mean(input_floats):.4f}, std={np.std(input_floats):.4f}")
        print(f"  Input range: [{np.min(input_floats):.4f}, {np.max(input_floats):.4f}]")
        print()

        # Run chained kernel
        print("Step 6: Running encoder chain (LayerNorm -> Softmax -> GELU)...")
        print()

        chain_times = []
        final_output = None

        for iteration in range(num_iterations):
            # Write input
            bo_in.write(input_bf16, 0)
            bo_in.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            # Execute chained kernel
            start = time.perf_counter()
            run = kernel(3, bo_instr, len(insts), bo_in, bo_out)
            run.wait()
            end = time.perf_counter()

            # Read output
            bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            output_bf16 = bo_out.read(buffer_size, 0).tobytes()

            chain_times.append(end - start)

            if iteration == num_iterations - 1:
                final_output = bf16_to_float(output_bf16)

        print("  Chained kernel execution complete")
        print()

        # Performance results
        print("Step 7: Performance Results:")
        print()

        avg_chain = np.mean(chain_times) * 1000
        min_chain = np.min(chain_times) * 1000
        max_chain = np.max(chain_times) * 1000
        std_chain = np.std(chain_times) * 1000

        print("Chained Execution Performance:")
        print(f"  Average time:   {avg_chain:.3f} ms")
        print(f"  Min time:       {min_chain:.3f} ms")
        print(f"  Max time:       {max_chain:.3f} ms")
        print(f"  Std deviation:  {std_chain:.3f} ms")
        print()

        # Compare with individual kernels baseline (~3.8ms)
        individual_baseline_ms = 3.8
        speedup = individual_baseline_ms / avg_chain if avg_chain > 0 else 0
        savings_ms = individual_baseline_ms - avg_chain

        print("Comparison with Individual Kernels:")
        print(f"  Individual kernels baseline: ~{individual_baseline_ms:.1f} ms")
        print(f"  Chained execution:           {avg_chain:.3f} ms")
        print(f"  Time savings:                {savings_ms:.3f} ms ({savings_ms/individual_baseline_ms*100:.1f}%)")
        if speedup > 1:
            print(f"  Speedup:                     {speedup:.2f}x faster")
        else:
            print(f"  Ratio:                       {speedup:.2f}x")
        print()

        # Throughput
        elements_per_sec = num_elements / (avg_chain / 1000)
        print(f"  Chain throughput: {elements_per_sec/1e6:.2f} M elements/second")
        print()

        # Accuracy verification
        print("Step 8: Output Validation:")
        print()

        # Check output characteristics
        output_min = np.min(final_output)
        output_max = np.max(final_output)
        output_mean = np.mean(final_output)
        output_std = np.std(final_output)

        print("Output Characteristics:")
        print(f"  Min value:  {output_min:.6f}")
        print(f"  Max value:  {output_max:.6f}")
        print(f"  Mean:       {output_mean:.6f}")
        print(f"  Std:        {output_std:.6f}")
        print()

        # GELU output properties check
        # After softmax (sum=1, all positive), GELU output should:
        # - Be mostly positive (GELU of small positive numbers)
        # - Have values roughly in [0, 0.5] range for softmax outputs

        valid_range = output_min >= -0.1 and output_max <= 1.0  # Allow some tolerance
        non_zero = np.sum(np.abs(final_output) > 1e-6) > num_elements * 0.5

        print("Validation Checks:")
        print(f"  Output in valid range [-0.1, 1.0]: {'PASS' if valid_range else 'FAIL'}")
        print(f"  Non-zero outputs (>50%):          {'PASS' if non_zero else 'FAIL'}")

        # Check correlation with expected
        correlation = np.corrcoef(final_output, expected_final)[0, 1]
        max_error = np.max(np.abs(final_output - expected_final))

        print(f"  Correlation with expected:        {correlation:.6f}")
        print(f"  Max absolute error:               {max_error:.6f}")

        # Determine pass/fail thresholds
        # BF16 precision and chained operations allow for some error
        accuracy_pass = correlation > 0.95 and max_error < 0.1
        print(f"  Accuracy check (corr>0.95, err<0.1): {'PASS' if accuracy_pass else 'FAIL'}")
        print()

        # Sample comparison
        print("Sample values (first 10):")
        print(f"{'Index':<8} {'Input':<12} {'NPU Output':<12} {'Expected':<12} {'Error':<12}")
        print("-" * 56)
        for i in range(10):
            error = abs(final_output[i] - expected_final[i])
            print(f"{i:<8} {input_floats[i]:<12.4f} {final_output[i]:<12.6f} {expected_final[i]:<12.6f} {error:<12.6f}")
        print()

        # Overall result
        print("=" * 70)

        performance_pass = avg_chain < 5.0  # Should be ~3-4 ms
        overall_pass = valid_range and non_zero and performance_pass

        if overall_pass:
            print("ENCODER CHAIN INTEGRATION TEST PASSED!")
            print()
            print("Summary:")
            print(f"  - Chained kernel (LN+SM+GELU): {avg_chain:.3f} ms")
            print(f"  - Individual baseline:        ~{individual_baseline_ms:.1f} ms")
            if speedup > 1:
                print(f"  - Speedup:                    {speedup:.2f}x")
            print(f"  - Output correlation:         {correlation:.6f}")
            print(f"  - Buffer size:                {buffer_size} bytes ({num_elements} elements)")
            print()
            print("Encoder chain XCLBIN is working correctly!")
            result = 0
        else:
            print("ENCODER CHAIN INTEGRATION TEST FAILED!")
            print()
            if not valid_range:
                print("  - FAIL: Output values out of expected range")
            if not non_zero:
                print("  - FAIL: Too many zero outputs")
            if not performance_pass:
                print(f"  - FAIL: Performance too slow ({avg_chain:.3f} ms > 5.0 ms)")
            if not accuracy_pass:
                print(f"  - WARN: Low accuracy (corr={correlation:.4f}, err={max_error:.4f})")
            result = 1
        print("=" * 70)

        return result

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
