#!/usr/bin/env python3
"""
Kernel Chain Integration Test
Tests sequential execution of multiple NPU kernels to simulate encoder operations

Chain tested: LayerNorm -> Softmax
This represents the normalization + attention score pattern in transformers

AMD Phoenix NPU - XDNA1
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

class NPUKernel:
    """Wrapper for NPU kernel execution"""
    def __init__(self, device, xclbin_path, insts_path):
        self.device = device

        # Load XCLBIN
        xclbin_obj = xrt.xclbin(xclbin_path)
        uuid = xclbin_obj.get_uuid()
        device.register_xclbin(xclbin_obj)

        # Create context and kernel
        self.hw_ctx = xrt.hw_context(device, uuid)
        self.kernel = xrt.kernel(self.hw_ctx, "MLIR_AIE")

        # Load instructions
        with open(insts_path, "rb") as f:
            self.insts = f.read()

        self.name = os.path.basename(os.path.dirname(xclbin_path))

    def allocate_buffers(self, buffer_size):
        """Allocate XRT buffers for this kernel"""
        self.bo_instr = xrt.bo(self.device, len(self.insts),
                               xrt.bo.flags.cacheable, self.kernel.group_id(1))
        self.bo_in = xrt.bo(self.device, buffer_size,
                            xrt.bo.flags.host_only, self.kernel.group_id(3))
        self.bo_out = xrt.bo(self.device, buffer_size,
                             xrt.bo.flags.host_only, self.kernel.group_id(4))

        # Sync instruction buffer
        self.bo_instr.write(self.insts, 0)
        self.bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        self.buffer_size = buffer_size

    def run(self, input_bf16):
        """Execute kernel with input data, return output"""
        # Write input
        self.bo_in.write(input_bf16, 0)
        self.bo_in.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Execute
        start = time.perf_counter()
        run = self.kernel(3, self.bo_instr, len(self.insts), self.bo_in, self.bo_out)
        run.wait()
        end = time.perf_counter()

        # Read output
        self.bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        output_bf16 = self.bo_out.read(self.buffer_size, 0).tobytes()

        return output_bf16, end - start

def main():
    print("=" * 70)
    print("Kernel Chain Integration Test - AMD Phoenix NPU")
    print("=" * 70)
    print()
    print("Testing: LayerNorm -> Softmax")
    print("This simulates: normalize -> attention scores")
    print()

    num_elements = 1024
    buffer_size = num_elements * 2  # 2048 bytes
    num_iterations = 10

    # Paths to kernel builds
    base_dir = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1"

    layernorm_xclbin = f"{base_dir}/build_layernorm/layernorm_bf16.xclbin"
    layernorm_insts = f"{base_dir}/build_layernorm/insts.bin"

    softmax_xclbin = f"{base_dir}/build_softmax_bf16/softmax_bf16.xclbin"
    softmax_insts = f"{base_dir}/build_softmax_bf16/insts.bin"

    # Verify files exist
    for path in [layernorm_xclbin, layernorm_insts, softmax_xclbin, softmax_insts]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            return 1

    try:
        # Initialize device
        print("Step 1: Initializing NPU device...")
        device = xrt.device(0)
        print("Device initialized")
        print()

        # Load both kernels
        print("Step 2: Loading kernels...")
        print("  Loading LayerNorm kernel...")
        layernorm_kernel = NPUKernel(device, layernorm_xclbin, layernorm_insts)
        layernorm_kernel.allocate_buffers(buffer_size)
        print(f"  LayerNorm loaded: {len(layernorm_kernel.insts)} bytes instructions")

        print("  Loading Softmax kernel...")
        softmax_kernel = NPUKernel(device, softmax_xclbin, softmax_insts)
        softmax_kernel.allocate_buffers(buffer_size)
        print(f"  Softmax loaded: {len(softmax_kernel.insts)} bytes instructions")
        print()

        # Prepare test data
        print("Step 3: Preparing test data...")
        # Input data representing attention logits before normalization
        input_floats = np.random.randn(num_elements).astype(np.float32) * 2.0 + 0.5
        input_bf16 = float_to_bf16(input_floats)

        # Compute expected results
        expected_after_layernorm = layernorm_ref(input_floats)
        expected_final = softmax_ref(expected_after_layernorm)

        print(f"Input: mean={np.mean(input_floats):.4f}, std={np.std(input_floats):.4f}")
        print()

        # Run kernel chain
        print("Step 4: Running kernel chain (LayerNorm -> Softmax)...")
        print()

        chain_times = []
        layernorm_times = []
        softmax_times = []

        for iteration in range(num_iterations):
            # Stage 1: LayerNorm
            stage1_output_bf16, ln_time = layernorm_kernel.run(input_bf16)
            layernorm_times.append(ln_time)

            # Stage 2: Softmax (uses LayerNorm output as input)
            stage2_output_bf16, sm_time = softmax_kernel.run(stage1_output_bf16)
            softmax_times.append(sm_time)

            chain_times.append(ln_time + sm_time)

            if iteration == num_iterations - 1:
                # Keep final outputs for verification
                layernorm_output = bf16_to_float(stage1_output_bf16)
                final_output = bf16_to_float(stage2_output_bf16)

        print("Kernel chain execution complete")
        print()

        # Performance results
        print("Step 5: Performance Results:")
        print()

        avg_ln = np.mean(layernorm_times) * 1000
        avg_sm = np.mean(softmax_times) * 1000
        avg_chain = np.mean(chain_times) * 1000

        print("Individual Kernel Times:")
        print(f"  LayerNorm: {avg_ln:.3f} ms (avg), {np.min(layernorm_times)*1000:.3f} ms (min)")
        print(f"  Softmax:   {avg_sm:.3f} ms (avg), {np.min(softmax_times)*1000:.3f} ms (min)")
        print()

        print("Chain Performance:")
        print(f"  Total chain time: {avg_chain:.3f} ms")
        print(f"  Min chain time:   {np.min(chain_times)*1000:.3f} ms")
        print(f"  Std deviation:    {np.std(chain_times)*1000:.3f} ms")
        print()

        # Throughput
        elements_per_sec = num_elements / (avg_chain / 1000)
        print(f"  Chain throughput: {elements_per_sec/1e6:.2f} M elements/second")
        print()

        # Accuracy verification
        print("Step 6: Accuracy Verification:")
        print()

        # Stage 1: LayerNorm accuracy
        ln_max_error = np.max(np.abs(layernorm_output - expected_after_layernorm))
        ln_correlation = np.corrcoef(layernorm_output, expected_after_layernorm)[0, 1]

        print("LayerNorm Stage:")
        print(f"  Output: mean={np.mean(layernorm_output):.6f}, std={np.std(layernorm_output):.6f}")
        print(f"  Max error: {ln_max_error:.6f}")
        print(f"  Correlation: {ln_correlation:.6f}")
        ln_pass = ln_max_error < 0.05 and ln_correlation > 0.999
        print(f"  Status: {'PASS' if ln_pass else 'FAIL'}")
        print()

        # Stage 2: Final chain accuracy (vs expected softmax of normalized input)
        final_max_error = np.max(np.abs(final_output - expected_final))
        final_correlation = np.corrcoef(final_output, expected_final)[0, 1]

        print("Full Chain (LayerNorm -> Softmax):")
        print(f"  Output sum: {np.sum(final_output):.6f} (should be ~1.0)")
        print(f"  Max error: {final_max_error:.6f}")
        print(f"  Correlation: {final_correlation:.6f}")
        chain_pass = final_max_error < 0.05 and final_correlation > 0.99
        print(f"  Status: {'PASS' if chain_pass else 'FAIL'}")
        print()

        # Sample comparison
        print("Sample values (first 5):")
        print(f"{'Stage':<15} {'Input':<10} {'LN Out':<10} {'SM Out':<10} {'Expected':<10}")
        print("-" * 55)
        for i in range(5):
            print(f"{i:<15} {input_floats[i]:<10.4f} {layernorm_output[i]:<10.4f} {final_output[i]:<10.6f} {expected_final[i]:<10.6f}")
        print()

        # Overall result
        print("=" * 70)
        if ln_pass and chain_pass:
            print("KERNEL CHAIN INTEGRATION TEST PASSED!")
            print()
            print("Summary:")
            print(f"  - LayerNorm: {avg_ln:.3f} ms, correlation {ln_correlation:.6f}")
            print(f"  - Softmax:   {avg_sm:.3f} ms")
            print(f"  - Total:     {avg_chain:.3f} ms for 1024 elements")
            print()
            print("Ready for encoder layer integration!")
            result = 0
        else:
            print("KERNEL CHAIN INTEGRATION TEST FAILED!")
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
