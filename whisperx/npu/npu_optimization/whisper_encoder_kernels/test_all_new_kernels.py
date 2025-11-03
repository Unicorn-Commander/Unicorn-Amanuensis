#!/usr/bin/env python3
"""
Unified Test Suite for All NPU Kernels
Tests attention 64x64, layer norm, and GELU using the WORKING mel kernel pattern

Based on successful mel_npu_execution.py pattern
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
import time
from pathlib import Path

# Test configurations
TESTS = {
    'matmul_16x16': {
        'xclbin': 'build/matmul_simple.xclbin',
        'insts': 'build/insts.bin',
        'kernel_name': 'MLIR_AIE',
        'input_size': 512,    # 2 matrices: 16x16 + 16x16 = 512 bytes
        'output_size': 256,   # 16x16 result
        'group_id_input': 3,
        'group_id_output': 4,
        'description': 'Matrix multiply C = A @ B (16x16 INT8)',
    },
    'attention_64x64': {
        'xclbin': 'build_attention_64x64/attention_64x64.xclbin',
        'insts': 'build_attention_64x64/insts.bin',
        'kernel_name': 'MLIR_AIE',
        'input_size': 12288,  # 64x64 x 3 (Q+K+V)
        'output_size': 4096,  # 64x64
        'group_id_input': 3,
        'group_id_output': 4,
        'description': 'Attention mechanism (64x64 tiles)',
    },
    'layernorm': {
        'xclbin': 'build_layernorm/layernorm_simple.xclbin',
        'insts': 'build_layernorm/insts.bin',
        'kernel_name': 'MLIR_AIE',
        'input_size': 768,    # 256 input + 256 gamma + 256 beta
        'output_size': 256,
        'group_id_input': 3,
        'group_id_output': 4,
        'description': 'Layer normalization (256 features)',
    },
    'gelu_512': {
        'xclbin': 'build_gelu/gelu_simple.xclbin',
        'insts': 'build_gelu/insts_512.bin',
        'kernel_name': 'MLIR_AIE',
        'input_size': 512,
        'output_size': 512,
        'group_id_input': 3,
        'group_id_output': 4,
        'description': 'GELU activation (512 elements)',
    },
    'gelu_2048': {
        'xclbin': 'build_gelu/gelu_2048.xclbin',
        'insts': 'build_gelu/insts_2048.bin',
        'kernel_name': 'MLIR_AIE',
        'input_size': 2048,
        'output_size': 2048,
        'group_id_input': 3,
        'group_id_output': 4,
        'description': 'GELU activation (2048 elements for FFN)',
    }
}

def test_kernel(kernel_name, config):
    """Test a single kernel using the WORKING mel pattern"""

    print("=" * 70)
    print(f"Testing: {kernel_name}")
    if 'description' in config:
        print(f"Description: {config['description']}")
    print("=" * 70)

    xclbin_path = Path(__file__).parent / config['xclbin']
    if not xclbin_path.exists():
        print(f"❌ XCLBIN not found: {xclbin_path}")
        return False

    print(f"XCLBIN: {xclbin_path}")
    print(f"Input size: {config['input_size']} bytes")
    print(f"Output size: {config['output_size']} bytes")
    print()

    try:
        # Step 1: Initialize NPU (SAME AS MEL KERNEL)
        print("Step 1: Initializing NPU...")
        device = xrt.device(0)
        print(f"✅ NPU device opened: /dev/accel/accel0")

        # Load XCLBIN
        xclbin_obj = xrt.xclbin(str(xclbin_path))
        uuid = xclbin_obj.get_uuid()
        print(f"✅ XCLBIN loaded")

        # Register XCLBIN
        device.register_xclbin(xclbin_obj)
        print(f"✅ XCLBIN registered")

        # Create hardware context
        hw_ctx = xrt.hw_context(device, uuid)
        print(f"✅ Hardware context created")

        # Get kernel
        kernel = xrt.kernel(hw_ctx, config['kernel_name'])
        print(f"✅ Kernel found: {config['kernel_name']}")
        print()

        # Step 2: Load instruction buffer (CRITICAL!)
        print("Step 2: Loading instruction buffer...")
        insts_path = Path(__file__).parent / config['insts']
        if not insts_path.exists():
            print(f"❌ Instructions file not found: {insts_path}")
            return False

        with open(insts_path, "rb") as f:
            insts_data = f.read()
        n_insts = len(insts_data)

        # Instruction buffer (group_id 1, cacheable)
        instr_bo = xrt.bo(device, n_insts,
                          xrt.bo.flags.cacheable,
                          kernel.group_id(1))
        instr_bo.write(insts_data, 0)
        instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

        print(f"✅ Instructions loaded: {n_insts} bytes (group 1)")
        print()

        # Step 3: Allocate buffers (SAME AS MEL KERNEL)
        print("Step 3: Allocating data buffers...")

        # Use group_id from config (matches MLIR specification)
        input_bo = xrt.bo(device, config['input_size'],
                          xrt.bo.flags.host_only,
                          kernel.group_id(config['group_id_input']))

        output_bo = xrt.bo(device, config['output_size'],
                           xrt.bo.flags.host_only,
                           kernel.group_id(config['group_id_output']))

        print(f"✅ Input buffer: {config['input_size']} bytes (group {config['group_id_input']})")
        print(f"✅ Output buffer: {config['output_size']} bytes (group {config['group_id_output']})")
        print()

        # Step 4: Prepare test data
        print("Step 4: Preparing test data...")
        input_data = np.random.randint(-64, 64, config['input_size'], dtype=np.int8)
        print(f"✅ Generated {config['input_size']} random INT8 values")
        print()

        # Step 5: Write input to NPU (SAME AS MEL KERNEL)
        print("Step 5: Writing input to NPU...")
        input_bo.write(input_data.tobytes(), 0)
        input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                      config['input_size'], 0)
        print(f"✅ Input synced to NPU")
        print()

        # Step 6: Execute kernel with ALL 5 arguments (CRITICAL!)
        print("Step 6: Executing kernel (warm-up)...")
        opcode = 3  # Standard opcode for NPU kernels
        run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)  # 5 arguments!
        state = run.wait(1000)  # 1 second timeout

        if state == xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            print(f"✅ Kernel execution COMPLETED")
        else:
            print(f"❌ Kernel execution failed: {state}")
            return False
        print()

        # Step 7: Performance test
        print("Step 7: Running performance test (10 iterations)...")
        times = []
        for i in range(10):
            start = time.perf_counter()
            run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)  # 5 arguments!
            state = run.wait(1000)
            end = time.perf_counter()

            if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                print(f"❌ Iteration {i} failed: {state}")
                return False

            times.append((end - start) * 1000)  # Convert to ms

        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"✅ Performance test complete")
        print(f"   Average: {avg_time:.3f} ms")
        print(f"   Std dev: {std_time:.3f} ms")
        print()

        # Step 8: Read output (SAME AS MEL KERNEL)
        print("Step 8: Reading output from NPU...")
        output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
                       config['output_size'], 0)

        output_data = np.frombuffer(output_bo.read(config['output_size'], 0),
                                    dtype=np.int8)

        print(f"✅ Output read: {config['output_size']} bytes")
        print(f"   Range: [{output_data.min()}, {output_data.max()}]")
        print(f"   Non-zero: {np.count_nonzero(output_data)}/{len(output_data)} ({100*np.count_nonzero(output_data)/len(output_data):.1f}%)")
        print(f"   Mean: {output_data.mean():.2f}")
        print()

        # Success!
        print("=" * 70)
        print(f"✅ {kernel_name}: SUCCESS!")
        print("=" * 70)
        print()

        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all kernel tests"""

    print("\n")
    print("=" * 70)
    print("NPU Kernel Test Suite - Using Working Mel Pattern")
    print("=" * 70)
    print()

    results = {}
    for kernel_name, config in TESTS.items():
        success = test_kernel(kernel_name, config)
        results[kernel_name] = success

        if not success:
            print(f"\n⚠️  {kernel_name} failed - continuing with other tests...\n")

    # Summary
    print("\n")
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for kernel_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {kernel_name}")

    total = len(results)
    passed = sum(1 for s in results.values() if s)
    print()
    print(f"Total: {passed}/{total} kernels working")
    print("=" * 70)

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
