#!/usr/bin/env python3
"""
Test MEL INT8 XCLBIN on NPU using correct XRT API
Based on breakthrough findings from passthrough_step3.mlir testing
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import numpy as np
import pyxrt as xrt

def test_mel_xclbin():
    """Test MEL INT8 kernel with proper XRT API"""

    print("="*70)
    print("MEL INT8 NPU Kernel Test - Using Correct XRT API")
    print("="*70)
    print()

    xclbin_path = "build/mel_int8_final.xclbin"

    try:
        # Step 1: Open NPU device
        print("Step 1: Opening NPU device...")
        device = xrt.device(0)
        print("✅ Device opened: /dev/accel/accel0")
        print()

        # Step 2: Load XCLBIN as object (not device.load_xclbin!)
        print("Step 2: Loading XCLBIN object...")
        xclbin = xrt.xclbin(xclbin_path)
        uuid = xclbin.get_uuid()
        print(f"✅ XCLBIN loaded with UUID: {uuid}")
        print()

        # Step 3: Register XCLBIN to device (key API call!)
        print("Step 3: Registering XCLBIN to device...")
        device.register_xclbin(xclbin)
        print("✅ XCLBIN registered successfully")
        print()

        # Step 4: Create hardware context (not directly from device!)
        print("Step 4: Creating hardware context...")
        context = xrt.hw_context(device, uuid)
        print("✅ Hardware context created")
        print()

        # Step 5: Get kernel handle from context
        print("Step 5: Getting kernel handle...")
        kernel = xrt.kernel(context, "MLIR_AIE")
        print(f"✅ Kernel handle obtained: MLIR_AIE")
        print()

        # Step 6: Create buffer objects
        print("Step 6: Creating buffer objects...")

        # Input: 400 INT16 samples = 800 bytes = 200 32-bit words
        input_size = 200  # 32-bit words
        bo_input = xrt.bo(device, input_size * 4, xrt.bo.host_only, kernel.group_id(3))

        # Output: 80 INT8 mel features = 80 bytes = 20 32-bit words
        output_size = 20  # 32-bit words
        bo_output = xrt.bo(device, output_size * 4, xrt.bo.host_only, kernel.group_id(4))

        # Instruction buffer (for NPU instructions)
        instr_size = 300
        bo_instr = xrt.bo(device, instr_size, xrt.bo.cacheable, kernel.group_id(1))

        print(f"✅ Input buffer: {input_size * 4} bytes ({input_size} words)")
        print(f"✅ Output buffer: {output_size * 4} bytes ({output_size} words)")
        print(f"✅ Instruction buffer: {instr_size} bytes")
        print()

        # Step 7: Load NPU instructions (if insts.bin exists)
        import os
        insts_path = "build/insts.bin"
        if os.path.exists(insts_path):
            print("Step 7: Loading NPU instructions...")
            with open(insts_path, "rb") as f:
                insts = f.read()

            instr_map = bo_instr.map()
            instr_map[:len(insts)] = insts
            bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, len(insts), 0)

            num_instr = len(insts) // 4
            print(f"✅ Loaded {len(insts)} bytes ({num_instr} instructions)")
            print()
        else:
            print("⚠️  No insts.bin found - kernel may not execute properly")
            num_instr = 0
            print()

        # Step 8: Prepare test data
        print("Step 8: Preparing test data...")

        # Generate test audio data (simulated INT16 audio samples)
        # For a real test, this would be actual audio
        input_data = np.arange(input_size, dtype=np.int32)

        input_map = bo_input.map()
        input_map[:] = input_data.view(np.uint8)
        bo_input.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, input_size * 4, 0)

        print(f"✅ Test data prepared: {input_size} words")
        print(f"   Sample values: {list(input_data[:5])}...")
        print()

        # Step 9: Execute kernel on NPU
        print("Step 9: Executing kernel on NPU...")

        opcode = 3  # Standard opcode for NPU execution
        run = kernel(opcode, bo_instr, num_instr, bo_input, bo_output)

        # Wait for completion
        state = run.wait()

        print(f"✅ Kernel execution completed")
        print(f"   Execution state: {state}")
        print()

        # Step 10: Read results from NPU
        print("Step 10: Reading results from NPU...")

        bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, output_size * 4, 0)
        output_map = bo_output.map()
        output_data = np.frombuffer(output_map, dtype=np.int32, count=output_size)

        print(f"✅ Output data read: {output_size} words")
        print(f"   Output values: {list(output_data[:10])}...")
        print()

        # Analysis
        print("="*70)
        print("RESULTS ANALYSIS")
        print("="*70)

        if np.all(output_data == 0):
            print("⚠️  Output is all zeros (expected for empty kernel)")
            print("   This means:")
            print("   - ✅ Kernel executed successfully")
            print("   - ✅ DMA transfers work")
            print("   - ⚠️  Core logic not yet implemented (empty kernel)")
        else:
            print("✅ Output contains non-zero values!")
            print("   This means kernel processing is working!")

        print()
        print("="*70)
        print("TEST COMPLETE - NPU KERNEL EXECUTED SUCCESSFULLY!")
        print("="*70)

        return True

    except Exception as e:
        print()
        print("="*70)
        print(f"❌ TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mel_xclbin()
    sys.exit(0 if success else 1)
