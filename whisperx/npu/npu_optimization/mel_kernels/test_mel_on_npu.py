#!/usr/bin/env python3
"""Test MEL kernel XCLBIN on Phoenix NPU"""
import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import numpy as np
import pyxrt as xrt

print("=" * 60)
print("Testing MEL Kernel on AMD Phoenix NPU")
print("=" * 60)
print()

try:
    # Step 1: Open device
    print("1. Opening NPU device...")
    device = xrt.device(0)
    print(f"   ✅ Device: {device}")
    
    # Step 2: Load XCLBIN with FFT
    print("\n2. Loading FFT XCLBIN...")
    xclbin = xrt.xclbin("build_fft/mel_fft_final.xclbin")
    device.register_xclbin(xclbin)
    print("   ✅ FFT XCLBIN registered!")

    # Step 3: Get kernel
    print("\n3. Getting kernel...")
    uuid = xclbin.get_uuid()
    hw_ctx = xrt.hw_context(device, uuid)
    kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
    print(f"   ✅ Kernel: {kernel}")

    # Step 4: Allocate buffers
    print("\n4. Allocating buffers...")
    input_size = 800  # 800 INT8 bytes (400 INT16 samples)
    output_size = 80   # 80 INT8 bytes (80 mel bins)

    # Read FFT instruction binary
    insts_bin = open("build_fft/insts_fft.bin", "rb").read()
    n_insts = len(insts_bin)

    # Allocate buffers with correct group IDs
    # group_id(1) = instruction buffer (SRAM)
    # group_id(3) = input buffer (HOST)
    # group_id(4) = output buffer (HOST)
    instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
    input_bo = xrt.bo(device, input_size, xrt.bo.flags.host_only, kernel.group_id(3))
    output_bo = xrt.bo(device, output_size, xrt.bo.flags.host_only, kernel.group_id(4))

    # Write instructions
    instr_bo.write(insts_bin, 0)
    instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)
    print("   ✅ Buffers allocated!")

    # Step 5: Run FFT kernel
    print("\n5. Executing FFT kernel on NPU...")
    # Generate test audio: simple sine wave (1 kHz at 16 kHz sample rate)
    # 400 INT16 samples = 800 bytes
    sample_rate = 16000
    freq = 1000  # 1 kHz tone
    duration = 400 / sample_rate  # ~25ms
    t = np.linspace(0, duration, 400)
    sine_wave = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

    # Convert INT16 to bytes (little-endian)
    input_data = sine_wave.astype(np.int16).tobytes()
    input_bo.write(input_data, 0)
    input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, input_size, 0)

    # Kernel invocation: opcode=3, instr_bo, n_insts, data_buffers...
    opcode = 3  # NPU execution opcode
    run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
    state = run.wait(5000)  # 5 second timeout (increased for FFT)

    if state == xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        print("   ✅ FFT Kernel executed successfully!")

        # Read output
        output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, output_size, 0)
        output_data = np.frombuffer(output_bo.read(output_size, 0), dtype=np.int8)

        print("\n6. FFT Output verification:")
        print(f"   Input: 1 kHz sine wave, {sample_rate} Hz sample rate")
        print(f"   Output mel bins (first 16): {output_data[:16]}")
        print(f"   Output mel bins (last 16):  {output_data[-16:]}")

        # Check for non-zero output (FFT should produce energy)
        non_zero_count = np.count_nonzero(output_data)
        print(f"\n   Non-zero bins: {non_zero_count}/80")

        if non_zero_count > 10:  # Expect at least some energy in bins
            print("\n   🎉 FFT OUTPUT DETECTED! Kernel processing audio!")
        else:
            print("\n   ⚠️  Low energy detected - may need scaling adjustment")
    else:
        print(f"   ❌ Kernel failed with state: {state}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
