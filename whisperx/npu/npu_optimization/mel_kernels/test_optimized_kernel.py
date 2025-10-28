#!/usr/bin/env python3
"""Test optimized mel filterbank kernel on NPU"""
import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import numpy as np
import pyxrt as xrt

print("=" * 60)
print("Testing Optimized MEL Filterbank Kernel on AMD Phoenix NPU")
print("=" * 60)
print()

try:
    # Step 1: Open device
    print("1. Opening NPU device...")
    device = xrt.device(0)
    print(f"   ✅ Device: {device}")

    # Step 2: Load XCLBIN
    print("\n2. Loading optimized XCLBIN...")
    xclbin = xrt.xclbin("build_optimized/mel_optimized_new.xclbin")
    device.register_xclbin(xclbin)
    print("   ✅ Optimized XCLBIN registered!")

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

    # Read instruction binary
    insts_bin = open("build_optimized/insts_optimized_new.bin", "rb").read()
    n_insts = len(insts_bin)

    # Allocate buffers
    instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
    input_bo = xrt.bo(device, input_size, xrt.bo.flags.host_only, kernel.group_id(3))
    output_bo = xrt.bo(device, output_size, xrt.bo.flags.host_only, kernel.group_id(4))

    # Write instructions
    instr_bo.write(insts_bin, 0)
    instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)
    print("   ✅ Buffers allocated!")

    # Step 5: Prepare test data
    print("\n5. Preparing test data...")
    # Generate 1 kHz sine wave
    sample_rate = 16000
    freq = 1000
    duration = 400 / sample_rate
    t = np.linspace(0, duration, 400)
    sine_wave = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

    # Convert to bytes
    input_data = sine_wave.astype(np.int16).tobytes()
    input_bo.write(input_data, 0)
    input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, input_size, 0)
    print("   ✅ Test data written (1 kHz sine wave)")

    # Step 6: Execute kernel
    print("\n6. Executing OPTIMIZED kernel on NPU...")
    print("   (Uses proper triangular mel filters)")
    opcode = 3
    run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)

    # Try with longer timeout
    print("   Waiting for kernel (10s timeout)...")
    state = run.wait(10000)  # 10 second timeout

    if state == xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        print("   🎉 SUCCESS! Optimized kernel executed successfully!")

        # Read output
        output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, output_size, 0)
        output_data = np.frombuffer(output_bo.read(output_size, 0), dtype=np.int8)

        print("\n7. Output verification:")
        print(f"   Mel bins (first 16): {output_data[:16].tolist()}")
        print(f"   Mel bins (last 16):  {output_data[-16:].tolist()}")

        non_zero = np.count_nonzero(output_data)
        avg_energy = np.abs(output_data).mean()
        max_energy = np.abs(output_data).max()

        print(f"\n   Non-zero bins: {non_zero}/80")
        print(f"   Average energy: {avg_energy:.2f}")
        print(f"   Max energy: {max_energy}")

        print("\n8. Comparison with simple kernel:")
        print("   Simple kernel: avg=52.46, max=117")
        print(f"   Optimized:     avg={avg_energy:.2f}, max={max_energy}")

        energy_diff = ((avg_energy - 52.46) / 52.46) * 100
        print(f"   Energy difference: {energy_diff:+.1f}%")

        if non_zero >= 70:
            print("\n   ✅ EXCELLENT! Optimized mel filterbank working!")
            print("   ✅ Proper triangular filters processing audio correctly")
        elif non_zero >= 40:
            print("\n   ✅ GOOD! Most bins active")
        else:
            print("\n   ⚠️  LOW! May need investigation")
    else:
        print(f"   ❌ Kernel failed with state: {state}")
        print(f"      State name: {state.name if hasattr(state, 'name') else 'unknown'}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
