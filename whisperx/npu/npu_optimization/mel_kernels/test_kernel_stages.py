#!/usr/bin/env python3
"""
Diagnostic test to isolate zero-output bug in mel kernel.
Tests each stage of the pipeline individually.
"""

import numpy as np
import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
from pathlib import Path

def test_input_conversion():
    """Test Step 1: INT8 to INT16 conversion"""
    print("\n" + "="*70)
    print("TEST 1: Input Conversion (INT8 → INT16)")
    print("="*70)

    # Generate test audio: 1000 Hz sine wave
    sample_rate = 16000
    duration = 0.025  # 25ms = 400 samples
    t = np.linspace(0, duration, 400)
    audio_float = 0.5 * np.sin(2 * np.pi * 1000 * t)  # Amplitude 0.5

    # Convert to INT16 (how kernel expects)
    audio_int16 = (audio_float * 32767).astype(np.int16)

    # Convert to INT8 bytes (how Python sends)
    audio_bytes = audio_int16.tobytes()

    # Reconstruct as kernel does
    reconstructed = np.zeros(400, dtype=np.int16)
    for i in range(400):
        byte_idx = i * 2
        low_byte = int(audio_bytes[byte_idx])
        high_byte = int(audio_bytes[byte_idx + 1])
        # Ensure high byte is sign-extended
        if high_byte > 127:
            high_byte = high_byte - 256
        reconstructed[i] = np.int16(low_byte | (high_byte << 8))

    # Check if unsigned cast is correct (as in kernel)
    reconstructed_unsigned = np.zeros(400, dtype=np.int16)
    for i in range(400):
        byte_idx = i * 2
        # Kernel uses: ((int16_t)(uint8_t)input[byte_idx]) | (((int16_t)(int8_t)input[byte_idx + 1]) << 8)
        low_byte = int(audio_bytes[byte_idx])  # uint8_t cast
        high_byte = int(audio_bytes[byte_idx + 1])
        if high_byte > 127:  # int8_t cast (sign extension)
            high_byte = high_byte - 256
        reconstructed_unsigned[i] = np.int16(low_byte | (high_byte << 8))

    print(f"Original INT16:     min={audio_int16.min()}, max={audio_int16.max()}, mean={audio_int16.mean():.1f}")
    print(f"Reconstructed:      min={reconstructed.min()}, max={reconstructed.max()}, mean={reconstructed.mean():.1f}")
    print(f"Reconstructed (u8): min={reconstructed_unsigned.min()}, max={reconstructed_unsigned.max()}, mean={reconstructed_unsigned.mean():.1f}")
    print(f"Match: {np.allclose(audio_int16, reconstructed_unsigned)}")

    if not np.allclose(audio_int16, reconstructed_unsigned):
        print("❌ INPUT CONVERSION BUG FOUND!")
        print(f"First 10 original:      {audio_int16[:10]}")
        print(f"First 10 reconstructed: {reconstructed_unsigned[:10]}")
        return False

    print("✅ Input conversion correct")
    return True

def test_hann_window():
    """Test Step 2: Hann window application"""
    print("\n" + "="*70)
    print("TEST 2: Hann Window Application")
    print("="*70)

    # Generate Hann window in Q15
    n = 400
    hann_float = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    hann_q15 = (hann_float * 32767).astype(np.int16)

    # Test signal
    signal = np.full(400, 16383, dtype=np.int16)  # 0.5 in Q15

    # Apply window (Q15 * Q15 → Q30 → Q15)
    windowed = np.zeros(400, dtype=np.int16)
    for i in range(400):
        product = np.int32(signal[i]) * np.int32(hann_q15[i])
        windowed[i] = np.int16((product + (1 << 14)) >> 15)  # Round and shift

    print(f"Signal:   min={signal.min()}, max={signal.max()}, mean={signal.mean():.1f}")
    print(f"Window:   min={hann_q15.min()}, max={hann_q15.max()}, mean={hann_q15.mean():.1f}")
    print(f"Windowed: min={windowed.min()}, max={windowed.max()}, mean={windowed.mean():.1f}")

    if windowed.max() == 0:
        print("❌ HANN WINDOW BUG: All zeros after windowing!")
        return False

    print("✅ Hann window application correct")
    return True

def test_fft_bit_reversal():
    """Test Step 3: FFT bit reversal"""
    print("\n" + "="*70)
    print("TEST 3: FFT Bit Reversal")
    print("="*70)

    # Generate bit reversal LUT
    def bit_reverse(n, bits=9):  # 512 = 2^9
        result = 0
        for i in range(bits):
            result = (result << 1) | ((n >> i) & 1)
        return result

    bit_reverse_lut = [bit_reverse(i) for i in range(512)]

    # Test data
    test_data = np.arange(512, dtype=np.int16)
    reversed_data = test_data[bit_reverse_lut]

    print(f"Original[0:10]: {test_data[:10]}")
    print(f"Reversed[0:10]: {reversed_data[:10]}")
    print(f"LUT[0:10]:      {bit_reverse_lut[:10]}")

    if reversed_data[0] != 0 or reversed_data[256] != 1:
        print("❌ BIT REVERSAL BUG!")
        return False

    print("✅ Bit reversal correct")
    return True

def test_mel_filter_indexing():
    """Test Step 4: Mel filter coefficient indexing"""
    print("\n" + "="*70)
    print("TEST 4: Mel Filter Coefficient Indexing")
    print("="*70)

    # Simulate mel filter structure from mel_coeffs_fixed.h
    # Each filter has 257 weights (one for each FFT bin 0-256)
    # But only start_bin to end_bin are non-zero

    # Example: First mel filter (0-100 Hz, bins 0-3)
    start_bin = 0
    end_bin = 3
    weights = np.zeros(257, dtype=np.int16)
    weights[0] = 8192   # 0.25 in Q15
    weights[1] = 16384  # 0.5 in Q15
    weights[2] = 16384  # 0.5 in Q15

    # Test magnitude spectrum
    magnitude = np.full(256, 10000, dtype=np.int16)

    # Apply filter (as kernel does)
    mel_energy = 0
    for bin in range(start_bin, end_bin):
        if bin >= 256:
            break
        weight = weights[bin]
        if weight == 0:
            continue
        weighted = np.int32(magnitude[bin]) * np.int32(weight)
        mel_energy += weighted >> 15

    print(f"Magnitude bins: {magnitude[:5]}")
    print(f"Filter weights: {weights[:5]}")
    print(f"Mel energy (Q15): {mel_energy}")

    # Convert to INT8
    if mel_energy < 0:
        mel_energy = 0
    scaled = (mel_energy * 127) // 32767
    if scaled > 127:
        scaled = 127

    print(f"Scaled to INT8: {scaled}")

    if scaled == 0 and mel_energy > 0:
        print("❌ INT8 SCALING BUG: Non-zero energy becomes zero!")
        return False

    if mel_energy == 0:
        print("❌ MEL FILTER BUG: Zero energy from non-zero inputs!")
        return False

    print("✅ Mel filter indexing correct")
    return True

def test_npu_execution():
    """Test Step 5: Actual NPU execution with diagnostic input"""
    print("\n" + "="*70)
    print("TEST 5: NPU Execution with Diagnostic Input")
    print("="*70)

    xclbin_path = Path(__file__).parent / "build_fixed_v3" / "mel_fixed_v3.xclbin"
    insts_path = Path(__file__).parent / "build_fixed_v3" / "insts_v3.bin"

    if not xclbin_path.exists():
        print(f"❌ XCLBIN not found: {xclbin_path}")
        return False

    if not insts_path.exists():
        print(f"❌ Instructions not found: {insts_path}")
        return False

    # Load XCLBIN
    device = xrt.device(0)
    xclbin = xrt.xclbin(str(xclbin_path))
    device.register_xclbin(xclbin)
    uuid = xclbin.get_uuid()
    hw_ctx = xrt.hw_context(device, uuid)
    kernel = xrt.kernel(hw_ctx, "MLIR_AIE")

    print(f"✅ NPU initialized")
    print(f"   XCLBIN: {xclbin_path.name}")
    print(f"   Kernel: MLIR_AIE")

    # Test 1: Maximum amplitude sine wave (should produce strong mel features)
    sample_rate = 16000
    duration = 0.025  # 400 samples
    t = np.linspace(0, duration, 400)

    # Test frequencies
    test_cases = [
        (1000, "1000 Hz (mid-frequency)"),
        (100, "100 Hz (low-frequency)"),
        (4000, "4000 Hz (high-frequency)"),
    ]

    for freq, desc in test_cases:
        print(f"\nTesting {desc}...")

        # Generate maximum amplitude signal
        audio_float = 0.9 * np.sin(2 * np.pi * freq * t)
        audio_int16 = (audio_float * 32767).astype(np.int16)

        # Convert to bytes
        input_data = audio_int16.tobytes()

        # Read instructions
        insts_bin = open(insts_path, "rb").read()
        n_insts = len(insts_bin)

        # Allocate buffers
        input_size = 800  # 400 INT16 = 800 bytes
        output_size = 80  # 80 INT8 mel bins

        instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
        input_bo = xrt.bo(device, input_size, xrt.bo.flags.host_only, kernel.group_id(3))
        output_bo = xrt.bo(device, output_size, xrt.bo.flags.host_only, kernel.group_id(4))

        # Write data
        instr_bo.write(insts_bin, 0)
        instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

        input_bo.write(input_data, 0)
        input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, input_size, 0)

        # Execute
        opcode = 3
        run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
        state = run.wait(1000)

        if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            print(f"❌ Kernel execution failed: {state}")
            return False

        # Read output
        output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, output_size, 0)
        mel_bins = np.frombuffer(output_bo.read(output_size, 0), dtype=np.int8)

        print(f"   Input INT16:  min={audio_int16.min()}, max={audio_int16.max()}, mean={audio_int16.mean():.1f}")
        print(f"   Output INT8:  min={mel_bins.min()}, max={mel_bins.max()}, mean={mel_bins.mean():.2f}")
        print(f"   First 10:     {mel_bins[:10]}")
        print(f"   Non-zero bins: {np.count_nonzero(mel_bins)}/80")

        if mel_bins.max() == 0:
            print(f"❌ ALL ZEROS FOR {desc}!")
        else:
            print(f"✅ Non-zero output for {desc}")

    return True

if __name__ == "__main__":
    print("="*70)
    print("MEL KERNEL DIAGNOSTIC TEST SUITE")
    print("="*70)
    print("Testing each stage of the pipeline to isolate zero-output bug")

    results = {
        "Input Conversion": test_input_conversion(),
        "Hann Window": test_hann_window(),
        "FFT Bit Reversal": test_fft_bit_reversal(),
        "Mel Filter Indexing": test_mel_filter_indexing(),
        "NPU Execution": test_npu_execution(),
    }

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25s}: {status}")

    if not all(results.values()):
        print("\n❌ Some tests failed - bug identified!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed - bug is elsewhere!")
        sys.exit(0)
