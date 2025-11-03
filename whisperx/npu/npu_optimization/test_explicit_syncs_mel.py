#!/usr/bin/env python3
"""
Test Explicit Buffer Synchronization Patterns for NPU Mel Kernel

MISSION: Test if explicit buffer syncs and device_only flags fix the NPU zeros issue.

PROBLEM: All NPU kernels return zeros despite fast execution times
ROOT CAUSE HYPOTHESIS: Missing explicit buffer syncs after kernel execution

THIS SCRIPT TESTS THREE VARIATIONS:
  Variation A: host_only + explicit syncs (current approach)
  Variation B: device_only + explicit syncs (NEW - may fix zeros!)
  Variation C: normal (no flags) + explicit syncs

Each variation tests:
  1. Explicit sync TO device before execution
  2. Kernel execution with wait
  3. CRITICAL: Explicit sync FROM device after execution
  4. Then read buffer

Expected results:
  - Variation A: May still return zeros (same as current)
  - Variation B: May produce non-zero output (device_only could fix DMA)
  - Variation C: May work as middle ground

Date: October 31, 2025
Team Lead: Buffer Synchronization Testing Expert
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
import time
import os

# Set debug logging BEFORE importing XRT
os.environ['XRT_LOG_LEVEL'] = 'debug'

# Configuration
XCLBIN_PATH = '/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/build_fixed_v3/mel_fixed_v3.xclbin'
INSTR_PATH = '/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/build_fixed_v3/insts_v3.bin'
AUDIO_PATH = '/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/test_audio_jfk.wav'
WINDOW_SIZE = 400  # INT16 audio samples
MEL_BINS = 80
INPUT_SIZE = 800    # 400 INT16 = 800 bytes
OUTPUT_SIZE = 80    # 80 INT8 = 80 bytes
OPCODE = 3          # Kernel opcode

def load_real_audio():
    """Load real audio from test file"""
    try:
        import librosa
        print(f"Loading audio from: {AUDIO_PATH}")
        audio, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)

        # Find a section with actual audio (not silence)
        # Skip first few seconds which may be silence
        start_offset = 8000  # Skip 0.5 seconds of potential silence
        audio_section = audio[start_offset:start_offset + WINDOW_SIZE]

        if len(audio_section) < WINDOW_SIZE:
            audio_section = np.pad(audio_section, (0, WINDOW_SIZE - len(audio_section)))

        # Convert to INT16
        audio_int16 = (audio_section * 32767).astype(np.int16)

        print(f"✅ Audio loaded: {len(audio_int16)} samples (offset {start_offset})")
        print(f"   Range: [{audio_int16.min()}, {audio_int16.max()}]")
        non_zero = np.count_nonzero(audio_int16)
        print(f"   Non-zero samples: {non_zero}/{len(audio_int16)} ({100*non_zero/len(audio_int16):.1f}%)")
        return audio_int16
    except Exception as e:
        print(f"⚠️  Failed to load audio: {e}")
        print("   Using synthetic sine wave instead")
        return generate_synthetic_audio()

def generate_synthetic_audio():
    """Generate synthetic test audio (1kHz sine wave - matches quick_correlation_test)"""
    sr = 16000
    # Generate 1 second of 1kHz sine wave (same as quick_correlation_test)
    audio = np.sin(2 * np.pi * 1000 * np.linspace(0, 1, sr, endpoint=False)).astype(np.float32)
    # Take first 400 samples
    audio_frame = audio[:WINDOW_SIZE]
    # Convert to INT16
    audio_int16 = (audio_frame * 32767).astype(np.int16)
    print(f"Generated 1kHz sine wave: {len(audio_int16)} samples")
    print(f"   Range: [{audio_int16.min()}, {audio_int16.max()}]")
    return audio_int16

def load_instructions():
    """Load instruction binary from file"""
    try:
        with open(INSTR_PATH, 'rb') as f:
            instr_bin = f.read()
        print(f"✅ Instructions loaded: {len(instr_bin)} bytes")
        return instr_bin
    except Exception as e:
        print(f"❌ Failed to load instructions: {e}")
        sys.exit(1)

def analyze_output(output_data, variation_name):
    """Analyze output data and print statistics"""
    print(f"\n{'='*70}")
    print(f"VARIATION {variation_name} RESULTS")
    print(f"{'='*70}")

    # Basic statistics
    non_zero_count = np.count_nonzero(output_data)
    non_zero_percent = (non_zero_count / len(output_data)) * 100

    print(f"Non-zero values: {non_zero_count}/{len(output_data)} ({non_zero_percent:.1f}%)")
    print(f"Output range: [{output_data.min()}, {output_data.max()}]")
    print(f"Mean: {output_data.mean():.2f}, Std: {output_data.std():.2f}")
    print(f"First 10 values: {output_data[:10]}")
    print(f"Last 10 values: {output_data[-10:]}")

    # Verdict
    if non_zero_percent > 80:
        print(f"\n✅ SUCCESS! {variation_name} produces non-zero output!")
        print(f"   This variation likely FIXES the zeros issue!")
        return True
    elif non_zero_percent > 20:
        print(f"\n⚠️  PARTIAL: {variation_name} produces some non-zero output")
        print(f"   May need additional fixes")
        return False
    else:
        print(f"\n❌ FAILURE: {variation_name} still returns mostly zeros")
        print(f"   This variation does not fix the issue")
        return False

def test_variation_a_host_only(device, xclbin_obj, uuid, hw_ctx, kernel, audio_int16, instr_bin):
    """
    VARIATION A: host_only + explicit syncs (current approach)

    This is the standard approach used in existing code.
    Uses xrt.bo.flags.host_only for data buffers.
    Uses xrt.bo.flags.cacheable for instruction buffer (required).
    """
    print(f"\n{'='*70}")
    print("TESTING VARIATION A: host_only + explicit syncs")
    print(f"{'='*70}")

    try:
        # Create buffers with host_only flag for data, cacheable for instructions
        print("Creating buffers...")
        n_insts = len(instr_bin)
        instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
        input_bo = xrt.bo(device, INPUT_SIZE, xrt.bo.flags.host_only, kernel.group_id(3))
        output_bo = xrt.bo(device, OUTPUT_SIZE, xrt.bo.flags.host_only, kernel.group_id(4))
        print("✅ Buffers created")

        # Write instructions
        print("Writing instructions...")
        instr_bo.write(instr_bin, 0)
        instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

        # Write input data (as raw bytes, not packed words)
        print("Writing input audio...")
        input_data = audio_int16.tobytes()
        input_bo.write(input_data, 0)

        # EXPLICIT SYNC TO DEVICE (before execution)
        print("⚡ Syncing input TO device (explicit)...")
        input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, INPUT_SIZE, 0)
        print("✅ Input synced to device")

        # Execute kernel with opcode
        print("Executing kernel on NPU...")
        start = time.perf_counter()
        run = kernel(OPCODE, instr_bo, n_insts, input_bo, output_bo)
        state = run.wait(10000)
        end = time.perf_counter()

        if state == xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            print(f"✅ Kernel completed in {(end-start)*1000:.3f} ms")
        else:
            print(f"❌ Kernel failed with state: {state}")
            return False

        # CRITICAL: EXPLICIT SYNC FROM DEVICE (after execution)
        print("⚡ Syncing output FROM device (EXPLICIT - CRITICAL)...")
        output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, OUTPUT_SIZE, 0)
        print("✅ Output synced from device")

        # Read output
        print("Reading output data...")
        output_data = np.frombuffer(output_bo.read(OUTPUT_SIZE, 0), dtype=np.int8)

        # Analyze
        success = analyze_output(output_data, "A (host_only)")

        return success

    except Exception as e:
        print(f"❌ Variation A failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_variation_b_device_only(device, xclbin_obj, uuid, hw_ctx, kernel, audio_int16, instr_bin):
    """
    VARIATION B: SKIPPED - device_only not supported on Phoenix NPU

    Phoenix NPU doesn't support device_only flag.
    This is expected behavior for this platform.
    """
    print(f"\n{'='*70}")
    print("TESTING VARIATION B: device_only + explicit syncs")
    print(f"{'='*70}")
    print("⚠️  SKIPPED: device_only flag not supported on Phoenix NPU")
    print("   This is expected - Phoenix uses host_only or cacheable")
    return False

def test_variation_c_cacheable(device, xclbin_obj, uuid, hw_ctx, kernel, audio_int16, instr_bin):
    """
    VARIATION C: cacheable + explicit syncs

    This uses xrt.bo.flags.cacheable for ALL buffers.
    May provide better DMA performance than host_only.
    """
    print(f"\n{'='*70}")
    print("TESTING VARIATION C: cacheable + explicit syncs")
    print(f"{'='*70}")

    try:
        # Create ALL buffers with cacheable flag
        print("Creating buffers with cacheable flag...")
        n_insts = len(instr_bin)
        instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
        input_bo = xrt.bo(device, INPUT_SIZE, xrt.bo.flags.cacheable, kernel.group_id(3))
        output_bo = xrt.bo(device, OUTPUT_SIZE, xrt.bo.flags.cacheable, kernel.group_id(4))
        print("✅ Buffers created")

        # Write instructions
        print("Writing instructions...")
        instr_bo.write(instr_bin, 0)
        instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

        # Write input data
        print("Writing input audio...")
        input_data = audio_int16.tobytes()
        input_bo.write(input_data, 0)

        # EXPLICIT SYNC TO DEVICE (before execution)
        print("⚡ Syncing input TO device (explicit)...")
        input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, INPUT_SIZE, 0)
        print("✅ Input synced to device")

        # Execute kernel with opcode
        print("Executing kernel on NPU...")
        start = time.perf_counter()
        run = kernel(OPCODE, instr_bo, n_insts, input_bo, output_bo)
        state = run.wait(10000)
        end = time.perf_counter()

        if state == xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            print(f"✅ Kernel completed in {(end-start)*1000:.3f} ms")
        else:
            print(f"❌ Kernel failed with state: {state}")
            return False

        # CRITICAL: EXPLICIT SYNC FROM DEVICE (after execution)
        print("⚡ Syncing output FROM device (EXPLICIT - CRITICAL)...")
        output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, OUTPUT_SIZE, 0)
        print("✅ Output synced from device")

        # Read output
        print("Reading output data...")
        output_data = np.frombuffer(output_bo.read(OUTPUT_SIZE, 0), dtype=np.int8)

        # Analyze
        success = analyze_output(output_data, "C (cacheable)")

        return success

    except Exception as e:
        print(f"❌ Variation C failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test orchestrator"""
    print(f"\n{'='*70}")
    print("EXPLICIT BUFFER SYNCHRONIZATION TEST")
    print("Testing 3 variations to fix NPU zeros issue")
    print(f"{'='*70}")

    print(f"\nKernel: {XCLBIN_PATH}")
    print(f"Audio: {AUDIO_PATH}")
    print(f"XRT_LOG_LEVEL: {os.environ.get('XRT_LOG_LEVEL', 'not set')}")

    # Load audio and instructions once
    # Use synthetic audio (same as quick_correlation_test which works)
    audio_int16 = generate_synthetic_audio()
    instr_bin = load_instructions()

    # Initialize NPU once
    print(f"\n{'='*70}")
    print("INITIALIZING NPU")
    print(f"{'='*70}")

    try:
        device = xrt.device(0)
        print("✅ NPU device opened")

        xclbin_obj = xrt.xclbin(XCLBIN_PATH)
        uuid = xclbin_obj.get_uuid()
        print(f"✅ XCLBIN loaded (UUID: {uuid})")

        device.register_xclbin(xclbin_obj)
        print("✅ XCLBIN registered")

        hw_ctx = xrt.hw_context(device, uuid)
        print("✅ Hardware context created")

        kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
        print("✅ Kernel handle obtained")

    except Exception as e:
        print(f"❌ NPU initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test all three variations
    results = {}

    # Variation A: host_only + explicit syncs
    results['A'] = test_variation_a_host_only(device, xclbin_obj, uuid, hw_ctx, kernel, audio_int16, instr_bin)

    # Variation B: device_only + explicit syncs (SKIPPED - not supported)
    results['B'] = test_variation_b_device_only(device, xclbin_obj, uuid, hw_ctx, kernel, audio_int16, instr_bin)

    # Variation C: cacheable + explicit syncs
    results['C'] = test_variation_c_cacheable(device, xclbin_obj, uuid, hw_ctx, kernel, audio_int16, instr_bin)

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    print("\nResults:")
    for var, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  Variation {var}: {status}")

    successful_vars = [var for var, success in results.items() if success]

    if successful_vars:
        print(f"\n🎉 BREAKTHROUGH! The following variation(s) produce non-zero output:")
        for var in successful_vars:
            print(f"   - Variation {var}")
        print(f"\n✅ Use this pattern in production code!")
    else:
        print(f"\n❌ None of the variations produced non-zero output")
        print(f"   Additional investigation needed:")
        print(f"   - Check if kernel is actually computing (not just DMA)")
        print(f"   - Verify instruction buffer format")
        print(f"   - Test with different buffer sizes")
        print(f"   - Check XRT debug logs for DMA issues")

    print(f"\n{'='*70}")

    return any(results.values())

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
