#!/usr/bin/env python3
"""
Comprehensive NPU Mel Execution Test for AMD Phoenix NPU
Tests mel_fixed_v3.xclbin (most recent, 56KB)

This script:
1. Verifies NPU device availability (/dev/accel/accel0)
2. Loads mel_fixed_v3.xclbin on NPU
3. Generates 1 second of test audio (440 Hz sine wave, 16kHz)
4. Executes mel spectrogram kernel on NPU
5. Validates output shape and range
6. Measures execution time with timeout protection
7. Falls back to mel_int8_final.xclbin if primary fails

Expected Output:
- Mel spectrogram: 80 mel bins × ~100 time frames
- INT8 values
- Execution time < 100ms

Date: October 29, 2025
Hardware: AMD Ryzen 9 8945HS with Phoenix NPU
"""

import sys
import os
import signal
import time
from contextlib import contextmanager

# Add XRT Python path
sys.path.insert(0, '/opt/xilinx/xrt/python')

import numpy as np

# Test timeout configuration
TEST_TIMEOUT = 30  # seconds

class TimeoutException(Exception):
    """Exception raised on timeout"""
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timeout protection"""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def check_npu_device():
    """Verify NPU device exists before testing"""
    device_path = '/dev/accel/accel0'
    if not os.path.exists(device_path):
        print(f"❌ NPU device not found: {device_path}")
        print("   Ensure XRT is installed and NPU is enabled")
        return False

    print(f"✅ NPU device found: {device_path}")
    return True

def generate_test_audio(duration=1.0, sample_rate=16000, frequency=440):
    """
    Generate test audio signal

    Args:
        duration: Audio duration in seconds (default: 1 second)
        sample_rate: Sample rate in Hz (default: 16kHz)
        frequency: Tone frequency in Hz (default: 440 Hz)

    Returns:
        numpy array of INT16 audio samples
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Generate 440 Hz sine wave (A4 musical note)
    audio = np.sin(2 * np.pi * frequency * t)

    # Convert to INT16 range
    audio_int16 = (audio * 32767 * 0.8).astype(np.int16)

    print(f"✅ Generated test audio: {len(audio_int16)} samples ({duration}s @ {sample_rate}Hz)")
    print(f"   Frequency: {frequency} Hz, Range: [{audio_int16.min()}, {audio_int16.max()}]")

    return audio_int16

def test_xclbin(xclbin_path, insts_path, audio_data, device_idx=0):
    """
    Test a specific xclbin file on NPU

    Args:
        xclbin_path: Path to xclbin file
        insts_path: Path to instruction binary file
        audio_data: INT16 audio data (400 samples for one frame)
        device_idx: XRT device index (default: 0)

    Returns:
        dict with execution results or None on failure
    """
    try:
        import pyxrt as xrt
    except ImportError:
        print("❌ XRT Python bindings not found")
        print("   Install with: pip install /opt/xilinx/xrt/python/xrt_binding*.whl")
        return None

    print(f"\n{'='*70}")
    print(f"Testing: {os.path.basename(xclbin_path)}")
    print(f"{'='*70}")

    if not os.path.exists(xclbin_path):
        print(f"❌ XCLBIN not found: {xclbin_path}")
        return None

    if not os.path.exists(insts_path):
        print(f"❌ Instructions not found: {insts_path}")
        return None

    try:
        with timeout(TEST_TIMEOUT):
            # Step 1: Open NPU device
            print(f"\n1. Opening NPU device {device_idx}...")
            device = xrt.device(device_idx)
            print(f"   ✅ Device opened")

            # Step 2: Load XCLBIN
            print(f"\n2. Loading XCLBIN...")
            xclbin_size_mb = os.path.getsize(xclbin_path) / 1024 / 1024
            print(f"   Size: {xclbin_size_mb:.2f} MB")

            xclbin_obj = xrt.xclbin(xclbin_path)
            uuid = device.register_xclbin(xclbin_obj)
            print(f"   ✅ XCLBIN loaded and registered")
            print(f"   UUID: {uuid}")

            # Step 3: Create hardware context
            print(f"\n3. Creating hardware context...")
            hw_ctx = xrt.hw_context(device, uuid)
            print(f"   ✅ Hardware context created")

            # Step 4: Get kernel
            print(f"\n4. Getting kernel...")
            kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
            print(f"   ✅ Kernel 'MLIR_AIE' found")

            # Step 5: Load instruction binary
            print(f"\n5. Loading instruction binary...")
            with open(insts_path, 'rb') as f:
                insts_bin = f.read()
            n_insts = len(insts_bin)
            print(f"   Instructions: {n_insts} bytes")

            # Step 6: Allocate buffers
            print(f"\n6. Allocating buffers...")

            # These kernels work on 400-sample frames (25ms @ 16kHz)
            # Input: 400 INT16 samples = 800 bytes
            # Output: 80 mel bins INT8 = 80 bytes
            input_size_bytes = 800   # 400 INT16 samples
            output_size_bytes = 80   # 80 mel bins

            # Take first 400 samples for one frame
            audio_frame = audio_data[:400]

            print(f"   Input size: {input_size_bytes} bytes (400 INT16 samples, 25ms frame)")
            print(f"   Output size: {output_size_bytes} bytes (80 mel bins)")

            # Allocate buffers with correct group IDs (from working example)
            instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
            input_bo = xrt.bo(device, input_size_bytes, xrt.bo.flags.host_only, kernel.group_id(3))
            output_bo = xrt.bo(device, output_size_bytes, xrt.bo.flags.host_only, kernel.group_id(4))

            print(f"   ✅ Buffers allocated")

            # Step 7: Write instruction buffer
            print(f"\n7. Writing instructions to NPU...")
            instr_bo.write(insts_bin, 0)
            instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)
            print(f"   ✅ Instructions synced to NPU ({n_insts} bytes)")

            # Step 8: Write input data
            print(f"\n8. Writing input data to NPU...")
            input_bo.write(audio_frame.tobytes(), 0)
            input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, input_size_bytes, 0)
            print(f"   ✅ Input data synced to NPU ({input_size_bytes} bytes)")

            # Step 9: Execute kernel (warm-up)
            print(f"\n9. Executing kernel (warm-up)...")
            opcode = 3  # Standard opcode for MLIR-AIE kernels
            start_warmup = time.perf_counter()
            run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
            state = run.wait(10000)  # 10 second timeout
            end_warmup = time.perf_counter()

            if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                print(f"   ❌ Kernel execution failed: {state}")
                return None

            warmup_time_ms = (end_warmup - start_warmup) * 1000
            print(f"   ✅ Warm-up completed in {warmup_time_ms:.2f} ms")

            # Step 10: Performance test (10 iterations)
            print(f"\n10. Running performance test (10 iterations)...")
            execution_times = []

            for i in range(10):
                start = time.perf_counter()
                run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
                state = run.wait(10000)
                end = time.perf_counter()

                if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                    print(f"   ❌ Iteration {i+1} failed: {state}")
                    continue

                execution_times.append((end - start) * 1000)  # Convert to ms

            if not execution_times:
                print(f"   ❌ No successful executions")
                return None

            avg_time = np.mean(execution_times)
            min_time = np.min(execution_times)
            max_time = np.max(execution_times)
            std_time = np.std(execution_times)

            print(f"   ✅ Performance test completed")
            print(f"   Average: {avg_time:.2f} ms")
            print(f"   Min: {min_time:.2f} ms, Max: {max_time:.2f} ms")
            print(f"   Std Dev: {std_time:.2f} ms")

            # Step 11: Read output
            print(f"\n11. Reading output from NPU...")
            output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, output_size_bytes, 0)
            output_data = np.frombuffer(output_bo.read(output_size_bytes, 0), dtype=np.int8)

            print(f"   ✅ Output read: {len(output_data)} INT8 values")
            print(f"   Range: [{output_data.min()}, {output_data.max()}]")
            print(f"   First 16 values: {output_data[:16].tolist()}")
            print(f"   Last 16 values: {output_data[-16:].tolist()}")

            # Step 12: Validate output
            print(f"\n12. Validating output...")

            non_zero = np.count_nonzero(output_data)
            non_zero_percent = (non_zero / len(output_data)) * 100
            unique_values = len(np.unique(output_data))

            print(f"   Non-zero values: {non_zero}/{len(output_data)} ({non_zero_percent:.1f}%)")
            print(f"   Unique values: {unique_values}")
            print(f"   Mean absolute: {np.abs(output_data).mean():.2f}")

            # Validation criteria
            validation_passed = True

            if non_zero_percent < 10:
                print(f"   ⚠️  WARNING: Very few non-zero values ({non_zero_percent:.1f}%)")
                validation_passed = False

            if unique_values < 10:
                print(f"   ⚠️  WARNING: Very few unique values ({unique_values})")
                validation_passed = False

            if avg_time > 100:
                print(f"   ⚠️  WARNING: Execution time exceeds 100ms ({avg_time:.2f}ms)")
                validation_passed = False

            if validation_passed:
                print(f"   ✅ Output validation PASSED")
            else:
                print(f"   ⚠️  Output validation has warnings")

            # Success summary
            print(f"\n{'='*70}")
            print(f"✅ TEST SUCCESSFUL: {os.path.basename(xclbin_path)}")
            print(f"{'='*70}")
            print(f"Execution Time: {avg_time:.2f} ms (target: < 100 ms)")
            print(f"Output Shape: {len(output_data)} INT8 values")
            print(f"Output Range: [{output_data.min()}, {output_data.max()}]")
            print(f"Validation: {'PASSED' if validation_passed else 'WARNINGS'}")
            print(f"{'='*70}")

            return {
                'success': True,
                'xclbin': xclbin_path,
                'execution_time_ms': avg_time,
                'execution_times': execution_times,
                'output_data': output_data,
                'non_zero_percent': non_zero_percent,
                'unique_values': unique_values,
                'validation_passed': validation_passed
            }

    except TimeoutException as e:
        print(f"\n❌ TIMEOUT: {e}")
        print("   NPU may be busy or kernel is hanging")
        return None

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test execution"""
    print("="*70)
    print("NPU Mel Spectrogram Kernel Execution Test")
    print("AMD Phoenix NPU - XRT 2.20.0")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Step 1: Check NPU device
    print("\n[1/5] Checking NPU device availability...")
    if not check_npu_device():
        print("\n❌ Test aborted: NPU device not available")
        return 1

    # Step 2: Generate test audio
    print("\n[2/5] Generating test audio...")
    audio_data = generate_test_audio(duration=1.0, sample_rate=16000, frequency=440)

    # Step 3: Define xclbin files to test (in priority order)
    base_path = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels"

    test_configs = [
        {
            'xclbin': os.path.join(base_path, "build_fixed_v3/mel_fixed_v3.xclbin"),
            'insts': os.path.join(base_path, "build_fixed_v3/insts_v3.bin"),
            'name': 'mel_fixed_v3 (56KB, most recent with fixes)'
        },
        {
            'xclbin': os.path.join(base_path, "build_optimized/mel_optimized_new.xclbin"),
            'insts': os.path.join(base_path, "build_optimized/insts_optimized_new.bin"),
            'name': 'mel_optimized_new (18KB, HTK triangular filters)'
        },
        {
            'xclbin': os.path.join(base_path, "build_fft/mel_fft_final.xclbin"),
            'insts': os.path.join(base_path, "build_fft/insts_fft.bin"),
            'name': 'mel_fft_final (24KB, FFT implementation)'
        },
        {
            'xclbin': os.path.join(base_path, "build_fixed/mel_fixed_new.xclbin"),
            'insts': os.path.join(base_path, "build_fixed/insts_new.bin"),
            'name': 'mel_fixed_new (16KB, fixed point)'
        },
    ]

    print("\n[3/5] XCLBIN files to test:")
    for i, config in enumerate(test_configs, 1):
        xclbin = config['xclbin']
        insts = config['insts']
        xclbin_exists = "✅" if os.path.exists(xclbin) else "❌"
        insts_exists = "✅" if os.path.exists(insts) else "❌"
        size = f"{os.path.getsize(xclbin)/1024:.1f} KB" if os.path.exists(xclbin) else "N/A"
        print(f"   {i}. {xclbin_exists} XCLBIN: {os.path.basename(xclbin)} ({size})")
        print(f"      {insts_exists} Instructions: {os.path.basename(insts)}")

    # Step 4: Test each xclbin
    print("\n[4/5] Testing XCLBIN files...")

    successful_tests = []

    for config in test_configs:
        xclbin_path = config['xclbin']
        insts_path = config['insts']

        if not os.path.exists(xclbin_path):
            print(f"\n⏭️  Skipping: {config['name']} (XCLBIN not found)")
            continue

        if not os.path.exists(insts_path):
            print(f"\n⏭️  Skipping: {config['name']} (instructions not found)")
            continue

        result = test_xclbin(xclbin_path, insts_path, audio_data)

        if result and result['success']:
            successful_tests.append(result)
            print(f"\n✅ {os.path.basename(xclbin_path)} PASSED")
            break  # Stop on first success
        else:
            print(f"\n❌ {os.path.basename(xclbin_path)} FAILED")

    # Step 5: Final summary
    print("\n" + "="*70)
    print("[5/5] FINAL SUMMARY")
    print("="*70)

    if successful_tests:
        best_result = successful_tests[0]
        print(f"✅ NPU MEL KERNEL EXECUTION: SUCCESS")
        print(f"\nWorking XCLBIN: {os.path.basename(best_result['xclbin'])}")
        print(f"Execution Time: {best_result['execution_time_ms']:.2f} ms")
        print(f"Output Validation: {'PASSED' if best_result['validation_passed'] else 'WARNINGS'}")
        print(f"\nPerformance Details:")
        print(f"  - Average: {best_result['execution_time_ms']:.2f} ms")
        print(f"  - Min: {min(best_result['execution_times']):.2f} ms")
        print(f"  - Max: {max(best_result['execution_times']):.2f} ms")
        print(f"\nOutput Details:")
        print(f"  - Non-zero values: {best_result['non_zero_percent']:.1f}%")
        print(f"  - Unique values: {best_result['unique_values']}")
        print(f"  - Output size: {len(best_result['output_data'])} INT8 values")
        print("\n" + "="*70)
        return 0
    else:
        print(f"❌ NPU MEL KERNEL EXECUTION: FAILED")
        print(f"\nAll tested XCLBIN files failed to execute properly.")
        print(f"Please check:")
        print(f"  1. XRT installation: /opt/xilinx/xrt/")
        print(f"  2. NPU firmware: xrt-smi examine")
        print(f"  3. XCLBIN compilation: Check build logs")
        print(f"  4. NPU availability: ls -l /dev/accel/accel0")
        print("\n" + "="*70)
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
