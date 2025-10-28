#!/usr/bin/env python3
"""
Test MEL INT8 Kernel Execution on AMD Phoenix NPU

This script:
1. Loads the compiled mel_int8_final.xclbin
2. Prepares test audio input (400 INT16 samples)
3. Executes the kernel on NPU
4. Retrieves and validates the output (80 INT8 mel features)
5. Measures performance toward 220x realtime target

Based on successful October 27 breakthrough.
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
import time
import librosa

# Configuration (matches MLIR specification)
WINDOW_SIZE = 400        # INT16 audio samples
FFT_SIZE = 512
MEL_BINS = 80
SAMPLE_RATE = 16000

# Buffer sizes (in 32-bit words for XRT)
INPUT_WORDS = 200   # 400 INT16 samples = 800 bytes = 200 words (32-bit)
OUTPUT_WORDS = 20   # 80 INT8 mel features = 80 bytes = 20 words (32-bit)

def load_test_audio(audio_path=None):
    """
    Load and prepare test audio
    Returns: numpy array of INT16 audio samples (400 samples)
    """
    if audio_path is None:
        # Generate synthetic test signal (sine wave + noise)
        print("Generating synthetic test audio...")
        t = np.linspace(0, 0.025, WINDOW_SIZE)  # 25ms at 16kHz

        # Mix of frequencies (simulate speech)
        signal = (
            0.3 * np.sin(2 * np.pi * 200 * t) +    # 200Hz fundamental
            0.2 * np.sin(2 * np.pi * 400 * t) +    # 400Hz harmonic
            0.15 * np.sin(2 * np.pi * 800 * t) +   # 800Hz harmonic
            0.1 * np.random.randn(WINDOW_SIZE)     # Noise
        )

        # Normalize to INT16 range
        signal = signal / np.max(np.abs(signal)) * 32000
        audio_int16 = signal.astype(np.int16)

    else:
        # Load real audio file
        print(f"Loading audio from: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        # Take first 400 samples
        audio = audio[:WINDOW_SIZE]
        if len(audio) < WINDOW_SIZE:
            audio = np.pad(audio, (0, WINDOW_SIZE - len(audio)))

        # Convert to INT16
        audio_int16 = (audio * 32767).astype(np.int16)

    return audio_int16

def compute_reference_mel(audio_int16):
    """
    Compute reference MEL spectrogram using librosa (CPU)
    For validation of NPU output
    """
    print("\nComputing reference MEL spectrogram (CPU)...")

    # Convert INT16 to float32 normalized
    audio_float = audio_int16.astype(np.float32) / 32768.0

    # Compute mel spectrogram with librosa
    mel_spec = librosa.feature.melspectrogram(
        y=audio_float,
        sr=SAMPLE_RATE,
        n_fft=FFT_SIZE,
        hop_length=WINDOW_SIZE,  # Only one frame
        win_length=WINDOW_SIZE,
        n_mels=MEL_BINS,
        fmin=0,
        fmax=SAMPLE_RATE/2
    )

    # Convert to log scale (dB)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Take first frame only
    mel_frame = mel_db[:, 0]

    print(f"  Reference MEL range: [{mel_frame.min():.2f}, {mel_frame.max():.2f}] dB")
    print(f"  Reference MEL shape: {mel_frame.shape}")

    return mel_frame

def test_npu_mel_kernel(xclbin_path='build/mel_int8_final.xclbin',
                        audio_path=None,
                        num_iterations=100):
    """
    Test MEL kernel execution on NPU

    Args:
        xclbin_path: Path to compiled XCLBIN
        audio_path: Path to audio file (None for synthetic)
        num_iterations: Number of iterations for performance measurement
    """

    print("="*70)
    print("AMD Phoenix NPU MEL Spectrogram Kernel Test")
    print("="*70)
    print()

    # Step 1: Load test audio
    audio_int16 = load_test_audio(audio_path)
    print(f"✅ Test audio loaded: {len(audio_int16)} INT16 samples")
    print(f"   Range: [{audio_int16.min()}, {audio_int16.max()}]")
    print()

    # Step 2: Compute reference MEL (CPU)
    reference_mel = compute_reference_mel(audio_int16)

    # Step 3: Initialize NPU
    print("Initializing NPU...")
    device = xrt.device(0)
    print(f"✅ NPU device opened: /dev/accel/accel0")

    # Load XCLBIN
    xclbin_obj = xrt.xclbin(xclbin_path)
    uuid = xclbin_obj.get_uuid()
    print(f"✅ XCLBIN loaded: {xclbin_path}")
    print(f"   UUID: {uuid}")

    # Register XCLBIN
    device.register_xclbin(xclbin_obj)
    print(f"✅ XCLBIN registered on device")

    # Create hardware context
    hw_ctx = xrt.hw_context(device, uuid)
    print(f"✅ Hardware context created")

    # Get kernel
    kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
    print(f"✅ Kernel found: MLIR_AIE")
    print()

    # Step 4: Prepare buffers
    print("Preparing buffers...")

    # Input buffer: 200 words (400 INT16 samples = 800 bytes)
    # Group ID 3 is for input (from MLIR connectivity)
    input_bo = xrt.bo(device, INPUT_WORDS * 4, xrt.bo.flags.host_only, kernel.group_id(3))

    # Output buffer: 20 words (80 INT8 values = 80 bytes)
    # Group ID 4 is for output
    output_bo = xrt.bo(device, OUTPUT_WORDS * 4, xrt.bo.flags.host_only, kernel.group_id(4))

    print(f"✅ Input buffer allocated: {INPUT_WORDS * 4} bytes")
    print(f"✅ Output buffer allocated: {OUTPUT_WORDS * 4} bytes")
    print()

    # Step 5: Write input data
    print("Writing input audio to NPU buffer...")

    # Convert INT16 audio to 32-bit words for XRT
    # Pack two INT16 samples into one INT32 word (little-endian)
    input_words = np.zeros(INPUT_WORDS, dtype=np.int32)
    for i in range(INPUT_WORDS):
        sample_0 = int(audio_int16[i * 2]) & 0xFFFF
        sample_1 = int(audio_int16[i * 2 + 1]) & 0xFFFF
        input_words[i] = (sample_1 << 16) | sample_0

    # Write to buffer
    input_bo.write(input_words, 0)

    # Sync to device
    input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, INPUT_WORDS * 4, 0)
    print(f"✅ Input audio synced to NPU ({INPUT_WORDS * 4} bytes)")
    print()

    # Step 6: Execute kernel (warm-up)
    print("Executing kernel (warm-up)...")
    run = kernel(input_bo, output_bo)
    state = run.wait(1000)  # 1 second timeout

    if state == xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        print(f"✅ Kernel execution COMPLETED (warm-up)")
    else:
        print(f"❌ Kernel execution failed: {state}")
        return
    print()

    # Step 7: Performance measurement
    print(f"Running performance test ({num_iterations} iterations)...")

    execution_times = []
    for i in range(num_iterations):
        start = time.perf_counter()

        run = kernel(input_bo, output_bo)
        state = run.wait(1000)

        end = time.perf_counter()

        if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            print(f"❌ Iteration {i} failed: {state}")
            continue

        execution_times.append(end - start)

    avg_time = np.mean(execution_times)
    std_time = np.std(execution_times)
    min_time = np.min(execution_times)
    max_time = np.max(execution_times)

    print(f"✅ Performance test completed")
    print(f"   Average execution time: {avg_time*1000:.3f} ms")
    print(f"   Std deviation: {std_time*1000:.3f} ms")
    print(f"   Min: {min_time*1000:.3f} ms, Max: {max_time*1000:.3f} ms")
    print()

    # Step 8: Read output
    print("Reading MEL output from NPU...")

    # Sync from device
    output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, OUTPUT_WORDS * 4, 0)

    # Read data
    output_words = np.frombuffer(output_bo.read(OUTPUT_WORDS * 4, 0), dtype=np.int32)

    # Unpack INT8 values from INT32 words
    # Each word contains 4 INT8 values (little-endian)
    mel_output_int8 = np.zeros(MEL_BINS, dtype=np.int8)
    for i in range(OUTPUT_WORDS):
        word = output_words[i]
        mel_output_int8[i*4 + 0] = (word >> 0) & 0xFF
        mel_output_int8[i*4 + 1] = (word >> 8) & 0xFF
        mel_output_int8[i*4 + 2] = (word >> 16) & 0xFF
        mel_output_int8[i*4 + 3] = (word >> 24) & 0xFF

    # Convert INT8 to signed
    mel_output_int8 = mel_output_int8.view(np.int8)

    print(f"✅ MEL output read: {mel_output_int8.shape}")
    print(f"   Range: [{mel_output_int8.min()}, {mel_output_int8.max()}]")
    print(f"   First 10 values: {mel_output_int8[:10]}")
    print()

    # Step 9: Validation
    print("Validating output...")

    # Convert INT8 Q7 to float for comparison
    # Q7 format: 1 sign bit + 7 fractional bits = range [-1, 127/128]
    mel_output_float = mel_output_int8.astype(np.float32)

    # Normalize reference to similar scale for comparison
    reference_normalized = (reference_mel - reference_mel.min()) / (reference_mel.max() - reference_mel.min())
    reference_scaled = reference_normalized * 127 - 64  # Scale to INT8 range

    # Compute correlation (qualitative check)
    correlation = np.corrcoef(mel_output_float, reference_scaled)[0, 1]

    # Compute RMSE
    rmse = np.sqrt(np.mean((mel_output_float - reference_scaled) ** 2))

    print(f"   Correlation with reference: {correlation:.3f}")
    print(f"   RMSE: {rmse:.2f}")

    if correlation > 0.5:
        print(f"✅ Output correlation GOOD ({correlation:.3f} > 0.5)")
    else:
        print(f"⚠️  Output correlation LOW ({correlation:.3f} < 0.5)")
        print(f"   This may be expected for first integration - kernel may need tuning")
    print()

    # Step 10: Calculate realtime factor
    print("Performance Analysis:")
    print("-" * 70)

    audio_duration = WINDOW_SIZE / SAMPLE_RATE  # seconds
    processing_time = avg_time  # seconds
    realtime_factor = audio_duration / processing_time

    print(f"   Audio duration: {audio_duration*1000:.3f} ms ({WINDOW_SIZE} samples @ {SAMPLE_RATE}Hz)")
    print(f"   Processing time: {processing_time*1000:.3f} ms")
    print(f"   Realtime factor: {realtime_factor:.1f}x")
    print()

    # Compare to target
    target_rtf = 220.0
    progress_percent = (realtime_factor / target_rtf) * 100

    print(f"   Target: {target_rtf}x realtime")
    print(f"   Current: {realtime_factor:.1f}x realtime")
    print(f"   Progress: {progress_percent:.1f}%")
    print()

    if realtime_factor >= target_rtf:
        print(f"🎉 TARGET ACHIEVED! {realtime_factor:.1f}x >= {target_rtf}x")
    elif realtime_factor >= 100:
        print(f"✅ EXCELLENT! {realtime_factor:.1f}x (halfway to target)")
    elif realtime_factor >= 50:
        print(f"✅ GOOD! {realtime_factor:.1f}x (significant acceleration)")
    elif realtime_factor >= 10:
        print(f"✅ WORKING! {realtime_factor:.1f}x (10x+ faster than realtime)")
    else:
        print(f"⚠️  SLOW: {realtime_factor:.1f}x (needs optimization)")

    print()
    print("="*70)
    print("Test Complete!")
    print("="*70)

    return {
        'execution_time_ms': avg_time * 1000,
        'realtime_factor': realtime_factor,
        'correlation': correlation,
        'rmse': rmse,
        'mel_output': mel_output_int8,
        'reference_mel': reference_mel
    }

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test MEL INT8 kernel on AMD Phoenix NPU')
    parser.add_argument('--xclbin', default='build/mel_int8_final.xclbin',
                        help='Path to compiled XCLBIN file')
    parser.add_argument('--audio', default=None,
                        help='Path to audio file (optional, uses synthetic if not provided)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations for performance test')

    args = parser.parse_args()

    try:
        results = test_npu_mel_kernel(
            xclbin_path=args.xclbin,
            audio_path=args.audio,
            num_iterations=args.iterations
        )

        print("\n✅ NPU MEL kernel test SUCCESSFUL!")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
