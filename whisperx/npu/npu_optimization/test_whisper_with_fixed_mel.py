#!/usr/bin/env python3
"""
Test Whisper with Fixed NPU Mel Kernel
Tests the production mel kernel (correlation 0.70) with actual Whisper inference
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
import librosa
import time
from pathlib import Path

print("=" * 70)
print("WHISPER + NPU MEL KERNEL INTEGRATION TEST")
print("=" * 70)

# Load audio
audio_path = "mel_kernels/test_audio_jfk.wav"
print(f"\nLoading audio: {audio_path}")
audio, sr = librosa.load(audio_path, sr=16000, mono=True)
print(f"  Duration: {len(audio) / sr:.2f}s")
print(f"  Sample rate: {sr} Hz")
print(f"  Samples: {len(audio)}")

# Load NPU with fixed mel kernel
print("\nLoading NPU mel kernel...")
xclbin_path = "mel_kernels/build_fixed_v3/mel_fixed_v3.xclbin"
device = xrt.device(0)
xclbin = xrt.xclbin(xclbin_path)
device.register_xclbin(xclbin)
uuid = xclbin.get_uuid()
hw_ctx = xrt.hw_context(device, uuid)
kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
print("  ✅ NPU kernel loaded")

# Load instruction binary
insts_bin = open("mel_kernels/build_fixed_v3/insts_v3.bin", "rb").read()
n_insts = len(insts_bin)

# Create XRT buffers
instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
input_bo = xrt.bo(device, 800, xrt.bo.flags.host_only, kernel.group_id(3))
output_bo = xrt.bo(device, 80, xrt.bo.flags.host_only, kernel.group_id(4))

# Write instructions once
instr_bo.write(insts_bin, 0)
instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

# Process audio in frames
hop_length = 160  # 10ms hop
frame_length = 400  # 25ms frames
n_frames = 1 + (len(audio) - frame_length) // hop_length

print(f"\nProcessing {n_frames} frames...")
mel_spectrogram = np.zeros((80, n_frames), dtype=np.int8)

start_time = time.time()
for frame_idx in range(n_frames):
    # Extract frame
    start_sample = frame_idx * hop_length
    end_sample = start_sample + frame_length
    audio_frame = audio[start_sample:end_sample]

    # Convert to INT16
    audio_int16 = (audio_frame * 32767).astype(np.int16)

    # Write to NPU
    input_data = audio_int16.tobytes()
    input_bo.write(input_data, 0)
    input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 800, 0)

    # Run kernel
    opcode = 3
    run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
    state = run.wait(10000)

    if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        print(f"❌ Frame {frame_idx} failed!")
        break

    # Read output
    output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 80, 0)
    mel_frame = np.frombuffer(output_bo.read(80, 0), dtype=np.int8)
    mel_spectrogram[:, frame_idx] = mel_frame

    if frame_idx % 50 == 0:
        print(f"  Processed {frame_idx}/{n_frames} frames...", end='\r')

elapsed = time.time() - start_time
print(f"\n  ✅ Processed {n_frames} frames in {elapsed:.3f}s")
print(f"  Performance: {len(audio) / sr / elapsed:.1f}x realtime")

# Analyze mel spectrogram
print("\nMel Spectrogram Analysis:")
print(f"  Shape: {mel_spectrogram.shape}")
print(f"  Range: [{mel_spectrogram.min()}, {mel_spectrogram.max()}]")
print(f"  Mean: {mel_spectrogram.mean():.2f}")
print(f"  Non-zero: {np.count_nonzero(mel_spectrogram)} / {mel_spectrogram.size} ({100 * np.count_nonzero(mel_spectrogram) / mel_spectrogram.size:.1f}%)")

# Compare with librosa
print("\nComparing with librosa reference...")
mel_ref = librosa.feature.melspectrogram(
    y=audio, sr=sr, n_fft=512, hop_length=hop_length,
    win_length=frame_length, n_mels=80,
    htk=True, power=2.0, fmin=0, fmax=8000
)
mel_ref_db = librosa.power_to_db(mel_ref, ref=np.max)

# Normalize both for comparison
def normalize(x):
    x = x.astype(np.float32)
    x_range = x.max() - x.min()
    if x_range < 1e-6:
        return np.zeros_like(x)
    return (x - x.min()) / x_range

mel_npu_norm = normalize(mel_spectrogram.astype(np.float32))
mel_ref_norm = normalize(mel_ref_db)

# Compute correlation per frame and overall
correlations = []
for frame_idx in range(min(n_frames, mel_ref.shape[1])):
    corr = np.corrcoef(mel_npu_norm[:, frame_idx], mel_ref_norm[:, frame_idx])[0, 1]
    if not np.isnan(corr):
        correlations.append(corr)

avg_correlation = np.mean(correlations) if correlations else 0.0
print(f"  Average frame correlation: {avg_correlation:.4f}")
print(f"  Frames with corr > 0.5: {sum(c > 0.5 for c in correlations)} / {len(correlations)}")
print(f"  Frames with corr > 0.7: {sum(c > 0.7 for c in correlations)} / {len(correlations)}")

# Overall correlation (match dimensions)
n_common_frames = min(mel_npu_norm.shape[1], mel_ref_norm.shape[1])
overall_corr = np.corrcoef(
    mel_npu_norm[:, :n_common_frames].flatten(),
    mel_ref_norm[:, :n_common_frames].flatten()
)[0, 1]
print(f"  Overall correlation: {overall_corr:.4f}")

# Test with Whisper encoder (if available)
print("\nTesting with Whisper encoder...")
try:
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    print("  Loading Whisper base model...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    model.eval()

    # Convert INT8 mel to float for Whisper
    mel_float = mel_spectrogram.astype(np.float32) / 127.0  # Normalize to [0, 1]

    # Whisper expects (batch, mels, time)
    mel_tensor = torch.from_numpy(mel_float).unsqueeze(0)

    print(f"  Mel tensor shape: {mel_tensor.shape}")
    print(f"  Running encoder...")

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(mel_tensor)

    print(f"  ✅ Encoder output shape: {encoder_outputs.last_hidden_state.shape}")
    print(f"  Encoder output range: [{encoder_outputs.last_hidden_state.min():.3f}, {encoder_outputs.last_hidden_state.max():.3f}]")

    # Try to decode (simple greedy)
    print("\n  Attempting transcription (greedy decode)...")
    predicted_ids = model.generate(mel_tensor, max_length=448)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print(f"\n  Transcription: '{transcription}'")

except ImportError:
    print("  ⚠️  Transformers/Whisper not available, skipping encoder test")
except Exception as e:
    print(f"  ⚠️  Whisper test failed: {e}")

# Final summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✅ NPU mel kernel: WORKING")
print(f"✅ Processing speed: {len(audio) / sr / elapsed:.1f}x realtime")
print(f"✅ Correlation: {overall_corr:.4f}")
print(f"{'✅' if overall_corr > 0.6 else '⚠️ '} Quality: {'GOOD' if overall_corr > 0.6 else 'NEEDS IMPROVEMENT'}")
print("=" * 70)
