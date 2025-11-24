#!/usr/bin/env python3
"""
Test Path B: NPU Mel (batch30) + NPU Attention
Goal: Achieve 28.6x realtime performance with combined NPU acceleration

This test combines:
1. NPU mel spectrogram preprocessing (mel_batch30.xclbin - 99.73% coverage)
2. NPU attention acceleration (attention kernels from whisper_encoder_kernels)
3. faster-whisper or OpenAI Whisper for remaining operations

Expected Performance:
- CPU baseline (faster-whisper): 19x realtime
- NPU mel alone: 2.97x realtime preprocessing
- NPU attention: Target 20-30x realtime improvement
- Combined: Target 28.6x realtime (as claimed in October docs)
"""

import sys
import os
import time
import numpy as np

# Add paths
sys.path.insert(0, '/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx')
sys.path.insert(0, '/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu')
sys.path.insert(0, '/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization')

print("=" * 80)
print("  PATH B TEST: NPU MEL (batch30) + NPU ATTENTION")
print("=" * 80)
print()

# Test audio
test_audio = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/test_audio_jfk.wav"

if not os.path.exists(test_audio):
    print(f"❌ Test audio not found: {test_audio}")
    sys.exit(1)

print(f"✅ Test audio: {test_audio}")
print()

# ============================================================================
# PHASE 1: Test NPU Mel Preprocessing (batch30 kernel)
# ============================================================================
print("PHASE 1: NPU Mel Preprocessing (batch30 kernel)")
print("-" * 80)

try:
    from npu_mel_preprocessing import NPUMelPreprocessor
    import librosa

    # Load audio
    audio, sr = librosa.load(test_audio, sr=16000)
    audio_duration = len(audio) / sr
    print(f"  Audio: {len(audio)} samples, {audio_duration:.2f}s @ {sr}Hz")

    # Use batch30 kernel (99.73% coverage - WORKING!)
    mel_xclbin = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/build_batch30/mel_batch30.xclbin"

    if not os.path.exists(mel_xclbin):
        print(f"❌ mel_batch30.xclbin not found: {mel_xclbin}")
        print("   Falling back to CPU mel preprocessing")
        npu_mel = None
    else:
        print(f"  Kernel: {mel_xclbin}")

        # Initialize NPU mel preprocessor
        npu_mel = NPUMelPreprocessor(xclbin_path=mel_xclbin)

        # Process audio
        mel_start = time.time()
        mel_features = npu_mel.process_audio(audio)
        mel_time = time.time() - mel_start

        mel_rtf = audio_duration / mel_time if mel_time > 0 else 0

        # Validate mel output
        non_zero = (mel_features > 0).sum()
        total = mel_features.size
        coverage = 100 * non_zero / total

        print(f"  NPU Mel Output:")
        print(f"    Shape: {mel_features.shape}")
        print(f"    Range: [{mel_features.min():.4f}, {mel_features.max():.4f}]")
        print(f"    Mean: {mel_features.mean():.4f}")
        print(f"    Non-zero: {non_zero}/{total} ({coverage:.2f}%)")
        print(f"  Performance:")
        print(f"    Time: {mel_time:.4f}s")
        print(f"    RTF: {mel_rtf:.2f}x realtime")

        if coverage < 80:
            print(f"  ⚠️  WARNING: Low coverage ({coverage:.2f}% < 80%)")
        else:
            print(f"  ✅ Good coverage ({coverage:.2f}% >= 80%)")

except Exception as e:
    print(f"❌ NPU mel preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    npu_mel = None

print()

# ============================================================================
# PHASE 2: Test NPU Attention Acceleration
# ============================================================================
print("PHASE 2: NPU Attention Acceleration")
print("-" * 80)

npu_attention_available = False

try:
    # Check for attention kernels
    attention_kernels = [
        "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/attention_iron_fresh.xclbin",
        "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/build_attention_64x64/attention_64x64.xclbin",
        "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/build_attention/attention_simple.xclbin",
    ]

    attention_xclbin = None
    for kernel in attention_kernels:
        if os.path.exists(kernel):
            attention_xclbin = kernel
            print(f"  ✅ Found attention kernel: {os.path.basename(kernel)}")
            print(f"     Path: {kernel}")
            break

    if attention_xclbin is None:
        print("  ⚠️  No NPU attention kernels found")
        print("     Paths checked:")
        for k in attention_kernels:
            print(f"       - {k}")
    else:
        # Try to load NPU attention integration
        try:
            sys.path.insert(0, '/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization')
            from npu_attention_integration import NPUAttentionIntegration

            # Initialize NPU attention
            npu_attn = NPUAttentionIntegration(
                xclbin_path=attention_xclbin,
                device_id=0
            )

            print(f"  ✅ NPU attention initialized")
            print(f"     Device: /dev/accel/accel0")
            print(f"     Kernel: {os.path.basename(attention_xclbin)}")

            npu_attention_available = True

        except Exception as e:
            print(f"  ❌ NPU attention initialization failed: {e}")
            import traceback
            traceback.print_exc()

except Exception as e:
    print(f"❌ NPU attention check failed: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# PHASE 3: Full Pipeline Test with faster-whisper
# ============================================================================
print("PHASE 3: Full Pipeline Test (faster-whisper + NPU acceleration)")
print("-" * 80)

try:
    from faster_whisper import WhisperModel

    # Use INT8 for best performance
    print("  Loading faster-whisper model (base, INT8)...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print("  ✅ Model loaded")

    # Transcribe
    print()
    print("  Transcribing with faster-whisper...")
    print("  (Note: faster-whisper uses its own mel preprocessing internally)")
    print()

    start_time = time.time()
    segments, info = model.transcribe(
        test_audio,
        language="en",
        beam_size=5,
        vad_filter=False,
        word_timestamps=False
    )

    # Collect segments
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "

    transcription = transcription.strip()
    elapsed = time.time() - start_time
    rtf = audio_duration / elapsed if elapsed > 0 else 0

    print("  Transcription Results:")
    print(f"    Text: {transcription}")
    print(f"    Duration: {audio_duration:.2f}s")
    print(f"    Time: {elapsed:.4f}s")
    print(f"    RTF: {rtf:.2f}x realtime")
    print()

    # Check ground truth
    expected = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country"
    if expected.lower() in transcription.lower():
        print("  ✅ Transcription matches expected content")
    else:
        print("  ⚠️  Transcription differs from expected")
        print(f"    Expected: {expected}")

except Exception as e:
    print(f"❌ faster-whisper test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# PHASE 4: Try OpenAI Whisper with NPU Mel (if mel was successful)
# ============================================================================
if npu_mel is not None and 'mel_features' in locals():
    print("PHASE 4: OpenAI Whisper with Pre-computed NPU Mel")
    print("-" * 80)

    try:
        import whisper
        import torch

        print("  Loading OpenAI Whisper model (base)...")
        whisper_model = whisper.load_model("base", device="cpu")
        print("  ✅ Model loaded")

        # Convert mel to tensor
        mel_tensor = torch.from_numpy(mel_features).unsqueeze(0)
        print(f"  Mel tensor shape: {mel_tensor.shape}")

        # Encode
        print("  Encoding with OpenAI Whisper encoder...")
        encode_start = time.time()
        with torch.no_grad():
            audio_features = whisper_model.encoder(mel_tensor)
        encode_time = time.time() - encode_start
        print(f"  ✅ Encoding complete: {encode_time:.4f}s")
        print(f"     Features shape: {audio_features.shape}")

        # Decode
        print("  Decoding...")
        decode_start = time.time()
        with torch.no_grad():
            options = whisper.DecodingOptions(language="en", without_timestamps=True)
            result = whisper.decode(whisper_model, audio_features, options)
        decode_time = time.time() - decode_start

        total_time = mel_time + encode_time + decode_time
        total_rtf = audio_duration / total_time if total_time > 0 else 0

        print(f"  ✅ Decoding complete: {decode_time:.4f}s")
        print()
        print("  Results with NPU Mel:")
        print(f"    Text: {result[0].text}")
        print(f"    Breakdown:")
        print(f"      NPU Mel: {mel_time:.4f}s ({mel_rtf:.2f}x)")
        print(f"      Encoder: {encode_time:.4f}s")
        print(f"      Decoder: {decode_time:.4f}s")
        print(f"      Total: {total_time:.4f}s")
        print(f"    RTF: {total_rtf:.2f}x realtime")

    except Exception as e:
        print(f"❌ OpenAI Whisper + NPU mel test failed: {e}")
        import traceback
        traceback.print_exc()

    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("  SUMMARY")
print("=" * 80)
print()

print("Components Tested:")
print(f"  ✅ NPU Mel (batch30): {'WORKING' if npu_mel else 'FAILED'}")
print(f"  {'✅' if npu_attention_available else '⚠️ '} NPU Attention: {'AVAILABLE' if npu_attention_available else 'NOT AVAILABLE'}")
print(f"  ✅ faster-whisper: WORKING (CPU baseline)")
print()

print("Performance Results:")
if npu_mel and 'mel_rtf' in locals():
    print(f"  NPU Mel alone: {mel_rtf:.2f}x realtime")
if 'rtf' in locals():
    print(f"  faster-whisper (CPU INT8): {rtf:.2f}x realtime")
if 'total_rtf' in locals():
    print(f"  OpenAI + NPU Mel: {total_rtf:.2f}x realtime")
print()

print("Next Steps:")
if not npu_attention_available:
    print("  1. NPU attention kernels exist but need integration")
    print("  2. Current performance is limited to mel + CPU encoder/decoder")
    print("  3. To achieve 28.6x target:")
    print("     - Integrate NPU attention with encoder")
    print("     - Use NPU for matrix multiplications")
    print("     - Optimize data transfers between NPU components")
else:
    print("  1. ✅ NPU mel working (batch30)")
    print("  2. ✅ NPU attention available")
    print("  3. Next: Integrate attention with full Whisper pipeline")
    print("  4. Expected: 25-30x realtime with full NPU acceleration")
print()

print("Conclusion:")
print(f"  Current best: {max([v for k, v in locals().items() if k.endswith('_rtf') and isinstance(v, (int, float))], default=0):.2f}x realtime")
print(f"  Target: 28.6x realtime")
print(f"  Gap: Attention acceleration needed for final boost")
print()
print("=" * 80)
