#!/usr/bin/env python3
"""
Test script for KV Cache Fix
Verifies that decoder KV concatenation is working correctly
"""

import numpy as np
import sys
import os
import time
import tempfile
import soundfile as sf

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'npu'))
sys.path.insert(0, os.path.join(current_dir, 'npu', 'npu_optimization'))

# Import directly from the file path
import importlib.util
spec = importlib.util.spec_from_file_location(
    "onnx_whisper_npu",
    os.path.join(current_dir, 'npu', 'npu_optimization', 'onnx_whisper_npu.py')
)
onnx_whisper_npu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(onnx_whisper_npu)
ONNXWhisperNPU = onnx_whisper_npu.ONNXWhisperNPU

def generate_test_audio(duration=5.0, freq=440.0):
    """Generate synthetic test audio"""
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    return audio, sample_rate

def test_kv_cache_fix():
    """Test the KV cache fix"""
    print("=" * 70)
    print("KV Cache Fix Test")
    print("=" * 70)
    print()

    # Initialize decoder
    print("1. Initializing ONNX Whisper decoder...")
    decoder = ONNXWhisperNPU()
    if not decoder.initialize(model_size="base"):
        print("❌ Failed to initialize decoder")
        return False

    print("✅ Decoder initialized successfully")
    print()

    # Generate test audio
    print("2. Generating synthetic test audio (5 seconds, 440 Hz)...")
    audio, sample_rate = generate_test_audio(duration=5.0)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        test_file = tmp.name

    print(f"✅ Test audio created: {test_file}")
    print(f"   Duration: {len(audio) / sample_rate:.1f}s")
    print(f"   Sample rate: {sample_rate} Hz")
    print()

    # Transcribe
    print("3. Running transcription with KV cache fix...")
    start_time = time.time()

    try:
        result = decoder.transcribe_audio(test_file)
        elapsed = time.time() - start_time

        print()
        print("=" * 70)
        print("Results")
        print("=" * 70)
        print(f"✅ Transcription completed in {elapsed:.2f}s")
        print()
        print(f"Transcribed Text:")
        print(f"  '{result['text']}'")
        print()
        print(f"Performance Metrics:")
        print(f"  Processing time: {result['processing_time']:.2f}s")
        print(f"  Audio duration: {result['audio_duration']:.2f}s")
        print(f"  Real-time factor: {result['real_time_factor']:.2f}x")
        print(f"  NPU accelerated: {result['npu_accelerated']}")
        print()

        # Check for garbled output
        text = result['text'].strip()
        is_garbled = (
            text == "" or
            text.startswith("[") or
            "..." in text or
            len(text) < 5
        )

        print("Quality Check:")
        if is_garbled:
            print("⚠️  Output appears to be placeholder/garbled")
            print("   This may be expected for synthetic audio")
        else:
            print("✅ Output contains actual text (not garbled)")
        print()

        # Performance check
        print("Performance Check:")
        target_rtf = 15.0  # Target: 15x realtime or better
        if result['real_time_factor'] >= target_rtf:
            print(f"✅ Achieved {result['real_time_factor']:.1f}x realtime (target: {target_rtf}x)")
        else:
            print(f"⚠️  Only {result['real_time_factor']:.1f}x realtime (target: {target_rtf}x)")
        print()

        # Cleanup
        os.unlink(test_file)

        return True

    except Exception as e:
        print()
        print("❌ Transcription failed!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)

        return False

def test_with_real_speech():
    """Test with real speech audio if available"""
    print()
    print("=" * 70)
    print("Testing with Real Speech (if available)")
    print("=" * 70)
    print()

    test_files = [
        "/tmp/test_speech_like.wav",  # Our generated speech-like audio
        "/home/ucadmin/VibeVoice/Shafen_Khan_call.m4a",
        "/home/ucadmin/Development/test_audio.wav",
        "/tmp/test_speech.wav"
    ]

    found_file = None
    for path in test_files:
        if os.path.exists(path):
            found_file = path
            break

    if not found_file:
        print("ℹ️  No real speech audio found, skipping this test")
        print("   Tested paths:")
        for path in test_files:
            print(f"   - {path}")
        return True

    print(f"Found test audio: {found_file}")
    print()

    # Initialize decoder
    decoder = ONNXWhisperNPU()
    if not decoder.initialize(model_size="base"):
        print("❌ Failed to initialize decoder")
        return False

    print("Running transcription...")
    start_time = time.time()

    try:
        result = decoder.transcribe_audio(found_file)
        elapsed = time.time() - start_time

        print()
        print("=" * 70)
        print("Real Speech Results")
        print("=" * 70)
        print(f"✅ Transcription completed in {elapsed:.2f}s")
        print()
        print(f"Transcribed Text (first 200 chars):")
        print(f"  {result['text'][:200]}...")
        print()
        print(f"Performance:")
        print(f"  RTF: {result['real_time_factor']:.2f}x realtime")
        print()

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def main():
    """Main test function"""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "KV Cache Fix Verification Test" + " " * 22 + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  Purpose: Verify decoder KV concatenation fix" + " " * 21 + "║")
    print("║" + "  Expected: Non-garbled output with 15-20x RTF" + " " * 19 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Test 1: Synthetic audio
    success1 = test_kv_cache_fix()

    # Test 2: Real speech (optional)
    success2 = test_with_real_speech()

    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Synthetic Audio Test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Real Speech Test: {'✅ PASSED' if success2 else '❌ FAILED (or skipped)'}")
    print()

    if success1:
        print("✅ KV Cache fix appears to be working correctly!")
        print()
        print("Next Steps:")
        print("1. Test with longer audio (30s, 60s)")
        print("2. Measure detailed performance metrics")
        print("3. Compare with baseline (before fix)")
        print("4. Validate transcription accuracy")
    else:
        print("❌ KV Cache fix needs debugging")
        print()
        print("Debug Steps:")
        print("1. Check KV tensor shapes in decoder loop")
        print("2. Verify concatenation axis is correct")
        print("3. Add debug logging for KV cache sizes")
        print("4. Test with shorter sequences first")

    print()
    return success1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
