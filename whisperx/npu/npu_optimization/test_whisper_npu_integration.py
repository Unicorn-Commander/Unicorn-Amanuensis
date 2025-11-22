#!/usr/bin/env python3
"""
Test script for Whisper + NPU integration
Validates that NPU attention is being invoked correctly
"""

import sys
import torch
import numpy as np
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_npu_attention_integration():
    """Test NPU attention integration layer"""
    print("=" * 70)
    print("TEST 1: NPU Attention Integration")
    print("=" * 70)
    print()

    try:
        from npu_attention_integration import create_npu_attention, NPU_AVAILABLE

        print(f"NPU Available: {NPU_AVAILABLE}")
        print()

        # Create attention module
        attn = create_npu_attention(n_state=512, n_head=8, use_npu=True)
        print(f"Created: {attn}")
        print()

        # Test with small input
        batch_size = 1
        seq_len = 64  # Small test
        n_state = 512

        x = torch.randn(batch_size, seq_len, n_state)

        print(f"Input shape: {x.shape}")

        # Forward pass
        import time
        start = time.perf_counter()
        output, _ = attn(x)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"Output shape: {output.shape}")
        print(f"Time: {elapsed:.2f}ms")
        print()

        # Check statistics
        stats = attn.get_stats()
        print("Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()

        # Verify NPU was actually used
        if NPU_AVAILABLE and stats.get('npu_calls', 0) > 0:
            print("✅ NPU attention is being invoked!")
            return True
        elif not NPU_AVAILABLE:
            print("⚠️  NPU not available, using CPU fallback (expected)")
            return True
        else:
            print("❌ NPU available but not being used!")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_whisper_encoder_replacement():
    """Test replacing Whisper encoder with NPU version"""
    print("=" * 70)
    print("TEST 2: Whisper Encoder Replacement")
    print("=" * 70)
    print()

    try:
        import whisper
        from whisper_npu_openai import replace_encoder_with_npu, NPU_AVAILABLE

        print("Loading Whisper base model...")
        model = whisper.load_model("base")
        print(f"Original encoder: {type(model.encoder).__name__}")
        print()

        print("Replacing encoder with NPU version...")
        model = replace_encoder_with_npu(model, use_npu=True)
        print(f"New encoder: {type(model.encoder).__name__}")
        print()

        # Test encoder forward pass
        n_mels = model.dims.n_mels
        n_ctx = 1500  # 30 seconds
        mel_input = torch.randn(1, n_mels, n_ctx)

        print(f"Testing encoder with input shape: {mel_input.shape}")

        import time
        start = time.perf_counter()
        with torch.no_grad():
            output = model.encoder(mel_input)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"Output shape: {output.shape}")
        print(f"Encoding time: {elapsed:.2f}ms")
        print()

        # Check encoder statistics
        if hasattr(model.encoder, 'get_stats'):
            stats = model.encoder.get_stats()
            print("Encoder Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
            print()

            # Verify NPU was used
            if NPU_AVAILABLE and stats.get('npu_calls', 0) > 0:
                print("✅ NPU encoder is working!")
                return True
            elif not NPU_AVAILABLE:
                print("⚠️  NPU not available, encoder using CPU (expected)")
                return True
            else:
                print("❌ NPU available but encoder not using it!")
                return False
        else:
            print("⚠️  Encoder doesn't have stats (using original Whisper encoder)")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_transcription():
    """Test full transcription with NPU encoder"""
    print("=" * 70)
    print("TEST 3: Full Transcription Pipeline")
    print("=" * 70)
    print()

    try:
        from whisper_npu_openai import load_whisper_with_npu
        import tempfile
        import soundfile as sf

        print("Loading Whisper model with NPU...")
        model = load_whisper_with_npu("base", use_npu=True)
        print()

        # Create synthetic test audio
        print("Creating synthetic test audio...")
        sample_rate = 16000
        duration = 5.0  # 5 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sample_rate)
            audio_path = tmp_file.name

        print(f"Test audio: {duration}s sine wave at 440 Hz")
        print(f"Saved to: {audio_path}")
        print()

        # Transcribe
        print("Running transcription...")
        import time
        start = time.perf_counter()
        result = model.transcribe(audio_path, fp16=False)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"Transcription time: {elapsed:.2f}ms")
        print(f"Text: '{result['text']}'")
        print(f"Language: {result['language']}")
        print()

        # Calculate realtime factor
        rtf = elapsed / (duration * 1000)
        speedup = 1.0 / rtf

        print(f"Performance:")
        print(f"  Audio duration: {duration}s")
        print(f"  Processing time: {elapsed:.2f}ms")
        print(f"  Realtime factor: {rtf:.4f}")
        print(f"  Speedup: {speedup:.1f}x realtime")
        print()

        # Check encoder stats
        if hasattr(model.encoder, 'get_stats'):
            stats = model.encoder.get_stats()
            print("Encoder Statistics:")
            print(f"  NPU calls: {stats.get('npu_calls', 0)}")
            print(f"  CPU fallback calls: {stats.get('cpu_fallback_calls', 0)}")
            print(f"  Total NPU time: {stats.get('total_npu_time_ms', 0):.2f}ms")
            print()

            if stats.get('npu_calls', 0) > 0:
                print("✅ Transcription using NPU encoder!")
                return True
            else:
                print("⚠️  Transcription completed but NPU not used")
                return False
        else:
            print("⚠️  Can't verify NPU usage (no stats)")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_npu_xclbin_exists():
    """Test that NPU XCLBIN file exists"""
    print("=" * 70)
    print("TEST 0: NPU Kernel Availability")
    print("=" * 70)
    print()

    xclbin_paths = [
        Path(__file__).parent / 'whisper_encoder_kernels' / 'build_attention_int32' / 'attention_64x64.xclbin',
        Path(__file__).parent / 'whisper_encoder_kernels' / 'build_attention_64x64' / 'attention_64x64.xclbin',
        Path(__file__).parent / 'whisper_encoder_kernels' / 'attention_64x64.xclbin',
    ]

    print("Checking for NPU attention kernel...")
    for xclbin_path in xclbin_paths:
        print(f"  {xclbin_path}: ", end='')
        if xclbin_path.exists():
            size_mb = xclbin_path.stat().st_size / (1024 * 1024)
            print(f"✅ Found ({size_mb:.2f} MB)")
            return True
        else:
            print("❌ Not found")

    print()
    print("⚠️  No NPU attention kernel found!")
    print("   This is expected - NPU kernels need to be compiled separately.")
    print("   The integration will fall back to CPU attention.")
    print()
    return False


def main():
    """Run all tests"""
    print()
    print("=" * 70)
    print("WHISPER NPU INTEGRATION TEST SUITE")
    print("=" * 70)
    print()

    results = {}

    # Test 0: Check XCLBIN exists
    print()
    results['xclbin_exists'] = test_npu_xclbin_exists()

    # Test 1: NPU attention integration
    print()
    results['attention_integration'] = test_npu_attention_integration()

    # Test 2: Whisper encoder replacement
    print()
    results['encoder_replacement'] = test_whisper_encoder_replacement()

    # Test 3: Full transcription
    print()
    results['full_transcription'] = test_full_transcription()

    # Summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")

    print()
    print(f"Results: {passed}/{total} tests passed")
    print()

    if results.get('xclbin_exists'):
        if passed == total:
            print("🎉 ALL TESTS PASSED - NPU integration is working!")
        else:
            print("⚠️  Some tests failed - check errors above")
    else:
        print("ℹ️  NPU kernel not found - integration uses CPU fallback")
        print("   To enable NPU acceleration:")
        print("   1. Compile attention kernel (see NPU_WHISPER_QUICK_START.md)")
        print("   2. Place XCLBIN at: whisper_encoder_kernels/build_attention_int32/attention_64x64.xclbin")
        print("   3. Re-run tests")

    print()
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
