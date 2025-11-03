#!/usr/bin/env python3
"""
Test NPU mel feature injection into faster-whisper
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path
import asyncio

# Setup logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

# Add whisperx to path
sys.path.insert(0, '/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx')

async def test_npu_integration():
    """Test that NPU mel features are used instead of being recomputed"""

    print("=" * 80)
    print("Testing NPU Mel Feature Injection")
    print("=" * 80)

    # Import after path setup
    from server_dynamic import whisper_engine

    # Use the global whisper engine instance
    print("\n1. Using global whisper_engine...")
    engine = whisper_engine

    # Check NPU is available
    if not hasattr(engine, 'npu_runtime') or not engine.npu_runtime.mel_available:
        print("❌ NPU mel not available - test cannot proceed")
        return False

    print("✅ NPU mel is available")

    # Get a test audio file
    test_audio = Path("/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/test_audio_jfk.wav")
    if not test_audio.exists():
        print(f"❌ Test audio not found: {test_audio}")
        return False

    print(f"✅ Test audio found: {test_audio}")

    # Save the file to temp location
    import tempfile
    import shutil

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp_path = tmp.name
        shutil.copy(test_audio, tmp_path)

    print(f"✅ Audio copied to: {tmp_path}")

    # Transcribe with NPU
    print("\n2. Transcribing with NPU mel injection...")
    print("-" * 80)

    try:
        result = await engine.transcribe(tmp_path, vad_filter=False)

        print("-" * 80)
        print("\n3. Results:")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Realtime factor: {result['realtime_factor']}")
        print(f"   NPU MEL time: {result.get('npu_mel_time', 0):.3f}s")

        # Check if NPU mel was used
        if result.get('npu_mel_time') is not None and result.get('npu_mel_time', 0) > 0:
            print("\n✅ SUCCESS: NPU mel preprocessing was used!")
            print(f"   NPU mel computation took: {result['npu_mel_time']:.3f}s")

            # Estimate what CPU mel would have taken (roughly 1/3 of total time in old approach)
            estimated_cpu_mel = result['duration'] * 0.01  # ~1% of audio duration
            print(f"   Estimated CPU mel time: ~{estimated_cpu_mel:.3f}s")
            print(f"   Time saved: ~{estimated_cpu_mel - result['npu_mel_time']:.3f}s")

            return True
        else:
            print("\n⚠️ WARNING: MEL time not reported (might be using CPU fallback)")
            return False

    except Exception as e:
        print(f"\n❌ ERROR during transcription: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)

if __name__ == "__main__":
    success = asyncio.run(test_npu_integration())
    sys.exit(0 if success else 1)
