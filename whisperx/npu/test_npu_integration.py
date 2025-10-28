#!/usr/bin/env python3
"""
Quick validation test for NPU integration

This script verifies that all components are working correctly.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("NPU INTEGRATION VALIDATION TEST")
print("=" * 70)

# Test 1: Import modules
print("\nTest 1: Importing modules...")
try:
    from npu_mel_preprocessing import NPUMelPreprocessor
    print("  ✅ npu_mel_preprocessing imported")
except Exception as e:
    print(f"  ❌ Failed to import npu_mel_preprocessing: {e}")
    sys.exit(1)

try:
    from whisperx_npu_wrapper import WhisperXNPU
    print("  ✅ whisperx_npu_wrapper imported")
except Exception as e:
    print(f"  ❌ Failed to import whisperx_npu_wrapper: {e}")
    sys.exit(1)

# Test 2: Check XRT availability
print("\nTest 2: Checking XRT availability...")
try:
    sys.path.insert(0, '/opt/xilinx/xrt/python')
    import pyxrt as xrt
    print("  ✅ PyXRT available")
except Exception as e:
    print(f"  ⚠️  PyXRT not available: {e}")
    print("  (Will fall back to CPU)")

# Test 3: Check NPU device
print("\nTest 3: Checking NPU device...")
if os.path.exists("/dev/accel/accel0"):
    print("  ✅ NPU device available: /dev/accel/accel0")
else:
    print("  ⚠️  NPU device not found: /dev/accel/accel0")
    print("  (Will fall back to CPU)")

# Test 4: Check XCLBIN
print("\nTest 4: Checking XCLBIN...")
xclbin_path = Path(__file__).parent / "npu_optimization" / "mel_kernels" / "build_fixed" / "mel_fixed.xclbin"
if xclbin_path.exists():
    size = xclbin_path.stat().st_size
    print(f"  ✅ XCLBIN found: {xclbin_path}")
    print(f"     Size: {size} bytes")
else:
    print(f"  ⚠️  XCLBIN not found: {xclbin_path}")
    print("  (Will fall back to CPU)")

# Test 5: Initialize NPU preprocessor
print("\nTest 5: Initializing NPU preprocessor...")
try:
    preprocessor = NPUMelPreprocessor(fallback_to_cpu=True)
    if preprocessor.npu_available:
        print("  ✅ NPU preprocessor initialized (NPU mode)")
    else:
        print("  ⚠️  NPU preprocessor initialized (CPU fallback)")
except Exception as e:
    print(f"  ❌ Failed to initialize preprocessor: {e}")
    sys.exit(1)

# Test 6: Process test audio
print("\nTest 6: Processing test audio...")
try:
    # Generate 1 second sine wave at 1 kHz
    sample_rate = 16000
    duration = 1.0
    freq = 1000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

    print(f"  Input: {len(audio)} samples ({duration}s @ {sample_rate}Hz)")

    # Process
    mel_features = preprocessor.process_audio(audio)

    print(f"  ✅ Output: {mel_features.shape} (mels, frames)")

    # Get metrics
    metrics = preprocessor.get_performance_metrics()
    print(f"     Frames processed: {metrics['total_frames']}")
    print(f"     Backend: {'NPU' if metrics['npu_available'] else 'CPU'}")

except Exception as e:
    print(f"  ❌ Failed to process audio: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Check WhisperX backend availability
print("\nTest 7: Checking WhisperX backend availability...")
try:
    from faster_whisper import WhisperModel
    print("  ✅ faster-whisper available")
    backend = "faster-whisper"
except ImportError:
    print("  ⚠️  faster-whisper not available")
    try:
        import whisperx
        print("  ✅ whisperx available")
        backend = "whisperx"
    except ImportError:
        print("  ❌ No Whisper backend available")
        backend = None

# Test 8: Quick WhisperX wrapper test (without transcription)
if backend:
    print("\nTest 8: Initializing WhisperXNPU wrapper...")
    try:
        model = WhisperXNPU(model_size="base", enable_npu=True)
        print(f"  ✅ WhisperXNPU initialized")
        print(f"     Backend: {model.backend}")
        print(f"     Model: {model.model_size}")
        print(f"     NPU available: {model.npu_available}")
        model.close()
    except Exception as e:
        print(f"  ❌ Failed to initialize WhisperXNPU: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\nTest 8: Skipping WhisperXNPU test (no backend)")

# Cleanup
preprocessor.close()

# Summary
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
print(f"✅ Modules: Imported successfully")
print(f"{'✅' if os.path.exists('/dev/accel/accel0') else '⚠️ '} NPU Device: {'Available' if os.path.exists('/dev/accel/accel0') else 'Not found (CPU fallback)'}")
print(f"{'✅' if xclbin_path.exists() else '⚠️ '} XCLBIN: {'Available' if xclbin_path.exists() else 'Not found (CPU fallback)'}")
print(f"✅ Preprocessor: Working ({('NPU' if preprocessor.npu_available else 'CPU')} mode)")
print(f"{'✅' if backend else '❌'} Whisper Backend: {backend if backend else 'Not available'}")
print("=" * 70)

if preprocessor.npu_available and backend:
    print("\n🎉 All components operational! Ready for production use.")
    print("\nNext steps:")
    print("  1. Test with real audio: python3 whisperx_npu_wrapper.py audio.wav")
    print("  2. Run benchmark: python3 npu_benchmark.py audio.wav")
elif backend:
    print("\n⚠️  NPU not available, but CPU fallback working.")
    print("   Check XRT installation and NPU device availability.")
    print("\nNext steps:")
    print("  1. Install XRT: See README_NPU_INTEGRATION.md")
    print("  2. Build XCLBIN: cd mel_kernels && ./build_mel_complete.sh")
else:
    print("\n❌ Whisper backend not available.")
    print("   Install faster-whisper: pip install faster-whisper")

print()
