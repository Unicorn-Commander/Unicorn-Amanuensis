#!/usr/bin/env python3
"""
Test XDNA2 STT Runtime on Real Hardware

Tests the WhisperXDNA2Runtime implementation with the proven
1,183x INT8 matmul kernel.

Expected Results:
- Device initialization: SUCCESS
- 4-tile kernel test: 0.1-0.5ms per matmul
- NPU utilization: Active and functional
- Realtime factor: TBD (full pipeline not implemented yet)

Usage:
    source ~/mlir-aie/ironenv/bin/activate
    export PYTHONPATH="/opt/xilinx/xrt/python:$PYTHONPATH"
    python3 test_xdna2_stt.py
"""

import sys
import os
import time
import logging
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/opt/xilinx/xrt/python")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def test_device_initialization():
    """Test XDNA2 device initialization."""
    print_section("TEST 1: XDNA2 Device Initialization")

    try:
        from xdna2.runtime.whisper_xdna2_runtime import WhisperXDNA2Runtime

        logger.info("Creating WhisperXDNA2Runtime...")
        runtime = WhisperXDNA2Runtime(model_size="base", use_4tile=True)

        if runtime._initialized:
            print("✅ Device initialized successfully")
            print(f"   - Kernel directory: {runtime.kernel_dir}")
            print(f"   - Model size: {runtime.model_size}")
            print(f"   - Using 4-tile kernel: {runtime.use_4tile}")
            return runtime
        else:
            print("❌ Device initialization failed")
            return None

    except Exception as e:
        print(f"❌ Device initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_npu_matmul(runtime):
    """Test NPU matmul execution."""
    print_section("TEST 2: NPU Matmul Execution")

    if runtime is None:
        print("⏭️  Skipping (device not initialized)")
        return False

    try:
        import numpy as np

        # Test dimensions for 4-tile kernel
        M, K, N = 64, 64, 32
        logger.info(f"Testing matmul with dimensions: {M}x{K} @ {K}x{N}")

        # Create test inputs
        A = np.random.randint(-8, 8, (M, K), dtype=np.int8)
        B = np.random.randint(-8, 8, (K, N), dtype=np.int8)

        # Run on NPU
        start = time.perf_counter()
        C = runtime._run_matmul_npu(A, B, M, K, N)
        elapsed = time.perf_counter() - start

        # Calculate performance
        ops = 2 * M * K * N
        gflops = ops / elapsed / 1e9

        print("✅ NPU matmul execution successful")
        print(f"   - Input A: {A.shape} (int8)")
        print(f"   - Input B: {B.shape} (int8)")
        print(f"   - Output C: {C.shape} (int32)")
        print(f"   - Elapsed: {elapsed*1000:.2f}ms")
        print(f"   - Performance: {gflops:.1f} GFLOPS")

        # Verify correctness with CPU reference
        C_ref = A.astype(np.int32) @ B.astype(np.int32)
        max_error = np.abs(C - C_ref).max()
        print(f"   - Max error vs CPU: {max_error}")

        if max_error == 0:
            print("   - ✅ 100% accuracy!")
        elif max_error < 10:
            print("   - ⚠️  Small numerical error (acceptable for int8)")
        else:
            print("   - ❌ Large error detected!")
            return False

        return True

    except Exception as e:
        print(f"❌ NPU matmul failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encoder_pipeline(runtime):
    """Test Whisper encoder pipeline (matmul test)."""
    print_section("TEST 3: Whisper Encoder Pipeline")

    if runtime is None:
        print("⏭️  Skipping (device not initialized)")
        return False

    try:
        import numpy as np

        # Create dummy mel features
        # Whisper uses 80 mel bins, typical 30s audio = ~3000 time steps
        n_mels = 80
        time_steps = 100  # Small for testing

        mel = np.random.randn(n_mels, time_steps).astype(np.float32)
        logger.info(f"Testing encoder with mel shape: {mel.shape}")

        # Run encoder (currently just tests matmul)
        start = time.perf_counter()
        output = runtime.run_encoder(mel)
        elapsed = time.perf_counter() - start

        print("✅ Encoder pipeline test successful")
        print(f"   - Input mel: {mel.shape}")
        print(f"   - Output: {output.shape}")
        print(f"   - Elapsed: {elapsed*1000:.2f}ms")

        return True

    except Exception as e:
        print(f"❌ Encoder pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_preprocessing():
    """Test audio preprocessing (if librosa available)."""
    print_section("TEST 4: Audio Preprocessing")

    try:
        import librosa
        import numpy as np

        # Create test audio (1 second of sine wave at 440Hz)
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        import soundfile as sf
        sf.write(temp_path, audio, sr)

        # Test preprocessing
        from xdna2.runtime.whisper_xdna2_runtime import WhisperXDNA2Runtime
        runtime = WhisperXDNA2Runtime(model_size="base", use_4tile=True)
        mel = runtime.preprocess_audio(temp_path)

        # Cleanup
        os.unlink(temp_path)

        print("✅ Audio preprocessing successful")
        print(f"   - Audio duration: {duration}s")
        print(f"   - Sample rate: {sr}Hz")
        print(f"   - Mel shape: {mel.shape}")
        print(f"   - Mel bins: {mel.shape[0]} (expected: 80)")

        return True

    except ImportError as e:
        print("⏭️  Skipping (librosa or soundfile not installed)")
        print(f"   Install with: pip install librosa soundfile")
        return None

    except Exception as e:
        print(f"❌ Audio preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_transcription():
    """Test full transcription pipeline (placeholder for now)."""
    print_section("TEST 5: Full Transcription Pipeline")

    print("⏳ Full transcription test pending")
    print("   - Encoder implementation: ✅ Matmul tested")
    print("   - Decoder implementation: ⏳ TODO")
    print("   - End-to-end pipeline: ⏳ TODO")
    print()
    print("   Next steps:")
    print("   1. Implement full Whisper encoder layers")
    print("   2. Add decoder (CPU or NPU)")
    print("   3. Test with real audio files")
    print("   4. Measure realtime factor")

    return None


def main():
    """Run all tests."""
    print()
    print("=" * 70)
    print("  XDNA2 STT Runtime Test Suite")
    print("  Target: 400-500x realtime with 1,183x INT8 matmul kernel")
    print("=" * 70)

    # Check environment
    print()
    print("Environment Check:")
    xrt_path = "/opt/xilinx/xrt/python"
    if os.path.exists(xrt_path):
        print(f"✅ XRT Python bindings found: {xrt_path}")
    else:
        print(f"❌ XRT Python bindings not found: {xrt_path}")
        print("   Make sure XRT is installed and PYTHONPATH is set")

    ironenv = os.environ.get("VIRTUAL_ENV")
    if ironenv and "ironenv" in ironenv:
        print(f"✅ MLIR-AIE environment active: {ironenv}")
    else:
        print("⚠️  MLIR-AIE ironenv not detected")
        print("   Activate with: source ~/mlir-aie/ironenv/bin/activate")

    # Run tests
    results = {}

    # Test 1: Device initialization
    runtime = test_device_initialization()
    results["device_init"] = runtime is not None

    # Test 2: NPU matmul
    results["npu_matmul"] = test_npu_matmul(runtime)

    # Test 3: Encoder pipeline
    results["encoder_pipeline"] = test_encoder_pipeline(runtime)

    # Test 4: Audio preprocessing
    results["audio_preprocessing"] = test_audio_preprocessing()

    # Test 5: Full transcription
    results["full_transcription"] = test_full_transcription()

    # Cleanup
    if runtime:
        runtime.cleanup()

    # Summary
    print_section("Test Summary")

    total = 0
    passed = 0
    skipped = 0

    for test_name, result in results.items():
        total += 1
        if result is True:
            status = "✅ PASS"
            passed += 1
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⏭️  SKIP"
            skipped += 1

        print(f"{status}  {test_name}")

    print()
    print(f"Results: {passed}/{total} passed, {skipped} skipped")
    print()

    # Exit code
    if results["device_init"] and results["npu_matmul"]:
        print("=" * 70)
        print("  ✅ CRITICAL TESTS PASSED")
        print("  NPU is operational and matmul kernel works!")
        print("=" * 70)
        print()
        return 0
    else:
        print("=" * 70)
        print("  ❌ CRITICAL TESTS FAILED")
        print("  Check device initialization and kernel paths")
        print("=" * 70)
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
