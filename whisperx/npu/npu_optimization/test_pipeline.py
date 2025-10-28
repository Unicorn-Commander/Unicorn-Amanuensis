#!/usr/bin/env python3
"""
Quick Integration Test for Librosa + ONNX Pipeline
==================================================
Tests the complete pipeline with real audio
"""

import sys
from pathlib import Path
import logging

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from librosa_onnx_pipeline import LibrosaONNXWhisper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pipeline():
    """Test the pipeline with existing audio"""

    # Paths
    model_path = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/onnx"
    audio_path = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/test_audio_jfk.wav"

    print("="*80)
    print("LIBROSA + ONNX PIPELINE INTEGRATION TEST")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Audio: {audio_path}")
    print("="*80)

    # Test 1: CPU Execution Provider
    print("\n" + "="*80)
    print("TEST 1: CPU EXECUTION PROVIDER")
    print("="*80)
    try:
        pipeline_cpu = LibrosaONNXWhisper(
            model_path=model_path,
            execution_provider='CPUExecutionProvider',
            use_int8=False
        )
        result_cpu = pipeline_cpu.transcribe(audio_path)

        print("\nRESULTS (CPU):")
        print(f"  Text: '{result_cpu['text']}'")
        print(f"  Duration: {result_cpu['duration']:.2f}s")
        print(f"  Total time: {result_cpu['timings']['total']:.2f}s")
        print(f"  Realtime factor: {result_cpu['realtime_factors']['total']:.1f}x")
        print("\n  Stage breakdown:")
        print(f"    Mel:     {result_cpu['timings']['mel_spectrogram']:.4f}s ({result_cpu['realtime_factors']['mel']:.1f}x)")
        print(f"    Encoder: {result_cpu['timings']['encoder']:.4f}s ({result_cpu['realtime_factors']['encoder']:.1f}x)")
        print(f"    Decoder: {result_cpu['timings']['decoder']:.4f}s ({result_cpu['realtime_factors']['decoder']:.1f}x)")
        print("\n  TEST 1: PASSED ✓")
    except Exception as e:
        print(f"\n  TEST 1: FAILED ✗")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: INT8 Models (if available)
    print("\n" + "="*80)
    print("TEST 2: INT8 MODELS (if available)")
    print("="*80)
    try:
        pipeline_int8 = LibrosaONNXWhisper(
            model_path=model_path,
            execution_provider='CPUExecutionProvider',
            use_int8=True
        )
        result_int8 = pipeline_int8.transcribe(audio_path)

        print("\nRESULTS (INT8):")
        print(f"  Text: '{result_int8['text']}'")
        print(f"  Duration: {result_int8['duration']:.2f}s")
        print(f"  Total time: {result_int8['timings']['total']:.2f}s")
        print(f"  Realtime factor: {result_int8['realtime_factors']['total']:.1f}x")

        # Compare with FP32
        if 'result_cpu' in locals():
            speedup = result_cpu['timings']['total'] / result_int8['timings']['total']
            print(f"\n  INT8 vs FP32 speedup: {speedup:.2f}x")

        print("\n  TEST 2: PASSED ✓")
    except FileNotFoundError as e:
        print(f"\n  TEST 2: SKIPPED (INT8 models not available)")
        print(f"  {e}")
    except Exception as e:
        print(f"\n  TEST 2: FAILED ✗")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: OpenVINO (if available)
    print("\n" + "="*80)
    print("TEST 3: OPENVINO EXECUTION PROVIDER (if available)")
    print("="*80)
    try:
        import onnxruntime as ort
        if 'OpenVINOExecutionProvider' in ort.get_available_providers():
            pipeline_ov = LibrosaONNXWhisper(
                model_path=model_path,
                execution_provider='OpenVINOExecutionProvider',
                use_int8=False
            )
            result_ov = pipeline_ov.transcribe(audio_path)

            print("\nRESULTS (OpenVINO):")
            print(f"  Text: '{result_ov['text']}'")
            print(f"  Duration: {result_ov['duration']:.2f}s")
            print(f"  Total time: {result_ov['timings']['total']:.2f}s")
            print(f"  Realtime factor: {result_ov['realtime_factors']['total']:.1f}x")

            # Compare with CPU
            if 'result_cpu' in locals():
                speedup = result_cpu['timings']['total'] / result_ov['timings']['total']
                print(f"\n  OpenVINO vs CPU speedup: {speedup:.2f}x")

            print("\n  TEST 3: PASSED ✓")
        else:
            print("\n  TEST 3: SKIPPED (OpenVINO not available)")
    except Exception as e:
        print(f"\n  TEST 3: FAILED ✗")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("\nIntegration test complete!")
    print("\nNext steps:")
    print("  1. Run full benchmark: python3 benchmark_librosa_onnx.py")
    print("  2. Test with different audio: python3 librosa_onnx_pipeline.py --audio <file>")
    print("  3. Optimize bottlenecks identified in benchmark")
    print("="*80)


if __name__ == "__main__":
    test_pipeline()
