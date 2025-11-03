#!/usr/bin/env python3
"""
Test with 35-second audio to trigger chunked processing
"""

import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'npu'))
sys.path.insert(0, os.path.join(current_dir, 'npu', 'npu_optimization'))

# Import directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "onnx_whisper_npu",
    os.path.join(current_dir, 'npu', 'npu_optimization', 'onnx_whisper_npu.py')
)
onnx_whisper_npu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(onnx_whisper_npu)
ONNXWhisperNPU = onnx_whisper_npu.ONNXWhisperNPU

def main():
    print("\n" + "="*70)
    print("LONG AUDIO TEST (35 seconds - triggers chunking)")
    print("="*70 + "\n")

    # Initialize
    decoder = ONNXWhisperNPU()
    if not decoder.initialize(model_size="base"):
        print("❌ Failed to initialize")
        return False

    print("✅ Decoder initialized\n")

    # Test with 35-second audio
    test_file = "/tmp/test_long_speech.wav"
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False

    print(f"Testing with: {test_file}")
    print()

    try:
        result = decoder.transcribe_audio(test_file)

        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"✅ Completed in {result['processing_time']:.2f}s")
        print(f"\nTranscription:")
        print(f"  '{result['text']}'")
        print(f"\nMetrics:")
        print(f"  Audio duration: {result['audio_duration']:.1f}s")
        print(f"  Real-time factor: {result['real_time_factor']:.2f}x")
        print(f"  Chunked: {result.get('chunked', False)}")
        print(f"  Num chunks: {result.get('num_chunks', 0)}")
        print(f"\nSegments:")
        for seg in result.get('segments', []):
            print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}")

        return True

    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
