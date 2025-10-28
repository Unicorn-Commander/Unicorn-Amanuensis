#!/usr/bin/env python3
"""
WhisperX NPU Integration Test Script
Tests end-to-end transcription with both simple and optimized kernels

Team 2: WhisperX Integration Lead
Date: October 28, 2025
"""

import sys
import os
import time
import json
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # whisperx/
sys.path.insert(0, str(Path(__file__).parent.parent))  # npu/

import numpy as np

# Known JFK transcript for WER calculation
JFK_REFERENCE = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate between reference and hypothesis

    Args:
        reference: Reference transcript (ground truth)
        hypothesis: Hypothesis transcript (model output)

    Returns:
        wer: Word Error Rate (0.0 = perfect, 1.0 = completely wrong)
    """
    # Simple WER implementation using Levenshtein distance
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Build distance matrix
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer


def test_kernel(kernel_name, xclbin_path, insts_path, audio_path, reference_text):
    """
    Test WhisperX with specific NPU kernel

    Args:
        kernel_name: Name of kernel (e.g., "simple", "optimized")
        xclbin_path: Path to XCLBIN file
        insts_path: Path to instruction binary
        audio_path: Path to test audio file
        reference_text: Reference transcript for WER calculation

    Returns:
        results: Dictionary with test results
    """
    print(f"\n{'='*80}")
    print(f"TESTING: {kernel_name.upper()} KERNEL")
    print(f"{'='*80}")
    print(f"XCLBIN: {xclbin_path}")
    print(f"Instructions: {insts_path}")
    print(f"Audio: {audio_path}")

    # Check files exist
    if not os.path.exists(xclbin_path):
        print(f"ERROR: XCLBIN not found: {xclbin_path}")
        return None
    if not os.path.exists(insts_path):
        print(f"ERROR: Instructions not found: {insts_path}")
        return None
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio not found: {audio_path}")
        return None

    try:
        # Import WhisperX NPU wrapper
        from whisperx_npu_wrapper import WhisperXNPU

        # Initialize model with NPU
        print(f"\nInitializing WhisperX with {kernel_name} kernel...")
        start_init = time.time()

        model = WhisperXNPU(
            model_size="base",
            npu_xclbin=xclbin_path,
            enable_npu=True
        )

        init_time = time.time() - start_init
        print(f"Initialization complete: {init_time:.3f}s")

        # Load audio to get duration
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_duration = len(audio) / sr
        print(f"Audio duration: {audio_duration:.2f}s")

        # Transcribe
        print(f"\nTranscribing with {kernel_name} kernel...")
        start_transcribe = time.time()

        result = model.transcribe(audio_path)

        transcribe_time = time.time() - start_transcribe

        # Extract results
        text = result.get("text", "").strip()
        npu_accelerated = result.get("npu_accelerated", False)
        npu_time = result.get("npu_time", 0.0)
        inference_time = result.get("inference_time", 0.0)
        processing_time = result.get("processing_time", 0.0)
        rtf = result.get("rtf", 0.0)

        # Calculate WER
        wer = calculate_wer(reference_text, text)

        # Print results
        print(f"\n{'-'*80}")
        print(f"RESULTS: {kernel_name.upper()}")
        print(f"{'-'*80}")
        print(f"Transcript: {text}")
        print(f"Reference:  {reference_text}")
        print(f"\nMetrics:")
        print(f"  Audio duration:     {audio_duration:.2f}s")
        print(f"  Processing time:    {processing_time:.4f}s")
        print(f"  NPU time:           {npu_time:.4f}s")
        print(f"  Inference time:     {inference_time:.4f}s")
        print(f"  Real-time factor:   {rtf:.2f}x")
        print(f"  NPU accelerated:    {npu_accelerated}")
        print(f"  Word Error Rate:    {wer:.4f} ({wer*100:.2f}%)")
        print(f"{'-'*80}")

        # Cleanup
        model.close()

        return {
            "kernel": kernel_name,
            "xclbin": xclbin_path,
            "audio_duration": audio_duration,
            "text": text,
            "reference": reference_text,
            "wer": wer,
            "npu_accelerated": npu_accelerated,
            "npu_time": npu_time,
            "inference_time": inference_time,
            "processing_time": processing_time,
            "rtf": rtf,
            "init_time": init_time,
            "success": True,
            "error": None
        }

    except Exception as e:
        print(f"\nERROR during {kernel_name} kernel test:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            "kernel": kernel_name,
            "xclbin": xclbin_path,
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def main():
    """Main test runner"""

    print("="*80)
    print("WHISPERX NPU INTEGRATION TEST")
    print("Testing mel preprocessing with Simple and Optimized kernels")
    print("="*80)

    # Test configuration
    base_dir = Path(__file__).parent
    audio_path = base_dir / "test_audio_jfk.wav"

    # Test cases
    tests = [
        {
            "name": "simple",
            "xclbin": base_dir / "build_fixed" / "mel_fixed_new.xclbin",
            "insts": base_dir / "build_fixed" / "insts_fixed.bin"
        },
        {
            "name": "optimized",
            "xclbin": base_dir / "build_optimized" / "mel_optimized_new.xclbin",
            "insts": base_dir / "build_optimized" / "insts_optimized.bin"
        }
    ]

    # Run tests
    results = []
    for test_config in tests:
        result = test_kernel(
            kernel_name=test_config["name"],
            xclbin_path=str(test_config["xclbin"]),
            insts_path=str(test_config["insts"]),
            audio_path=str(audio_path),
            reference_text=JFK_REFERENCE
        )

        if result:
            results.append(result)

    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    if len(results) == 2:
        simple = results[0]
        optimized = results[1]

        print(f"\n{'Metric':<25} {'Simple':<20} {'Optimized':<20} {'Improvement':<15}")
        print(f"{'-'*80}")

        if simple["success"] and optimized["success"]:
            # WER comparison
            wer_improvement = ((simple["wer"] - optimized["wer"]) / simple["wer"] * 100) if simple["wer"] > 0 else 0
            print(f"{'Word Error Rate':<25} {simple['wer']*100:>18.2f}% {optimized['wer']*100:>18.2f}% {wer_improvement:>13.1f}%")

            # Speed comparison
            rtf_improvement = ((optimized["rtf"] - simple["rtf"]) / simple["rtf"] * 100) if simple["rtf"] > 0 else 0
            print(f"{'Real-time Factor':<25} {simple['rtf']:>18.2f}x {optimized['rtf']:>18.2f}x {rtf_improvement:>13.1f}%")

            # Processing time comparison
            time_improvement = ((simple["processing_time"] - optimized["processing_time"]) / simple["processing_time"] * 100) if simple["processing_time"] > 0 else 0
            print(f"{'Processing Time':<25} {simple['processing_time']:>18.4f}s {optimized['processing_time']:>18.4f}s {time_improvement:>13.1f}%")

            # NPU time comparison
            npu_improvement = ((simple["npu_time"] - optimized["npu_time"]) / simple["npu_time"] * 100) if simple["npu_time"] > 0 else 0
            print(f"{'NPU Time':<25} {simple['npu_time']:>18.4f}s {optimized['npu_time']:>18.4f}s {npu_improvement:>13.1f}%")

            print(f"\n{'Transcript Quality':<25}")
            print(f"{'-'*80}")
            print(f"Simple:    {simple['text']}")
            print(f"Optimized: {optimized['text']}")
            print(f"Reference: {JFK_REFERENCE}")

            # Success criteria
            print(f"\n{'='*80}")
            print("SUCCESS CRITERIA")
            print(f"{'='*80}")

            criteria = {
                "Both kernels integrate successfully": simple["success"] and optimized["success"],
                "Optimized kernel better WER (target: 25-30% improvement)": wer_improvement >= 25.0,
                "No crashes or errors": simple["error"] is None and optimized["error"] is None,
                "Realtime factor measured": simple["rtf"] > 0 and optimized["rtf"] > 0
            }

            for criterion, passed in criteria.items():
                status = "PASS" if passed else "FAIL"
                print(f"[{status}] {criterion}")
        else:
            if not simple["success"]:
                print(f"Simple kernel FAILED: {simple.get('error', 'Unknown error')}")
            if not optimized["success"]:
                print(f"Optimized kernel FAILED: {optimized.get('error', 'Unknown error')}")

    # Save results to JSON
    results_path = base_dir / "integration_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    results = main()
