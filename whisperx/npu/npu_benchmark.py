#!/usr/bin/env python3
"""
NPU Mel Preprocessing Benchmark

This script benchmarks NPU mel preprocessing against CPU librosa to measure:
- Speed improvement (target: 6x for preprocessing alone)
- Accuracy (correlation with CPU reference)
- End-to-end WhisperX performance

Author: Magic Unicorn Unconventional Technology & Stuff Inc.
Date: October 28, 2025
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

# Add NPU modules to path
sys.path.insert(0, str(Path(__file__).parent))

from npu_mel_preprocessing import NPUMelPreprocessor
from whisperx_npu_wrapper import WhisperXNPU

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def benchmark_mel_preprocessing(audio_path: str, n_runs: int = 10) -> Dict:
    """
    Benchmark NPU mel preprocessing vs CPU librosa.

    Args:
        audio_path: Path to audio file
        n_runs: Number of benchmark runs (default: 10)

    Returns:
        results: Dictionary with benchmark results
    """
    logger.info("=" * 70)
    logger.info("MEL PREPROCESSING BENCHMARK: NPU vs CPU")
    logger.info("=" * 70)

    # Load audio
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / sr
    logger.info(f"\nAudio: {audio_path}")
    logger.info(f"Duration: {duration:.2f}s")
    logger.info(f"Samples: {len(audio)}")
    logger.info(f"Sample rate: {sr}Hz")

    # ===== CPU Benchmark (librosa) =====
    logger.info("\n" + "-" * 70)
    logger.info("CPU Benchmark (librosa)")
    logger.info("-" * 70)

    cpu_times = []
    for i in range(n_runs):
        start = time.time()

        mel_cpu = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=80,
            fmin=0,
            fmax=sr // 2
        )
        mel_cpu_log = librosa.power_to_db(mel_cpu, ref=np.max)

        elapsed = time.time() - start
        cpu_times.append(elapsed)
        logger.info(f"  Run {i+1}/{n_runs}: {elapsed*1000:.2f}ms")

    cpu_time_avg = np.mean(cpu_times)
    cpu_time_std = np.std(cpu_times)
    cpu_rtf = duration / cpu_time_avg

    logger.info(f"\nCPU Results:")
    logger.info(f"  Average time: {cpu_time_avg*1000:.2f} ± {cpu_time_std*1000:.2f}ms")
    logger.info(f"  Real-time factor: {cpu_rtf:.2f}x")
    logger.info(f"  Output shape: {mel_cpu.shape}")

    # ===== NPU Benchmark =====
    logger.info("\n" + "-" * 70)
    logger.info("NPU Benchmark")
    logger.info("-" * 70)

    try:
        # Initialize NPU preprocessor
        npu_preprocessor = NPUMelPreprocessor(fallback_to_cpu=False)

        if not npu_preprocessor.npu_available:
            logger.warning("NPU not available - skipping NPU benchmark")
            npu_preprocessor.close()
            return {
                "cpu_time": cpu_time_avg,
                "npu_time": None,
                "speedup": None,
                "accuracy": None,
                "npu_available": False
            }

        npu_times = []
        for i in range(n_runs):
            npu_preprocessor.reset_metrics()
            start = time.time()

            mel_npu = npu_preprocessor.process_audio(audio)

            elapsed = time.time() - start
            npu_times.append(elapsed)
            logger.info(f"  Run {i+1}/{n_runs}: {elapsed*1000:.2f}ms")

        npu_time_avg = np.mean(npu_times)
        npu_time_std = np.std(npu_times)
        npu_rtf = duration / npu_time_avg

        logger.info(f"\nNPU Results:")
        logger.info(f"  Average time: {npu_time_avg*1000:.2f} ± {npu_time_std*1000:.2f}ms")
        logger.info(f"  Real-time factor: {npu_rtf:.2f}x")
        logger.info(f"  Output shape: {mel_npu.shape}")

        # ===== Accuracy Comparison =====
        logger.info("\n" + "-" * 70)
        logger.info("Accuracy Comparison")
        logger.info("-" * 70)

        # Reshape if needed
        if mel_cpu.shape != mel_npu.shape:
            logger.warning(f"Shape mismatch: CPU {mel_cpu.shape} vs NPU {mel_npu.shape}")
            # Transpose NPU to match CPU shape
            mel_npu_cmp = mel_npu.T if mel_npu.shape[0] != mel_cpu.shape[0] else mel_npu
        else:
            mel_npu_cmp = mel_npu

        # Normalize both for comparison
        mel_cpu_norm = (mel_cpu - np.mean(mel_cpu)) / (np.std(mel_cpu) + 1e-8)
        mel_npu_norm = (mel_npu_cmp - np.mean(mel_npu_cmp)) / (np.std(mel_npu_cmp) + 1e-8)

        # Calculate correlation
        correlation = np.corrcoef(mel_cpu_norm.flatten(), mel_npu_norm.flatten())[0, 1]

        # Calculate MSE
        mse = np.mean((mel_cpu_norm - mel_npu_norm) ** 2)

        # Calculate max absolute difference
        max_diff = np.max(np.abs(mel_cpu_norm - mel_npu_norm))

        logger.info(f"  Correlation: {correlation:.6f}")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  Max diff: {max_diff:.6f}")

        # ===== Speedup Calculation =====
        speedup = cpu_time_avg / npu_time_avg

        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 70)
        logger.info(f"CPU time:       {cpu_time_avg*1000:.2f}ms ({cpu_rtf:.2f}x realtime)")
        logger.info(f"NPU time:       {npu_time_avg*1000:.2f}ms ({npu_rtf:.2f}x realtime)")
        logger.info(f"Speedup:        {speedup:.2f}x")
        logger.info(f"Accuracy:       {correlation:.6f} correlation")
        logger.info("=" * 70)

        npu_preprocessor.close()

        return {
            "cpu_time": cpu_time_avg,
            "npu_time": npu_time_avg,
            "speedup": speedup,
            "accuracy": correlation,
            "mse": mse,
            "max_diff": max_diff,
            "cpu_rtf": cpu_rtf,
            "npu_rtf": npu_rtf,
            "npu_available": True
        }

    except Exception as e:
        logger.error(f"NPU benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "cpu_time": cpu_time_avg,
            "npu_time": None,
            "speedup": None,
            "accuracy": None,
            "npu_available": False,
            "error": str(e)
        }


def benchmark_end_to_end(audio_path: str, model_size: str = "base") -> Dict:
    """
    Benchmark end-to-end WhisperX performance with and without NPU.

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (default: "base")

    Returns:
        results: Dictionary with end-to-end benchmark results
    """
    logger.info("\n" + "=" * 70)
    logger.info("END-TO-END WHISPERX BENCHMARK")
    logger.info("=" * 70)

    # Test with NPU
    logger.info("\n" + "-" * 70)
    logger.info("Testing with NPU acceleration...")
    logger.info("-" * 70)

    model_npu = WhisperXNPU(model_size=model_size, enable_npu=True)
    result_npu = model_npu.transcribe(audio_path)
    summary_npu = model_npu.get_performance_summary()
    model_npu.close()

    # Test without NPU
    logger.info("\n" + "-" * 70)
    logger.info("Testing without NPU acceleration...")
    logger.info("-" * 70)

    model_cpu = WhisperXNPU(model_size=model_size, enable_npu=False)
    result_cpu = model_cpu.transcribe(audio_path)
    summary_cpu = model_cpu.get_performance_summary()
    model_cpu.close()

    # Compare results
    logger.info("\n" + "=" * 70)
    logger.info("END-TO-END COMPARISON")
    logger.info("=" * 70)
    logger.info(f"Duration:           {result_npu['duration']:.2f}s")
    logger.info(f"\nWith NPU:")
    logger.info(f"  Processing time:  {result_npu['processing_time']:.4f}s")
    logger.info(f"  NPU time:         {result_npu['npu_time']:.4f}s")
    logger.info(f"  Inference time:   {result_npu['inference_time']:.4f}s")
    logger.info(f"  RTF:              {result_npu['rtf']:.2f}x")
    logger.info(f"\nWithout NPU:")
    logger.info(f"  Processing time:  {result_cpu['processing_time']:.4f}s")
    logger.info(f"  Inference time:   {result_cpu['inference_time']:.4f}s")
    logger.info(f"  RTF:              {result_cpu['rtf']:.2f}x")
    logger.info(f"\nSpeedup:            {result_cpu['processing_time']/result_npu['processing_time']:.2f}x")
    logger.info("=" * 70)

    # Check text accuracy
    text_match = result_npu['text'] == result_cpu['text']
    logger.info(f"\nText match: {text_match}")
    if not text_match:
        logger.warning("Text mismatch detected!")
        logger.info(f"NPU text length: {len(result_npu['text'])}")
        logger.info(f"CPU text length: {len(result_cpu['text'])}")

    return {
        "npu_result": result_npu,
        "cpu_result": result_cpu,
        "speedup": result_cpu['processing_time'] / result_npu['processing_time'],
        "text_match": text_match
    }


def main():
    """Main benchmark function."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark NPU mel preprocessing vs CPU")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs (default: 10)")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size (default: base)")
    parser.add_argument("--mel-only", action="store_true", help="Only benchmark mel preprocessing")
    parser.add_argument("--end-to-end", action="store_true", help="Only benchmark end-to-end")

    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        logger.error(f"Audio file not found: {args.audio_file}")
        sys.exit(1)

    # Run benchmarks
    if args.mel_only:
        results = benchmark_mel_preprocessing(args.audio_file, args.runs)
    elif args.end_to_end:
        results = benchmark_end_to_end(args.audio_file, args.model)
    else:
        # Run both
        logger.info("\n" + "="*70)
        logger.info("RUNNING COMPLETE BENCHMARK SUITE")
        logger.info("="*70)

        mel_results = benchmark_mel_preprocessing(args.audio_file, args.runs)
        e2e_results = benchmark_end_to_end(args.audio_file, args.model)

        logger.info("\n" + "="*70)
        logger.info("FINAL SUMMARY")
        logger.info("="*70)
        logger.info("\nMel Preprocessing:")
        logger.info(f"  CPU time:    {mel_results.get('cpu_time', 0)*1000:.2f}ms")
        logger.info(f"  NPU time:    {mel_results.get('npu_time', 0)*1000:.2f}ms" if mel_results.get('npu_time') else "  NPU:         Not available")
        logger.info(f"  Speedup:     {mel_results.get('speedup', 0):.2f}x" if mel_results.get('speedup') else "  Speedup:     N/A")
        logger.info(f"  Accuracy:    {mel_results.get('accuracy', 0):.6f}" if mel_results.get('accuracy') else "  Accuracy:    N/A")

        logger.info("\nEnd-to-End WhisperX:")
        logger.info(f"  With NPU:    {e2e_results['npu_result']['rtf']:.2f}x realtime")
        logger.info(f"  Without NPU: {e2e_results['cpu_result']['rtf']:.2f}x realtime")
        logger.info(f"  Speedup:     {e2e_results['speedup']:.2f}x")
        logger.info(f"  Text match:  {e2e_results['text_match']}")
        logger.info("="*70)


if __name__ == "__main__":
    main()
