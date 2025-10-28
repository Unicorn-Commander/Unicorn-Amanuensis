#!/usr/bin/env python3
"""
Benchmark Librosa + ONNX Runtime Pipeline
=========================================
Performance testing for 220x realtime target

Tests:
- Various audio lengths (1s, 5s, 10s, 30s, 60s)
- Stage-by-stage timing breakdown
- Realtime factor calculation
- Comparison with 220x target

Goal: Measure current performance and identify bottlenecks
"""

import numpy as np
import librosa
import soundfile as sf
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from librosa_onnx_pipeline import LibrosaONNXWhisper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run comprehensive benchmarks on librosa + ONNX pipeline"""

    def __init__(self,
                 model_path: str,
                 execution_provider: str = 'CPUExecutionProvider',
                 use_int8: bool = False):
        """
        Initialize benchmark runner

        Args:
            model_path: Path to ONNX models
            execution_provider: ONNX Runtime provider
            use_int8: Use INT8 models
        """
        self.model_path = model_path
        self.execution_provider = execution_provider
        self.use_int8 = use_int8

        # Initialize pipeline
        logger.info("Initializing pipeline for benchmarking...")
        self.pipeline = LibrosaONNXWhisper(
            model_path=model_path,
            execution_provider=execution_provider,
            use_int8=use_int8
        )

    def generate_test_audio(self, duration: float, output_path: str) -> str:
        """
        Generate synthetic test audio

        Args:
            duration: Audio duration in seconds
            output_path: Where to save audio file

        Returns:
            Path to generated audio file
        """
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Generate sine wave (440 Hz A note)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        # Add some harmonics for more realistic signal
        audio += 0.3 * np.sin(2 * np.pi * 880 * t)  # Octave
        audio += 0.2 * np.sin(2 * np.pi * 1320 * t)  # Fifth

        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8

        # Save
        sf.write(output_path, audio, sample_rate)
        logger.info(f"Generated {duration}s test audio: {output_path}")

        return output_path

    def benchmark_single(self, audio_path: str) -> Dict[str, Any]:
        """
        Benchmark single audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Benchmark results
        """
        logger.info(f"\nBenchmarking: {audio_path}")

        # Get audio duration first
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(audio) / sr

        # Run transcription
        result = self.pipeline.transcribe(audio_path)

        # Extract timing breakdown
        timings = result['timings']
        rtf = result['realtime_factors']

        # Calculate percentages
        total_time = timings['total']
        breakdown = {
            'audio_load': {
                'time': timings['audio_load'],
                'percent': (timings['audio_load'] / total_time * 100) if total_time > 0 else 0
            },
            'mel_spectrogram': {
                'time': timings['mel_spectrogram'],
                'percent': (timings['mel_spectrogram'] / total_time * 100) if total_time > 0 else 0,
                'rtf': rtf['mel']
            },
            'encoder': {
                'time': timings['encoder'],
                'percent': (timings['encoder'] / total_time * 100) if total_time > 0 else 0,
                'rtf': rtf['encoder']
            },
            'decoder': {
                'time': timings['decoder'],
                'percent': (timings['decoder'] / total_time * 100) if total_time > 0 else 0,
                'rtf': rtf['decoder']
            }
        }

        return {
            'audio_path': audio_path,
            'duration': duration,
            'total_time': total_time,
            'realtime_factor': rtf['total'],
            'breakdown': breakdown,
            'text': result['text'],
            'tokens_generated': result['tokens_generated']
        }

    def run_benchmark_suite(self, test_durations: List[float]) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite

        Args:
            test_durations: List of audio durations to test (in seconds)

        Returns:
            Complete benchmark results
        """
        logger.info("="*80)
        logger.info("LIBROSA + ONNX RUNTIME BENCHMARK SUITE")
        logger.info("="*80)
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Execution Provider: {self.execution_provider}")
        logger.info(f"INT8: {self.use_int8}")
        logger.info(f"Test durations: {test_durations}")
        logger.info("="*80)

        results = []
        test_audio_dir = Path("/tmp/benchmark_audio")
        test_audio_dir.mkdir(exist_ok=True)

        for duration in test_durations:
            # Generate test audio
            audio_path = test_audio_dir / f"test_{duration}s.wav"
            self.generate_test_audio(duration, str(audio_path))

            # Run benchmark
            result = self.benchmark_single(str(audio_path))
            results.append(result)

            # Print summary
            self._print_result(result)

        # Print comparison table
        self._print_comparison_table(results)

        # Calculate statistics
        stats = self._calculate_statistics(results)

        return {
            'results': results,
            'statistics': stats,
            'config': {
                'model_path': str(self.model_path),
                'execution_provider': self.execution_provider,
                'int8': self.use_int8
            }
        }

    def _print_result(self, result: Dict[str, Any]):
        """Print single benchmark result"""
        print(f"\n{'='*80}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Total time: {result['total_time']:.4f}s")
        print(f"Realtime factor: {result['realtime_factor']:.1f}x")
        print(f"\nBreakdown:")
        for stage, data in result['breakdown'].items():
            rtf_str = f" ({data['rtf']:.1f}x)" if 'rtf' in data else ""
            print(f"  {stage:20s}: {data['time']:8.4f}s ({data['percent']:5.1f}%){rtf_str}")
        print(f"{'='*80}")

    def _print_comparison_table(self, results: List[Dict[str, Any]]):
        """Print comparison table for all results"""
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON TABLE")
        print("="*80)
        print(f"{'Duration':<12} {'Total Time':<12} {'RTF':<10} {'Status':<20} {'vs 220x Target'}")
        print("-"*80)

        target_rtf = 220.0
        for result in results:
            duration = result['duration']
            total_time = result['total_time']
            rtf = result['realtime_factor']

            # Status based on performance
            if rtf >= target_rtf:
                status = "TARGET MET!"
            elif rtf >= target_rtf * 0.5:
                status = "Good (50%+ target)"
            elif rtf >= target_rtf * 0.1:
                status = "Moderate (10%+ target)"
            else:
                status = "Below target"

            # Gap to target
            gap_percent = (rtf / target_rtf * 100) if target_rtf > 0 else 0

            print(f"{duration:>10.1f}s  {total_time:>10.4f}s  {rtf:>8.1f}x  {status:<20} {gap_percent:>6.1f}%")

        print("="*80)

    def _calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate statistics"""
        rtfs = [r['realtime_factor'] for r in results]
        total_times = [r['total_time'] for r in results]

        # Average breakdown percentages
        avg_breakdown = {}
        for stage in ['audio_load', 'mel_spectrogram', 'encoder', 'decoder']:
            times = [r['breakdown'][stage]['percent'] for r in results]
            avg_breakdown[stage] = {
                'avg_percent': np.mean(times),
                'min_percent': np.min(times),
                'max_percent': np.max(times)
            }

        stats = {
            'realtime_factor': {
                'mean': np.mean(rtfs),
                'min': np.min(rtfs),
                'max': np.max(rtfs),
                'std': np.std(rtfs)
            },
            'total_time': {
                'mean': np.mean(total_times),
                'min': np.min(total_times),
                'max': np.max(total_times),
                'std': np.std(total_times)
            },
            'avg_breakdown': avg_breakdown,
            'target_rtf': 220.0,
            'target_achievement': (np.mean(rtfs) / 220.0 * 100) if 220.0 > 0 else 0
        }

        # Print statistics
        print("\n" + "="*80)
        print("AGGREGATE STATISTICS")
        print("="*80)
        print(f"Average RTF: {stats['realtime_factor']['mean']:.1f}x (min: {stats['realtime_factor']['min']:.1f}x, max: {stats['realtime_factor']['max']:.1f}x)")
        print(f"Target RTF: {stats['target_rtf']:.1f}x")
        print(f"Target achievement: {stats['target_achievement']:.1f}%")
        print("\nAverage breakdown:")
        for stage, data in stats['avg_breakdown'].items():
            print(f"  {stage:20s}: {data['avg_percent']:5.1f}%")
        print("="*80)

        return stats

    def identify_bottlenecks(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify performance bottlenecks"""
        # Average percentages
        avg_percentages = {}
        for stage in ['audio_load', 'mel_spectrogram', 'encoder', 'decoder']:
            percentages = [r['breakdown'][stage]['percent'] for r in results]
            avg_percentages[stage] = np.mean(percentages)

        # Sort by percentage (highest first)
        sorted_stages = sorted(avg_percentages.items(), key=lambda x: x[1], reverse=True)

        print("\n" + "="*80)
        print("BOTTLENECK ANALYSIS")
        print("="*80)
        print("Stages ranked by time consumption:")
        for i, (stage, percent) in enumerate(sorted_stages, 1):
            print(f"{i}. {stage:20s}: {percent:5.1f}%")

        # Recommendations
        print("\nOptimization recommendations:")
        if sorted_stages[0][0] == 'encoder':
            print("  1. ENCODER is the bottleneck - consider:")
            print("     - Using INT8 quantization (2-4x speedup)")
            print("     - Using OpenVINO execution provider")
            print("     - Custom NPU kernels for attention/matmul")
        if sorted_stages[1][0] == 'decoder':
            print("  2. DECODER is second bottleneck - consider:")
            print("     - Optimizing KV cache implementation")
            print("     - Beam search optimization")
            print("     - Custom NPU kernels for autoregressive decoding")
        if avg_percentages['mel_spectrogram'] > 10:
            print("  3. MEL SPECTROGRAM taking >10% - consider:")
            print("     - Custom NPU mel kernel (potential 10-20x speedup)")
            print("     - Batch processing multiple chunks")

        print("="*80)

        return {
            'sorted_stages': sorted_stages,
            'recommendations': "See printed analysis above"
        }


def main():
    """Run benchmark suite"""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark librosa + ONNX pipeline')
    parser.add_argument('--model', type=str,
                       default='/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/onnx',
                       help='Path to ONNX model directory')
    parser.add_argument('--provider', type=str, default='CPUExecutionProvider',
                       choices=['CPUExecutionProvider', 'OpenVINOExecutionProvider'],
                       help='ONNX Runtime execution provider')
    parser.add_argument('--int8', action='store_true',
                       help='Use INT8 quantized models')
    parser.add_argument('--durations', type=str, default='1,5,10,30,60',
                       help='Comma-separated list of test durations (seconds)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output JSON file for results')

    args = parser.parse_args()

    # Parse durations
    test_durations = [float(d) for d in args.durations.split(',')]

    # Initialize benchmark runner
    runner = BenchmarkRunner(
        model_path=args.model,
        execution_provider=args.provider,
        use_int8=args.int8
    )

    # Run benchmark suite
    results = runner.run_benchmark_suite(test_durations)

    # Identify bottlenecks
    bottlenecks = runner.identify_bottlenecks(results['results'])
    results['bottlenecks'] = bottlenecks

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_path}")

    # Final summary
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"Average RTF: {results['statistics']['realtime_factor']['mean']:.1f}x")
    print(f"Target RTF: {results['statistics']['target_rtf']:.1f}x")
    print(f"Achievement: {results['statistics']['target_achievement']:.1f}%")
    print(f"Results saved: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
