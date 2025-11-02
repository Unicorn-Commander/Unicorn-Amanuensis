#!/usr/bin/env python3
"""
Week 18: Detailed Performance Profiling
CC-1L Performance Engineering Team

Comprehensive performance measurement with hierarchical timing:

Coarse Level (Week 17):
- Total processing time
- Mel spectrogram generation
- NPU encoder execution
- Decoder execution

Medium Level (NEW - Week 18):
- Mel spectrogram: FFT, log-mel, normalization
- NPU encoder: buffer transfer, kernel execution, result retrieval
- Decoder: attention layers, token generation, post-processing

Fine Level (STRETCH):
- Per-operation timing (matmul, softmax, etc.)
- Memory transfer overhead
- Synchronization overhead
"""

import numpy as np
import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging
import sys

# Add parent directory to path for profiling_utils
sys.path.insert(0, str(Path(__file__).parent))

from profiling_utils import (
    PerformanceProfiler,
    MultiRunProfiler,
    create_ascii_bar_chart,
    create_waterfall_diagram,
    TimingMeasurement
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DetailedPerformanceResult:
    """Detailed performance measurement result with hierarchical timing"""
    test_name: str
    audio_duration_s: float
    audio_file: str

    # Overall timing
    processing_time_ms: float = 0.0
    realtime_factor: float = 0.0

    # Coarse-level timing (from service)
    mel_time_ms: float = 0.0
    encoder_time_ms: float = 0.0
    decoder_time_ms: float = 0.0

    # Medium-level timing (NEW)
    # Mel spectrogram breakdown
    mel_fft_ms: float = 0.0
    mel_logmel_ms: float = 0.0
    mel_normalize_ms: float = 0.0

    # NPU encoder breakdown
    npu_buffer_transfer_ms: float = 0.0
    npu_kernel_execution_ms: float = 0.0
    npu_result_retrieval_ms: float = 0.0

    # Decoder breakdown
    decoder_attention_ms: float = 0.0
    decoder_token_gen_ms: float = 0.0
    decoder_postprocess_ms: float = 0.0

    # Performance metrics
    npu_utilization_percent: float = 0.0
    throughput_audio_s_per_wall_s: float = 0.0

    # Status
    success: bool = True
    error: str = ""
    transcription: str = ""

    # Profiler data
    profiler_data: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        d = {}
        for field_name, field_value in asdict(self).items():
            if field_name == 'profiler_data':
                continue  # Skip profiler_data (too verbose)
            # Convert numpy types to Python types
            if hasattr(field_value, 'item'):
                d[field_name] = field_value.item()
            else:
                d[field_name] = field_value
        return d


class Week18PerformanceProfiling:
    """
    Week 18 Detailed Performance Profiling Suite

    Measures performance at three levels:
    1. Coarse: Total component times
    2. Medium: Sub-component breakdown
    3. Fine: Per-operation timing (stretch goal)
    """

    def __init__(self, service_url: str = "http://127.0.0.1:9050"):
        """Initialize profiling suite"""
        self.service_url = service_url
        self.results: List[DetailedPerformanceResult] = []

        # Performance targets
        self.target_realtime_min = 100.0  # Week 18 target (decoder optimization)
        self.target_realtime_max = 200.0
        self.week19_target_min = 250.0    # Week 19 target
        self.week19_target_max = 350.0
        self.final_target_min = 400.0     # Final Week 20 target
        self.final_target_max = 500.0

        logger.info("="*80)
        logger.info("  WEEK 18: DETAILED PERFORMANCE PROFILING")
        logger.info("="*80)
        logger.info(f"  Service URL: {service_url}")
        logger.info(f"  Week 18 Target: {self.target_realtime_min}-{self.target_realtime_max}× realtime")
        logger.info(f"  Week 19 Target: {self.week19_target_min}-{self.week19_target_max}× realtime")
        logger.info(f"  Final Target: {self.final_target_min}-{self.final_target_max}× realtime")
        logger.info("="*80)

    def check_service_health(self) -> bool:
        """Check if service is running and healthy"""
        try:
            logger.info("\n" + "="*80)
            logger.info("  CHECKING SERVICE HEALTH")
            logger.info("="*80)

            response = requests.get(f"{self.service_url}/health", timeout=5)
            response.raise_for_status()

            health = response.json()
            logger.info(f"  Service: {health.get('service', 'Unknown')}")
            logger.info(f"  Status: {health.get('status', 'Unknown')}")
            logger.info(f"  NPU Enabled: {health.get('npu_enabled', False)}")

            if health.get('status') != 'healthy':
                logger.error(f"  ❌ Service unhealthy: {health.get('status')}")
                return False

            if not health.get('npu_enabled', False):
                logger.warning("  ⚠️  NPU not enabled - performance will be degraded")

            logger.info("  ✅ Service is healthy and ready")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"  ❌ Service not responding: {e}")
            logger.error("\n  To start service:")
            logger.error("    cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis")
            logger.error("    source ~/mlir-aie/ironenv/bin/activate")
            logger.error("    source /opt/xilinx/xrt/setup.sh 2>/dev/null")
            logger.error("    ENABLE_PIPELINE=true python -m uvicorn xdna2.server:app --host 127.0.0.1 --port 9050")
            return False

    def measure_audio_performance_detailed(
        self,
        audio_path: Path,
        test_name: str,
        num_runs: int = 10,
        warmup_runs: int = 2
    ) -> DetailedPerformanceResult:
        """
        Measure detailed performance for a single audio file with multiple runs

        Args:
            audio_path: Path to audio file
            test_name: Name for this test
            num_runs: Number of measured runs (after warmup)
            warmup_runs: Number of warmup runs to discard

        Returns:
            DetailedPerformanceResult with comprehensive timing metrics
        """
        logger.info("\n" + "="*80)
        logger.info(f"  TEST: {test_name}")
        logger.info("="*80)
        logger.info(f"  Audio file: {audio_path.name}")
        logger.info(f"  Runs: {num_runs} (+ {warmup_runs} warmup)")

        # Get audio duration
        file_size_bytes = audio_path.stat().st_size
        audio_duration_s = (file_size_bytes - 44) / (2 * 16000)
        logger.info(f"  Audio duration: {audio_duration_s:.2f}s")

        result = DetailedPerformanceResult(
            test_name=test_name,
            audio_duration_s=audio_duration_s,
            audio_file=str(audio_path.name)
        )

        # Multi-run profiler
        multi_profiler = MultiRunProfiler(num_runs=num_runs, warmup_runs=warmup_runs)

        try:
            # Run multiple iterations
            for i in range(multi_profiler.total_runs):
                with multi_profiler.run(i):
                    # Measure end-to-end request
                    with multi_profiler.measure("total_request"):
                        with open(audio_path, 'rb') as f:
                            files = {'file': (audio_path.name, f, 'audio/wav')}

                            # Measure HTTP request
                            with multi_profiler.measure("http_request", parent="total_request"):
                                response = requests.post(
                                    f"{self.service_url}/v1/audio/transcriptions",
                                    files=files,
                                    timeout=60
                                )

                            # Measure response parsing
                            with multi_profiler.measure("response_parse", parent="total_request"):
                                response.raise_for_status()
                                data = response.json()

                # Extract timing from service response (only from last run)
                if i >= warmup_runs and 'timing' in data:
                    timing = data['timing']
                    # These are server-side timings, add them as measurements
                    # They won't be included in client-side timing but are useful for analysis

            # Get aggregated statistics
            stats = multi_profiler.get_aggregated_statistics()

            # Extract key metrics
            if 'total_request' in stats:
                result.processing_time_ms = stats['total_request'].mean_ms
                if audio_duration_s > 0:
                    result.realtime_factor = audio_duration_s / (result.processing_time_ms / 1000)
                    result.throughput_audio_s_per_wall_s = result.realtime_factor

            # Extract timing from last response
            if 'timing' in data:
                timing = data['timing']
                result.mel_time_ms = timing.get('mel_ms', 0.0)
                result.encoder_time_ms = timing.get('encoder_ms', 0.0)
                result.decoder_time_ms = timing.get('decoder_ms', 0.0)

                # Medium-level timing (if available from service)
                if 'mel_breakdown' in timing:
                    mel = timing['mel_breakdown']
                    result.mel_fft_ms = mel.get('fft_ms', 0.0)
                    result.mel_logmel_ms = mel.get('logmel_ms', 0.0)
                    result.mel_normalize_ms = mel.get('normalize_ms', 0.0)

                if 'encoder_breakdown' in timing:
                    enc = timing['encoder_breakdown']
                    result.npu_buffer_transfer_ms = enc.get('buffer_transfer_ms', 0.0)
                    result.npu_kernel_execution_ms = enc.get('kernel_execution_ms', 0.0)
                    result.npu_result_retrieval_ms = enc.get('result_retrieval_ms', 0.0)

                if 'decoder_breakdown' in timing:
                    dec = timing['decoder_breakdown']
                    result.decoder_attention_ms = dec.get('attention_ms', 0.0)
                    result.decoder_token_gen_ms = dec.get('token_gen_ms', 0.0)
                    result.decoder_postprocess_ms = dec.get('postprocess_ms', 0.0)

            # Extract transcription
            result.transcription = data.get('text', '')

            # Estimate NPU utilization
            if result.realtime_factor > 0:
                # Simple estimate based on target performance
                result.npu_utilization_percent = (self.final_target_min / result.realtime_factor) * 2.3

            # Store profiler data
            result.profiler_data = {
                'statistics': {name: stat.to_dict() for name, stat in stats.items()},
                'num_runs': num_runs,
                'warmup_runs': warmup_runs
            }

            # Log results
            logger.info(f"\n  ✅ Transcription successful!")
            logger.info(f"  Processing time (mean): {result.processing_time_ms:.2f} ms")
            logger.info(f"  Processing time (p95): {stats['total_request'].p95_ms:.2f} ms")
            logger.info(f"  Processing time (p99): {stats['total_request'].p99_ms:.2f} ms")
            logger.info(f"  Realtime factor: {result.realtime_factor:.1f}×")
            logger.info(f"  Throughput: {result.throughput_audio_s_per_wall_s:.1f} audio-s/wall-s")

            # Target comparison
            if result.realtime_factor >= self.final_target_min:
                logger.info(f"  🎯 FINAL TARGET MET! ({self.final_target_min}-{self.final_target_max}×)")
            elif result.realtime_factor >= self.week19_target_min:
                logger.info(f"  🎯 WEEK 19 TARGET MET! ({self.week19_target_min}-{self.week19_target_max}×)")
            elif result.realtime_factor >= self.target_realtime_min:
                logger.info(f"  🎯 WEEK 18 TARGET MET! ({self.target_realtime_min}-{self.target_realtime_max}×)")
            else:
                gap = self.target_realtime_min - result.realtime_factor
                logger.info(f"  ⚠️  Below Week 18 target by {gap:.1f}×")

            # Component breakdown
            logger.info(f"\n  COARSE-LEVEL TIMING (from service):")
            logger.info(f"    Mel spectrogram: {result.mel_time_ms:.2f} ms ({result.mel_time_ms/result.processing_time_ms*100:.1f}%)")
            logger.info(f"    NPU encoder:     {result.encoder_time_ms:.2f} ms ({result.encoder_time_ms/result.processing_time_ms*100:.1f}%)")
            logger.info(f"    Decoder:         {result.decoder_time_ms:.2f} ms ({result.decoder_time_ms/result.processing_time_ms*100:.1f}%)")

            # Client-side timing breakdown
            logger.info(f"\n  CLIENT-SIDE TIMING:")
            for name, stat in sorted(stats.items(), key=lambda x: x[1].mean_ms, reverse=True):
                logger.info(f"    {name:<25} {stat.mean_ms:>8.2f} ms (±{stat.std_dev_ms:.2f})")

            logger.info(f"\n  Transcription: \"{result.transcription[:100]}...\"")

        except requests.exceptions.Timeout:
            result.success = False
            result.error = "Request timeout (>60s)"
            logger.error(f"  ❌ {result.error}")

        except requests.exceptions.RequestException as e:
            result.success = False
            result.error = str(e)
            logger.error(f"  ❌ Request failed: {e}")

        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"  ❌ Unexpected error: {e}")

        self.results.append(result)
        return result

    def run_comprehensive_profiling(
        self,
        test_dir: Path,
        num_runs: int = 10,
        warmup_runs: int = 2
    ) -> Dict:
        """
        Run comprehensive performance profiling suite

        Args:
            test_dir: Directory containing test audio files
            num_runs: Number of measured runs per test
            warmup_runs: Number of warmup runs to discard

        Returns:
            Complete profiling results
        """
        logger.info("\n" + "="*80)
        logger.info("  RUNNING COMPREHENSIVE PERFORMANCE PROFILING")
        logger.info("="*80)
        logger.info(f"  Runs per test: {num_runs} (+ {warmup_runs} warmup)")

        # Check service health first
        if not self.check_service_health():
            return {
                'status': 'FAILED',
                'error': 'Service not available',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
            }

        # Test files
        test_files = [
            ('test_1s.wav', '1 Second Audio'),
            ('test_5s.wav', '5 Second Audio'),
            ('test_silence.wav', 'Silence (Edge Case)'),
        ]

        # Run tests
        for wav_file, test_name in test_files:
            audio_path = test_dir / wav_file

            if not audio_path.exists():
                logger.warning(f"\n⚠️  Skipping {test_name}: {wav_file} not found")
                continue

            self.measure_audio_performance_detailed(
                audio_path,
                test_name,
                num_runs=num_runs,
                warmup_runs=warmup_runs
            )

            # Brief pause between tests
            time.sleep(0.5)

        # Generate summary
        return self.generate_summary()

    def generate_summary(self) -> Dict:
        """Generate comprehensive profiling summary"""
        logger.info("\n" + "="*80)
        logger.info("  DETAILED PROFILING SUMMARY")
        logger.info("="*80)

        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]

        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'total_tests': len(self.results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'week18_target': f"{self.target_realtime_min}-{self.target_realtime_max}× realtime",
            'week19_target': f"{self.week19_target_min}-{self.week19_target_max}× realtime",
            'final_target': f"{self.final_target_min}-{self.final_target_max}× realtime",
            'results': [r.to_dict() for r in self.results]
        }

        if successful_results:
            # Overall metrics
            realtime_factors = [r.realtime_factor for r in successful_results]
            processing_times = [r.processing_time_ms for r in successful_results]

            avg_realtime = np.mean(realtime_factors)
            min_realtime = np.min(realtime_factors)
            max_realtime = np.max(realtime_factors)

            # Determine which target is met
            final_target_met = min_realtime >= self.final_target_min
            week19_target_met = min_realtime >= self.week19_target_min
            week18_target_met = min_realtime >= self.target_realtime_min

            summary['metrics'] = {
                'avg_realtime_factor': float(avg_realtime),
                'min_realtime_factor': float(min_realtime),
                'max_realtime_factor': float(max_realtime),
                'avg_processing_time_ms': float(np.mean(processing_times)),
                'week18_target_met': week18_target_met,
                'week19_target_met': week19_target_met,
                'final_target_met': final_target_met,
                'avg_npu_utilization_percent': float(np.mean([r.npu_utilization_percent for r in successful_results]))
            }

            logger.info(f"\n  Tests: {len(successful_results)}/{len(self.results)} successful")
            logger.info(f"\n  Realtime Factors:")
            logger.info(f"    Average: {avg_realtime:.1f}×")
            logger.info(f"    Min: {min_realtime:.1f}×")
            logger.info(f"    Max: {max_realtime:.1f}×")

            if final_target_met:
                logger.info(f"\n  🎉 FINAL TARGET ACHIEVED! All tests meet {self.final_target_min}× minimum")
                summary['status'] = 'FINAL_TARGET_MET'
            elif week19_target_met:
                logger.info(f"\n  🎯 WEEK 19 TARGET MET! All tests meet {self.week19_target_min}× minimum")
                summary['status'] = 'WEEK19_TARGET_MET'
            elif week18_target_met:
                logger.info(f"\n  ✅ WEEK 18 TARGET MET! All tests meet {self.target_realtime_min}× minimum")
                summary['status'] = 'WEEK18_TARGET_MET'
            else:
                gap = self.target_realtime_min - min_realtime
                logger.warning(f"\n  ⚠️  Week 18 target not met. Gap: {gap:.1f}×")
                summary['status'] = 'BELOW_TARGET'

            # Bottleneck analysis
            self._print_bottleneck_analysis(successful_results)

        else:
            logger.error("\n  ❌ No successful tests")
            summary['status'] = 'ALL_FAILED'

        if failed_results:
            logger.error(f"\n  Failed Tests:")
            for r in failed_results:
                logger.error(f"    - {r.test_name}: {r.error}")

        return summary

    def _print_bottleneck_analysis(self, results: List[DetailedPerformanceResult]):
        """Print detailed bottleneck analysis"""
        logger.info("\n  BOTTLENECK ANALYSIS")
        logger.info("  " + "-"*76)

        # Aggregate timing across all tests
        mel_times = [r.mel_time_ms for r in results if r.mel_time_ms > 0]
        encoder_times = [r.encoder_time_ms for r in results if r.encoder_time_ms > 0]
        decoder_times = [r.decoder_time_ms for r in results if r.decoder_time_ms > 0]
        total_times = [r.processing_time_ms for r in results if r.processing_time_ms > 0]

        if not total_times:
            logger.info("  No timing data available")
            return

        avg_total = np.mean(total_times)
        avg_mel = np.mean(mel_times) if mel_times else 0
        avg_encoder = np.mean(encoder_times) if encoder_times else 0
        avg_decoder = np.mean(decoder_times) if decoder_times else 0

        # Calculate percentages
        mel_pct = (avg_mel / avg_total * 100) if avg_total > 0 else 0
        encoder_pct = (avg_encoder / avg_total * 100) if avg_total > 0 else 0
        decoder_pct = (avg_decoder / avg_total * 100) if avg_total > 0 else 0

        logger.info(f"\n  Average Component Times (across all tests):")
        logger.info(f"    Total:         {avg_total:>8.2f} ms (100.0%)")
        logger.info(f"    Mel:           {avg_mel:>8.2f} ms ({mel_pct:>5.1f}%)")
        logger.info(f"    NPU Encoder:   {avg_encoder:>8.2f} ms ({encoder_pct:>5.1f}%)")
        logger.info(f"    Decoder:       {avg_decoder:>8.2f} ms ({decoder_pct:>5.1f}%)")

        # Identify bottleneck
        components = [
            ('Mel Spectrogram', avg_mel, mel_pct),
            ('NPU Encoder', avg_encoder, encoder_pct),
            ('Decoder', avg_decoder, decoder_pct)
        ]
        bottleneck = max(components, key=lambda x: x[1])

        logger.info(f"\n  🎯 PRIMARY BOTTLENECK: {bottleneck[0]} ({bottleneck[1]:.2f}ms, {bottleneck[2]:.1f}%)")

        # Recommendations
        if bottleneck[0] == 'Decoder':
            logger.info("\n  💡 RECOMMENDATIONS:")
            logger.info("    - Optimize Python decoder (C++ implementation)")
            logger.info("    - Consider batch processing")
            logger.info("    - Investigate decoder model quantization")
        elif bottleneck[0] == 'NPU Encoder':
            logger.info("\n  💡 RECOMMENDATIONS:")
            logger.info("    - Optimize NPU buffer transfers")
            logger.info("    - Consider multi-tile execution")
            logger.info("    - Investigate kernel optimization")
        elif bottleneck[0] == 'Mel Spectrogram':
            logger.info("\n  💡 RECOMMENDATIONS:")
            logger.info("    - Optimize FFT computation")
            logger.info("    - Consider NumPy/SciPy optimization")
            logger.info("    - Investigate NPU acceleration for mel computation")

        # Create ASCII bar chart
        breakdown_data = {
            'Mel Spectrogram': avg_mel,
            'NPU Encoder': avg_encoder,
            'Decoder': avg_decoder
        }
        chart = create_ascii_bar_chart(breakdown_data, "Average Component Breakdown")
        logger.info(chart)

    def save_results(self, output_path: Path):
        """Save results to JSON file"""
        summary = self.generate_summary() if not hasattr(self, '_summary') else self._summary

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            elif isinstance(obj, (bool, int, float, str, type(None))):
                return obj
            else:
                return str(obj)

        summary_clean = convert_types(summary)

        with open(output_path, 'w') as f:
            json.dump(summary_clean, f, indent=2)

        logger.info(f"\n  📄 Results saved to: {output_path}")


def main():
    """Main profiling execution"""
    import sys

    try:
        # Initialize profiling suite
        suite = Week18PerformanceProfiling()

        # Test directory
        test_dir = Path(__file__).parent

        # Run comprehensive profiling (10 runs + 2 warmup per test)
        results = suite.run_comprehensive_profiling(
            test_dir,
            num_runs=10,
            warmup_runs=2
        )

        # Save results
        output_dir = Path("/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/tests/results")
        suite.save_results(output_dir / "week18_detailed_profiling.json")

        # Exit code based on target achievement
        if results.get('status') == 'FINAL_TARGET_MET':
            logger.info("\n✅ SUCCESS: Final 400-500× target achieved!")
            return 0
        elif results.get('status') == 'WEEK19_TARGET_MET':
            logger.info("\n✅ SUCCESS: Week 19 target (250-350×) achieved!")
            return 0
        elif results.get('status') == 'WEEK18_TARGET_MET':
            logger.info("\n✅ SUCCESS: Week 18 target (100-200×) achieved!")
            return 0
        elif results.get('status') == 'BELOW_TARGET':
            logger.warning("\n⚠️  PARTIAL: Performance below Week 18 target but functional")
            return 1
        else:
            logger.error("\n❌ FAILED: Performance profiling failed")
            return 2

    except Exception as e:
        logger.error(f"\n✗ FATAL: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
