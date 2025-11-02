#!/usr/bin/env python3
"""
Week 18: Multi-Stream Concurrent Testing
CC-1L Performance Engineering Team

Validates concurrent transcription request handling:
- 4 concurrent streams (baseline)
- 8 concurrent streams (target)
- 16 concurrent streams (stress test)
- Mixed audio lengths (1s, 5s combinations)

Measures:
- Throughput: Total audio seconds processed per wall-clock second
- Latency: Per-request processing time
- NPU utilization: Estimate from timing
- Resource contention: CPU, memory, NPU queueing
- Scalability: How throughput scales with concurrent requests
"""

import asyncio
import aiohttp
import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from profiling_utils import (
    PerformanceProfiler,
    create_ascii_bar_chart,
    TimingStatistics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RequestResult:
    """Single request result"""
    request_id: int
    audio_file: str
    audio_duration_s: float
    processing_time_ms: float
    realtime_factor: float
    success: bool
    error: str = ""
    transcription: str = ""


@dataclass
class MultiStreamResult:
    """Multi-stream test result"""
    test_name: str
    num_streams: int
    total_requests: int
    successful_requests: int
    failed_requests: int

    # Timing metrics
    wall_clock_time_s: float
    total_audio_duration_s: float
    throughput_realtime_factor: float  # total_audio / wall_clock_time

    # Per-request metrics
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float

    # NPU utilization estimate
    estimated_npu_utilization_percent: float

    # Request results
    requests: List[RequestResult]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        d = asdict(self)
        return d


class Week18MultiStreamTest:
    """
    Week 18 Multi-Stream Concurrent Testing Suite

    Tests system behavior under concurrent load to validate:
    1. Multi-stream scalability
    2. Throughput characteristics
    3. Latency distribution
    4. Resource contention
    """

    def __init__(self, service_url: str = "http://127.0.0.1:9050"):
        """Initialize multi-stream test suite"""
        self.service_url = service_url
        self.results: List[MultiStreamResult] = []

        logger.info("="*80)
        logger.info("  WEEK 18: MULTI-STREAM CONCURRENT TESTING")
        logger.info("="*80)
        logger.info(f"  Service URL: {service_url}")
        logger.info("="*80)

    async def check_service_health(self) -> bool:
        """Check if service is running and healthy"""
        try:
            logger.info("\n" + "="*80)
            logger.info("  CHECKING SERVICE HEALTH")
            logger.info("="*80)

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.service_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    response.raise_for_status()
                    health = await response.json()

                    logger.info(f"  Service: {health.get('service', 'Unknown')}")
                    logger.info(f"  Status: {health.get('status', 'Unknown')}")
                    logger.info(f"  NPU Enabled: {health.get('npu_enabled', False)}")

                    if health.get('status') != 'healthy':
                        logger.error(f"  ❌ Service unhealthy: {health.get('status')}")
                        return False

                    logger.info("  ✅ Service is healthy and ready")
                    return True

        except Exception as e:
            logger.error(f"  ❌ Service not responding: {e}")
            return False

    async def transcribe_audio(
        self,
        session: aiohttp.ClientSession,
        audio_path: Path,
        request_id: int
    ) -> RequestResult:
        """
        Transcribe audio file asynchronously

        Args:
            session: aiohttp client session
            audio_path: Path to audio file
            request_id: Request identifier

        Returns:
            RequestResult with timing and transcription
        """
        # Get audio duration
        file_size_bytes = audio_path.stat().st_size
        audio_duration_s = (file_size_bytes - 44) / (2 * 16000)

        result = RequestResult(
            request_id=request_id,
            audio_file=str(audio_path.name),
            audio_duration_s=audio_duration_s,
            processing_time_ms=0.0,
            realtime_factor=0.0,
            success=False
        )

        try:
            # Prepare form data
            with open(audio_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file',
                              f,
                              filename=audio_path.name,
                              content_type='audio/wav')

                # Measure request time
                start_time = time.perf_counter()

                async with session.post(
                    f"{self.service_url}/v1/audio/transcriptions",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    end_time = time.perf_counter()

                    response.raise_for_status()
                    response_data = await response.json()

            # Calculate timing
            processing_time_s = end_time - start_time
            result.processing_time_ms = processing_time_s * 1000

            if audio_duration_s > 0:
                result.realtime_factor = audio_duration_s / processing_time_s

            result.transcription = response_data.get('text', '')
            result.success = True

        except asyncio.TimeoutError:
            result.error = "Request timeout"
            logger.warning(f"  Request {request_id}: Timeout")

        except aiohttp.ClientError as e:
            result.error = f"HTTP error: {str(e)}"
            logger.warning(f"  Request {request_id}: {result.error}")

        except Exception as e:
            result.error = f"Error: {str(e)}"
            logger.warning(f"  Request {request_id}: {result.error}")

        return result

    async def run_concurrent_streams(
        self,
        test_name: str,
        audio_files: List[Path],
        num_streams: int
    ) -> MultiStreamResult:
        """
        Run concurrent stream test

        Args:
            test_name: Test name
            audio_files: List of audio files to transcribe
            num_streams: Number of concurrent streams

        Returns:
            MultiStreamResult with metrics
        """
        logger.info("\n" + "="*80)
        logger.info(f"  TEST: {test_name}")
        logger.info("="*80)
        logger.info(f"  Concurrent streams: {num_streams}")
        logger.info(f"  Total requests: {len(audio_files)}")

        # Start timing
        wall_start = time.perf_counter()

        # Create tasks
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.transcribe_audio(session, audio_path, i)
                for i, audio_path in enumerate(audio_files)
            ]

            # Run with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(num_streams)

            async def bounded_transcribe(task, request_id):
                async with semaphore:
                    return await task

            # Execute with concurrency limit
            bounded_tasks = [
                bounded_transcribe(self.transcribe_audio(session, audio_files[i], i), i)
                for i in range(len(audio_files))
            ]

            results = await asyncio.gather(*bounded_tasks)

        # End timing
        wall_end = time.perf_counter()
        wall_clock_time_s = wall_end - wall_start

        # Analyze results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        total_audio_duration_s = sum(r.audio_duration_s for r in results)
        throughput_realtime = total_audio_duration_s / wall_clock_time_s if wall_clock_time_s > 0 else 0

        # Latency statistics
        latencies = [r.processing_time_ms for r in successful_results]

        if latencies:
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            sorted_latencies = sorted(latencies)
            p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)] if len(sorted_latencies) > 0 else 0
            p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)] if len(sorted_latencies) > 0 else 0
        else:
            avg_latency = median_latency = min_latency = max_latency = p95_latency = p99_latency = 0

        # Estimate NPU utilization
        # Simple heuristic: higher throughput = higher utilization
        # Target is 400-500× realtime with ~2.3% utilization
        estimated_npu_utilization = (throughput_realtime / 400.0) * 2.3 if throughput_realtime > 0 else 0
        estimated_npu_utilization = min(100.0, estimated_npu_utilization)  # Cap at 100%

        result = MultiStreamResult(
            test_name=test_name,
            num_streams=num_streams,
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            wall_clock_time_s=wall_clock_time_s,
            total_audio_duration_s=total_audio_duration_s,
            throughput_realtime_factor=throughput_realtime,
            avg_latency_ms=avg_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            estimated_npu_utilization_percent=estimated_npu_utilization,
            requests=results
        )

        # Log results
        logger.info(f"\n  ✅ Test complete!")
        logger.info(f"  Wall-clock time: {wall_clock_time_s:.2f} s")
        logger.info(f"  Total audio processed: {total_audio_duration_s:.2f} s")
        logger.info(f"  Throughput: {throughput_realtime:.1f}× realtime")
        logger.info(f"  Successful requests: {len(successful_results)}/{len(results)}")

        if failed_results:
            logger.warning(f"  Failed requests: {len(failed_results)}")
            for r in failed_results[:5]:  # Show first 5 failures
                logger.warning(f"    Request {r.request_id}: {r.error}")

        logger.info(f"\n  LATENCY STATISTICS:")
        logger.info(f"    Average:  {avg_latency:.2f} ms")
        logger.info(f"    Median:   {median_latency:.2f} ms")
        logger.info(f"    Min:      {min_latency:.2f} ms")
        logger.info(f"    Max:      {max_latency:.2f} ms")
        logger.info(f"    P95:      {p95_latency:.2f} ms")
        logger.info(f"    P99:      {p99_latency:.2f} ms")

        logger.info(f"\n  NPU UTILIZATION (estimated): {estimated_npu_utilization:.2f}%")

        self.results.append(result)
        return result

    async def run_full_multi_stream_suite(self, test_dir: Path) -> Dict:
        """
        Run full multi-stream test suite

        Tests:
        1. 4 concurrent streams (baseline)
        2. 8 concurrent streams (target)
        3. 16 concurrent streams (stress test)
        4. Mixed audio lengths

        Args:
            test_dir: Directory containing test audio files

        Returns:
            Complete test results
        """
        logger.info("\n" + "="*80)
        logger.info("  RUNNING FULL MULTI-STREAM TEST SUITE")
        logger.info("="*80)

        # Check service health
        if not await self.check_service_health():
            return {
                'status': 'FAILED',
                'error': 'Service not available',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
            }

        # Prepare test audio files
        test_1s = test_dir / "test_1s.wav"
        test_5s = test_dir / "test_5s.wav"

        if not test_1s.exists() or not test_5s.exists():
            logger.error("  ❌ Required test files not found")
            return {
                'status': 'FAILED',
                'error': 'Test files not found',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
            }

        # Test 1: 4 concurrent streams (1s audio)
        audio_files_4 = [test_1s] * 8  # 8 requests, 4 concurrent
        await self.run_concurrent_streams(
            "4 Concurrent Streams (1s audio)",
            audio_files_4,
            num_streams=4
        )

        await asyncio.sleep(1)  # Brief pause

        # Test 2: 8 concurrent streams (1s audio)
        audio_files_8 = [test_1s] * 16  # 16 requests, 8 concurrent
        await self.run_concurrent_streams(
            "8 Concurrent Streams (1s audio)",
            audio_files_8,
            num_streams=8
        )

        await asyncio.sleep(1)  # Brief pause

        # Test 3: 4 concurrent streams (5s audio)
        audio_files_5s = [test_5s] * 8  # 8 requests, 4 concurrent
        await self.run_concurrent_streams(
            "4 Concurrent Streams (5s audio)",
            audio_files_5s,
            num_streams=4
        )

        await asyncio.sleep(1)  # Brief pause

        # Test 4: 16 concurrent streams (stress test, 1s audio)
        audio_files_16 = [test_1s] * 32  # 32 requests, 16 concurrent
        await self.run_concurrent_streams(
            "16 Concurrent Streams (1s audio) - STRESS TEST",
            audio_files_16,
            num_streams=16
        )

        await asyncio.sleep(1)  # Brief pause

        # Test 5: Mixed audio lengths (4 streams)
        audio_files_mixed = [test_1s, test_5s] * 4  # 8 requests, mixed
        await self.run_concurrent_streams(
            "4 Concurrent Streams (mixed 1s/5s)",
            audio_files_mixed,
            num_streams=4
        )

        # Generate summary
        return self.generate_summary()

    def generate_summary(self) -> Dict:
        """Generate comprehensive summary"""
        logger.info("\n" + "="*80)
        logger.info("  MULTI-STREAM TEST SUMMARY")
        logger.info("="*80)

        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'total_tests': len(self.results),
            'results': [r.to_dict() for r in self.results]
        }

        if self.results:
            # Overall metrics
            throughputs = [r.throughput_realtime_factor for r in self.results]
            avg_throughput = statistics.mean(throughputs)
            max_throughput = max(throughputs)

            avg_latencies = [r.avg_latency_ms for r in self.results]
            overall_avg_latency = statistics.mean(avg_latencies)

            summary['metrics'] = {
                'avg_throughput_realtime': float(avg_throughput),
                'max_throughput_realtime': float(max_throughput),
                'overall_avg_latency_ms': float(overall_avg_latency),
                'total_requests': sum(r.total_requests for r in self.results),
                'total_successful': sum(r.successful_requests for r in self.results),
                'total_failed': sum(r.failed_requests for r in self.results)
            }

            logger.info(f"\n  Tests completed: {len(self.results)}")
            logger.info(f"  Total requests: {summary['metrics']['total_requests']}")
            logger.info(f"  Successful: {summary['metrics']['total_successful']}")
            logger.info(f"  Failed: {summary['metrics']['total_failed']}")

            logger.info(f"\n  THROUGHPUT:")
            logger.info(f"    Average: {avg_throughput:.1f}× realtime")
            logger.info(f"    Maximum: {max_throughput:.1f}× realtime")

            logger.info(f"\n  LATENCY:")
            logger.info(f"    Overall average: {overall_avg_latency:.2f} ms")

            # Per-test breakdown
            logger.info(f"\n  PER-TEST RESULTS:")
            logger.info("  " + "-"*76)
            logger.info(f"  {'Test':<40} {'Streams':>8} {'Throughput':>12} {'Avg Latency':>12}")
            logger.info("  " + "-"*76)

            for r in self.results:
                logger.info(f"  {r.test_name:<40} {r.num_streams:>8} {r.throughput_realtime_factor:>11.1f}× {r.avg_latency_ms:>11.2f}ms")

            logger.info("  " + "-"*76)

            # Scalability analysis
            self._print_scalability_analysis()

            summary['status'] = 'COMPLETE'
        else:
            logger.error("\n  ❌ No tests completed")
            summary['status'] = 'NO_TESTS'

        return summary

    def _print_scalability_analysis(self):
        """Print scalability analysis"""
        logger.info("\n  SCALABILITY ANALYSIS")
        logger.info("  " + "-"*76)

        # Group by number of streams
        by_streams = {}
        for r in self.results:
            if r.num_streams not in by_streams:
                by_streams[r.num_streams] = []
            by_streams[r.num_streams].append(r)

        # Calculate average throughput per stream count
        for num_streams in sorted(by_streams.keys()):
            results = by_streams[num_streams]
            avg_throughput = statistics.mean([r.throughput_realtime_factor for r in results])
            avg_latency = statistics.mean([r.avg_latency_ms for r in results])

            logger.info(f"  {num_streams} streams: {avg_throughput:.1f}× throughput, {avg_latency:.2f}ms latency")

        # Efficiency calculation
        logger.info("\n  💡 INSIGHTS:")

        if 4 in by_streams and 8 in by_streams:
            throughput_4 = statistics.mean([r.throughput_realtime_factor for r in by_streams[4]])
            throughput_8 = statistics.mean([r.throughput_realtime_factor for r in by_streams[8]])

            if throughput_4 > 0:
                scaling_efficiency = (throughput_8 / throughput_4) / 2.0  # Should be ~2× for perfect scaling
                logger.info(f"    4→8 stream scaling efficiency: {scaling_efficiency*100:.1f}%")

                if scaling_efficiency > 0.8:
                    logger.info(f"    ✅ Good scalability (>80%)")
                elif scaling_efficiency > 0.6:
                    logger.info(f"    ⚠️  Moderate scalability (60-80%)")
                else:
                    logger.info(f"    ❌ Poor scalability (<60%)")

    def save_results(self, output_path: Path):
        """Save results to JSON file"""
        summary = self.generate_summary() if not hasattr(self, '_summary') else self._summary

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n  📄 Results saved to: {output_path}")


async def main_async():
    """Main async execution"""
    try:
        # Initialize test suite
        suite = Week18MultiStreamTest()

        # Test directory
        test_dir = Path(__file__).parent

        # Run full multi-stream test suite
        results = await suite.run_full_multi_stream_suite(test_dir)

        # Save results
        output_dir = Path("/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/tests/results")
        suite.save_results(output_dir / "week18_multi_stream_results.json")

        # Exit code
        if results.get('status') == 'COMPLETE':
            logger.info("\n✅ SUCCESS: Multi-stream testing complete!")
            return 0
        else:
            logger.error("\n❌ FAILED: Multi-stream testing failed")
            return 2

    except Exception as e:
        logger.error(f"\n✗ FATAL: {e}", exc_info=True)
        return 1


def main():
    """Main entry point"""
    return asyncio.run(main_async())


if __name__ == "__main__":
    import sys
    sys.exit(main())
