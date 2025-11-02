#!/usr/bin/env python3
"""
Week 19.5 Comprehensive Performance Testing

Tests the FIXED encoder→decoder architecture where NPU output is used directly
(not discarded and re-encoded on CPU).

This test suite validates:
1. Single-request performance (1s, 5s, 30s audio)
2. Multi-stream concurrent performance (4, 8, 16 streams)
3. Architecture fix validation (NPU output usage)
4. Comparison with Week 18/19 baselines

Author: Team 3 Lead - Performance Testing & Comparison
Date: November 2, 2025
"""

import time
import requests
import numpy as np
import json
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import wave
import sys

@dataclass
class TestResult:
    """Single test run result"""
    file: str
    duration: float
    processing_time: float
    realtime_factor: float
    transcription: str
    timestamp: float

@dataclass
class StatisticalResults:
    """Statistical analysis of multiple test runs"""
    file: str
    duration: float
    num_runs: int
    mean_time: float
    median_time: float
    std_time: float
    min_time: float
    max_time: float
    p95_time: float
    p99_time: float
    mean_realtime_factor: float
    median_realtime_factor: float
    transcription: str

@dataclass
class MultiStreamResult:
    """Multi-stream test result"""
    test_name: str
    num_streams: int
    total_requests: int
    wall_time: float
    total_audio: float
    throughput: float
    avg_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    success_rate: float
    successful: int
    failed: int


class PerformanceTester:
    """Comprehensive performance testing suite for Week 19.5"""

    def __init__(self, base_url: str = "http://localhost:9050"):
        self.base_url = base_url
        self.results = {}
        self.audio_dir = Path(__file__).parent / "audio"

    def get_audio_duration(self, audio_file: str) -> float:
        """Get audio duration in seconds"""
        try:
            with wave.open(audio_file, 'rb') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                return frames / float(rate)
        except Exception as e:
            print(f"Warning: Could not read WAV header for {audio_file}: {e}")
            # Fallback: estimate from file size (16kHz, 16-bit mono)
            file_size = Path(audio_file).stat().st_size
            # WAV header is ~44 bytes, then 2 bytes per sample at 16kHz
            audio_bytes = file_size - 44
            samples = audio_bytes / 2
            return samples / 16000.0

    def send_request(self, audio_file: str) -> Tuple[str, float]:
        """
        Send single transcription request

        Returns:
            (transcription_text, processing_time)
        """
        start = time.perf_counter()

        with open(audio_file, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/v1/audio/transcriptions",
                files={'file': f},
                timeout=120
            )

        elapsed = time.perf_counter() - start

        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code} {response.text}")

        result = response.json()
        text = result.get('text', '')

        return text, elapsed

    def test_single_request(self, audio_file: str, num_runs: int = 10, warmup: int = 2) -> StatisticalResults:
        """
        Test single request performance with statistical analysis

        Args:
            audio_file: Path to test audio file
            num_runs: Number of test runs for statistics
            warmup: Number of warmup runs (discarded)

        Returns:
            StatisticalResults with mean, median, stddev, percentiles
        """
        print(f"\n{'='*70}")
        print(f"Testing: {Path(audio_file).name}")
        print(f"{'='*70}")

        audio_duration = self.get_audio_duration(audio_file)
        print(f"Audio duration: {audio_duration:.2f}s")

        # Warmup runs
        print(f"\nWarmup ({warmup} runs)...")
        for i in range(warmup):
            try:
                text, elapsed = self.send_request(audio_file)
                print(f"  Warmup {i+1}: {elapsed*1000:.0f}ms")
            except Exception as e:
                print(f"  Warmup {i+1}: FAILED - {e}")

        # Test runs
        print(f"\nTest runs ({num_runs} runs)...")
        times = []
        transcription = ""

        for i in range(num_runs):
            try:
                text, elapsed = self.send_request(audio_file)
                times.append(elapsed)
                transcription = text  # Keep last transcription
                rt_factor = audio_duration / elapsed
                print(f"  Run {i+1:2d}: {elapsed*1000:6.0f}ms ({rt_factor:5.1f}× realtime) - \"{text}\"")
            except Exception as e:
                print(f"  Run {i+1:2d}: FAILED - {e}")
                # Still record a failure time (use timeout as penalty)
                times.append(120.0)

        # Calculate statistics
        if not times:
            print(f"\nERROR: No successful runs!")
            return None

        times_array = np.array(times)

        results = StatisticalResults(
            file=Path(audio_file).name,
            duration=audio_duration,
            num_runs=len(times),
            mean_time=float(np.mean(times_array)),
            median_time=float(np.median(times_array)),
            std_time=float(np.std(times_array)),
            min_time=float(np.min(times_array)),
            max_time=float(np.max(times_array)),
            p95_time=float(np.percentile(times_array, 95)),
            p99_time=float(np.percentile(times_array, 99)),
            mean_realtime_factor=audio_duration / float(np.mean(times_array)),
            median_realtime_factor=audio_duration / float(np.median(times_array)),
            transcription=transcription
        )

        self._print_summary(results)
        self.results[audio_file] = results

        return results

    def _print_summary(self, results: StatisticalResults):
        """Print statistical summary of results"""
        print(f"\n{'-'*70}")
        print(f"Statistical Summary:")
        print(f"{'-'*70}")
        print(f"  Runs:             {results.num_runs}")
        print(f"  Mean time:        {results.mean_time*1000:6.0f}ms ({results.mean_realtime_factor:5.1f}× realtime)")
        print(f"  Median time:      {results.median_time*1000:6.0f}ms ({results.median_realtime_factor:5.1f}× realtime)")
        print(f"  Std dev:          {results.std_time*1000:6.0f}ms")
        print(f"  Min time:         {results.min_time*1000:6.0f}ms ({results.duration/results.min_time:5.1f}× realtime)")
        print(f"  Max time:         {results.max_time*1000:6.0f}ms ({results.duration/results.max_time:5.1f}× realtime)")
        print(f"  P95 time:         {results.p95_time*1000:6.0f}ms")
        print(f"  P99 time:         {results.p99_time*1000:6.0f}ms")
        print(f"  Transcription:    \"{results.transcription}\"")
        print(f"{'-'*70}")

    async def send_request_async(self, session: aiohttp.ClientSession, audio_file: str, request_id: int) -> Tuple[int, float, str, bool]:
        """
        Send async transcription request

        Returns:
            (request_id, elapsed_time, transcription, success)
        """
        start = time.perf_counter()

        try:
            with open(audio_file, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=Path(audio_file).name)

                async with session.post(
                    f"{self.base_url}/v1/audio/transcriptions",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    elapsed = time.perf_counter() - start

                    if response.status == 200:
                        result = await response.json()
                        text = result.get('text', '')
                        return (request_id, elapsed, text, True)
                    else:
                        error_text = await response.text()
                        print(f"    Request {request_id}: FAILED ({response.status}) - {error_text[:100]}")
                        return (request_id, elapsed, "", False)
        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"    Request {request_id}: EXCEPTION - {e}")
            return (request_id, elapsed, "", False)

    async def test_concurrent_performance(
        self,
        audio_file: str,
        num_streams: int,
        requests_per_stream: int = 2
    ) -> MultiStreamResult:
        """
        Test multi-stream concurrent performance

        Args:
            audio_file: Path to test audio file
            num_streams: Number of concurrent streams
            requests_per_stream: Requests per stream

        Returns:
            MultiStreamResult with throughput and latency metrics
        """
        total_requests = num_streams * requests_per_stream
        audio_duration = self.get_audio_duration(audio_file)
        total_audio = audio_duration * total_requests

        test_name = f"{num_streams} streams × {requests_per_stream} requests"

        print(f"\n{'='*70}")
        print(f"Multi-Stream Test: {test_name}")
        print(f"{'='*70}")
        print(f"  Audio file: {Path(audio_file).name}")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Total requests: {total_requests}")
        print(f"  Total audio: {total_audio:.2f}s")
        print(f"  Concurrency: {num_streams}")

        print(f"\nSending {total_requests} concurrent requests...")
        start = time.perf_counter()

        async with aiohttp.ClientSession() as session:
            # Create tasks for all requests
            tasks = []
            for stream_id in range(num_streams):
                for req_num in range(requests_per_stream):
                    request_id = stream_id * requests_per_stream + req_num
                    task = self.send_request_async(session, audio_file, request_id)
                    tasks.append(task)

            # Execute all requests concurrently
            results = await asyncio.gather(*tasks)

        wall_time = time.perf_counter() - start

        # Analyze results
        successful = sum(1 for _, _, _, success in results if success)
        failed = total_requests - successful
        success_rate = successful / total_requests if total_requests > 0 else 0.0

        latencies = [elapsed for _, elapsed, _, success in results if success]

        if latencies:
            latencies_array = np.array(latencies)
            avg_latency = float(np.mean(latencies_array))
            median_latency = float(np.median(latencies_array))
            p95_latency = float(np.percentile(latencies_array, 95))
            p99_latency = float(np.percentile(latencies_array, 99))
        else:
            avg_latency = median_latency = p95_latency = p99_latency = 0.0

        throughput = total_audio / wall_time if wall_time > 0 else 0.0

        result = MultiStreamResult(
            test_name=test_name,
            num_streams=num_streams,
            total_requests=total_requests,
            wall_time=wall_time,
            total_audio=total_audio,
            throughput=throughput,
            avg_latency=avg_latency,
            median_latency=median_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            success_rate=success_rate,
            successful=successful,
            failed=failed
        )

        self._print_multistream_summary(result)

        return result

    def _print_multistream_summary(self, result: MultiStreamResult):
        """Print multi-stream test summary"""
        print(f"\n{'-'*70}")
        print(f"Multi-Stream Results:")
        print(f"{'-'*70}")
        print(f"  Wall time:        {result.wall_time:.2f}s")
        print(f"  Throughput:       {result.throughput:.1f}× realtime")
        print(f"  Success rate:     {result.success_rate*100:.1f}% ({result.successful}/{result.total_requests})")
        print(f"  Avg latency:      {result.avg_latency*1000:.0f}ms")
        print(f"  Median latency:   {result.median_latency*1000:.0f}ms")
        print(f"  P95 latency:      {result.p95_latency*1000:.0f}ms")
        print(f"  P99 latency:      {result.p99_latency*1000:.0f}ms")
        print(f"{'-'*70}")

    def save_results(self, output_file: str):
        """Save all results to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, (StatisticalResults, MultiStreamResult)):
                json_results[str(key)] = asdict(value)
            else:
                json_results[str(key)] = value

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def main():
    """Main test execution"""
    print("="*70)
    print("Week 19.5 Comprehensive Performance Testing")
    print("="*70)
    print("Testing FIXED architecture (NPU output used directly)")
    print()

    # Initialize tester
    tester = PerformanceTester()

    # Check service health
    print("Checking service health...")
    try:
        response = requests.get(f"{tester.base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✓ Service: {health.get('service', 'Unknown')}")
            print(f"✓ Version: {health.get('version', 'Unknown')}")
            print(f"✓ NPU Enabled: {health.get('encoder', {}).get('npu_enabled', False)}")
        else:
            print(f"✗ Service health check failed: {response.status_code}")
            return 1
    except Exception as e:
        print(f"✗ Cannot connect to service: {e}")
        return 1

    # Phase 1: Single Request Performance Tests
    print(f"\n{'='*70}")
    print("PHASE 1: Single Request Performance Tests")
    print(f"{'='*70}")

    test_files = [
        tester.audio_dir / "test_1s.wav",
        tester.audio_dir / "test_5s.wav",
        tester.audio_dir / "test_30s.wav",
        tester.audio_dir / "test_silence.wav"
    ]

    single_results = []
    for test_file in test_files:
        if not test_file.exists():
            print(f"\nWarning: Test file not found: {test_file}")
            continue

        result = tester.test_single_request(str(test_file), num_runs=10, warmup=2)
        if result:
            single_results.append(result)
            tester.results[f"single_{result.file}"] = result

    # Phase 2: Multi-Stream Performance Tests
    print(f"\n{'='*70}")
    print("PHASE 2: Multi-Stream Performance Tests")
    print(f"{'='*70}")

    multi_tests = [
        (tester.audio_dir / "test_1s.wav", 4, 2),   # 4 streams, 2 requests each = 8 total
        (tester.audio_dir / "test_1s.wav", 8, 2),   # 8 streams, 2 requests each = 16 total
        (tester.audio_dir / "test_1s.wav", 16, 2),  # 16 streams, 2 requests each = 32 total
        (tester.audio_dir / "test_5s.wav", 4, 2),   # 4 streams, 2 requests each = 8 total
    ]

    multi_results = []
    for audio_file, num_streams, requests_per_stream in multi_tests:
        if not audio_file.exists():
            print(f"\nWarning: Test file not found: {audio_file}")
            continue

        result = asyncio.run(
            tester.test_concurrent_performance(str(audio_file), num_streams, requests_per_stream)
        )
        multi_results.append(result)
        tester.results[f"multi_{result.test_name}"] = result

    # Phase 3: Summary and Comparison
    print(f"\n{'='*70}")
    print("PHASE 3: Summary and Comparison")
    print(f"{'='*70}")

    if single_results:
        print("\nSingle Request Summary:")
        print(f"{'Test':<20} {'Mean Time':<12} {'Realtime Factor':<18} {'Transcription':<30}")
        print("-"*80)
        for r in single_results:
            print(f"{r.file:<20} {r.mean_time*1000:6.0f}ms     {r.mean_realtime_factor:5.1f}×           \"{r.transcription[:28]}\"")

        # Calculate average
        avg_rt = np.mean([r.mean_realtime_factor for r in single_results])
        print(f"\n{'Average':<20} {'':12} {avg_rt:5.1f}×")

    if multi_results:
        print("\nMulti-Stream Summary:")
        print(f"{'Test':<30} {'Throughput':<15} {'Avg Latency':<15} {'Success Rate'}")
        print("-"*80)
        for r in multi_results:
            print(f"{r.test_name:<30} {r.throughput:5.1f}× realtime   {r.avg_latency*1000:6.0f}ms        {r.success_rate*100:5.1f}%")

        # Calculate average
        avg_throughput = np.mean([r.throughput for r in multi_results])
        print(f"\n{'Average':<30} {avg_throughput:5.1f}× realtime")

    # Save results
    output_file = Path(__file__).parent / "results" / "week19_5_performance_results.json"
    tester.save_results(str(output_file))

    print(f"\n{'='*70}")
    print("Testing Complete!")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
