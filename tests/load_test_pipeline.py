#!/usr/bin/env python3
"""
Pipeline Load Testing Script

Comprehensive load testing for multi-stream pipeline performance validation:
- Variable concurrency levels (1, 5, 10, 15, 20 concurrent requests)
- Sustained load testing (30-60 seconds)
- Throughput measurement (requests per second)
- Latency distribution (p50, p95, p99)
- Error rate tracking
- Pipeline vs sequential comparison

Usage:
    # Run full load test suite
    python load_test_pipeline.py

    # Test specific concurrency level
    python load_test_pipeline.py --concurrency 10 --duration 30

    # Compare pipeline vs sequential
    python load_test_pipeline.py --compare

    # Quick test (10 seconds)
    python load_test_pipeline.py --quick

Requirements:
    - Service running on localhost:9050
    - Test audio file available
    - ENABLE_PIPELINE=true for pipeline mode

Performance Targets:
    - Sequential: 15.6 req/s baseline
    - Pipeline: 67 req/s (+329% improvement)
    - Individual latency: <70ms
    - Error rate: <1%

Author: CC-1L Multi-Stream Integration Team
Date: November 1, 2025
"""

import asyncio
import aiohttp
import time
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class LoadTestResult:
    """Results from a load test run"""
    concurrency: int
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    throughput_rps: float
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100

    @property
    def mean_latency(self) -> float:
        """Mean latency in seconds"""
        return np.mean(self.latencies) if self.latencies else 0.0

    @property
    def p50_latency(self) -> float:
        """50th percentile latency"""
        return np.percentile(self.latencies, 50) if self.latencies else 0.0

    @property
    def p95_latency(self) -> float:
        """95th percentile latency"""
        return np.percentile(self.latencies, 95) if self.latencies else 0.0

    @property
    def p99_latency(self) -> float:
        """99th percentile latency"""
        return np.percentile(self.latencies, 99) if self.latencies else 0.0

    def print_summary(self):
        """Print formatted summary"""
        print(f"\n{'='*70}")
        print(f"  Load Test Results - {self.concurrency} Concurrent Requests")
        print(f"{'='*70}")
        print(f"  Duration:           {self.duration:.1f}s")
        print(f"  Total Requests:     {self.total_requests}")
        print(f"  Successful:         {self.successful_requests} ({self.success_rate:.1f}%)")
        print(f"  Failed:             {self.failed_requests} ({self.error_rate:.1f}%)")
        print(f"  ")
        print(f"  Throughput:         {self.throughput_rps:.2f} req/s")
        print(f"  ")
        print(f"  Latency Statistics:")
        print(f"    Mean:             {self.mean_latency*1000:.1f}ms")
        print(f"    P50:              {self.p50_latency*1000:.1f}ms")
        print(f"    P95:              {self.p95_latency*1000:.1f}ms")
        print(f"    P99:              {self.p99_latency*1000:.1f}ms")

        if self.errors:
            print(f"  ")
            print(f"  Top Errors:")
            error_counts = {}
            for error in self.errors:
                error_counts[error] = error_counts.get(error, 0) + 1

            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    {error}: {count}x")

        print(f"{'='*70}\n")


class LoadTester:
    """Load testing orchestrator"""

    def __init__(self, base_url: str, test_audio_path: Path):
        """
        Initialize load tester.

        Args:
            base_url: Service base URL (e.g., http://localhost:9050)
            test_audio_path: Path to test audio file
        """
        self.base_url = base_url
        self.test_audio_path = test_audio_path

        if not test_audio_path.exists():
            raise FileNotFoundError(f"Test audio not found: {test_audio_path}")

        # Read audio data once (reused for all requests)
        with open(test_audio_path, "rb") as f:
            self.audio_data = f.read()

        print(f"[LoadTester] Initialized")
        print(f"  Base URL: {base_url}")
        print(f"  Audio file: {test_audio_path}")
        print(f"  Audio size: {len(self.audio_data)} bytes")

    async def send_request(self, session: aiohttp.ClientSession, request_id: int) -> Dict[str, Any]:
        """
        Send a single transcription request.

        Args:
            session: aiohttp session
            request_id: Request identifier

        Returns:
            Result dictionary with success, latency, error
        """
        start = time.perf_counter()

        try:
            # Create form data
            data = aiohttp.FormData()
            data.add_field("file", self.audio_data, filename=f"load_test_{request_id}.wav", content_type="audio/wav")

            # Send request
            async with session.post(
                f"{self.base_url}/v1/audio/transcriptions",
                data=data,
                timeout=aiohttp.ClientTimeout(total=30.0)
            ) as resp:
                latency = time.perf_counter() - start

                if resp.status == 200:
                    result = await resp.json()
                    return {
                        "success": True,
                        "latency": latency,
                        "text_length": len(result.get("text", ""))
                    }
                else:
                    error_text = await resp.text()
                    return {
                        "success": False,
                        "latency": latency,
                        "error": f"HTTP {resp.status}: {error_text[:100]}"
                    }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "latency": time.perf_counter() - start,
                "error": "Request timeout (30s)"
            }

        except Exception as e:
            return {
                "success": False,
                "latency": time.perf_counter() - start,
                "error": f"{type(e).__name__}: {str(e)[:100]}"
            }

    async def run_load_test(self, concurrency: int, duration: float) -> LoadTestResult:
        """
        Run load test with specified concurrency and duration.

        Args:
            concurrency: Number of concurrent requests
            duration: Test duration in seconds

        Returns:
            LoadTestResult with performance metrics
        """
        print(f"\n[LoadTest] Starting load test")
        print(f"  Concurrency: {concurrency}")
        print(f"  Duration: {duration}s")

        successful = 0
        failed = 0
        latencies = []
        errors = []

        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            request_id = 0

            while time.time() - start_time < duration:
                # Launch N concurrent requests
                tasks = [
                    self.send_request(session, request_id + i)
                    for i in range(concurrency)
                ]

                results = await asyncio.gather(*tasks)

                # Process results
                for result in results:
                    if result["success"]:
                        successful += 1
                        latencies.append(result["latency"])
                    else:
                        failed += 1
                        errors.append(result.get("error", "Unknown error"))

                request_id += concurrency

                # Progress update every 5 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                    current_throughput = (successful + failed) / elapsed
                    print(f"  [{elapsed:.0f}s] Throughput: {current_throughput:.2f} req/s, "
                          f"Success: {successful}, Failed: {failed}")

        # Calculate final metrics
        total_time = time.time() - start_time
        total_requests = successful + failed
        throughput = total_requests / total_time if total_time > 0 else 0.0

        return LoadTestResult(
            concurrency=concurrency,
            duration=total_time,
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            throughput_rps=throughput,
            latencies=latencies,
            errors=errors
        )

    async def check_service_mode(self) -> str:
        """
        Check if service is in pipeline or sequential mode.

        Returns:
            "pipeline" or "sequential"
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("mode", "unknown")
        except Exception as e:
            print(f"Warning: Could not determine service mode: {e}")

        return "unknown"


async def run_full_load_test_suite(base_url: str, test_audio_path: Path, duration: float = 30.0):
    """
    Run full load test suite with multiple concurrency levels.

    Args:
        base_url: Service base URL
        test_audio_path: Path to test audio file
        duration: Duration for each test (seconds)
    """
    tester = LoadTester(base_url, test_audio_path)

    # Check service mode
    mode = await tester.check_service_mode()
    print(f"\n{'='*70}")
    print(f"  LOAD TEST SUITE - {mode.upper()} MODE")
    print(f"{'='*70}")

    # Concurrency levels to test
    concurrency_levels = [1, 5, 10, 15, 20]

    results = []

    for concurrency in concurrency_levels:
        result = await tester.run_load_test(concurrency, duration)
        result.print_summary()
        results.append(result)

        # Brief pause between tests
        await asyncio.sleep(2)

    # Print comparison summary
    print(f"\n{'='*70}")
    print(f"  LOAD TEST SUMMARY - {mode.upper()} MODE")
    print(f"{'='*70}")
    print(f"  {'Concurrency':<15} {'Throughput':>15} {'Mean Latency':>15} {'P95 Latency':>15} {'Success Rate':>15}")
    print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")

    for result in results:
        print(f"  {result.concurrency:<15} "
              f"{result.throughput_rps:>14.2f}r/s "
              f"{result.mean_latency*1000:>14.1f}ms "
              f"{result.p95_latency*1000:>14.1f}ms "
              f"{result.success_rate:>14.1f}%")

    print(f"{'='*70}\n")

    # Find best throughput
    best_result = max(results, key=lambda r: r.throughput_rps)
    print(f"  Best Throughput: {best_result.throughput_rps:.2f} req/s at {best_result.concurrency} concurrent requests")

    # Performance targets
    if mode == "pipeline":
        target_throughput = 67.0  # +329% improvement
        print(f"  Target: 67 req/s (+329%)")
        if best_result.throughput_rps >= target_throughput:
            print(f"  ✅ Target achieved! (+{(best_result.throughput_rps/15.6 - 1)*100:.0f}% improvement)")
        else:
            print(f"  ⚠️  Target not met ({best_result.throughput_rps/target_throughput*100:.1f}% of target)")
    else:
        baseline_throughput = 15.6
        print(f"  Baseline: 15.6 req/s (sequential mode)")
        if best_result.throughput_rps >= baseline_throughput:
            print(f"  ✅ Baseline achieved!")
        else:
            print(f"  ⚠️  Below baseline ({best_result.throughput_rps/baseline_throughput*100:.1f}% of baseline)")

    print(f"{'='*70}\n")


async def run_comparison_test(base_url: str, test_audio_path: Path, duration: float = 30.0):
    """
    Compare pipeline vs sequential performance.

    Note: This requires ability to toggle ENABLE_PIPELINE environment variable
    and restart service, which is not automated here.

    Args:
        base_url: Service base URL
        test_audio_path: Path to test audio file
        duration: Duration for each test
    """
    tester = LoadTester(base_url, test_audio_path)

    mode = await tester.check_service_mode()

    print(f"\n{'='*70}")
    print(f"  PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    print(f"  Current mode: {mode}")
    print(f"  ")
    print(f"  To run full comparison:")
    print(f"    1. Test with ENABLE_PIPELINE=false (sequential)")
    print(f"    2. Test with ENABLE_PIPELINE=true (pipeline)")
    print(f"    3. Compare throughput improvement")
    print(f"  ")
    print(f"  Running test in current mode ({mode})...")
    print(f"{'='*70}\n")

    # Run test at optimal concurrency
    concurrency = 15 if mode == "pipeline" else 1

    result = await tester.run_load_test(concurrency, duration)
    result.print_summary()

    # Expected results
    print(f"\n{'='*70}")
    print(f"  Expected Performance (from profiling):")
    print(f"{'='*70}")
    print(f"  Sequential mode (1 concurrent):  15.6 req/s")
    print(f"  Pipeline mode (15 concurrent):   67 req/s (+329%)")
    print(f"  ")
    print(f"  Actual Result:")
    print(f"  {mode.capitalize()} mode ({concurrency} concurrent):  {result.throughput_rps:.2f} req/s")
    print(f"{'='*70}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Pipeline Load Testing")
    parser.add_argument("--url", default="http://localhost:9050", help="Service base URL")
    parser.add_argument("--audio", type=Path, help="Path to test audio file")
    parser.add_argument("--concurrency", type=int, help="Single concurrency level to test")
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration in seconds")
    parser.add_argument("--compare", action="store_true", help="Run comparison test")
    parser.add_argument("--quick", action="store_true", help="Quick test (10 seconds)")

    args = parser.parse_args()

    # Determine test audio path
    if args.audio:
        test_audio_path = args.audio
    else:
        # Default to tests/audio/test_audio.wav
        test_audio_path = Path(__file__).parent / "audio" / "test_audio.wav"

        if not test_audio_path.exists():
            print(f"Error: Test audio not found: {test_audio_path}")
            print(f"Please provide --audio argument or create test audio file")
            sys.exit(1)

    # Adjust duration for quick test
    duration = 10.0 if args.quick else args.duration

    # Run appropriate test
    if args.compare:
        asyncio.run(run_comparison_test(args.url, test_audio_path, duration))
    elif args.concurrency:
        # Single concurrency test
        tester = LoadTester(args.url, test_audio_path)
        result = asyncio.run(tester.run_load_test(args.concurrency, duration))
        result.print_summary()
    else:
        # Full test suite
        asyncio.run(run_full_load_test_suite(args.url, test_audio_path, duration))


if __name__ == "__main__":
    main()
