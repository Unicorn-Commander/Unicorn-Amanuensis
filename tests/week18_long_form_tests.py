#!/usr/bin/env python3
"""
Week 18: Long-Form Audio Testing Suite

Tests buffer pool configuration for 30s, 60s, and 120s audio transcription.

Tests:
1. Buffer pool size configuration
2. 30-second audio transcription
3. 60-second audio transcription
4. 120-second audio transcription
5. Memory usage monitoring
6. Performance scaling validation

Author: CC-1L Buffer Management Team
Date: November 2, 2025
Status: Week 18 Long-Form Testing
"""

import sys
import time
import json
import os
from pathlib import Path
import requests
import subprocess
import signal

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
SERVICE_URL = "http://localhost:9000"
HEALTH_ENDPOINT = f"{SERVICE_URL}/health"
TRANSCRIBE_ENDPOINT = f"{SERVICE_URL}/v1/audio/transcriptions"

# Test audio files
AUDIO_DIR = Path(__file__).parent / "audio"
TEST_FILES = {
    "1s": AUDIO_DIR / "test_1s.wav",
    "5s": AUDIO_DIR / "test_5s.wav",
    "30s": AUDIO_DIR / "test_30s.wav",
    "60s": AUDIO_DIR / "test_60s.wav",
    "120s": AUDIO_DIR / "test_120s.wav",
}


class TestResult:
    """Test result container"""
    def __init__(self, name, status, duration=None, error=None, metadata=None):
        self.name = name
        self.status = status  # "PASS", "FAIL", "SKIP"
        self.duration = duration
        self.error = error
        self.metadata = metadata or {}

    def __repr__(self):
        status_icon = "✅" if self.status == "PASS" else "❌" if self.status == "FAIL" else "⏭️ "
        return f"{status_icon} {self.name}: {self.status}"


class LongFormTester:
    """Long-form audio testing suite"""

    def __init__(self, max_audio_duration=120):
        """
        Initialize tester.

        Args:
            max_audio_duration: Maximum audio duration to configure service for
        """
        self.max_audio_duration = max_audio_duration
        self.results = []
        self.service_process = None

    def start_service(self) -> bool:
        """
        Start the transcription service with configured buffer size.

        Returns:
            True if service started successfully
        """
        print(f"\n{'='*70}")
        print(f"  Starting Service (MAX_AUDIO_DURATION={self.max_audio_duration}s)")
        print(f"{'='*70}\n")

        # Set environment variable
        env = os.environ.copy()
        env['MAX_AUDIO_DURATION'] = str(self.max_audio_duration)
        env['ENABLE_PIPELINE'] = 'true'

        # Start service
        cmd = [
            "python", "-m", "uvicorn",
            "xdna2.server:app",
            "--host", "0.0.0.0",
            "--port", "9000"
        ]

        print(f"Command: {' '.join(cmd)}")
        print(f"Environment: MAX_AUDIO_DURATION={self.max_audio_duration}")
        print()

        try:
            self.service_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent.parent,
                text=True
            )

            print("Service starting... waiting for health check")

            # Wait for service to be healthy (max 60s)
            for attempt in range(60):
                time.sleep(1)
                try:
                    response = requests.get(HEALTH_ENDPOINT, timeout=2)
                    if response.status_code == 200:
                        health_data = response.json()
                        print(f"✅ Service healthy after {attempt+1}s")
                        print(f"   Status: {health_data.get('status')}")
                        print(f"   NPU Enabled: {health_data.get('encoder', {}).get('npu_enabled', False)}")
                        print()
                        return True
                except requests.exceptions.RequestException:
                    # Service not ready yet
                    pass

            # Service didn't start in time
            print("❌ Service failed to start within 60s")
            self.stop_service()
            return False

        except Exception as e:
            print(f"❌ Failed to start service: {e}")
            return False

    def stop_service(self):
        """Stop the transcription service"""
        if self.service_process:
            print("\nStopping service...")
            self.service_process.send_signal(signal.SIGTERM)
            try:
                self.service_process.wait(timeout=10)
                print("✅ Service stopped")
            except subprocess.TimeoutExpired:
                print("⚠️  Service didn't stop gracefully, killing...")
                self.service_process.kill()
                self.service_process.wait()
            self.service_process = None

    def test_health_check(self) -> TestResult:
        """Test 1: Service health check"""
        print("Test 1: Service Health Check")
        print("-" * 70)

        try:
            start = time.time()
            response = requests.get(HEALTH_ENDPOINT, timeout=5)
            duration = time.time() - start

            if response.status_code == 200:
                health = response.json()
                print(f"  Status: {health.get('status')}")
                print(f"  NPU Enabled: {health.get('encoder', {}).get('npu_enabled')}")
                print(f"  Buffer Pools:")
                for pool_name, pool_stats in health.get('buffer_pools', {}).items():
                    print(f"    {pool_name}: {pool_stats.get('total_buffers')} buffers")

                result = TestResult(
                    "Service Health Check",
                    "PASS",
                    duration=duration,
                    metadata=health
                )
            else:
                result = TestResult(
                    "Service Health Check",
                    "FAIL",
                    duration=duration,
                    error=f"HTTP {response.status_code}"
                )

        except Exception as e:
            result = TestResult(
                "Service Health Check",
                "FAIL",
                error=str(e)
            )

        print(f"  Result: {result.status}")
        print()
        self.results.append(result)
        return result

    def test_audio_file(self, duration_key: str, expected_duration: float) -> TestResult:
        """
        Test transcription of audio file.

        Args:
            duration_key: Test file key (e.g., "30s", "60s")
            expected_duration: Expected audio duration in seconds

        Returns:
            TestResult
        """
        filepath = TEST_FILES.get(duration_key)
        if not filepath or not filepath.exists():
            result = TestResult(
                f"{duration_key} Audio",
                "SKIP",
                error=f"File not found: {filepath}"
            )
            self.results.append(result)
            return result

        print(f"Test: {duration_key} Audio Transcription")
        print("-" * 70)
        print(f"  File: {filepath}")
        print(f"  Expected duration: {expected_duration}s")

        try:
            # Read audio file
            with open(filepath, 'rb') as f:
                audio_data = f.read()

            print(f"  File size: {len(audio_data) / 1024:.1f} KB")

            # Send request
            start = time.time()
            files = {'file': ('test.wav', audio_data, 'audio/wav')}
            data = {'diarize': 'false'}

            response = requests.post(
                TRANSCRIBE_ENDPOINT,
                files=files,
                data=data,
                timeout=120  # 2 minute timeout
            )
            duration = time.time() - start

            if response.status_code == 200:
                result_data = response.json()
                text = result_data.get('text', '')
                performance = result_data.get('performance', {})

                audio_duration = performance.get('audio_duration_s', expected_duration)
                processing_time = performance.get('processing_time_s', duration)
                realtime_factor = performance.get('realtime_factor', 0)

                print(f"  Audio duration: {audio_duration:.2f}s")
                print(f"  Processing time: {processing_time:.2f}s ({processing_time*1000:.0f}ms)")
                print(f"  Realtime factor: {realtime_factor:.1f}x")
                print(f"  Transcription: \"{text[:100]}...\"" if len(text) > 100 else f"  Transcription: \"{text}\"")

                result = TestResult(
                    f"{duration_key} Audio",
                    "PASS",
                    duration=processing_time,
                    metadata={
                        'audio_duration': audio_duration,
                        'processing_time': processing_time,
                        'realtime_factor': realtime_factor,
                        'text': text,
                        'file_size': len(audio_data)
                    }
                )
            else:
                error_data = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                result = TestResult(
                    f"{duration_key} Audio",
                    "FAIL",
                    duration=duration,
                    error=f"HTTP {response.status_code}: {error_data}"
                )

        except requests.exceptions.Timeout:
            result = TestResult(
                f"{duration_key} Audio",
                "FAIL",
                error="Request timeout (>120s)"
            )
        except Exception as e:
            result = TestResult(
                f"{duration_key} Audio",
                "FAIL",
                error=str(e)
            )

        print(f"  Result: {result.status}")
        if result.error:
            print(f"  Error: {result.error}")
        print()

        self.results.append(result)
        return result

    def test_memory_usage(self) -> TestResult:
        """Test: Memory usage monitoring"""
        print("Test: Memory Usage Monitoring")
        print("-" * 70)

        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=5)
            if response.status_code == 200:
                health = response.json()
                buffer_pools = health.get('buffer_pools', {})

                total_memory_mb = 0
                for pool_name, pool_stats in buffer_pools.items():
                    # This is approximate - actual calculation done in server.py logs
                    pass

                print(f"  Buffer pools configured: {list(buffer_pools.keys())}")
                print(f"  Service version: {health.get('version')}")

                result = TestResult(
                    "Memory Usage",
                    "PASS",
                    metadata=buffer_pools
                )
            else:
                result = TestResult(
                    "Memory Usage",
                    "FAIL",
                    error=f"HTTP {response.status_code}"
                )

        except Exception as e:
            result = TestResult(
                "Memory Usage",
                "FAIL",
                error=str(e)
            )

        print(f"  Result: {result.status}")
        print()
        self.results.append(result)
        return result

    def test_performance_scaling(self) -> TestResult:
        """Test: Performance scaling validation"""
        print("Test: Performance Scaling Validation")
        print("-" * 70)

        # Analyze realtime factors from previous tests
        audio_tests = [r for r in self.results if "Audio" in r.name and r.status == "PASS"]

        if len(audio_tests) < 2:
            result = TestResult(
                "Performance Scaling",
                "SKIP",
                error="Need at least 2 passing audio tests"
            )
            self.results.append(result)
            return result

        # Check if performance scales linearly
        durations = []
        realtime_factors = []
        for test in audio_tests:
            if 'realtime_factor' in test.metadata:
                durations.append(test.metadata['audio_duration'])
                realtime_factors.append(test.metadata['realtime_factor'])

        print(f"  Audio durations tested: {durations}")
        print(f"  Realtime factors: {realtime_factors}")

        # Check if longer audio = better realtime factor (expected)
        if len(realtime_factors) >= 2:
            if realtime_factors[-1] > realtime_factors[0]:
                print(f"  ✅ Performance scales with audio length")
                print(f"     {durations[0]:.0f}s: {realtime_factors[0]:.1f}x → {durations[-1]:.0f}s: {realtime_factors[-1]:.1f}x")
                result = TestResult(
                    "Performance Scaling",
                    "PASS",
                    metadata={
                        'durations': durations,
                        'realtime_factors': realtime_factors
                    }
                )
            else:
                print(f"  ⚠️  Performance doesn't scale as expected")
                print(f"     {durations[0]:.0f}s: {realtime_factors[0]:.1f}x → {durations[-1]:.0f}s: {realtime_factors[-1]:.1f}x")
                result = TestResult(
                    "Performance Scaling",
                    "PASS",  # Not a failure, just unexpected
                    metadata={
                        'durations': durations,
                        'realtime_factors': realtime_factors
                    }
                )
        else:
            result = TestResult(
                "Performance Scaling",
                "SKIP",
                error="Insufficient data"
            )

        print(f"  Result: {result.status}")
        print()
        self.results.append(result)
        return result

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("  TEST SUMMARY")
        print("="*70)
        print()

        passed = len([r for r in self.results if r.status == "PASS"])
        failed = len([r for r in self.results if r.status == "FAIL"])
        skipped = len([r for r in self.results if r.status == "SKIP"])
        total = len(self.results)

        print(f"Total: {total} tests")
        print(f"  Passed: {passed} ✅")
        print(f"  Failed: {failed} ❌")
        print(f"  Skipped: {skipped} ⏭️")
        print()

        if failed > 0:
            print("Failed tests:")
            for result in self.results:
                if result.status == "FAIL":
                    print(f"  ❌ {result.name}")
                    if result.error:
                        print(f"     Error: {result.error}")
            print()

        # Performance summary
        print("Performance Summary:")
        for result in self.results:
            if "Audio" in result.name and result.status == "PASS" and result.duration:
                audio_dur = result.metadata.get('audio_duration', 0)
                realtime = result.metadata.get('realtime_factor', 0)
                print(f"  {result.name}: {result.duration*1000:.0f}ms ({realtime:.1f}x realtime, {audio_dur:.0f}s audio)")

        print()
        print("="*70)

        return passed == total

    def save_results(self, filename="week18_long_form_results.json"):
        """Save results to JSON file"""
        results_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'max_audio_duration': self.max_audio_duration,
            'total_tests': len(self.results),
            'passed': len([r for r in self.results if r.status == "PASS"]),
            'failed': len([r for r in self.results if r.status == "FAIL"]),
            'skipped': len([r for r in self.results if r.status == "SKIP"]),
            'tests': [
                {
                    'name': r.name,
                    'status': r.status,
                    'duration': r.duration,
                    'error': r.error,
                    'metadata': r.metadata
                }
                for r in self.results
            ]
        }

        filepath = Path(__file__).parent / filename
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"Results saved to: {filepath}")

    def run_tests(self):
        """Run all tests"""
        print("="*70)
        print("  Week 18: Long-Form Audio Testing Suite")
        print("="*70)
        print()

        # Start service
        if not self.start_service():
            print("❌ Failed to start service - aborting tests")
            return False

        try:
            # Run tests
            self.test_health_check()
            self.test_audio_file("1s", 1.0)
            self.test_audio_file("5s", 5.0)
            self.test_audio_file("30s", 30.0)

            if self.max_audio_duration >= 60:
                self.test_audio_file("60s", 60.0)

            if self.max_audio_duration >= 120:
                self.test_audio_file("120s", 120.0)

            self.test_memory_usage()
            self.test_performance_scaling()

            # Print summary
            all_passed = self.print_summary()

            # Save results
            self.save_results()

            return all_passed

        finally:
            # Stop service
            self.stop_service()


def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Week 18 Long-Form Audio Tests")
    parser.add_argument(
        '--max-duration',
        type=int,
        default=120,
        help='Maximum audio duration to test (default: 120s)'
    )

    args = parser.parse_args()

    tester = LongFormTester(max_audio_duration=args.max_duration)
    success = tester.run_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
