#!/usr/bin/env python3
"""
Week 19.5 Pipeline Validation Tests
Validates the fixed encoder→decoder architecture

This test suite validates that:
1. NPU encoder output is USED (not discarded)
2. NO CPU re-encoding happens
3. Accuracy is maintained (>95% vs baseline)
4. Performance meets targets (>25× realtime)
5. Pipeline handles concurrent requests

Author: Team 2 Lead - Pipeline Optimization & Validation
Date: November 2, 2025
Status: Ready for Team 1's fix
"""

import sys
import time
import json
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Test configuration
SERVICE_URL = "http://localhost:9050"
TRANSCRIPTION_ENDPOINT = f"{SERVICE_URL}/v1/audio/transcriptions"
HEALTH_ENDPOINT = f"{SERVICE_URL}/health"
STATS_ENDPOINT = f"{SERVICE_URL}/stats"

# Test audio files (from Week 17 testing)
TEST_AUDIO_DIR = Path(__file__).parent.parent / "tests" / "audio"
TEST_FILES = {
    "test_1s.wav": {
        "duration": 1.0,
        "baseline_text": " Ooh.",
        "baseline_time_ms": 328,  # Week 18 baseline
        "target_time_ms": 100,    # Week 19.5 target
    },
    "test_5s.wav": {
        "duration": 5.0,
        "baseline_text": " Whoa! Whoa! Whoa! Whoa!",
        "baseline_time_ms": 495,  # Week 18 baseline
        "target_time_ms": 200,    # Week 19.5 target
    }
}


@dataclass
class TestResult:
    """Result from a single test"""
    name: str
    passed: bool
    message: str
    details: Dict[str, Any]
    duration_ms: float


class Week195Validator:
    """Validation suite for Week 19.5 pipeline fix"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.service_available = False

    def check_service_health(self) -> bool:
        """Check if service is available and healthy"""
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=5)
            if response.status_code == 200:
                health = response.json()
                self.service_available = True
                print(f"✅ Service healthy: {health.get('status', 'unknown')}")
                return True
            else:
                print(f"❌ Service unhealthy: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Service not available: {e}")
            return False

    def test_no_cpu_reencoding(self) -> TestResult:
        """
        CRITICAL TEST: Verify CPU is NOT re-encoding

        Method: Compare timing between stages. If CPU re-encoding,
        encoder stage would take 300+ ms. With fix, should be ~20ms.
        """
        start = time.perf_counter()

        try:
            # Get baseline stats
            stats_before = requests.get(STATS_ENDPOINT).json()

            # Send test request
            test_file = TEST_AUDIO_DIR / "test_5s.wav"

            if not test_file.exists():
                return TestResult(
                    name="no_cpu_reencoding",
                    passed=False,
                    message=f"Test file not found: {test_file}",
                    details={},
                    duration_ms=0
                )

            with open(test_file, 'rb') as f:
                files = {'file': ('test_5s.wav', f, 'audio/wav')}
                response = requests.post(TRANSCRIPTION_ENDPOINT, files=files, timeout=30)

            elapsed_ms = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                return TestResult(
                    name="no_cpu_reencoding",
                    passed=False,
                    message=f"Request failed: HTTP {response.status_code}",
                    details={'response': response.text},
                    duration_ms=elapsed_ms
                )

            result = response.json()

            # Get stats after
            stats_after = requests.get(STATS_ENDPOINT).json()

            # Check timing breakdown if available
            timing = result.get('timing', {})
            encoder_time = timing.get('encoder_ms', 0)
            decoder_time = timing.get('decoder_ms', 0)

            # If re-encoding, encoder would be 300+ ms (CPU Whisper encoder)
            # With fix, should be ~20ms (NPU encoder only)
            cpu_reencoding_detected = encoder_time > 100  # Threshold

            # Also check total time - if too slow, likely re-encoding
            target_time = TEST_FILES["test_5s.wav"]["target_time_ms"]
            performance_acceptable = elapsed_ms < (target_time * 1.5)  # 50% margin

            passed = not cpu_reencoding_detected and performance_acceptable

            details = {
                'total_time_ms': elapsed_ms,
                'encoder_time_ms': encoder_time,
                'decoder_time_ms': decoder_time,
                'target_time_ms': target_time,
                'cpu_reencoding_detected': cpu_reencoding_detected,
                'performance_acceptable': performance_acceptable,
                'text': result.get('text', ''),
            }

            if passed:
                message = f"✓ No CPU re-encoding detected (encoder: {encoder_time}ms, total: {elapsed_ms:.0f}ms)"
            else:
                if cpu_reencoding_detected:
                    message = f"✗ CPU re-encoding detected! Encoder took {encoder_time}ms (should be ~20ms)"
                else:
                    message = f"✗ Performance too slow: {elapsed_ms:.0f}ms (target: <{target_time * 1.5:.0f}ms)"

            return TestResult(
                name="no_cpu_reencoding",
                passed=passed,
                message=message,
                details=details,
                duration_ms=elapsed_ms
            )

        except Exception as e:
            return TestResult(
                name="no_cpu_reencoding",
                passed=False,
                message=f"Test failed with error: {e}",
                details={'error': str(e)},
                duration_ms=(time.perf_counter() - start) * 1000
            )

    def test_encoder_output_used(self) -> TestResult:
        """
        Verify NPU encoder output flows to decoder

        This checks that encoder output exists and is being used.
        Ideally we'd instrument the code, but we can infer from timing.
        """
        start = time.perf_counter()

        try:
            test_file = TEST_AUDIO_DIR / "test_1s.wav"

            if not test_file.exists():
                return TestResult(
                    name="encoder_output_used",
                    passed=False,
                    message=f"Test file not found: {test_file}",
                    details={},
                    duration_ms=0
                )

            with open(test_file, 'rb') as f:
                files = {'file': ('test_1s.wav', f, 'audio/wav')}
                response = requests.post(TRANSCRIPTION_ENDPOINT, files=files, timeout=30)

            elapsed_ms = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                return TestResult(
                    name="encoder_output_used",
                    passed=False,
                    message=f"Request failed: HTTP {response.status_code}",
                    details={},
                    duration_ms=elapsed_ms
                )

            result = response.json()
            timing = result.get('timing', {})

            # Check if timing breakdown exists (indicates instrumentation)
            has_timing = 'encoder_ms' in timing and 'decoder_ms' in timing

            # If encoder is fast (~20ms) and decoder is reasonable (~150ms),
            # that indicates encoder output is being used
            encoder_time = timing.get('encoder_ms', 0)
            decoder_time = timing.get('decoder_ms', 0)

            encoder_fast = encoder_time < 50  # Should be ~20ms with NPU
            decoder_reasonable = 50 < decoder_time < 500  # Decoder work

            passed = has_timing and encoder_fast and decoder_reasonable

            details = {
                'has_timing_breakdown': has_timing,
                'encoder_ms': encoder_time,
                'decoder_ms': decoder_time,
                'encoder_fast': encoder_fast,
                'decoder_reasonable': decoder_reasonable,
                'total_ms': elapsed_ms,
            }

            if passed:
                message = f"✓ Encoder output appears to be used (encoder: {encoder_time}ms, decoder: {decoder_time}ms)"
            else:
                if not has_timing:
                    message = "⚠ No timing breakdown available - cannot verify"
                else:
                    message = f"✗ Timing suggests re-encoding (encoder: {encoder_time}ms, decoder: {decoder_time}ms)"

            return TestResult(
                name="encoder_output_used",
                passed=passed,
                message=message,
                details=details,
                duration_ms=elapsed_ms
            )

        except Exception as e:
            return TestResult(
                name="encoder_output_used",
                passed=False,
                message=f"Test failed: {e}",
                details={'error': str(e)},
                duration_ms=(time.perf_counter() - start) * 1000
            )

    def test_accuracy_maintained(self) -> TestResult:
        """
        Verify transcription accuracy vs baseline

        Compare Week 19.5 transcriptions against Week 17/18 baselines.
        Accuracy should be >95% similar (allowing for minor variations).
        """
        start = time.perf_counter()

        results = []
        all_passed = True

        try:
            for filename, info in TEST_FILES.items():
                test_file = TEST_AUDIO_DIR / filename

                if not test_file.exists():
                    results.append({
                        'file': filename,
                        'status': 'missing',
                        'message': f'File not found: {test_file}'
                    })
                    all_passed = False
                    continue

                with open(test_file, 'rb') as f:
                    files = {'file': (filename, f, 'audio/wav')}
                    response = requests.post(TRANSCRIPTION_ENDPOINT, files=files, timeout=30)

                if response.status_code != 200:
                    results.append({
                        'file': filename,
                        'status': 'error',
                        'message': f'HTTP {response.status_code}'
                    })
                    all_passed = False
                    continue

                result = response.json()
                actual_text = result.get('text', '').strip()
                baseline_text = info['baseline_text'].strip()

                # Calculate similarity (simple word overlap)
                actual_words = set(actual_text.lower().split())
                baseline_words = set(baseline_text.lower().split())

                if len(baseline_words) == 0:
                    similarity = 1.0 if len(actual_words) == 0 else 0.0
                else:
                    overlap = len(actual_words & baseline_words)
                    similarity = overlap / len(baseline_words)

                passed = similarity >= 0.95

                results.append({
                    'file': filename,
                    'status': 'pass' if passed else 'fail',
                    'actual': actual_text,
                    'baseline': baseline_text,
                    'similarity': similarity,
                })

                if not passed:
                    all_passed = False

            elapsed_ms = (time.perf_counter() - start) * 1000

            if all_passed:
                message = f"✓ All {len(results)} files matched baseline (>95% similarity)"
            else:
                failed = [r for r in results if r['status'] != 'pass']
                message = f"✗ {len(failed)}/{len(results)} files failed accuracy check"

            return TestResult(
                name="accuracy_maintained",
                passed=all_passed,
                message=message,
                details={'results': results},
                duration_ms=elapsed_ms
            )

        except Exception as e:
            return TestResult(
                name="accuracy_maintained",
                passed=False,
                message=f"Test failed: {e}",
                details={'error': str(e), 'results': results},
                duration_ms=(time.perf_counter() - start) * 1000
            )

    def test_performance_target(self) -> TestResult:
        """
        Verify performance meets Week 19.5 targets

        Target: >25× realtime (5s audio in <200ms)
        Baseline: 10.1× realtime (5s audio in 495ms)
        """
        start = time.perf_counter()

        try:
            test_file = TEST_AUDIO_DIR / "test_5s.wav"
            duration = TEST_FILES["test_5s.wav"]["duration"]
            target_time_ms = TEST_FILES["test_5s.wav"]["target_time_ms"]

            if not test_file.exists():
                return TestResult(
                    name="performance_target",
                    passed=False,
                    message=f"Test file not found: {test_file}",
                    details={},
                    duration_ms=0
                )

            # Run 5 times for statistical significance
            times = []
            for i in range(5):
                run_start = time.perf_counter()

                with open(test_file, 'rb') as f:
                    files = {'file': ('test_5s.wav', f, 'audio/wav')}
                    response = requests.post(TRANSCRIPTION_ENDPOINT, files=files, timeout=30)

                run_elapsed = time.perf_counter() - run_start

                if response.status_code == 200:
                    times.append(run_elapsed)

            if len(times) == 0:
                return TestResult(
                    name="performance_target",
                    passed=False,
                    message="All runs failed",
                    details={},
                    duration_ms=(time.perf_counter() - start) * 1000
                )

            avg_time = np.mean(times)
            avg_time_ms = avg_time * 1000
            min_time_ms = np.min(times) * 1000
            max_time_ms = np.max(times) * 1000
            std_time_ms = np.std(times) * 1000

            realtime_factor = duration / avg_time

            # Check if meets target (>25× realtime = <200ms for 5s)
            meets_target = avg_time_ms < target_time_ms

            # Calculate improvement vs baseline
            baseline_time_ms = TEST_FILES["test_5s.wav"]["baseline_time_ms"]
            speedup = baseline_time_ms / avg_time_ms

            details = {
                'runs': len(times),
                'avg_time_ms': avg_time_ms,
                'min_time_ms': min_time_ms,
                'max_time_ms': max_time_ms,
                'std_time_ms': std_time_ms,
                'realtime_factor': realtime_factor,
                'target_time_ms': target_time_ms,
                'baseline_time_ms': baseline_time_ms,
                'speedup_vs_baseline': speedup,
                'meets_target': meets_target,
            }

            if meets_target:
                message = (
                    f"✓ Performance target MET: {avg_time_ms:.0f}ms avg "
                    f"({realtime_factor:.1f}× realtime, {speedup:.2f}× speedup)"
                )
            else:
                message = (
                    f"✗ Performance target NOT MET: {avg_time_ms:.0f}ms avg "
                    f"({realtime_factor:.1f}× realtime), target: <{target_time_ms}ms"
                )

            return TestResult(
                name="performance_target",
                passed=meets_target,
                message=message,
                details=details,
                duration_ms=(time.perf_counter() - start) * 1000
            )

        except Exception as e:
            return TestResult(
                name="performance_target",
                passed=False,
                message=f"Test failed: {e}",
                details={'error': str(e)},
                duration_ms=(time.perf_counter() - start) * 1000
            )

    def test_concurrent_requests(self) -> TestResult:
        """
        Test pipeline handles concurrent requests correctly

        Send 10 concurrent requests and verify all succeed.
        """
        start = time.perf_counter()

        try:
            import concurrent.futures

            test_file = TEST_AUDIO_DIR / "test_1s.wav"

            if not test_file.exists():
                return TestResult(
                    name="concurrent_requests",
                    passed=False,
                    message=f"Test file not found: {test_file}",
                    details={},
                    duration_ms=0
                )

            def send_request(i):
                """Send a single transcription request"""
                try:
                    with open(test_file, 'rb') as f:
                        files = {'file': (f'test_{i}.wav', f, 'audio/wav')}
                        response = requests.post(
                            TRANSCRIPTION_ENDPOINT,
                            files=files,
                            timeout=30
                        )

                    return {
                        'index': i,
                        'status': response.status_code,
                        'success': response.status_code == 200,
                        'text': response.json().get('text', '') if response.status_code == 200 else None,
                    }
                except Exception as e:
                    return {
                        'index': i,
                        'status': None,
                        'success': False,
                        'error': str(e),
                    }

            # Send 10 concurrent requests
            num_requests = 10
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
                futures = [executor.submit(send_request, i) for i in range(num_requests)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            elapsed_ms = (time.perf_counter() - start) * 1000

            # Check results
            successes = [r for r in results if r['success']]
            failures = [r for r in results if not r['success']]

            all_passed = len(successes) == num_requests

            details = {
                'total_requests': num_requests,
                'successes': len(successes),
                'failures': len(failures),
                'results': results,
                'total_time_ms': elapsed_ms,
                'avg_time_ms': elapsed_ms / num_requests,
            }

            if all_passed:
                message = f"✓ All {num_requests} concurrent requests succeeded ({elapsed_ms:.0f}ms total)"
            else:
                message = f"✗ {len(failures)}/{num_requests} concurrent requests failed"

            return TestResult(
                name="concurrent_requests",
                passed=all_passed,
                message=message,
                details=details,
                duration_ms=elapsed_ms
            )

        except Exception as e:
            return TestResult(
                name="concurrent_requests",
                passed=False,
                message=f"Test failed: {e}",
                details={'error': str(e)},
                duration_ms=(time.perf_counter() - start) * 1000
            )

    def run_all_tests(self) -> bool:
        """Run all validation tests"""
        print("\n" + "="*70)
        print("  WEEK 19.5 PIPELINE VALIDATION")
        print("="*70 + "\n")

        # Check service health
        if not self.check_service_health():
            print("\n❌ Service not available. Cannot run tests.")
            return False

        print()

        # Run all tests
        tests = [
            ("Critical: No CPU Re-encoding", self.test_no_cpu_reencoding),
            ("Critical: Encoder Output Used", self.test_encoder_output_used),
            ("Accuracy: Baseline Maintained", self.test_accuracy_maintained),
            ("Performance: Target Met", self.test_performance_target),
            ("Stress: Concurrent Requests", self.test_concurrent_requests),
        ]

        for test_name, test_func in tests:
            print(f"Running: {test_name}...")
            result = test_func()
            self.results.append(result)
            print(f"  {result.message}")
            print(f"  Duration: {result.duration_ms:.1f}ms\n")

        # Print summary
        self.print_summary()

        # Return overall pass/fail
        return all(r.passed for r in self.results)

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("  TEST SUMMARY")
        print("="*70 + "\n")

        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]

        print(f"Total Tests: {len(self.results)}")
        print(f"Passed:      {len(passed)} ✓")
        print(f"Failed:      {len(failed)} ✗\n")

        if failed:
            print("Failed Tests:")
            for result in failed:
                print(f"  ✗ {result.name}: {result.message}")
            print()

        # Print detailed results
        print("Detailed Results:")
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {status} - {result.name}")
            print(f"    {result.message}")

            # Print key details
            if 'realtime_factor' in result.details:
                print(f"    Realtime: {result.details['realtime_factor']:.1f}×")
            if 'speedup_vs_baseline' in result.details:
                print(f"    Speedup: {result.details['speedup_vs_baseline']:.2f}×")
            if 'total_time_ms' in result.details:
                print(f"    Time: {result.details['total_time_ms']:.0f}ms")

            print()

        print("="*70 + "\n")

    def export_results(self, output_file: Path):
        """Export test results to JSON"""
        data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'service_url': SERVICE_URL,
            'service_available': self.service_available,
            'total_tests': len(self.results),
            'passed': sum(1 for r in self.results if r.passed),
            'failed': sum(1 for r in self.results if not r.passed),
            'results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'message': r.message,
                    'details': r.details,
                    'duration_ms': r.duration_ms,
                }
                for r in self.results
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Results exported to: {output_file}\n")


def main():
    """Run Week 19.5 validation suite"""
    validator = Week195Validator()

    success = validator.run_all_tests()

    # Export results
    output_file = Path(__file__).parent / "week19_5_validation_results.json"
    validator.export_results(output_file)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
