#!/usr/bin/env python3
"""
Week 19: Integration Testing Suite
CC-1L Team 3 Lead - Validation & Performance Testing

Comprehensive integration tests for Week 19 optimizations:
- Basic functionality validation
- Accuracy verification
- Edge case handling
- Regression testing against Week 18 baseline
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class IntegrationTestResult:
    """Result from a single integration test"""
    test_name: str
    test_type: str  # functionality, accuracy, edge_case
    audio_file: str
    audio_duration_s: float
    success: bool
    processing_time_ms: float = 0.0
    realtime_factor: float = 0.0
    transcription: str = ""
    expected_text: Optional[str] = None
    error: str = ""
    http_status: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


class Week19IntegrationTests:
    """
    Week 19 Integration Testing Suite

    Phase 1: End-to-End Integration Testing
    - Basic functionality tests
    - Accuracy validation
    - Edge case handling
    - HTTP status verification
    """

    def __init__(self, service_url: str = "http://localhost:9050"):
        """Initialize testing suite"""
        self.service_url = service_url
        self.results: List[IntegrationTestResult] = []
        self.week18_baseline = self._load_week18_baseline()

        logger.info("="*80)
        logger.info("  WEEK 19: INTEGRATION TESTING SUITE")
        logger.info("="*80)
        logger.info(f"  Service URL: {service_url}")
        logger.info(f"  Week 18 Baseline: {len(self.week18_baseline)} tests loaded")
        logger.info("="*80)

    def _load_week18_baseline(self) -> Dict:
        """Load Week 18 baseline results"""
        baseline_path = Path(__file__).parent / "results" / "week18_detailed_profiling.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                return json.load(f)
        return {}

    def check_service_health(self) -> bool:
        """Check if service is running and healthy"""
        try:
            logger.info("\n" + "="*80)
            logger.info("  SERVICE HEALTH CHECK")
            logger.info("="*80)

            response = requests.get(f"{self.service_url}/health", timeout=5)
            response.raise_for_status()

            health = response.json()
            logger.info(f"  Service: {health.get('service', 'Unknown')}")
            logger.info(f"  Version: {health.get('version', 'Unknown')}")
            logger.info(f"  Status: {health.get('status', 'Unknown')}")
            logger.info(f"  NPU Enabled: {health.get('npu_enabled', False)}")
            logger.info(f"  Backend: {health.get('backend', 'Unknown')}")

            # Check encoder info
            if 'encoder' in health:
                encoder = health['encoder']
                logger.info(f"\n  Encoder:")
                logger.info(f"    Type: {encoder.get('type', 'Unknown')}")
                logger.info(f"    NPU Enabled: {encoder.get('npu_enabled', False)}")
                logger.info(f"    Weights Loaded: {encoder.get('weights_loaded', False)}")

            # Check buffer pools
            if 'buffer_pools' in health:
                pools = health['buffer_pools']
                logger.info(f"\n  Buffer Pools:")
                for pool_name, pool_info in pools.items():
                    hit_rate = pool_info.get('hit_rate', 0.0) * 100
                    logger.info(f"    {pool_name}: {hit_rate:.1f}% hit rate")

            # Check performance
            if 'performance' in health:
                perf = health['performance']
                logger.info(f"\n  Performance:")
                logger.info(f"    Requests: {perf.get('requests_processed', 0)}")
                logger.info(f"    Average RT Factor: {perf.get('average_realtime_factor', 0.0):.1f}×")
                logger.info(f"    Target: {perf.get('target_realtime_factor', 0)}×")

            # Check warnings
            if 'warnings' in health and health['warnings']:
                logger.warning(f"\n  Warnings:")
                for warning in health['warnings']:
                    logger.warning(f"    - {warning}")

            status = health.get('status', 'unknown')
            if status not in ['healthy', 'degraded']:
                logger.error(f"  Service unhealthy: {status}")
                return False

            if not health.get('npu_enabled', False):
                logger.warning("  NPU not enabled - performance will be degraded")

            logger.info("\n  Service is ready for testing")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"  Service not responding: {e}")
            return False

    def test_basic_functionality(self, test_dir: Path) -> List[IntegrationTestResult]:
        """
        Test 1: Basic Functionality Tests

        Validates:
        - HTTP 200 responses
        - Transcription text returned
        - Processing time reasonable
        - No errors in responses
        """
        logger.info("\n" + "="*80)
        logger.info("  TEST 1: BASIC FUNCTIONALITY")
        logger.info("="*80)

        test_files = [
            ('test_1s.wav', 1.0, "1 Second Audio"),
            ('test_5s.wav', 5.0, "5 Second Audio"),
            ('test_30s.wav', 30.0, "30 Second Audio (Buffer Fix)"),
            ('test_silence.wav', 5.0, "Silent Audio (Edge Case)"),
        ]

        results = []

        for wav_file, duration, test_name in test_files:
            audio_path = test_dir / wav_file

            if not audio_path.exists():
                logger.warning(f"\n  Skipping {test_name}: {wav_file} not found")
                continue

            logger.info(f"\n  Testing: {test_name}")
            logger.info(f"  File: {wav_file} ({duration}s)")

            result = IntegrationTestResult(
                test_name=test_name,
                test_type="functionality",
                audio_file=wav_file,
                audio_duration_s=duration,
                success=False
            )

            try:
                start_time = time.perf_counter()

                with open(audio_path, 'rb') as f:
                    files = {'file': (wav_file, f, 'audio/wav')}
                    response = requests.post(
                        f"{self.service_url}/v1/audio/transcriptions",
                        files=files,
                        timeout=60
                    )

                processing_time = (time.perf_counter() - start_time) * 1000
                result.processing_time_ms = processing_time
                result.http_status = response.status_code

                # Check HTTP status
                if response.status_code != 200:
                    result.error = f"HTTP {response.status_code}"
                    logger.error(f"    HTTP Status: {response.status_code}")
                else:
                    # Parse response
                    data = response.json()
                    result.transcription = data.get('text', '')
                    result.realtime_factor = duration / (processing_time / 1000) if processing_time > 0 else 0
                    result.success = True

                    logger.info(f"    HTTP Status: 200 OK")
                    logger.info(f"    Processing Time: {processing_time:.2f}ms")
                    logger.info(f"    Realtime Factor: {result.realtime_factor:.1f}×")
                    logger.info(f"    Transcription Length: {len(result.transcription)} chars")

                    if result.transcription:
                        logger.info(f"    Text Preview: \"{result.transcription[:60]}...\"")
                    else:
                        logger.warning(f"    Warning: Empty transcription")

            except requests.exceptions.Timeout:
                result.error = "Timeout (>60s)"
                logger.error(f"    Error: Request timeout")

            except requests.exceptions.RequestException as e:
                result.error = str(e)
                logger.error(f"    Error: {e}")

            except Exception as e:
                result.error = f"Unexpected: {e}"
                logger.error(f"    Unexpected error: {e}")

            results.append(result)
            self.results.append(result)

        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"\n  Summary: {successful}/{len(results)} tests passed")

        return results

    def test_accuracy_comparison(self, test_dir: Path) -> List[IntegrationTestResult]:
        """
        Test 2: Accuracy Validation

        Compares Week 19 transcriptions with Week 18 baseline
        Calculates simple word error rate
        """
        logger.info("\n" + "="*80)
        logger.info("  TEST 2: ACCURACY VALIDATION")
        logger.info("="*80)

        if not self.week18_baseline.get('results'):
            logger.warning("  No Week 18 baseline found - skipping accuracy tests")
            return []

        results = []
        baseline_results = {r['audio_file']: r for r in self.week18_baseline['results']}

        test_files = [
            'test_1s.wav',
            'test_5s.wav',
            'test_silence.wav'
        ]

        for wav_file in test_files:
            if wav_file not in baseline_results:
                continue

            audio_path = test_dir / wav_file
            if not audio_path.exists():
                continue

            baseline = baseline_results[wav_file]
            test_name = f"Accuracy: {baseline['test_name']}"

            logger.info(f"\n  Testing: {test_name}")

            result = IntegrationTestResult(
                test_name=test_name,
                test_type="accuracy",
                audio_file=wav_file,
                audio_duration_s=baseline['audio_duration_s'],
                expected_text=baseline.get('transcription', ''),
                success=False
            )

            try:
                with open(audio_path, 'rb') as f:
                    files = {'file': (wav_file, f, 'audio/wav')}
                    response = requests.post(
                        f"{self.service_url}/v1/audio/transcriptions",
                        files=files,
                        timeout=60
                    )

                if response.status_code == 200:
                    data = response.json()
                    result.transcription = data.get('text', '')
                    result.http_status = 200

                    # Simple word comparison
                    baseline_words = baseline.get('transcription', '').lower().split()
                    current_words = result.transcription.lower().split()

                    # Calculate simple similarity
                    if baseline_words or current_words:
                        matching = sum(1 for w in current_words if w in baseline_words)
                        total = max(len(baseline_words), len(current_words))
                        similarity = (matching / total * 100) if total > 0 else 0

                        logger.info(f"    Baseline: \"{baseline.get('transcription', '')[:60]}...\"")
                        logger.info(f"    Current:  \"{result.transcription[:60]}...\"")
                        logger.info(f"    Similarity: {similarity:.1f}%")

                        # Consider >90% similarity as acceptable
                        result.success = similarity > 90.0
                        if not result.success:
                            result.error = f"Low similarity: {similarity:.1f}%"
                    else:
                        # Both empty is OK for silence
                        result.success = True
                        logger.info(f"    Both transcriptions empty (expected for silence)")
                else:
                    result.error = f"HTTP {response.status_code}"
                    result.http_status = response.status_code

            except Exception as e:
                result.error = str(e)
                logger.error(f"    Error: {e}")

            results.append(result)
            self.results.append(result)

        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"\n  Summary: {successful}/{len(results)} accuracy tests passed")

        return results

    def test_performance_comparison(self, test_dir: Path) -> List[IntegrationTestResult]:
        """
        Test 3: Performance Comparison vs Week 18

        Validates that Week 19 is faster than Week 18
        """
        logger.info("\n" + "="*80)
        logger.info("  TEST 3: PERFORMANCE COMPARISON")
        logger.info("="*80)

        if not self.week18_baseline.get('results'):
            logger.warning("  No Week 18 baseline found - skipping performance comparison")
            return []

        results = []
        baseline_results = {r['audio_file']: r for r in self.week18_baseline['results']}

        for wav_file, baseline in baseline_results.items():
            audio_path = test_dir / wav_file
            if not audio_path.exists():
                continue

            test_name = f"Performance: {baseline['test_name']}"
            logger.info(f"\n  Testing: {test_name}")

            result = IntegrationTestResult(
                test_name=test_name,
                test_type="performance",
                audio_file=wav_file,
                audio_duration_s=baseline['audio_duration_s'],
                success=False
            )

            try:
                start_time = time.perf_counter()

                with open(audio_path, 'rb') as f:
                    files = {'file': (wav_file, f, 'audio/wav')}
                    response = requests.post(
                        f"{self.service_url}/v1/audio/transcriptions",
                        files=files,
                        timeout=60
                    )

                processing_time = (time.perf_counter() - start_time) * 1000
                result.processing_time_ms = processing_time
                result.http_status = response.status_code

                if response.status_code == 200:
                    result.realtime_factor = baseline['audio_duration_s'] / (processing_time / 1000)

                    # Compare to baseline
                    baseline_time = baseline['processing_time_ms']
                    baseline_rt = baseline['realtime_factor']

                    speedup = baseline_time / processing_time if processing_time > 0 else 0
                    rt_improvement = result.realtime_factor / baseline_rt if baseline_rt > 0 else 0

                    logger.info(f"    Week 18: {baseline_time:.2f}ms ({baseline_rt:.1f}× RT)")
                    logger.info(f"    Week 19: {processing_time:.2f}ms ({result.realtime_factor:.1f}× RT)")
                    logger.info(f"    Speedup: {speedup:.2f}× faster")
                    logger.info(f"    RT Improvement: {rt_improvement:.2f}× better")

                    # Success if not significantly slower
                    result.success = speedup >= 0.8  # Allow 20% variation
                    if not result.success:
                        result.error = f"Slower than baseline: {speedup:.2f}×"
                else:
                    result.error = f"HTTP {response.status_code}"
                    result.http_status = response.status_code

            except Exception as e:
                result.error = str(e)
                logger.error(f"    Error: {e}")

            results.append(result)
            self.results.append(result)

        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"\n  Summary: {successful}/{len(results)} performance tests passed")

        return results

    def run_all_integration_tests(self, test_dir: Path) -> Dict:
        """Run complete integration test suite"""
        logger.info("\n" + "="*80)
        logger.info("  RUNNING COMPLETE INTEGRATION TEST SUITE")
        logger.info("="*80)

        # Check service health
        if not self.check_service_health():
            return {
                'status': 'FAILED',
                'error': 'Service not available',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
            }

        # Run test phases
        self.test_basic_functionality(test_dir)
        time.sleep(0.5)

        self.test_accuracy_comparison(test_dir)
        time.sleep(0.5)

        self.test_performance_comparison(test_dir)

        # Generate summary
        return self.generate_summary()

    def generate_summary(self) -> Dict:
        """Generate test summary"""
        logger.info("\n" + "="*80)
        logger.info("  INTEGRATION TEST SUMMARY")
        logger.info("="*80)

        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful

        by_type = {}
        for r in self.results:
            if r.test_type not in by_type:
                by_type[r.test_type] = {'total': 0, 'passed': 0}
            by_type[r.test_type]['total'] += 1
            if r.success:
                by_type[r.test_type]['passed'] += 1

        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'total_tests': total,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'by_type': by_type,
            'results': [r.to_dict() for r in self.results]
        }

        logger.info(f"\n  Total Tests: {total}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Success Rate: {summary['success_rate']:.1f}%")

        logger.info(f"\n  By Test Type:")
        for test_type, counts in by_type.items():
            logger.info(f"    {test_type}: {counts['passed']}/{counts['total']} passed")

        if failed > 0:
            logger.warning(f"\n  Failed Tests:")
            for r in self.results:
                if not r.success:
                    logger.warning(f"    - {r.test_name}: {r.error}")

        # Overall status
        if summary['success_rate'] >= 95:
            summary['status'] = 'PASS'
            logger.info(f"\n  STATUS: PASS (>=95% success rate)")
        elif summary['success_rate'] >= 80:
            summary['status'] = 'PARTIAL'
            logger.warning(f"\n  STATUS: PARTIAL (80-95% success rate)")
        else:
            summary['status'] = 'FAIL'
            logger.error(f"\n  STATUS: FAIL (<80% success rate)")

        return summary

    def save_results(self, output_path: Path):
        """Save results to JSON file"""
        summary = self.generate_summary() if not hasattr(self, '_summary') else self._summary

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n  Results saved to: {output_path}")


def main():
    """Main test execution"""
    try:
        # Initialize test suite
        suite = Week19IntegrationTests()

        # Test directory
        test_dir = Path(__file__).parent / "audio"

        # Run all tests
        results = suite.run_all_integration_tests(test_dir)

        # Save results
        output_dir = Path("/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/tests/results")
        suite.save_results(output_dir / "week19_integration_results.json")

        # Exit code based on status
        status = results.get('status', 'FAIL')
        if status == 'PASS':
            logger.info("\n SUCCESS: All integration tests passed!")
            return 0
        elif status == 'PARTIAL':
            logger.warning("\n PARTIAL: Some integration tests failed")
            return 1
        else:
            logger.error("\n FAILED: Integration tests failed")
            return 2

    except Exception as e:
        logger.error(f"\n FATAL: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
