#!/usr/bin/env python3
"""
Week 19.6 Comprehensive Validation & Testing Suite

Mission: Validate Week 18 baseline is restored after Week 19.5 rollback

Test Categories:
1. Baseline Validation (1s, 5s, 30s, silence)
2. Multi-Stream Reliability (4, 8, 16 concurrent streams)
3. Long-Form Audio (30s, 60s)
4. Regression Testing (Week 19.5 issues resolved)

Success Criteria:
- All baseline tests passing (4/4 or 5/5 if 30s works)
- Performance >= 7.9x realtime (Week 18 parity)
- Multi-stream: 100% success at 4, 8, 16 streams
- 30s audio: Working (not failing)
- No Week 19.5 regressions (empty transcriptions, hallucinations)

Author: Week 19.6 Team 3 (Validation & Testing)
Date: November 2, 2025
"""

import asyncio
import aiohttp
import time
import json
import numpy as np
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://localhost:9050"
AUDIO_DIR = Path(__file__).parent.parent / "tests" / "audio"
RESULTS_DIR = Path(__file__).parent.parent / "tests" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Expected transcriptions (from Week 17/18 baseline)
EXPECTED_TRANSCRIPTIONS = {
    "test_1s.wav": " Ooh.",
    "test_5s.wav": " Whoa! Whoa! Whoa! Whoa!",
    "test_silence.wav": "",  # Silence should transcribe to empty string
}

# Week 18 baseline performance (from WEEK18_COMPLETE.md)
WEEK18_BASELINE = {
    "1s_audio": {"realtime_factor": 3.0, "processing_time_ms": 328},
    "5s_audio": {"realtime_factor": 10.1, "processing_time_ms": 495},
    "silence": {"realtime_factor": 10.6, "processing_time_ms": 473},
    "average_realtime": 7.9,
    "multi_stream_success_rate": 100.0,  # 100% success in Week 18
}


@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    status: str  # "PASS", "FAIL", "ERROR"
    audio_file: str
    audio_duration_s: float
    processing_time_ms: float
    realtime_factor: float
    transcription: str
    expected_transcription: Optional[str]
    accuracy_match: bool
    error_message: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class MultiStreamResult:
    """Multi-stream test result"""
    test_name: str
    num_streams: int
    num_successful: int
    num_failed: int
    success_rate_pct: float
    avg_processing_time_ms: float
    avg_realtime_factor: float
    p50_processing_time_ms: float
    p95_processing_time_ms: float
    errors: List[str]

    def to_dict(self):
        return asdict(self)


class Week196ValidationSuite:
    """Week 19.6 comprehensive validation test suite"""

    def __init__(self, base_url: str = BASE_URL, audio_dir: Path = AUDIO_DIR):
        self.base_url = base_url
        self.audio_dir = audio_dir
        self.results: List[TestResult] = []
        self.multi_stream_results: List[MultiStreamResult] = []

    async def check_health(self) -> bool:
        """Check if service is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health", timeout=10) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        logger.info(f"Health check: {health_data.get('status', 'unknown')}")

                        # Check decoder configuration
                        config = health_data.get('config', {})
                        use_custom = config.get('USE_CUSTOM_DECODER', False)
                        use_faster = config.get('USE_FASTER_WHISPER', False)

                        logger.info(f"Decoder config - Custom: {use_custom}, Faster: {use_faster}")

                        if use_custom or use_faster:
                            logger.warning("⚠️  Week 19.5 decoders still enabled! Expected both false for Week 18 baseline.")
                            logger.warning("    Set USE_CUSTOM_DECODER=false USE_FASTER_WHISPER=false")
                            return False

                        # Check buffer pool configuration
                        buffer_pools = health_data.get('buffer_pools', {})
                        audio_pool = buffer_pools.get('audio', {})
                        audio_total = audio_pool.get('total_buffers', 0)

                        logger.info(f"Audio buffer pool size: {audio_total}")

                        if audio_total < 50:
                            logger.warning(f"⚠️  Audio buffer pool size is {audio_total} (Week 18 had 5)")
                            logger.warning("    For multi-stream tests, recommend 50+ buffers")

                        return True
                    else:
                        logger.error(f"Health check failed: HTTP {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False

    async def transcribe_audio(self, audio_file: str, timeout: int = 30) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Transcribe audio file via API

        Returns:
            (result_dict, error_message)
        """
        audio_path = self.audio_dir / audio_file

        if not audio_path.exists():
            return None, f"Audio file not found: {audio_path}"

        try:
            async with aiohttp.ClientSession() as session:
                with open(audio_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('file', f, filename=audio_file, content_type='audio/wav')

                    start_time = time.time()

                    async with session.post(
                        f"{self.base_url}/v1/audio/transcriptions",
                        data=data,
                        timeout=timeout
                    ) as response:
                        elapsed_ms = (time.time() - start_time) * 1000

                        if response.status == 200:
                            result = await response.json()
                            result['processing_time_ms'] = elapsed_ms
                            return result, None
                        else:
                            error_text = await response.text()
                            return None, f"HTTP {response.status}: {error_text}"
        except asyncio.TimeoutError:
            return None, f"Timeout after {timeout}s"
        except Exception as e:
            return None, f"Exception: {str(e)}"

    async def test_baseline_single(self, audio_file: str, expected_transcription: Optional[str] = None) -> TestResult:
        """Test single audio file (baseline validation)"""
        logger.info(f"Testing {audio_file}...")

        # Get audio duration
        import wave
        audio_path = self.audio_dir / audio_file
        try:
            with wave.open(str(audio_path), 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration_s = frames / rate
        except Exception as e:
            return TestResult(
                test_name=f"baseline_{audio_file}",
                status="ERROR",
                audio_file=audio_file,
                audio_duration_s=0,
                processing_time_ms=0,
                realtime_factor=0,
                transcription="",
                expected_transcription=expected_transcription,
                accuracy_match=False,
                error_message=f"Cannot read audio file: {e}"
            )

        # Transcribe
        result, error = await self.transcribe_audio(audio_file, timeout=60)

        if error:
            return TestResult(
                test_name=f"baseline_{audio_file}",
                status="ERROR",
                audio_file=audio_file,
                audio_duration_s=duration_s,
                processing_time_ms=0,
                realtime_factor=0,
                transcription="",
                expected_transcription=expected_transcription,
                accuracy_match=False,
                error_message=error
            )

        # Extract metrics
        processing_time_ms = result.get('processing_time_ms', 0)
        transcription = result.get('text', '')
        realtime_factor = (duration_s * 1000) / processing_time_ms if processing_time_ms > 0 else 0

        # Check accuracy
        accuracy_match = True
        if expected_transcription is not None:
            accuracy_match = (transcription.strip() == expected_transcription.strip())

        # Determine status
        status = "PASS" if accuracy_match and processing_time_ms > 0 else "FAIL"

        logger.info(f"  Result: {status}")
        logger.info(f"  Transcription: '{transcription}'")
        if expected_transcription:
            logger.info(f"  Expected: '{expected_transcription}'")
        logger.info(f"  Processing time: {processing_time_ms:.1f}ms")
        logger.info(f"  Realtime factor: {realtime_factor:.1f}x")

        return TestResult(
            test_name=f"baseline_{audio_file}",
            status=status,
            audio_file=audio_file,
            audio_duration_s=duration_s,
            processing_time_ms=processing_time_ms,
            realtime_factor=realtime_factor,
            transcription=transcription,
            expected_transcription=expected_transcription,
            accuracy_match=accuracy_match
        )

    async def test_baseline_validation(self) -> Dict:
        """
        Test 1: Baseline Validation

        Tests: 1s, 5s, 30s, silence
        Success: All passing with Week 18 performance
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 1: BASELINE VALIDATION")
        logger.info("="*80)

        test_files = [
            ("test_1s.wav", EXPECTED_TRANSCRIPTIONS.get("test_1s.wav")),
            ("test_5s.wav", EXPECTED_TRANSCRIPTIONS.get("test_5s.wav")),
            ("test_30s.wav", None),  # No expected transcription yet
            ("test_silence.wav", EXPECTED_TRANSCRIPTIONS.get("test_silence.wav")),
        ]

        baseline_results = []

        for audio_file, expected in test_files:
            result = await self.test_baseline_single(audio_file, expected)
            baseline_results.append(result)
            self.results.append(result)

        # Calculate summary statistics
        passed = sum(1 for r in baseline_results if r.status == "PASS")
        failed = sum(1 for r in baseline_results if r.status == "FAIL")
        errors = sum(1 for r in baseline_results if r.status == "ERROR")

        # Calculate average realtime factor (excluding errors)
        valid_results = [r for r in baseline_results if r.status in ["PASS", "FAIL"] and r.realtime_factor > 0]
        avg_realtime = statistics.mean([r.realtime_factor for r in valid_results]) if valid_results else 0

        summary = {
            "test_name": "baseline_validation",
            "total_tests": len(baseline_results),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "success_rate_pct": (passed / len(baseline_results)) * 100 if len(baseline_results) > 0 else 0,
            "avg_realtime_factor": avg_realtime,
            "week18_avg_realtime": WEEK18_BASELINE["average_realtime"],
            "performance_vs_week18_pct": (avg_realtime / WEEK18_BASELINE["average_realtime"]) * 100 if WEEK18_BASELINE["average_realtime"] > 0 else 0,
            "results": [r.to_dict() for r in baseline_results]
        }

        logger.info(f"\nBaseline Validation Summary:")
        logger.info(f"  Tests: {passed}/{len(baseline_results)} passed")
        logger.info(f"  Average realtime factor: {avg_realtime:.1f}x")
        logger.info(f"  Week 18 baseline: {WEEK18_BASELINE['average_realtime']:.1f}x")
        logger.info(f"  Performance vs Week 18: {summary['performance_vs_week18_pct']:.1f}%")

        return summary

    async def test_multi_stream(self, num_streams: int, audio_file: str = "test_1s.wav", runs: int = 2) -> MultiStreamResult:
        """
        Test concurrent streams

        Args:
            num_streams: Number of concurrent requests
            audio_file: Audio file to use
            runs: Number of times to run the test
        """
        logger.info(f"\nTesting {num_streams} concurrent streams ({runs} runs)...")

        all_processing_times = []
        all_errors = []
        total_successful = 0
        total_failed = 0

        for run in range(runs):
            # Launch concurrent requests
            tasks = [self.transcribe_audio(audio_file, timeout=30) for _ in range(num_streams)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result_tuple in enumerate(results):
                if isinstance(result_tuple, Exception):
                    all_errors.append(str(result_tuple))
                    total_failed += 1
                else:
                    result, error = result_tuple
                    if error:
                        all_errors.append(error)
                        total_failed += 1
                    else:
                        all_processing_times.append(result.get('processing_time_ms', 0))
                        total_successful += 1

        # Calculate statistics
        total_requests = num_streams * runs
        success_rate = (total_successful / total_requests) * 100 if total_requests > 0 else 0

        avg_time = statistics.mean(all_processing_times) if all_processing_times else 0
        p50_time = statistics.median(all_processing_times) if all_processing_times else 0
        p95_time = statistics.quantiles(all_processing_times, n=20)[18] if len(all_processing_times) >= 20 else (max(all_processing_times) if all_processing_times else 0)

        # Get audio duration for realtime factor
        import wave
        audio_path = self.audio_dir / audio_file
        with wave.open(str(audio_path), 'rb') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration_s = frames / rate

        avg_realtime = (duration_s * 1000) / avg_time if avg_time > 0 else 0

        logger.info(f"  Success rate: {success_rate:.1f}% ({total_successful}/{total_requests})")
        logger.info(f"  Avg processing time: {avg_time:.1f}ms (realtime: {avg_realtime:.1f}x)")
        logger.info(f"  p50: {p50_time:.1f}ms, p95: {p95_time:.1f}ms")
        if all_errors:
            logger.info(f"  Errors: {len(all_errors)} unique errors")
            for error in set(all_errors[:5]):  # Show first 5 unique errors
                logger.info(f"    - {error}")

        return MultiStreamResult(
            test_name=f"multi_stream_{num_streams}",
            num_streams=num_streams,
            num_successful=total_successful,
            num_failed=total_failed,
            success_rate_pct=success_rate,
            avg_processing_time_ms=avg_time,
            avg_realtime_factor=avg_realtime,
            p50_processing_time_ms=p50_time,
            p95_processing_time_ms=p95_time,
            errors=list(set(all_errors))
        )

    async def test_multi_stream_reliability(self) -> Dict:
        """
        Test 2: Multi-Stream Reliability

        Tests: 4, 8, 16 concurrent streams
        Success: 100% success rate at all levels
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 2: MULTI-STREAM RELIABILITY")
        logger.info("="*80)

        stream_counts = [4, 8, 16]
        multi_stream_results = []

        for num_streams in stream_counts:
            result = await self.test_multi_stream(num_streams, "test_1s.wav", runs=2)
            multi_stream_results.append(result)
            self.multi_stream_results.append(result)

        # Calculate summary
        avg_success_rate = statistics.mean([r.success_rate_pct for r in multi_stream_results])
        all_100_pct = all(r.success_rate_pct == 100.0 for r in multi_stream_results)

        summary = {
            "test_name": "multi_stream_reliability",
            "stream_counts_tested": stream_counts,
            "avg_success_rate_pct": avg_success_rate,
            "all_100_pct_success": all_100_pct,
            "week18_success_rate": WEEK18_BASELINE["multi_stream_success_rate"],
            "results": [r.to_dict() for r in multi_stream_results]
        }

        logger.info(f"\nMulti-Stream Reliability Summary:")
        logger.info(f"  Average success rate: {avg_success_rate:.1f}%")
        logger.info(f"  Week 18 baseline: {WEEK18_BASELINE['multi_stream_success_rate']:.1f}%")
        logger.info(f"  All 100% success: {all_100_pct}")

        return summary

    async def test_long_form_audio(self) -> Dict:
        """
        Test 3: Long-Form Audio

        Tests: 30s, 60s audio
        Success: Both working without failures
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 3: LONG-FORM AUDIO")
        logger.info("="*80)

        test_files = ["test_30s.wav", "test_60s.wav"]
        long_form_results = []

        for audio_file in test_files:
            result = await self.test_baseline_single(audio_file, expected_transcription=None)
            long_form_results.append(result)
            self.results.append(result)

        # Check if both working
        all_working = all(r.status in ["PASS", "FAIL"] and r.error_message is None for r in long_form_results)

        summary = {
            "test_name": "long_form_audio",
            "files_tested": test_files,
            "all_working": all_working,
            "results": [r.to_dict() for r in long_form_results]
        }

        logger.info(f"\nLong-Form Audio Summary:")
        logger.info(f"  All files working: {all_working}")
        for r in long_form_results:
            logger.info(f"  {r.audio_file}: {r.status} ({r.realtime_factor:.1f}x realtime)")

        return summary

    async def test_regression_checks(self) -> Dict:
        """
        Test 4: Regression Testing

        Verify Week 19.5 issues resolved:
        - No empty transcriptions (1s audio should have text)
        - No hallucinations (silence should be empty)
        - Consistent results across runs (5s audio)
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 4: REGRESSION CHECKS")
        logger.info("="*80)

        regression_tests = []

        # Test 1: No empty transcriptions (run 1s audio 3 times)
        logger.info("\nChecking for empty transcription bug (1s audio, 3 runs)...")
        empty_count = 0
        for i in range(3):
            result, error = await self.transcribe_audio("test_1s.wav")
            if result:
                text = result.get('text', '').strip()
                if text == '':
                    empty_count += 1
                    logger.warning(f"  Run {i+1}: EMPTY transcription (Week 19.5 bug!)")
                else:
                    logger.info(f"  Run {i+1}: '{text}' (OK)")

        no_empty_bug = (empty_count == 0)
        regression_tests.append({
            "test": "no_empty_transcriptions",
            "passed": no_empty_bug,
            "empty_count": empty_count,
            "total_runs": 3
        })

        # Test 2: No hallucinations (silence should be empty)
        logger.info("\nChecking for hallucination bug (silence)...")
        result, error = await self.transcribe_audio("test_silence.wav")
        hallucination_text = ""
        if result:
            hallucination_text = result.get('text', '').strip()
            if hallucination_text != '':
                logger.warning(f"  HALLUCINATION detected: '{hallucination_text}' (Week 19.5 bug!)")
            else:
                logger.info(f"  Silence correctly transcribed as empty (OK)")

        no_hallucination = (hallucination_text == '')
        regression_tests.append({
            "test": "no_hallucinations",
            "passed": no_hallucination,
            "hallucination_text": hallucination_text
        })

        # Test 3: Consistent results (5s audio, 3 runs)
        logger.info("\nChecking for consistency (5s audio, 3 runs)...")
        transcriptions = []
        for i in range(3):
            result, error = await self.transcribe_audio("test_5s.wav")
            if result:
                text = result.get('text', '').strip()
                transcriptions.append(text)
                logger.info(f"  Run {i+1}: '{text}'")

        # Check if all transcriptions are the same (or very similar)
        unique_transcriptions = set(transcriptions)
        consistent = len(unique_transcriptions) <= 1  # All same

        regression_tests.append({
            "test": "consistent_results",
            "passed": consistent,
            "unique_transcriptions": list(unique_transcriptions),
            "total_runs": len(transcriptions)
        })

        logger.info(f"  Consistency: {'PASS' if consistent else 'FAIL'} ({len(unique_transcriptions)} unique transcriptions)")

        # Summary
        all_passed = all(t["passed"] for t in regression_tests)

        summary = {
            "test_name": "regression_checks",
            "all_passed": all_passed,
            "tests": regression_tests
        }

        logger.info(f"\nRegression Checks Summary:")
        logger.info(f"  All tests passed: {all_passed}")
        logger.info(f"  No empty transcriptions: {no_empty_bug}")
        logger.info(f"  No hallucinations: {no_hallucination}")
        logger.info(f"  Consistent results: {consistent}")

        return summary

    async def run_all_tests(self) -> Dict:
        """Run all validation tests"""
        logger.info("\n" + "="*80)
        logger.info("WEEK 19.6 VALIDATION & TESTING SUITE")
        logger.info("="*80)
        logger.info(f"Base URL: {self.base_url}")
        logger.info(f"Audio directory: {self.audio_dir}")
        logger.info(f"Results directory: {RESULTS_DIR}")

        # Check health
        logger.info("\nChecking service health...")
        is_healthy = await self.check_health()
        if not is_healthy:
            logger.error("Service health check failed! Cannot proceed with tests.")
            return {
                "status": "ERROR",
                "error": "Service health check failed"
            }

        # Run all test suites
        test_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "service_url": self.base_url,
            "week18_baseline": WEEK18_BASELINE,
        }

        # Test 1: Baseline Validation
        test_results["baseline_validation"] = await self.test_baseline_validation()

        # Test 2: Multi-Stream Reliability
        test_results["multi_stream_reliability"] = await self.test_multi_stream_reliability()

        # Test 3: Long-Form Audio
        test_results["long_form_audio"] = await self.test_long_form_audio()

        # Test 4: Regression Checks
        test_results["regression_checks"] = await self.test_regression_checks()

        # Overall summary
        baseline_passed = test_results["baseline_validation"]["passed"]
        baseline_total = test_results["baseline_validation"]["total_tests"]
        multistream_success = test_results["multi_stream_reliability"]["avg_success_rate_pct"]
        longform_working = test_results["long_form_audio"]["all_working"]
        regression_passed = test_results["regression_checks"]["all_passed"]

        test_results["overall_summary"] = {
            "baseline_tests": f"{baseline_passed}/{baseline_total} passed",
            "multi_stream_avg_success": f"{multistream_success:.1f}%",
            "long_form_working": longform_working,
            "regression_tests_passed": regression_passed,
            "week18_parity_achieved": (
                baseline_passed == baseline_total and
                multistream_success == 100.0 and
                longform_working and
                regression_passed
            )
        }

        # Save results
        results_file = RESULTS_DIR / "week19_6_baseline.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)

        # Also save multi-stream results separately
        multistream_file = RESULTS_DIR / "week19_6_multistream.json"
        with open(multistream_file, 'w') as f:
            json.dump({
                "timestamp": test_results["timestamp"],
                "results": test_results["multi_stream_reliability"]
            }, f, indent=2)

        logger.info("\n" + "="*80)
        logger.info("VALIDATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Multi-stream results saved to: {multistream_file}")
        logger.info(f"\nOverall Summary:")
        logger.info(f"  Baseline: {baseline_passed}/{baseline_total} passed")
        logger.info(f"  Multi-stream: {multistream_success:.1f}% success rate")
        logger.info(f"  Long-form: {'Working' if longform_working else 'Issues detected'}")
        logger.info(f"  Regression tests: {'All passed' if regression_passed else 'Some failed'}")
        logger.info(f"  Week 18 parity: {'ACHIEVED' if test_results['overall_summary']['week18_parity_achieved'] else 'NOT ACHIEVED'}")

        return test_results


async def main():
    """Main entry point"""
    suite = Week196ValidationSuite()
    results = await suite.run_all_tests()

    # Exit code based on results
    if results.get("status") == "ERROR":
        return 1

    week18_parity = results.get("overall_summary", {}).get("week18_parity_achieved", False)
    return 0 if week18_parity else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
