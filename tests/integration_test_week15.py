#!/usr/bin/env python3
"""
Week 15 Integration Test Suite - NPU Audio Transcription Pipeline

Tests end-to-end audio transcription with NPU acceleration.
Documents current integration status and validates system components.

Author: CC-1L Integration Testing Team Lead
Date: November 2, 2025
Status: Week 15 Testing
"""

import requests
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List

# Service configuration
SERVICE_URL = "http://127.0.0.1:9050"
TEST_AUDIO_DIR = Path(__file__).parent / "audio"

# Test audio files
TEST_CASES = [
    {"file": "test_1s.wav", "expected_duration": 1.0, "description": "1 second audio"},
    {"file": "test_5s.wav", "expected_duration": 5.0, "description": "5 second audio"},
    {"file": "test_30s.wav", "expected_duration": 30.0, "description": "30 second audio"},
    {"file": "test_silence.wav", "expected_duration": 5.0, "description": "Silent audio (edge case)"},
]


class IntegrationTestResults:
    """Track integration test results"""

    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
        self.service_status = None
        self.npu_status = None

    def add_result(self, test_name: str, passed: bool, details: Dict[str, Any]):
        """Add a test result"""
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1

        self.test_results.append({
            "test_name": test_name,
            "passed": passed,
            "details": details
        })

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("  WEEK 15 INTEGRATION TEST SUMMARY")
        print("="*80)
        print(f"  Tests Run: {self.tests_run}")
        print(f"  Tests Passed: {self.tests_passed}")
        print(f"  Tests Failed: {self.tests_failed}")
        print(f"  Success Rate: {self.tests_passed/self.tests_run*100:.1f}%")
        print("="*80)

        print("\n  Service Status:")
        if self.service_status:
            print(f"    Status: {self.service_status.get('status')}")
            print(f"    NPU Enabled: {self.service_status.get('encoder', {}).get('npu_enabled')}")
            print(f"    Weights Loaded: {self.service_status.get('encoder', {}).get('weights_loaded')}")

        print("\n  Test Results:")
        for result in self.test_results:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"    {status}: {result['test_name']}")
            if not result["passed"]:
                print(f"      Error: {result['details'].get('error')}")

        print("\n" + "="*80)


def test_service_health() -> Dict[str, Any]:
    """Test 1: Service health check"""
    print("\n[Test 1] Service Health Check...")
    try:
        response = requests.get(f"{SERVICE_URL}/health", timeout=5)
        response.raise_for_status()
        health = response.json()

        print(f"  Service: {health.get('service')}")
        print(f"  Status: {health.get('status')}")
        print(f"  NPU Enabled: {health.get('encoder', {}).get('npu_enabled')}")
        print(f"  Weights Loaded: {health.get('encoder', {}).get('weights_loaded')}")

        return {
            "passed": health.get("status") == "healthy",
            "health": health
        }
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {
            "passed": False,
            "error": str(e)
        }


def test_transcription(audio_file: str, expected_duration: float) -> Dict[str, Any]:
    """Test transcription with audio file"""
    print(f"\n[Test] Transcription: {audio_file}...")

    audio_path = TEST_AUDIO_DIR / audio_file
    if not audio_path.exists():
        return {
            "passed": False,
            "error": f"Audio file not found: {audio_path}"
        }

    try:
        with open(audio_path, 'rb') as f:
            files = {'file': (audio_file, f, 'audio/wav')}

            start_time = time.perf_counter()
            response = requests.post(
                f"{SERVICE_URL}/v1/audio/transcriptions",
                files=files,
                timeout=60
            )
            elapsed_time = time.perf_counter() - start_time

        if response.status_code == 200:
            result = response.json()
            transcription = result.get('text', '')
            performance = result.get('performance', {})

            audio_duration = performance.get('audio_duration_s', expected_duration)
            processing_time = performance.get('processing_time_s', elapsed_time)
            realtime_factor = performance.get('realtime_factor', audio_duration / processing_time if processing_time > 0 else 0)

            print(f"  ✅ Transcription: \"{transcription[:100]}...\"")
            print(f"  Audio Duration: {audio_duration:.2f}s")
            print(f"  Processing Time: {processing_time*1000:.1f}ms")
            print(f"  Realtime Factor: {realtime_factor:.1f}x")

            return {
                "passed": True,
                "transcription": transcription,
                "audio_duration": audio_duration,
                "processing_time": processing_time,
                "realtime_factor": realtime_factor,
                "performance": performance
            }
        else:
            error_data = response.json()
            error_msg = error_data.get('error', 'Unknown error')
            details = error_data.get('details', '')

            print(f"  ❌ Error ({response.status_code}): {error_msg}")
            if details:
                print(f"     Details: {details}")

            return {
                "passed": False,
                "error": f"{error_msg}: {details}",
                "status_code": response.status_code
            }

    except requests.exceptions.Timeout:
        print(f"  ❌ Timeout after 60s")
        return {
            "passed": False,
            "error": "Request timeout after 60s"
        }
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {
            "passed": False,
            "error": str(e)
        }


def main():
    """Run integration tests"""
    print("="*80)
    print("  WEEK 15 INTEGRATION TEST - NPU Audio Transcription Pipeline")
    print("="*80)
    print(f"  Service URL: {SERVICE_URL}")
    print(f"  Test Audio: {TEST_AUDIO_DIR}")
    print("="*80)

    results = IntegrationTestResults()

    # Test 1: Service Health
    health_result = test_service_health()
    results.add_result("Service Health Check", health_result["passed"], health_result)
    results.service_status = health_result.get("health")

    if not health_result["passed"]:
        print("\n❌ Service health check failed. Cannot continue tests.")
        results.print_summary()
        return 1

    # Get NPU status
    npu_enabled = health_result.get("health", {}).get("encoder", {}).get("npu_enabled", False)
    results.npu_status = npu_enabled

    # Test 2-5: Transcription with different audio lengths
    for test_case in TEST_CASES:
        test_name = f"Transcription: {test_case['description']}"
        result = test_transcription(test_case["file"], test_case["expected_duration"])
        results.add_result(test_name, result["passed"], result)

    # Print summary
    results.print_summary()

    # Save results to JSON
    output_file = Path(__file__).parent / "integration_test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "service_status": results.service_status,
            "npu_enabled": results.npu_status,
            "tests_run": results.tests_run,
            "tests_passed": results.tests_passed,
            "tests_failed": results.tests_failed,
            "test_results": results.test_results
        }, f, indent=2)

    print(f"\n  Results saved to: {output_file}")

    # Return exit code
    return 0 if results.tests_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
