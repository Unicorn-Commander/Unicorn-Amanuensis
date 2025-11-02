#!/usr/bin/env python3
"""
Week 16: Comprehensive NPU Validation Suite

Validates end-to-end NPU pipeline performance and measures actual
400-500x realtime target achievement.

This script orchestrates all validation tests:
1. Smoke test (standalone NPU)
2. Integration tests (full pipeline)
3. Performance measurement (10x runs)
4. Report generation

Author: Week 16 Validation Team Lead
Date: November 2, 2025
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import requests

# Configuration
SERVICE_URL = "http://127.0.0.1:9050"
SERVICE_TIMEOUT = 60
TEST_AUDIO_DIR = Path(__file__).parent / "audio"
RESULTS_DIR = Path(__file__).parent / "validation_results"


@dataclass
class ValidationResults:
    """Container for all validation results"""
    timestamp: str
    smoke_test: Dict[str, Any]
    integration_tests: Dict[str, Any]
    performance_tests: Dict[str, Any]
    overall_status: str
    go_no_go: str
    summary: Dict[str, Any]


class Week16ValidationSuite:
    """
    Comprehensive validation suite for Week 16 NPU pipeline validation
    """

    def __init__(self):
        """Initialize validation suite"""
        self.results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'smoke_test': {},
            'integration_tests': {},
            'performance_tests': {},
            'overall_status': 'pending',
            'go_no_go': 'pending',
            'summary': {}
        }

        # Create results directory
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        print("="*80)
        print("  WEEK 16: NPU PIPELINE VALIDATION SUITE")
        print("="*80)
        print(f"  Date: {self.results['timestamp']}")
        print(f"  Mission: Validate 400-500x realtime performance target")
        print(f"  Service: {SERVICE_URL}")
        print(f"  Results: {RESULTS_DIR}")
        print("="*80)

    def run_smoke_test(self) -> Dict[str, Any]:
        """
        Phase 1: Smoke Test - Standalone NPU kernel execution

        Tests basic NPU functionality without full pipeline.

        Returns:
            Dictionary with smoke test results
        """
        print("\n" + "="*80)
        print("  PHASE 1: SMOKE TEST - Standalone NPU")
        print("="*80)

        smoke_test_script = Path(__file__).parent.parent / "WEEK15_NPU_SIMPLE_TEST.py"

        if not smoke_test_script.exists():
            print(f"  [ERROR] Smoke test script not found: {smoke_test_script}")
            return {
                'status': 'error',
                'error': f'Smoke test script not found: {smoke_test_script}'
            }

        print(f"  Running: {smoke_test_script.name}")

        try:
            # Run smoke test
            result = subprocess.run(
                [sys.executable, str(smoke_test_script)],
                capture_output=True,
                text=True,
                timeout=60
            )

            success = result.returncode == 0
            output = result.stdout

            # Parse output for key metrics
            metrics = self._parse_smoke_test_output(output)

            print(f"\n  Status: {'✅ PASS' if success else '❌ FAIL'}")
            if metrics:
                print(f"  Execution Time: {metrics.get('execution_time_ms', 'N/A')} ms")
                print(f"  Memory Transfer: {metrics.get('transfer_speed_gbps', 'N/A')} GB/s")

            return {
                'status': 'pass' if success else 'fail',
                'returncode': result.returncode,
                'output': output,
                'error': result.stderr if result.stderr else None,
                'metrics': metrics
            }

        except subprocess.TimeoutExpired:
            print("  [ERROR] Smoke test timeout (60s)")
            return {
                'status': 'timeout',
                'error': 'Smoke test timeout after 60s'
            }
        except Exception as e:
            print(f"  [ERROR] Smoke test failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _parse_smoke_test_output(self, output: str) -> Dict[str, Any]:
        """Parse smoke test output for metrics"""
        metrics = {}

        # Look for common patterns in output
        lines = output.split('\n')
        for line in lines:
            if 'execution time' in line.lower():
                try:
                    # Extract timing (e.g., "Execution time: 0.74 ms")
                    parts = line.split(':')
                    if len(parts) >= 2:
                        time_str = parts[1].strip().split()[0]
                        metrics['execution_time_ms'] = float(time_str)
                except:
                    pass

            if 'transfer' in line.lower() and 'gb/s' in line.lower():
                try:
                    # Extract transfer speed
                    parts = line.split(':')
                    if len(parts) >= 2:
                        speed_str = parts[1].strip().split()[0]
                        metrics['transfer_speed_gbps'] = float(speed_str)
                except:
                    pass

        return metrics

    def check_service_running(self) -> bool:
        """Check if NPU service is running"""
        print("\n  Checking service status...")
        try:
            response = requests.get(f"{SERVICE_URL}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                npu_enabled = health.get('encoder', {}).get('npu_enabled', False)
                print(f"  Service: ✅ Running (NPU: {'enabled' if npu_enabled else 'disabled'})")
                return True
            else:
                print(f"  Service: ❌ Not healthy (status {response.status_code})")
                return False
        except Exception as e:
            print(f"  Service: ❌ Not responding ({e})")
            return False

    def run_integration_tests(self) -> Dict[str, Any]:
        """
        Phase 2: Integration Tests - Full pipeline

        Runs comprehensive integration test suite.

        Returns:
            Dictionary with integration test results
        """
        print("\n" + "="*80)
        print("  PHASE 2: INTEGRATION TESTS - Full Pipeline")
        print("="*80)

        # Check service first
        if not self.check_service_running():
            print("\n  [ERROR] Service not running. Start with:")
            print("    cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis")
            print("    source ~/mlir-aie/ironenv/bin/activate")
            print("    source /opt/xilinx/xrt/setup.sh 2>/dev/null")
            print("    ENABLE_PIPELINE=true python -m uvicorn xdna2.server:app --host 127.0.0.1 --port 9050")
            return {
                'status': 'error',
                'error': 'Service not running'
            }

        integration_test_script = Path(__file__).parent / "integration_test_week15.py"

        if not integration_test_script.exists():
            print(f"  [ERROR] Integration test script not found: {integration_test_script}")
            return {
                'status': 'error',
                'error': f'Integration test script not found: {integration_test_script}'
            }

        print(f"  Running: {integration_test_script.name}")

        try:
            # Run integration tests
            result = subprocess.run(
                [sys.executable, str(integration_test_script)],
                capture_output=True,
                text=True,
                timeout=120
            )

            success = result.returncode == 0
            output = result.stdout

            # Load JSON results if available
            results_file = Path(__file__).parent / "integration_test_results.json"
            test_results = None
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        test_results = json.load(f)
                except:
                    pass

            print(f"\n  Status: {'✅ PASS' if success else '❌ FAIL'}")
            if test_results:
                print(f"  Tests Run: {test_results.get('tests_run', 0)}")
                print(f"  Tests Passed: {test_results.get('tests_passed', 0)}")
                print(f"  Tests Failed: {test_results.get('tests_failed', 0)}")

            return {
                'status': 'pass' if success else 'fail',
                'returncode': result.returncode,
                'output': output,
                'error': result.stderr if result.stderr else None,
                'test_results': test_results
            }

        except subprocess.TimeoutExpired:
            print("  [ERROR] Integration tests timeout (120s)")
            return {
                'status': 'timeout',
                'error': 'Integration tests timeout after 120s'
            }
        except Exception as e:
            print(f"  [ERROR] Integration tests failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def run_performance_tests(self, num_runs: int = 10) -> Dict[str, Any]:
        """
        Phase 3: Performance Tests - Measure realtime factor

        Runs multiple transcriptions and measures average performance.

        Args:
            num_runs: Number of test runs (default 10)

        Returns:
            Dictionary with performance test results
        """
        print("\n" + "="*80)
        print("  PHASE 3: PERFORMANCE TESTS - 400-500x Target")
        print("="*80)

        # Check service first
        if not self.check_service_running():
            return {
                'status': 'error',
                'error': 'Service not running'
            }

        test_audio = TEST_AUDIO_DIR / "test_30s.wav"
        if not test_audio.exists():
            print(f"  [ERROR] Test audio not found: {test_audio}")
            return {
                'status': 'error',
                'error': f'Test audio not found: {test_audio}'
            }

        print(f"  Test: 30-second audio transcription")
        print(f"  Runs: {num_runs}")
        print(f"  Target: 400-500x realtime")

        results = []
        for i in range(num_runs):
            try:
                with open(test_audio, 'rb') as f:
                    files = {'file': ('test_30s.wav', f, 'audio/wav')}

                    start = time.perf_counter()
                    response = requests.post(
                        f"{SERVICE_URL}/v1/audio/transcriptions",
                        files=files,
                        timeout=SERVICE_TIMEOUT
                    )
                    elapsed = time.perf_counter() - start

                if response.status_code == 200:
                    result = response.json()
                    perf = result.get('performance', {})

                    run_result = {
                        'run': i + 1,
                        'status': 'success',
                        'elapsed_s': elapsed,
                        'realtime_factor': perf.get('realtime_factor', 0),
                        'processing_time_s': perf.get('processing_time_s', elapsed),
                        'encoder_time_ms': perf.get('encoder_time_ms', 0),
                        'audio_duration_s': perf.get('audio_duration_s', 30.0)
                    }

                    print(f"  Run {i+1:2d}: {run_result['realtime_factor']:.1f}x realtime, "
                          f"encoder: {run_result['encoder_time_ms']:.1f}ms")
                else:
                    run_result = {
                        'run': i + 1,
                        'status': 'error',
                        'error': f"HTTP {response.status_code}",
                        'elapsed_s': elapsed
                    }
                    print(f"  Run {i+1:2d}: ❌ ERROR ({response.status_code})")

                results.append(run_result)

            except Exception as e:
                run_result = {
                    'run': i + 1,
                    'status': 'error',
                    'error': str(e)
                }
                print(f"  Run {i+1:2d}: ❌ ERROR ({e})")
                results.append(run_result)

        # Calculate statistics
        successful_runs = [r for r in results if r['status'] == 'success']

        if successful_runs:
            realtime_factors = [r['realtime_factor'] for r in successful_runs]
            encoder_times = [r['encoder_time_ms'] for r in successful_runs]

            import numpy as np
            stats = {
                'num_runs': len(results),
                'successful_runs': len(successful_runs),
                'failed_runs': len(results) - len(successful_runs),
                'mean_realtime_factor': float(np.mean(realtime_factors)),
                'median_realtime_factor': float(np.median(realtime_factors)),
                'std_realtime_factor': float(np.std(realtime_factors)),
                'min_realtime_factor': float(np.min(realtime_factors)),
                'max_realtime_factor': float(np.max(realtime_factors)),
                'mean_encoder_time_ms': float(np.mean(encoder_times)),
                'target_achieved': float(np.mean(realtime_factors)) >= 400,
                'target_percentage': (float(np.mean(realtime_factors)) / 400) * 100
            }

            print(f"\n  Performance Summary:")
            print(f"    Mean Realtime Factor: {stats['mean_realtime_factor']:.1f}x")
            print(f"    Median Realtime Factor: {stats['median_realtime_factor']:.1f}x")
            print(f"    Std Deviation: {stats['std_realtime_factor']:.1f}x")
            print(f"    Target (400x): {stats['target_percentage']:.1f}%")

            if stats['target_achieved']:
                print(f"    Verdict: ✅ TARGET ACHIEVED!")
            elif stats['mean_realtime_factor'] >= 1:
                print(f"    Verdict: 🟡 Faster than realtime, below target")
            else:
                print(f"    Verdict: ❌ Slower than realtime")

            return {
                'status': 'complete',
                'runs': results,
                'statistics': stats
            }
        else:
            print(f"\n  [ERROR] No successful runs")
            return {
                'status': 'error',
                'error': 'No successful runs',
                'runs': results
            }

    def generate_report(self) -> str:
        """
        Phase 4: Generate validation report

        Creates markdown report with all validation results.

        Returns:
            Path to generated report
        """
        print("\n" + "="*80)
        print("  PHASE 4: REPORT GENERATION")
        print("="*80)

        # Determine overall status
        smoke_ok = self.results['smoke_test'].get('status') == 'pass'
        integration_ok = self.results['integration_tests'].get('status') == 'pass'
        performance_ok = self.results['performance_tests'].get('status') == 'complete'

        self.results['overall_status'] = 'success' if (smoke_ok and integration_ok and performance_ok) else 'partial'

        # Determine go/no-go
        if smoke_ok and integration_ok:
            stats = self.results['performance_tests'].get('statistics', {})
            mean_rt = stats.get('mean_realtime_factor', 0)

            if mean_rt >= 1:
                self.results['go_no_go'] = 'GO'
            else:
                self.results['go_no_go'] = 'NO-GO'
        else:
            self.results['go_no_go'] = 'NO-GO'

        # Save JSON results
        json_file = RESULTS_DIR / f"validation_results_{int(time.time())}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"  JSON Results: {json_file}")

        # Generate markdown report
        report_file = RESULTS_DIR / f"VALIDATION_REPORT_{int(time.time())}.md"
        self._write_markdown_report(report_file)

        print(f"  Markdown Report: {report_file}")

        return str(report_file)

    def _write_markdown_report(self, filepath: Path):
        """Write markdown validation report"""

        report = f"""# Week 16: NPU Pipeline Validation Report

**Date**: {self.results['timestamp']}
**Overall Status**: {'✅ SUCCESS' if self.results['overall_status'] == 'success' else '🟡 PARTIAL SUCCESS'}
**Go/No-Go for Week 17**: **{self.results['go_no_go']}**
**Validation Lead**: Validation Team Lead

---

## Executive Summary

"""

        # Add smoke test summary
        smoke = self.results['smoke_test']
        if smoke.get('status') == 'pass':
            report += "✅ **Smoke Test**: PASSED - NPU kernel execution working\n"
        else:
            report += f"❌ **Smoke Test**: FAILED - {smoke.get('error', 'Unknown error')}\n"

        # Add integration test summary
        integration = self.results['integration_tests']
        if integration.get('status') == 'pass':
            test_res = integration.get('test_results', {})
            passed = test_res.get('tests_passed', 0)
            total = test_res.get('tests_run', 0)
            report += f"✅ **Integration Tests**: {passed}/{total} PASSED\n"
        else:
            report += f"❌ **Integration Tests**: FAILED - {integration.get('error', 'Unknown error')}\n"

        # Add performance summary
        performance = self.results['performance_tests']
        if performance.get('status') == 'complete':
            stats = performance.get('statistics', {})
            mean_rt = stats.get('mean_realtime_factor', 0)
            target_pct = stats.get('target_percentage', 0)
            report += f"{'✅' if stats.get('target_achieved') else '🟡'} **Performance**: {mean_rt:.1f}x realtime ({target_pct:.1f}% of 400x target)\n"
        else:
            report += f"❌ **Performance**: FAILED - {performance.get('error', 'Unknown error')}\n"

        report += f"""

---

## Full Results

See attached JSON file for complete details:
`{RESULTS_DIR}/validation_results_*.json`

---

**Built with 🦄 by Magic Unicorn Unconventional Technology & Stuff Inc**
"""

        with open(filepath, 'w') as f:
            f.write(report)

    def run_full_validation(self) -> int:
        """
        Run complete validation suite

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        print("\n  Starting full validation suite...")
        print(f"  Timestamp: {self.results['timestamp']}\n")

        # Phase 1: Smoke test
        self.results['smoke_test'] = self.run_smoke_test()

        # Phase 2: Integration tests
        if self.results['smoke_test'].get('status') == 'pass':
            self.results['integration_tests'] = self.run_integration_tests()
        else:
            print("\n  [SKIP] Integration tests (smoke test failed)")
            self.results['integration_tests'] = {
                'status': 'skipped',
                'reason': 'Smoke test failed'
            }

        # Phase 3: Performance tests
        if self.results['integration_tests'].get('status') == 'pass':
            self.results['performance_tests'] = self.run_performance_tests(num_runs=10)
        else:
            print("\n  [SKIP] Performance tests (integration tests failed)")
            self.results['performance_tests'] = {
                'status': 'skipped',
                'reason': 'Integration tests failed'
            }

        # Phase 4: Generate report
        report_path = self.generate_report()

        # Print final summary
        print("\n" + "="*80)
        print("  VALIDATION COMPLETE")
        print("="*80)
        print(f"  Overall Status: {self.results['overall_status'].upper()}")
        print(f"  Go/No-Go: {self.results['go_no_go']}")
        print(f"  Report: {report_path}")
        print("="*80)

        return 0 if self.results['go_no_go'] == 'GO' else 1


def main():
    """Main entry point"""
    try:
        suite = Week16ValidationSuite()
        exit_code = suite.run_full_validation()
        return exit_code

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n[ERROR] Validation suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
