"""
Unit Tests for ComponentTimer

Tests for the hierarchical timing system used in Week 19.6 timing instrumentation.

Coverage:
- Basic timing operations
- Hierarchical timing
- Statistical calculations
- Thread safety
- Overhead measurement
- JSON export
- Disabled timer mode

Author: CC-1L Week 19.6 Team 2 (Timing Instrumentation)
Date: November 2, 2025
Status: Production Ready
"""

import unittest
import time
import threading
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from xdna2.component_timer import ComponentTimer, GlobalTimingManager


class TestComponentTimer(unittest.TestCase):
    """Test cases for ComponentTimer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.timer = ComponentTimer(enabled=True)

    def test_basic_timing(self):
        """Test basic timing operation"""
        with self.timer.time('test'):
            time.sleep(0.01)  # Sleep 10ms

        breakdown = self.timer.get_breakdown()

        # Should have exactly one component
        self.assertEqual(len(breakdown), 1)
        self.assertIn('test', breakdown)

        # Should have recorded time close to 10ms
        self.assertGreater(breakdown['test']['mean'], 8.0)  # At least 8ms
        self.assertLess(breakdown['test']['mean'], 15.0)    # At most 15ms

        # Should have count of 1
        self.assertEqual(breakdown['test']['count'], 1)

    def test_hierarchical_timing(self):
        """Test hierarchical timing with nested components"""
        with self.timer.time('total'):
            time.sleep(0.005)  # 5ms

            with self.timer.time('stage1'):
                time.sleep(0.003)  # 3ms

            with self.timer.time('stage2'):
                time.sleep(0.002)  # 2ms

        breakdown = self.timer.get_breakdown()

        # Should have 3 components
        self.assertEqual(len(breakdown), 3)
        self.assertIn('total', breakdown)
        self.assertIn('total.stage1', breakdown)
        self.assertIn('total.stage2', breakdown)

        # Total should be sum of sleep times (~10ms)
        self.assertGreater(breakdown['total']['mean'], 8.0)

        # Stage1 should be ~3ms
        self.assertGreater(breakdown['total.stage1']['mean'], 2.0)
        self.assertLess(breakdown['total.stage1']['mean'], 5.0)

        # Stage2 should be ~2ms
        self.assertGreater(breakdown['total.stage2']['mean'], 1.0)
        self.assertLess(breakdown['total.stage2']['mean'], 4.0)

    def test_multiple_samples(self):
        """Test timing with multiple samples for statistics"""
        # Record 20 samples
        for i in range(20):
            with self.timer.time('repeated'):
                time.sleep(0.001)  # 1ms

        breakdown = self.timer.get_breakdown()

        # Should have 20 samples
        self.assertEqual(breakdown['repeated']['count'], 20)

        # All statistics should be present
        self.assertIn('mean', breakdown['repeated'])
        self.assertIn('p50', breakdown['repeated'])
        self.assertIn('p95', breakdown['repeated'])
        self.assertIn('p99', breakdown['repeated'])
        self.assertIn('min', breakdown['repeated'])
        self.assertIn('max', breakdown['repeated'])

        # Mean should be ~1ms
        self.assertGreater(breakdown['repeated']['mean'], 0.5)
        self.assertLess(breakdown['repeated']['mean'], 2.0)

    def test_percentile_calculation(self):
        """Test percentile calculations with sufficient samples"""
        # Create data with known distribution
        for i in range(100):
            with self.timer.time('percentile_test'):
                # Sleep 0-10ms (linear distribution)
                time.sleep(i * 0.0001)

        breakdown = self.timer.get_breakdown()

        # P50 should be around 4.5ms (middle of 0-9ms range)
        p50 = breakdown['percentile_test']['p50']
        self.assertGreater(p50, 3.0)
        self.assertLess(p50, 6.0)

        # P95 should be higher than P50
        p95 = breakdown['percentile_test']['p95']
        self.assertGreater(p95, p50)

        # P99 should be higher than P95
        p99 = breakdown['percentile_test']['p99']
        self.assertGreater(p99, p95)

    def test_disabled_timer(self):
        """Test that disabled timer has minimal overhead"""
        disabled_timer = ComponentTimer(enabled=False)

        # Time something with disabled timer
        start = time.perf_counter()
        for i in range(1000):
            with disabled_timer.time('disabled'):
                pass  # Empty block
        elapsed = time.perf_counter() - start

        # Should complete very quickly (<1ms for 1000 iterations)
        self.assertLess(elapsed * 1000, 1.0)

        # Should have no timing data
        breakdown = disabled_timer.get_breakdown()
        self.assertEqual(len(breakdown), 0)

    def test_reset(self):
        """Test reset functionality"""
        # Record some data
        with self.timer.time('test'):
            time.sleep(0.001)

        # Verify data exists
        breakdown = self.timer.get_breakdown()
        self.assertEqual(len(breakdown), 1)

        # Reset
        self.timer.reset()

        # Verify data is cleared
        breakdown = self.timer.get_breakdown()
        self.assertEqual(len(breakdown), 0)

    def test_json_export(self):
        """Test JSON serialization"""
        with self.timer.time('json_test'):
            time.sleep(0.001)

        # Get JSON
        json_data = self.timer.get_json()

        # Should be serializable
        json_str = json.dumps(json_data)
        self.assertIsInstance(json_str, str)

        # Should be deserializable
        parsed = json.loads(json_str)
        self.assertIn('json_test', parsed)
        self.assertIn('mean', parsed['json_test'])

    def test_overhead_measurement(self):
        """Test overhead measurement functionality"""
        # Record some data
        for i in range(100):
            with self.timer.time('overhead_test'):
                pass  # Empty block

        # Get overhead estimate
        overhead = self.timer.get_overhead_estimate()

        # Should have expected keys
        self.assertIn('per_measurement_us', overhead)
        self.assertIn('total_measurements', overhead)
        self.assertIn('total_overhead_ms', overhead)

        # Per-measurement should be low (<10μs)
        self.assertLess(overhead['per_measurement_us'], 10.0)

        # Total overhead should be <5ms for 100 measurements
        self.assertLess(overhead['total_overhead_ms'], 5.0)

    def test_thread_safety(self):
        """Test thread safety of timing operations"""
        num_threads = 10
        iterations_per_thread = 50

        def worker():
            for i in range(iterations_per_thread):
                with self.timer.time('threaded'):
                    time.sleep(0.0001)  # 0.1ms

        # Create and start threads
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Should have recorded all samples
        breakdown = self.timer.get_breakdown()
        expected_count = num_threads * iterations_per_thread
        self.assertEqual(breakdown['threaded']['count'], expected_count)

    def test_exception_handling(self):
        """Test that timing works correctly even with exceptions"""
        try:
            with self.timer.time('exception_test'):
                time.sleep(0.001)
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # Timing should still be recorded
        breakdown = self.timer.get_breakdown()
        self.assertEqual(len(breakdown), 1)
        self.assertIn('exception_test', breakdown)
        self.assertEqual(breakdown['exception_test']['count'], 1)

    def test_get_summary(self):
        """Test get_summary method"""
        with self.timer.time('summary_test'):
            time.sleep(0.001)

        # Get summary for specific component
        summary = self.timer.get_summary('summary_test')
        self.assertIn('mean', summary)
        self.assertIn('count', summary)

        # Get summary for all components
        all_summary = self.timer.get_summary()
        self.assertIn('summary_test', all_summary)

    def test_print_summary(self):
        """Test print_summary doesn't crash"""
        with self.timer.time('print_test'):
            time.sleep(0.001)

        # Should not raise exception
        try:
            self.timer.print_summary()
        except Exception as e:
            self.fail(f"print_summary raised exception: {e}")


class TestGlobalTimingManager(unittest.TestCase):
    """Test cases for GlobalTimingManager"""

    def test_singleton(self):
        """Test singleton pattern"""
        manager1 = GlobalTimingManager.instance()
        manager2 = GlobalTimingManager.instance()

        # Should be same instance
        self.assertIs(manager1, manager2)

    def test_global_timer(self):
        """Test global timer access"""
        timer = GlobalTimingManager.get_timer()

        # Should be a ComponentTimer instance
        self.assertIsInstance(timer, ComponentTimer)

    def test_enable_disable(self):
        """Test global enable/disable"""
        # Enable
        GlobalTimingManager.enable()
        self.assertTrue(GlobalTimingManager.is_enabled())

        # Disable
        GlobalTimingManager.disable()
        self.assertFalse(GlobalTimingManager.is_enabled())

        # Re-enable for other tests
        GlobalTimingManager.enable()

    def test_global_reset(self):
        """Test global reset"""
        timer = GlobalTimingManager.get_timer()

        # Record some data
        with timer.time('global_test'):
            time.sleep(0.001)

        # Reset
        GlobalTimingManager.reset()

        # Should be cleared
        breakdown = timer.get_breakdown()
        self.assertEqual(len(breakdown), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests simulating real pipeline usage"""

    def test_pipeline_simulation(self):
        """Simulate transcription pipeline timing"""
        timer = ComponentTimer(enabled=True)

        # Simulate 10 requests through 3-stage pipeline
        for request_id in range(10):
            # Stage 1: Load + Mel
            with timer.time('stage1'):
                with timer.time('audio_loading'):
                    time.sleep(0.005)  # 5ms
                with timer.time('mel_computation'):
                    time.sleep(0.003)  # 3ms

            # Stage 2: Encoder
            with timer.time('stage2'):
                with timer.time('conv1d_preprocessing'):
                    time.sleep(0.001)  # 1ms
                with timer.time('npu_encoder'):
                    time.sleep(0.002)  # 2ms

            # Stage 3: Decoder + Alignment
            with timer.time('stage3'):
                with timer.time('decoder'):
                    time.sleep(0.010)  # 10ms
                with timer.time('alignment'):
                    time.sleep(0.005)  # 5ms
                with timer.time('postprocessing'):
                    time.sleep(0.001)  # 1ms

        # Get breakdown
        breakdown = timer.get_breakdown()

        # Should have all expected components (correct hierarchical paths)
        expected_components = [
            'stage1',
            'stage1.audio_loading',
            'stage1.mel_computation',
            'stage2',
            'stage2.conv1d_preprocessing',
            'stage2.npu_encoder',
            'stage3',
            'stage3.decoder',
            'stage3.alignment',
            'stage3.postprocessing'
        ]

        for component in expected_components:
            self.assertIn(component, breakdown, f"Missing component: {component}")
            self.assertEqual(breakdown[component]['count'], 10, f"Wrong count for {component}")

        # Print summary for visual inspection
        print("\n" + "="*80)
        print("INTEGRATION TEST: Pipeline Simulation Results")
        print("="*80)
        timer.print_summary()

    def test_overhead_validation(self):
        """Validate that total overhead is <5ms"""
        timer = ComponentTimer(enabled=True)

        # Simulate realistic workload
        for i in range(100):
            with timer.time('workload'):
                with timer.time('sub1'):
                    pass
                with timer.time('sub2'):
                    pass
                with timer.time('sub3'):
                    pass

        # Measure overhead
        overhead = timer.get_overhead_estimate()

        # Total overhead should be <5ms
        self.assertLess(
            overhead['total_overhead_ms'],
            5.0,
            f"Overhead {overhead['total_overhead_ms']:.3f}ms exceeds 5ms target"
        )

        print(f"\n✅ Overhead validation: {overhead['total_overhead_ms']:.3f}ms (target: <5ms)")


def run_tests():
    """Run all tests and display results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestComponentTimer))
    suite.addTests(loader.loadTestsFromTestCase(TestGlobalTimingManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")

    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(run_tests())
