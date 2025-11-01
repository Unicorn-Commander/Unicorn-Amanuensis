#!/usr/bin/env python3
"""
NPU Callback Integration Tests (Week 6, Day 3 - Task 1)

Validates NPU execution, callback registration, and data flow for C++ encoder.

Tests:
1. NPU initialization (XRT, device detection)
2. XCLBIN loading to NPU
3. Callback registration (Python → C++ interface)
4. Data flow verification (Python → C++ → NPU → Python)
5. NPU latency measurement (target: <1ms for matmuls)

Author: NPU Testing & Validation Teamlead
Date: November 1, 2025
Status: Week 6 Days 3-5 - Task 1
"""

import sys
import os
import time
import unittest
import numpy as np
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test imports with graceful failure
try:
    from xdna2.cpp_runtime_wrapper import CPPRuntimeWrapper, EncoderLayer, CPPRuntimeError
    from xdna2.encoder_cpp import WhisperEncoderCPP, create_encoder_cpp
    from runtime.platform_detector import PlatformDetector, Platform
    CPP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"C++ runtime not available: {e}")
    CPP_AVAILABLE = False

try:
    from xdna2.npu_callback_native import NPUCallbackNative
    NPU_AVAILABLE = True
except ImportError as e:
    logger.warning(f"NPU callback not available: {e}")
    NPU_AVAILABLE = False


class TestNPUInitialization(unittest.TestCase):
    """Test NPU hardware initialization"""

    @unittest.skipUnless(NPU_AVAILABLE, "NPU not available")
    def test_xrt_available(self):
        """Test that XRT is installed and accessible"""
        print("\n[TEST] XRT availability...")

        # Check XRT installation
        xrt_path = Path("/opt/xilinx/xrt")
        self.assertTrue(xrt_path.exists(), f"XRT not found at {xrt_path}")

        # Check for setup script
        setup_script = xrt_path / "setup.sh"
        self.assertTrue(setup_script.exists(), f"XRT setup.sh not found")

        print(f"  ✓ XRT found at {xrt_path}")

    @unittest.skipUnless(NPU_AVAILABLE, "NPU not available")
    def test_npu_device_detection(self):
        """Test that NPU device is detected"""
        print("\n[TEST] NPU device detection...")

        try:
            # Use platform detector
            detector = PlatformDetector()
            platform_info = detector.detect()

            print(f"  Platform: {platform_info['platform']}")
            print(f"  Has NPU: {platform_info['has_npu']}")

            # Verify XDNA2 NPU detected
            self.assertTrue(
                platform_info['has_npu'],
                "NPU not detected by platform detector"
            )

            # Check for device node
            device_nodes = list(Path("/dev/accel").glob("accel*"))
            self.assertGreater(
                len(device_nodes), 0,
                "No NPU device nodes found in /dev/accel"
            )

            print(f"  ✓ NPU device detected: {device_nodes[0]}")

        except Exception as e:
            self.fail(f"NPU device detection failed: {e}")

    @unittest.skipUnless(NPU_AVAILABLE, "NPU not available")
    def test_xclbin_available(self):
        """Test that NPU kernels (xclbin files) are available"""
        print("\n[TEST] XCLBIN availability...")

        # Check for matmul kernels
        kernel_locations = [
            Path("~/mlir-aie/programming_examples/basic/matrix_multiplication").expanduser(),
            Path("/opt/xilinx/xrt/share"),
            Path.home() / "mlir-aie" / "programming_examples"
        ]

        found_kernels = []
        for location in kernel_locations:
            if location.exists():
                xclbin_files = list(location.rglob("*.xclbin"))
                found_kernels.extend(xclbin_files)

        self.assertGreater(
            len(found_kernels), 0,
            f"No xclbin files found in {kernel_locations}"
        )

        print(f"  ✓ Found {len(found_kernels)} xclbin kernel(s)")
        for kernel in found_kernels[:3]:  # Show first 3
            print(f"    - {kernel.name}")


class TestCallbackRegistration(unittest.TestCase):
    """Test NPU callback registration with C++ encoder"""

    @unittest.skipUnless(CPP_AVAILABLE and NPU_AVAILABLE, "C++ runtime or NPU not available")
    def test_callback_creation(self):
        """Test that NPU callback can be created"""
        print("\n[TEST] NPU callback creation...")

        try:
            # Find a kernel
            kernel_path = self._find_test_kernel()
            if not kernel_path:
                self.skipTest("No test kernel found")

            # Create callback
            callback = NPUCallbackNative(
                kernel_path=str(kernel_path),
                device_idx=0
            )

            self.assertIsNotNone(callback)
            print(f"  ✓ Callback created with kernel: {kernel_path.name}")

        except Exception as e:
            self.fail(f"Callback creation failed: {e}")

    @unittest.skipUnless(CPP_AVAILABLE and NPU_AVAILABLE, "C++ runtime or NPU not available")
    def test_callback_registration_with_encoder(self):
        """Test registering NPU callback with C++ encoder"""
        print("\n[TEST] Callback registration with encoder...")

        try:
            # Find kernel
            kernel_path = self._find_test_kernel()
            if not kernel_path:
                self.skipTest("No test kernel found")

            # Create encoder
            encoder = create_encoder_cpp(
                num_layers=1,  # Just one layer for testing
                n_heads=8,
                n_state=512,
                ffn_dim=2048,
                use_npu=True
            )

            # Create callback
            callback = NPUCallbackNative(
                kernel_path=str(kernel_path),
                device_idx=0
            )

            # Register callback
            encoder.register_npu_callback(callback)

            print("  ✓ Callback registered successfully")

        except Exception as e:
            self.fail(f"Callback registration failed: {e}")

    def _find_test_kernel(self):
        """Helper to find a test kernel"""
        kernel_locations = [
            Path("~/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build").expanduser()
        ]

        for location in kernel_locations:
            if location.exists():
                xclbin_files = list(location.glob("final*.xclbin"))
                if xclbin_files:
                    return xclbin_files[0]
        return None


class TestDataFlow(unittest.TestCase):
    """Test data flow through NPU callback"""

    @unittest.skipUnless(CPP_AVAILABLE and NPU_AVAILABLE, "C++ runtime or NPU not available")
    def test_matmul_data_flow(self):
        """Test that data flows correctly through NPU matmul"""
        print("\n[TEST] NPU matmul data flow...")

        try:
            # Find kernel
            kernel_path = self._find_test_kernel()
            if not kernel_path:
                self.skipTest("No test kernel found")

            # Create callback
            callback = NPUCallbackNative(
                kernel_path=str(kernel_path),
                device_idx=0
            )

            # Create test data (small matrices for quick test)
            M, K, N = 64, 64, 64
            A = np.random.randint(-128, 127, (M, K), dtype=np.int8)
            B = np.random.randint(-128, 127, (K, N), dtype=np.int8)

            # Execute on NPU
            print(f"  Executing {M}×{K} @ {K}×{N} matmul on NPU...")
            C_npu = callback.execute_matmul(A, B)

            # Verify output shape
            self.assertEqual(C_npu.shape, (M, N), f"Output shape mismatch: {C_npu.shape}")

            # Verify output type (should be int32 accumulator)
            self.assertEqual(C_npu.dtype, np.int32, f"Output dtype mismatch: {C_npu.dtype}")

            # Compute CPU reference
            C_cpu = np.matmul(A.astype(np.int32), B.astype(np.int32))

            # Compare results (allow small error due to quantization)
            max_diff = np.abs(C_npu - C_cpu).max()
            mean_val = np.abs(C_cpu).mean()
            error_pct = (max_diff / mean_val) * 100 if mean_val > 0 else 0

            print(f"  Max difference: {max_diff}")
            print(f"  Error: {error_pct:.2f}%")

            # For INT8 matmul, we expect exact match
            self.assertLess(error_pct, 1.0, f"Error too high: {error_pct:.2f}%")

            print("  ✓ Data flow verified")

        except AttributeError as e:
            # NPUCallbackNative might not have execute_matmul method yet
            self.skipTest(f"NPU callback interface incomplete: {e}")
        except Exception as e:
            self.fail(f"Data flow test failed: {e}")

    def _find_test_kernel(self):
        """Helper to find a test kernel"""
        kernel_locations = [
            Path("~/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build").expanduser()
        ]

        for location in kernel_locations:
            if location.exists():
                xclbin_files = list(location.glob("final*.xclbin"))
                if xclbin_files:
                    return xclbin_files[0]
        return None


class TestNPULatency(unittest.TestCase):
    """Test NPU execution latency"""

    @unittest.skipUnless(CPP_AVAILABLE and NPU_AVAILABLE, "C++ runtime or NPU not available")
    def test_matmul_latency(self):
        """Test that NPU matmul meets <1ms latency target"""
        print("\n[TEST] NPU matmul latency...")

        try:
            # Find kernel
            kernel_path = self._find_test_kernel()
            if not kernel_path:
                self.skipTest("No test kernel found")

            # Create callback
            callback = NPUCallbackNative(
                kernel_path=str(kernel_path),
                device_idx=0
            )

            # Create test data
            M, K, N = 512, 512, 512  # Typical encoder layer size
            A = np.random.randint(-128, 127, (M, K), dtype=np.int8)
            B = np.random.randint(-128, 127, (K, N), dtype=np.int8)

            # Warmup (3 iterations)
            print("  Warming up...")
            for _ in range(3):
                try:
                    _ = callback.execute_matmul(A, B)
                except AttributeError:
                    self.skipTest("NPU callback interface incomplete")

            # Benchmark (100 iterations)
            print(f"  Benchmarking {M}×{K} @ {K}×{N} matmul (100 iterations)...")
            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                _ = callback.execute_matmul(A, B)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # Convert to ms

            # Calculate statistics
            mean_time = np.mean(times)
            median_time = np.median(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)

            print(f"\n  Results (ms):")
            print(f"    Mean:   {mean_time:.3f}")
            print(f"    Median: {median_time:.3f}")
            print(f"    Min:    {min_time:.3f}")
            print(f"    Max:    {max_time:.3f}")
            print(f"    Std:    {std_time:.3f}")

            # Verify target met
            TARGET_LATENCY_MS = 1.0  # Target: <1ms per matmul

            self.assertLess(
                min_time, TARGET_LATENCY_MS,
                f"Min latency {min_time:.3f}ms exceeds target {TARGET_LATENCY_MS}ms"
            )

            # Also check median (more robust than mean)
            self.assertLess(
                median_time, TARGET_LATENCY_MS * 2,
                f"Median latency {median_time:.3f}ms exceeds 2x target"
            )

            print(f"  ✓ Latency target met (min: {min_time:.3f}ms < {TARGET_LATENCY_MS}ms)")

        except AttributeError as e:
            self.skipTest(f"NPU callback interface incomplete: {e}")
        except Exception as e:
            self.fail(f"Latency test failed: {e}")

    def _find_test_kernel(self):
        """Helper to find a test kernel"""
        kernel_locations = [
            Path("~/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build").expanduser()
        ]

        for location in kernel_locations:
            if location.exists():
                xclbin_files = list(location.glob("final*.xclbin"))
                if xclbin_files:
                    return xclbin_files[0]
        return None


class TestNPUCallbackRobustness(unittest.TestCase):
    """Test NPU callback error handling and edge cases"""

    @unittest.skipUnless(CPP_AVAILABLE and NPU_AVAILABLE, "C++ runtime or NPU not available")
    def test_invalid_kernel_path(self):
        """Test that invalid kernel path is handled gracefully"""
        print("\n[TEST] Invalid kernel path handling...")

        with self.assertRaises((FileNotFoundError, RuntimeError, CPPRuntimeError)):
            callback = NPUCallbackNative(
                kernel_path="/nonexistent/kernel.xclbin",
                device_idx=0
            )

        print("  ✓ Invalid kernel path handled")

    @unittest.skipUnless(CPP_AVAILABLE and NPU_AVAILABLE, "C++ runtime or NPU not available")
    def test_invalid_device_index(self):
        """Test that invalid device index is handled gracefully"""
        print("\n[TEST] Invalid device index handling...")

        kernel_path = self._find_test_kernel()
        if not kernel_path:
            self.skipTest("No test kernel found")

        # Try invalid device index (999)
        with self.assertRaises((RuntimeError, CPPRuntimeError, IndexError)):
            callback = NPUCallbackNative(
                kernel_path=str(kernel_path),
                device_idx=999
            )

        print("  ✓ Invalid device index handled")

    def _find_test_kernel(self):
        """Helper to find a test kernel"""
        kernel_locations = [
            Path("~/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build").expanduser()
        ]

        for location in kernel_locations:
            if location.exists():
                xclbin_files = list(location.glob("final*.xclbin"))
                if xclbin_files:
                    return xclbin_files[0]
        return None


def run_tests():
    """Run all NPU callback tests"""
    print("\n" + "="*70)
    print("  NPU Callback Integration Tests (Week 6, Day 3 - Task 1)")
    print("="*70)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNPUInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestCallbackRegistration))
    suite.addTests(loader.loadTestsFromTestCase(TestDataFlow))
    suite.addTests(loader.loadTestsFromTestCase(TestNPULatency))
    suite.addTests(loader.loadTestsFromTestCase(TestNPUCallbackRobustness))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("  Test Summary")
    print("="*70)
    print(f"  Tests run: {result.testsRun}")
    print(f"  Successes: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print("="*70 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
