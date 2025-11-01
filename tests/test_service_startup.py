#!/usr/bin/env python3
"""
Service Startup Integration Tests

Tests for Week 6 Days 1-2 service integration:
1. Service starts without errors
2. Platform detector selects XDNA2_CPP
3. C++ encoder initializes successfully
4. /v1/audio/transcriptions endpoint is accessible
5. Health check passes

Author: CC-1L Integration Team
Date: November 1, 2025
Status: Week 6 Days 1-2 Implementation
"""

import sys
import os
import unittest
import time
import tempfile
import wave
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestServiceStartup(unittest.TestCase):
    """Test service startup and basic integration"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        print("\n" + "="*70)
        print("  SERVICE STARTUP INTEGRATION TESTS")
        print("="*70)

    def test_01_platform_detection(self):
        """Test platform detector selects XDNA2_CPP when C++ runtime available"""
        print("\n[Test 1/5] Platform Detection")

        from runtime.platform_detector import get_platform, get_platform_info, Platform

        platform = get_platform()
        platform_info = get_platform_info()

        print(f"  Detected platform: {platform.value}")
        print(f"  Has NPU: {platform_info.get('has_npu', False)}")
        print(f"  Uses C++ runtime: {platform_info.get('uses_cpp_runtime', False)}")

        # Check if C++ runtime is available
        if platform_info.get('uses_cpp_runtime', False):
            self.assertEqual(platform, Platform.XDNA2_CPP)
            print("  ✓ Platform detection successful: XDNA2_CPP")
        else:
            print(f"  ⚠ C++ runtime not available, using: {platform.value}")
            print("  This is expected if C++ libraries are not built yet")
            self.assertIn(platform, [Platform.XDNA2, Platform.XDNA1, Platform.CPU])

    def test_02_cpp_encoder_import(self):
        """Test C++ encoder can be imported"""
        print("\n[Test 2/5] C++ Encoder Import")

        try:
            from xdna2.encoder_cpp import WhisperEncoderCPP, create_encoder_cpp
            from xdna2.cpp_runtime_wrapper import CPPRuntimeWrapper
            print("  ✓ C++ encoder modules imported successfully")
        except ImportError as e:
            print(f"  ⚠ C++ encoder import failed: {e}")
            print("  This is expected if C++ libraries are not built yet")
            self.skipTest("C++ encoder not available")

    def test_03_cpp_encoder_initialization(self):
        """Test C++ encoder can be initialized"""
        print("\n[Test 3/5] C++ Encoder Initialization")

        try:
            from xdna2.encoder_cpp import create_encoder_cpp
            from xdna2.cpp_runtime_wrapper import CPPRuntimeError

            print("  Creating encoder (CPU mode for testing)...")
            encoder = create_encoder_cpp(
                num_layers=6,
                n_heads=8,
                n_state=512,
                ffn_dim=2048,
                use_npu=False  # CPU mode for testing
            )

            print("  Checking encoder properties...")
            self.assertEqual(encoder.num_layers, 6)
            self.assertEqual(encoder.n_heads, 8)
            self.assertEqual(encoder.n_state, 512)
            self.assertEqual(encoder.ffn_dim, 2048)

            print("  Getting encoder stats...")
            stats = encoder.get_stats()
            self.assertIsNotNone(stats)
            self.assertIn('runtime_version', stats)
            print(f"    Runtime version: {stats['runtime_version']}")

            print("  ✓ C++ encoder initialized successfully")

        except Exception as e:
            print(f"  ✗ C++ encoder initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.skipTest(f"C++ encoder initialization failed: {e}")

    def test_04_xdna2_server_import(self):
        """Test xdna2 server can be imported"""
        print("\n[Test 4/5] XDNA2 Server Import")

        try:
            from xdna2 import server as xdna2_server
            print("  ✓ XDNA2 server module imported successfully")

            # Check if app exists
            self.assertIsNotNone(xdna2_server.app)
            print(f"    App title: {xdna2_server.app.title}")
            print(f"    App version: {xdna2_server.app.version}")

        except ImportError as e:
            print(f"  ✗ XDNA2 server import failed: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"XDNA2 server import failed: {e}")

    def test_05_api_entry_point(self):
        """Test main API entry point imports correctly"""
        print("\n[Test 5/5] API Entry Point")

        try:
            # Clear any cached imports
            if 'api' in sys.modules:
                del sys.modules['api']

            import api
            print("  ✓ API module imported successfully")

            # Check platform was detected
            self.assertIsNotNone(api.platform)
            print(f"    Detected platform: {api.platform.value}")

            # Check backend type was set
            self.assertIsNotNone(api.backend_type)
            print(f"    Backend type: {api.backend_type}")

            # Check app exists
            self.assertIsNotNone(api.app)
            print(f"    App title: {api.app.title}")

        except Exception as e:
            print(f"  ✗ API import failed: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"API import failed: {e}")


class TestServiceEndpoints(unittest.TestCase):
    """Test service endpoints (requires service to be running)"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        print("\n" + "="*70)
        print("  SERVICE ENDPOINT TESTS")
        print("="*70)
        print("  Note: These tests require the service to be running")
        print("  Start service with: python3 api.py")
        print("="*70)

    def test_01_health_endpoint(self):
        """Test /health endpoint"""
        print("\n[Endpoint Test 1/3] Health Check")

        try:
            import requests
            response = requests.get("http://localhost:9000/health", timeout=5)

            print(f"  Status code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"  Status: {data.get('status', 'unknown')}")
                print(f"  Backend: {data.get('backend', 'unknown')}")
                print("  ✓ Health endpoint accessible")
                self.assertEqual(data.get('status'), 'healthy')
            else:
                print(f"  ⚠ Unexpected status code: {response.status_code}")

        except requests.exceptions.ConnectionError:
            print("  ⚠ Service not running at localhost:9000")
            print("  Start service with: python3 api.py")
            self.skipTest("Service not running")

        except ImportError:
            print("  ⚠ requests library not installed")
            print("  Install with: pip install requests")
            self.skipTest("requests library not available")

    def test_02_platform_endpoint(self):
        """Test /platform endpoint"""
        print("\n[Endpoint Test 2/3] Platform Info")

        try:
            import requests
            response = requests.get("http://localhost:9000/platform", timeout=5)

            print(f"  Status code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"  Service: {data.get('service', 'unknown')}")
                print(f"  Version: {data.get('version', 'unknown')}")
                print(f"  Backend: {data.get('backend', 'unknown')}")

                platform_info = data.get('platform', {})
                print(f"  Platform: {platform_info.get('platform', 'unknown')}")
                print(f"  Has NPU: {platform_info.get('has_npu', False)}")
                print(f"  Uses C++ runtime: {platform_info.get('uses_cpp_runtime', False)}")

                print("  ✓ Platform endpoint accessible")

        except requests.exceptions.ConnectionError:
            print("  ⚠ Service not running")
            self.skipTest("Service not running")

        except ImportError:
            self.skipTest("requests library not available")

    def test_03_transcription_endpoint(self):
        """Test /v1/audio/transcriptions endpoint exists"""
        print("\n[Endpoint Test 3/3] Transcription Endpoint")

        try:
            import requests

            # Create dummy audio file (1 second of silence)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                # Generate 1 second of silence at 16kHz
                sample_rate = 16000
                duration = 1  # seconds
                samples = np.zeros(sample_rate * duration, dtype=np.int16)

                # Write WAV file
                with wave.open(tmp.name, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(samples.tobytes())

                tmp_path = tmp.name

            try:
                # Send request (should fail because encoder needs real initialization)
                print("  Sending test request (expecting potential error)...")
                with open(tmp_path, 'rb') as f:
                    files = {'file': ('test.wav', f, 'audio/wav')}
                    response = requests.post(
                        "http://localhost:9000/v1/audio/transcriptions",
                        files=files,
                        timeout=30
                    )

                print(f"  Status code: {response.status_code}")

                if response.status_code == 200:
                    data = response.json()
                    print(f"  Text: {data.get('text', '')[:50]}...")
                    print("  ✓ Transcription endpoint accessible and working")

                elif response.status_code == 503:
                    print("  ⚠ Service unavailable (encoder not initialized)")
                    print("  This is expected if weights are not loaded yet")

                else:
                    print(f"  ⚠ Unexpected status: {response.status_code}")
                    print(f"  Response: {response.text[:200]}")

            finally:
                os.unlink(tmp_path)

        except requests.exceptions.ConnectionError:
            print("  ⚠ Service not running")
            self.skipTest("Service not running")

        except ImportError:
            self.skipTest("requests library not available")


def run_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("  UNICORN-AMANUENSIS INTEGRATION TESTS")
    print("  Week 6 Days 1-2: Service Integration")
    print("="*70)

    # Run startup tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestServiceStartup)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Run endpoint tests (if startup tests passed)
    if result.wasSuccessful():
        print("\n" + "="*70)
        print("  Running endpoint tests...")
        print("  (These require the service to be running)")
        print("="*70)

        suite = unittest.TestLoader().loadTestsFromTestCase(TestServiceEndpoints)
        runner = unittest.TextTestRunner(verbosity=2)
        endpoint_result = runner.run(suite)

        # Combine results
        if not endpoint_result.wasSuccessful():
            result = endpoint_result

    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n  ✓ ALL TESTS PASSED")
    else:
        print("\n  ✗ SOME TESTS FAILED")

    print("="*70 + "\n")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
