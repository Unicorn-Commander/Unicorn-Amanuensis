#!/usr/bin/env python3
"""
Week 16 Integration Test - XRTApp Buffer Operations

Tests the complete integration of real XRT buffer operations in the
Unicorn-Amanuensis service. Validates:

1. XRT buffer allocation and registration
2. Data write and read operations
3. Buffer synchronization (host ↔ device)
4. NPU callback chain integration
5. End-to-end service initialization

This test verifies that the service now uses real XRT buffers instead of
the stub implementation, enabling actual NPU execution.

Author: Service Integration Team
Date: November 2, 2025
Status: Week 16 - Real Buffer Operations
"""

import sys
import numpy as np
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_xrt_buffer_allocation():
    """Test 1: XRT buffer allocation and registration"""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: XRT Buffer Allocation")
    logger.info("="*70)

    try:
        # Import XRT
        import pyxrt as xrt

        # Load device and xclbin
        xclbin_path = Path(__file__).parent.parent.parent.parent / \
                     "kernels/common/build_bf16_1tile/matmul_1tile_bf16.xclbin"

        if not xclbin_path.exists():
            logger.warning(f"xclbin not found: {xclbin_path}")
            logger.warning("Skipping test (NPU hardware required)")
            return False

        logger.info(f"Loading xclbin: {xclbin_path.name}")

        # Open device
        device = xrt.device(0)
        logger.info("✓ Device opened")

        # Load xclbin
        xclbin = xrt.xclbin(str(xclbin_path))
        device.register_xclbin(xclbin)
        logger.info("✓ xclbin registered")

        # Create context
        uuid = xclbin.get_uuid()
        context = xrt.hw_context(device, uuid)
        logger.info("✓ Hardware context created")

        # Get kernel
        kernel = xrt.kernel(context, "MLIR_AIE")
        logger.info("✓ Kernel loaded")

        # Test buffer allocation
        test_shape = (512, 1024)
        test_dtype = np.float32
        size = int(np.prod(test_shape) * np.dtype(test_dtype).itemsize)

        logger.info(f"\nAllocating test buffer: {test_dtype} {test_shape}")
        logger.info(f"  Size: {size:,} bytes ({size/(1024*1024):.2f} MB)")

        bo = xrt.bo(device, size, xrt.bo.host_only, kernel.group_id(0))
        logger.info("✓ Buffer allocated successfully")

        # Test write
        test_data = np.random.randn(*test_shape).astype(test_dtype)
        bo.write(test_data, 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        logger.info("✓ Data written and synced to device")

        # Test read
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        read_data = np.frombuffer(bytes(bo.map())[:size], dtype=test_dtype).reshape(test_shape)
        logger.info("✓ Data read and synced from device")

        # Verify
        if np.allclose(test_data, read_data):
            logger.info("✓ Data integrity verified (write/read match)")
        else:
            logger.error("✗ Data mismatch after write/read")
            return False

        logger.info("\n" + "="*70)
        logger.info("TEST 1: PASSED ✓")
        logger.info("="*70)
        return True

    except ImportError as e:
        logger.error(f"pyxrt not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_xrtapp_class():
    """Test 2: XRTApp class integration"""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: XRTApp Class Integration")
    logger.info("="*70)

    try:
        # Import server module to get XRTApp
        from xdna2.server import load_xrt_npu_application

        logger.info("Loading XRT NPU application...")
        npu_app = load_xrt_npu_application()
        logger.info(f"✓ XRTApp created: {npu_app.__class__.__name__}")

        # Verify it's the real implementation (not stub)
        if hasattr(npu_app, 'xrt_buffers'):
            logger.info("✓ Real XRTApp (has xrt_buffers attribute)")
        else:
            logger.error("✗ Still using stub (no xrt_buffers attribute)")
            return False

        if hasattr(npu_app, 'buffer_metadata'):
            logger.info("✓ Has buffer_metadata attribute")
        else:
            logger.error("✗ Missing buffer_metadata attribute")
            return False

        # Test buffer registration
        logger.info("\nTesting buffer registration...")
        test_idx = 0
        test_dtype = np.float32
        test_shape = (100, 200)

        npu_app.register_buffer(test_idx, test_dtype, test_shape)
        logger.info(f"✓ Buffer {test_idx} registered")

        # Verify buffer exists
        if test_idx in npu_app.xrt_buffers:
            logger.info("✓ Buffer object created")
        else:
            logger.error("✗ Buffer object not created")
            return False

        if test_idx in npu_app.buffer_metadata:
            metadata = npu_app.buffer_metadata[test_idx]
            logger.info(f"✓ Metadata stored: {metadata}")
        else:
            logger.error("✗ Metadata not stored")
            return False

        # Test write/read
        logger.info("\nTesting write/read operations...")
        test_data = np.random.randn(*test_shape).astype(test_dtype)

        npu_app.write_buffer(test_idx, test_data)
        logger.info("✓ Data written to buffer")

        read_data = npu_app.read_buffer(test_idx)
        logger.info("✓ Data read from buffer")

        if np.allclose(test_data, read_data):
            logger.info("✓ Data integrity verified")
        else:
            logger.error("✗ Data mismatch")
            return False

        logger.info("\n" + "="*70)
        logger.info("TEST 2: PASSED ✓")
        logger.info("="*70)
        return True

    except FileNotFoundError as e:
        logger.warning(f"xclbin not found: {e}")
        logger.warning("Skipping test (NPU hardware required)")
        return False
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encoder_npu_callback():
    """Test 3: Encoder NPU callback registration"""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Encoder NPU Callback Registration")
    logger.info("="*70)

    try:
        from xdna2.encoder_cpp import create_encoder_cpp
        from xdna2.server import load_xrt_npu_application

        logger.info("Creating C++ encoder...")
        encoder = create_encoder_cpp(num_layers=6, use_npu=True)
        logger.info("✓ Encoder created")

        logger.info("\nLoading XRT application...")
        npu_app = load_xrt_npu_application()
        logger.info("✓ XRT application loaded")

        logger.info("\nRegistering NPU callback...")
        success = encoder.register_npu_callback(npu_app)

        if success:
            logger.info("✓ NPU callback registered successfully")
        else:
            logger.error("✗ NPU callback registration failed")
            return False

        # Verify callback is active
        stats = encoder.get_stats()
        if 'npu_stats' in stats:
            logger.info("✓ NPU statistics available")
            logger.info(f"  NPU stats: {stats['npu_stats']}")
        else:
            logger.warning("⚠ NPU statistics not available (may not be used yet)")

        logger.info("\n" + "="*70)
        logger.info("TEST 3: PASSED ✓")
        logger.info("="*70)
        return True

    except FileNotFoundError as e:
        logger.warning(f"xclbin not found: {e}")
        logger.warning("Skipping test (NPU hardware required)")
        return False
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_service_initialization():
    """Test 4: Full service initialization"""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Service Initialization")
    logger.info("="*70)

    try:
        logger.info("Importing server module...")
        from xdna2.server import initialize_encoder

        logger.info("\nInitializing encoder (this may take a while)...")
        logger.info("Note: Full initialization requires Whisper model download")

        success = initialize_encoder()

        if success:
            logger.info("✓ Service initialized successfully")
        else:
            logger.error("✗ Service initialization failed")
            return False

        logger.info("\n" + "="*70)
        logger.info("TEST 4: PASSED ✓")
        logger.info("="*70)
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("  Week 16 Integration Tests - XRTApp Buffer Operations")
    print("="*70)
    print("\nTesting real XRT buffer implementation in Unicorn-Amanuensis")
    print("Service Integration Team - November 2, 2025\n")

    # Run tests
    results = {}

    # Test 1: Low-level XRT buffer operations
    results['buffer_allocation'] = test_xrt_buffer_allocation()

    # Test 2: XRTApp class methods
    results['xrtapp_class'] = test_xrtapp_class()

    # Test 3: NPU callback registration
    results['npu_callback'] = test_encoder_npu_callback()

    # Test 4: Full service initialization
    # Skipping by default as it's slow - uncomment to run
    # results['service_init'] = test_service_initialization()

    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name:30s}: {status}")

    print("="*70)
    print(f"  Total: {passed}/{total} tests passed")
    print("="*70 + "\n")

    if passed == total:
        print("✅ All tests passed! XRTApp integration successful.")
        return 0
    else:
        print("❌ Some tests failed. Check logs above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
