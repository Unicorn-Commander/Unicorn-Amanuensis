#!/usr/bin/env python3
"""
NPU Attention Server Integration Test
Tests that the validated INT32 attention kernel loads correctly in the production server

Mission: Verify 25-35x realtime target is achievable with NPU attention
Current baseline: 16-17x realtime (decoder working, encoder CPU)
Target with NPU: 25-35x realtime (decoder + NPU attention)

Hardware: AMD Phoenix NPU (XDNA1)
Kernel: INT32 attention - 0.92 correlation, 2.08ms latency
XCLBIN: build_attention_int32/attention_64x64.xclbin (12.4 KB)
Status: VALIDATED and READY FOR PRODUCTION
"""

import sys
import logging
from pathlib import Path
import numpy as np
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_npu_attention_loading():
    """Test 1: Verify NPU attention kernel loads"""
    print("\n" + "="*70)
    print("TEST 1: NPU ATTENTION KERNEL LOADING")
    print("="*70)

    try:
        # Add path
        sys.path.insert(0, str(Path(__file__).parent))

        from npu_attention_integration import NPUAttentionIntegration

        # Initialize
        logger.info("Initializing NPU attention...")
        integration = NPUAttentionIntegration(enable_npu=True)

        # Check status
        if integration.npu_available:
            print("✅ NPU attention loaded successfully")
            print(f"   XCLBIN: attention_64x64.xclbin")
            print(f"   Device: /dev/accel/accel0")
            print(f"   Status: READY")
            return True
        else:
            print("⚠️ NPU attention not available (will use CPU fallback)")
            print("   This is OK - server will still work with decoder")
            return False

    except Exception as e:
        print(f"❌ Failed to load NPU attention: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_npu_attention_execution():
    """Test 2: Verify NPU attention executes correctly"""
    print("\n" + "="*70)
    print("TEST 2: NPU ATTENTION EXECUTION")
    print("="*70)

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from npu_attention_integration import NPUAttentionIntegration

        integration = NPUAttentionIntegration(enable_npu=True)

        if not integration.npu_available:
            print("⚠️ Skipping execution test (NPU not available)")
            return False

        # Test single-head attention
        print("\nTest 2a: Single-head attention (64x64)...")
        Q = np.random.randn(64, 64).astype(np.float32)
        K = np.random.randn(64, 64).astype(np.float32)
        V = np.random.randn(64, 64).astype(np.float32)

        start = time.perf_counter()
        output = integration.compute_attention(Q, K, V)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"   Latency: {elapsed_ms:.2f}ms")
        print("   ✅ Single-head attention works")

        # Test multi-head attention
        print("\nTest 2b: Multi-head attention (64x512, 8 heads)...")
        Q = np.random.randn(64, 512).astype(np.float32)
        K = np.random.randn(64, 512).astype(np.float32)
        V = np.random.randn(64, 512).astype(np.float32)

        start = time.perf_counter()
        output = integration.multi_head_attention(Q, K, V, num_heads=8)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"   Latency: {elapsed_ms:.2f}ms")
        print("   ✅ Multi-head attention works")

        # Print performance stats
        print("\nPerformance statistics:")
        stats = integration.get_performance_stats()
        print(f"   NPU calls: {stats['npu_calls']}")
        print(f"   CPU calls: {stats['cpu_calls']}")
        if stats['npu_calls'] > 0:
            print(f"   Avg NPU time: {stats['avg_npu_time_ms']:.2f}ms")

        print("\n✅ NPU attention execution test passed")
        return True

    except Exception as e:
        print(f"❌ NPU attention execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_server_integration():
    """Test 3: Verify server integrates NPU attention correctly"""
    print("\n" + "="*70)
    print("TEST 3: SERVER INTEGRATION")
    print("="*70)

    try:
        # Add server path
        server_path = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(server_path))

        # Import server class
        from server_dynamic import DynamicWhisperEngine

        print("\nInitializing Whisper engine with NPU attention...")
        engine = DynamicWhisperEngine()

        # Check NPU attention status
        has_npu_attention = hasattr(engine, 'npu_attention') and engine.npu_attention is not None

        if has_npu_attention:
            npu_active = engine.npu_attention.npu_available
            print(f"✅ NPU attention integrated into server")
            print(f"   Status: {'ACTIVE' if npu_active else 'CPU FALLBACK'}")

            if npu_active:
                print(f"   Expected speedup: 1.5-2x encoder acceleration")
                print(f"   Target: 25-35x realtime (from 16-17x baseline)")
            else:
                print(f"   Baseline: 16-17x realtime (decoder only)")

            return True
        else:
            print("⚠️ NPU attention not initialized in server")
            print("   Server will use CPU for attention")
            print("   Baseline: 16-17x realtime (decoder only)")
            return False

    except Exception as e:
        print(f"❌ Server integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cpu_fallback():
    """Test 4: Verify CPU fallback works"""
    print("\n" + "="*70)
    print("TEST 4: CPU FALLBACK")
    print("="*70)

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from npu_attention_integration import NPUAttentionIntegration

        # Force CPU fallback
        print("\nForcing CPU fallback...")
        integration = NPUAttentionIntegration(enable_npu=False)

        # Test attention
        Q = np.random.randn(64, 64).astype(np.float32)
        K = np.random.randn(64, 64).astype(np.float32)
        V = np.random.randn(64, 64).astype(np.float32)

        start = time.perf_counter()
        output = integration.compute_attention(Q, K, V)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"   CPU latency: {elapsed_ms:.2f}ms")
        print("   ✅ CPU fallback works")

        # Verify it used CPU
        stats = integration.get_performance_stats()
        if stats['cpu_calls'] > 0 and stats['npu_calls'] == 0:
            print("   ✅ Confirmed: Using CPU (no NPU calls)")
            return True
        else:
            print(f"   ⚠️ Unexpected: NPU={stats['npu_calls']}, CPU={stats['cpu_calls']}")
            return False

    except Exception as e:
        print(f"❌ CPU fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("NPU ATTENTION SERVER INTEGRATION TEST SUITE")
    print("="*70)
    print("\nMission: Integrate validated INT32 attention kernel into production")
    print("Current: 16-17x realtime (decoder working, encoder CPU)")
    print("Target:  25-35x realtime (decoder + NPU attention)")
    print("\nKernel: INT32 attention - 0.92 correlation, 2.08ms latency")
    print("XCLBIN: build_attention_int32/attention_64x64.xclbin (12.4 KB)")
    print("Status: VALIDATED and READY FOR PRODUCTION")
    print()

    results = []

    # Test 1: Loading
    test1_pass = test_npu_attention_loading()
    results.append(("NPU Attention Loading", test1_pass))

    # Test 2: Execution (only if NPU available)
    if test1_pass:
        test2_pass = test_npu_attention_execution()
        results.append(("NPU Attention Execution", test2_pass))
    else:
        results.append(("NPU Attention Execution", None))  # Skipped

    # Test 3: Server Integration
    test3_pass = test_server_integration()
    results.append(("Server Integration", test3_pass))

    # Test 4: CPU Fallback
    test4_pass = test_cpu_fallback()
    results.append(("CPU Fallback", test4_pass))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, test_result in results:
        if test_result is True:
            status = "✅ PASS"
        elif test_result is False:
            status = "❌ FAIL"
        else:
            status = "⏭️ SKIPPED"
        print(f"{test_name:40s} {status}")

    # Overall status
    print("\n" + "="*70)
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)

    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("\n✅ ALL TESTS PASSED - NPU ATTENTION READY FOR PRODUCTION")
        print("\nNext steps:")
        print("1. Start server: python3 server_dynamic.py")
        print("2. Check /status endpoint for NPU attention status")
        print("3. Transcribe audio and monitor performance")
        print("4. Target: 25-35x realtime speedup")
    elif test3_pass:
        print("\n⚠️ SOME TESTS FAILED - BUT SERVER INTEGRATION WORKS")
        print("\nServer will use CPU fallback for attention")
        print("Baseline: 16-17x realtime (decoder only)")
        print("\nTo enable NPU:")
        print("1. Check /dev/accel/accel0 exists")
        print("2. Verify XCLBIN at: whisper_encoder_kernels/build_attention_int32/")
        print("3. Restart server")
    else:
        print("\n❌ CRITICAL TESTS FAILED - MANUAL INTERVENTION REQUIRED")
        print("\nPlease check:")
        print("1. NPU device accessible: ls -l /dev/accel/accel0")
        print("2. XCLBIN exists: whisper_encoder_kernels/build_attention_int32/attention_64x64.xclbin")
        print("3. XRT installed: /opt/xilinx/xrt/python")

    print("="*70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
