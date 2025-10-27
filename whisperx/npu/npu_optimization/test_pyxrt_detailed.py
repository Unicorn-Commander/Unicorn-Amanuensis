#!/usr/bin/env python3
"""
Detailed PyXRT testing to understand the exact API limitation.
Tests multiple XRT Python API methods to find what works.
"""

import sys
import os

def test_pyxrt_api():
    """Test various PyXRT API methods to understand limitations."""

    print("=" * 60)
    print("PyXRT API Detailed Testing")
    print("=" * 60)
    print()

    # Import pyxrt
    try:
        import pyxrt
        print("[✓] pyxrt module imported successfully")
        print(f"[*] pyxrt version: {pyxrt.__version__ if hasattr(pyxrt, '__version__') else 'unknown'}")
    except ImportError as e:
        print(f"[!] ERROR: Cannot import pyxrt: {e}")
        return False

    print()

    # Initialize device
    try:
        print("[*] Initializing device 0 (/dev/accel/accel0)...")
        device = pyxrt.device(0)
        print("[✓] Device initialized successfully")
    except Exception as e:
        print(f"[!] ERROR initializing device: {e}")
        return False

    print()

    # Test XCLBIN files
    xclbin_files = [
        "passthrough_with_pdi.xclbin",
        "passthrough_complete.xclbin",
        "passthrough_minimal.xclbin",
    ]

    for xclbin_file in xclbin_files:
        if not os.path.exists(xclbin_file):
            continue

        print(f"[*] Testing with: {xclbin_file}")
        print(f"    Size: {os.path.getsize(xclbin_file)} bytes")
        print()

        # Test 1: register_xclbin
        print("    Test 1: device.register_xclbin()")
        try:
            result = device.register_xclbin(xclbin_file)
            print(f"    [✓] register_xclbin() SUCCESS!")
            print(f"        Result: {result}")
            print(f"        Type: {type(result)}")
        except Exception as e:
            print(f"    [!] register_xclbin() FAILED: {e}")

        print()

        # Test 2: load_xclbin
        print("    Test 2: device.load_xclbin()")
        try:
            uuid = device.load_xclbin(xclbin_file)
            print(f"    [✓] load_xclbin() SUCCESS!")
            print(f"        UUID: {uuid}")
            print(f"        Type: {type(uuid)}")
        except Exception as e:
            print(f"    [!] load_xclbin() FAILED: {e}")
            print(f"        Exception type: {type(e).__name__}")

        print()
        print("-" * 60)
        print()

    # Test device info
    print("[*] Device Information:")
    try:
        # Try various device info methods
        try:
            name = device.get_info(pyxrt.device.get_info.name)
            print(f"    Name: {name}")
        except:
            pass

        try:
            # List available methods
            print("    Available device methods:")
            for attr in dir(device):
                if not attr.startswith('_'):
                    print(f"        - {attr}")
        except:
            pass

    except Exception as e:
        print(f"    [!] Error getting device info: {e}")

    print()
    print("=" * 60)
    print()

    return True


if __name__ == "__main__":
    os.chdir("/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization")
    test_pyxrt_api()
