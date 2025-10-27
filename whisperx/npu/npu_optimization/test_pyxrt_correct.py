#!/usr/bin/env python3
"""
Corrected PyXRT testing using proper xrt::xclbin object.
Based on the error message, register_xclbin expects xrt::xclbin, not a string.
"""

import sys
import os

def test_pyxrt_correct():
    """Test PyXRT with correct API usage."""

    print("=" * 70)
    print("PyXRT API Corrected Testing - Using xrt::xclbin Object")
    print("=" * 70)
    print()

    # Import pyxrt
    try:
        import pyxrt
        print("[✓] pyxrt module imported successfully")
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
    ]

    for xclbin_file in xclbin_files:
        if not os.path.exists(xclbin_file):
            continue

        print(f"[*] Testing with: {xclbin_file}")
        print(f"    Size: {os.path.getsize(xclbin_file)} bytes")
        print()

        # Method 1: Load xclbin object first, then register
        print("    Method 1: Create xrt.xclbin object, then register")
        try:
            # Create xclbin object from file
            print("    [*] Creating xrt.xclbin object...")
            xclbin = pyxrt.xclbin(xclbin_file)
            print(f"    [✓] xrt.xclbin object created!")
            print(f"        Type: {type(xclbin)}")

            # Now register it
            print("    [*] Registering xclbin with device...")
            uuid = device.register_xclbin(xclbin)
            print(f"    [✓] register_xclbin() SUCCESS!")
            print(f"        UUID: {uuid}")
            print(f"        UUID type: {type(uuid)}")

            print()
            print("    [*] Trying to access kernel...")
            try:
                # Try to get kernel
                kernel = pyxrt.kernel(device, uuid, "DPU:passthrough")
                print(f"    [✓] Kernel object created!")
                print(f"        Kernel type: {type(kernel)}")

            except Exception as e:
                print(f"    [!] Kernel access failed: {e}")
                print(f"        (This may be expected if kernel incomplete)")

        except Exception as e:
            print(f"    [!] Method 1 FAILED: {e}")
            print(f"        Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()

        print()

        # Method 2: Try load_xclbin for comparison
        print("    Method 2: Direct load_xclbin() for comparison")
        try:
            uuid2 = device.load_xclbin(xclbin_file)
            print(f"    [✓] load_xclbin() SUCCESS!")
            print(f"        UUID: {uuid2}")
        except Exception as e:
            print(f"    [!] load_xclbin() FAILED (expected): {e}")

        print()
        print("-" * 70)
        print()

    print("=" * 70)
    print()

    return True


if __name__ == "__main__":
    os.chdir("/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization")
    test_pyxrt_correct()
