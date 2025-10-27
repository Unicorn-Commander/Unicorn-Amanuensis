#!/usr/bin/env python3
"""
Test XCLBIN with working mobilenet PDI to validate our structure.
This will tell us if the issue is ONLY the PDI or if there are other problems.
"""

import sys
import os
import pyxrt

def test_with_mobilenet_pdi():
    """Test our XCLBIN structure with a known-working PDI."""

    print("=" * 70)
    print("Testing XCLBIN Structure with Working Mobilenet PDI")
    print("=" * 70)
    print()

    xclbin_file = "test_with_mobilenet_pdi.xclbin"

    if not os.path.exists(xclbin_file):
        print(f"[!] ERROR: {xclbin_file} not found!")
        return False

    print(f"[*] XCLBIN file: {xclbin_file}")
    print(f"    Size: {os.path.getsize(xclbin_file)} bytes")
    print()

    # Initialize device
    try:
        print("[*] Initializing NPU device 0...")
        device = pyxrt.device(0)
        print("[✓] Device initialized")
    except Exception as e:
        print(f"[!] Device init failed: {e}")
        return False

    print()

    # Create xclbin object
    try:
        print("[*] Creating xrt.xclbin object...")
        xclbin = pyxrt.xclbin(xclbin_file)
        print("[✓] xclbin object created")
    except Exception as e:
        print(f"[!] xclbin creation failed: {e}")
        return False

    print()

    # Register xclbin
    try:
        print("[*] Registering xclbin with NPU...")
        uuid = device.register_xclbin(xclbin)
        print("[✓] XCLBIN registered successfully!")
        print(f"    UUID: {uuid}")
    except Exception as e:
        print(f"[!] Registration failed: {e}")
        return False

    print()

    # Try to access kernel
    print("[*] Attempting to access DPU kernel...")
    try:
        kernel = pyxrt.kernel(device, uuid, "DPU:passthrough")
        print("[✓] SUCCESS! Kernel object created!")
        print(f"    Kernel type: {type(kernel)}")
        print()
        print("=" * 70)
        print("🎉 OUR XCLBIN STRUCTURE IS CORRECT!")
        print("    The issue was ONLY the PDI file format.")
        print("    Next step: Generate proper PDI from our MLIR compilation.")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"[!] Kernel access failed: {e}")
        print()
        print("Analysis:")
        print("  - XCLBIN registers successfully ✓")
        print("  - But kernel still not accessible ✗")
        print("  - This suggests:")
        if "No valid DPU kernel" in str(e):
            print("    → Mobilenet PDI doesn't match our kernel name")
            print("    → We need a PDI specifically for 'passthrough' kernel")
        else:
            print(f"    → Unexpected error: {e}")
        print()
        return False

    print()


if __name__ == "__main__":
    os.chdir("/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization")
    success = test_with_mobilenet_pdi()
    sys.exit(0 if success else 1)
