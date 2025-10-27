#!/usr/bin/env python3
"""
Test loading compiled XCLBIN on AMD Phoenix NPU
Tests Phase 2.3 INT8 optimized kernel
"""
import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import pyxrt as xrt
import numpy as np
from pathlib import Path

def test_xclbin_loading():
    """Test loading Phase 2.3 INT8 XCLBIN on NPU"""
    print("=" * 70)
    print("TESTING NPU KERNEL LOADING (Phase 2.3 INT8)")
    print("=" * 70 + "\n")

    # Path to compiled kernel
    kernel_path = Path(__file__).parent / "build" / "mel_int8_optimized.xclbin"

    if not kernel_path.exists():
        print(f"❌ XCLBIN not found: {kernel_path}")
        print("Run: ./compile_mel_int8.sh")
        return False

    print(f"✅ XCLBIN found: {kernel_path}")
    print(f"   Size: {kernel_path.stat().st_size} bytes\n")

    try:
        # Open NPU device
        print("Opening NPU device /dev/accel/accel0...")
        device = xrt.device(0)
        print("✅ NPU device opened successfully\n")

        # Load XCLBIN
        print(f"Loading XCLBIN: {kernel_path}")
        xclbin_uuid = device.load_xclbin(str(kernel_path))
        print(f"✅ XCLBIN loaded successfully!")
        print(f"   UUID: {xclbin_uuid}\n")

        # Get device info
        print("NPU Device Information:")
        print(f"  Device index: 0")
        print(f"  Device type: AMD Phoenix NPU (XDNA1)")
        print(f"  XCLBIN: mel_int8_optimized.xclbin")
        print(f"  Status: Loaded and ready\n")

        print("=" * 70)
        print("✅ NPU KERNEL LOADING TEST PASSED!")
        print("=" * 70)
        print("\nPhase 2.3 INT8 kernel successfully loaded on NPU hardware.")
        print("Ready for mel spectrogram computation testing.\n")

        return True

    except Exception as e:
        print(f"❌ Failed to load XCLBIN: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check XRT is running: /opt/xilinx/xrt/bin/xrt-smi examine")
        print(f"  2. Check device permissions: ls -l /dev/accel/accel0")
        print(f"  3. Verify firmware: /opt/xilinx/xrt/bin/xrt-smi examine -r aie")
        return False


if __name__ == "__main__":
    success = test_xclbin_loading()
    sys.exit(0 if success else 1)
