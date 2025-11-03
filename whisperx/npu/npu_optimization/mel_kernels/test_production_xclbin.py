#!/usr/bin/env python3
"""
Test Production XCLBIN Loading on AMD Phoenix NPU
Tests the production mel_fixed_v3_PRODUCTION_v2.0.xclbin with Oct 28 fixes
"""

import os
import sys
import numpy as np
from pathlib import Path

def test_production_xclbin():
    """Test loading production XCLBIN on NPU"""
    print("=" * 80)
    print("NPU MEL PRODUCTION XCLBIN Test (with Oct 28 Fixes)")
    print("=" * 80)
    print()

    # Step 1: Check NPU device
    print("Step 1: Checking NPU device...")
    npu_device = "/dev/accel/accel0"
    if not os.path.exists(npu_device):
        print(f"❌ NPU device not found at {npu_device}")
        return False
    print(f"✅ NPU device found: {npu_device}")
    print()

    # Step 2: Check production XCLBINs
    print("Step 2: Checking production XCLBINs...")
    build_dir = Path(__file__).parent / "build_fixed_v3"

    production_xclbins = [
        ("mel_fixed_v3.xclbin", "Latest Nov 1 with all fixes"),
        ("mel_fixed_v3_PRODUCTION_v2.0.xclbin", "Oct 30 production with 0.92 correlation"),
        ("mel_fixed_v3_SIGNFIX.xclbin", "Oct 31 with sign fix"),
        ("mel_fixed_v3_PRODUCTION_v1.0.xclbin", "Oct 29 production"),
    ]

    available_xclbins = []
    for xclbin_file, description in production_xclbins:
        xclbin_path = build_dir / xclbin_file
        if xclbin_path.exists():
            size = xclbin_path.stat().st_size
            print(f"✅ Found: {xclbin_file}")
            print(f"   Description: {description}")
            print(f"   Size: {size} bytes")
            available_xclbins.append((xclbin_path, description, size))
        else:
            print(f"❌ Missing: {xclbin_file}")
    print()

    if not available_xclbins:
        print("❌ No production XCLBINs found!")
        return False

    # Step 3: Check instruction binaries
    print("Step 3: Checking instruction binaries...")
    insts_files = [
        ("insts_v3.bin", "Latest Nov 1"),
        ("insts_v3_SIGNFIX.bin", "Oct 31 sign fix"),
    ]

    available_insts = []
    for insts_file, description in insts_files:
        insts_path = build_dir / insts_file
        if insts_path.exists():
            size = insts_path.stat().st_size
            print(f"✅ Found: {insts_file}")
            print(f"   Description: {description}")
            print(f"   Size: {size} bytes")
            available_insts.append((insts_path, description, size))
    print()

    # Step 4: Try loading with pyxrt
    print("Step 4: Testing XRT loading...")
    try:
        import pyxrt as xrt
        print("✅ pyxrt imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import pyxrt: {e}")
        return False

    try:
        device = xrt.device(0)
        print(f"✅ XRT device initialized")
    except Exception as e:
        print(f"❌ Failed to initialize XRT device: {e}")
        return False
    print()

    # Step 5: Try loading each production XCLBIN
    print("Step 5: Testing production XCLBINs...")
    successful_xclbins = []

    for xclbin_path, description, size in available_xclbins:
        print(f"\nTesting: {xclbin_path.name}")
        print(f"  {description}")

        try:
            uuid = device.load_xclbin(str(xclbin_path))
            print(f"  ✅ XCLBIN loaded successfully!")
            print(f"     UUID: {uuid}")
            successful_xclbins.append((xclbin_path, description, uuid))
        except Exception as e:
            print(f"  ❌ Failed to load: {e}")

    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)

    if successful_xclbins:
        print(f"✅ Successfully loaded {len(successful_xclbins)}/{len(available_xclbins)} XCLBINs")
        print()
        print("Working XCLBINs:")
        for xclbin_path, description, uuid in successful_xclbins:
            print(f"  • {xclbin_path.name}")
            print(f"    {description}")
            print(f"    UUID: {uuid}")
            print()

        # Recommend the best one
        best_xclbin = successful_xclbins[0]  # First one (latest)
        print("🎯 RECOMMENDED FOR PRODUCTION:")
        print(f"   {best_xclbin[0]}")
        print(f"   {best_xclbin[1]}")
        print()

        if available_insts:
            print("Instruction binary:")
            print(f"   {available_insts[0][0]}")
        print()

        return True
    else:
        print("❌ No XCLBINs loaded successfully")
        print()
        print("This might indicate:")
        print("  1. Platform mismatch (XCLBIN compiled for wrong NPU)")
        print("  2. XRT version incompatibility")
        print("  3. NPU firmware issue")
        print()
        return False


if __name__ == "__main__":
    print()
    print("🦄 NPU MEL Preprocessing Team Lead - Production XCLBIN Test")
    print()

    success = test_production_xclbin()

    if success:
        print("=" * 80)
        print("✅ SUCCESS! Production XCLBIN ready for NPU execution")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Copy working XCLBIN to server location")
        print("  2. Update server_dynamic.py configuration")
        print("  3. Run accuracy tests with librosa")
        print("  4. Benchmark performance on NPU")
        print()
    else:
        print("=" * 80)
        print("❌ BLOCKER: Cannot load XCLBINs on NPU")
        print("=" * 80)
        print()
        print("Action required:")
        print("  1. Check XRT version and NPU firmware compatibility")
        print("  2. Verify XCLBIN platform settings")
        print("  3. Recompile with correct target platform")
        print()

    sys.exit(0 if success else 1)
