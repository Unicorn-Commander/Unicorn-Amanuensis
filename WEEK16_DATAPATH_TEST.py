#!/usr/bin/env python3
"""
Week 16: NPU Datapath Test

Test if we can successfully write to and read from NPU buffers.
This isolates the data transfer issue from computation.

Author: Week 16 NPU Debugging Team
Date: November 2, 2025
"""

import sys
import numpy as np
from pathlib import Path

try:
    import pyxrt as xrt
    print("[OK] XRT Python bindings loaded")
except ImportError:
    print("[ERROR] pyxrt not available")
    sys.exit(1)


def main():
    print("\n" + "="*70)
    print("  WEEK 16: NPU DATAPATH TEST")
    print("  Testing data write/read without computation")
    print("="*70)

    # Find kernel
    kernel_dir = Path("/home/ccadmin/CC-1L/kernels/common/build_bf16_1tile")
    xclbin_path = kernel_dir / "matmul_1tile_bf16.xclbin"

    if not xclbin_path.exists():
        print(f"[ERROR] xclbin not found: {xclbin_path}")
        return 1

    try:
        # Initialize device
        print("\n[1/6] Opening NPU device...")
        device = xrt.device(0)
        print("  Device 0 opened")

        # Load xclbin
        print("[2/6] Loading xclbin...")
        xclbin = xrt.xclbin(str(xclbin_path))
        device.register_xclbin(xclbin)
        uuid = xclbin.get_uuid()
        context = xrt.hw_context(device, uuid)
        print("  xclbin registered")

        # Get kernel
        print("[3/6] Loading kernel...")
        kernels = xclbin.get_kernels()
        kernel_name = kernels[0].get_name()
        kernel = xrt.kernel(context, kernel_name)
        print(f"  Kernel: {kernel_name}")

        # Create test pattern
        print("[4/6] Creating test pattern...")
        # Use simple pattern to verify data integrity
        test_size = 512 * 512  # Match kernel size
        test_pattern = np.arange(test_size, dtype=np.uint16)  # Sequential pattern
        print(f"  Pattern: 0, 1, 2, ... {test_size-1} (as uint16)")

        # Create buffers
        print("[5/6] Testing buffer write/read...")
        size_bytes = test_size * 2  # 2 bytes per uint16

        # Test each buffer independently
        for group_id, name in [(0, "arg0"), (1, "arg1"), (2, "arg2")]:
            print(f"\n  Testing buffer {name} (group_id {group_id}):")

            try:
                # Create buffer
                bo = xrt.bo(device, size_bytes, xrt.bo.host_only, kernel.group_id(group_id))
                print(f"    ✓ Buffer created ({size_bytes} bytes)")

                # Write test pattern
                bo.write(test_pattern.tobytes(), 0)
                print(f"    ✓ Pattern written")

                # Read back WITHOUT sync (test host-side memory)
                readback = np.frombuffer(bo.map()[:size_bytes], dtype=np.uint16)

                # Verify
                if np.array_equal(readback, test_pattern):
                    print(f"    ✓ Host-side readback matches (first 5: {readback[:5]})")
                else:
                    print(f"    ✗ Host-side readback MISMATCH!")
                    print(f"      Expected: {test_pattern[:5]}")
                    print(f"      Got:      {readback[:5]}")
                    return 1

                # Now test device sync
                bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                print(f"    ✓ Synced to device")

                bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                print(f"    ✓ Synced from device")

                # Read again after device sync
                readback2 = np.frombuffer(bo.map()[:size_bytes], dtype=np.uint16)

                if np.array_equal(readback2, test_pattern):
                    print(f"    ✓ Device round-trip successful!")
                else:
                    print(f"    ✗ Device round-trip FAILED!")
                    print(f"      Expected: {test_pattern[:5]}")
                    print(f"      Got:      {readback2[:5]}")
                    return 1

            except Exception as e:
                print(f"    ✗ Error: {e}")
                return 1

        # Now test if kernel execution preserves data (passthrough test)
        print("\n[6/6] Testing kernel execution (NO computation expected)...")

        # Create buffers with test pattern
        bo_A = xrt.bo(device, size_bytes, xrt.bo.host_only, kernel.group_id(0))
        bo_B = xrt.bo(device, size_bytes, xrt.bo.host_only, kernel.group_id(1))
        bo_C = xrt.bo(device, size_bytes, xrt.bo.host_only, kernel.group_id(2))

        # Write patterns
        pattern_A = np.full(test_size, 1, dtype=np.uint16)  # All 1s
        pattern_B = np.full(test_size, 2, dtype=np.uint16)  # All 2s

        bo_A.write(pattern_A.tobytes(), 0)
        bo_B.write(pattern_B.tobytes(), 0)

        # Sync to device
        bo_A.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bo_B.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        print("  Executing kernel...")
        run = kernel(bo_A, bo_B, bo_C)
        run.wait()
        print("  ✓ Kernel completed")

        # Read result
        bo_C.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        result = np.frombuffer(bo_C.map()[:size_bytes], dtype=np.uint16)

        print(f"\n  Result analysis:")
        print(f"    First 10 elements: {result[:10]}")
        print(f"    Min value: {result.min()}")
        print(f"    Max value: {result.max()}")
        print(f"    Mean value: {result.mean():.2f}")
        print(f"    Non-zero count: {np.count_nonzero(result)}/{test_size}")

        if np.all(result == 0):
            print("\n  ⚠ WARNING: All zeros in output buffer!")
            print("  This confirms the issue - kernel writes nothing to output.")
            print("\n  Possible causes:")
            print("    1. Kernel DMA not configured correctly")
            print("    2. Buffer memory bank mismatch")
            print("    3. Kernel needs explicit buffer addressing")
            print("    4. Runtime sequence not executing")
        else:
            print(f"\n  ✓ Output has non-zero values!")
            print(f"    This suggests kernel IS writing something.")

        print("\n" + "="*70)
        print("  DATAPATH TEST COMPLETE")
        print("="*70)

        return 0

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
