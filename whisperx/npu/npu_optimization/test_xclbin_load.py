#!/usr/bin/env python3
"""
Test loading our minimal XCLBIN onto the NPU device
"""
import sys

try:
    import pyxrt
    print("[✓] pyxrt module loaded")
except ImportError:
    print("[!] pyxrt not available, trying basic device open...")
    pyxrt = None

print("\n" + "="*60)
print("Testing Minimal XCLBIN Load on NPU")
print("="*60 + "\n")

xclbin_file = "passthrough_minimal.xclbin"
print(f"[*] XCLBIN: {xclbin_file}")

try:
    if pyxrt:
        # Try with pyxrt if available
        print("\n[*] Opening NPU device with pyxrt...")
        device = pyxrt.device(0)
        print("[✓] Device opened successfully")

        print(f"\n[*] Loading XCLBIN: {xclbin_file}...")
        uuid = device.load_xclbin(xclbin_file)
        print(f"[✓] XCLBIN loaded successfully!")
        print(f"    UUID: {uuid}")

    else:
        # Fallback: just check if device exists
        import os
        if os.path.exists("/dev/accel/accel0"):
            print("[✓] NPU device /dev/accel/accel0 exists")
            print("[i] pyxrt not available for loading test")
        else:
            print("[!] NPU device /dev/accel/accel0 not found")
            sys.exit(1)

except Exception as e:
    print(f"\n[!] ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("Test completed!")
print("="*60)
