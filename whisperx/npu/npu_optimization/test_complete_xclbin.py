#!/usr/bin/env python3
"""Test loading complete XCLBIN with metadata onto NPU"""
import sys

try:
    import pyxrt
    print("[✓] pyxrt module loaded")
except ImportError:
    print("[!] pyxrt not available")
    sys.exit(1)

print("\n" + "="*60)
print("Testing Complete XCLBIN Load on NPU")
print("="*60 + "\n")

xclbin_file = "passthrough_complete.xclbin"
print(f"[*] XCLBIN: {xclbin_file}")
print(f"[*] Size: 3,174 bytes")
print(f"[*] Sections: BITSTREAM, MEM_TOPOLOGY, IP_LAYOUT, AIE_PARTITION")
print(f"[*] Platform: xilinx_v1_ipu_0_0")

try:
    print("\n[*] Opening NPU device...")
    device = pyxrt.device(0)
    print("[✓] Device opened successfully")
    
    print(f"\n[*] Loading XCLBIN: {xclbin_file}...")
    uuid = device.load_xclbin(xclbin_file)
    print(f"[✓] XCLBIN LOADED SUCCESSFULLY!")
    print(f"    UUID: {uuid}")
    
    print("\n" + "="*60)
    print("SUCCESS! NPU accepted the XCLBIN!")
    print("="*60)
    
except Exception as e:
    print(f"\n[!] ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
