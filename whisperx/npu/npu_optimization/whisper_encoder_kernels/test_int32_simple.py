#!/usr/bin/env python3
"""Quick INT32 XCLBIN load test"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import pyxrt
from pathlib import Path

xclbin_path = Path("build_attention_int32/attention_int32.xclbin")
if not xclbin_path.exists():
    print(f"❌ XCLBIN not found: {xclbin_path}")
    sys.exit(1)

print("="*70)
print("INT32 Attention XCLBIN Load Test")
print("="*70)
print()

print("Loading NPU device...")
try:
    device = pyxrt.device(0)
    print(f"✅ Device opened: {device}")
    
    xclbin = pyxrt.xclbin(str(xclbin_path))
    print(f"✅ XCLBIN parsed: {xclbin_path} ({xclbin_path.stat().st_size} bytes)")
    
    device.register_xclbin(xclbin)
    print(f"✅ XCLBIN registered on NPU")
    
    # Get kernel info
    kernels = xclbin.get_kernels()
    print(f"✅ Found {len(kernels)} kernel(s)")
    for i, k in enumerate(kernels):
        print(f"   Kernel {i}: {k.get_name()}")
    
    print()
    print("="*70)
    print("SUCCESS - INT32 XCLBIN loads correctly on NPU!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Kernel executable and loads on NPU ✅")  
    print("  2. Need to test accuracy with actual data")
    print("  3. Expected correlation: 0.70-0.90 (target)")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
