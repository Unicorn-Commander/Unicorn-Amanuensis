#!/usr/bin/env python3
"""Quick test to verify XRT setup with working XCLBIN"""
import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
from pathlib import Path

xclbin_path = "attention_64x64.xclbin"
insts_path = "build_attention_64x64/insts.bin"

print("Testing with working XCLBIN...")
device = xrt.device(0)
xclbin_obj = xrt.xclbin(xclbin_path)
uuid = xclbin_obj.get_uuid()
device.register_xclbin(xclbin_obj)
print("✅ XCLBIN loaded")

hw_ctx = xrt.hw_context(device, uuid)
print("✅ Hardware context created")

kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
print("✅ Kernel found")

# Now test INT32 XCLBIN
print("\nTesting INT32 XCLBIN...")
xclbin_path_int32 = "build_attention_int32/attention_int32.xclbin"
device2 = xrt.device(0)
xclbin_obj2 = xrt.xclbin(xclbin_path_int32)
uuid2 = xclbin_obj2.get_uuid()
device2.register_xclbin(xclbin_obj2)
print("✅ INT32 XCLBIN loaded")

try:
    hw_ctx2 = xrt.hw_context(device2, uuid2)
    print("✅ INT32 Hardware context created")
    kernel2 = xrt.kernel(hw_ctx2, "MLIR_AIE")
    print("✅ INT32 Kernel found")
except Exception as e:
    print(f"❌ INT32 failed: {e}")
    # Try without hw_context
    print("\nTrying legacy API...")
    try:
        kernel2 = xrt.kernel(device2, uuid2, "MLIR_AIE")
        print("✅ INT32 Kernel found with legacy API")
    except Exception as e2:
        print(f"❌ Legacy API also failed: {e2}")
