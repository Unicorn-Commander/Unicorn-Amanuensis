#!/bin/bash
# Quick NPU detection test

echo "🔍 Testing NPU Detection..."
echo ""

cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx

python3 << 'PYTHON_SCRIPT'
import os
import sys
import subprocess

print("1️⃣  Checking NPU device...")
if os.path.exists("/dev/accel/accel0"):
    print("   ✅ /dev/accel/accel0 exists")
else:
    print("   ❌ NPU device not found!")
    sys.exit(1)

print("\n2️⃣  Checking XRT...")
try:
    result = subprocess.run(
        ["/opt/xilinx/xrt/bin/xrt-smi", "examine"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if "NPU Phoenix" in result.stdout:
        print("   ✅ XRT sees NPU Phoenix")
    else:
        print("   ⚠️  XRT output doesn't show NPU")
except Exception as e:
    print(f"   ❌ XRT error: {e}")

print("\n3️⃣  Checking kernel directory...")
kernel_dir = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels"
if os.path.exists(kernel_dir):
    kernels = [f for f in os.listdir(kernel_dir) if f.endswith('.xclbin')]
    print(f"   ✅ Found {len(kernels)} kernels in whisper_encoder_kernels/")
else:
    print(f"   ❌ Kernel directory not found!")

print("\n4️⃣  Running hardware detection from server...")
sys.path.insert(0, '.')

# Simplified detection
hardware_info = {
    "type": "cpu",
    "name": "CPU",
    "npu_available": False,
    "npu_kernels": 0
}

try:
    if os.path.exists("/dev/accel/accel0"):
        result = subprocess.run(
            ["/opt/xilinx/xrt/bin/xrt-smi", "examine"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and "NPU Phoenix" in result.stdout:
            hardware_info["npu_available"] = True
            hardware_info["type"] = "npu"
            hardware_info["name"] = "AMD Phoenix NPU"

            kernel_dir = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels"
            if os.path.exists(kernel_dir):
                kernels = [f for f in os.listdir(kernel_dir) if f.endswith('.xclbin')]
                hardware_info["npu_kernels"] = len(kernels)

            print(f"   ✅ Hardware detected as: {hardware_info['name']}")
            print(f"   ✅ NPU available: {hardware_info['npu_available']}")
            print(f"   ✅ Kernels found: {hardware_info['npu_kernels']}")
        else:
            print("   ⚠️  XRT command failed or NPU Phoenix not in output")
except Exception as e:
    print(f"   ❌ Detection error: {e}")

print("\n5️⃣  Checking NPU runtime module...")
try:
    sys.path.insert(0, 'npu')
    from npu_runtime_unified import UnifiedNPURuntime
    print("   ✅ NPU runtime module can be imported")

    print("\n6️⃣  Attempting to initialize NPU runtime...")
    try:
        runtime = UnifiedNPURuntime()
        print(f"   ✅ NPU runtime initialized!")
        print(f"   • Mel available: {runtime.mel_available}")
        print(f"   • GELU available: {runtime.gelu_available}")
        print(f"   • Attention available: {runtime.attention_available}")
    except Exception as e:
        print(f"   ❌ Runtime initialization failed: {e}")
        import traceback
        traceback.print_exc()

except ImportError as e:
    print(f"   ❌ Cannot import NPU runtime: {e}")

print("\n" + "="*60)
print("Summary:")
if hardware_info["npu_available"]:
    print("✅ NPU IS DETECTED")
    print(f"   Device: {hardware_info['name']}")
    print(f"   Kernels: {hardware_info['npu_kernels']}")
    print("\n🎯 Server should show NPU if runtime initializes!")
else:
    print("❌ NPU NOT DETECTED")
    print("   Server will show CPU")
print("="*60)

PYTHON_SCRIPT
