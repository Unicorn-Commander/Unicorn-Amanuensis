#!/usr/bin/env python3
"""
Test Intel Extension for PyTorch (IPEX) GPU acceleration
"""

import torch
import intel_extension_for_pytorch as ipex
import time
import numpy as np

print("🔍 Testing Intel Extension for PyTorch (IPEX)")
print("=" * 50)

# Check available devices
print("\n📊 Available devices:")
print(f"  - CUDA available: {torch.cuda.is_available()}")
print(f"  - XPU available: {torch.xpu.is_available()}")
print(f"  - CPU available: True")

if torch.xpu.is_available():
    print(f"\n✅ Intel XPU (GPU) detected!")
    print(f"  - Device count: {torch.xpu.device_count()}")
    print(f"  - Device name: {torch.xpu.get_device_name(0)}")
    
    # Test on XPU
    print("\n🚀 Testing on Intel XPU (iGPU)...")
    
    # Create test model
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024)
    )
    
    # Move to XPU
    model = model.to('xpu')
    model = ipex.optimize(model)
    
    # Test input
    input_data = torch.randn(32, 1024).to('xpu')
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_data)
    
    # Benchmark
    torch.xpu.synchronize()
    start = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            _ = model(input_data)
    
    torch.xpu.synchronize()
    elapsed = time.time() - start
    
    print(f"  ⏱️ Time: {elapsed:.2f}s for 100 iterations")
    print(f"  📈 FPS: {100/elapsed:.1f}")
    print(f"  🎯 Latency: {elapsed/100*1000:.1f}ms per inference")
    
else:
    print("\n❌ Intel XPU not available")
    print("💡 IPEX XPU support requires Intel GPU drivers and runtime")
    
    # Try CPU optimization instead
    print("\n🔄 Testing IPEX CPU optimization...")
    
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024)
    )
    
    # Optimize for CPU (eval mode)
    model.eval()
    model = ipex.optimize(model)
    
    input_data = torch.randn(32, 1024)
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_data)
    
    # Benchmark
    start = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            _ = model(input_data)
    
    elapsed = time.time() - start
    
    print(f"  ⏱️ Time: {elapsed:.2f}s for 100 iterations")
    print(f"  📈 FPS: {100/elapsed:.1f}")
    print(f"  🎯 Latency: {elapsed/100*1000:.1f}ms per inference")

print("\n" + "=" * 50)
print("📋 Summary:")
if torch.xpu.is_available():
    print("✅ Intel GPU (XPU) is available through IPEX!")
    print("🎯 This provides TRUE hardware acceleration on Intel iGPU")
else:
    print("❌ Intel GPU (XPU) support not available")
    print("\n💡 To enable Intel GPU support:")
    print("  1. Install Intel GPU drivers")
    print("  2. Install Intel Compute Runtime")
    print("  3. Install Level Zero loader")
    print("  4. Set environment variables:")
    print("     export ONEAPI_ROOT=/opt/intel/oneapi")
    print("     source /opt/intel/oneapi/setvars.sh")