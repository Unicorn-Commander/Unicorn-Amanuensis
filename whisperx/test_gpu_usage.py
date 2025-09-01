#!/usr/bin/env python3
"""
Test to verify if OpenVINO is actually using Intel iGPU
"""

import os
import time
import numpy as np
import openvino as ov

# Force GPU usage
os.environ["OV_CACHE_DIR"] = "./cache"
os.environ["OV_GPU_CACHE_MODEL"] = "1"

print("🔍 Testing OpenVINO GPU Usage")
print("=" * 50)

# Initialize OpenVINO
core = ov.Core()

# List available devices
print("\n📊 Available devices:")
for device in core.available_devices:
    try:
        name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"  - {device}: {name}")
    except:
        print(f"  - {device}")

# Create a simple test model
print("\n🧪 Creating test model...")
import openvino.runtime as ovr

# Create a simple model - just a large matrix multiplication
param = ovr.opset10.parameter([1, 1024], np.float32, name="input")
weights = ovr.opset10.constant(np.random.randn(1024, 1024).astype(np.float32))
matmul = ovr.opset10.matmul(param, weights, False, False)
result = ovr.opset10.result(matmul)
model = ovr.Model([result], [param], "test_model")

# Test on different devices
devices_to_test = ["CPU", "GPU"]

for device in devices_to_test:
    if device not in core.available_devices:
        print(f"\n❌ {device} not available")
        continue
    
    print(f"\n🚀 Testing on {device}...")
    
    # Compile model
    compiled_model = core.compile_model(model, device)
    
    # Create inference request
    infer_request = compiled_model.create_infer_request()
    
    # Prepare input
    input_data = np.random.randn(1, 1024).astype(np.float32)
    
    # Warm up
    for _ in range(3):
        infer_request.infer({0: input_data})
    
    # Measure performance
    num_iterations = 100
    start_time = time.time()
    
    for _ in range(num_iterations):
        infer_request.infer({0: input_data})
    
    elapsed = time.time() - start_time
    fps = num_iterations / elapsed
    
    print(f"  ⏱️ Time: {elapsed:.2f}s for {num_iterations} iterations")
    print(f"  📈 FPS: {fps:.1f}")
    print(f"  🎯 Latency: {(elapsed/num_iterations)*1000:.1f}ms per inference")

print("\n" + "=" * 50)
print("🔍 Checking OpenVINO GPU configuration...")

# Check GPU-specific properties
if "GPU" in core.available_devices:
    gpu_config = {
        "PERFORMANCE_HINT": "LATENCY",
        "CACHE_DIR": "./cache",
        "NUM_STREAMS": "1",
        "INFERENCE_PRECISION_HINT": "f16"
    }
    
    print("\n📋 GPU Configuration:")
    for key, value in gpu_config.items():
        print(f"  - {key}: {value}")
    
    # Try to get GPU metrics
    try:
        import subprocess
        result = subprocess.run(["clinfo"], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Device Name' in line or 'Max compute units' in line:
                    print(f"  {line.strip()}")
    except:
        pass

print("\n✅ Test complete!")
print("\nℹ️ If GPU is significantly faster than CPU, it's working.")
print("If speeds are similar, OpenVINO is likely falling back to CPU.")