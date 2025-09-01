#!/usr/bin/env python3
"""
Verify which device Whisper is actually using during inference
"""

import os
import time
import psutil
import threading
import numpy as np

# Set environment for GPU
os.environ["OV_CACHE_DIR"] = "./cache"
os.environ["OV_GPU_CACHE_MODEL"] = "1"

print("🔍 Monitoring Whisper Model Device Usage")
print("=" * 50)

# CPU monitoring thread
cpu_samples = []
monitoring = True

def monitor_cpu():
    while monitoring:
        cpu_samples.append(psutil.cpu_percent(interval=0.1))
        
# Start CPU monitoring
monitor_thread = threading.Thread(target=monitor_cpu)
monitor_thread.start()

try:
    # Load the model
    print("\n🔄 Loading Whisper model...")
    from whisperx_ov_simple import WhisperXOpenVINO
    
    model = WhisperXOpenVINO("large-v3", device="GPU", compute_type="int8")
    
    # Create test audio
    print("\n🎤 Creating test audio (10 seconds)...")
    audio = np.random.randn(16000 * 10).astype(np.float32) * 0.1  # 10 seconds
    
    # Clear CPU samples before inference
    cpu_samples.clear()
    
    print("\n🚀 Running inference...")
    start_time = time.time()
    
    result = model.transcribe(audio, batch_size=1)
    
    inference_time = time.time() - start_time
    
    # Stop monitoring
    monitoring = False
    monitor_thread.join()
    
    print(f"\n📊 Results:")
    print(f"  - Inference time: {inference_time:.2f}s")
    print(f"  - Text: '{result['text']}'")
    print(f"  - Average CPU usage during inference: {np.mean(cpu_samples):.1f}%")
    print(f"  - Peak CPU usage: {max(cpu_samples):.1f}%")
    
    # Check what device was actually used
    import openvino as ov
    core = ov.Core()
    
    print(f"\n🎯 Device Analysis:")
    if np.mean(cpu_samples) > 80:
        print("  ❌ HIGH CPU usage detected - likely running on CPU!")
    elif np.mean(cpu_samples) > 40:
        print("  ⚠️ Moderate CPU usage - might be partially on CPU")
    else:
        print("  ✅ Low CPU usage - possibly using GPU")
    
    # Try to get more detailed info
    print(f"\n📋 Model Configuration:")
    print(f"  - Requested device: GPU")
    print(f"  - Available devices: {core.available_devices}")
    
    # Check if GPU is being utilized
    try:
        import subprocess
        
        # Try to check Intel GPU usage
        print(f"\n🔍 Checking Intel GPU status...")
        
        # Check if we can access GPU metrics
        gpu_files = [
            "/sys/class/drm/card0/gt/gt0/freq_cur",
            "/sys/class/drm/card1/gt/gt0/freq_cur",
            "/sys/kernel/debug/dri/0/i915_frequency_info"
        ]
        
        for gpu_file in gpu_files:
            if os.path.exists(gpu_file):
                try:
                    with open(gpu_file, 'r') as f:
                        content = f.read().strip()
                        print(f"  - GPU frequency info: {content}")
                        break
                except:
                    pass
        
        # Check OpenVINO cache
        cache_dir = "./cache"
        if os.path.exists(cache_dir):
            cache_files = os.listdir(cache_dir)
            gpu_cache = [f for f in cache_files if 'GPU' in f or 'gpu' in f]
            if gpu_cache:
                print(f"  ✅ GPU cache files found: {len(gpu_cache)} files")
            else:
                print(f"  ❌ No GPU cache files found")
    except Exception as e:
        print(f"  ⚠️ Could not check GPU status: {e}")
        
except Exception as e:
    monitoring = False
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("🔍 Diagnosis:")
if inference_time > 5:
    print("❌ SLOW inference detected - model is likely using CPU")
    print("💡 Recommendation: The model is NOT properly utilizing Intel iGPU")
else:
    print("✅ Fast inference - may be using GPU acceleration")
    
print("\n💡 To force GPU usage, we may need to:")
print("  1. Use a different backend (not OpenVINO)")
print("  2. Use Intel's native GPU libraries directly")
print("  3. Consider using smaller batch sizes")
print("  4. Check if model layers are GPU-compatible")