#!/usr/bin/env python3
"""
Comprehensive NPU Benchmark: Custom Runtime vs VitisAI EP vs faster_whisper
Tests all three approaches and compares performance
"""

import time
import sys
import os
from pathlib import Path

print("="*70)
print("🦄 Unicorn Amanuensis - Comprehensive NPU Benchmark")
print("="*70)
print()

# Test audio file
test_audio = "/app/whisper-cpp-igpu/bindings/go/samples/jfk.wav"
if not Path(test_audio).exists():
    print(f"❌ Test audio not found: {test_audio}")
    sys.exit(1)

print(f"📁 Test audio: {test_audio}")
print()

results = {}

# Test 1: Current NPU Hybrid (ONNX + Custom Kernels)
print("="*70)
print("Test 1: Custom NPU Hybrid Runtime (Current Implementation)")
print("="*70)
try:
    sys.path.insert(0, '/app/npu')
    from npu_runtime import NPURuntime
    
    runtime = NPURuntime()
    if runtime.is_available():
        print("✅ Custom NPU runtime available")
        start = time.time()
        result = runtime.transcribe(test_audio)
        elapsed = time.time() - start
        
        results['custom_npu'] = {
            'time': elapsed,
            'text': result.get('text', '')[:100],
            'status': 'success' if 'error' not in result else 'error'
        }
        print(f"⏱️  Time: {elapsed:.2f}s")
        print(f"📝 Text: {result.get('text', '')[:100]}...")
    else:
        print("⚠️  NPU runtime not available")
        results['custom_npu'] = {'status': 'unavailable'}
except Exception as e:
    print(f"❌ Error: {e}")
    results['custom_npu'] = {'status': 'error', 'error': str(e)}

print()

# Test 2: VitisAI Execution Provider (if available)
print("="*70)
print("Test 2: VitisAI Execution Provider (AMD Official)")
print("="*70)
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"Available ONNX providers: {providers}")
    
    if 'VitisAIExecutionProvider' in providers:
        print("✅ VitisAI EP available - testing...")
        # Test with VitisAI EP
        # TODO: Implement when VitisAI is installed
        results['vitisai'] = {'status': 'available_not_tested'}
    else:
        print("⚠️  VitisAI EP not installed")
        print("   Install Ryzen AI Software to enable")
        results['vitisai'] = {'status': 'not_installed'}
except Exception as e:
    print(f"❌ Error: {e}")
    results['vitisai'] = {'status': 'error', 'error': str(e)}

print()

# Test 3: faster_whisper (Production Baseline)
print("="*70)
print("Test 3: faster_whisper (Production Baseline)")
print("="*70)
try:
    from faster_whisper import WhisperModel
    
    print("✅ faster_whisper available")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    start = time.time()
    segments, info = model.transcribe(test_audio, beam_size=5)
    text = " ".join([seg.text for seg in segments])
    elapsed = time.time() - start
    
    results['faster_whisper'] = {
        'time': elapsed,
        'text': text[:100],
        'status': 'success'
    }
    print(f"⏱️  Time: {elapsed:.2f}s")
    print(f"📝 Text: {text[:100]}...")
except Exception as e:
    print(f"❌ Error: {e}")
    results['faster_whisper'] = {'status': 'error', 'error': str(e)}

print()

# Results Summary
print("="*70)
print("📊 BENCHMARK RESULTS SUMMARY")
print("="*70)
print()

for approach, data in results.items():
    print(f"**{approach.upper()}**:")
    if data['status'] == 'success':
        print(f"  ✅ Time: {data['time']:.2f}s")
        print(f"  📝 Text: {data['text']}")
    else:
        print(f"  ⚠️  Status: {data['status']}")
    print()

# Calculate speedups if we have timings
if 'faster_whisper' in results and results['faster_whisper']['status'] == 'success':
    baseline = results['faster_whisper']['time']
    print(f"Baseline (faster_whisper): {baseline:.2f}s")
    
    if 'custom_npu' in results and results['custom_npu']['status'] == 'success':
        speedup = baseline / results['custom_npu']['time']
        print(f"Custom NPU speedup: {speedup:.1f}x")
    
    if 'vitisai' in results and results['vitisai']['status'] == 'success':
        speedup = baseline / results['vitisai']['time']
        print(f"VitisAI EP speedup: {speedup:.1f}x")

print()
print("="*70)
print("🎯 Next Steps:")
if results.get('vitisai', {}).get('status') == 'not_installed':
    print("  1. Download Ryzen AI Software from AMD")
    print("  2. Install onnxruntime_vitisai wheel")
    print("  3. Re-run this benchmark")
print("="*70)
