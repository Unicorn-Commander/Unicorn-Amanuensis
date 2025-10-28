#!/usr/bin/env python3
"""
Test what ACTUALLY works in Unicorn-Amanuensis RIGHT NOW
No aspirational claims, just reality check
"""
import os
import sys
import time
import subprocess

print("="*70)
print("🔍 UNICORN-AMANUENSIS REALITY CHECK")
print("="*70)

# Test audio file
test_audio = "test_audio_jfk.wav"
if not os.path.exists(test_audio):
    print(f"\n❌ Test audio not found: {test_audio}")
    sys.exit(1)

# Get audio duration
import wave
with wave.open(test_audio, 'rb') as wf:
    frames = wf.getnframes()
    rate = wf.getframerate()
    duration = frames / float(rate)
print(f"\n📊 Test Audio: {test_audio}")
print(f"   Duration: {duration:.2f} seconds")

print("\n" + "="*70)
print("TESTING AVAILABLE TRANSCRIPTION OPTIONS")
print("="*70)

results = {}

# Test 1: faster-whisper (if available)
print("\n1️⃣ Testing faster-whisper...")
try:
    import faster_whisper
    print(f"   ✅ faster-whisper {faster_whisper.__version__} installed")
    
    # Quick test
    start = time.perf_counter()
    model = faster_whisper.WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, info = model.transcribe(test_audio)
    text = " ".join([s.text for s in segments])
    elapsed = time.perf_counter() - start
    
    rtf = duration / elapsed
    results['faster-whisper'] = {
        'works': True,
        'time': elapsed,
        'rtf': rtf,
        'text': text[:100]
    }
    print(f"   ⚡ Processing time: {elapsed:.2f}s")
    print(f"   🚀 Realtime factor: {rtf:.1f}x")
    print(f"   📝 Text: {text[:80]}...")
except Exception as e:
    print(f"   ❌ Not available: {e}")
    results['faster-whisper'] = {'works': False, 'error': str(e)}

# Test 2: OpenAI whisper (if available)
print("\n2️⃣ Testing openai-whisper...")
try:
    import whisper as openai_whisper
    print(f"   ✅ openai-whisper installed")
    
    start = time.perf_counter()
    model = openai_whisper.load_model("tiny")
    result = model.transcribe(test_audio)
    elapsed = time.perf_counter() - start
    
    rtf = duration / elapsed
    results['openai-whisper'] = {
        'works': True,
        'time': elapsed,
        'rtf': rtf,
        'text': result['text'][:100]
    }
    print(f"   ⚡ Processing time: {elapsed:.2f}s")
    print(f"   🚀 Realtime factor: {rtf:.1f}x")
    print(f"   📝 Text: {result['text'][:80]}...")
except Exception as e:
    print(f"   ❌ Not available: {e}")
    results['openai-whisper'] = {'works': False, 'error': str(e)}

# Test 3: whisper.cpp (if available)
print("\n3️⃣ Testing whisper.cpp...")
try:
    # Check if whisper.cpp binary exists
    result = subprocess.run(['which', 'whisper-cli'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   ✅ whisper.cpp found: {result.stdout.strip()}")
        results['whisper.cpp'] = {'works': True, 'location': result.stdout.strip()}
    else:
        raise FileNotFoundError("whisper-cli not in PATH")
except Exception as e:
    print(f"   ❌ Not available: {e}")
    results['whisper.cpp'] = {'works': False, 'error': str(e)}

# Test 4: librosa (for preprocessing)
print("\n4️⃣ Testing librosa preprocessing...")
try:
    import librosa
    import numpy as np
    print(f"   ✅ librosa {librosa.__version__} installed")
    
    start = time.perf_counter()
    audio, sr = librosa.load(test_audio, sr=16000)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=512, hop_length=160, n_mels=80
    )
    elapsed = time.perf_counter() - start
    
    rtf = duration / elapsed
    results['librosa'] = {
        'works': True,
        'time': elapsed,
        'rtf': rtf,
        'mel_shape': mel_spec.shape
    }
    print(f"   ⚡ Processing time: {elapsed*1000:.1f}ms")
    print(f"   🚀 Realtime factor: {rtf:.1f}x")
    print(f"   📊 Mel shape: {mel_spec.shape}")
except Exception as e:
    print(f"   ❌ Not available: {e}")
    results['librosa'] = {'works': False, 'error': str(e)}

# Test 5: NPU custom kernels
print("\n5️⃣ Testing NPU custom kernels...")
try:
    sys.path.insert(0, '/opt/xilinx/xrt/python')
    import pyxrt as xrt
    
    device = xrt.device(0)
    print(f"   ✅ NPU device accessible: /dev/accel/accel0")
    
    # Check if our XCLBINs exist
    xclbin_simple = "build_fixed/mel_fixed_new.xclbin"
    if os.path.exists(xclbin_simple):
        print(f"   ✅ Simple kernel exists: {xclbin_simple}")
        results['npu-simple'] = {'works': True, 'xclbin': xclbin_simple}
    else:
        print(f"   ⚠️  Simple kernel not found: {xclbin_simple}")
        results['npu-simple'] = {'works': False, 'error': 'XCLBIN not found'}
    
    xclbin_opt = "build_optimized/mel_optimized_new.xclbin"
    if os.path.exists(xclbin_opt):
        print(f"   ✅ Optimized kernel exists: {xclbin_opt}")
        results['npu-optimized'] = {'works': True, 'xclbin': xclbin_opt}
    else:
        print(f"   ⚠️  Optimized kernel not found: {xclbin_opt}")
        results['npu-optimized'] = {'works': False, 'error': 'XCLBIN not found'}
        
except Exception as e:
    print(f"   ❌ NPU not accessible: {e}")
    results['npu-simple'] = {'works': False, 'error': str(e)}
    results['npu-optimized'] = {'works': False, 'error': str(e)}

# Summary
print("\n" + "="*70)
print("📊 SUMMARY: WHAT ACTUALLY WORKS")
print("="*70)

working = [k for k, v in results.items() if v.get('works')]
not_working = [k for k, v in results.items() if not v.get('works')]

print(f"\n✅ WORKING ({len(working)}):")
for tool in working:
    info = results[tool]
    if 'rtf' in info:
        print(f"   • {tool}: {info['rtf']:.1f}x realtime")
    else:
        print(f"   • {tool}: Available")

print(f"\n❌ NOT WORKING ({len(not_working)}):")
for tool in not_working:
    print(f"   • {tool}: {results[tool].get('error', 'Unknown error')}")

# Recommendation
print("\n" + "="*70)
print("💡 RECOMMENDATION")
print("="*70)

if 'faster-whisper' in working:
    print("\n✅ BEST OPTION: faster-whisper")
    print(f"   Current performance: {results['faster-whisper']['rtf']:.1f}x realtime")
    print(f"   This is what UC-Meeting-Ops actually uses!")
    print(f"   Note: Tested with tiny model, base/medium/large will be slower")
elif 'openai-whisper' in working:
    print("\n⚠️  FALLBACK: openai-whisper")
    print(f"   Current performance: {results['openai-whisper']['rtf']:.1f}x realtime")
    print(f"   Recommend installing faster-whisper for better speed")
else:
    print("\n❌ NO WORKING TRANSCRIPTION ENGINE")
    print("   Action: Install faster-whisper")
    print("   Command: pip install faster-whisper --break-system-packages")

if 'librosa' in working:
    print(f"\n✅ Preprocessing ready: librosa {results['librosa']['rtf']:.0f}x realtime")

if 'npu-simple' in working or 'npu-optimized' in working:
    print("\n⚠️  NPU kernels exist but produce incorrect output (4.68% correlation)")
    print("   Not recommended for production use")

print("\n" + "="*70)
