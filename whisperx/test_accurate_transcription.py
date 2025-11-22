#!/usr/bin/env python3
"""
Test accurate transcription with Whisper base and large-v3 models.
Uses faster-whisper's built-in CTranslate2 mel preprocessing for accuracy.
"""

import requests
import time
from pathlib import Path

SERVER_URL = "http://localhost:9004"

# Find a test audio file
test_files = [
    "/home/ucadmin/VibeVoice/Shafen_Khan_call.m4a",
    "/home/ucadmin/UC-1/test_audio.wav",
    "/tmp/test.wav"
]

test_file = None
for f in test_files:
    if Path(f).exists():
        test_file = f
        break

if not test_file:
    print("❌ No test audio file found!")
    exit(1)

print(f"🎵 Testing with: {test_file}")
print(f"📦 File size: {Path(test_file).stat().st_size / 1024:.1f} KB")
print()

# Test with Whisper base
print("=" * 70)
print("📊 TEST 1: Whisper BASE Model")
print("=" * 70)

start = time.time()
with open(test_file, 'rb') as f:
    files = {'file': f}
    data = {'model': 'base', 'vad_filter': 'false'}

    print("⏳ Sending request...")
    response = requests.post(f"{SERVER_URL}/transcribe", files=files, data=data, timeout=300)

elapsed = time.time() - start

if response.status_code == 200:
    result = response.json()
    segments = result.get('segments', [])
    text = result.get('text', '')

    print(f"✅ Transcription SUCCESS")
    print(f"⏱️  Processing time: {elapsed:.2f}s")
    print(f"📝 Segments: {len(segments)}")
    print(f"📄 Text length: {len(text)} chars")
    print()
    print("First 500 chars:")
    print(text[:500])
    print()
else:
    print(f"❌ ERROR: {response.status_code}")
    print(response.text)

print()

# Test with Whisper large-v3
print("=" * 70)
print("📊 TEST 2: Whisper LARGE-V3 Model")
print("=" * 70)

start = time.time()
with open(test_file, 'rb') as f:
    files = {'file': f}
    data = {'model': 'large-v3', 'vad_filter': 'false'}

    print("⏳ Sending request...")
    response = requests.post(f"{SERVER_URL}/transcribe", files=files, data=data, timeout=600)

elapsed = time.time() - start

if response.status_code == 200:
    result = response.json()
    segments = result.get('segments', [])
    text = result.get('text', '')

    print(f"✅ Transcription SUCCESS")
    print(f"⏱️  Processing time: {elapsed:.2f}s")
    print(f"📝 Segments: {len(segments)}")
    print(f"📄 Text length: {len(text)} chars")
    print()
    print("First 500 chars:")
    print(text[:500])
    print()
else:
    print(f"❌ ERROR: {response.status_code}")
    print(response.text)

print()
print("=" * 70)
print("✅ All tests completed!")
print("=" * 70)
