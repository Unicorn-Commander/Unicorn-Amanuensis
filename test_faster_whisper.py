#!/usr/bin/env python3
"""
Test faster-whisper performance (optimized decoder)
"""
from faster_whisper import WhisperModel
import numpy as np
import time

print('='*70)
print('FASTER-WHISPER PERFORMANCE TEST')
print('='*70)
print()

# Generate test audio
print('Generating test audio (30s)...')
audio = np.sin(2 * np.pi * 440 * np.linspace(0, 30, 30*16000)).astype(np.float32)
print(f'✅ Audio: {len(audio)/16000:.1f}s @ 16kHz')
print()

# Load model
print('Loading faster-whisper model (base, int8)...')
model = WhisperModel("base", device="cpu", compute_type="int8")
print('✅ Model loaded')
print()

# Transcribe
print('Transcribing...')
start = time.perf_counter()

segments, info = model.transcribe(
    audio,
    beam_size=5,
    language="en"
)

# Consume segments iterator
transcription = ""
for segment in segments:
    transcription += segment.text

total_time = (time.perf_counter() - start) * 1000

print()
print('='*70)
print('RESULTS:')
print('='*70)
print(f'Total time: {total_time:.1f} ms')
print(f'Audio duration: 30s')
print(f'Realtime factor: {30000 / total_time:.1f}x')
print()
print(f'Transcription: "{transcription[:100]}..."')
print()

if 30000 / total_time >= 220:
    print('🎉 ALREADY EXCEEDS 220x TARGET!')
elif 30000 / total_time >= 180:
    print(f'🎯 Very close! Only need {220 / (30000/total_time):.2f}x more')
else:
    print(f'⚠️  Need {220 / (30000/total_time):.1f}x more speedup')

print('='*70)
