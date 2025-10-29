#!/usr/bin/env python3
"""
Test full Whisper pipeline to find actual bottleneck
"""
import sys
sys.path.insert(0, '/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx')

import onnxruntime as ort
import numpy as np
import librosa
import time

print('='*70)
print('FULL WHISPER PIPELINE - PERFORMANCE BREAKDOWN')
print('='*70)
print()

# Generate test audio (30 seconds)
print('Generating test audio (30s)...')
audio = np.sin(2 * np.pi * 440 * np.linspace(0, 30, 30*16000)).astype(np.float32)
print(f'✅ Audio: {len(audio)/16000:.1f}s @ 16kHz')
print()

# ======================
# STEP 1: MEL PREPROCESSING
# ======================
print('Step 1: Mel Spectrogram (librosa)...')
start = time.perf_counter()

mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=16000,
    n_fft=512,
    hop_length=160,
    n_mels=80,
    fmin=0,
    fmax=8000,
    htk=True,
    power=2.0
)

# Pad to 3000 frames if needed
if mel_spec.shape[1] < 3000:
    mel_spec = np.pad(mel_spec, ((0, 0), (0, 3000 - mel_spec.shape[1])))
else:
    mel_spec = mel_spec[:, :3000]

mel_time = (time.perf_counter() - start) * 1000
mel_input = mel_spec[np.newaxis, :, :].astype(np.float32)

print(f'   Time: {mel_time:.1f} ms')
print(f'   Shape: {mel_input.shape}')
print(f'   Realtime: {30000 / mel_time:.1f}x')
print()

# ======================
# STEP 2: ENCODER
# ======================
print('Step 2: Encoder (ONNX Runtime)...')
encoder_path = 'whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/onnx/encoder_model.onnx'
encoder = ort.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])

start = time.perf_counter()
encoder_out = encoder.run(None, {'input_features': mel_input})
encoder_time = (time.perf_counter() - start) * 1000

print(f'   Time: {encoder_time:.1f} ms')
print(f'   Output shape: {encoder_out[0].shape}')
print(f'   Realtime: {30000 / encoder_time:.1f}x')
print()

# ======================
# STEP 3: DECODER (Single pass test)
# ======================
print('Step 3: Decoder (single forward pass)...')
decoder_path = 'whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/onnx/decoder_model.onnx'
decoder = ort.InferenceSession(decoder_path, providers=['CPUExecutionProvider'])

# Start token
input_ids = np.array([[50258]], dtype=np.int64)  # Start of transcript token

start = time.perf_counter()
decoder_out = decoder.run(
    None,
    {
        'input_ids': input_ids,
        'encoder_hidden_states': encoder_out[0]
    }
)
decoder_single = (time.perf_counter() - start) * 1000

print(f'   Single pass: {decoder_single:.1f} ms')
print(f'   Output shape: {decoder_out[0].shape}')
print()

# Estimate full generation (assume 50 tokens average)
estimated_tokens = 50
estimated_decoder_time = decoder_single * estimated_tokens

print(f'   Estimated for {estimated_tokens} tokens: {estimated_decoder_time:.1f} ms')
print(f'   Est. realtime: {30000 / estimated_decoder_time:.1f}x')
print()

# ======================
# SUMMARY
# ======================
print('='*70)
print('PIPELINE BREAKDOWN (30s audio):')
print('='*70)
total_time = mel_time + encoder_time + estimated_decoder_time

print(f'Mel preprocessing:  {mel_time:6.1f} ms ({mel_time/total_time*100:5.1f}%)')
print(f'Encoder:            {encoder_time:6.1f} ms ({encoder_time/total_time*100:5.1f}%)')
print(f'Decoder (est):      {estimated_decoder_time:6.1f} ms ({estimated_decoder_time/total_time*100:5.1f}%)')
print(f'{"─"*70}')
print(f'Total (estimated):  {total_time:6.1f} ms')
print()

overall_rtf = 30000 / total_time
print(f'Overall Realtime Factor: {overall_rtf:.1f}x')
print()

if overall_rtf >= 220:
    print('✅ ALREADY EXCEEDS 220x TARGET!')
elif overall_rtf >= 180:
    print(f'🎯 Close to target! Need {220/overall_rtf:.2f}x more speedup')
else:
    print(f'⚠️  Need {220/overall_rtf:.1f}x speedup')

print('='*70)
print()
print('🔍 Key Finding: Decoder autoregressive generation is likely bottleneck')
print('   Solution: Optimize beam search, use faster-whisper, or batch tokens')
print('='*70)
