#!/usr/bin/env python3
"""
Debug script to trace KV cache shapes and identify the reshape issue
"""

import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from npu.npu_optimization.onnx_whisper_npu import ONNXWhisperNPU
import librosa

print("=" * 70)
print("KV Cache Debug - Shape Tracing")
print("=" * 70)
print()

# Initialize decoder
decoder = ONNXWhisperNPU()
if not decoder.initialize(model_size='base'):
    print("Failed to initialize")
    sys.exit(1)

# Load short audio
audio_file = '/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/test_audio_jfk.wav'
audio, sample_rate = librosa.load(audio_file, sr=16000)
audio_duration = len(audio) / sample_rate
print(f"Audio: {audio_duration:.1f}s")

# Extract mel features
mel_features = decoder.extract_mel_features(audio, sample_rate)
input_features = np.expand_dims(mel_features, axis=0)

# Pad to 3000
if input_features.shape[2] < 3000:
    padding = 3000 - input_features.shape[2]
    input_features = np.pad(input_features, ((0, 0), (0, 0), (0, padding)), mode='constant')

print(f"Input features shape: {input_features.shape}")

# Run encoder
encoder_outputs = decoder.encoder_session.run(None, {
    'input_features': input_features
})
hidden_states = encoder_outputs[0]
print(f"Encoder output: {hidden_states.shape}")
print()

# Tokenizer
from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")

# Initial tokens
decoder_input_ids = np.array([[50258, 50259, 50360, 50365]], dtype=np.int64)
print(f"Initial input_ids: {decoder_input_ids.shape} - {decoder_input_ids}")
print()

# First pass - regular decoder
print("=" * 70)
print("FIRST DECODER PASS (no KV cache)")
print("=" * 70)
print(f"Input: input_ids={decoder_input_ids.shape}, encoder_hidden_states={hidden_states.shape}")

decoder_outputs = decoder.decoder_session.run(None, {
    'input_ids': decoder_input_ids,
    'encoder_hidden_states': hidden_states
})

print(f"Number of outputs: {len(decoder_outputs)}")
print(f"Logits shape: {decoder_outputs[0].shape}")
print()

# Extract KV cache
if len(decoder_outputs) == 25:
    print("KV cache outputs (present tensors):")
    past_key_values = []
    for i in range(6):  # 6 decoder layers
        decoder_key = decoder_outputs[i*2 + 1]
        decoder_value = decoder_outputs[i*2 + 2]
        encoder_key = decoder_outputs[i*2 + 13]
        encoder_value = decoder_outputs[i*2 + 14]

        print(f"  Layer {i}:")
        print(f"    decoder.key:   {decoder_key.shape}")
        print(f"    decoder.value: {decoder_value.shape}")
        print(f"    encoder.key:   {encoder_key.shape}")
        print(f"    encoder.value: {encoder_value.shape}")

        past_key_values.append((
            decoder_key,
            decoder_value,
            encoder_key,
            encoder_value
        ))
    print()

    # Get next token
    logits = decoder_outputs[0]
    next_token_id = np.argmax(logits[0, -1, :])
    print(f"Next token: {next_token_id} -> '{tokenizer.decode([next_token_id])}'")

    # Add to sequence
    decoder_input_ids = np.concatenate([
        decoder_input_ids,
        np.array([[next_token_id]], dtype=np.int64)
    ], axis=1)
    print(f"Updated input_ids: {decoder_input_ids.shape}")
    print()

    # Second pass - with KV cache
    print("=" * 70)
    print("SECOND DECODER PASS (with KV cache)")
    print("=" * 70)
    print(f"Input: input_ids={decoder_input_ids[:, -1:].shape} (last token only)")

    # Build inputs with KV cache
    inputs = {'input_ids': decoder_input_ids[:, -1:]}
    for i, kv in enumerate(past_key_values):
        inputs[f'past_key_values.{i}.decoder.key'] = kv[0]
        inputs[f'past_key_values.{i}.decoder.value'] = kv[1]
        inputs[f'past_key_values.{i}.encoder.key'] = kv[2]
        inputs[f'past_key_values.{i}.encoder.value'] = kv[3]

        print(f"  past_key_values.{i}.decoder.key: {kv[0].shape}")
        print(f"  past_key_values.{i}.decoder.value: {kv[1].shape}")

    print()

    # Run decoder with past
    try:
        decoder_outputs = decoder.decoder_with_past_session.run(None, inputs)
        print(f"✅ SUCCESS!")
        print(f"Logits shape: {decoder_outputs[0].shape}")

        # Show new KV shapes
        print()
        print("New KV cache outputs (present tensors):")
        for i in range(6):
            print(f"  Layer {i}:")
            print(f"    present.{i}.decoder.key:   {decoder_outputs[i*2 + 1].shape}")
            print(f"    present.{i}.decoder.value: {decoder_outputs[i*2 + 2].shape}")

        # Test concatenation
        print()
        print("=" * 70)
        print("TESTING KV CACHE CONCATENATION")
        print("=" * 70)
        for i in range(6):
            old_key = past_key_values[i][0]
            new_key = decoder_outputs[i*2 + 1]

            print(f"Layer {i}:")
            print(f"  Old decoder.key: {old_key.shape}")
            print(f"  New decoder.key: {new_key.shape}")

            try:
                concatenated = np.concatenate([old_key, new_key], axis=2)
                print(f"  ✅ Concatenated:  {concatenated.shape}")
            except Exception as e:
                print(f"  ❌ Concatenation failed: {e}")
            print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()

else:
    print(f"❌ Unexpected number of outputs: {len(decoder_outputs)} (expected 25)")
