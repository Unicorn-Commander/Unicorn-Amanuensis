#!/usr/bin/env python3
"""
Check the decoder-with-past model's input/output structure
"""

import onnxruntime as ort
import numpy as np

# Load the decoder-with-past model
model_path = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/onnx/decoder_with_past_model.onnx"

session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

print("=" * 70)
print("Whisper ONNX Decoder-with-Past Structure")
print("=" * 70)
print()

print("Decoder-with-Past Inputs:")
for inp in session.get_inputs():
    print(f"  {inp.name}: {inp.shape}")
print()

print("Decoder-with-Past Outputs:")
for i, out in enumerate(session.get_outputs()):
    print(f"  [{i:2d}] {out.name}: {out.shape}")
print()

# Count inputs
decoder_kv_inputs = [inp for inp in session.get_inputs() if 'decoder.key' in inp.name or 'decoder.value' in inp.name]
encoder_kv_inputs = [inp for inp in session.get_inputs() if 'encoder.key' in inp.name or 'encoder.value' in inp.name]

print(f"Total inputs: {len(session.get_inputs())}")
print(f"Decoder KV inputs: {len(decoder_kv_inputs)}")
print(f"Encoder KV inputs: {len(encoder_kv_inputs)}")
print()

# Show expected KV input format
print("Expected KV input names (first 3 layers):")
for i in range(3):
    print(f"  Layer {i}:")
    for inp in session.get_inputs():
        if f'.{i}.' in inp.name and 'past_key_values' in inp.name:
            print(f"    {inp.name}")
