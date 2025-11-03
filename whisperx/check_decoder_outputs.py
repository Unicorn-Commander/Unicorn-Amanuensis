#!/usr/bin/env python3
"""
Check the actual output structure of the Whisper ONNX decoder
"""

import onnxruntime as ort
import numpy as np

# Load the decoder model
model_path = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/onnx/decoder_model.onnx"

session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

print("=" * 70)
print("Whisper ONNX Decoder Output Structure")
print("=" * 70)
print()

print("Decoder Inputs:")
for inp in session.get_inputs():
    print(f"  {inp.name}: {inp.shape}")
print()

print("Decoder Outputs:")
for i, out in enumerate(session.get_outputs()):
    print(f"  [{i:2d}] {out.name}: {out.shape}")
print()

# Test run
input_ids = np.array([[50258, 50259, 50360, 50365]], dtype=np.int64)
encoder_hidden_states = np.random.randn(1, 1500, 512).astype(np.float32)

print("Running test inference...")
outputs = session.run(None, {
    'input_ids': input_ids,
    'encoder_hidden_states': encoder_hidden_states
})

print(f"Number of outputs: {len(outputs)}")
print()

print("Output shapes:")
for i, out in enumerate(outputs):
    print(f"  [{i:2d}] {out.shape}")
print()

# Try to identify the pattern
print("=" * 70)
print("Pattern Analysis")
print("=" * 70)
print()

# Logits should be (1, 4, 51865)
print(f"Output 0 (logits): {outputs[0].shape}")
print()

# Group remaining outputs
print("KV cache tensors (outputs 1-24):")
for i in range(1, min(25, len(outputs))):
    shape = outputs[i].shape
    name = session.get_outputs()[i].name
    print(f"  [{i:2d}] {name:50s} {shape}")
