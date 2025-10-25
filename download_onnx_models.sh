#!/bin/bash
# Download ONNX Whisper models from HuggingFace
set -e

echo "📥 Downloading ONNX Whisper models from HuggingFace..."

# Create model directory
mkdir -p models/whisper-base-onnx/onnx

# Base URL
BASE_URL="https://huggingface.co/onnx-community/whisper-base/resolve/main/onnx"

# Download encoder
echo "Downloading encoder model..."
wget -q --show-progress "$BASE_URL/encoder_model.onnx" -O "models/whisper-base-onnx/onnx/encoder_model.onnx"

# Download decoder  
echo "Downloading decoder model..."
wget -q --show-progress "$BASE_URL/decoder_model.onnx" -O "models/whisper-base-onnx/onnx/decoder_model.onnx"

# Download decoder with past (optional)
echo "Downloading decoder_with_past model (optional)..."
wget -q --show-progress "$BASE_URL/decoder_with_past_model.onnx" -O "models/whisper-base-onnx/onnx/decoder_with_past_model.onnx" || echo "decoder_with_past not found, skipping..."

# Download config
echo "Downloading config..."
wget -q --show-progress "https://huggingface.co/onnx-community/whisper-base/resolve/main/config.json" -O "models/whisper-base-onnx/config.json" || true

# Download tokenizer
echo "Downloading tokenizer..."
wget -q --show-progress "https://huggingface.co/onnx-community/whisper-base/resolve/main/tokenizer.json" -O "models/whisper-base-onnx/tokenizer.json" || true

echo "✅ ONNX models downloaded successfully!"
ls -lh models/whisper-base-onnx/onnx/
