#!/bin/bash
#
# Whisper ONNX Model Download Script for AMD NPU
# Downloads INT8 quantized models from onnx-community
#

set -e

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/models" && pwd)"

echo "🦄 Unicorn Amanuensis - Model Downloader"
echo "========================================"
echo ""
echo "This script downloads Whisper ONNX INT8 models for AMD NPU acceleration"
echo "Models directory: $MODELS_DIR"
echo ""

# Function to download a model
download_model() {
    local repo_id="$1"
    local output_dir="$2"
    local model_name="$3"

    echo ""
    echo "📥 Downloading $model_name..."
    echo "   Repo: $repo_id"
    echo "   Target: $output_dir"
    echo ""

    python3 << EOF
from huggingface_hub import hf_hub_download
import os

model_id = "$repo_id"
output_dir = "$output_dir"

files = [
    "onnx/encoder_model_int8.onnx",
    "onnx/decoder_model_int8.onnx",
    "onnx/decoder_with_past_model_int8.onnx",
    "config.json",
    "tokenizer.json"
]

print("Downloading files...")
for file in files:
    print(f"  📥 {file}...", end=" ", flush=True)
    try:
        hf_hub_download(
            repo_id=model_id,
            filename=file,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        filepath = os.path.join(output_dir, file)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024*1024)
            print(f"✅ ({size:.1f} MB)")
        else:
            print(f"✅")
    except Exception as e:
        print(f"❌ {e}")

print("")
print(f"✅ {model_id.split('/')[-1]} downloaded to: {output_dir}")
EOF
}

# Check if python3 and huggingface_hub are available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "📦 Installing huggingface_hub..."
    pip3 install --user huggingface_hub
fi

# Create models directory
mkdir -p "$MODELS_DIR"

# Menu
echo "Select models to download:"
echo ""
echo "  1) Whisper Base INT8 (~121 MB) - Fast, good for most uses"
echo "  2) Whisper Large-v3-Turbo INT8 (~1.4 GB) - Best accuracy"
echo "  3) Both models"
echo "  4) Skip download (use existing models)"
echo ""
read -p "Choice [1-4]: " choice

case $choice in
    1)
        download_model \
            "onnx-community/whisper-base" \
            "$MODELS_DIR/whisper-base-onnx-int8" \
            "Whisper Base INT8"
        ;;
    2)
        download_model \
            "onnx-community/whisper-large-v3-turbo" \
            "$MODELS_DIR/whisper-large-v3-onnx-int8" \
            "Whisper Large-v3-Turbo INT8"
        ;;
    3)
        download_model \
            "onnx-community/whisper-base" \
            "$MODELS_DIR/whisper-base-onnx-int8" \
            "Whisper Base INT8"

        download_model \
            "onnx-community/whisper-large-v3-turbo" \
            "$MODELS_DIR/whisper-large-v3-onnx-int8" \
            "Whisper Large-v3-Turbo INT8"
        ;;
    4)
        echo "⏭️  Skipping download"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# Verify models
echo ""
echo "📊 Model Summary:"
echo "================"

if [ -d "$MODELS_DIR/whisper-base-onnx-int8/onnx" ]; then
    base_size=$(du -sh "$MODELS_DIR/whisper-base-onnx-int8" | cut -f1)
    echo "✅ Whisper Base INT8: $base_size"
else
    echo "⚠️  Whisper Base INT8: Not downloaded"
fi

if [ -d "$MODELS_DIR/whisper-large-v3-onnx-int8/onnx" ]; then
    large_size=$(du -sh "$MODELS_DIR/whisper-large-v3-onnx-int8" | cut -f1)
    echo "✅ Whisper Large-v3-Turbo INT8: $large_size"
else
    echo "⚠️  Whisper Large-v3-Turbo INT8: Not downloaded"
fi

echo ""
echo "✅ Done! Models are ready for use."
echo ""
echo "Next steps:"
echo "  1. Build the Docker image: docker compose -f docker-compose-npu.yml build"
echo "  2. Start the service: docker compose -f docker-compose-npu.yml up -d"
echo "  3. Access the GUI: http://localhost:9000/web"
echo ""
