#!/bin/bash
#
# Download NPU-optimized Whisper models
# Run this script to download the actual model files
#

set -e

echo "🦄 Downloading NPU-optimized Whisper Models"
echo "============================================"
echo ""

# Function to download if not exists
download_if_missing() {
    local repo=$1
    local dest=$2

    if [ ! -d "$dest" ] || [ -z "$(ls -A $dest/*.onnx 2>/dev/null)" ]; then
        echo "📥 Downloading from $repo..."
        python3 << EOF
from huggingface_hub import snapshot_download
try:
    path = snapshot_download(
        repo_id="$repo",
        local_dir="$dest",
        token=False
    )
    print(f"✅ Downloaded to: {path}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
EOF
    else
        echo "✅ Already downloaded: $dest"
    fi
}

# Whisper Base - Already available in main whisperx cache
echo "1️⃣  Whisper Base (122 MB)"
echo "   Location: ../whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/"
if [ -d "../whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/onnx/" ]; then
    echo "   ✅ Already available"
else
    echo "   ⚠️  Not found - will be downloaded when needed"
fi
echo ""

# Whisper Medium
echo "2️⃣  Whisper Medium (1.3 GB)"
download_if_missing "PraveenJesu/whisper-medium-v2.2.4_onnx_quantized" "whisper-medium-int8"
echo ""

# Whisper Large
echo "3️⃣  Whisper Large (2.4 GB)"
download_if_missing "Intel/whisper-large-int8-static-inc" "whisper-large-int8"
echo ""

echo "✅ Model download complete!"
echo ""
echo "📊 Disk usage:"
du -sh whisper-* 2>/dev/null || true
echo ""
echo "🚀 Models are ready for NPU acceleration"
echo "   See INSTALL.md for usage instructions"
