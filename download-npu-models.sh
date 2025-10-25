#!/bin/bash
# Download AMD NPU models for Unicorn Amanuensis

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}📥 Downloading AMD NPU Whisper models from magicunicorn...${NC}"
echo ""

# Navigate to whisperx directory
cd "$(dirname "$0")/whisperx"

# Fix permissions
echo -e "${YELLOW}🔧 Fixing permissions...${NC}"
sudo chown -R $USER:$USER models/ 2>/dev/null || true

# Create directories
mkdir -p models/whisper-base-amd-npu-int8
mkdir -p models/whisperx-large-v3-npu

# Download base model
echo -e "${BLUE}📥 Downloading whisper-base-amd-npu-int8...${NC}"
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

os.chdir("models")
snapshot_download(
    "magicunicorn/whisper-base-amd-npu-int8",
    local_dir="whisper-base-amd-npu-int8"
)
print("✅ Base model downloaded!")
EOF

# Download large-v3 model
echo -e "${BLUE}📥 Downloading whisperx-large-v3-npu...${NC}"
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

os.chdir("models")
snapshot_download(
    "magicunicorn/whisperx-large-v3-npu",
    local_dir="whisperx-large-v3-npu"
)
print("✅ Large-v3 model downloaded!")
EOF

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   🎉 Models Downloaded! 🎉              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""

# List downloaded files
echo -e "${YELLOW}📁 Downloaded files:${NC}"
ls -lh models/whisper-base-amd-npu-int8/
echo ""
ls -lh models/whisperx-large-v3-npu/

echo ""
echo -e "${GREEN}✅ Ready to rebuild container!${NC}"
echo "Run: ${BLUE}./rebuild-npu.sh${NC}"
