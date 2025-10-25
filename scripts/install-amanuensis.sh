#!/bin/bash
set -e

echo "🦄 Installing Unicorn Amanuensis (STT Service)"
echo "================================================"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ Error: requirements.txt not found${NC}"
    echo "Please run this script from the Unicorn-Amanuensis directory"
    exit 1
fi

echo ""
echo "📦 Step 1: Installing unicorn-npu-core library"
echo "-----------------------------------------------"

# Check if unicorn-npu-core exists
CORE_PATH="/home/ucadmin/UC-1/unicorn-npu-core"
if [ ! -d "$CORE_PATH" ]; then
    echo -e "${RED}❌ unicorn-npu-core not found at: $CORE_PATH${NC}"
    echo "Please ensure unicorn-npu-core is cloned/created first"
    exit 1
fi

# Install unicorn-npu-core in development mode
cd "$CORE_PATH"
pip install -e . --break-system-packages
cd - > /dev/null

echo -e "${GREEN}✅ unicorn-npu-core installed${NC}"

echo ""
echo "📦 Step 2: Setting up NPU on host system"
echo "-----------------------------------------------"

# Run NPU host setup (via core library)
python3 -c "from unicorn_npu.scripts.install_host import main; main()" || true

echo ""
echo "📦 Step 3: Installing Amanuensis dependencies"
echo "-----------------------------------------------"

# Install Amanuensis requirements
pip install -r requirements.txt --break-system-packages

echo -e "${GREEN}✅ Dependencies installed${NC}"

echo ""
echo "📦 Step 4: Downloading Whisper models (optional)"
echo "-----------------------------------------------"

# Check if download script exists
if [ -f "download_onnx_models.sh" ]; then
    read -p "Download Whisper ONNX models? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        bash download_onnx_models.sh
    else
        echo "Skipping model download"
    fi
else
    echo -e "${YELLOW}⚠️  Model download script not found${NC}"
    echo "Models can be downloaded later"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}🎉 Amanuensis installation complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Log out and log back in (for group changes)"
echo "2. Set NPU to performance mode:"
echo "   python3 -c \"from unicorn_npu import NPUDevice; npu = NPUDevice(); npu.set_power_mode('performance')\""
echo "3. Start the service:"
echo "   cd whisperx && python3 server_whisperx_npu.py"
echo ""
