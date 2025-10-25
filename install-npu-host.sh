#!/bin/bash
#
# Unicorn Amanuensis - NPU Host System Setup
# This script must be run on the HOST system before deploying Docker containers
#
# Prerequisites:
# - AMD Ryzen 7040/8040 series CPU with Phoenix NPU
# - Ubuntu 22.04+ or compatible Linux distribution
# - sudo access
#

set -e

echo "🦄 Unicorn Amanuensis - NPU Host System Setup"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then
   echo -e "${RED}❌ Please do NOT run this script as root${NC}"
   echo "   Run as regular user with sudo access: ./install-npu-host.sh"
   exit 1
fi

# Function to check command existence
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Step 1: Checking Prerequisites"
echo "-------------------------------"

# Check for AMD NPU
if [ ! -e "/sys/bus/pci/devices/0000:c7:00.1" ] && [ ! -e "/dev/accel/accel0" ]; then
    echo -e "${YELLOW}⚠️  Warning: AMD NPU may not be present${NC}"
    echo "   This setup requires AMD Ryzen 7040/8040 series with Phoenix NPU"
    read -p "   Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Step 2: Installing XRT for AMD NPU"
echo "-----------------------------------"

# Check if XRT is already installed
if [ -f "/opt/xilinx/xrt/bin/xrt-smi" ]; then
    echo -e "${GREEN}✅ XRT already installed${NC}"
    /opt/xilinx/xrt/bin/xrt-smi examine | head -5 || true
else
    echo "📥 Installing XRT (Xilinx Runtime) for AMD NPU..."
    echo "   This will take 15-30 minutes..."

    # Install build dependencies
    sudo apt-get update
    sudo apt-get install -y \\
        build-essential \\
        cmake \\
        git \\
        libboost-all-dev \\
        libssl-dev \\
        uuid-dev \\
        libudev-dev \\
        libdrm-dev \\
        pkg-config

    # Clone and build XRT
    cd ~
    if [ ! -d "xdna-driver" ]; then
        git clone https://github.com/amd/xdna-driver
    fi
    cd xdna-driver

    echo "🔨 Building XRT..."
    ./build.sh -release

    echo "📦 Installing XRT package..."
    sudo dpkg -i build/Release/*.deb

    # Load kernel module
    sudo modprobe amdxdna || echo "Module may need reboot to load"

    cd ~/UC-1/Unicorn-Amanuensis
fi

echo ""
echo "Step 3: Verifying NPU Device"
echo "-----------------------------"

if [ -e "/dev/accel/accel0" ]; then
    echo -e "${GREEN}✅ NPU device found: /dev/accel/accel0${NC}"
    ls -la /dev/accel/accel0
else
    echo -e "${YELLOW}⚠️  NPU device not found. Trying to load module...${NC}"
    sudo modprobe amdxdna || echo "Failed to load module"
    sleep 2
    if [ -e "/dev/accel/accel0" ]; then
        echo -e "${GREEN}✅ NPU device now available${NC}"
    else
        echo -e "${RED}❌ NPU device still not found${NC}"
        echo "   You may need to reboot for the driver to load"
    fi
fi

# Check DRI devices
if [ -e "/dev/dri/card0" ]; then
    echo -e "${GREEN}✅ DRI devices found${NC}"
    ls -la /dev/dri/ | head -5
else
    echo -e "${RED}❌ DRI devices not found${NC}"
    echo "   These are required for XRT memory operations"
fi

echo ""
echo "Step 4: Setting Up User Permissions"
echo "------------------------------------"

# Add user to render group (for NPU access)
if groups | grep -q render; then
    echo -e "${GREEN}✅ User already in render group${NC}"
else
    echo "➕ Adding user to render group..."
    sudo usermod -aG render $USER
    echo -e "${YELLOW}⚠️  You must LOG OUT and LOG BACK IN for group changes to take effect${NC}"
fi

# Add user to docker group
if groups | grep -q docker; then
    echo -e "${GREEN}✅ User already in docker group${NC}"
else
    if command_exists docker; then
        echo "➕ Adding user to docker group..."
        sudo usermod -aG docker $USER
        echo -e "${YELLOW}⚠️  You must LOG OUT and LOG BACK IN for group changes to take effect${NC}"
    fi
fi

echo ""
echo "Step 5: Installing Docker (if needed)"
echo "--------------------------------------"

if command_exists docker; then
    echo -e "${GREEN}✅ Docker already installed${NC}"
    docker --version
else
    echo "📥 Installing Docker..."
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo sh /tmp/get-docker.sh
    rm /tmp/get-docker.sh
    echo -e "${GREEN}✅ Docker installed${NC}"
fi

if command_exists docker-compose || docker compose version >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker Compose available${NC}"
else
    echo "📥 Installing Docker Compose plugin..."
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin
fi

echo ""
echo "Step 6: Downloading ONNX Whisper Models"
echo "----------------------------------------"

MODELS_DIR="./models/whisper_onnx_cache"
mkdir -p "$MODELS_DIR"

if [ -f "$MODELS_DIR/models--onnx-community--whisper-base/onnx/encoder_model.onnx" ]; then
    echo -e "${GREEN}✅ Whisper base model already downloaded${NC}"
else
    echo "📥 Downloading ONNX Whisper base model..."
    echo "   This is ~500MB and will take a few minutes..."
    cd "$MODELS_DIR"

    # Check if huggingface-cli is available
    if command_exists huggingface-cli || command_exists hf; then
        huggingface-cli download onnx-community/whisper-base \\
            --local-dir models--onnx-community--whisper-base || \\
        hf download onnx-community/whisper-base \\
            --local-dir models--onnx-community--whisper-base
        echo -e "${GREEN}✅ Base model downloaded${NC}"
    else
        echo -e "${YELLOW}⚠️  huggingface-cli not found${NC}"
        echo "   Install with: pip install huggingface-hub"
        echo "   Then run: huggingface-cli download onnx-community/whisper-base --local-dir $MODELS_DIR/models--onnx-community--whisper-base"
    fi

    cd - > /dev/null
fi

echo ""
echo "🎉 Setup Complete!"
echo "==================="
echo ""
echo "Next Steps:"
echo ""
echo "1. ${YELLOW}LOG OUT and LOG BACK IN${NC} for group changes to take effect"
echo ""
echo "2. Build and start the container:"
echo "   cd ~/UC-1"
echo "   docker compose -f docker-compose-uc1-optimized.yml build unicorn-amanuensis"
echo "   docker compose -f docker-compose-uc1-optimized.yml up -d unicorn-amanuensis"
echo ""
echo "3. Test transcription:"
echo "   curl -X POST -F \"file=@test.wav\" http://localhost:9000/transcribe"
echo ""
echo "4. Expected performance:"
echo "   - Base model: 25-51x realtime"
echo "   - Large-v3 model: 8.5x realtime"
echo ""
echo "For detailed documentation, see: NPU-SETUP.md"
echo ""

# Final verification
echo "System Verification:"
echo "--------------------"
echo -n "NPU Device: "
[ -e "/dev/accel/accel0" ] && echo -e "${GREEN}✅ Present${NC}" || echo -e "${RED}❌ Missing${NC}"

echo -n "DRI Devices: "
[ -e "/dev/dri/card0" ] && echo -e "${GREEN}✅ Present${NC}" || echo -e "${RED}❌ Missing${NC}"

echo -n "XRT Installed: "
[ -f "/opt/xilinx/xrt/bin/xrt-smi" ] && echo -e "${GREEN}✅ Yes${NC}" || echo -e "${RED}❌ No${NC}"

echo -n "Docker: "
command_exists docker && echo -e "${GREEN}✅ Installed${NC}" || echo -e "${RED}❌ Not installed${NC}"

echo -n "ONNX Models: "
[ -f "$MODELS_DIR/models--onnx-community--whisper-base/onnx/encoder_model.onnx" ] && echo -e "${GREEN}✅ Downloaded${NC}" || echo -e "${YELLOW}⚠️  Not found${NC}"

echo ""
echo "If NPU Device is missing, you may need to:"
echo "  1. Reboot the system"
echo "  2. Check BIOS settings for NPU enablement"
echo "  3. Verify you have AMD Ryzen 7040/8040 series CPU"
echo ""
