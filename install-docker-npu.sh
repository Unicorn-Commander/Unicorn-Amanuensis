#!/bin/bash
# Unicorn Amanuensis - Docker NPU Installation Script
# Specifically for AMD Phoenix NPU (XDNA1)

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║   🦄 Unicorn Amanuensis Docker NPU Installer 🦄     ║${NC}"
echo -e "${PURPLE}║        AMD Phoenix NPU (XDNA1) Edition              ║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in the right directory
if [ ! -d "whisperx" ]; then
    echo -e "${RED}❌ Error: whisperx directory not found${NC}"
    echo "Please run this script from the Unicorn-Amanuensis root directory"
    exit 1
fi

# Check for NPU device
echo -e "${YELLOW}🔍 Checking for AMD Phoenix NPU...${NC}"
if [ ! -e "/dev/accel/accel0" ]; then
    echo -e "${RED}❌ NPU device /dev/accel/accel0 not found!${NC}"
    echo ""
    echo "Make sure:"
    echo "  1. NPU driver is installed (xrt/xdna)"
    echo "  2. Device permissions are correct"
    echo "  3. User is in 'render' group"
    echo ""
    exit 1
fi

if ! lspci | grep -i "Phoenix" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Warning: Phoenix device not detected in lspci${NC}"
    echo "Continuing anyway since /dev/accel/accel0 exists..."
else
    echo -e "${GREEN}✅ AMD Phoenix NPU detected!${NC}"
fi

# Check device permissions
echo -e "${YELLOW}🔍 Checking device permissions...${NC}"
ls -l /dev/accel/accel0
if [ ! -r "/dev/accel/accel0" ] || [ ! -w "/dev/accel/accel0" ]; then
    echo -e "${RED}❌ Cannot read/write /dev/accel/accel0${NC}"
    echo "Run: sudo chmod 666 /dev/accel/accel0"
    exit 1
fi
echo -e "${GREEN}✅ NPU device accessible${NC}"

# Check Docker
echo -e "${YELLOW}🔍 Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✅ Docker is installed${NC}"

# Check if user is in docker group
if ! groups | grep -q docker; then
    echo -e "${YELLOW}⚠️  You are not in the 'docker' group${NC}"
    echo "Run: sudo usermod -aG docker $USER"
    echo "Then log out and log back in"
fi

# Create necessary directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p whisperx/uploads
mkdir -p whisperx/models
mkdir -p whisperx/static
mkdir -p whisperx/templates
echo -e "${GREEN}✅ Directories created${NC}"

# Verify required files exist
echo -e "${YELLOW}🔍 Checking required files...${NC}"
REQUIRED_FILES=(
    "whisperx/Dockerfile.npu"
    "whisperx/docker-compose-npu.yml"
    "whisperx/server_whisperx_npu.py"
    "whisperx/onnx_whisper_npu.py"
    "whisperx/requirements_npu.txt"
)

MISSING_FILES=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}❌ Missing: $file${NC}"
        MISSING_FILES=$((MISSING_FILES + 1))
    else
        echo -e "${GREEN}✅ Found: $file${NC}"
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo -e "${RED}❌ Missing $MISSING_FILES required files${NC}"
    exit 1
fi

# Check for NPU implementation
if [ ! -d "whisperx/npu" ]; then
    echo -e "${YELLOW}⚠️  NPU implementation directory not found${NC}"
    echo "Creating directory..."
    mkdir -p whisperx/npu
fi

# Check for GUI files
if [ ! -f "whisperx/templates/index.html" ]; then
    echo -e "${YELLOW}⚠️  GUI template not found, but will be created in container${NC}"
fi

# Download Whisper ONNX models for NPU
echo -e "${YELLOW}📥 Downloading Whisper ONNX INT8 models...${NC}"
echo "These are real ONNX models from onnx-community (not magicunicorn placeholders)"
echo "Total download size: ~1.6 GB"
echo ""

# Fix permissions first
sudo chown -R $USER:$USER whisperx/models/ 2>/dev/null || true

# Check if models already exist
if [ -d "whisperx/models/whisper-base-onnx-int8/onnx" ] && [ -d "whisperx/models/whisper-large-v3-onnx-int8/onnx" ]; then
    echo -e "${GREEN}✅ Models already downloaded${NC}"
    echo ""
    read -p "Re-download models? [y/N]: " redownload
    if [[ ! "$redownload" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}ℹ️  Skipping model download${NC}"
    else
        cd whisperx
        ./download-models.sh
        cd ..
    fi
else
    echo -e "${BLUE}📥 Running model download script...${NC}"
    cd whisperx

    # Check if download script exists
    if [ ! -f "download-models.sh" ]; then
        echo -e "${RED}❌ download-models.sh not found!${NC}"
        echo "Creating it now..."
        # The script should already be created, but just in case
        echo -e "${YELLOW}Please run: git pull or re-clone the repository${NC}"
        exit 1
    fi

    # Make executable
    chmod +x download-models.sh

    # Run with option 3 (both models)
    echo "3" | ./download-models.sh
    cd ..
fi

echo -e "${GREEN}✅ Models ready!${NC}"

# Create Docker network
echo -e "${YELLOW}🌐 Creating Docker network...${NC}"
docker network create unicorn-network 2>/dev/null && echo -e "${GREEN}✅ Network created${NC}" || echo -e "${BLUE}ℹ️  Network already exists${NC}"

# Build Docker image
echo -e "${BLUE}🔨 Building NPU Docker image...${NC}"
echo "This may take 5-10 minutes (downloads ONNX models from HuggingFace)..."
cd whisperx
docker build -f Dockerfile.npu -t unicorn-amanuensis-npu:latest .
cd ..
echo -e "${GREEN}✅ Docker image built!${NC}"

# Start container
echo -e "${BLUE}🚀 Starting NPU container...${NC}"
cd whisperx
docker compose -f docker-compose-npu.yml up -d
cd ..

# Wait for container to be healthy
echo -e "${YELLOW}⏳ Waiting for container to be ready...${NC}"
sleep 15

# Check health
echo -e "${YELLOW}🏥 Checking health...${NC}"
HEALTH_RESPONSE=$(curl -s http://localhost:9000/health || echo "failed")
if [[ $HEALTH_RESPONSE == *"status"* ]]; then
    echo -e "${GREEN}✅ Health check passed!${NC}"
else
    echo -e "${YELLOW}⚠️  Health check didn't respond yet, checking logs...${NC}"
    docker logs amanuensis-npu --tail 20
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          🎉 Installation Complete! 🎉               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${PURPLE}Access Points:${NC}"
echo -e "  🌐 Web GUI: ${BLUE}http://localhost:9000/web${NC}"
echo -e "  📡 API: ${BLUE}http://localhost:9000/v1/audio/transcriptions${NC}"
echo -e "  📊 Status: ${BLUE}http://localhost:9000/status${NC}"
echo -e "  🏥 Health: ${BLUE}http://localhost:9000/health${NC}"
echo ""
echo -e "${YELLOW}💡 Useful Commands:${NC}"
echo "  View logs: ${GREEN}docker logs -f amanuensis-npu${NC}"
echo "  Stop: ${GREEN}cd whisperx && docker compose -f docker-compose-npu.yml down${NC}"
echo "  Rebuild: ${GREEN}./rebuild-npu.sh${NC}"
echo "  Test API: ${GREEN}curl -X POST http://localhost:9000/v1/audio/transcriptions -F 'file=@audio.mp3'${NC}"
echo ""
echo -e "${BLUE}📜 The GUI is accessible at http://localhost:9000/web${NC}"
echo -e "${BLUE}🦄 Powered by AMD Phoenix NPU - Efficient AI Transcription!${NC}"
