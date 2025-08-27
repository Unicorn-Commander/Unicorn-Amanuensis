#!/bin/bash
# Unicorn Amanuensis - Hardware-Optimized Container Launcher
# Automatically detects hardware and runs the appropriate container

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🦄 Unicorn Amanuensis - Hardware-Optimized Edition${NC}"
echo "=================================================="

# Function to check Intel GPU
check_intel_gpu() {
    if lspci | grep -E "(Intel.*Graphics|Intel.*Iris|Intel.*Arc)" > /dev/null 2>&1; then
        GPU_NAME=$(lspci | grep -E "(Intel.*Graphics|Intel.*Iris|Intel.*Arc)" | head -1)
        echo -e "${GREEN}✅ Intel GPU detected: ${GPU_NAME}${NC}"
        return 0
    fi
    return 1
}

# Function to check AMD NPU
check_amd_npu() {
    if lspci | grep -i "AMD.*1502" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ AMD NPU (XDNA) detected - Phoenix/Hawk Point${NC}"
        return 0
    fi
    return 1
}

# Function to check NVIDIA GPU
check_nvidia_gpu() {
    if nvidia-smi > /dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo -e "${GREEN}✅ NVIDIA GPU detected: ${GPU_NAME}${NC}"
        return 0
    fi
    return 1
}

# Detect hardware and select profile
echo -e "${YELLOW}🔍 Detecting hardware...${NC}"

PROFILE=""
HARDWARE=""
MODEL="base"

if check_intel_gpu; then
    PROFILE="igpu"
    HARDWARE="Intel iGPU with OpenVINO"
    MODEL=${WHISPER_MODEL:-base}
    echo -e "${BLUE}📦 Using Intel iGPU optimized container${NC}"
elif check_amd_npu; then
    PROFILE="npu"
    HARDWARE="AMD NPU (XDNA)"
    MODEL=${WHISPER_MODEL:-base}
    echo -e "${BLUE}📦 Using AMD NPU optimized container${NC}"
elif check_nvidia_gpu; then
    PROFILE="cuda"
    HARDWARE="NVIDIA GPU with CUDA"
    MODEL=${WHISPER_MODEL:-large-v3}
    echo -e "${BLUE}📦 Using NVIDIA CUDA optimized container${NC}"
else
    PROFILE="cpu"
    HARDWARE="CPU (no GPU detected)"
    MODEL=${WHISPER_MODEL:-base}
    echo -e "${YELLOW}⚠️ No GPU detected, using CPU-optimized container${NC}"
fi

# Export model for docker-compose
export WHISPER_MODEL=$MODEL

# Stop any existing containers
echo -e "${YELLOW}Stopping existing containers...${NC}"
docker-compose -f docker-compose.hardware.yml --profile igpu down 2>/dev/null || true
docker-compose -f docker-compose.hardware.yml --profile cuda down 2>/dev/null || true
docker-compose -f docker-compose.hardware.yml --profile npu down 2>/dev/null || true
docker-compose -f docker-compose.hardware.yml --profile cpu down 2>/dev/null || true

# Build if needed (with cache)
echo -e "${YELLOW}Building ${PROFILE} container (using cache if available)...${NC}"
docker-compose -f docker-compose.hardware.yml --profile $PROFILE build

# Start the appropriate container
echo -e "${GREEN}🚀 Starting Unicorn Amanuensis with ${HARDWARE}...${NC}"
docker-compose -f docker-compose.hardware.yml --profile $PROFILE up -d

# Wait for service
echo -e "${YELLOW}⏳ Waiting for service to be ready...${NC}"
for i in {1..60}; do
    if curl -s http://localhost:9000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Service is ready!${NC}"
        
        # Get actual hardware info from service
        HARDWARE_INFO=$(curl -s http://localhost:9000/api/hardware | python3 -m json.tool 2>/dev/null || echo "{}")
        
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# Show final status
echo ""
echo -e "${GREEN}🎉 Unicorn Amanuensis is running!${NC}"
echo "=================================================="
echo -e "🌐 Web Interface:    ${BLUE}http://localhost:9000${NC}"
echo -e "🔌 API Endpoint:     ${BLUE}http://localhost:9000/v1/audio/transcriptions${NC}"
echo -e "💻 Hardware:         ${GREEN}${HARDWARE}${NC}"
echo -e "🧠 Model:           ${GREEN}${MODEL}${NC}"
echo -e "🐳 Container:       ${GREEN}unicorn-amanuensis:${PROFILE}${NC}"
echo -e "📊 Health:          ${BLUE}http://localhost:9000/health${NC}"
echo ""
echo -e "${BLUE}Features:${NC}"
echo "  • Hardware-optimized inference"
echo "  • Word-level timestamps"
echo "  • Speaker diarization"
echo "  • 100+ language support"
echo "  • OpenAI-compatible API"
echo ""
echo -e "${YELLOW}View logs:${NC} docker logs -f unicorn-amanuensis"
echo -e "${YELLOW}Stop service:${NC} docker-compose -f docker-compose.hardware.yml --profile ${PROFILE} down"