#!/bin/bash
# Unicorn Amanuensis Pro - Start with hardware acceleration
# Automatically detects and uses: AMD NPU, Intel iGPU, NVIDIA GPU, or CPU

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🦄 Unicorn Amanuensis Pro - Enterprise Transcription Suite${NC}"
echo "=================================================="

# Detect hardware
echo -e "${YELLOW}🔍 Detecting hardware acceleration...${NC}"

# Check for AMD NPU
if lspci | grep -i "AMD.*1502" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ AMD NPU (XDNA) detected - Phoenix/Hawk Point${NC}"
    export WHISPER_DEVICE=npu
    HARDWARE="AMD NPU"
# Check for Intel GPU
elif lspci | grep -E "(Intel.*Graphics|Intel.*Iris|Intel.*Arc)" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Intel iGPU detected - Using OpenVINO acceleration${NC}"
    export WHISPER_DEVICE=igpu
    HARDWARE="Intel iGPU"
# Check for NVIDIA GPU
elif nvidia-smi > /dev/null 2>&1; then
    echo -e "${GREEN}✅ NVIDIA GPU detected - Using CUDA acceleration${NC}"
    export WHISPER_DEVICE=cuda
    HARDWARE="NVIDIA GPU"
else
    echo -e "${YELLOW}⚠️ No GPU detected - Using optimized CPU mode${NC}"
    export WHISPER_DEVICE=cpu
    HARDWARE="CPU"
fi

# Set model based on hardware
if [ "$WHISPER_DEVICE" = "npu" ] || [ "$WHISPER_DEVICE" = "igpu" ]; then
    # Use smaller model for NPU/iGPU
    export WHISPER_MODEL=${WHISPER_MODEL:-base}
    echo -e "${BLUE}📦 Using model: $WHISPER_MODEL (optimized for $HARDWARE)${NC}"
else
    # Use larger model for NVIDIA/CPU
    export WHISPER_MODEL=${WHISPER_MODEL:-large-v3}
    echo -e "${BLUE}📦 Using model: $WHISPER_MODEL${NC}"
fi

# Features
echo -e "\n${BLUE}✨ Features:${NC}"
echo "  • Word-level timestamps with phoneme alignment"
echo "  • Speaker diarization (multi-speaker detection)"
echo "  • 100+ language support with auto-detection"
echo "  • OpenAI-compatible API (drop-in replacement)"
echo "  • Professional web interface"
echo ""

# Check if container is already running
if docker ps | grep unicorn-amanuensis > /dev/null; then
    echo -e "${YELLOW}⚠️ Stopping existing container...${NC}"
    docker-compose -f docker-compose.whisperx.yml down
fi

# Start the service
echo -e "${GREEN}🚀 Starting Unicorn Amanuensis on $HARDWARE...${NC}"

# Add device flags based on hardware
DOCKER_FLAGS=""
if [ "$WHISPER_DEVICE" = "igpu" ]; then
    DOCKER_FLAGS="--device /dev/dri:/dev/dri"
elif [ "$WHISPER_DEVICE" = "cuda" ]; then
    DOCKER_FLAGS="--gpus all"
fi

# Export for docker-compose
export DOCKER_FLAGS

# Start with docker-compose
docker-compose -f docker-compose.whisperx.yml up -d

# Wait for service to be ready
echo -e "${YELLOW}⏳ Waiting for service to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:9000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Service is ready!${NC}"
        break
    fi
    sleep 1
done

# Show status
echo ""
echo -e "${GREEN}🎉 Unicorn Amanuensis Pro is running!${NC}"
echo "=================================================="
echo -e "🌐 Web Interface:    ${BLUE}http://localhost:9000${NC}"
echo -e "🔌 API Endpoint:     ${BLUE}http://localhost:9000/v1/audio/transcriptions${NC}"
echo -e "💻 Hardware:         ${GREEN}$HARDWARE${NC}"
echo -e "🧠 Model:           ${GREEN}$WHISPER_MODEL${NC}"
echo -e "📊 Health Check:     ${BLUE}http://localhost:9000/health${NC}"
echo ""
echo -e "${YELLOW}Tips:${NC}"
echo "  • Upload audio/video files via the web interface"
echo "  • Record directly from your microphone"
echo "  • Export transcripts as TXT, SRT, VTT, or JSON"
echo "  • View speaker timeline and word-level timestamps"
echo ""
echo -e "${BLUE}View logs: docker logs -f unicorn-amanuensis${NC}"