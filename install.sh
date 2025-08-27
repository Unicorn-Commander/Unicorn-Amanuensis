#!/bin/bash

# Unicorn Amanuensis Installation Script
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║        📜 Unicorn Amanuensis Installer 📜           ║${NC}"
echo -e "${PURPLE}║       Professional AI Transcription Service          ║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

echo -e "${GREEN}✓ Docker is installed${NC}"

# Hardware detection
echo ""
echo "Detecting hardware..."

# Run Python hardware detection if available
if command -v python3 &> /dev/null && [ -f "hardware-detect/detect.py" ]; then
    python3 hardware-detect/detect.py
    BACKEND="auto"
else
    # Simple detection
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null 2>&1; then
        echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
        BACKEND="cuda"
    elif [ -d "/dev/dri" ] && [ -e "/dev/dri/renderD128" ]; then
        echo -e "${BLUE}✓ Intel GPU detected${NC}"
        BACKEND="igpu"
    else
        echo -e "${YELLOW}⚠ No GPU detected - using CPU${NC}"
        BACKEND="cpu"
    fi
fi

# Create .env file
if [ ! -f .env ]; then
    echo ""
    echo "Creating configuration..."
    cp .env.template .env
    
    # Update backend
    sed -i "s/WHISPER_BACKEND=auto/WHISPER_BACKEND=$BACKEND/" .env
    
    # Configure based on backend
    case $BACKEND in
        cuda)
            sed -i "s/COMPUTE_TYPE=int8/COMPUTE_TYPE=float16/" .env
            echo -e "${GREEN}✓ Configured for NVIDIA GPU${NC}"
            ;;
        igpu)
            echo -e "${BLUE}✓ Configured for Intel iGPU${NC}"
            ;;
        npu)
            echo -e "${PURPLE}✓ Configured for AMD NPU${NC}"
            ;;
        *)
            echo -e "${GREEN}✓ Configured for CPU${NC}"
            ;;
    esac
    
    # Ask for HuggingFace token
    echo ""
    echo -e "${BLUE}Optional: HuggingFace Token for Speaker Diarization${NC}"
    echo "Get a token at: https://huggingface.co/settings/tokens"
    read -p "Enter token (or press Enter to skip): " HF_TOKEN
    
    if [ -n "$HF_TOKEN" ]; then
        sed -i "s/HF_TOKEN=/HF_TOKEN=$HF_TOKEN/" .env
        echo -e "${GREEN}✓ Diarization enabled${NC}"
    else
        sed -i "s/ENABLE_DIARIZATION=true/ENABLE_DIARIZATION=false/" .env
        echo -e "${YELLOW}⚠ Diarization disabled (no token)${NC}"
    fi
    
    # Ask for domain
    echo ""
    read -p "Domain/IP for remote access (Enter for localhost): " EXTERNAL_HOST
    if [ -n "$EXTERNAL_HOST" ] && [ "$EXTERNAL_HOST" != "localhost" ]; then
        sed -i "s/EXTERNAL_HOST=localhost/EXTERNAL_HOST=$EXTERNAL_HOST/" .env
        echo -e "${GREEN}✓ Remote access configured${NC}"
    fi
else
    echo -e "${GREEN}✓ Configuration exists${NC}"
fi

# Build and start
echo ""
echo -e "${BLUE}Building service...${NC}"
docker-compose build

echo ""
echo -e "${BLUE}Starting Unicorn Amanuensis...${NC}"
docker-compose up -d

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          🎉 Installation Complete! 🎉               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Show URLs
source .env
echo -e "${PURPLE}Access Points:${NC}"
if [ "$EXTERNAL_HOST" != "localhost" ] && [ -n "$EXTERNAL_HOST" ]; then
    echo "  📡 API: ${BLUE}${EXTERNAL_PROTOCOL:-http}://${EXTERNAL_HOST}:9000${NC}"
    echo "  🌐 Web UI: ${BLUE}${EXTERNAL_PROTOCOL:-http}://${EXTERNAL_HOST}:9001${NC}"
    echo "  📚 API Docs: ${BLUE}${EXTERNAL_PROTOCOL:-http}://${EXTERNAL_HOST}:9000/docs${NC}"
else
    echo "  📡 API: ${BLUE}http://localhost:9000${NC}"
    echo "  🌐 Web UI: ${BLUE}http://localhost:9001${NC}"
    echo "  📚 API Docs: ${BLUE}http://localhost:9000/docs${NC}"
fi

echo ""
echo -e "${YELLOW}Note: First startup may take a few minutes while models download${NC}"
echo ""
echo "Commands:"
echo "  View logs: ${GREEN}docker-compose logs -f${NC}"
echo "  Stop: ${GREEN}docker-compose down${NC}"
echo "  Test: ${GREEN}curl -X POST http://localhost:9000/v1/audio/transcriptions -F 'file=@audio.mp3'${NC}"
echo ""
echo -e "${PURPLE}📜 Unicorn Amanuensis - Transcribe with Intelligence 📜${NC}"