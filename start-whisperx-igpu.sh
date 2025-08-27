#!/bin/bash

echo "========================================="
echo "Unicorn Amanuensis - WhisperX with Intel iGPU"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for Intel GPU
echo -e "${YELLOW}Checking for Intel GPU...${NC}"
if lspci | grep -i "VGA.*Intel" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Intel GPU detected${NC}"
    lspci | grep -i "VGA.*Intel"
else
    echo -e "${YELLOW}⚠️ No Intel GPU detected via lspci${NC}"
fi

# Check DRI devices
echo -e "\n${YELLOW}Checking DRI devices...${NC}"
if ls /dev/dri/card* > /dev/null 2>&1; then
    echo -e "${GREEN}✅ DRI card devices found:${NC}"
    ls -la /dev/dri/card*
else
    echo -e "${RED}❌ No DRI card devices found${NC}"
fi

if ls /dev/dri/renderD* > /dev/null 2>&1; then
    echo -e "${GREEN}✅ DRI render devices found:${NC}"
    ls -la /dev/dri/renderD*
else
    echo -e "${RED}❌ No DRI render devices found${NC}"
fi

# Check VA-API support
echo -e "\n${YELLOW}Checking VA-API support...${NC}"
if command -v vainfo &> /dev/null; then
    if vainfo 2>&1 | grep -q "Driver version"; then
        echo -e "${GREEN}✅ VA-API is working${NC}"
        vainfo 2>&1 | grep "Driver version"
    else
        echo -e "${RED}❌ VA-API not working properly${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ vainfo not installed${NC}"
fi

# Check OpenCL support
echo -e "\n${YELLOW}Checking OpenCL support...${NC}"
if command -v clinfo &> /dev/null; then
    if clinfo 2>&1 | grep -q "Intel"; then
        echo -e "${GREEN}✅ Intel OpenCL device found${NC}"
        clinfo 2>&1 | grep -A 2 "Device Name"
    else
        echo -e "${YELLOW}⚠️ No Intel OpenCL devices found${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ clinfo not installed${NC}"
fi

# Check group memberships
echo -e "\n${YELLOW}Checking group memberships...${NC}"
VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3 2>/dev/null || echo "")

echo "Current user: $(whoami)"
echo "Video group GID: $VIDEO_GID"
if [ -n "$RENDER_GID" ]; then
    echo "Render group GID: $RENDER_GID"
fi

# Check if current user is in video group
if groups | grep -q video; then
    echo -e "${GREEN}✅ User is in video group${NC}"
else
    echo -e "${YELLOW}⚠️ User is not in video group. You may need to run:${NC}"
    echo "   sudo usermod -aG video $USER"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "\n${YELLOW}Creating default .env file...${NC}"
    cat > .env << EOF
# Unicorn Amanuensis Configuration
WHISPER_MODEL=large-v3
WHISPER_DEVICE=igpu
COMPUTE_TYPE=int8
BATCH_SIZE=16
ENABLE_DIARIZATION=true
MAX_SPEAKERS=10
HF_TOKEN=

# Add your Hugging Face token above for speaker diarization
EOF
    echo -e "${GREEN}✅ Created .env file${NC}"
else
    echo -e "\n${GREEN}✅ Using existing .env file${NC}"
fi

# Pull or build the image
echo -e "\n${YELLOW}Building WhisperX + OpenVINO Docker image...${NC}"
docker-compose -f docker-compose.whisperx.yml build

# Start the service
echo -e "\n${YELLOW}Starting Unicorn Amanuensis with WhisperX + OpenVINO...${NC}"
docker-compose -f docker-compose.whisperx.yml up -d

# Wait for service to be ready
echo -e "\n${YELLOW}Waiting for service to be ready...${NC}"
MAX_ATTEMPTS=60
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:9000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Service is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 5
    ATTEMPT=$((ATTEMPT + 1))
done

if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
    echo -e "\n${RED}❌ Service failed to start in time${NC}"
    echo "Checking logs..."
    docker-compose -f docker-compose.whisperx.yml logs --tail=50
    exit 1
fi

# Show service status
echo -e "\n${YELLOW}Service Status:${NC}"
curl -s http://localhost:9000/health | python3 -m json.tool

echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}Unicorn Amanuensis is running!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "API Endpoint: http://localhost:9000/v1/audio/transcriptions"
echo "Web UI: http://localhost:9001/"
echo "Health Check: http://localhost:9000/health"
echo "GPU Status: http://localhost:9000/gpu-status"
echo ""
echo "To view logs: docker-compose -f docker-compose.whisperx.yml logs -f"
echo "To stop: docker-compose -f docker-compose.whisperx.yml down"
echo ""