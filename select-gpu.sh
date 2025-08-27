#!/bin/bash
# Unicorn Amanuensis - GPU Selector for Dual-GPU Systems
# Choose between Intel iGPU and NVIDIA RTX GPU

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}🦄 Unicorn Amanuensis - GPU Selection${NC}"
echo "=================================================="

# Detect available GPUs
echo -e "${YELLOW}🔍 Detecting GPUs...${NC}"
echo ""

# Check Intel iGPU
INTEL_GPU=""
if lspci | grep -E "(Intel.*Graphics|Intel.*Iris|Intel.*Arc)" > /dev/null 2>&1; then
    INTEL_GPU=$(lspci | grep -E "VGA.*Intel" | sed 's/.*: //')
    echo -e "${GREEN}[1] Intel iGPU detected:${NC}"
    echo -e "    $INTEL_GPU"
    echo -e "    ${PURPLE}• Power efficient (25-35W)${NC}"
    echo -e "    ${PURPLE}• OpenVINO optimized${NC}"
    echo -e "    ${PURPLE}• Best for: whisper-base model${NC}"
    echo ""
fi

# Check NVIDIA GPU
NVIDIA_GPU=""
if nvidia-smi > /dev/null 2>&1; then
    NVIDIA_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo -e "${GREEN}[2] NVIDIA GPU detected:${NC}"
    echo -e "    $NVIDIA_GPU ($VRAM)"
    echo -e "    ${PURPLE}• Maximum performance${NC}"
    echo -e "    ${PURPLE}• CUDA acceleration${NC}"
    echo -e "    ${PURPLE}• Best for: whisper-large-v3 model${NC}"
    echo ""
fi

# Selection menu
echo -e "${BLUE}Select which GPU to use for Unicorn Amanuensis:${NC}"
echo ""
echo "  [1] Intel iGPU    - Power efficient, OpenVINO optimized"
echo "  [2] NVIDIA GPU    - Maximum performance, CUDA acceleration"
echo "  [3] Auto-select   - Let the system choose (prefers iGPU for efficiency)"
echo "  [4] Benchmark     - Test both and show performance comparison"
echo ""

read -p "Enter your choice [1-4]: " choice

case $choice in
    1)
        echo -e "\n${GREEN}✅ Selected: Intel iGPU${NC}"
        export SELECTED_GPU="intel"
        export WHISPER_DEVICE="igpu"
        export WHISPER_MODEL="${WHISPER_MODEL:-base}"
        PROFILE="igpu"
        ;;
    2)
        echo -e "\n${GREEN}✅ Selected: NVIDIA GPU${NC}"
        export SELECTED_GPU="nvidia"
        export WHISPER_DEVICE="cuda"
        export WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
        PROFILE="cuda"
        ;;
    3)
        echo -e "\n${GREEN}✅ Auto-selecting: Intel iGPU (power efficient)${NC}"
        export SELECTED_GPU="intel"
        export WHISPER_DEVICE="igpu"
        export WHISPER_MODEL="${WHISPER_MODEL:-base}"
        PROFILE="igpu"
        ;;
    4)
        echo -e "\n${YELLOW}🏁 Starting benchmark...${NC}"
        ./benchmark-gpus.sh
        exit 0
        ;;
    *)
        echo -e "\n${YELLOW}Invalid choice. Using Intel iGPU by default.${NC}"
        export SELECTED_GPU="intel"
        export WHISPER_DEVICE="igpu"
        export WHISPER_MODEL="${WHISPER_MODEL:-base}"
        PROFILE="igpu"
        ;;
esac

# Save selection for future use
echo "SELECTED_GPU=$SELECTED_GPU" > .gpu_selection
echo "WHISPER_DEVICE=$WHISPER_DEVICE" >> .gpu_selection
echo "WHISPER_MODEL=$WHISPER_MODEL" >> .gpu_selection
echo "PROFILE=$PROFILE" >> .gpu_selection

echo ""
echo -e "${BLUE}Starting Unicorn Amanuensis with $SELECTED_GPU...${NC}"
echo "=================================================="

# Stop any existing containers
docker-compose -f docker-compose.hardware.yml down > /dev/null 2>&1 || true

# Start with selected profile
docker-compose -f docker-compose.hardware.yml --profile $PROFILE up -d

# Wait for service
echo -e "${YELLOW}⏳ Waiting for service to be ready...${NC}"
for i in {1..60}; do
    if curl -s http://localhost:9000/health > /dev/null 2>&1; then
        echo -e "\n${GREEN}✅ Service is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# Display final info
HARDWARE_INFO=$(curl -s http://localhost:9000/api/hardware 2>/dev/null || echo "{}")

echo ""
echo -e "${GREEN}🎉 Unicorn Amanuensis is running!${NC}"
echo "=================================================="
echo -e "🌐 Web Interface:    ${BLUE}http://localhost:9000${NC}"
echo -e "🔌 API Endpoint:     ${BLUE}http://localhost:9000/v1/audio/transcriptions${NC}"
echo -e "💻 Hardware:         ${GREEN}$SELECTED_GPU${NC}"
echo -e "🧠 Model:           ${GREEN}$WHISPER_MODEL${NC}"
echo ""
echo -e "${YELLOW}To switch GPUs, run this script again${NC}"
echo -e "${YELLOW}To always use the same GPU:${NC}"
echo -e "  Intel iGPU:  ${BLUE}export WHISPER_DEVICE=igpu${NC}"
echo -e "  NVIDIA GPU:  ${BLUE}export WHISPER_DEVICE=cuda${NC}"