#!/bin/bash
# Unicorn Amanuensis - Bare Metal Installation
# Run WhisperX directly without Docker

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🦄 Unicorn Amanuensis - Bare Metal Installation${NC}"
echo "=================================================="

# Function to detect Python version
detect_python() {
    if command -v python3.10 &> /dev/null; then
        PYTHON="python3.10"
    elif command -v python3.11 &> /dev/null; then
        PYTHON="python3.11"
    elif command -v python3 &> /dev/null; then
        PYTHON="python3"
    else
        echo -e "${RED}❌ Python 3.10+ not found${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Using $PYTHON${NC}"
}

# Function to create virtual environment
create_venv() {
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    $PYTHON -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
}

# Function to detect hardware
detect_hardware() {
    echo -e "${YELLOW}🔍 Detecting hardware...${NC}"
    
    # Check for multiple GPUs
    GPUS_FOUND=()
    
    # Check Intel GPU
    if lspci | grep -E "(Intel.*Graphics|Intel.*Iris|Intel.*Arc)" > /dev/null 2>&1; then
        INTEL_GPU=$(lspci | grep -E "(Intel.*Graphics|Intel.*Iris|Intel.*Arc)" | head -1)
        echo -e "${GREEN}✅ Intel GPU: ${INTEL_GPU}${NC}"
        GPUS_FOUND+=("intel")
    fi
    
    # Check NVIDIA GPU
    if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
        NVIDIA_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo -e "${GREEN}✅ NVIDIA GPU: ${NVIDIA_GPU}${NC}"
        GPUS_FOUND+=("nvidia")
    fi
    
    # Check AMD NPU (Phoenix XDNA1)
    if lspci | grep -i "Phoenix" > /dev/null 2>&1 && [ -e "/dev/accel/accel0" ]; then
        echo -e "${GREEN}✅ AMD NPU (Phoenix XDNA1) detected at /dev/accel/accel0${NC}"
        GPUS_FOUND+=("npu")
    fi
    
    # Check AMD GPU
    if lspci | grep -E "AMD.*Radeon" > /dev/null 2>&1; then
        AMD_GPU=$(lspci | grep -E "AMD.*Radeon" | head -1)
        echo -e "${GREEN}✅ AMD GPU: ${AMD_GPU}${NC}"
        GPUS_FOUND+=("amd")
    fi
    
    # If no GPUs, use CPU
    if [ ${#GPUS_FOUND[@]} -eq 0 ]; then
        echo -e "${YELLOW}⚠️ No GPU detected, will use CPU${NC}"
        GPUS_FOUND+=("cpu")
    fi
}

# Function to select hardware
select_hardware() {
    if [ ${#GPUS_FOUND[@]} -gt 1 ]; then
        echo -e "\n${BLUE}Multiple hardware accelerators detected!${NC}"
        echo "Please select which one to use:"
        
        PS3="Enter your choice (number): "
        select HARDWARE in "${GPUS_FOUND[@]}" "benchmark_all"; do
            case $HARDWARE in
                "benchmark_all")
                    echo -e "${YELLOW}Benchmarking all hardware...${NC}"
                    SELECTED_HARDWARE="benchmark"
                    break
                    ;;
                *)
                    if [ -n "$HARDWARE" ]; then
                        SELECTED_HARDWARE=$HARDWARE
                        echo -e "${GREEN}Selected: $HARDWARE${NC}"
                        break
                    fi
                    ;;
            esac
        done
    else
        SELECTED_HARDWARE=${GPUS_FOUND[0]}
        echo -e "${GREEN}Using: $SELECTED_HARDWARE${NC}"
    fi
}

# Function to install dependencies based on hardware
install_dependencies() {
    echo -e "${YELLOW}Installing dependencies for $SELECTED_HARDWARE...${NC}"
    
    # Common dependencies
    pip install --no-cache-dir \
        fastapi==0.110.0 \
        uvicorn[standard]==0.27.0 \
        python-multipart==0.0.9 \
        aiofiles==23.2.1 \
        librosa==0.10.1 \
        soundfile==0.12.1 \
        pydub==0.25.1 \
        huggingface-hub>=0.20.0 \
        tqdm \
        pandas \
        numpy'<2.0'
    
    case $SELECTED_HARDWARE in
        "nvidia")
            echo -e "${YELLOW}Installing NVIDIA CUDA dependencies...${NC}"
            pip install --no-cache-dir \
                torch==2.1.0 \
                torchaudio==2.1.0 \
                --index-url https://download.pytorch.org/whl/cu121
            pip install --no-cache-dir \
                git+https://github.com/m-bain/whisperx.git \
                pyannote.audio==3.1.1 \
                speechbrain==0.5.16
            ;;
            
        "intel")
            echo -e "${YELLOW}Installing Intel OpenVINO dependencies...${NC}"
            pip install --no-cache-dir \
                openvino==2024.0.0 \
                openvino-dev==2024.0.0 \
                optimum[openvino,nncf]>=1.16.0 \
                optimum-intel
            pip install --no-cache-dir \
                git+https://github.com/m-bain/whisperx.git
            ;;
            
        "npu")
            echo -e "${YELLOW}Installing AMD NPU dependencies...${NC}"
            pip install --no-cache-dir \
                onnx \
                onnxruntime \
                torch==2.1.0+cpu \
                torchaudio==2.1.0+cpu \
                --index-url https://download.pytorch.org/whl/cpu
            pip install --no-cache-dir \
                git+https://github.com/m-bain/whisperx.git
            ;;
            
        "cpu"|*)
            echo -e "${YELLOW}Installing CPU dependencies...${NC}"
            pip install --no-cache-dir \
                torch==2.1.0+cpu \
                torchaudio==2.1.0+cpu \
                --index-url https://download.pytorch.org/whl/cpu
            pip install --no-cache-dir \
                git+https://github.com/m-bain/whisperx.git \
                pyannote.audio==3.1.1
            ;;
    esac
}

# Function to download server files
download_server() {
    echo -e "${YELLOW}Downloading server files...${NC}"
    
    # Create directories
    mkdir -p whisperx/templates
    mkdir -p whisperx/npu
    
    # Download appropriate server file
    BASE_URL="https://raw.githubusercontent.com/Unicorn-Commander/Unicorn-Amanuensis/main/whisperx"
    
    case $SELECTED_HARDWARE in
        "nvidia")
            wget -q "$BASE_URL/server_whisperx_cuda.py" -O whisperx/server.py || \
            wget -q "$BASE_URL/server_whisperx_openvino.py" -O whisperx/server.py
            ;;
        "intel")
            wget -q "$BASE_URL/server_whisperx_igpu.py" -O whisperx/server.py
            ;;
        "npu")
            wget -q "$BASE_URL/server_whisperx_npu.py" -O whisperx/server.py
            wget -q "$BASE_URL/npu/npu_runtime.py" -O whisperx/npu/npu_runtime.py
            ;;
        *)
            wget -q "$BASE_URL/server_whisperx_cpu.py" -O whisperx/server.py || \
            wget -q "$BASE_URL/server_whisperx_openvino.py" -O whisperx/server.py
            ;;
    esac
    
    # Download web interface
    wget -q "$BASE_URL/templates/index.html" -O whisperx/templates/index.html
    
    echo -e "${GREEN}✅ Server files downloaded${NC}"
}

# Function to create start script
create_start_script() {
    cat > start-amanuensis.sh << 'EOF'
#!/bin/bash
# Start Unicorn Amanuensis

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export WHISPER_MODEL=${WHISPER_MODEL:-base}
export ENABLE_DIARIZATION=${ENABLE_DIARIZATION:-true}
export HF_TOKEN=${HF_TOKEN:-}

# Hardware-specific settings
EOF

    case $SELECTED_HARDWARE in
        "nvidia")
            echo 'export WHISPER_DEVICE=cuda' >> start-amanuensis.sh
            echo 'export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}' >> start-amanuensis.sh
            ;;
        "intel")
            echo 'export WHISPER_DEVICE=igpu' >> start-amanuensis.sh
            echo 'export OPENVINO_DEVICE=GPU' >> start-amanuensis.sh
            ;;
        "npu")
            echo 'export WHISPER_DEVICE=npu' >> start-amanuensis.sh
            echo 'export XDNA_VISIBLE_DEVICES=0' >> start-amanuensis.sh
            ;;
        *)
            echo 'export WHISPER_DEVICE=cpu' >> start-amanuensis.sh
            ;;
    esac

    cat >> start-amanuensis.sh << 'EOF'

# Start server
cd whisperx
echo "🦄 Starting Unicorn Amanuensis..."
echo "🌐 Web Interface: http://localhost:9000"
echo "🔌 API Endpoint: http://localhost:9000/v1/audio/transcriptions"
echo ""
python -m uvicorn server:app --host 0.0.0.0 --port 9000
EOF

    chmod +x start-amanuensis.sh
    echo -e "${GREEN}✅ Start script created: ./start-amanuensis.sh${NC}"
}

# Function to create systemd service
create_service() {
    echo -e "${YELLOW}Creating systemd service...${NC}"
    
    SERVICE_FILE="/tmp/unicorn-amanuensis.service"
    cat > $SERVICE_FILE << EOF
[Unit]
Description=Unicorn Amanuensis - AI Transcription Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$(pwd)/venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 9000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    echo -e "${YELLOW}To install as system service, run:${NC}"
    echo "sudo cp $SERVICE_FILE /etc/systemd/system/"
    echo "sudo systemctl daemon-reload"
    echo "sudo systemctl enable unicorn-amanuensis"
    echo "sudo systemctl start unicorn-amanuensis"
}

# Main installation flow
main() {
    echo -e "${BLUE}Starting installation...${NC}"
    
    # Check prerequisites
    detect_python
    
    # Detect hardware
    detect_hardware
    
    # Select hardware if multiple options
    select_hardware
    
    # Create virtual environment
    create_venv
    
    # Install dependencies
    install_dependencies
    
    # Download server files
    download_server
    
    # Create start script
    create_start_script
    
    # Optionally create service
    read -p "Create systemd service? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        create_service
    fi
    
    echo ""
    echo -e "${GREEN}🎉 Installation complete!${NC}"
    echo "=================================================="
    echo -e "To start Unicorn Amanuensis:"
    echo -e "  ${BLUE}./start-amanuensis.sh${NC}"
    echo ""
    echo -e "To use different hardware:"
    echo -e "  ${BLUE}export WHISPER_DEVICE=cuda${NC}  # For NVIDIA GPU"
    echo -e "  ${BLUE}export WHISPER_DEVICE=igpu${NC}  # For Intel iGPU"
    echo -e "  ${BLUE}export WHISPER_DEVICE=cpu${NC}   # For CPU only"
    echo ""
    echo -e "Web Interface: ${BLUE}http://localhost:9000${NC}"
    echo -e "API Endpoint:  ${BLUE}http://localhost:9000/v1/audio/transcriptions${NC}"
}

# Run main installation
main