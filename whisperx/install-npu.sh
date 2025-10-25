#!/bin/bash
#
# Unicorn Amanuensis NPU Installation Script
# Sets up Whisper transcription with AMD Phoenix NPU acceleration
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🦄 Unicorn Amanuensis - NPU Installation"
echo "========================================="
echo ""
echo "This will set up Whisper transcription with AMD NPU acceleration"
echo ""

# Check NPU device
echo "1️⃣  Checking NPU device..."
if [ -e "/dev/accel/accel0" ]; then
    echo "   ✅ NPU device found: /dev/accel/accel0"
else
    echo "   ❌ NPU device not found!"
    echo "   Make sure amdxdna driver is loaded: lsmod | grep amdxdna"
    exit 1
fi

# Check Docker
echo ""
echo "2️⃣  Checking Docker..."
if command -v docker &> /dev/null; then
    echo "   ✅ Docker installed: $(docker --version)"
else
    echo "   ❌ Docker not found!"
    echo "   Install Docker first: https://docs.docker.com/engine/install/"
    exit 1
fi

# Check Docker Compose
if docker compose version &> /dev/null; then
    echo "   ✅ Docker Compose installed: $(docker compose version)"
else
    echo "   ❌ Docker Compose not found!"
    exit 1
fi

# Download models
echo ""
echo "3️⃣  Downloading Whisper models..."
cd "$SCRIPT_DIR"

if [ -d "models/whisper-base-onnx-int8/onnx" ] && [ -d "models/whisper-large-v3-onnx-int8/onnx" ]; then
    echo "   ✅ Models already downloaded"
    echo ""
    read -p "   Re-download models? [y/N]: " redownload
    if [[ "$redownload" =~ ^[Yy]$ ]]; then
        ./download-models.sh
    fi
else
    echo "   Models not found, running download script..."
    ./download-models.sh
fi

# Create network
echo ""
echo "4️⃣  Creating Docker network..."
if docker network inspect unicorn-network &> /dev/null; then
    echo "   ✅ Network 'unicorn-network' already exists"
else
    docker network create unicorn-network
    echo "   ✅ Created network 'unicorn-network'"
fi

# Build image
echo ""
echo "5️⃣  Building Docker image..."
read -p "   Build now? [Y/n]: " build_now
if [[ ! "$build_now" =~ ^[Nn]$ ]]; then
    docker compose -f docker-compose-npu.yml build
    echo "   ✅ Image built successfully"
fi

# Start service
echo ""
echo "6️⃣  Starting service..."
read -p "   Start Amanuensis now? [Y/n]: " start_now
if [[ ! "$start_now" =~ ^[Nn]$ ]]; then
    docker compose -f docker-compose-npu.yml up -d
    echo "   ✅ Service started"

    echo ""
    echo "   Waiting for service to be ready..."
    sleep 5

    if docker ps | grep -q amanuensis-npu; then
        echo "   ✅ Container is running"
        echo ""
        echo "   Checking logs for NPU detection..."
        docker logs amanuensis-npu 2>&1 | grep -E "(NPU|AMD Phoenix|AIE)" | head -5 || true
    else
        echo "   ⚠️  Container not running, check logs:"
        echo "      docker logs amanuensis-npu"
    fi
fi

# Summary
echo ""
echo "✅ Installation Complete!"
echo "========================"
echo ""
echo "Access points:"
echo "  Web GUI:  http://localhost:9000/web"
echo "  API:      http://localhost:9000"
echo "  Health:   http://localhost:9000/health"
echo ""
echo "Commands:"
echo "  View logs:     docker logs -f amanuensis-npu"
echo "  Restart:       docker restart amanuensis-npu"
echo "  Stop:          docker compose -f docker-compose-npu.yml down"
echo "  Rebuild:       docker compose -f docker-compose-npu.yml up -d --build"
echo ""
echo "Model info:"
echo "  Location:      $SCRIPT_DIR/models/"
echo "  Base model:    whisper-base-onnx-int8 (125 MB)"
echo "  Large model:   whisper-large-v3-onnx-int8 (1.5 GB)"
echo ""
echo "Test transcription:"
echo "  curl -X POST -F 'file=@audio.wav' http://localhost:9000/transcribe"
echo ""
