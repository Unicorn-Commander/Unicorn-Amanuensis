#!/bin/bash
# Rebuild Unicorn Amanuensis NPU Container
# This script stops, removes, and rebuilds the NPU container

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║   🦄 Unicorn Amanuensis NPU Rebuild Script 🦄       ║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Navigate to whisperx directory
cd "$(dirname "$0")/whisperx"

# Stop and remove existing container
echo -e "${YELLOW}🛑 Stopping existing NPU container...${NC}"
docker compose -f docker-compose-npu.yml down 2>/dev/null || echo "No container to stop"

# Remove old image
echo -e "${YELLOW}🗑️  Removing old NPU image...${NC}"
docker rmi unicorn-amanuensis-npu:latest 2>/dev/null || echo "No image to remove"

# Clean up dangling images
echo -e "${YELLOW}🧹 Cleaning up Docker cache...${NC}"
docker image prune -f

# Build new image
echo -e "${BLUE}🔨 Building NPU container...${NC}"
docker build -f Dockerfile.npu -t unicorn-amanuensis-npu:latest .

# Create network if it doesn't exist
echo -e "${BLUE}🌐 Ensuring unicorn-network exists...${NC}"
docker network create unicorn-network 2>/dev/null || echo "Network already exists"

# Start container
echo -e "${GREEN}🚀 Starting NPU container...${NC}"
docker compose -f docker-compose-npu.yml up -d

# Wait for health check
echo -e "${YELLOW}⏳ Waiting for health check...${NC}"
sleep 10

# Show status
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║            🎉 Rebuild Complete! 🎉                   ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Check container status
docker ps | grep amanuensis-npu

echo ""
echo -e "${PURPLE}Access Points:${NC}"
echo -e "  🌐 GUI: ${BLUE}http://localhost:9000/web${NC}"
echo -e "  📡 API: ${BLUE}http://localhost:9000/v1/audio/transcriptions${NC}"
echo -e "  📊 Health: ${BLUE}http://localhost:9000/health${NC}"
echo ""

# Show logs
echo -e "${YELLOW}📜 Recent logs:${NC}"
docker logs amanuensis-npu --tail 20

echo ""
echo -e "${GREEN}✅ Container rebuilt successfully!${NC}"
echo ""
echo "Commands:"
echo "  View logs: ${GREEN}docker logs -f amanuensis-npu${NC}"
echo "  Stop: ${GREEN}cd whisperx && docker compose -f docker-compose-npu.yml down${NC}"
echo "  Restart: ${GREEN}./rebuild-npu.sh${NC}"
