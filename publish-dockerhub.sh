#!/bin/bash
# Publish Unicorn Amanuensis NPU to Docker Hub
# Organization: magicunicorn

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║   🦄 Publish to Docker Hub - Magic Unicorn 🦄       ║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if logged in to Docker Hub
echo -e "${YELLOW}🔍 Checking Docker Hub login...${NC}"
if ! docker info | grep -q "Username"; then
    echo -e "${YELLOW}⚠️  Not logged in to Docker Hub${NC}"
    echo "Please log in:"
    docker login
fi

# Get version
echo ""
read -p "Enter version tag (e.g., 1.0.0, latest): " VERSION
if [ -z "$VERSION" ]; then
    VERSION="latest"
fi

# Docker Hub organization and repository
ORG="magicunicorn"
REPO="unicorn-amanuensis-npu"
LOCAL_IMAGE="unicorn-amanuensis-npu:latest"

echo ""
echo -e "${BLUE}📋 Publishing Configuration:${NC}"
echo "  Local Image: $LOCAL_IMAGE"
echo "  Docker Hub: $ORG/$REPO:$VERSION"
echo ""

# Confirm
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Check if local image exists
echo -e "${YELLOW}🔍 Checking local image...${NC}"
if ! docker image inspect $LOCAL_IMAGE > /dev/null 2>&1; then
    echo -e "${RED}❌ Local image not found: $LOCAL_IMAGE${NC}"
    echo "Build it first with: ./rebuild-npu.sh"
    exit 1
fi
echo -e "${GREEN}✅ Local image found${NC}"

# Tag for Docker Hub
echo -e "${BLUE}🏷️  Tagging image...${NC}"
docker tag $LOCAL_IMAGE $ORG/$REPO:$VERSION
if [ "$VERSION" != "latest" ]; then
    docker tag $LOCAL_IMAGE $ORG/$REPO:latest
fi
echo -e "${GREEN}✅ Image tagged${NC}"

# Push to Docker Hub
echo -e "${BLUE}📤 Pushing to Docker Hub...${NC}"
echo "This may take several minutes (image is ~9GB)..."
docker push $ORG/$REPO:$VERSION

if [ "$VERSION" != "latest" ]; then
    echo -e "${BLUE}📤 Pushing 'latest' tag...${NC}"
    docker push $ORG/$REPO:latest
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          🎉 Published Successfully! 🎉              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${PURPLE}Docker Hub Images:${NC}"
echo -e "  ${BLUE}https://hub.docker.com/r/$ORG/$REPO${NC}"
echo ""
echo -e "${YELLOW}Users can now install with:${NC}"
echo -e "  ${GREEN}docker pull $ORG/$REPO:$VERSION${NC}"
echo -e "  ${GREEN}docker pull $ORG/$REPO:latest${NC}"
echo ""
echo -e "${BLUE}Quick Start Command:${NC}"
cat << 'EOF'
docker run -d \
  --name amanuensis-npu \
  --device /dev/accel/accel0 \
  --device /dev/dri \
  -p 9000:9000 \
  magicunicorn/unicorn-amanuensis-npu:latest
EOF
echo ""
echo -e "${GREEN}✅ Publication complete!${NC}"
