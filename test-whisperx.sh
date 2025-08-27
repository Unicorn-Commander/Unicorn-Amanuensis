#!/bin/bash

echo "========================================="
echo "Testing Unicorn Amanuensis - WhisperX"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if service is running
echo -e "\n${YELLOW}Checking service health...${NC}"
HEALTH_RESPONSE=$(curl -s http://localhost:9000/health)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Service is healthy${NC}"
    echo "$HEALTH_RESPONSE" | python3 -m json.tool
else
    echo -e "${RED}❌ Service is not responding${NC}"
    exit 1
fi

# Check GPU status
echo -e "\n${YELLOW}Checking GPU status...${NC}"
GPU_STATUS=$(curl -s http://localhost:9000/gpu-status)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ GPU status retrieved${NC}"
    echo "$GPU_STATUS" | python3 -m json.tool
else
    echo -e "${YELLOW}⚠️ Could not retrieve GPU status${NC}"
fi

# Create a test audio file if it doesn't exist
TEST_AUDIO="/tmp/test_audio.wav"
if [ ! -f "$TEST_AUDIO" ]; then
    echo -e "\n${YELLOW}Creating test audio file...${NC}"
    # Generate a simple test audio with speech synthesis or use ffmpeg to create a tone
    ffmpeg -f lavfi -i "sine=frequency=440:duration=3" -ac 1 -ar 16000 "$TEST_AUDIO" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Test audio created${NC}"
    else
        echo -e "${RED}❌ Failed to create test audio${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Using existing test audio${NC}"
fi

# Test basic transcription
echo -e "\n${YELLOW}Testing basic transcription...${NC}"
RESPONSE=$(curl -s -X POST http://localhost:9000/v1/audio/transcriptions \
    -F "file=@$TEST_AUDIO" \
    -F "language=en" \
    -F "response_format=json")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Basic transcription successful${NC}"
    echo "$RESPONSE" | python3 -m json.tool | head -20
else
    echo -e "${RED}❌ Basic transcription failed${NC}"
fi

# Test with word timestamps
echo -e "\n${YELLOW}Testing transcription with word timestamps...${NC}"
RESPONSE=$(curl -s -X POST http://localhost:9000/v1/audio/transcriptions \
    -F "file=@$TEST_AUDIO" \
    -F "language=en" \
    -F "word_timestamps=true" \
    -F "timestamps=true" \
    -F "response_format=json")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Word timestamp transcription successful${NC}"
    # Check if words are present in response
    if echo "$RESPONSE" | grep -q "words"; then
        echo -e "${GREEN}✅ Word-level timestamps present${NC}"
    else
        echo -e "${YELLOW}⚠️ No word-level timestamps in response${NC}"
    fi
else
    echo -e "${RED}❌ Word timestamp transcription failed${NC}"
fi

# Test with speaker diarization (if HF_TOKEN is set)
if [ -n "$HF_TOKEN" ]; then
    echo -e "\n${YELLOW}Testing transcription with speaker diarization...${NC}"
    RESPONSE=$(curl -s -X POST http://localhost:9000/v1/audio/transcriptions \
        -F "file=@$TEST_AUDIO" \
        -F "language=en" \
        -F "diarize=true" \
        -F "min_speakers=1" \
        -F "max_speakers=2" \
        -F "response_format=json")
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Diarization transcription successful${NC}"
        # Check if speaker info is present
        if echo "$RESPONSE" | grep -q "speaker"; then
            echo -e "${GREEN}✅ Speaker diarization present${NC}"
        else
            echo -e "${YELLOW}⚠️ No speaker information in response${NC}"
        fi
    else
        echo -e "${RED}❌ Diarization transcription failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ Skipping diarization test (HF_TOKEN not set)${NC}"
fi

# Performance test
echo -e "\n${YELLOW}Checking performance metrics...${NC}"
RESPONSE=$(curl -s -X POST http://localhost:9000/v1/audio/transcriptions \
    -F "file=@$TEST_AUDIO" \
    -F "language=auto" \
    -F "response_format=json")

if [ $? -eq 0 ]; then
    # Extract performance metrics
    RTF=$(echo "$RESPONSE" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('performance', {}).get('rtf', 'N/A'))" 2>/dev/null)
    TOTAL_TIME=$(echo "$RESPONSE" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('performance', {}).get('total_time', 'N/A'))" 2>/dev/null)
    
    echo -e "${GREEN}Performance Metrics:${NC}"
    echo "  Real-time factor (RTF): $RTF"
    echo "  Total processing time: $TOTAL_TIME"
    
    # Check device being used
    DEVICE=$(echo "$RESPONSE" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('config', {}).get('device', 'N/A'))" 2>/dev/null)
    echo "  Device: $DEVICE"
fi

echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}Testing complete!${NC}"
echo -e "${GREEN}=========================================${NC}"