#!/bin/bash
# Benchmark Intel iGPU vs NVIDIA GPU for WhisperX

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🏁 Unicorn Amanuensis - GPU Benchmark${NC}"
echo "=================================================="

# Create test audio file if it doesn't exist
TEST_FILE="test_audio.wav"
if [ ! -f "$TEST_FILE" ]; then
    echo -e "${YELLOW}Creating 30-second test audio file...${NC}"
    # Use ffmpeg to generate a test tone
    ffmpeg -f lavfi -i "sine=frequency=440:duration=30" -ac 2 -ar 16000 "$TEST_FILE" 2>/dev/null
fi

# Function to benchmark a GPU
benchmark_gpu() {
    local gpu_type=$1
    local profile=$2
    local model=$3
    
    echo -e "\n${YELLOW}Testing $gpu_type...${NC}"
    
    # Stop any existing containers
    docker-compose -f docker-compose.hardware.yml down > /dev/null 2>&1 || true
    
    # Start container with selected profile
    export WHISPER_MODEL=$model
    docker-compose -f docker-compose.hardware.yml --profile $profile up -d
    
    # Wait for service
    echo -n "Waiting for service..."
    for i in {1..60}; do
        if curl -s http://localhost:9000/health > /dev/null 2>&1; then
            echo " Ready!"
            break
        fi
        sleep 1
    done
    
    # Warm up run
    echo "Warming up..."
    curl -s -X POST \
        -F "file=@$TEST_FILE" \
        -F "model=$model" \
        http://localhost:9000/v1/audio/transcriptions > /dev/null
    
    # Benchmark runs
    echo "Running benchmark (3 runs)..."
    
    local total_time=0
    local power_samples=()
    
    for run in 1 2 3; do
        echo -n "  Run $run: "
        
        # Measure power (if available)
        if [ "$gpu_type" == "Intel iGPU" ]; then
            # Try to get Intel GPU power
            power=$(cat /sys/class/drm/card0/power/energy_uj 2>/dev/null || echo "0")
        elif [ "$gpu_type" == "NVIDIA GPU" ]; then
            # Get NVIDIA GPU power
            power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null || echo "0")
        fi
        power_samples+=($power)
        
        # Time the transcription
        start_time=$(date +%s.%N)
        
        response=$(curl -s -X POST \
            -F "file=@$TEST_FILE" \
            -F "model=$model" \
            -F "response_format=json" \
            http://localhost:9000/v1/audio/transcriptions)
        
        end_time=$(date +%s.%N)
        elapsed=$(echo "$end_time - $start_time" | bc)
        total_time=$(echo "$total_time + $elapsed" | bc)
        
        echo "${elapsed}s"
    done
    
    # Calculate average
    avg_time=$(echo "scale=2; $total_time / 3" | bc)
    
    # Stop container
    docker-compose -f docker-compose.hardware.yml --profile $profile down > /dev/null 2>&1
    
    # Return results
    echo "$gpu_type|$model|$avg_time|${power_samples[1]}"
}

# Run benchmarks
echo -e "\n${BLUE}Starting benchmarks...${NC}"
echo "This will test both GPUs with appropriate models."
echo ""

# Benchmark Intel iGPU
INTEL_RESULTS=$(benchmark_gpu "Intel iGPU" "igpu" "base")

# Benchmark NVIDIA GPU with base model (for comparison)
NVIDIA_BASE_RESULTS=$(benchmark_gpu "NVIDIA GPU (base)" "cuda" "base")

# Benchmark NVIDIA GPU with large model
NVIDIA_LARGE_RESULTS=$(benchmark_gpu "NVIDIA GPU (large-v3)" "cuda" "large-v3")

# Display results
echo ""
echo -e "${GREEN}📊 Benchmark Results${NC}"
echo "=================================================="
echo ""

# Parse results
IFS='|' read -r gpu1 model1 time1 power1 <<< "$INTEL_RESULTS"
IFS='|' read -r gpu2 model2 time2 power2 <<< "$NVIDIA_BASE_RESULTS"
IFS='|' read -r gpu3 model3 time3 power3 <<< "$NVIDIA_LARGE_RESULTS"

# Create comparison table
printf "%-25s %-15s %-12s %-12s\n" "Hardware" "Model" "Time (30s)" "Power"
printf "%-25s %-15s %-12s %-12s\n" "--------" "-----" "-----------" "-----"
printf "%-25s %-15s %-12s %-12s\n" "$gpu1" "$model1" "${time1}s" "${power1:-N/A}W"
printf "%-25s %-15s %-12s %-12s\n" "$gpu2" "$model2" "${time2}s" "${power2:-N/A}W"
printf "%-25s %-15s %-12s %-12s\n" "$gpu3" "$model3" "${time3}s" "${power3:-N/A}W"

echo ""

# Calculate speed ratios
if (( $(echo "$time1 > 0" | bc -l) )); then
    speedup_base=$(echo "scale=1; $time1 / $time2" | bc)
    speedup_large=$(echo "scale=1; $time1 / $time3" | bc)
    
    echo -e "${BLUE}Performance Analysis:${NC}"
    echo "• NVIDIA GPU (base) is ${speedup_base}x faster than Intel iGPU"
    echo "• NVIDIA GPU (large-v3) is ${speedup_large}x faster than Intel iGPU (base)"
    
    # Power efficiency (if available)
    if [ "$power1" != "0" ] && [ "$power2" != "0" ]; then
        efficiency1=$(echo "scale=2; 1000 / ($power1 * $time1)" | bc)
        efficiency2=$(echo "scale=2; 1000 / ($power2 * $time2)" | bc)
        echo ""
        echo -e "${BLUE}Power Efficiency:${NC}"
        echo "• Intel iGPU: ${efficiency1} transcriptions/kWh"
        echo "• NVIDIA GPU: ${efficiency2} transcriptions/kWh"
    fi
fi

echo ""
echo -e "${GREEN}💡 Recommendations:${NC}"
echo "• For battery/mobile use: Intel iGPU (power efficient)"
echo "• For maximum speed: NVIDIA GPU with large-v3 model"
echo "• For balanced usage: NVIDIA GPU with base model"
echo ""

# Clean up test file
rm -f "$TEST_FILE"

echo -e "${YELLOW}Benchmark complete!${NC}"