#!/bin/bash
#
# build_bfp16_kernels.sh
# Build BFP16 matrix multiplication kernels for Whisper encoder on XDNA2 NPU
#
# This script:
# 1. Activates MLIR-AIE environment
# 2. Generates MLIR for all three Whisper dimensions
# 3. Compiles mm_bfp.cc kernel to object files
# 4. (Optional) Compiles XCLBin files for NPU deployment
#
# Usage:
#   ./build_bfp16_kernels.sh            # Generate MLIR only (fast)
#   ./build_bfp16_kernels.sh --compile  # Generate MLIR + compile kernels (slow)
#   ./build_bfp16_kernels.sh --xclbin   # Full build including XCLBin (very slow)
#
# Copyright (C) 2025, Magic Unicorn Unconventional Technology & Stuff Inc
# Licensed under the Apache License v2.0 with LLVM Exceptions

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Build configuration
MLIR_AIE_ENV="${HOME}/mlir-aie/ironenv"
BUILD_DIR="$SCRIPT_DIR/build"
MLIR_DIR="$BUILD_DIR/mlir"
OBJ_DIR="$BUILD_DIR/obj"
XCLBIN_DIR="$BUILD_DIR/xclbin"

# Parse command line arguments
COMPILE_KERNELS=false
BUILD_XCLBIN=false

for arg in "$@"; do
    case $arg in
        --compile)
            COMPILE_KERNELS=true
            ;;
        --xclbin)
            COMPILE_KERNELS=true
            BUILD_XCLBIN=true
            ;;
        -h|--help)
            echo "Usage: $0 [--compile] [--xclbin]"
            echo ""
            echo "Options:"
            echo "  --compile   Compile C++ kernels to object files (adds ~5 minutes)"
            echo "  --xclbin    Full build including XCLBin generation (adds ~30 minutes per kernel)"
            echo ""
            echo "Default: Generate MLIR only (fast, ~30 seconds)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            exit 1
            ;;
    esac
done

# Print configuration
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}BFP16 Kernel Build Script${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""
echo "Configuration:"
echo "  Working directory: $SCRIPT_DIR"
echo "  MLIR-AIE env: $MLIR_AIE_ENV"
echo "  Build directory: $BUILD_DIR"
echo "  Compile kernels: $COMPILE_KERNELS"
echo "  Build XCLBin: $BUILD_XCLBIN"
echo ""

# Create build directories
mkdir -p "$BUILD_DIR"
mkdir -p "$MLIR_DIR"
mkdir -p "$OBJ_DIR"
mkdir -p "$XCLBIN_DIR"

# Check for MLIR-AIE environment
if [ ! -f "$MLIR_AIE_ENV/bin/activate" ]; then
    echo -e "${RED}ERROR: MLIR-AIE environment not found at $MLIR_AIE_ENV${NC}"
    echo "Please install MLIR-AIE first:"
    echo "  cd ~/mlir-aie"
    echo "  python3 -m venv ironenv"
    echo "  source ironenv/bin/activate"
    echo "  pip install -r python/requirements.txt"
    exit 1
fi

# Activate MLIR-AIE environment
echo -e "${GREEN}[1/4] Activating MLIR-AIE environment...${NC}"
source "$MLIR_AIE_ENV/bin/activate"
python3 --version
echo ""

# Whisper encoder dimensions
# Kernel 1: 512x512x512 (attention Q/K/V/out projections)
# Kernel 2: 512x512x2048 (FFN fc1 expansion)
# Kernel 3: 512x2048x512 (FFN fc2 reduction)
KERNELS=(
    "512:512:512:attention"
    "512:512:2048:ffn_fc1"
    "512:2048:512:ffn_fc2"
)

# Generate MLIR for all kernels
echo -e "${GREEN}[2/4] Generating MLIR for Whisper encoder kernels...${NC}"
echo ""

for kernel in "${KERNELS[@]}"; do
    IFS=':' read -r M K N NAME <<< "$kernel"

    echo -e "${YELLOW}Generating MLIR for ${NAME} (M=${M}, K=${K}, N=${N})...${NC}"

    OUTPUT_FILE="$MLIR_DIR/matmul_${M}x${K}x${N}_bfp16.mlir"

    python3 generate_whisper_bfp16.py \
        --dev npu2 \
        -M "$M" -K "$K" -N "$N" \
        --dtype_in bf16 \
        --dtype_out bf16 \
        --emulate-bf16-mmul-with-bfp16 true \
        > "$OUTPUT_FILE"

    if [ $? -eq 0 ]; then
        FILE_SIZE=$(stat -c%s "$OUTPUT_FILE")
        echo -e "${GREEN}✓ Generated: $OUTPUT_FILE (${FILE_SIZE} bytes)${NC}"
    else
        echo -e "${RED}✗ Failed to generate MLIR for ${NAME}${NC}"
        exit 1
    fi
    echo ""
done

echo -e "${GREEN}MLIR generation complete!${NC}"
echo ""
echo "Generated files:"
ls -lh "$MLIR_DIR"/*.mlir
echo ""

# Compile C++ kernels (optional)
if [ "$COMPILE_KERNELS" = true ]; then
    echo -e "${GREEN}[3/4] Compiling C++ kernels...${NC}"
    echo ""

    # Check for AIE compiler
    if ! command -v xchesscc &> /dev/null; then
        echo -e "${YELLOW}WARNING: xchesscc not found in PATH${NC}"
        echo "Kernel compilation requires Vitis AIE tools"
        echo "Skipping kernel compilation..."
        echo ""
    else
        for kernel in "${KERNELS[@]}"; do
            IFS=':' read -r M K N NAME <<< "$kernel"

            echo -e "${YELLOW}Compiling kernel for ${NAME} (${M}x${K}x${N})...${NC}"

            OUTPUT_OBJ="$OBJ_DIR/mm_${M}x${K}x${N}.o"

            # Compile with AIE-ML compiler
            xchesscc \
                -c mm_bfp.cc \
                -o "$OUTPUT_OBJ" \
                -DDIM_M=$M -DDIM_K=$K -DDIM_N=$N \
                -I../aie_kernel_utils.h \
                -std=c++20 \
                -target aie-ml

            if [ $? -eq 0 ]; then
                FILE_SIZE=$(stat -c%s "$OUTPUT_OBJ")
                echo -e "${GREEN}✓ Compiled: $OUTPUT_OBJ (${FILE_SIZE} bytes)${NC}"
            else
                echo -e "${RED}✗ Failed to compile kernel for ${NAME}${NC}"
                exit 1
            fi
            echo ""
        done

        echo -e "${GREEN}Kernel compilation complete!${NC}"
        echo ""
    fi
else
    echo -e "${YELLOW}[3/4] Skipping kernel compilation (use --compile flag)${NC}"
    echo ""
fi

# Build XCLBin (optional, very slow)
if [ "$BUILD_XCLBIN" = true ]; then
    echo -e "${GREEN}[4/4] Building XCLBin files...${NC}"
    echo -e "${YELLOW}WARNING: This will take 10-30 minutes per kernel!${NC}"
    echo ""

    # Check for Vitis
    if ! command -v aiecompiler &> /dev/null; then
        echo -e "${RED}ERROR: aiecompiler not found in PATH${NC}"
        echo "XCLBin build requires Vitis AIE tools"
        exit 1
    fi

    for kernel in "${KERNELS[@]}"; do
        IFS=':' read -r M K N NAME <<< "$kernel"

        echo -e "${YELLOW}Building XCLBin for ${NAME} (${M}x${K}x${N})...${NC}"
        echo "This may take 10-30 minutes..."

        MLIR_FILE="$MLIR_DIR/matmul_${M}x${K}x${N}_bfp16.mlir"
        XCLBIN_FILE="$XCLBIN_DIR/matmul_${M}x${K}x${N}_bfp16.xclbin"

        # Compile MLIR to XCLBin
        # Note: This is a simplified version - actual compilation requires more steps
        aiecc.py \
            --aie-generate-cdo \
            --aie-generate-npu \
            --no-compile-host \
            --xclbin-name="$XCLBIN_FILE" \
            "$MLIR_FILE"

        if [ $? -eq 0 ]; then
            FILE_SIZE=$(stat -c%s "$XCLBIN_FILE")
            echo -e "${GREEN}✓ Built: $XCLBIN_FILE (${FILE_SIZE} bytes)${NC}"
        else
            echo -e "${RED}✗ Failed to build XCLBin for ${NAME}${NC}"
            exit 1
        fi
        echo ""
    done

    echo -e "${GREEN}XCLBin build complete!${NC}"
    echo ""
else
    echo -e "${YELLOW}[4/4] Skipping XCLBin build (use --xclbin flag)${NC}"
    echo ""
fi

# Summary
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Build Summary${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""
echo "Build directory: $BUILD_DIR"
echo ""
echo "Generated MLIR files:"
ls -lh "$MLIR_DIR"/*.mlir 2>/dev/null || echo "  (none)"
echo ""

if [ "$COMPILE_KERNELS" = true ]; then
    echo "Compiled kernel objects:"
    ls -lh "$OBJ_DIR"/*.o 2>/dev/null || echo "  (none)"
    echo ""
fi

if [ "$BUILD_XCLBIN" = true ]; then
    echo "Built XCLBin files:"
    ls -lh "$XCLBIN_DIR"/*.xclbin 2>/dev/null || echo "  (none)"
    echo ""
fi

echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review MLIR files in $MLIR_DIR"
echo "  2. Test MLIR generation with: python3 generate_whisper_bfp16.py --help"
echo "  3. Compile kernels with: $0 --compile"
echo "  4. Build XCLBin with: $0 --xclbin (very slow!)"
echo ""
