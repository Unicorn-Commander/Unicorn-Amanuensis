#!/bin/bash
#
# compile_xclbin.sh
# Compile BFP16 matrix multiplication XCLBin files for XDNA2 NPU
#
# This script compiles complete XCLBin files from MLIR and C++ kernel sources.
# Based on AMD MLIR-AIE programming examples workflow.
#
# Usage:
#   ./compile_xclbin.sh                    # Compile all 3 Whisper kernels
#   ./compile_xclbin.sh 512 512 512        # Compile specific M K N dimensions
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
CYAN='\033[0;36m'
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
LOG_DIR="$BUILD_DIR/logs"

# Compiler paths (from MLIR-AIE ironenv)
PEANO_INSTALL_DIR="${MLIR_AIE_ENV}/lib/python3.13/site-packages/llvm-aie"
PEANO_CC="${PEANO_INSTALL_DIR}/bin/clang++"

# Kernel configuration
KERNEL_SRC="mm_bfp.cc"
TILE_M=64
TILE_K=64
TILE_N=64

# Print configuration
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}BFP16 XCLBin Compilation Script${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""
echo "Configuration:"
echo "  Working directory: $SCRIPT_DIR"
echo "  MLIR-AIE env: $MLIR_AIE_ENV"
echo "  Peano compiler: $PEANO_CC"
echo "  Build directory: $BUILD_DIR"
echo "  Tile size: ${TILE_M}x${TILE_K}x${TILE_N}"
echo ""

# Create build directories
mkdir -p "$BUILD_DIR"
mkdir -p "$MLIR_DIR"
mkdir -p "$OBJ_DIR"
mkdir -p "$XCLBIN_DIR"
mkdir -p "$LOG_DIR"

# Check for MLIR-AIE environment
if [ ! -f "$MLIR_AIE_ENV/bin/activate" ]; then
    echo -e "${RED}ERROR: MLIR-AIE environment not found at $MLIR_AIE_ENV${NC}"
    echo "Please install MLIR-AIE first"
    exit 1
fi

# Check for Peano compiler
if [ ! -f "$PEANO_CC" ]; then
    echo -e "${RED}ERROR: Peano compiler not found at $PEANO_CC${NC}"
    exit 1
fi

# Activate MLIR-AIE environment
echo -e "${GREEN}[1/4] Activating MLIR-AIE environment...${NC}"
source "$MLIR_AIE_ENV/bin/activate"
echo "  Python: $(which python3)"
echo "  aiecc.py: $(which aiecc.py)"
echo ""

# Whisper encoder dimensions
# Kernel 1: 512x512x512 (attention Q/K/V/out projections)
# Kernel 2: 512x512x2048 (FFN fc1 expansion)
# Kernel 3: 512x2048x512 (FFN fc2 reduction)

# If specific dimensions provided, use them
if [ $# -eq 3 ]; then
    KERNELS=(
        "$1:$2:$3:custom"
    )
else
    KERNELS=(
        "512:512:512:attention"
        "512:512:2048:ffn_fc1"
        "512:2048:512:ffn_fc2"
    )
fi

# Function to compile kernel object
compile_kernel_obj() {
    local M=$1
    local K=$2
    local N=$3
    local NAME=$4

    echo -e "${CYAN}>>> Compiling kernel object for ${NAME} (${M}x${K}x${N})...${NC}"

    local OUTPUT_OBJ="$OBJ_DIR/mm_${TILE_M}x${TILE_K}x${TILE_N}.o"
    local LOG_FILE="$LOG_DIR/kernel_${M}x${K}x${N}.log"

    # Compile with Peano (llvm-aie) compiler
    # Note: We compile with tile dimensions, not full matrix dimensions
    "$PEANO_CC" \
        -c "$KERNEL_SRC" \
        -o "$OUTPUT_OBJ" \
        -DDIM_M=$TILE_M -DDIM_K=$TILE_K -DDIM_N=$TILE_N \
        -std=c++23 \
        --target=aie2p \
        > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ] && [ -f "$OUTPUT_OBJ" ]; then
        FILE_SIZE=$(stat -c%s "$OUTPUT_OBJ")
        echo -e "${GREEN}  ✓ Compiled: $OUTPUT_OBJ (${FILE_SIZE} bytes)${NC}"
        return 0
    else
        echo -e "${RED}  ✗ Failed to compile kernel${NC}"
        echo -e "${YELLOW}  Log: $LOG_FILE${NC}"
        cat "$LOG_FILE"
        return 1
    fi
}

# Function to generate MLIR
generate_mlir() {
    local M=$1
    local K=$2
    local N=$3
    local NAME=$4

    echo -e "${CYAN}>>> Generating MLIR for ${NAME} (${M}x${K}x${N})...${NC}"

    local OUTPUT_FILE="$MLIR_DIR/matmul_${M}x${K}x${N}_bfp16.mlir"
    local LOG_FILE="$LOG_DIR/mlir_${M}x${K}x${N}.log"

    python3 generate_whisper_bfp16.py \
        --dev npu2 \
        -M "$M" -K "$K" -N "$N" \
        -m "$TILE_M" -k "$TILE_K" -n "$TILE_N" \
        --dtype_in bf16 \
        --dtype_out bf16 \
        --emulate-bf16-mmul-with-bfp16 True \
        > "$OUTPUT_FILE" 2> "$LOG_FILE"

    if [ $? -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
        FILE_SIZE=$(stat -c%s "$OUTPUT_FILE")
        echo -e "${GREEN}  ✓ Generated: $OUTPUT_FILE (${FILE_SIZE} bytes)${NC}"
        return 0
    else
        echo -e "${RED}  ✗ Failed to generate MLIR${NC}"
        echo -e "${YELLOW}  Log: $LOG_FILE${NC}"
        cat "$LOG_FILE"
        return 1
    fi
}

# Function to compile XCLBin
compile_xclbin() {
    local M=$1
    local K=$2
    local N=$3
    local NAME=$4

    echo -e "${CYAN}>>> Compiling XCLBin for ${NAME} (${M}x${K}x${N})...${NC}"
    echo -e "${YELLOW}    This may take 5-15 minutes...${NC}"

    local MLIR_FILE="$MLIR_DIR/matmul_${M}x${K}x${N}_bfp16.mlir"
    local KERNEL_OBJ="$OBJ_DIR/mm_${TILE_M}x${TILE_K}x${TILE_N}.o"
    local XCLBIN_FILE="$XCLBIN_DIR/matmul_${M}x${K}x${N}_bfp16.xclbin"
    local INSTS_FILE="$XCLBIN_DIR/insts_${M}x${K}x${N}_bfp16.txt"
    local LOG_FILE="$LOG_DIR/xclbin_${M}x${K}x${N}.log"

    # Check prerequisites
    if [ ! -f "$MLIR_FILE" ]; then
        echo -e "${RED}  ✗ MLIR file not found: $MLIR_FILE${NC}"
        return 1
    fi

    if [ ! -f "$KERNEL_OBJ" ]; then
        echo -e "${RED}  ✗ Kernel object not found: $KERNEL_OBJ${NC}"
        return 1
    fi

    # Start timer
    local START_TIME=$(date +%s)

    # Compile XCLBin using aiecc.py
    cd "$BUILD_DIR"
    aiecc.py \
        --aie-generate-xclbin \
        --no-compile-host \
        --xclbin-name="$(basename "$XCLBIN_FILE")" \
        --aie-generate-npu-insts \
        --npu-insts-name="$(basename "$INSTS_FILE")" \
        --no-xchesscc \
        --no-xbridge \
        --peano "$PEANO_INSTALL_DIR" \
        --dynamic-objFifos \
        "mlir/$(basename "$MLIR_FILE")" \
        > "$LOG_FILE" 2>&1

    cd "$SCRIPT_DIR"

    # End timer
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))

    if [ $? -eq 0 ] && [ -f "$XCLBIN_FILE" ]; then
        FILE_SIZE=$(stat -c%s "$XCLBIN_FILE")
        FILE_SIZE_KB=$((FILE_SIZE / 1024))
        echo -e "${GREEN}  ✓ Built: $XCLBIN_FILE (${FILE_SIZE_KB} KB in ${DURATION}s)${NC}"

        # Also check for instruction file
        if [ -f "$INSTS_FILE" ]; then
            INSTS_SIZE=$(stat -c%s "$INSTS_FILE")
            echo -e "${GREEN}  ✓ Instructions: $INSTS_FILE (${INSTS_SIZE} bytes)${NC}"
        fi

        return 0
    else
        echo -e "${RED}  ✗ Failed to build XCLBin (took ${DURATION}s)${NC}"
        echo -e "${YELLOW}  Log: $LOG_FILE${NC}"
        tail -50 "$LOG_FILE"
        return 1
    fi
}

# Main compilation loop
echo -e "${GREEN}[2/4] Compiling kernel objects...${NC}"
echo ""

# Compile kernel object once (same for all dimensions)
compile_kernel_obj 512 512 512 "generic" || exit 1
echo ""

echo -e "${GREEN}[3/4] Generating MLIR for all configurations...${NC}"
echo ""

for kernel in "${KERNELS[@]}"; do
    IFS=':' read -r M K N NAME <<< "$kernel"
    generate_mlir "$M" "$K" "$N" "$NAME" || exit 1
    echo ""
done

echo -e "${GREEN}[4/4] Compiling XCLBin files...${NC}"
echo -e "${YELLOW}WARNING: This step is very slow (5-15 minutes per kernel)${NC}"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

for kernel in "${KERNELS[@]}"; do
    IFS=':' read -r M K N NAME <<< "$kernel"

    if compile_xclbin "$M" "$K" "$N" "$NAME"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
        echo -e "${YELLOW}  Continuing with next kernel...${NC}"
    fi
    echo ""
done

# Summary
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Compilation Summary${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""
echo "Build directory: $BUILD_DIR"
echo ""
echo "Generated MLIR files:"
ls -lh "$MLIR_DIR"/*.mlir 2>/dev/null || echo "  (none)"
echo ""
echo "Compiled kernel objects:"
ls -lh "$OBJ_DIR"/*.o 2>/dev/null || echo "  (none)"
echo ""
echo "Built XCLBin files:"
ls -lh "$XCLBIN_DIR"/*.xclbin 2>/dev/null || echo "  (none)"
echo ""
echo "XCLBin compilation results:"
echo "  Success: $SUCCESS_COUNT"
echo "  Failed:  $FAIL_COUNT"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo -e "${GREEN}✓ Successfully compiled $SUCCESS_COUNT XCLBin file(s)!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Validate XCLBin files with: xclbinutil --info --input <xclbin>"
    echo "  2. Test with XRT runtime"
    echo "  3. Integrate into Whisper encoder"
else
    echo -e "${RED}✗ No XCLBin files were successfully compiled${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check logs in: $LOG_DIR"
    echo "  2. Verify Vitis/Peano installation"
    echo "  3. Check MLIR syntax with: aie-opt --verify-diagnostics <mlir>"
    exit 1
fi

echo ""
