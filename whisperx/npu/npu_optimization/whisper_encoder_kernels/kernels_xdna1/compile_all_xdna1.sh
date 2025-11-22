#!/bin/bash
#==============================================================================
# XDNA1 (Phoenix NPU) Kernel Compilation Script
#==============================================================================
#
# Purpose: Compile all XDNA1-optimized kernels for AMD Phoenix NPU
# Target: npu1 device (4 columns, AIE2 architecture)
# Compiler: Peano C++ compiler for AIE2
#
# Usage:
#   bash compile_all_xdna1.sh                    # Compile all kernels
#   bash compile_all_xdna1.sh softmax            # Compile single kernel
#   bash compile_all_xdna1.sh --verbose          # Verbose output
#
# Output: .o files ready for MLIR integration
#
#==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Compilation settings
TARGET="aie2"
VERBOSE=0

# Check for verbose flag
if [[ "$1" == "--verbose" ]] || [[ "$2" == "--verbose" ]]; then
    VERBOSE=1
fi

#==============================================================================
# Function: Print colored message
#==============================================================================
print_msg() {
    local color=$1
    local msg=$2
    echo -e "${color}${msg}${NC}"
}

#==============================================================================
# Function: Find Peano compiler
#==============================================================================
find_peano() {
    # Try common locations
    local peano_paths=(
        "$HOME/.local/lib/python3.13/site-packages/llvm-aie/bin/clang"
        "/opt/mlir-aie/bin/peano"
        "/usr/local/bin/peano"
        "$HOME/.local/bin/peano"
        "/opt/xilinx/mlir-aie/bin/peano"
    )

    for path in "${peano_paths[@]}"; do
        if [[ -x "$path" ]]; then
            echo "$path"
            return 0
        fi
    done

    # Try to find in PATH
    if command -v peano &> /dev/null; then
        echo "peano"
        return 0
    fi

    # Try chess (alternative compiler name)
    if command -v xchesscc &> /dev/null; then
        echo "xchesscc"
        return 0
    fi

    return 1
}

#==============================================================================
# Function: Compile single kernel
#==============================================================================
compile_kernel() {
    local source=$1
    local output="${source%.cc}.o"
    local kernel_name=$(basename "$source" .cc)

    print_msg "$BLUE" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_msg "$YELLOW" "Compiling: $kernel_name"
    print_msg "$BLUE" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check if source exists
    if [[ ! -f "$source" ]]; then
        print_msg "$RED" "❌ ERROR: Source file not found: $source"
        return 1
    fi

    # Include paths for AIE API headers
    local include_paths=(
        "-I/home/ucadmin/mlir-aie-source/third_party/aie_api/include"
        "-I/home/ucadmin/mlir-aie-source/aie_runtime_lib/AIE2"
        "-I/home/ucadmin/mlir-aie-source/aie_runtime_lib"
        "-I/home/ucadmin/mlir-aie-source/aie_kernels"
    )

    # Compilation command with proper flags
    local compile_cmd="$PEANO -O2 -std=c++20 --target=aie2-none-unknown-elf ${include_paths[@]} -c $source -o $output"

    if [[ $VERBOSE -eq 1 ]]; then
        print_msg "$BLUE" "Command: $compile_cmd"
    fi

    # Compile
    local start_time=$(date +%s.%N)

    if $compile_cmd 2>&1 | tee compile_${kernel_name}.log; then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)

        # Get file sizes
        local src_size=$(du -h "$source" | cut -f1)
        local obj_size=$(du -h "$output" | cut -f1)

        print_msg "$GREEN" "✅ SUCCESS: $kernel_name compiled in ${duration}s"
        print_msg "$GREEN" "   Source: $src_size → Object: $obj_size"
        print_msg "$GREEN" "   Output: $output"

        # Store success
        echo "$kernel_name,$duration,$src_size,$obj_size,SUCCESS" >> compilation_results.csv

        return 0
    else
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)

        print_msg "$RED" "❌ FAILED: $kernel_name compilation failed after ${duration}s"
        print_msg "$RED" "   See compile_${kernel_name}.log for details"

        # Store failure
        echo "$kernel_name,$duration,N/A,N/A,FAILED" >> compilation_results.csv

        return 1
    fi
}

#==============================================================================
# Main script
#==============================================================================
print_msg "$BLUE" "╔════════════════════════════════════════════════════════════╗"
print_msg "$BLUE" "║   XDNA1 (Phoenix NPU) Kernel Compilation                  ║"
print_msg "$BLUE" "║   Target: npu1 (4 columns, 16 TOPS INT8)                  ║"
print_msg "$BLUE" "╚════════════════════════════════════════════════════════════╝"
echo ""

# Find Peano compiler
print_msg "$YELLOW" "🔍 Locating Peano compiler..."
if ! PEANO=$(find_peano); then
    print_msg "$RED" "❌ ERROR: Peano compiler not found!"
    print_msg "$RED" ""
    print_msg "$RED" "Please install Peano compiler from:"
    print_msg "$RED" "  - AMD/Xilinx MLIR-AIE package"
    print_msg "$RED" "  - Or use xchesscc (Vitis AI alternative)"
    print_msg "$RED" ""
    print_msg "$RED" "Installation paths searched:"
    print_msg "$RED" "  /opt/mlir-aie/bin/peano"
    print_msg "$RED" "  /usr/local/bin/peano"
    print_msg "$RED" "  ~/.local/bin/peano"
    print_msg "$RED" "  /opt/xilinx/mlir-aie/bin/peano"
    exit 1
fi

print_msg "$GREEN" "✅ Found Peano: $PEANO"
echo ""

# Initialize CSV results
echo "Kernel,Duration(s),SourceSize,ObjectSize,Status" > compilation_results.csv

# Kernel list
KERNELS=(
    "softmax_xdna1.cc"
    "gelu_optimized_xdna1.cc"
    "swiglu_xdna1.cc"
    "softmax_bf16_xdna1.cc"
)

# Check if specific kernel requested
if [[ -n "$1" ]] && [[ "$1" != "--verbose" ]]; then
    KERNELS=("${1}_xdna1.cc")
    print_msg "$YELLOW" "Compiling single kernel: $1"
    echo ""
fi

# Compile all kernels
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL=${#KERNELS[@]}

for kernel in "${KERNELS[@]}"; do
    if compile_kernel "$kernel"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
    echo ""
done

#==============================================================================
# Summary
#==============================================================================
print_msg "$BLUE" "╔════════════════════════════════════════════════════════════╗"
print_msg "$BLUE" "║   Compilation Summary                                      ║"
print_msg "$BLUE" "╚════════════════════════════════════════════════════════════╝"
echo ""

print_msg "$YELLOW" "Total Kernels: $TOTAL"
print_msg "$GREEN" "✅ Successful: $SUCCESS_COUNT"

if [[ $FAIL_COUNT -gt 0 ]]; then
    print_msg "$RED" "❌ Failed: $FAIL_COUNT"
fi

echo ""

# Display results table
if command -v column &> /dev/null; then
    print_msg "$BLUE" "Detailed Results:"
    cat compilation_results.csv | column -t -s ','
else
    print_msg "$BLUE" "Detailed Results (see compilation_results.csv):"
    cat compilation_results.csv
fi

echo ""

# Final status
if [[ $FAIL_COUNT -eq 0 ]]; then
    print_msg "$GREEN" "╔════════════════════════════════════════════════════════════╗"
    print_msg "$GREEN" "║   ✅ ALL KERNELS COMPILED SUCCESSFULLY!                   ║"
    print_msg "$GREEN" "╚════════════════════════════════════════════════════════════╝"
    echo ""
    print_msg "$GREEN" "Next steps:"
    print_msg "$GREEN" "  1. Integrate .o files with MLIR designs"
    print_msg "$GREEN" "  2. Generate XCLBIN with aie-translate"
    print_msg "$GREEN" "  3. Test on Phoenix NPU with XRT"
    echo ""
    exit 0
else
    print_msg "$RED" "╔════════════════════════════════════════════════════════════╗"
    print_msg "$RED" "║   ❌ SOME KERNELS FAILED TO COMPILE                       ║"
    print_msg "$RED" "╚════════════════════════════════════════════════════════════╝"
    echo ""
    print_msg "$RED" "Check compile_*.log files for error details"
    echo ""
    exit 1
fi
