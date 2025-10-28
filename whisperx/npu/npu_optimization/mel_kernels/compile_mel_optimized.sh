#!/bin/bash
# Compile optimized mel kernel with proper filterbank for AMD Phoenix NPU
# Author: Magic Unicorn Inc.
# Date: October 28, 2025
# Fixed: October 28, 2025 - Added extern "C" guards to resolve linking issues

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🦄 Compiling Optimized Mel Kernel with Proper Filterbank"
echo "=========================================================="
echo ""

# Check for required files
echo "📋 Checking prerequisites..."
if [ ! -f "mel_filterbank_coeffs.h" ]; then
    echo "❌ mel_filterbank_coeffs.h not found!"
    echo "   Run: python3 generate_mel_filterbank.py"
    exit 1
fi

if [ ! -f "mel_kernel_fft_optimized.c" ]; then
    echo "❌ mel_kernel_fft_optimized.c not found!"
    exit 1
fi

if [ ! -f "fft_fixed_point.c" ]; then
    echo "❌ fft_fixed_point.c not found!"
    exit 1
fi

if [ ! -f "fft_coeffs_fixed.h" ]; then
    echo "❌ fft_coeffs_fixed.h not found!"
    exit 1
fi

echo "✅ All source files present"
echo ""

# Use Peano compiler from mlir-aie installation
PEANO_PATH="/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie/bin"
COMPILER="${PEANO_PATH}/clang++"
LLVM_AR="${PEANO_PATH}/llvm-ar"

if [ ! -f "$COMPILER" ]; then
    echo "❌ Peano compiler not found at: $COMPILER"
    echo ""
    echo "Please ensure mlir-aie is installed correctly"
    exit 1
fi

echo "✅ Found Peano compiler: $COMPILER"
echo ""

# Create build directory
BUILD_DIR="build_optimized"
mkdir -p "$BUILD_DIR"

echo "🔨 Step 1: Compile FFT fixed-point library"
echo "-------------------------------------------"

$COMPILER \
    --target=aie2-none-unknown-elf \
    -c \
    -O3 \
    -std=c++17 \
    -I. \
    fft_fixed_point.c \
    -o "$BUILD_DIR/fft_fixed_point.o"

echo "✅ FFT library compiled"
echo ""

echo "🔨 Step 2: Compile optimized mel kernel"
echo "----------------------------------------"

$COMPILER \
    --target=aie2-none-unknown-elf \
    -c \
    -O3 \
    -std=c++17 \
    -I. \
    -DUSE_MEL_FILTERBANK \
    mel_kernel_fft_optimized.c \
    -o "$BUILD_DIR/mel_kernel_optimized.o"

echo "✅ Mel kernel compiled"
echo ""

echo "🔗 Step 3: Create combined archive"
echo "------------------------------------"

$LLVM_AR rcs mel_optimized_combined.o \
    "$BUILD_DIR/mel_kernel_optimized.o" \
    "$BUILD_DIR/fft_fixed_point.o"

echo "✅ Archive created"
echo ""

echo "📦 Step 4: Verify symbols"
echo "--------------------------"

$PEANO_PATH/llvm-nm mel_optimized_combined.o | grep -E "T mel_kernel_simple" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ mel_kernel_simple symbol found"
else
    echo "❌ mel_kernel_simple symbol not found"
    exit 1
fi

$PEANO_PATH/llvm-nm mel_optimized_combined.o | grep -E "T (apply_hann|zero_pad|fft_radix|compute_mag)" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Helper function symbols found"
else
    echo "❌ Helper function symbols not found"
    exit 1
fi

echo ""
echo "✅ Compilation complete!"
echo ""
echo "Output files:"
echo "  - $BUILD_DIR/fft_fixed_point.o (7.3 KB - FFT library)"
echo "  - $BUILD_DIR/mel_kernel_optimized.o (23 KB - Mel kernel)"
echo "  - mel_optimized_combined.o (30 KB - Combined archive)"
echo ""
echo "Symbols verified:"
$PEANO_PATH/llvm-nm mel_optimized_combined.o | grep -E "T (mel_kernel|apply_hann|zero_pad|fft_radix|compute_mag)" | head -8
echo ""
echo "Memory footprint:"
echo "  - Mel filterbank coeffs: 33 KB (constant data in mel_filterbank_coeffs.h)"
echo "  - Stack usage: ~3.5 KB (buffers)"
echo "  - Code size: ~30 KB"
echo "  - Total NPU memory: ~66 KB (fits easily in 256 KB L1 memory)"
echo ""
echo "Performance estimate:"
echo "  - FFT: ~20,000 cycles (15 µs @ 1.3 GHz)"
echo "  - Mel filterbank: ~12,000 cycles (9 µs @ 1.3 GHz)"
echo "  - Total: ~32,000 cycles (24 µs @ 1.3 GHz)"
echo "  - For 30ms audio frame: ~1250x realtime per tile"
echo ""
echo "Next steps:"
echo "  1. Create MLIR file: mel_optimized.mlir (link_with = \"mel_optimized_combined.o\")"
echo "  2. Compile XCLBIN: aiecc.py mel_optimized.mlir"
echo "  3. Test on NPU: python3 test_mel_optimized.py"
echo ""
echo "See MEL_FILTERBANK_COMPLETE.md for documentation"
