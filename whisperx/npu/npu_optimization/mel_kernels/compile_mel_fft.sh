#!/bin/bash
# Compilation script for mel_fft kernel (Phase 2.2)
# Compiles C kernels to AIE2 ELF, generates CDO, and packages XCLBIN

set -e  # Exit on error

echo "=== Phase 2.2: Compiling Mel Spectrogram Kernel with FFT ==="
echo ""

# Directories
KERNEL_DIR="/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels"
BUILD_DIR="${KERNEL_DIR}/build"
MLIR_AIE_SOURCE="/home/ucadmin/mlir-aie-source"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${KERNEL_DIR}"

echo "Step 1: Compiling C kernel to AIE2 ELF with Peano..."
echo "  Kernel: mel_fft_basic.c"
echo "  Features: FFT + Hann window + Mel filterbank"
echo ""

# Compile mel_fft_basic kernel with LUT include
${MLIR_AIE_SOURCE}/ironenv/lib/python3.13/site-packages/llvm-aie/bin/clang++ \
  -O2 -std=c++20 \
  --target=aie2-none-unknown-elf \
  -I${KERNEL_DIR} \
  -c mel_fft_basic.c \
  -o ${BUILD_DIR}/mel_fft_basic.o

echo "✅ C kernel compiled:"
ls -lh ${BUILD_DIR}/mel_fft_basic.o
echo ""

echo "Step 2: Lowering MLIR (aie-opt)..."
${MLIR_AIE_SOURCE}/build/bin/aie-opt \
  --aie-canonicalize-device \
  --aie-objectFifo-stateful-transform \
  --aie-create-pathfinder-flows \
  --aie-assign-buffer-addresses \
  mel_fft.mlir \
  -o ${BUILD_DIR}/mel_fft_lowered.mlir

echo "✅ MLIR lowered: ${BUILD_DIR}/mel_fft_lowered.mlir"
ls -lh ${BUILD_DIR}/mel_fft_lowered.mlir
echo ""

echo "Step 3: Generating CDO files (aie-translate)..."
cd ${BUILD_DIR}
${MLIR_AIE_SOURCE}/build/bin/aie-translate \
  --aie-generate-cdo \
  mel_fft_lowered.mlir

echo "✅ CDO files generated:"
ls -lh main_aie_cdo_*.bin 2>/dev/null || echo "⚠️ CDO files not found"
echo ""

echo "Step 4: Combining CDO files for XDNA..."
# For XDNA NPU, combine CDO files directly
cat main_aie_cdo_elfs.bin main_aie_cdo_init.bin main_aie_cdo_enable.bin > mel_fft_cdo_combined.bin

echo "✅ Combined CDO files: ${BUILD_DIR}/mel_fft_cdo_combined.bin"
ls -lh mel_fft_cdo_combined.bin
echo ""

echo "Step 5: Packaging XCLBIN (xclbinutil)..."
# Create minimal XCLBIN with just PDI section
/opt/xilinx/xrt/bin/xclbinutil \
  --add-section PDI:RAW:mel_fft_cdo_combined.bin \
  --force \
  --output mel_fft.xclbin

echo "✅ XCLBIN packaged: ${BUILD_DIR}/mel_fft.xclbin"
ls -lh mel_fft.xclbin
echo ""

echo "Step 6: Verifying XCLBIN structure..."
file mel_fft.xclbin
/opt/xilinx/xrt/bin/xclbinutil --info --input mel_fft.xclbin | head -20
echo ""

echo "===================================================================="
echo "✅ PHASE 2.2 COMPILATION COMPLETE!"
echo "===================================================================="
echo ""
echo "Generated files:"
echo "  - ${BUILD_DIR}/mel_fft_basic.o (AIE2 ELF kernel)"
echo "  - ${BUILD_DIR}/mel_fft_lowered.mlir (lowered MLIR)"
echo "  - ${BUILD_DIR}/mel_fft_cdo_combined.bin (Combined CDO)"
echo "  - ${BUILD_DIR}/mel_fft.xclbin (NPU executable)"
echo ""
echo "Kernel capabilities:"
echo "  ✅ Hann window application"
echo "  ✅ 512-point FFT (Cooley-Tukey radix-2)"
echo "  ✅ Magnitude spectrum computation"
echo "  ✅ Mel filterbank (80 bins)"
echo ""
echo "Next: Test on NPU hardware to measure performance"
echo "===================================================================="
