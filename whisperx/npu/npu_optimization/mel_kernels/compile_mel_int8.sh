#!/bin/bash
# Compilation script for mel_int8_optimized kernel (Phase 2.3)
# INT8 + AIE2 SIMD optimization for maximum performance

set -e  # Exit on error

echo "=== Phase 2.3: Compiling INT8 Optimized Mel Spectrogram Kernel ==="
echo ""

# Directories
KERNEL_DIR="/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels"
BUILD_DIR="${KERNEL_DIR}/build"
MLIR_AIE_SOURCE="/home/ucadmin/mlir-aie-source"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${KERNEL_DIR}"

echo "Step 1: Compiling INT8 optimized kernel to AIE2 ELF with Peano..."
echo "  Kernel: mel_int8_optimized.c"
echo "  Features:"
echo "    - Full INT8 quantization (Q7 format)"
echo "    - AIE2 SIMD vectorization"
echo "    - Block floating-point FFT"
echo "    - Vectorized mel filterbank"
echo "    - Log magnitude lookup table"
echo ""

# Compile INT8 optimized kernel with LUT include
${MLIR_AIE_SOURCE}/ironenv/lib/python3.13/site-packages/llvm-aie/bin/clang++ \
  -O3 -std=c++20 \
  --target=aie2-none-unknown-elf \
  -I${KERNEL_DIR} \
  -c mel_int8_optimized.c \
  -o ${BUILD_DIR}/mel_int8_optimized.o

echo "✅ INT8 kernel compiled:"
ls -lh ${BUILD_DIR}/mel_int8_optimized.o
echo ""

echo "Step 2: Lowering MLIR (aie-opt)..."
${MLIR_AIE_SOURCE}/build/bin/aie-opt \
  --aie-canonicalize-device \
  --aie-objectFifo-stateful-transform \
  --aie-create-pathfinder-flows \
  --aie-assign-buffer-addresses \
  mel_int8.mlir \
  -o ${BUILD_DIR}/mel_int8_lowered.mlir

echo "✅ MLIR lowered: ${BUILD_DIR}/mel_int8_lowered.mlir"
ls -lh ${BUILD_DIR}/mel_int8_lowered.mlir
echo ""

echo "Step 3: Generating CDO files (aie-translate)..."
cd ${BUILD_DIR}
${MLIR_AIE_SOURCE}/build/bin/aie-translate \
  --aie-generate-cdo \
  mel_int8_lowered.mlir

echo "✅ CDO files generated:"
ls -lh main_aie_cdo_*.bin 2>/dev/null || echo "⚠️ CDO files not found"
echo ""

echo "Step 4: Combining CDO files for XDNA..."
# For XDNA NPU, combine CDO files directly
cat main_aie_cdo_elfs.bin main_aie_cdo_init.bin main_aie_cdo_enable.bin > mel_int8_cdo_combined.bin

echo "✅ Combined CDO files: ${BUILD_DIR}/mel_int8_cdo_combined.bin"
ls -lh mel_int8_cdo_combined.bin
echo ""

echo "Step 5: Packaging XCLBIN (xclbinutil)..."
# Create minimal XCLBIN with just PDI section
/opt/xilinx/xrt/bin/xclbinutil \
  --add-section PDI:RAW:mel_int8_cdo_combined.bin \
  --force \
  --output mel_int8_optimized.xclbin

echo "✅ XCLBIN packaged: ${BUILD_DIR}/mel_int8_optimized.xclbin"
ls -lh mel_int8_optimized.xclbin
echo ""

echo "Step 6: Verifying XCLBIN structure..."
file mel_int8_optimized.xclbin
/opt/xilinx/xrt/bin/xclbinutil --info --input mel_int8_optimized.xclbin | head -20
echo ""

echo "===================================================================="
echo "✅ PHASE 2.3 COMPILATION COMPLETE!"
echo "===================================================================="
echo ""
echo "Generated files:"
echo "  - ${BUILD_DIR}/mel_int8_optimized.o (AIE2 ELF kernel)"
echo "  - ${BUILD_DIR}/mel_int8_lowered.mlir (lowered MLIR)"
echo "  - ${BUILD_DIR}/mel_int8_cdo_combined.bin (Combined CDO)"
echo "  - ${BUILD_DIR}/mel_int8_optimized.xclbin (NPU executable)"
echo ""
echo "Kernel optimizations:"
echo "  ✅ Full INT8 quantization (Q7 format)"
echo "  ✅ AIE2 SIMD vectorization (32 INT8 ops/cycle)"
echo "  ✅ Block floating-point FFT"
echo "  ✅ Vectorized mel filterbank"
echo "  ✅ Log magnitude lookup table"
echo "  ✅ Optimized memory access patterns"
echo ""
echo "Expected performance: 60-80x realtime"
echo ""
echo "Next: Phase 2.4 - Full pipeline integration for 220x target"
echo "===================================================================="
