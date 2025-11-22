#!/bin/bash
# Compile GELU BF16 Kernel for XDNA1 Phoenix NPU
# Uses tanh approximation with vectorized BF16 operations

set -e

echo "======================================================================"
echo "GELU BF16 Kernel Compilation for XDNA1"
echo "======================================================================"
echo
echo "Source files:"
echo "  - gelu_simple_xdna1.cc - BF16 GELU with fast tanh approximation"
echo "  - gelu_bf16.mlir - MLIR wrapper with ObjectFIFO pattern"
echo
echo "Target: 1024 bfloat16 GELU activation (2048 bytes)"
echo "======================================================================"
echo

# Setup environment
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH

# Verify tools
echo "Step 0: Verify toolchain..."
if [ ! -f "$PEANO_INSTALL_DIR/bin/clang" ]; then
    echo "ERROR: Peano clang not found"
    exit 1
fi
echo "Peano clang: $PEANO_INSTALL_DIR/bin/clang"
echo

# Create build directory
WORK_DIR=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1
BUILD_DIR=$WORK_DIR/build_gelu
mkdir -p $BUILD_DIR
cd $WORK_DIR

echo "Step 1: Compile GELU kernel..."
$PEANO_INSTALL_DIR/bin/clang \
    -O2 \
    -std=c++20 \
    --target=aie2-none-unknown-elf \
    -I/home/ucadmin/mlir-aie-source/third_party/aie_api/include \
    -I/home/ucadmin/mlir-aie-source/aie_runtime_lib/AIE2 \
    -I/home/ucadmin/mlir-aie-source/aie_runtime_lib \
    -I/home/ucadmin/mlir-aie-source/aie_kernels \
    -c gelu_simple_xdna1.cc \
    -o $BUILD_DIR/gelu_simple_xdna1.o

echo "GELU kernel compiled: $(stat -c%s $BUILD_DIR/gelu_simple_xdna1.o) bytes"
echo

echo "Step 2: Create combined object archive..."
$PEANO_INSTALL_DIR/bin/llvm-ar rcs \
    $BUILD_DIR/gelu_simple_xdna1_combined.o \
    $BUILD_DIR/gelu_simple_xdna1.o

echo "Combined archive: $(stat -c%s $BUILD_DIR/gelu_simple_xdna1_combined.o) bytes"
echo

echo "Step 3: Verify symbols in archive..."
$PEANO_INSTALL_DIR/bin/llvm-nm $BUILD_DIR/gelu_simple_xdna1_combined.o | grep -E "gelu" || true
echo

echo "Step 4: Copy MLIR file..."
cp gelu_bf16.mlir $BUILD_DIR/gelu_bf16.mlir
echo "MLIR file prepared"
echo

echo "Step 5: Generate XCLBIN with aiecc.py..."
cd $BUILD_DIR

/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=gelu_bf16.xclbin \
    --npu-insts-name=insts.bin \
    gelu_bf16.mlir

cd $WORK_DIR
echo
echo "======================================================================"
echo "COMPILATION COMPLETE!"
echo "======================================================================"
echo

echo "Generated Files:"
ls -lh $BUILD_DIR/gelu_bf16.xclbin $BUILD_DIR/insts.bin 2>/dev/null || echo "XCLBIN files not found"
echo

echo "Next Step: Test GELU on NPU"
echo "  python3 test_gelu.py"
echo
