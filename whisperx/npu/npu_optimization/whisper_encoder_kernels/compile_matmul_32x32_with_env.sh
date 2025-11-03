#!/bin/bash
set -e

echo "==================================================="
echo "32x32 Matmul Kernel Compilation (Phoenix NPU)"
echo "4 columns, 4 AIE-ML cores, 15 TOPS INT8"
echo "==================================================="
echo ""

# Source AIETools environment
echo "Step 1: Setting up environment..."
source /home/ucadmin/aietools_env.sh 2>&1 | grep -E "✅|Chess compiler"

# Set up Peano
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PEANO=$PEANO_INSTALL_DIR/bin
export PATH=$PEANO:$PATH

echo "  AIETOOLS_ROOT: $AIETOOLS_ROOT"
echo "  PEANO_INSTALL_DIR: $PEANO_INSTALL_DIR"
echo ""

# Verify compilers
echo "Verifying compilers in PATH:"
which xchesscc && echo "  ✅ xchesscc found" || echo "  ❌ xchesscc not found"
which clang | grep llvm-aie && echo "  ✅ Peano clang found" || echo "  ⚠️  clang found but may not be Peano"
echo ""

# Create build directory
mkdir -p build_matmul_32x32
cd build_matmul_32x32

# Compile C kernel
echo "Step 2: Compiling C kernel to object file..."
$PEANO/clang \
    --target=aie2 \
    -I$PEANO/../aie_kernels/aie2/include \
    -c ../matmul_int8_32x32.c \
    -o matmul_32x32.o

if [ $? -eq 0 ]; then
    echo "  ✅ C kernel compiled: matmul_32x32.o ($(ls -lh matmul_32x32.o | awk '{print $5}'))"
else
    echo "  ❌ C kernel compilation failed"
    exit 1
fi
echo ""

# Compile MLIR kernel
echo "Step 3: Compiling MLIR to XCLBIN with aiecc.py..."
echo "  Using both Chess and Peano compilers..."

/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --sysroot=$PEANO/../sysroot \
    --host-target=x86_64-amd-linux-gnu \
    ../matmul_32x32.mlir \
    -I$PEANO/../aie_kernels/aie2/include \
    -o matmul_32x32.xclbin \
    --xclbin-kernel-name=MLIR_AIE \
    --peano-install-dir=$PEANO_INSTALL_DIR \
    --unified \
    matmul_32x32.o

if [ $? -eq 0 ] && [ -f "matmul_32x32.xclbin" ]; then
    echo ""
    echo "==================================================="
    echo "✅ SUCCESS! XCLBIN Generated"
    echo "==================================================="
    echo ""
    echo "Build artifacts:"
    ls -lh matmul_32x32.o matmul_32x32.xclbin *.bin 2>/dev/null
    echo ""
    echo "Next step: Test kernel on NPU"
    echo "  python3 test_matmul_32x32.py"
else
    echo ""
    echo "❌ XCLBIN generation failed"
    echo "Check errors above"
    exit 1
fi
