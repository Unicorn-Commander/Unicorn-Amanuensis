#!/bin/bash
set -e

echo "=================================================="
echo "Compiling 32x32 Matmul Kernel (Working Method)"
echo "=================================================="
echo

# Set environment
export PEANO_INSTALL_DIR=/home/ucadmin/.local/lib/python3.13/site-packages/llvm-aie
export PATH=/opt/xilinx/xrt/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages:$PYTHONPATH

echo "Step 1: Recompiling C kernel to object file..."
$PEANO_INSTALL_DIR/bin/clang \
    --target=aie2 \
    -I$PEANO_INSTALL_DIR/../aie_kernels/aie2/include \
    -c matmul_int8_32x32.c \
    -o matmul_32x32.o
echo "✅ C kernel compiled: matmul_32x32.o"
echo

echo "Step 2: Creating archive with object file..."
$PEANO_INSTALL_DIR/bin/llvm-ar rcs \
    build_matmul_32x32/matmul_combined.o \
    matmul_32x32.o
echo "✅ Archive created"
echo

echo "Step 3: Copying to project directory..."
mkdir -p build_matmul_32x32/matmul_32x32.mlir.prj
cp build_matmul_32x32/matmul_combined.o build_matmul_32x32/matmul_32x32.mlir.prj/main_input.o
echo "✅ Object file staged"
echo

echo "Step 4: Generating XCLBIN with aiecc.py (no-chess mode)..."
cd build_matmul_32x32

/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/python3 \
    /home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=matmul_32x32.xclbin \
    --npu-insts-name=insts_32x32.bin \
    ../matmul_32x32.mlir

echo
echo "=================================================="
echo "✅ 32x32 Matmul Kernel Compiled Successfully!"
echo "=================================================="
echo
echo "Build artifacts:"
ls -lh matmul_32x32.xclbin insts_32x32.bin main_sequence.bin
echo
echo "Next step: Test with benchmark"
echo "  python3 test_batched_matmul_benchmark.py --tile-size=32"
