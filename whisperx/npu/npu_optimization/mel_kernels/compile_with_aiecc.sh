#!/bin/bash
# Compile MLIR kernel using official aiecc.py toolchain
# This should generate XCLBIN with proper metadata

set -e

# Set up environment
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-source/ironenv/lib/python3.13/site-packages/llvm-aie
export PATH=/home/ucadmin/mlir-aie-source/build/bin:$PEANO_INSTALL_DIR/bin:$PATH
export PYTHONPATH=/home/ucadmin/mlir-aie-source/build/python:$PYTHONPATH

MLIR_AIE_BUILD=/home/ucadmin/mlir-aie-source/build
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/build" && pwd)"
cd "$BUILD_DIR"

echo "==================================================================="
echo "Compiling MEL INT8 Kernel with Official aiecc.py"
echo "==================================================================="
echo

echo "Tools available:"
which aiecc.py
which aie-opt
which aie-translate
which bootgen
echo

echo "Step 1: Compile MLIR to XCLBIN with aiecc.py..."
cd ..

# Compile with aiecc.py - this should generate complete XCLBIN
$MLIR_AIE_BUILD/bin/aiecc.py \
    --aie-generate-xclbin \
    --xclbin-name=mel_int8_proper.xclbin \
    --xclbin-device=phoenix \
    mel_int8.mlir

echo
echo "✅ Compilation complete!"
echo

echo "Step 2: Inspect generated XCLBIN..."
/opt/xilinx/xrt/bin/xclbinutil --info --input build/mel_int8_proper.xclbin

echo
echo "==================================================================="
echo "Done!"
echo "==================================================================="
