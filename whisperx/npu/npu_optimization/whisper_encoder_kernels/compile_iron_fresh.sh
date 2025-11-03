#!/bin/bash
# Compile Fresh IRON-Generated Attention Kernel

set -e

echo "======================================================================"
echo "Compiling Fresh IRON-Generated Multi-Core Attention Kernel"
echo "======================================================================"
echo

# Setup environment
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH

echo "Step 1: Verify kernel object file..."
if [ ! -f "attention_int8_64x64_tiled.o" ]; then
    echo "Copying kernel object from build_attention_iron..."
    cp build_attention_iron/attention_int8_64x64_tiled.o .
fi
ls -lh attention_int8_64x64_tiled.o
echo

echo "Step 2: Compile MLIR to XCLBIN..."
/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=attention_iron_fresh.xclbin \
    --npu-insts-name=insts_iron_fresh.bin \
    fresh_attention_iron.mlir

echo
echo "======================================================================"
echo "✅ COMPILATION COMPLETE!"
echo "======================================================================"
echo

echo "Generated Files:"
ls -lh attention_iron_fresh.xclbin insts_iron_fresh.bin 2>/dev/null || echo "⚠️  Files not found"
echo

echo "Next Step: Test the fresh kernel"
echo "  python3 test_iron_fresh.py"
echo
