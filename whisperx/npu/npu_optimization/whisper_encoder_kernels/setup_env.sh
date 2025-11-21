#!/bin/bash
#
# NPU Kernel Development Environment Setup
# Sets up all required paths for MLIR-AIE2 and Peano compilation
#

# Peano Compiler (AIE2 C++ compiler)
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PEANO=$PEANO_INSTALL_DIR/bin/clang

# MLIR-AIE Python packages
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages:$PYTHONPATH

# XRT and NPU runtime
export PATH=/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/opt/xilinx/xrt/python:$PYTHONPATH

# MLIR and Peano tools
export PATH=$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH

# Verify tools
echo "=== NPU Kernel Development Environment ==="
echo "PEANO: $PEANO"
echo "aiecc.py: $(which aiecc.py)"
echo "XRT: $(which xrt-smi)"
echo ""
echo "Peano version:"
$PEANO --version | head -1
echo ""
echo "aiecc.py available: $(test -f $(which aiecc.py) && echo "YES" || echo "NO")"
echo "XRT available: $(test -f /opt/xilinx/xrt/bin/xrt-smi && echo "YES" || echo "NO")"
echo ""
echo "Environment ready!"
