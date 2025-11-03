#!/bin/bash
# Verification Script - AIE Toolchain for Multi-Core Compilation
# Run this to verify all tools are available

echo "======================================================================"
echo "AIE Toolchain Verification"
echo "======================================================================"
echo

# Setup environment
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH

echo "1. Checking Peano Compiler (AIE2 LLVM)..."
if [ -f "$PEANO_INSTALL_DIR/bin/clang" ]; then
    echo "   ✅ Peano clang: $PEANO_INSTALL_DIR/bin/clang"
    $PEANO_INSTALL_DIR/bin/clang --version | head -1
else
    echo "   ❌ Peano clang NOT FOUND"
    exit 1
fi
echo

echo "2. Checking MLIR-AIE Compiler..."
if command -v aiecc.py &> /dev/null; then
    echo "   ✅ aiecc.py: $(which aiecc.py)"
    aiecc.py --version 2>&1 | head -1 || echo "   (version check not available)"
else
    echo "   ❌ aiecc.py NOT FOUND"
    exit 1
fi
echo

echo "3. Checking LLVM Tools..."
for tool in llvm-ar llvm-nm llvm-objdump; do
    if [ -f "$PEANO_INSTALL_DIR/bin/$tool" ]; then
        echo "   ✅ $tool: available"
    else
        echo "   ❌ $tool: NOT FOUND"
    fi
done
echo

echo "4. Checking XRT Runtime..."
if command -v xrt-smi &> /dev/null; then
    echo "   ✅ XRT installed: $(which xrt-smi)"
    xrt-smi examine 2>&1 | grep -E "(NPU|Device)" | head -3
else
    echo "   ❌ XRT NOT FOUND"
fi
echo

echo "5. Checking NPU Device..."
if [ -e "/dev/accel/accel0" ]; then
    echo "   ✅ NPU device: /dev/accel/accel0"
    ls -l /dev/accel/accel0
else
    echo "   ⚠️  NPU device not found (may need module load)"
fi
echo

echo "6. Testing Compilation (Single-Core)..."
cd "$(dirname "$0")"
if [ -f "compile_attention_64x64.sh" ]; then
    echo "   Testing: compile_attention_64x64.sh"
    bash compile_attention_64x64.sh > /tmp/compile_test.log 2>&1
    if [ $? -eq 0 ]; then
        echo "   ✅ Compilation SUCCESS"
        ls -lh build_attention_64x64/attention_64x64.xclbin 2>/dev/null | awk '{print "   " $0}'
    else
        echo "   ❌ Compilation FAILED - check /tmp/compile_test.log"
    fi
else
    echo "   ⚠️  compile_attention_64x64.sh not found"
fi
echo

echo "======================================================================"
echo "Verification Complete"
echo "======================================================================"
echo
echo "Summary:"
echo "  - Peano Compiler: Available at $PEANO_INSTALL_DIR"
echo "  - MLIR-AIE Tools: Available in venv313"
echo "  - Chess Compiler: NOT NEEDED (use --no-xchesscc)"
echo "  - Single-Core: Compiles successfully"
echo "  - Multi-Core: Use IRON API or batched execution"
echo
echo "For multi-core compilation, see:"
echo "  - MULTICORE_COMPILATION_GUIDE.md (detailed guide)"
echo "  - INVESTIGATION_SUMMARY.md (quick reference)"
echo
