#!/bin/bash
#
# Compile simple MEL kernel - bypassing the core compilation issue
# Strategy: Use empty core with elf_file pointing to pre-compiled C kernel
#

set -e

echo "=== Simple MEL Kernel Compilation (v2) ==="
echo ""

MLIR_AIE=/home/ucadmin/mlir-aie-source
BUILD_DIR=build_simple_v2
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "[1/6] Compiling C kernel with Peano..."
$MLIR_AIE/ironenv/lib/python3.13/site-packages/llvm-aie/bin/clang++ \
  --target=aie2-none-unknown-elf \
  -I$MLIR_AIE/runtime_lib/x86_64/test_lib/include \
  -c ../mel_kernel_simple.c \
  -o mel_kernel_simple.o
echo "✓ C kernel compiled"

echo ""
echo "[2/6] Creating MLIR with pre-compiled kernel (no core body compilation)..."
# We'll use the working mel_int8_complete.mlir pattern but reference our kernel
cp ../mel_int8_complete.mlir mel_modified.mlir

# Update the elf_file reference
sed -i 's/mel_kernel_test_main.o/mel_kernel_simple.o/' mel_modified.mlir
echo "✓ MLIR modified"

echo ""
echo "[3/6] Lowering MLIR..."
$MLIR_AIE/build/bin/aie-opt \
  --aie-objectFifo-stateful-transform \
  --aie-lower-broadcast-packet \
  --aie-dma-to-npu \
  mel_modified.mlir \
  -o mel_lowered.mlir
echo "✓ MLIR lowered"

echo ""
echo "[4/6] Generating CDO..."
$MLIR_AIE/build/bin/aie-translate \
  --aie-generate-cdo \
  --aie-generate-npu \
  --npu-insts-name=insts.txt \
  mel_lowered.mlir
echo "✓ CDO generated"

echo ""
echo "[5/6] Creating XCLBIN with metadata..."
/opt/xilinx/xrt/bin/xclbinutil \
  --add-section PDI:RAW:aie.pdi \
  --output temp.xclbin

# Create kernel metadata
cat > kernel_metadata.xml <<EOF
<?xml version="1.0" encoding="utf-8"?>
<kernel_metadata>
  <kernel name="MLIR_AIE" language="c" attributes="uses_memory(DDR[1])">
    <arg name="opcode" type="uint32_t*" id="0" port="insts" memory="DDR[1]"/>
    <arg name="instr_buffer" type="uint32_t*" id="1" port="instr" memory="DDR[1]"/>
    <arg name="ninstr" type="uint32_t" id="2" port="ninstr" memory=""/>
    <arg name="input" type="uint8_t*" id="3" port="in" memory="DDR[1]"/>
    <arg name="output" type="uint8_t*" id="4" port="out" memory="DDR[1]"/>
  </kernel>
</kernel_metadata>
EOF

/opt/xilinx/xrt/bin/xclbinutil \
  --input temp.xclbin \
  --add-section EMBEDDED_METADATA:RAW:kernel_metadata.xml \
  --output mel_simple.xclbin
echo "✓ XCLBIN created with metadata"

echo ""
echo "[6/6] Verifying..."
SIZE=$(stat -c%s mel_simple.xclbin)
echo "✓ XCLBIN: mel_simple.xclbin ($SIZE bytes)"

echo ""
echo "=== COMPILATION COMPLETE ==="
echo "Output: build_simple_v2/mel_simple.xclbin"
echo ""
echo "To test:"
echo "  python3 ../test_simple_kernel.py"
