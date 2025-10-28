#!/bin/bash
#
# Compile simple MEL kernel (single call, no loop) to XCLBIN
# This pattern works with manual aie-opt compilation
#

set -e  # Exit on error

echo "=== Simple MEL Kernel Compilation ==="
echo ""

MLIR_AIE=/home/ucadmin/mlir-aie-source
BUILD_DIR=build_simple
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "[1/8] Lowering MLIR with aie-opt..."
$MLIR_AIE/build/bin/aie-opt \
  --aie-objectFifo-stateful-transform \
  --aie-localize-locks \
  --aie-normalize-address-spaces \
  ../mel_simple_single_call.mlir \
  -o mel_lowered.mlir
echo "✓ Lowered MLIR created"

echo ""
echo "[2/8] Assigning buffer addresses..."
$MLIR_AIE/build/bin/aie-opt \
  --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" \
  mel_lowered.mlir \
  -o mel_buffers.mlir
echo "✓ Buffer addresses assigned"

echo ""
echo "[3/8] Standard lowering (extracting core body)..."
$MLIR_AIE/build/bin/aie-opt \
  --aie-standard-lowering="tilecol=0 tilerow=2" \
  mel_buffers.mlir \
  -o mel_core_extracted.mlir
echo "✓ Core body extracted"

# Check if core function was created
if grep -q "func.func @core_0_2" mel_core_extracted.mlir; then
    echo "✓ Core function found: @core_0_2"
else
    echo "✗ ERROR: Core function not extracted!"
    exit 1
fi

echo ""
echo "[4/8] Compiling C kernel with Peano..."
$MLIR_AIE/ironenv/lib/python3.13/site-packages/llvm-aie/bin/clang++ \
  --target=aie2-none-unknown-elf \
  -I$MLIR_AIE/runtime_lib/x86_64/test_lib/include \
  -c ../mel_kernel_simple.c \
  -o mel_kernel_simple.o
echo "✓ C kernel compiled"

echo ""
echo "[5/8] Generating CDO (control/data overlay)..."
$MLIR_AIE/build/bin/aie-translate \
  --aie-generate-cdo \
  mel_buffers.mlir \
  -o aie_cdo.bin
echo "✓ CDO generated"

echo ""
echo "[6/8] Creating PDI (platform device image)..."
$MLIR_AIE/build/bin/bootgen \
  -arch versal \
  -image bootgen.bif \
  -o aie.pdi \
  -w 2>&1 | grep -v "WARNING" || true

# If bootgen needs BIF file, create it
if [ ! -f aie.pdi ]; then
    echo "Creating bootgen.bif..."
    cat > bootgen.bif <<EOF
all:
{
  id_code = 0x14ca8093
  extended_id_code = 0x01
  id = 0x2
  image
  {
    name = aie_subsystem, id = 0x1c000000
    { type = cdo
      file = aie_cdo.bin
    }
  }
}
EOF
    $MLIR_AIE/build/bin/bootgen \
      -arch versal \
      -image bootgen.bif \
      -o aie.pdi \
      -w 2>&1 | grep -v "WARNING" || true
fi
echo "✓ PDI created"

echo ""
echo "[7/8] Packaging XCLBIN..."
/opt/xilinx/xrt/bin/xclbinutil \
  --add-section PDI:RAW:aie.pdi \
  --output mel_simple.xclbin

# Add EMBEDDED_METADATA for XRT recognition
if [ -f mel_simple.xclbin ]; then
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
      --input mel_simple.xclbin \
      --add-section EMBEDDED_METADATA:RAW:kernel_metadata.xml \
      --output mel_simple_final.xclbin

    mv mel_simple_final.xclbin mel_simple.xclbin
fi
echo "✓ XCLBIN packaged with metadata"

echo ""
echo "[8/8] Verifying XCLBIN..."
/opt/xilinx/xrt/bin/xclbinutil --info --input mel_simple.xclbin > xclbin_info.txt
SIZE=$(stat -c%s mel_simple.xclbin)
echo "✓ XCLBIN created: mel_simple.xclbin ($SIZE bytes)"

echo ""
echo "=== COMPILATION COMPLETE ==="
echo "Output: build_simple/mel_simple.xclbin"
echo ""
echo "To test:"
echo "  python3 ../test_simple_kernel.py"
