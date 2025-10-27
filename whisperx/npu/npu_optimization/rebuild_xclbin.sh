#!/bin/bash
# Complete XCLBIN rebuild pipeline with passthrough core
# October 27, 2025 - After breakthrough execution

set -e  # Exit on error

echo "======================================================================"
echo "NPU XCLBIN Rebuild Pipeline - With Real Passthrough"
echo "======================================================================"
echo

# Setup paths - use source build tools
PEANO=/home/ucadmin/mlir-aie-source/ironenv/lib/python3.13/site-packages/llvm-aie/bin/clang++
AIE_OPT=/home/ucadmin/mlir-aie-source/build/bin/aie-opt
AIE_TRANSLATE=/home/ucadmin/mlir-aie-source/build/bin/aie-translate
BOOTGEN=/home/ucadmin/mlir-aie-source/build/bin/bootgen
XCLBINUTIL=/opt/xilinx/xrt/bin/xclbinutil

WORK_DIR=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization
BUILD_DIR=$WORK_DIR/build

cd $WORK_DIR

echo "Step 1: Clean build directory..."
rm -f $BUILD_DIR/core_passthrough.o
rm -f $BUILD_DIR/passthrough_kernel_new.o
rm -f $BUILD_DIR/input_with_addresses.mlir
rm -f $BUILD_DIR/input_physical.mlir
rm -f $BUILD_DIR/npu_insts.mlir
rm -f $BUILD_DIR/insts.bin
rm -f $BUILD_DIR/main_aie_cdo_*.bin
rm -f $BUILD_DIR/passthrough_complete.pdi
rm -f $BUILD_DIR/final_passthrough.xclbin
echo "✅ Build directory cleaned"
echo

echo "Step 2: Compile passthrough core with Peano..."
$PEANO -O2 -std=c++20 \
    --target=aie2-none-unknown-elf \
    -c core_passthrough.c \
    -o $BUILD_DIR/core_passthrough.o
echo "✅ Passthrough core compiled: $(stat -c%s $BUILD_DIR/core_passthrough.o) bytes"

# Copy to expected name
cp $BUILD_DIR/core_passthrough.o $BUILD_DIR/passthrough_kernel_new.o
echo "✅ ELF file ready: passthrough_kernel_new.o"
echo

echo "Step 3: Phase 1 - MLIR Lowering (aie-opt)..."
$AIE_OPT \
    --aie-canonicalize-device \
    --aie-assign-tile-ids \
    passthrough_step3.mlir -o $BUILD_DIR/input_with_addresses.mlir
echo "✅ Tile IDs assigned"

$AIE_OPT \
    --aie-objectFifo-stateful-transform \
    --aie-create-pathfinder-flows \
    --aie-assign-buffer-addresses \
    $BUILD_DIR/input_with_addresses.mlir -o $BUILD_DIR/input_physical.mlir
echo "✅ Physical placement complete: $(stat -c%s $BUILD_DIR/input_physical.mlir) bytes"
echo

echo "Step 4: Phase 3 - NPU Instruction Generation..."
$AIE_TRANSLATE \
    --aie-generate-npu-dpu-sequence=workDir=$BUILD_DIR \
    $BUILD_DIR/input_physical.mlir -o $BUILD_DIR/npu_insts.mlir
echo "✅ NPU instructions MLIR generated"

$AIE_TRANSLATE \
    --aie-npu-instgen \
    $BUILD_DIR/npu_insts.mlir -o $BUILD_DIR/insts.bin
echo "✅ Binary instructions: $(stat -c%s $BUILD_DIR/insts.bin) bytes ($(( $(stat -c%s $BUILD_DIR/insts.bin) / 4 )) instructions)"
echo

echo "Step 5: Phase 4 - CDO Generation..."
cd $BUILD_DIR
$AIE_TRANSLATE \
    --aie-generate-cdo \
    --aie-cdo-dir=$BUILD_DIR \
    input_physical.mlir
echo "✅ CDO files generated:"
ls -lh main_aie_cdo_*.bin 2>/dev/null || echo "  (checking...)"
TOTAL_CDO=$(du -ch main_aie_cdo_*.bin 2>/dev/null | grep total | cut -f1)
echo "  Total CDO size: $TOTAL_CDO"
cd $WORK_DIR
echo

echo "Step 6: Phase 5 - PDI Generation (bootgen)..."
$BOOTGEN \
    -arch versal \
    -image $BUILD_DIR/design.bif \
    -o $BUILD_DIR/passthrough_complete.pdi \
    -w on
echo "✅ PDI generated: $(stat -c%s $BUILD_DIR/passthrough_complete.pdi) bytes"
echo

echo "Step 7: Phase 6 - XCLBIN Packaging..."
cd $BUILD_DIR
$XCLBINUTIL \
    --add-section AIE_PARTITION:JSON:aie_partition.json \
    --add-section MEM_TOPOLOGY:JSON:mem_topology.json \
    --add-section IP_LAYOUT:JSON:ip_layout.json \
    --add-section CONNECTIVITY:JSON:connectivity.json \
    --add-section GROUP_CONNECTIVITY:JSON:group_connectivity.json \
    --add-section GROUP_TOPOLOGY:JSON:mem_topology.json \
    --output final_passthrough.xclbin
echo "✅ XCLBIN packaged: $(stat -c%s final_passthrough.xclbin) bytes"
cd $WORK_DIR
echo

echo "Step 8: Validation..."
file $BUILD_DIR/final_passthrough.xclbin
echo
$XCLBINUTIL --info --input $BUILD_DIR/final_passthrough.xclbin | head -30
echo

echo "======================================================================"
echo "✅ XCLBIN REBUILD COMPLETE!"
echo "======================================================================"
echo
echo "Generated Files:"
echo "  - core_passthrough.o     : AIE2 ELF with passthrough logic"
echo "  - input_physical.mlir    : Physical MLIR"
echo "  - insts.bin              : NPU instructions"
echo "  - main_aie_cdo_*.bin     : Configuration data"
echo "  - passthrough_complete.pdi : Platform device image"
echo "  - final_passthrough.xclbin : Complete NPU executable"
echo
echo "Next: Run test_xclbin_correct_api.py with new XCLBIN!"
echo "  python3 test_xclbin_correct_api.py"
echo
