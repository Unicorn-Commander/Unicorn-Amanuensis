#!/bin/bash
#
# Complete MEL Kernel build with WORKING LOOP pattern (Pattern A)
# This uses link_with instead of elf_file to allow loop in core
#
# Based on October 27 breakthrough + October 28 solution
#

set -e

echo "======================================================================"
echo "MEL Kernel - Pattern A (Loop in Core)"
echo "======================================================================"
echo

# Paths
PEANO=/home/ucadmin/mlir-aie-source/ironenv/lib/python3.13/site-packages/llvm-aie/bin/clang++
AIE_OPT=/home/ucadmin/mlir-aie-source/build/bin/aie-opt
AIE_TRANSLATE=/home/ucadmin/mlir-aie-source/build/bin/aie-translate
BOOTGEN=/home/ucadmin/mlir-aie-source/build/bin/bootgen
XCLBINUTIL=/opt/xilinx/xrt/bin/xclbinutil
BUILD_DIR=build

cd "$(dirname "$0")"

echo "Step 1: Compile C kernel (NO main, pure function)..."
$PEANO -O2 --target=aie2-none-unknown-elf \
    -c mel_kernel_simple.c -o $BUILD_DIR/mel_kernel_simple.o
echo "✅ C kernel compiled: $(stat -c%s $BUILD_DIR/mel_kernel_simple.o) bytes"
echo

echo "Step 2: Fix MLIR (elf_file → link_with)..."
sed 's/{ elf_file = "mel_kernel_simple.o" }/{ link_with = "mel_kernel_simple.o" }/' \
    mel_with_loop_fixed.mlir > $BUILD_DIR/mel_with_loop_corrected.mlir
echo "✅ MLIR corrected"
echo

echo "Step 3: Lower MLIR..."
$AIE_OPT \
    --aie-canonicalize-device \
    --aie-objectFifo-stateful-transform \
    $BUILD_DIR/mel_with_loop_corrected.mlir -o $BUILD_DIR/mel_loop_physical.mlir
echo "✅ MLIR lowered: $(stat -c%s $BUILD_DIR/mel_loop_physical.mlir) bytes"
echo

echo "Step 4: Extract core and compile..."
$AIE_OPT \
    --aie-standard-lowering="tilecol=0 tilerow=2" \
    $BUILD_DIR/mel_loop_physical.mlir -o $BUILD_DIR/mel_core_lowered.mlir
echo "✅ Core extracted: $(stat -c%s $BUILD_DIR/mel_core_lowered.mlir) bytes"
echo

echo "Step 5: Generate CDO..."
cd $BUILD_DIR
$AIE_TRANSLATE \
    --aie-generate-cdo \
    mel_loop_physical.mlir
cd ..

if [ -f $BUILD_DIR/main_aie_cdo_init.bin ]; then
    echo "✅ CDO files generated:"
    ls -lh $BUILD_DIR/*_aie_cdo_*.bin | awk '{print "   " $9 " - " $5}'
else
    echo "❌ CDO generation failed"
    exit 1
fi
echo

echo "Step 6: Create PDI..."
cat > $BUILD_DIR/mel_loop.bif <<'BIFEOF'
all:
{
    id_code = 0x14ca8093
    extended_id_code = 0x01
    image
    {
        name=aie_image, id=0x1c000000
        { type=cdo
          file=main_aie_cdo_init.bin
        }
        { type=cdo
          file=main_aie_cdo_enable.bin
        }
    }
}
BIFEOF

$BOOTGEN -arch versal -image $BUILD_DIR/mel_loop.bif -w -o $BUILD_DIR/mel_loop.pdi
echo "✅ PDI generated: $(stat -c%s $BUILD_DIR/mel_loop.pdi) bytes"
echo

echo "Step 7: Copy PDI with UUID..."
UUID="87654321-4321-8765-4321-876543218765"
cp $BUILD_DIR/mel_loop.pdi $BUILD_DIR/${UUID}.pdi
echo "✅ PDI copied: ${UUID}.pdi"
echo

echo "Step 8: Create AIE partition JSON..."
cat > $BUILD_DIR/aie_partition_loop.json <<'JSONEOF'
{
    "aie_partition": {
        "name": "QoS",
        "operations_per_cycle": "2048",
        "inference_fingerprint": "23423",
        "pre_post_fingerprint": "12345",
        "partition": {
            "column_width": 1,
            "start_columns": [0]
        },
        "PDIs": [
            {
                "uuid": "87654321-4321-8765-4321-876543218765",
                "file_name": "87654321-4321-8765-4321-876543218765.pdi",
                "cdo_groups": [
                    {
                        "name": "DPU",
                        "type": "PRIMARY",
                        "pdi_id": "0x01",
                        "dpu_kernel_ids": ["0x901"],
                        "pre_cdo_groups": ["0xC1"]
                    }
                ]
            }
        ]
    }
}
JSONEOF
echo "✅ AIE partition JSON created"
echo

echo "Step 9: Package XCLBIN..."
$XCLBINUTIL \
    --add-section MEM_TOPOLOGY:JSON:mem_topology_mel.json \
    --add-section AIE_PARTITION:JSON:$BUILD_DIR/aie_partition_loop.json \
    --add-section EMBEDDED_METADATA:RAW:embedded_metadata.xml \
    --add-section IP_LAYOUT:JSON:ip_layout_mel.json \
    --add-section CONNECTIVITY:JSON:connectivity_mel.json \
    --add-section GROUP_CONNECTIVITY:JSON:group_connectivity_mel.json \
    --add-section GROUP_TOPOLOGY:JSON:group_topology.json \
    --output $BUILD_DIR/mel_loop_final.xclbin

if [ -f $BUILD_DIR/mel_loop_final.xclbin ]; then
    echo "✅ XCLBIN packaged: $(stat -c%s $BUILD_DIR/mel_loop_final.xclbin) bytes"
else
    echo "❌ XCLBIN packaging failed"
    exit 1
fi
echo

echo "Step 10: Validate XCLBIN..."
file $BUILD_DIR/mel_loop_final.xclbin
$XCLBINUTIL --info --input $BUILD_DIR/mel_loop_final.xclbin | head -30
echo

echo "======================================================================"
echo "✅ SUCCESS! MEL Kernel with Loop Pattern Compiled"
echo "======================================================================"
echo
echo "Output: $BUILD_DIR/mel_loop_final.xclbin"
echo
echo "Next: Test on NPU with:"
echo "  python3 test_mel_npu_execution.py --xclbin build/mel_loop_final.xclbin"
echo
