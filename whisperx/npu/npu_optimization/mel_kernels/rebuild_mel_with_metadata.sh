#!/bin/bash
# Build mel_int8 kernel with proper metadata using proven approach
# Based on working rebuild_xclbin.sh from October 27, 2025

set -e  # Exit on error

echo "======================================================================"
echo "MEL INT8 Kernel - XCLBIN with Proper Metadata"
echo "======================================================================"
echo

# Setup paths - use source build tools
PEANO=/home/ucadmin/mlir-aie-source/ironenv/lib/python3.13/site-packages/llvm-aie/bin/clang++
AIE_OPT=/home/ucadmin/mlir-aie-source/build/bin/aie-opt
AIE_TRANSLATE=/home/ucadmin/mlir-aie-source/build/bin/aie-translate
BOOTGEN=/home/ucadmin/mlir-aie-source/build/bin/bootgen
XCLBINUTIL=/opt/xilinx/xrt/bin/xclbinutil

WORK_DIR=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels
BUILD_DIR=$WORK_DIR/build
TEMPLATE_DIR=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/build

cd $WORK_DIR

echo "Step 1: Copy existing CDO files (already compiled)..."
# We already have mel_int8_cdo_combined.bin from previous build
ls -lh $BUILD_DIR/mel_int8_cdo_combined.bin
echo "✅ CDO file ready: $(stat -c%s $BUILD_DIR/mel_int8_cdo_combined.bin) bytes"
echo

echo "Step 2: Create metadata JSON files for mel_int8 kernel..."
cat > $BUILD_DIR/aie_partition_mel.json <<'EOF'
{
  "aie_partition": {
    "name": "MEL_INT8",
    "operations_per_cycle": "2048",
    "inference_fingerprint": "23424",
    "pre_post_fingerprint": "12346",
    "partition": {
      "column_width": 1,
      "start_columns": [0]
    },
    "PDIs": [
      {
        "uuid": "87654321-4321-8765-4321-876543218765",
        "file_name": "mel_int8_cdo_combined.bin",
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
EOF
echo "✅ Created aie_partition_mel.json"

# Copy other metadata files from template
cp $TEMPLATE_DIR/mem_topology.json $BUILD_DIR/mem_topology_mel.json
cp $TEMPLATE_DIR/ip_layout.json $BUILD_DIR/ip_layout_mel.json
cp $TEMPLATE_DIR/connectivity.json $BUILD_DIR/connectivity_mel.json
cp $TEMPLATE_DIR/group_connectivity.json $BUILD_DIR/group_connectivity_mel.json

echo "✅ Metadata files prepared"
echo

echo "Step 3: Package XCLBIN with metadata..."
cd $BUILD_DIR
$XCLBINUTIL \
    --add-section AIE_PARTITION:JSON:aie_partition_mel.json \
    --add-section MEM_TOPOLOGY:JSON:mem_topology_mel.json \
    --add-section IP_LAYOUT:JSON:ip_layout_mel.json \
    --add-section CONNECTIVITY:JSON:connectivity_mel.json \
    --add-section GROUP_CONNECTIVITY:JSON:group_connectivity_mel.json \
    --add-section GROUP_TOPOLOGY:JSON:mem_topology_mel.json \
    --output mel_int8_with_metadata.xclbin

echo "✅ XCLBIN packaged: $(stat -c%s mel_int8_with_metadata.xclbin) bytes"
cd $WORK_DIR
echo

echo "Step 4: Validation..."
file $BUILD_DIR/mel_int8_with_metadata.xclbin
echo
$XCLBINUTIL --info --input $BUILD_DIR/mel_int8_with_metadata.xclbin | head -50
echo

echo "======================================================================"
echo "✅ MEL INT8 XCLBIN WITH METADATA COMPLETE!"
echo "======================================================================"
echo
echo "Generated: $BUILD_DIR/mel_int8_with_metadata.xclbin"
echo
echo "Next: Test loading on NPU"
echo "  cd $WORK_DIR"
echo "  python3 test_xclbin_load.py"
echo
