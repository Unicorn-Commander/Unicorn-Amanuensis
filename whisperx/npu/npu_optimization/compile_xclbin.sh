#!/bin/bash
# Complete XCLBIN Compilation Pipeline (Phases 3-6)
# Based on MLIR_AIE_XCLBIN_COMPILATION_PIPELINE.md
set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🦄 Unicorn XCLBIN Compilation Pipeline${NC}"
echo "=========================================="
echo ""

# Configuration
INPUT_MLIR="input_physical.mlir"  # Output from Phase 1
DEVICE_NAME="passthrough_complete"
BUILD_DIR="build"

# Tool paths
OPT="/home/ucadmin/mlir-aie-source/build/bin/aie-opt"
TRANSLATE="/home/ucadmin/mlir-aie-source/build/bin/aie-translate"
BOOTGEN="/home/ucadmin/mlir-aie-source/build/bin/bootgen"
XCLBINUTIL="/opt/xilinx/xrt/bin/xclbinutil"

# Change to build directory
cd "${BUILD_DIR}"

echo -e "${BLUE}📂 Working Directory: $(pwd)${NC}"
echo ""

# Verify input file exists
if [ ! -f "${INPUT_MLIR}" ]; then
    echo -e "${RED}❌ ERROR: ${INPUT_MLIR} not found!${NC}"
    echo "   Run ./test_phase1.sh first to generate this file."
    exit 1
fi

echo -e "${GREEN}✅ Input file found: ${INPUT_MLIR} ($(stat -c%s ${INPUT_MLIR}) bytes)${NC}"
echo ""

# ============================================================================
# PHASE 3: NPU INSTRUCTION GENERATION
# ============================================================================
echo -e "${BLUE}🔧 Phase 3: NPU Instruction Generation${NC}"
echo "==========================================="

echo "Step 3a: Lowering runtime sequence to NPU instructions..."
${OPT} \
  --pass-pipeline="builtin.module(
    aie.device(
      aie-materialize-bd-chains,
      aie-substitute-shim-dma-allocations,
      aie-assign-runtime-sequence-bd-ids,
      aie-dma-tasks-to-npu,
      aie-dma-to-npu,
      aie-lower-set-lock
    )
  )" \
  ${INPUT_MLIR} \
  -o npu_insts.mlir

if [ $? -eq 0 ] && [ -f npu_insts.mlir ]; then
    echo -e "${GREEN}✅ Step 3a PASSED: npu_insts.mlir generated${NC}"
    echo "   Size: $(stat -c%s npu_insts.mlir) bytes"
else
    echo -e "${RED}❌ Step 3a FAILED${NC}"
    exit 1
fi

echo ""
echo "Step 3b: Translating to binary instructions..."

${TRANSLATE} \
  --aie-npu-to-binary \
  --aie-output-binary \
  npu_insts.mlir \
  -o insts.bin

if [ $? -eq 0 ] && [ -f insts.bin ]; then
    echo -e "${GREEN}✅ Step 3b PASSED: insts.bin generated${NC}"
    echo "   Size: $(stat -c%s insts.bin) bytes"
else
    echo -e "${RED}❌ Step 3b FAILED${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅✅✅ Phase 3 COMPLETE! ✅✅✅${NC}"
echo ""

# ============================================================================
# PHASE 4: CDO GENERATION
# ============================================================================
echo -e "${BLUE}🔧 Phase 4: CDO Generation${NC}"
echo "==========================================="

echo "Generating CDO files with aie-translate..."
${TRANSLATE} \
  --aie-generate-cdo \
  --cdo-unified=false \
  input_physical.mlir

if [ $? -eq 0 ]; then
    # Verify all 3 CDO files were created
    CDO_FILES=(
        "passthrough_complete_aie_cdo_elfs.bin"
        "passthrough_complete_aie_cdo_init.bin"
        "passthrough_complete_aie_cdo_enable.bin"
    )

    ALL_EXIST=true
    for file in "${CDO_FILES[@]}"; do
        if [ -f "${file}" ]; then
            echo -e "${GREEN}✅ Generated: ${file} ($(stat -c%s ${file}) bytes)${NC}"
        else
            echo -e "${RED}❌ Missing: ${file}${NC}"
            ALL_EXIST=false
        fi
    done

    if [ "${ALL_EXIST}" = true ]; then
        echo ""
        echo -e "${GREEN}✅✅✅ Phase 4 COMPLETE! ✅✅✅${NC}"
        echo ""
    else
        echo -e "${RED}❌ Phase 4 FAILED - Not all CDO files generated${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Phase 4 FAILED${NC}"
    exit 1
fi

# ============================================================================
# PHASE 5: PDI GENERATION
# ============================================================================
echo -e "${BLUE}🔧 Phase 5: PDI Generation${NC}"
echo "==========================================="

echo "Step 5a: Creating BIF file..."
cat > design.bif << 'EOF'
all:
{
  id_code = 0x14ca8093
  extended_id_code = 0x01
  image
  {
    name=aie_image, id=0x1c000000
    { type=cdo
      file=passthrough_complete_aie_cdo_elfs.bin
      file=passthrough_complete_aie_cdo_init.bin
      file=passthrough_complete_aie_cdo_enable.bin
    }
  }
}
EOF

if [ -f design.bif ]; then
    echo -e "${GREEN}✅ BIF file created${NC}"
else
    echo -e "${RED}❌ Failed to create BIF file${NC}"
    exit 1
fi

echo ""
echo "Step 5b: Running bootgen..."
${BOOTGEN} \
  -arch versal \
  -image design.bif \
  -o passthrough_complete.pdi \
  -w

if [ $? -eq 0 ] && [ -f passthrough_complete.pdi ]; then
    echo -e "${GREEN}✅ Step 5b PASSED: passthrough_complete.pdi generated${NC}"
    echo "   Size: $(stat -c%s passthrough_complete.pdi) bytes"
else
    echo -e "${RED}❌ Step 5b FAILED${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅✅✅ Phase 5 COMPLETE! ✅✅✅${NC}"
echo ""

# ============================================================================
# PHASE 6: XCLBIN GENERATION
# ============================================================================
echo -e "${BLUE}🔧 Phase 6: XCLBIN Generation${NC}"
echo "==========================================="

echo "Step 6a: Creating JSON metadata files..."

# mem_topology.json
cat > mem_topology.json << 'EOF'
{
  "mem_topology": {
    "m_count": "2",
    "m_mem_data": [
      {
        "m_type": "MEM_DRAM",
        "m_used": "1",
        "m_sizeKB": "0x10000",
        "m_tag": "HOST",
        "m_base_address": "0x4000000"
      },
      {
        "m_type": "MEM_DRAM",
        "m_used": "1",
        "m_sizeKB": "0xc000",
        "m_tag": "SRAM",
        "m_base_address": "0x4000000"
      }
    ]
  }
}
EOF

# aie_partition.json
cat > aie_partition.json << 'EOF'
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
        "uuid": "12345678-1234-5678-1234-567812345678",
        "file_name": "passthrough_complete.pdi",
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

# kernels.json
cat > kernels.json << 'EOF'
{
  "ps-kernels": {
    "kernels": [
      {
        "name": "MLIR_AIE",
        "type": "dpu",
        "extended-data": {
          "subtype": "DPU",
          "functional": "0",
          "dpu_kernel_id": "0x901"
        },
        "arguments": [
          {
            "name": "opcode",
            "address-qualifier": "SCALAR",
            "type": "uint64_t",
            "offset": "0x00"
          },
          {
            "name": "instr",
            "memory-connection": "SRAM",
            "address-qualifier": "GLOBAL",
            "type": "char *",
            "offset": "0x08"
          },
          {
            "name": "ninstr",
            "address-qualifier": "SCALAR",
            "type": "uint32_t",
            "offset": "0x10"
          },
          {
            "name": "bo0",
            "memory-connection": "HOST",
            "address-qualifier": "GLOBAL",
            "type": "void*",
            "offset": "0x14"
          },
          {
            "name": "bo1",
            "memory-connection": "HOST",
            "address-qualifier": "GLOBAL",
            "type": "void*",
            "offset": "0x1c"
          }
        ],
        "instances": [
          {
            "name": "MLIRAIE"
          }
        ]
      }
    ]
  }
}
EOF

echo -e "${GREEN}✅ JSON metadata files created${NC}"

echo ""
echo "Step 6b: Running xclbinutil..."
${XCLBINUTIL} \
  --add-replace-section MEM_TOPOLOGY:JSON:mem_topology.json \
  --add-kernel kernels.json \
  --add-replace-section AIE_PARTITION:JSON:aie_partition.json \
  --force \
  --quiet \
  --output final.xclbin

if [ $? -eq 0 ] && [ -f final.xclbin ]; then
    echo -e "${GREEN}✅ Step 6b PASSED: final.xclbin generated${NC}"
    echo "   Size: $(stat -c%s final.xclbin) bytes"
else
    echo -e "${RED}❌ Step 6b FAILED${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅✅✅ Phase 6 COMPLETE! ✅✅✅${NC}"
echo ""

# ============================================================================
# FINAL STATUS
# ============================================================================
echo -e "${GREEN}==========================================="
echo "🎉🎉🎉 ALL PHASES COMPLETE! 🎉🎉🎉"
echo "===========================================${NC}"
echo ""
echo "Generated files:"
echo "  ✅ npu_insts.mlir - NPU instruction MLIR"
echo "  ✅ insts.bin - Binary instruction stream ($(stat -c%s insts.bin) bytes)"
echo "  ✅ passthrough_complete_aie_cdo_elfs.bin - ELF loading CDO"
echo "  ✅ passthrough_complete_aie_cdo_init.bin - Initialization CDO"
echo "  ✅ passthrough_complete_aie_cdo_enable.bin - Core enable CDO"
echo "  ✅ passthrough_complete.pdi - Platform Device Image ($(stat -c%s passthrough_complete.pdi) bytes)"
echo "  ✅ final.xclbin - NPU-executable binary ($(stat -c%s final.xclbin) bytes)"
echo ""
echo -e "${YELLOW}Next step: Test on NPU hardware with PyXRT${NC}"
echo ""
echo "Test command:"
echo "  python3 ../test_xclbin_load.py build/final.xclbin build/insts.bin"
echo ""
