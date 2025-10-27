#!/bin/bash
# Compilation script for mel_simple kernel (Phase 2.1)
# Compiles C kernels to AIE2 ELF, generates CDO, PDI, and XCLBIN

set -e  # Exit on error

echo "=== Phase 2.1: Compiling Simple Mel Spectrogram Kernel ==="
echo ""

# Directories
KERNEL_DIR="/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels"
BUILD_DIR="${KERNEL_DIR}/build"
MLIR_AIE_SOURCE="/home/ucadmin/mlir-aie-source"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${KERNEL_DIR}"

echo "Step 1: Compiling C kernels to AIE2 ELF with Peano..."
echo "  Using minimal kernel for Phase 2.1 proof-of-concept..."
echo "  (Full FFT will be added in Phase 2.2)"

# Compile minimal mel kernel (simpler for initial testing)
${MLIR_AIE_SOURCE}/ironenv/lib/python3.13/site-packages/llvm-aie/bin/clang++ \
  -O2 -std=c++20 \
  --target=aie2-none-unknown-elf \
  -c mel_simple_minimal.c \
  -o ${BUILD_DIR}/mel_simple.o

echo "✅ C kernel compiled:"
ls -lh ${BUILD_DIR}/mel_simple.o
echo ""

echo "Step 2: Lowering MLIR (aie-opt)..."
# Use the source-built aie-opt
${MLIR_AIE_SOURCE}/build/bin/aie-opt \
  --aie-canonicalize-device \
  --aie-objectFifo-stateful-transform \
  --aie-create-pathfinder-flows \
  --aie-assign-buffer-addresses \
  mel_simple.mlir \
  -o ${BUILD_DIR}/mel_simple_lowered.mlir

echo "✅ MLIR lowered: ${BUILD_DIR}/mel_simple_lowered.mlir"
echo ""

echo "Step 3: Generating CDO files (aie-translate)..."
cd ${BUILD_DIR}
${MLIR_AIE_SOURCE}/build/bin/aie-translate \
  --aie-generate-cdo \
  mel_simple_lowered.mlir

echo "✅ CDO files generated:"
ls -lh main_aie_cdo_*.bin 2>/dev/null || echo "⚠️ CDO files not found"
echo ""

echo "Step 4: Combining CDO files for XDNA..."
# For XDNA NPU, combine CDO files directly (no PDI needed)
cat main_aie_cdo_elfs.bin main_aie_cdo_init.bin main_aie_cdo_enable.bin > aie_cdo_combined.bin

echo "✅ Combined CDO files: ${BUILD_DIR}/aie_cdo_combined.bin"
ls -lh aie_cdo_combined.bin
echo ""

echo "Step 6: Creating XCLBIN metadata JSON files..."

# AIE Partition
cat > aie_partition.json <<'EOF'
{
  "aie_partition": {
    "name": "mel_simple_partition",
    "partition": {
      "column_width": 1,
      "start_columns": [0],
      "num_columns": 4,
      "operations_per_cycle": 400000000
    },
    "PDI_data": [
      {
        "image_type": "CONTROL",
        "pdi_id": 1
      }
    ]
  }
}
EOF

# Memory Topology
cat > mem_topology.json <<'EOF'
{
  "mem_topology": {
    "m_count": 2,
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

# IP Layout
cat > ip_layout.json <<'EOF'
{
  "ip_layout": {
    "m_count": 1,
    "m_ip_data": [
      {
        "m_type": "IP_PS_KERNEL",
        "m_subtype": "DPU",
        "m_functional": "DPU",
        "m_kernel_id": "0x901",
        "m_base_address": "not_used",
        "m_name": "MLIR_AIE:MLIRAIE"
      }
    ]
  }
}
EOF

# Connectivity
cat > connectivity.json <<'EOF'
{
  "connectivity": {
    "m_count": 3,
    "m_connection": [
      {"arg_index": "1", "m_ip_layout_index": "0", "mem_data_index": "1"},
      {"arg_index": "3", "m_ip_layout_index": "0", "mem_data_index": "0"},
      {"arg_index": "4", "m_ip_layout_index": "0", "mem_data_index": "0"}
    ]
  }
}
EOF

# Group Connectivity
cat > group_connectivity.json <<'EOF'
{
  "group_connectivity": {
    "m_count": 6,
    "m_connection": [
      {"arg_index": "1", "m_ip_layout_index": "0", "mem_data_index": "1"},
      {"arg_index": "3", "m_ip_layout_index": "0", "mem_data_index": "0"},
      {"arg_index": "4", "m_ip_layout_index": "0", "mem_data_index": "0"},
      {"arg_index": "1", "m_ip_layout_index": "0", "mem_data_index": "1"},
      {"arg_index": "3", "m_ip_layout_index": "0", "mem_data_index": "0"},
      {"arg_index": "4", "m_ip_layout_index": "0", "mem_data_index": "0"}
    ]
  }
}
EOF

echo "✅ XCLBIN metadata JSON files created"
echo ""

echo "Step 5: Packaging XCLBIN (xclbinutil)..."
# For Phase 2.1 proof-of-concept: create minimal XCLBIN with just PDI
/opt/xilinx/xrt/bin/xclbinutil \
  --add-section PDI:RAW:aie_cdo_combined.bin \
  --force \
  --output mel_simple.xclbin

echo "✅ XCLBIN packaged: ${BUILD_DIR}/mel_simple.xclbin"
ls -lh mel_simple.xclbin
echo ""

echo "Step 6: Verifying XCLBIN structure..."
file mel_simple.xclbin
/opt/xilinx/xrt/bin/xclbinutil --info --input mel_simple.xclbin | head -20
echo ""

echo "==================================================================="
echo "✅ COMPILATION COMPLETE!"
echo "==================================================================="
echo ""
echo "Generated files:"
echo "  - ${BUILD_DIR}/mel_simple.o (AIE2 ELF kernel)"
echo "  - ${BUILD_DIR}/mel_simple_lowered.mlir (lowered MLIR)"
echo "  - ${BUILD_DIR}/aie_cdo_combined.bin (Combined CDO files)"
echo "  - ${BUILD_DIR}/mel_simple.xclbin (NPU executable)"
echo ""
echo "Next step: Run test_mel_simple.py to execute on NPU"
echo "==================================================================="
