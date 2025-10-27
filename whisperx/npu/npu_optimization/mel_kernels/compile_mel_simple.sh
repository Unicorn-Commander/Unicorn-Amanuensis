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
# Compile both FFT and main kernel together
${MLIR_AIE_SOURCE}/ironenv/lib/python3.13/site-packages/llvm-aie/bin/clang++ \
  -O2 -std=c++20 \
  --target=aie2-none-unknown-elf \
  -c fft_radix2.c mel_simple.c \
  -o ${BUILD_DIR}/mel_simple.o

echo "✅ C kernel compiled: ${BUILD_DIR}/mel_simple.o"
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

echo "Step 4: Creating bootgen BIF file..."
cat > design.bif <<EOF
all:
{
  main_aie_cdo_elfs.bin
  main_aie_cdo_init.bin
  main_aie_cdo_enable.bin
}
EOF

echo "✅ BIF file created"
echo ""

echo "Step 5: Generating PDI (bootgen)..."
${MLIR_AIE_SOURCE}/build/bin/bootgen \
  -arch versal \
  -image design.bif \
  -o mel_simple.pdi \
  -w on

echo "✅ PDI generated: ${BUILD_DIR}/mel_simple.pdi"
ls -lh mel_simple.pdi
echo ""

echo "Step 6: Creating XCLBIN metadata JSON files..."

# AIE Partition
cat > aie_partition.json <<'EOF'
{
  "aie_partition": {
    "name": "mel_simple_partition",
    "column_width": 1,
    "start_columns": [0],
    "num_columns": 4,
    "operations_per_cycle": 400000000
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

echo "Step 7: Packaging XCLBIN (xclbinutil)..."
/opt/xilinx/xrt/bin/xclbinutil \
  --add-section AIE_PARTITION:JSON:aie_partition.json \
  --add-section MEM_TOPOLOGY:JSON:mem_topology.json \
  --add-section IP_LAYOUT:JSON:ip_layout.json \
  --add-section CONNECTIVITY:JSON:connectivity.json \
  --add-section GROUP_CONNECTIVITY:JSON:group_connectivity.json \
  --add-section PDI:RAW:mel_simple.pdi \
  --force \
  --output mel_simple.xclbin

echo "✅ XCLBIN packaged: ${BUILD_DIR}/mel_simple.xclbin"
ls -lh mel_simple.xclbin
echo ""

echo "Step 8: Verifying XCLBIN structure..."
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
echo "  - ${BUILD_DIR}/mel_simple.pdi (Platform Device Image)"
echo "  - ${BUILD_DIR}/mel_simple.xclbin (NPU executable)"
echo ""
echo "Next step: Run test_mel_simple.py to execute on NPU"
echo "==================================================================="
