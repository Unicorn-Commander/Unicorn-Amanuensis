#!/bin/bash
# Create proper XCLBIN with platform metadata for AMD Phoenix NPU
# This script adds the missing metadata that prevents XCLBIN loading

set -e

BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/build" && pwd)"
cd "$BUILD_DIR"

echo "==================================================================="
echo "Creating Proper XCLBIN with Platform Metadata"
echo "==================================================================="
echo

# Function to create platform metadata JSON
create_platform_json() {
    local output_file="$1"
    cat > "$output_file" <<'EOF'
{
  "platform_vbnv": "xilinx_phoenix_1x4_202410_1",
  "feature_rom": {
    "major": 1,
    "minor": 0
  }
}
EOF
    echo "✅ Created $output_file"
}

# Function to create kernel metadata JSON
create_kernel_json() {
    local kernel_name="$1"
    local output_file="$2"
    cat > "$output_file" <<EOF
{
  "ip_layout": {
    "m_count": 1,
    "m_ip_data": [
      {
        "m_type": 4,
        "properties": "0",
        "m_base_address": "0",
        "m_name": "$kernel_name"
      }
    ]
  }
}
EOF
    echo "✅ Created $output_file"
}

# Function to create connectivity JSON
create_connectivity_json() {
    local output_file="$1"
    cat > "$output_file" <<'EOF'
{
  "connectivity": {
    "m_count": 0,
    "m_connection": []
  }
}
EOF
    echo "✅ Created $output_file"
}

# Function to create memory topology JSON
create_mem_topology_json() {
    local output_file="$1"
    cat > "$output_file" <<'EOF'
{
  "mem_topology": {
    "m_count": 1,
    "m_mem_data": [
      {
        "m_type": "MEM_DDR4",
        "m_used": "1",
        "m_sizeKB": "0x10000",
        "m_tag": "bank0",
        "m_base_address": "0x0"
      }
    ]
  }
}
EOF
    echo "✅ Created $output_file"
}

# Create metadata JSONs
echo "Step 1: Creating metadata JSON files..."
create_platform_json "platform.json"
create_kernel_json "mel_int8_kernel" "kernels.json"
create_connectivity_json "connectivity.json"
create_mem_topology_json "mem_topology.json"
echo

# Rebuild XCLBIN with metadata for INT8 kernel
echo "Step 2: Rebuilding mel_int8_optimized.xclbin with metadata..."
/opt/xilinx/xrt/bin/xclbinutil \
    --add-section PDI:RAW:mel_int8_cdo_combined.bin \
    --add-section PLATFORM_JSON:JSON:platform.json \
    --add-section IP_LAYOUT:JSON:kernels.json \
    --add-section CONNECTIVITY:JSON:connectivity.json \
    --add-section MEM_TOPOLOGY:JSON:mem_topology.json \
    --output mel_int8_optimized_with_metadata.xclbin

echo "✅ Created mel_int8_optimized_with_metadata.xclbin"
echo

# Verify the new XCLBIN
echo "Step 3: Verifying new XCLBIN structure..."
/opt/xilinx/xrt/bin/xclbinutil --info --input mel_int8_optimized_with_metadata.xclbin

echo
echo "==================================================================="
echo "✅ XCLBIN Created Successfully!"
echo "==================================================================="
echo
echo "Output: $BUILD_DIR/mel_int8_optimized_with_metadata.xclbin"
echo
echo "Test loading with:"
echo "  cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels"
echo "  python3 test_xclbin_load.py"
echo
