#!/bin/bash
# Quick test of simple kernel with working infrastructure
set -e

cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/build

echo "Creating PDI..."
cat > mel_simple.bif <<EOF
all:
{
    id_code = 0x14ca8093
    extended_id_code = 0x01
    id = 0x2
    image {
        name=aie_image, id=0x1c000000
        { type=cdo
          file=main_aie_cdo_init.bin
        }
        { type=cdo
          file=main_aie_cdo_enable.bin
        }
    }
}
EOF

/home/ucadmin/mlir-aie-source/build/bin/bootgen -arch versal -image mel_simple.bif -o mel_simple.pdi -w

UUID="12345678-1234-5678-1234-567812345678"
cp mel_simple.pdi ${UUID}.pdi

echo "Creating XCLBIN..."
cat > aie_partition.json <<EOF
{
  "aie_partition": {
    "name": "",
    "operations_per_cycle": "2048",
    "PDIs": [
      {
        "uuid": "${UUID}",
        "file_name": "${UUID}.pdi",
        "cdo_groups": [
          {
            "name": "DPU",
            "type": "PRIMARY",
            "pdi_id": "0x1",
            "dpu_kernel_ids": ["0x901"],
            "pre_cdo_groups": ["0xc1"]
          }
        ]
      }
    ]
  }
}
EOF

/opt/xilinx/xrt/bin/xclbinutil \
  --add-section MEM_TOPOLOGY:JSON:mem_topology_mel.json \
  --add-section AIE_PARTITION:JSON:aie_partition.json \
  --add-section EMBEDDED_METADATA:RAW:embedded_metadata.xml \
  --add-section IP_LAYOUT:JSON:ip_layout_mel.json \
  --add-section CONNECTIVITY:JSON:connectivity_mel.json \
  --add-section GROUP_CONNECTIVITY:JSON:group_connectivity_mel.json \
  --add-section GROUP_TOPOLOGY:JSON:group_topology.json \
  --force \
  --output mel_simple_test.xclbin

echo "✅ XCLBIN created: $(stat -c%s mel_simple_test.xclbin) bytes"

echo ""
echo "Testing on NPU..."
python3 <<'PYTEST'
import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np

print("Loading XCLBIN...")
device = xrt.device(0)
xclbin_obj = xrt.xclbin('mel_simple_test.xclbin')
uuid = xclbin_obj.get_uuid()
device.register_xclbin(xclbin_obj)
hw_ctx = xrt.hw_context(device, uuid)
kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
print("✅ Loaded")

# Create buffers
input_bo = xrt.bo(device, 800, xrt.bo.flags.host_only, kernel.group_id(3))
output_bo = xrt.bo(device, 80, xrt.bo.flags.host_only, kernel.group_id(4))
instr_bo = xrt.bo(device, 4096, xrt.bo.flags.cacheable, kernel.group_id(1))

# Fill input with test data
input_data = np.arange(200, dtype=np.int32)
input_bo.write(input_data, 0)
input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 800, 0)

instr_data = np.zeros(1024, dtype=np.uint32)
instr_bo.write(instr_data, 0)
instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 4096, 0)

# Execute
print("Executing kernel...")
run = kernel(0, instr_bo, 0, input_bo, output_bo)
run.wait()
print("✅ Execution complete")

# Read output
output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 80, 0)
output_data = np.frombuffer(output_bo.read(80, 0), dtype=np.int8)

print("\nOutput (first 20):", output_data[:20])

# Check if kernel executed
if np.all(output_data == 0):
    print("❌ All zeros - kernel did NOT execute")
elif np.array_equal(output_data, np.arange(80, dtype=np.int8)):
    print("✅ SUCCESS! Pattern [0-79] found - kernel EXECUTED!")
else:
    print("⚠️ Partial execution or unexpected pattern")
    print("Expected:", np.arange(20, dtype=np.int8))
    print("Got:", output_data[:20])
PYTEST
