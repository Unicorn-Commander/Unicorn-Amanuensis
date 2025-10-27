#!/usr/bin/env python3
"""
Minimal XCLBIN generator for Phoenix NPU
Bypasses aiecc.py Python API issues by using C++ tools directly
"""

import json
import subprocess
import sys
import os

def run_cmd(cmd, desc):
    """Run command and check result"""
    print(f"[*] {desc}...")
    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] FAILED: {desc}")
        print(f"    stdout: {result.stdout}")
        print(f"    stderr: {result.stderr}")
        sys.exit(1)
    print(f"[✓] {desc} - SUCCESS")
    return result

def main():
    print("\n" + "="*60)
    print("NPU XCLBIN Generator - Direct xclbinutil Approach")
    print("="*60 + "\n")

    # Files we have
    mlir_file = "passthrough_step3.mlir"  # With BD IDs
    kernel_elf = "passthrough_kernel_new.o"
    npu_bin = "passthrough_npu.bin"

    # Output file
    xclbin_file = "passthrough.xclbin"

    print(f"[*] Input files:")
    print(f"    - MLIR: {mlir_file}")
    print(f"    - Kernel ELF: {kernel_elf}")
    print(f"    - NPU instructions: {npu_bin}")
    print(f"\n[*] Output: {xclbin_file}\n")

    # Check files exist
    for f in [mlir_file, kernel_elf, npu_bin]:
        if not os.path.exists(f):
            print(f"[!] ERROR: {f} not found!")
            sys.exit(1)

    # Step 1: Skip CDO generation - use NPU binary directly
    print("\n--- Step 1: Using NPU binary as PDI ---")
    print("[i] NPU devices use instruction sequences, not CDO files")
    print(f"[✓] Using {npu_bin} directly")

    # Step 2: Create minimal JSON metadata
    print("\n--- Step 2: Create JSON metadata ---")

    # Memory topology - minimal for NPU
    mem_topology = {
        "mem_topology": {
            "m_count": "1",
            "m_mem_data": [
                {
                    "m_type": "MEM_DRAM",
                    "m_used": "1",
                    "m_sizeKB": "0x10000",
                    "m_tag": "DDR",
                    "m_base_address": "0x0"
                }
            ]
        }
    }

    with open("mem_topology.json", "w") as f:
        json.dump(mem_topology, f, indent=2)
    print("[✓] mem_topology.json created")

    # Kernel metadata - use PS (Processing System) type for NPU
    # or DNASC (DNA Soft Controller) which is used for AI engines
    kernels = {
        "ip_layout": {
            "m_count": "1",
            "m_ip_data": [
                {
                    "m_type": "DNASC",  # DNA Soft Controller for AI Engine
                    "m_name": "MLIR_AIE",
                    "m_base_address": "0",
                    "properties": "0"
                }
            ]
        }
    }

    with open("kernels.json", "w") as f:
        json.dump(kernels, f, indent=2)
    print("[✓] kernels.json created")

    # AIE partition - for Phoenix NPU (with 'partition' node as xclbinutil expects)
    aie_partition = {
        "partition": {
            "name": "QoS",
            "column_width": 1,
            "start_columns": [0],
            "num_columns": 4,
            "num_rows": 6,
            "is_active": True,
            "operations_per_cycle": 2000
        }
    }

    with open("aie_partition.json", "w") as f:
        json.dump(aie_partition, f, indent=2)
    print("[✓] aie_partition.json created")

    # Step 3: Package into XCLBIN
    print("\n--- Step 3: Package XCLBIN ---")

    # Use xclbinutil to create XCLBIN with NPU binary
    # For NPU, we package the instruction sequence directly
    # Try minimal XCLBIN without IP_LAYOUT for now
    # NPU devices may not need traditional IP_LAYOUT
    xclbin_cmd = [
        "/opt/xilinx/xrt/bin/xclbinutil",
        "--add-section", f"BITSTREAM:RAW:{npu_bin}",  # NPU instructions as "bitstream"
        "--add-section", "MEM_TOPOLOGY:JSON:mem_topology.json",
        # Skip IP_LAYOUT for now - unknown IP type
        "--add-section", "AIE_PARTITION:JSON:aie_partition.json",
        "--force",
        "--output", xclbin_file
    ]

    run_cmd(xclbin_cmd, "Package XCLBIN")

    # Verify result
    if os.path.exists(xclbin_file):
        size = os.path.getsize(xclbin_file)
        print(f"\n[✓] SUCCESS! Generated {xclbin_file} ({size} bytes)")

        # Show info
        print("\n--- XCLBIN Info ---")
        subprocess.run(["xclbinutil", "--info", "--input", xclbin_file])

        return 0
    else:
        print(f"\n[!] ERROR: {xclbin_file} was not created!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
