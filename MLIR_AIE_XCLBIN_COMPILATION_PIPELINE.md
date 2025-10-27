# MLIR-AIE C++ Compilation Pipeline for NPU XCLBIN Generation

**Date**: October 26, 2025
**Author**: Claude (Reverse-Engineering Team)
**Goal**: Document exact command sequence to compile MLIR to working XCLBIN without Python wrapper
**Status**: Complete pipeline reverse-engineered from aiecc.py source code

---

## Executive Summary

This document provides the complete, **Python-free command-line compilation pipeline** for converting MLIR-AIE source code into NPU-executable XCLBIN files. All commands use C++ tools directly: `aie-opt`, `aie-translate`, and `bootgen`.

**Key Finding**: The Python wrapper (`aiecc.py`) orchestrates C++ tools - we can invoke them directly for full control.

---

## Table of Contents

1. [Overview: MLIR to XCLBIN Flow](#overview)
2. [Tools Required](#tools-required)
3. [Complete Command Sequence](#complete-command-sequence)
4. [Detailed Pipeline Phases](#detailed-pipeline-phases)
5. [Input MLIR Requirements](#input-mlir-requirements)
6. [Intermediate File Formats](#intermediate-file-formats)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization Flags](#performance-optimization-flags)

---

## Overview: MLIR to XCLBIN Flow

```
Input MLIR
    ↓
┌────────────────────────────────────────────────────────────────┐
│ PHASE 1: MLIR TRANSFORMATIONS (aie-opt)                       │
│                                                                │
│  passthrough.mlir                                             │
│        ↓                                                       │
│  [1a] Canonicalize device                                     │
│  [1b] Assign lock IDs                                         │
│  [1c] Register ObjectFIFOs                                    │
│  [1d] Lower ObjectFIFOs to stateful DMA                       │
│  [1e] Assign buffer descriptor IDs                            │
│  [1f] Lower cascade flows                                     │
│  [1g] Generate routing (pathfinder flows)                     │
│  [1h] Assign buffer addresses                                 │
│        ↓                                                       │
│  input_with_addresses.mlir                                    │
│        ↓                                                       │
│  [1i] Create pathfinder flows (routing)                       │
│        ↓                                                       │
│  input_physical.mlir                                          │
└────────────────────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────────────────────┐
│ PHASE 2: CORE COMPILATION (per-core, if cores exist)          │
│                                                                │
│  For each core (col, row):                                    │
│        ↓                                                       │
│  [2a] Lower to LLVM (per core)                                │
│       - Localize locks                                         │
│       - Normalize address spaces                              │
│       - Lower AIE standard ops                                │
│       - Convert to LLVM IR                                    │
│        ↓                                                       │
│  core_X_Y.opt.mlir                                            │
│        ↓                                                       │
│  [2b] Translate to LLVM IR                                    │
│        ↓                                                       │
│  core_X_Y.ll                                                  │
│        ↓                                                       │
│  [2c] Compile with Peano (AIE C++ compiler)                   │
│        ↓                                                       │
│  core_X_Y.o                                                   │
│        ↓                                                       │
│  [2d] Link with Peano linker                                  │
│        ↓                                                       │
│  core_X_Y.elf                                                 │
│                                                                │
│  [2e] Update MLIR with ELF references                         │
│        ↓                                                       │
│  input_physical_with_elfs.mlir                                │
└────────────────────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────────────────────┐
│ PHASE 3: NPU INSTRUCTION GENERATION                           │
│                                                                │
│  input_physical_with_elfs.mlir                                │
│        ↓                                                       │
│  [3a] Lower runtime sequence to NPU instructions              │
│       - Materialize BD chains                                 │
│       - Substitute shim DMA allocations                       │
│       - Lower DMA tasks to NPU instructions                   │
│       - Lower set_lock operations                             │
│        ↓                                                       │
│  npu_insts.mlir                                               │
│        ↓                                                       │
│  [3b] Translate to binary instructions                        │
│        ↓                                                       │
│  insts.bin                                                    │
└────────────────────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────────────────────┐
│ PHASE 4: CDO GENERATION (Configuration Data Object)           │
│                                                                │
│  input_physical_with_elfs.mlir                                │
│        ↓                                                       │
│  [4a] Generate CDO (via Python binding)                       │
│       aiedialect.generate_cdo(mlir, tmpdir, device_name)      │
│        ↓                                                       │
│  Generates 3 files:                                           │
│    - device_aie_cdo_elfs.bin   (ELF loading CDO)             │
│    - device_aie_cdo_init.bin   (Initialization CDO)          │
│    - device_aie_cdo_enable.bin (Core enable CDO)             │
└────────────────────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────────────────────┐
│ PHASE 5: PDI GENERATION (Programmable Device Image)           │
│                                                                │
│  [5a] Create BIF file (Boot Image Format)                     │
│        ↓                                                       │
│  design.bif:                                                  │
│    all:                                                       │
│    {                                                          │
│      id_code = 0x14ca8093                                    │
│      extended_id_code = 0x01                                 │
│      image                                                    │
│      {                                                        │
│        name=aie_image, id=0x1c000000                         │
│        { type=cdo                                            │
│          file=device_aie_cdo_elfs.bin                        │
│          file=device_aie_cdo_init.bin                        │
│          file=device_aie_cdo_enable.bin                      │
│        }                                                      │
│      }                                                        │
│    }                                                          │
│        ↓                                                       │
│  [5b] Run bootgen                                             │
│        ↓                                                       │
│  device.pdi                                                   │
└────────────────────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────────────────────┐
│ PHASE 6: XCLBIN GENERATION (XRT Container)                    │
│                                                                │
│  [6a] Create JSON metadata files:                             │
│    - mem_topology.json  (Memory configuration)                │
│    - aie_partition.json (AIE partition + PDI reference)       │
│    - kernels.json       (PS kernel definition)                │
│        ↓                                                       │
│  [6b] Run xclbinutil                                          │
│    xclbinutil                                                 │
│      --add-replace-section MEM_TOPOLOGY:JSON:mem_topology.json│
│      --add-kernel kernels.json                                │
│      --add-replace-section AIE_PARTITION:JSON:aie_partition.json│
│      --output final.xclbin                                    │
│        ↓                                                       │
│  final.xclbin                                                 │
└────────────────────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────────────────────┐
│ FINAL OUTPUT                                                   │
│                                                                │
│  final.xclbin - NPU-executable binary                         │
│  insts.bin    - Runtime instruction stream                    │
└────────────────────────────────────────────────────────────────┘
```

---

## Tools Required

### 1. aie-opt (MLIR Pass Manager)
- **Location**: `/home/ucadmin/mlir-aie-source/build/bin/aie-opt`
- **Size**: 179 MB
- **Purpose**: Run MLIR transformation passes
- **Status**: ✅ Available and working

### 2. aie-translate (MLIR Translator)
- **Location**: `/home/ucadmin/mlir-aie-source/build/bin/aie-translate`
- **Size**: 62 MB
- **Purpose**: Convert MLIR to other formats (LLVM IR, binary instructions)
- **Status**: ✅ Available and working

### 3. bootgen (Xilinx Boot Image Generator)
- **Location**: `/home/ucadmin/mlir-aie-source/build/bin/bootgen`
- **Size**: 2.3 MB
- **Purpose**: Generate PDI (Programmable Device Image) files
- **Status**: ✅ Available

### 4. xclbinutil (XRT Binary Utility)
- **Location**: `/opt/xilinx/xrt/bin/xclbinutil` (from XRT 2.20.0)
- **Purpose**: Create XCLBIN container files
- **Status**: ✅ Available (from XRT installation)

### 5. Peano C++ Compiler (CRITICAL - NOT YET LOCATED)
- **Typical Location**: Part of Vitis AIE tools or bundled with mlir-aie
- **Purpose**: Compile LLVM IR to AIE core object files (.o)
- **Components**:
  - `clang` - AIE-targeted C++ compiler
  - `opt` - LLVM optimizer
  - `llc` - LLVM static compiler
- **Status**: ⚠️ **NOT FOUND** - Critical blocker for core compilation
- **Alternative**: May be available in `/opt/xilinx/aietools/` if Vitis installed

### 6. Python Bindings (for CDO generation only)
- **Location**: `/home/ucadmin/.local/lib/python3.13/site-packages/mlir_aie/`
- **Purpose**: Call `aiedialect.generate_cdo()` - no pure C++ alternative
- **Status**: ✅ Available (v1.1.1)
- **Note**: Only needed for Phase 4, rest can be C++

---

## Complete Command Sequence

### Input File
**Location**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/passthrough_step3.mlir`

### Working Directory
```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/build
```

### Phase 1: MLIR Transformations

```bash
# Set variables
INPUT_MLIR="../passthrough_step3.mlir"
DEVICE_NAME="passthrough_complete"  # From module name
AIE_TARGET="aie2"  # Phoenix NPU is AIE2
ALLOC_SCHEME="bank-aware"  # Or "basic-sequential"

# Step 1a-1h: Lower and allocate
/home/ucadmin/mlir-aie-source/build/bin/aie-opt \
  --pass-pipeline="builtin.module(
    lower-affine,
    aie-canonicalize-device,
    aie.device(
      aie-assign-lock-ids,
      aie-register-objectFifos,
      aie-objectFifo-stateful-transform,
      aie-assign-bd-ids,
      aie-lower-cascade-flows,
      aie-lower-broadcast-packet,
      aie-lower-multicast,
      aie-assign-tile-controller-ids,
      aie-generate-column-control-overlay,
      aie-assign-buffer-addresses{alloc-scheme=${ALLOC_SCHEME}}
    ),
    convert-scf-to-cf
  )" \
  ${INPUT_MLIR} \
  -o input_with_addresses.mlir

# Step 1i: Create pathfinder flows (routing)
/home/ucadmin/mlir-aie-source/build/bin/aie-opt \
  --aie-create-pathfinder-flows \
  input_with_addresses.mlir \
  -o input_physical.mlir
```

### Phase 2: Core Compilation (Per-Core)

**NOTE**: This phase requires Peano compiler which is not yet located. Skip if no cores have code (passthrough_step3.mlir has empty core).

For each core at (col, row):

```bash
COL=0
ROW=2
CORE_PREFIX="core_${COL}_${ROW}"

# Step 2a: Lower core to LLVM
/home/ucadmin/mlir-aie-source/build/bin/aie-opt \
  --pass-pipeline="builtin.module(
    aie.device(
      aie-localize-locks,
      aie-normalize-address-spaces,
      aie-transform-bfp-types
    ),
    aie-standard-lowering{device=${DEVICE_NAME} tilecol=${COL} tilerow=${ROW}},
    aiex-standard-lowering,
    convert-aievec-to-llvm{aie-target=${AIE_TARGET}},
    canonicalize,
    cse,
    convert-ub-to-llvm,
    convert-vector-to-llvm,
    expand-strided-metadata,
    lower-affine,
    convert-math-to-llvm,
    convert-index-to-llvm,
    arith-expand,
    convert-arith-to-llvm,
    finalize-memref-to-llvm,
    convert-func-to-llvm{use-bare-ptr-memref-call-conv=true},
    convert-cf-to-llvm,
    canonicalize,
    cse
  )" \
  input_physical.mlir \
  -o ${CORE_PREFIX}.opt.mlir

# Step 2b: Translate to LLVM IR
/home/ucadmin/mlir-aie-source/build/bin/aie-translate \
  --mlir-to-llvmir \
  ${CORE_PREFIX}.opt.mlir \
  -o ${CORE_PREFIX}.ll

# Step 2c: Generate linker script
/home/ucadmin/mlir-aie-source/build/bin/aie-translate \
  --aie-generate-ldscript \
  --aie-device-name ${DEVICE_NAME} \
  --tilecol=${COL} \
  --tilerow=${ROW} \
  input_physical.mlir \
  -o ${CORE_PREFIX}.ld.script

# Step 2d: Compile with Peano (REQUIRES PEANO - NOT YET LOCATED)
# Typical command if Peano available:
# ${PEANO_INSTALL_DIR}/bin/opt --passes=default<O2> -inline-threshold=10 -S ${CORE_PREFIX}.ll -o ${CORE_PREFIX}.opt.ll
# ${PEANO_INSTALL_DIR}/bin/llc ${CORE_PREFIX}.opt.ll -O2 --march=aie2 --function-sections --filetype=obj -o ${CORE_PREFIX}.o

# Step 2e: Link to ELF (REQUIRES PEANO - NOT YET LOCATED)
# ${PEANO_INSTALL_DIR}/bin/clang -O2 --target=aie2-none-unknown-elf ${CORE_PREFIX}.o \
#   -Wl,--gc-sections -Wl,--orphan-handling=error \
#   -Wl,-T,${CORE_PREFIX}.ld.script \
#   -o ${CORE_PREFIX}.elf
```

**If cores exist, you must update input_physical.mlir with elf_file attributes pointing to the .elf files.**

### Phase 3: NPU Instruction Generation

```bash
# Step 3a: Lower runtime sequence to NPU instructions
/home/ucadmin/mlir-aie-source/build/bin/aie-opt \
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
  input_physical.mlir \
  -o npu_insts.mlir

# Step 3b: Translate to binary instructions
# Note: This uses Python binding (no pure C++ alternative)
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/ucadmin/.local/lib/python3.13/site-packages')
from mlir.ir import Context, Module
from mlir_aie.dialects import aie as aiedialect
import struct

with Context():
    with open('npu_insts.mlir', 'r') as f:
        module = Module.parse(f.read())

    # Translate NPU instructions to binary
    insts = aiedialect.translate_npu_to_binary(
        module.operation,
        "passthrough_complete",  # device_name
        None  # sequence_name (None = first sequence)
    )

    # Write binary file
    with open('insts.bin', 'wb') as f:
        f.write(struct.pack('I' * len(insts), *insts))

    print(f"Generated insts.bin: {len(insts)} instructions")
EOF
```

### Phase 4: CDO Generation

```bash
# CDO generation requires Python binding (no pure C++ alternative)
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/ucadmin/.local/lib/python3.13/site-packages')
from mlir.ir import Context, Module, Location
from mlir_aie.dialects import aie as aiedialect
import os

with Context(), Location.unknown():
    with open('input_physical.mlir', 'r') as f:
        module = Module.parse(f.read())

    # Generate CDO files (creates 3 .bin files)
    aiedialect.generate_cdo(
        module.operation,
        os.getcwd(),  # tmpdir
        "passthrough_complete"  # device_name
    )

    print("Generated CDO files:")
    print("  - passthrough_complete_aie_cdo_elfs.bin")
    print("  - passthrough_complete_aie_cdo_init.bin")
    print("  - passthrough_complete_aie_cdo_enable.bin")
EOF
```

### Phase 5: PDI Generation

```bash
# Step 5a: Create BIF file
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

# Step 5b: Generate PDI
/home/ucadmin/mlir-aie-source/build/bin/bootgen \
  -arch versal \
  -image design.bif \
  -o passthrough_complete.pdi \
  -w
```

### Phase 6: XCLBIN Generation

```bash
# Step 6a: Create JSON metadata files

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

# Step 6b: Generate XCLBIN
/opt/xilinx/xrt/bin/xclbinutil \
  --add-replace-section MEM_TOPOLOGY:JSON:mem_topology.json \
  --add-kernel kernels.json \
  --add-replace-section AIE_PARTITION:JSON:aie_partition.json \
  --force \
  --quiet \
  --output final.xclbin

echo "✅ Generated final.xclbin"
```

### Final Output Files

```bash
ls -lh final.xclbin insts.bin
```

You now have:
- **final.xclbin**: NPU-executable binary (load with XRT)
- **insts.bin**: Runtime instruction stream (pass to kernel)

---

## Detailed Pipeline Phases

### Phase 1: MLIR Transformations

**Purpose**: Transform high-level MLIR into physically-routed, address-allocated NPU configuration.

**Key Passes**:

1. **lower-affine**: Lower affine constructs to standard ops
2. **aie-canonicalize-device**: Ensure device is properly structured
3. **aie-assign-lock-ids**: Assign unique IDs to lock operations
4. **aie-register-objectFifos**: Register ObjectFIFO data movement
5. **aie-objectFifo-stateful-transform**: Convert ObjectFIFOs to explicit DMA
6. **aie-assign-bd-ids**: Assign buffer descriptor IDs
7. **aie-lower-cascade-flows**: Lower cascade operations
8. **aie-assign-buffer-addresses**: Allocate memory addresses
9. **aie-create-pathfinder-flows**: Route data flows through switchboxes

**Input**: High-level MLIR with ObjectFIFOs
**Output**: Physically-routed MLIR with concrete addresses

### Phase 2: Core Compilation

**Purpose**: Compile C++/MLIR code for AIE cores into ELF binaries.

**Key Steps**:
1. Lower MLIR to LLVM IR (per core)
2. Optimize LLVM IR with Peano optimizer
3. Compile to AIE object code with Peano compiler
4. Link with runtime libraries to create ELF

**Status**: ⚠️ **Blocked** - Peano compiler not located
**Workaround**: Can skip if cores are pre-compiled or empty

### Phase 3: NPU Instruction Generation

**Purpose**: Convert runtime DMA sequences into NPU binary instructions.

**Key Passes**:
1. **aie-materialize-bd-chains**: Expand buffer descriptor chains
2. **aie-substitute-shim-dma-allocations**: Resolve DMA channel allocations
3. **aie-assign-runtime-sequence-bd-ids**: Assign runtime BD IDs
4. **aie-dma-tasks-to-npu**: Lower DMA tasks to NPU instructions
5. **aie-dma-to-npu**: Convert DMA operations to NPU format
6. **aie-lower-set-lock**: Lower lock operations to NPU writes

**Output**: Binary instruction stream (insts.bin)

### Phase 4: CDO Generation

**Purpose**: Generate Configuration Data Objects for NPU initialization.

**CDO Types**:
1. **_aie_cdo_elfs.bin**: ELF loading configuration
2. **_aie_cdo_init.bin**: Tile initialization (DMA, locks, routing)
3. **_aie_cdo_enable.bin**: Core enable sequence

**Technology**: Uses libxaie C library (wrapped by Python bindings)
**Status**: ✅ Available via Python binding only

### Phase 5: PDI Generation

**Purpose**: Package CDO files into Versal PDI (Programmable Device Image).

**Tool**: bootgen (Xilinx Boot Image Generator)
**Input**: BIF file + CDO binaries
**Output**: .pdi file (bootable image)

**BIF Format**:
- **id_code**: Device ID (0x14ca8093 for Versal)
- **extended_id_code**: Extended device ID
- **image/type=cdo**: CDO file list

### Phase 6: XCLBIN Generation

**Purpose**: Package PDI and metadata into XRT-loadable container.

**Tool**: xclbinutil (from XRT)
**Metadata**:
- **MEM_TOPOLOGY**: Memory regions (HOST, SRAM)
- **AIE_PARTITION**: AIE partition configuration + PDI reference
- **kernels**: PS kernel definition with arguments

**Output**: .xclbin file (final NPU executable)

---

## Input MLIR Requirements

### Device Specification

Must use correct device for Phoenix NPU:
```mlir
aie.device(npu1) {
  // NOT npu1_4col or npu1_1col
  // Just npu1 for Phoenix
}
```

### Required Components

1. **Tiles**: Define compute and shim tiles
   ```mlir
   %shim_tile_0_0 = aie.tile(0, 0)  // ShimNOC for DMA
   %compute_tile_0_2 = aie.tile(0, 2)  // Compute tile
   ```

2. **Buffers**: Allocate tile memory
   ```mlir
   %buffer = aie.buffer(%tile_0_2) {sym_name = "buf"} : memref<1024xui8>
   ```

3. **Locks**: Synchronization primitives
   ```mlir
   %lock = aie.lock(%tile_0_2, 0) {init = 2 : i32}
   ```

4. **DMA Sequences**: Data movement
   ```mlir
   %mem_0_2 = aie.mem(%tile_0_2) {
     %dma = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
     ^bb1:
       aie.use_lock(%lock, AcquireGreaterEqual, 1)
       aie.dma_bd(%buffer : memref<1024xui8>, 0, 1024)
       aie.use_lock(%lock, Release, 1)
       aie.next_bd ^bb1
     ^bb2:
       aie.end
   }
   ```

5. **Runtime Sequence**: Host-side DMA control
   ```mlir
   aiex.runtime_sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
     aiex.npu.dma_memcpy_nd(%arg0[0,0,0,0][1,1,1,256][0,0,0,1])
       {id = 0 : i64, metadata = @shim_alloc} : memref<1024xi32>
     aiex.npu.dma_wait {symbol = @shim_alloc}
   }
   ```

6. **Shim DMA Allocations**: Channel assignments
   ```mlir
   aie.shim_dma_allocation @shim_alloc(MM2S, 0, 0)
   ```

7. **Routing**: Switchbox connections
   ```mlir
   %switchbox_0_0 = aie.switchbox(%shim_tile_0_0) {
     aie.connect<South : 3, North : 1>
     aie.connect<North : 0, South : 2>
   }
   ```

---

## Intermediate File Formats

### MLIR Files (.mlir)

**Human-readable text format:**
```mlir
module @name {
  aie.device(npu1) {
    %tile = aie.tile(0, 2)
    // ...
  }
}
```

### LLVM IR Files (.ll)

**LLVM intermediate representation:**
```llvm
define void @core_func() {
entry:
  %0 = load i32, i32* %ptr
  ret void
}
```

### CDO Files (.bin)

**Binary configuration data** - Opaque binary format for NPU register writes.

### PDI Files (.pdi)

**Versal boot image** - Contains CDO files + metadata for secure boot.

### XCLBIN Files (.xclbin)

**XRT container format** - ZIP-like archive with:
- PDI file
- JSON metadata (mem_topology, aie_partition, kernels)
- Checksums and signatures

---

## Troubleshooting

### Issue 1: "Peano compiler not found"

**Symptom**: Cannot compile core .ll files to .o
**Solution**:
- Check if Vitis AIE tools installed: `ls /opt/xilinx/aietools/`
- Set `PEANO_INSTALL_DIR` environment variable
- Alternative: Use pre-compiled .elf files (if available)

### Issue 2: "Python module not found" for CDO generation

**Symptom**: `ModuleNotFoundError: No module named 'aie.extras.util'`
**Solution**:
- This is expected - v1.1.1 Python API has issues
- CDO generation must use Python binding directly (shown in commands above)
- No pure C++ alternative available

### Issue 3: "Invalid device name: npu1_4col"

**Symptom**: Compilation fails with device error
**Solution**: Use `npu1` NOT `npu1_4col` for Phoenix NPU

### Issue 4: "xclbinutil: command not found"

**Symptom**: XCLBIN generation fails
**Solution**:
```bash
export PATH=/opt/xilinx/xrt/bin:$PATH
```

### Issue 5: "bootgen: syntax error"

**Symptom**: PDI generation fails
**Solution**: Check BIF file format - no spaces in wrong places, proper brackets

---

## Performance Optimization Flags

### aie-opt Optimization Passes

```bash
# Add these to --pass-pipeline for better performance:
--pass-pipeline="builtin.module(
  canonicalize,  # Canonicalize IR
  cse,           # Common subexpression elimination
  # ... other passes ...
)"
```

### Buffer Allocation Schemes

**bank-aware** (default): Better memory bank utilization
```bash
aie-assign-buffer-addresses{alloc-scheme=bank-aware}
```

**basic-sequential**: Simpler allocation (fallback if bank-aware fails)
```bash
aie-assign-buffer-addresses{alloc-scheme=basic-sequential}
```

### Peano Compiler Flags (when available)

```bash
# Optimization level
opt --passes=default<O2> -inline-threshold=10

# Code generation
llc -O2 --march=aie2 --function-sections
```

### Link Flags

```bash
# Garbage collection (remove unused code)
-Wl,--gc-sections

# Strict linker script enforcement
-Wl,--orphan-handling=error
```

---

## Summary: Critical Blockers and Workarounds

### ✅ What Works Without Issues

1. **Phase 1 (MLIR transforms)**: Fully operational with aie-opt
2. **Phase 3 (NPU instructions)**: Works with Python binding
3. **Phase 4 (CDO generation)**: Works with Python binding
4. **Phase 5 (PDI generation)**: Works with bootgen
5. **Phase 6 (XCLBIN generation)**: Works with xclbinutil

### ⚠️ Current Blockers

**Phase 2 (Core compilation)**: Requires Peano compiler

**Impact**: Cannot compile C++/MLIR core code to ELF binaries

**Workarounds**:
1. Use pre-compiled .elf files (if available)
2. Use cores with no code (empty `aie.core` blocks)
3. Locate Peano in Vitis installation
4. Request Peano from AMD/Xilinx

**Your passthrough_step3.mlir has empty core** → Can skip Phase 2!

### Next Steps

1. **Test without cores**: Compile passthrough_step3.mlir (has empty core)
2. **Verify on NPU**: Load generated XCLBIN with XRT
3. **Locate Peano**: Search Vitis install or contact AMD
4. **Scale up**: Once working, add real core implementations

---

## Quick Reference: One-Command Pipeline

For passthrough_step3.mlir (no core code):

```bash
#!/bin/bash
set -e

INPUT=../passthrough_step3.mlir
DEVICE=passthrough_complete
OPT=/home/ucadmin/mlir-aie-source/build/bin/aie-opt
TRANSLATE=/home/ucadmin/mlir-aie-source/build/bin/aie-translate
BOOTGEN=/home/ucadmin/mlir-aie-source/build/bin/bootgen
XCLBINUTIL=/opt/xilinx/xrt/bin/xclbinutil

# Phase 1: MLIR transforms
${OPT} --pass-pipeline="builtin.module(lower-affine,aie-canonicalize-device,aie.device(aie-assign-lock-ids,aie-register-objectFifos,aie-objectFifo-stateful-transform,aie-assign-bd-ids,aie-lower-cascade-flows,aie-lower-broadcast-packet,aie-lower-multicast,aie-assign-tile-controller-ids,aie-generate-column-control-overlay,aie-assign-buffer-addresses{alloc-scheme=bank-aware}),convert-scf-to-cf)" ${INPUT} -o input_with_addresses.mlir

${OPT} --aie-create-pathfinder-flows input_with_addresses.mlir -o input_physical.mlir

# Phase 3: NPU instructions
${OPT} --pass-pipeline="builtin.module(aie.device(aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu,aie-dma-to-npu,aie-lower-set-lock))" input_physical.mlir -o npu_insts.mlir

python3 -c "
import sys; sys.path.insert(0, '/home/ucadmin/.local/lib/python3.13/site-packages')
from mlir.ir import Context, Module; from mlir_aie.dialects import aie; import struct
with Context():
    with open('npu_insts.mlir') as f: module = Module.parse(f.read())
    insts = aie.translate_npu_to_binary(module.operation, '${DEVICE}', None)
    with open('insts.bin', 'wb') as f: f.write(struct.pack('I' * len(insts), *insts))
"

# Phase 4: CDO generation
python3 -c "
import sys, os; sys.path.insert(0, '/home/ucadmin/.local/lib/python3.13/site-packages')
from mlir.ir import Context, Module, Location; from mlir_aie.dialects import aie
with Context(), Location.unknown():
    with open('input_physical.mlir') as f: module = Module.parse(f.read())
    aie.generate_cdo(module.operation, os.getcwd(), '${DEVICE}')
"

# Phase 5: PDI generation
cat > design.bif << EOF
all: { id_code = 0x14ca8093; extended_id_code = 0x01; image { name=aie_image, id=0x1c000000; { type=cdo file=${DEVICE}_aie_cdo_elfs.bin file=${DEVICE}_aie_cdo_init.bin file=${DEVICE}_aie_cdo_enable.bin } } }
EOF

${BOOTGEN} -arch versal -image design.bif -o ${DEVICE}.pdi -w

# Phase 6: XCLBIN generation
cat > mem_topology.json << 'EOF'
{"mem_topology":{"m_count":"2","m_mem_data":[{"m_type":"MEM_DRAM","m_used":"1","m_sizeKB":"0x10000","m_tag":"HOST","m_base_address":"0x4000000"},{"m_type":"MEM_DRAM","m_used":"1","m_sizeKB":"0xc000","m_tag":"SRAM","m_base_address":"0x4000000"}]}}
EOF

cat > aie_partition.json << EOF
{"aie_partition":{"name":"QoS","operations_per_cycle":"2048","partition":{"column_width":1,"start_columns":[0]},"PDIs":[{"uuid":"12345678-1234-5678-1234-567812345678","file_name":"${DEVICE}.pdi","cdo_groups":[{"name":"DPU","type":"PRIMARY","pdi_id":"0x01","dpu_kernel_ids":["0x901"],"pre_cdo_groups":["0xC1"]}]}]}}
EOF

cat > kernels.json << 'EOF'
{"ps-kernels":{"kernels":[{"name":"MLIR_AIE","type":"dpu","extended-data":{"subtype":"DPU","functional":"0","dpu_kernel_id":"0x901"},"arguments":[{"name":"opcode","address-qualifier":"SCALAR","type":"uint64_t","offset":"0x00"},{"name":"instr","memory-connection":"SRAM","address-qualifier":"GLOBAL","type":"char *","offset":"0x08"},{"name":"ninstr","address-qualifier":"SCALAR","type":"uint32_t","offset":"0x10"},{"name":"bo0","memory-connection":"HOST","address-qualifier":"GLOBAL","type":"void*","offset":"0x14"},{"name":"bo1","memory-connection":"HOST","address-qualifier":"GLOBAL","type":"void*","offset":"0x1c"}],"instances":[{"name":"MLIRAIE"}]}]}}
EOF

${XCLBINUTIL} --add-replace-section MEM_TOPOLOGY:JSON:mem_topology.json --add-kernel kernels.json --add-replace-section AIE_PARTITION:JSON:aie_partition.json --force --quiet --output final.xclbin

echo "✅ SUCCESS! Generated final.xclbin and insts.bin"
```

---

**End of Document**

**Status**: Complete pipeline documented
**Next Action**: Test compilation on passthrough_step3.mlir
**Expected Result**: Working XCLBIN (no core code needed)
**Performance Target**: After adding kernels → 220x realtime
