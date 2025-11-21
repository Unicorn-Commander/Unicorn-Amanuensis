# AIETOOLS AND MLIR-AIE COMPILATION ENVIRONMENT REPORT

**Date**: November 20, 2025
**Working Directory**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels`
**System**: AMD Phoenix NPU (XDNA1) with Linux 6.14.0-34-generic

---

## EXECUTIVE SUMMARY

The MLIR-AIE compilation environment is **fully operational** with:
- ✅ **aietools installed**: Peano compiler at `/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie/`
- ✅ **aiecc.py available**: Python compilation orchestrator found in multiple locations
- ✅ **Working XCLBIN examples**: Mel, matmul, gelu, softmax, attention kernels compiled
- ✅ **Complete build scripts**: Shell scripts for all major kernel types

---

## 1. AIETOOLS INSTALLATION PATHS

### Primary Installation (RECOMMENDED)
```
/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie/
├── bin/
│   ├── clang                (203K)  - AIE2 C/C++ compiler
│   ├── clang++              (203K)
│   ├── clang-19             (203K)
│   ├── llvm-ar              (131K)  - Object file archiver
│   ├── llvm-nm              (40K)   - Symbol viewer
│   ├── llvm-objdump         (1.3M)  - Object disassembler
│   ├── llvm-readelf         (2.5M)
│   ├── llvm-readobj         (2.5M)
│   ├── llc                  (271K)  - LLVM compiler
│   ├── ld.lld               (8.2M)  - Linker
│   ├── lld-link             (8.2M)
│   ├── ld64.lld             (8.2M)
│   ├── llvm-size            (109K)
│   ├── llvm-cxxfilt         (40K)
│   └── llvm-dwarfdump       (288K)
├── include/                 - AIE2 kernel headers
└── lib/                     - AIE2 runtime libraries
```

### Alternative Installations
- `/home/ucadmin/mlir-aie-source/install/bin/` - Source build installation
- `/home/ucadmin/mlir-aie-source/build/bin/` - Build directory

### Available aiecc.py Locations
```
1. /home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py
2. /home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/mlir_aie/bin/aiecc.py
3. /home/ucadmin/mlir-aie-source/build/bin/aiecc.py
4. /home/ucadmin/mlir-aie-source/install/bin/aiecc.py
5. /home/ucadmin/mlir-aie-prebuilt/mlir_aie/bin/aiecc.py
```

**Recommended**: Use venv313 version - most recent and most tested.

---

## 2. ENVIRONMENT VARIABLES FOR MLIR COMPILATION

### Standard Configuration (Used in All Working Scripts)

```bash
# Peano Compiler Location
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie

# Python Path for AIE Libraries
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH

# Updated PATH for compilation tools
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH
```

### Explanation
1. **PEANO_INSTALL_DIR**: Points to the Peano AIE2 LLVM compiler tools
2. **PYTHONPATH**: Enables Python imports for MLIR-AIE library (used by aiecc.py)
3. **PATH**: Ensures aiecc.py and clang are found in shell

### Optional But Useful
```bash
# For XRT operations
export XRT_INI_PATH=/etc/xrt.ini

# For MLIR debugging (verbose output)
export MLIR_ENABLE_DIAGNOSTICS=1
```

---

## 3. COMPILATION WORKFLOW - HOW EXISTING XCLBINS WERE COMPILED

### Standard 5-Step Process (Used for All Kernels)

All compiled XCLBINs (mel, matmul, gelu, softmax, attention, layernorm) follow this pattern:

#### Step 1: Compile C/C++ Kernel to Object File
```bash
$PEANO_INSTALL_DIR/bin/clang \
    -O2 \
    -std=c11 \
    --target=aie2-none-unknown-elf \
    -c kernel_source.c \
    -o build_dir/kernel_source.o
```

**What it does**:
- Compiles C kernel code to AIE2 ELF object file
- Optimized with `-O2`
- Targets AIE2 instruction set (`aie2-none-unknown-elf`)

**Examples that worked**:
- `gelu_int8.c` → `gelu_int8.o`
- `layernorm_int8.c` → `layernorm_int8.o`
- `attention_int8_64x64.c` → `attention_int8_64x64.o`
- `softmax_bf16_xdna1.cc` → `softmax_bf16_xdna1.o`

#### Step 2: Create Object Archive
```bash
$PEANO_INSTALL_DIR/bin/llvm-ar rcs \
    build_dir/kernel_combined.o \
    build_dir/kernel_source.o
```

**What it does**:
- Creates an archive of object files
- `llvm-ar` is LLVM's archiver (equivalent to `ar`)
- `rcs` means: replace, create, and create symbol table

#### Step 3: Verify Symbols (Optional but Recommended)
```bash
$PEANO_INSTALL_DIR/bin/llvm-nm build_dir/kernel_combined.o | grep -E "kernel_name"
```

**What it does**:
- Lists symbols in the object file
- Verifies kernel function was compiled correctly
- Shows if external functions are present

#### Step 4: Prepare MLIR Files
```bash
cp kernel_wrapper.mlir build_dir/kernel_wrapper.mlir
```

**What it does**:
- Copies MLIR description to build directory
- MLIR file defines data flow and scheduling

#### Step 5: Generate XCLBIN (Executable Binary)
```bash
cd build_dir

/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=kernel_final.xclbin \
    --npu-insts-name=insts.bin \
    kernel_wrapper.mlir
```

**What it does**:
- Lowers MLIR to AIE2 hardware configuration
- Generates XCLBIN (Xilinx Compiled Binary)
- Generates NPU instruction sequences (insts.bin)
- Options:
  - `--alloc-scheme=basic-sequential`: Simple memory allocation
  - `--aie-generate-xclbin`: Create executable binary
  - `--aie-generate-npu-insts`: Create instruction sequences
  - `--no-compile-host`: Skip host C++ compilation
  - `--no-xchesscc`: Skip Chess compiler (not available/needed)
  - `--no-xbridge`: Skip bridge generation
  - `--xclbin-name`: Output binary filename
  - `--npu-insts-name`: Output instruction filename

---

## 4. WORKING XCLBIN EXAMPLES AND THEIR BUILD PATTERNS

### Example 1: GELU Activation Kernel

**Location**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/`

**Build Script**: `compile_gelu.sh`

**Source Files**:
- `gelu_int8.c` - Kernel implementation (LUT-based GELU)
- `gelu_simple.mlir` - MLIR wrapper for 512 elements
- `gelu_2048.mlir` - MLIR wrapper for 2048 elements (FFN)

**Compilation Command**:
```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels
bash compile_gelu.sh
```

**Generated Files**:
```
build_gelu/
├── gelu_int8.o
├── gelu_combined.o
├── gelu_simple.mlir
├── gelu_2048.mlir
├── gelu_simple.xclbin        (512-element version)
├── insts_512.bin
├── gelu_2048.xclbin          (2048-element version)
└── insts_2048.bin
```

**Performance**: <0.5ms for 512 elements, <1.3ms for 2048 elements

---

### Example 2: Matrix Multiplication Kernel

**Location**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/`

**Build Script**: `compile_matmul_32x32.sh` (most recent working version)

**Source Files**:
- `matmul_int8.c` - 32x32 INT8 matrix multiply
- `matmul_simple.mlir` - MLIR wrapper

**Compilation Command**:
```bash
bash compile_matmul_32x32.sh
```

**Generated Files**:
```
build_matmul_32x32/
├── matmul_int8.o
├── matmul_combined.o
├── matmul_simple.mlir
└── matmul_simple.xclbin
```

**MLIR Wrapper Pattern**:
```mlir
module @matmul_npu {
    aie.device(npu1) {
        func.func private @matmul_int8_16x16(memref<256xi8>, memref<256xi8>, memref<256xi8>)
        
        %tile00 = aie.tile(0, 0)   // ShimNOC (DMA)
        %tile02 = aie.tile(0, 2)   // Compute core
        
        aie.objectfifo @of_matA(%tile00, {%tile02}, 2 : i32)
        aie.objectfifo @of_matB(%tile00, {%tile02}, 2 : i32)
        aie.objectfifo @of_matC(%tile02, {%tile00}, 2 : i32)
        
        %core02 = aie.core(%tile02) {
            // Infinite loop with DMA-triggered execution
            scf.for %iter = ... {
                // Acquire buffers
                // Call kernel
                // Release buffers
            }
        } {link_with="matmul_combined.o"}
        
        aiex.runtime_sequence(...) {
            // DMA transfers and synchronization
        }
    }
}
```

---

### Example 3: Attention Mechanism Kernel

**Location**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/`

**Build Script**: `compile_attention_64x64.sh`

**Source Files**:
- `attention_int8_64x64_tiled.c` - 64x64 INT8 attention with tiling
- `attention_64x64.mlir` - MLIR wrapper

**Compilation Command**:
```bash
bash compile_attention_64x64.sh
```

**Key Features**:
- 64x64 matrix operations (4096 bytes per Q/K/V)
- INT32 accumulator (fits in AIE2 32KB SRAM)
- 8-10ms per tile execution
- Tiled approach for larger sequences

---

### Example 4: Softmax Activation Kernel (XDNA1)

**Location**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1/`

**Build Script**: `compile_softmax_bf16.sh`

**Source Files**:
- `softmax_bf16_xdna1.cc` - C++ BF16 softmax implementation
- `softmax_bf16.mlir` - MLIR wrapper

**Compilation Command**:
```bash
bash kernels_xdna1/compile_softmax_bf16.sh
```

**Key Differences from INT8**:
- Uses BF16 (bfloat16) floating point
- Includes AIE2 API headers:
  - `/home/ucadmin/mlir-aie-source/third_party/aie_api/include`
  - `/home/ucadmin/mlir-aie-source/aie_runtime_lib/AIE2`
- More numerically stable for activation functions
- Supports C++20 features

---

### Example 5: Layer Normalization Kernel

**Location**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/`

**Build Script**: `compile_layernorm.sh`

**Source Files**:
- `layernorm_int8.c` - INT8 layer normalization with RMSNorm support
- `layernorm_simple.mlir` - MLIR wrapper for 256-dimensional normalization

**Purpose**: Normalize 768-byte input to 256-byte output

---

## 5. COMPILATION SCRIPTS LOCATION AND PATTERNS

### Available Compilation Scripts

**Main Directory** (`whisper_encoder_kernels/`):
```
compile_layernorm.sh
compile_gelu.sh
compile_matmul.sh
compile_matmul_simple.sh
compile_matmul_32x32.sh
compile_matmul_32x32_direct.sh
compile_matmul_32x32_working.sh
compile_matmul_32x32_final.sh
compile_matmul_64x64.sh
compile_attention.sh
compile_attention_64x64.sh
compile_attention_int32.sh
compile_attention_multicore.sh
compile_attention_iron.sh
compile_attention_iron_fixed.sh
compile_iron_fresh.sh
verify_toolchain.sh
```

**XDNA1 Kernel Directory** (`kernels_xdna1/`):
```
compile_softmax_bf16.sh
compile_softmax_batched.sh
compile_gelu.sh
compile_all_xdna1.sh
```

### Script Pattern for New Kernels

To create a new compilation script:

```bash
#!/bin/bash
set -e

# 1. Environment setup
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH

# 2. Verify tools
if [ ! -f "$PEANO_INSTALL_DIR/bin/clang" ]; then
    echo "ERROR: Peano clang not found"
    exit 1
fi

# 3. Create build directory
WORK_DIR=$(pwd)
BUILD_DIR=$WORK_DIR/build_mykernel
mkdir -p $BUILD_DIR
cd $WORK_DIR

# 4. Compile kernel
$PEANO_INSTALL_DIR/bin/clang \
    -O2 \
    -std=c11 \
    --target=aie2-none-unknown-elf \
    -c mykernel.c \
    -o $BUILD_DIR/mykernel.o

# 5. Create archive
$PEANO_INSTALL_DIR/bin/llvm-ar rcs \
    $BUILD_DIR/mykernel_combined.o \
    $BUILD_DIR/mykernel.o

# 6. Copy MLIR
cp mykernel.mlir $BUILD_DIR/mykernel.mlir

# 7. Generate XCLBIN
cd $BUILD_DIR
/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=mykernel.xclbin \
    --npu-insts-name=insts.bin \
    mykernel.mlir

cd $WORK_DIR
echo "✅ Compilation complete!"
```

---

## 6. MLIR WRAPPER PATTERNS - ANATOMY OF A WORKING KERNEL

### Minimal MLIR Wrapper Structure

All working MLIR files follow this pattern:

```mlir
//===- kernel_name.mlir ------------------------------------------------===//
module @kernel_npu {
    aie.device(npu1) {  // CRITICAL: Must specify npu1 for Phoenix NPU
        
        // 1. DECLARE KERNEL FUNCTION
        func.func private @kernel_function_c(memref<Nxi8>, memref<Mxi8>)
        
        // 2. DECLARE TILES
        %tile00 = aie.tile(0, 0)   // ShimNOC (DMA control)
        %tile02 = aie.tile(0, 2)   // Compute core
        
        // 3. DECLARE ObjectFIFOs (data movement)
        aie.objectfifo @of_input(%tile00, {%tile02}, 2 : i32)
        aie.objectfifo @of_output(%tile02, {%tile00}, 2 : i32)
        
        // 4. CORE LOGIC (infinite loop with DMA-triggered execution)
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index
            
            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire input
                %subviewIn = aie.objectfifo.acquire @of_input(Consume, 1)
                %elemIn = aie.objectfifo.subview.access %subviewIn[0]
                
                // Acquire output
                %subviewOut = aie.objectfifo.acquire @of_output(Produce, 1)
                %elemOut = aie.objectfifo.subview.access %subviewOut[0]
                
                // Call kernel
                func.call @kernel_function_c(%elemIn, %elemOut)
                
                // Release
                aie.objectfifo.release @of_input(Consume, 1)
                aie.objectfifo.release @of_output(Produce, 1)
            }
            aie.end
        } {link_with="kernel_combined.o"}  // Link compiled C code
        
        // 5. RUNTIME SEQUENCE (DMA transfers and synchronization)
        aiex.runtime_sequence(%input : memref<Nxi8>, %output : memref<Mxi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %cN_i64 = arith.constant N : i64
            
            // Host to NPU DMA
            aiex.npu.dma_memcpy_nd(%input[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %cN_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input,
                id = 1 : i64
            } : memref<Nxi8>
            
            // NPU to Host DMA
            aiex.npu.dma_memcpy_nd(%output[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                           [%c1_i64, %c1_i64, %c1_i64, %cM_i64]
                                           [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output,
                id = 0 : i64
            } : memref<Mxi8>
            
            // Wait for completion
            aiex.npu.dma_wait {symbol = @of_output}
        }
    }
}
```

### Key Components Explained

1. **`aie.device(npu1)`**: CRITICAL - specifies Phoenix NPU hardware
2. **Tiles**:
   - `(0, 0)`: ShimNOC - handles DMA to/from host memory
   - `(0, 2)`: Compute tile - executes kernel code
3. **ObjectFIFOs**: Modern MLIR-AIE data movement (replaces manual DMA)
4. **Core Logic**: Infinite loop triggered by DMA availability
5. **Runtime Sequence**: DMA setup and synchronization

---

## 7. WORKING XCLBIN FILES AND THEIR LOCATIONS

### Compiled XCLBIN Files Found

```
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/build/
├── final.xclbin                          (integrated pipeline)
├── final_passthrough.xclbin
├── final_passthrough_with_pdi.xclbin
└── core_0_2.elf

/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/
├── build_attention_64x64/
│   └── attention_64x64.xclbin
├── build_attention_int32/
│   └── attention_64x64.xclbin
├── build_gelu/
│   ├── gelu_simple.xclbin
│   └── gelu_2048.xclbin
├── build_layernorm/
│   └── layernorm_simple.xclbin
└── build_matmul_32x32/
    └── matmul_simple.xclbin

/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/
├── build/
│   ├── mel_simple.xclbin
│   ├── mel_test_final.xclbin
│   ├── mel_int8_final.xclbin
│   └── mel_fft.xclbin
├── build_fft/
│   └── mel_fft_final.xclbin
└── build_fixed_v3/
    ├── mel_fixed_v3_PRODUCTION_v1.0.xclbin
    └── mel_fixed_v3_PRODUCTION_v2.0.xclbin
```

### Testing Compiled Kernels

Each XCLBIN has a corresponding test script:

```bash
# Test GELU
cd whisper_encoder_kernels/build_gelu
python3 ../test_gelu.py

# Test MatMul
cd whisper_encoder_kernels/build_matmul_32x32
python3 ../test_matmul_32x32.py

# Test Attention
cd whisper_encoder_kernels/build_attention_64x64
python3 ../test_attention_64x64.py

# Test LayerNorm
cd whisper_encoder_kernels/build_layernorm
python3 ../test_layernorm.py

# Test Mel Spectrogram
cd mel_kernels/build_fixed_v3
python3 ../../test_mel.py
```

---

## 8. REQUIRED ENVIRONMENT VARIABLES FOR MLIR COMPILATION

### Complete Environment Setup Script

Create a script `setup_mlir_env.sh`:

```bash
#!/bin/bash

# Set PEANO compiler location
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie

# Set Python path for MLIR-AIE library
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH

# Add tools to PATH
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH

# Verify setup
echo "PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR"
echo "Checking tools..."
echo "  clang: $(which $PEANO_INSTALL_DIR/bin/clang)"
echo "  aiecc.py: $(which aiecc.py)"
echo "  llvm-ar: $(which llvm-ar)"
echo "✅ MLIR environment ready"
```

Usage:
```bash
source setup_mlir_env.sh
bash compile_gelu.sh
```

---

## 9. BEST PRACTICES AND RECOMMENDATIONS

### Compilation Workflow

1. **Setup environment first**
   ```bash
   export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
   export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH
   export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH
   ```

2. **Verify toolchain**
   ```bash
   bash verify_toolchain.sh
   ```

3. **For new kernels, copy working script**
   - Use `compile_gelu.sh` or `compile_matmul_32x32.sh` as template
   - Replace kernel filename, MLIR filename, and output names
   - Test with simple 16x16 or 512-element sizes first

4. **Common Compilation Options**
   ```
   --alloc-scheme=basic-sequential    # Simple memory allocation (recommended)
   --aie-generate-xclbin              # Generate executable binary
   --aie-generate-npu-insts           # Generate instruction sequences
   --no-compile-host                  # Skip host code (we don't need it)
   --no-xchesscc                      # Skip Chess compiler (not available)
   --no-xbridge                       # Skip bridge generation
   ```

5. **Optimization compiler flags**
   ```bash
   -O2              # Good balance of speed and compile time
   -O3              # Maximum optimization (slower compile)
   -std=c11         # C11 standard (for most kernels)
   -std=c++20       # C++20 for advanced features (softmax example)
   ```

---

## 10. SUMMARY TABLE

| Aspect | Location/Value |
|--------|---|
| **Peano Compiler** | `/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie/bin/clang` |
| **aiecc.py** | `/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py` |
| **LLVM Tools** | `/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie/bin/` |
| **Kernel Sources** | `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/` |
| **MLIR Files** | `*.mlir` in kernel directories |
| **Compiled XCLBINs** | `build_*/` subdirectories |
| **Build Scripts** | `compile_*.sh` in kernel directories |
| **Default Target Device** | `aie2-none-unknown-elf` (via clang) |
| **NPU Device Spec** | `aie.device(npu1)` in MLIR |
| **Compilation Time** | <1 second per XCLBIN (typical) |
| **XRT Version** | 2.20.0 (required for Phoenix NPU) |

---

## 11. QUICK START CHECKLIST

To compile a new kernel:

- [ ] Set environment variables (PEANO_INSTALL_DIR, PYTHONPATH, PATH)
- [ ] Verify tools exist: `which clang`, `which aiecc.py`, `which llvm-ar`
- [ ] Create C/C++ kernel source file (`kernel_name.c`)
- [ ] Create MLIR wrapper file (`kernel_name.mlir`) using pattern above
- [ ] Create build script (`compile_kernel_name.sh`) using template
- [ ] Run: `bash compile_kernel_name.sh`
- [ ] Check output: `build_kernel_name/*.xclbin` should exist
- [ ] Test with Python script using XRT

---

**Report Complete**
Generated: November 20, 2025
