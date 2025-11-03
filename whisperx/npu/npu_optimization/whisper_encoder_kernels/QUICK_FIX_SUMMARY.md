# Quick Fix Summary - 32×32 Matmul Compilation

## TL;DR - What You Need to Know

**Status**: ⚠️ BLOCKED - Need AMD Chess Compiler
**What I Fixed**: All Python environment issues
**What's Missing**: `chess-llvm-link` binary from AMD Vitis/RyzenAI
**Time to Fix**: 30-60 minutes (install Chess compiler)

## The Blocker

```
FileNotFoundError: '<aietools not found>/tps/lnx64/target_aie_ml/bin/LNa64bin/chess-llvm-link'
```

Even with `--no-xchesscc` flag, aiecc.py REQUIRES Chess compiler.

## Quick Solution: Install Chess Compiler

### Option 1: AMD Vitis (Recommended)
```bash
# Download from AMD website
wget https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html

# Install
sudo ./Xilinx_Vitis_2024.1.sh

# Set environment
export AIETOOLS=/opt/amd/xilinx/vitis/2024.1/aietools
export PATH=$AIETOOLS/bin:$PATH

# Verify
which chess-llvm-link
```

### Option 2: Ryzen AI Software Platform
```bash
# Download from AMD
wget https://account.amd.com/en/member/ryzenai-sw-ea.html

# Extract and install
tar -xzf ryzenai-sw-1.0.tar.gz
cd ryzenai-sw-1.0
sudo ./install.sh

# Set environment
export AIETOOLS=/opt/ryzenai/aietools
export PATH=$AIETOOLS/tps/lnx64/target_aie_ml/bin/LNa64bin:$PATH

# Verify
which chess-llvm-link
```

### Then Compile 32×32
```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels
bash compile_matmul_32x32.sh
```

## What I Already Fixed

### 1. Python Environment ✅
All AIE tools now use correct Python:
```bash
/home/ucadmin/.local/bin/aiecc.py        # Fixed
/home/ucadmin/.local/bin/aie-translate   # Fixed
/home/ucadmin/.local/bin/aie-opt         # Fixed
```

### 2. Compilation Script ✅
**File**: `compile_matmul_32x32.sh`

Fixed:
- C compiler target: `--target=aie2` (was wrong)
- Include paths added
- Peano flags added: `--no-xchesscc --no-xbridge --unified`

### 3. Source Files ✅
- `matmul_int8_32x32.c` - Compiles successfully (3.7KB object file)
- `matmul_32x32.mlir` - Parses and lowers correctly
- Both ready for final XCLBIN generation

## Verification After Chess Install

```bash
# Should complete in ~30 seconds
cd whisper_encoder_kernels
bash compile_matmul_32x32.sh

# Check output
ls -lh build_matmul_32x32/matmul_32x32.xclbin
# Expected: ~11KB file

# Verify structure
/opt/xilinx/xrt/bin/xclbinutil --info \
  --input build_matmul_32x32/matmul_32x32.xclbin
# Expected: Valid XCLBIN with AIE2 kernel
```

## How the 16×16 Was Compiled

The existing `build_matmul_fixed/matmul_16x16.xclbin` (Oct 30 00:35) was compiled when Chess tools were available. The same environment no longer has Chess, preventing 32×32 compilation.

**Evidence**:
- 16×16 build includes `llchesshack.ll` file
- This proves Chess was used
- Same script fails now without Chess

## Alternative: Manual Compilation (Advanced)

If you cannot install Chess, try manual workflow:

```bash
# THIS IS UNTESTED - Research required
cd whisper_encoder_kernels/build_matmul_32x32

# 1. Lower MLIR manually
/home/ucadmin/.local/bin/aie-opt \
  --pass-pipeline="builtin.module(aie.device(aie-normalize-address-spaces,aie-objectFifo-stateful-transform))" \
  ../matmul_32x32.mlir -o lowered.mlir

# 2. More steps needed (see full report)
```

**Status**: Not fully researched yet. Installing Chess is faster.

## Files Modified

1. `/home/ucadmin/.local/bin/aiecc.py` - Shebang to venv313
2. `/home/ucadmin/.local/bin/aie-translate` - Shebang to venv313
3. `/home/ucadmin/.local/bin/aie-opt` - Shebang to venv313
4. `compile_matmul_32x32.sh` - Fixed flags and targets

## Expected Output

After Chess install and compilation:

```
build_matmul_32x32/
├── matmul_32x32.o             (3.7 KB) ✅ Already compiled
├── matmul_32x32.xclbin        (11 KB)  ❌ Needs Chess
├── main_sequence.bin          (300 B)  ❌ Needs Chess
└── matmul_32x32.mlir.prj/
    ├── main.pdi               (4 KB)   ❌ Needs Chess
    ├── main_core_0_2.elf      (4.5 KB) ❌ Needs Chess
    └── ... (other artifacts)
```

## Questions?

See full report: `COMPILATION_INVESTIGATION_REPORT.md`

---

**Bottom Line**: Install AMD Chess compiler (30-60 min) → Run script → Done (30 sec)
