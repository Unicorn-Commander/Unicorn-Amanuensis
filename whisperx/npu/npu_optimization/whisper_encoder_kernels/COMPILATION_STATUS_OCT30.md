# Whisper Encoder Kernel Compilation Status - October 30, 2025

## Executive Summary

**Goal**: Compile 32×32 matmul kernel for AMD Phoenix NPU to achieve 29-38x realtime performance (1.5-2× improvement from current 19.1×).

**Status**: **BLOCKED** on toolchain limitations

**What Works**: ✅ MLIR lowering with aie-opt, NPU binary generation
**What's Blocked**: ❌ XCLBIN packaging requires tools we don't have or can't configure

---

## What We Accomplished Today

### 1. Successfully Researched Solution Approaches ✅

- Reviewed `/home/ucadmin/NPU_SOLUTION_PACKAGE/SOLUTION_FOR_CHESS_COMPILER_ISSUE.md`
- Found that Meeting-Ops achieved 220× speedup with MLIR-AIE2
- Discovered their compilation approach also requires `v++` or working `aiecc.py`

### 2. Successfully Lowered MLIR Kernel ✅

**File**: `build_direct/matmul_lowered.mlir` (6.2 KB)

Used aie-opt with passes available in mlir-aie v1.1.1:
```bash
aie-opt \
    --aie-canonicalize-device \
    --aie-objectFifo-stateful-transform \
    --aie-create-pathfinder-flows \
    --aie-assign-buffer-addresses \
    matmul_32x32.mlir -o matmul_lowered.mlir
```

✅ **This worked!** MLIR successfully lowered without aiecc.py.

### 3. Generated NPU Binary ✅

**File**: `build_direct/matmul_npu.bin` (16 bytes)

```bash
aie-translate \
    --aie-npu-to-binary \
    --aie-output-binary \
    matmul_lowered.mlir -o matmul_npu.bin
```

✅ **This worked!** Generated NPU instruction sequence (DMA commands).

### 4. Compiled C Kernel ✅

**File**: `matmul_32x32.o` (3.7 KB)

```bash
$PEANO/clang \
    --target=aie2 \
    -I$PEANO/../aie_kernels/aie2/include \
    -c matmul_int8_32x32.c \
    -o matmul_32x32.o
```

✅ **This worked!** AIE2 object code compiled with Peano.

---

## What's Blocking Us ❌

### The XCLBIN Problem

To load and execute kernels on Phoenix NPU via XRT, we need a complete **XCLBIN file**. This requires:

1. **NPU instruction sequence** ✅ (we have this: matmul_npu.bin)
2. **AIE core ELF files** ⚠️ (we have .o file, need linking)
3. **Kernel metadata** ❌ (JSON config)
4. **XCLBIN packaging** ❌ (requires v++ or aiecc.py)

### Three Paths to XCLBIN (All Blocked)

#### Path A: Use aiecc.py ❌

**Issue**: Path detection failure

```
FileNotFoundError: '<aietools not found>/tps/lnx64/target_aie_ml/bin/LNa64bin/chess-llvm-link'
```

**What we tried**:
- ✅ Extracted vitis_aie_essentials with Chess compiler
- ✅ Set AIETOOLS_ROOT, XILINX_VITIS_AIETOOLS environment variables
- ✅ Added xchesscc to PATH (confirmed with `which xchesscc`)
- ✅ Set AIETOOLS before running Python
- ❌ aiecc.py's internal path detection (lines 1810-1843) still fails
- ❌ Even with --verbose flag, it hangs without output

**Root Cause**: aiecc.py runs its own path detection that looks for:
1. `ryzen_ai` Python package (we don't have this)
2. `xchesscc` in PATH using `shutil.which()` (we have this, but detection fails)

When detection fails, it sets `opts.aietools_path = "<aietools not found>"` and then tries to use that path.

#### Path B: Use v++ (Vitis Compiler) ❌

**Issue**: Not installed

The solution package (`aie2_kernel_driver.py`) shows:
```python
cmd = [
    "v++",
    "--platform", "xilinx_vck5000_gen4x8_qdma_2_202220_1",
    "--target", "hw",
    "--compile", "--optimize", "3",
    "--config", aie_config,
    "-o", str(self.xclbin_path)
]
```

**What we don't have**:
- ❌ Vitis HLS/v++ compiler (~100+ GB installation)
- ❌ Platform files for Phoenix NPU
- ❌ AMD SDK with proper device support

**Why v++**: Full Vitis toolchain includes Chess, Peano, and XCLBIN packaging in one tool.

#### Path C: Custom Runtime (Direct NPU Access) ❌

**Issue**: Too complex for current scope

Would require:
1. Writing custom XRT kernel driver
2. Directly programming NPU control registers
3. Manually managing DMA channels
4. Bypassing XCLBIN format entirely

**Complexity**: Several weeks of development, deep hardware knowledge required.

---

## Available Tools Analysis

### ✅ What We Have

| Tool | Location | Status |
|------|----------|--------|
| **Peano Compiler** | `venv313/lib/python3.13/site-packages/llvm-aie/bin/clang` | ✅ Working |
| **Chess Compiler** | `/home/ucadmin/tools/vitis_aie_essentials/tps/lnx64/target_aie_ml/bin/LNa64bin/chess-llvm-link` | ✅ Binary exists |
| **aie-opt** | `/home/ucadmin/.local/bin/aie-opt` | ✅ Working |
| **aie-translate** | `/home/ucadmin/.local/bin/aie-translate` | ✅ Working |
| **aiecc.py** | `/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py` | ⚠️ Path detection broken |
| **XRT 2.20.0** | `/opt/xilinx/xrt` | ✅ Working |
| **xclbinutil** | `/opt/xilinx/xrt/bin/xclbinutil` | ✅ Available |

### ❌ What We Need But Don't Have

- **v++** (Vitis compiler) - Not installed
- **Working aiecc.py** - Path detection broken
- **ryzen_ai Python package** - Not installed (would help aiecc.py detect tools)
- **Phoenix NPU platform files** - Not available

---

## Comparison: Solution Package vs Our Situation

### Solution Package Approach (`SOLUTION_FOR_CHESS_COMPILER_ISSUE.md`)

**Claim**: "Bypassed Chess compiler entirely and got 220× speedup!"

**Reality** (from reading `aie2_kernel_driver.py`):
1. Still uses `v++` for XCLBIN generation (line 89-95)
2. When v++ fails, falls back to **emulation mode** with mock XCLBIN (lines 114-143)
3. Emulation binary has no actual NPU code - just metadata

**Their compilation flow**:
```python
# Step 1: aie-opt (same as us) ✅
aie-opt --aie-lower-to-aie --aie-assign-tile-ids ...

# Step 2: aie-translate to JSON (we can do this) ✅
aie-translate --aie-generate-json ...

# Step 3: v++ to XCLBIN (we CAN'T do this) ❌
v++ --platform ... --target hw ...
```

**Conclusion**: They also rely on `v++` (which includes Chess). The "bypassing Chess" claim is misleading.

---

## Hardware Specifications (Corrected)

**AMD Phoenix XDNA1 NPU**:
- **Performance**: 15 TOPS INT8 (not 16)
- **Architecture**: 4 columns, NOT 4×6 array
- **AIE-ML Cores**: 4 total (1 per column at row 2)
- **Memory Tiles**: 4 (row 1)
- **Shim NOC Tiles**: 4 (row 0)

```
Row 2: [Compute] [Compute] [Compute] [Compute]  ← 4 AIE-ML cores
Row 1: [Memory]  [Memory]  [Memory]  [Memory]   ← Memory tiles
Row 0: [Shim]    [Shim]    [Shim]    [Shim]     ← DMA/NOC
       Col 0     Col 1     Col 2     Col 3
```

---

## Viable Paths Forward

### Option 1: Install Ryzen AI SDK ⏱️ 1-2 hours

**What**: Install AMD Ryzen AI Software Platform

**Where**: https://account.amd.com/en/ryzen-ai-sw.html (requires approval)

**Benefits**:
- ✅ Includes `ryzen_ai` Python package
- ✅ aiecc.py would auto-detect tools
- ✅ Provides proper Phoenix NPU platform files
- ✅ May include v++ or equivalent compiler

**Drawbacks**:
- ⏰ Requires AMD account approval (may take days)
- 💾 Large download (10-20 GB)
- 📋 License agreement required

**Likelihood of Success**: 90% - This is the official toolchain

### Option 2: Fix aiecc.py Path Detection ⏱️ 2-4 hours

**Approach**: Patch aiecc.py to use our Chess installation

**Steps**:
1. Create wrapper script that sets environment before importing aiecc
2. OR: Modify `/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/mlir_aie/python/aie/compiler/aiecc/main.py`
3. Force `opts.aietools_path` to our vitis_aie_essentials path
4. Test compilation

**Risks**:
- 🔧 Modifying installed package (updates will break it)
- 🐛 May encounter other compatibility issues
- ⚠️ Python 3.13 vs Python 3.10 incompatibilities

**Likelihood of Success**: 60% - May work but brittle

### Option 3: Use Pre-Compiled Matmul Kernel ⏱️ 30 min

**Approach**: Find and test matmul kernels that Team Lead 1 compiled

**Where to look**:
- Check other NPU optimization directories
- Search for `matmul_fixed.xclbin` mentioned in summary
- Test existing DMA pipelining code (already working at 19.1×)

**Benefits**:
- ✅ Immediate testing possible
- ✅ No compilation issues
- ✅ Can benchmark current solution

**Drawbacks**:
- ❌ May not have 32×32 version (might be 16×16)
- ❌ Doesn't solve future compilation needs

**Likelihood of Success**: 70% - If pre-compiled kernels exist

### Option 4: Accept Current Performance ⏱️ Immediate

**Approach**: Use current 19.1× realtime as baseline

**Rationale**:
- ✅ DMA pipelining already deployed (1.37× improvement)
- ✅ Encoder/decoder use ONNX Runtime (working)
- ✅ 19.1× is excellent performance (target was 29-38×)
- ⏸️ Focus on other optimizations (batch processing, etc.)

**Benefits**:
- ✅ No toolchain fighting
- ✅ Can deploy now
- ✅ Incremental improvements possible

**Drawbacks**:
- ❌ Not meeting 29-38× target
- ❌ Not reaching 220× ultimate goal

---

## Recommendation

**Short-term (Today)**: **Option 3** - Look for pre-compiled kernels to test

**Medium-term (Next Week)**: **Option 1** - Apply for Ryzen AI SDK access

**Long-term (Future)**: Evaluate custom runtime if needed for 220× target

---

## Key Insights Learned

1. **"Bypassing Chess compiler"** claims in solution docs are misleading - they still use v++
2. **aiecc.py path detection** is fragile and doesn't respect environment variables properly
3. **Phoenix NPU has 4 AIE-ML cores** (not 16), so kernel sizing matters
4. **MLIR-AIE v1.1.1** has different flags than newer versions in solution package
5. **XCLBIN creation** requires full Vitis toolchain or working aiecc.py - no easy shortcuts

---

## Files Generated Today

### ✅ Successfully Generated

- `build_direct/matmul_lowered.mlir` (6.2 KB) - Lowered MLIR ✅
- `build_direct/matmul_npu.bin` (16 bytes) - NPU instruction sequence ✅
- `matmul_32x32.o` (3.7 KB) - Compiled C kernel ✅

### ❌ Missing (Blocked)

- `matmul_32x32.xclbin` - Final NPU binary for XRT ❌

---

## Next Steps

1. **Search for existing compiled kernels**:
   ```bash
   find /home/ucadmin -name "matmul*.xclbin" 2>/dev/null
   find /home/ucadmin -name "*_fixed.xclbin" 2>/dev/null
   ```

2. **If found, test immediately**:
   ```bash
   python3 test_matmul_32x32.py
   ```

3. **If not found, apply for Ryzen AI SDK**:
   - Go to https://account.amd.com/en/ryzen-ai-sw.html
   - Request Early Access
   - Wait for approval

4. **Document current performance**:
   - 19.1× realtime is already excellent
   - DMA pipelining deployed successfully
   - Focus on other optimizations while waiting for SDK

---

**Status**: Blocked on toolchain, multiple viable paths forward identified

**Created**: October 30, 2025 05:00 UTC
**Author**: Claude Code (Sonnet 4.5)
