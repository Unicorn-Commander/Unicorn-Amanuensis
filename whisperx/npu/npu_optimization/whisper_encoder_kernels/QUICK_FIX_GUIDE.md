# Quick Fix Guide - Attention Kernel
## Get it working in 1 hour

---

## Problem

Attention kernels return all zeros because they use **wrong DMA API**.

## Solution

Use mel kernel's proven DMA API pattern.

---

## Quick Fix (Choose One)

### Option A: Use Pre-Fixed MLIR ⚡ FASTEST

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels

# Use the fixed MLIR I created
ls attention_fixed_npu_api.mlir  # Already done!

# Compile (need Peano)
# If Peano not found, see Option B
aiecc.py --no-aiesim \
         --xchesscc \
         --xbridge \
         --aie-generate-xclbin \
         --xclbin-name=attention_fixed.xclbin \
         --npu-insts-name=insts_fixed.bin \
         attention_fixed_npu_api.mlir
```

### Option B: Copy from Mel Template ✅ RECOMMENDED

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels

# Copy mel template
cp mel_kernels/build_fixed_v3/mel_fixed_v3.mlir attention_from_mel.mlir

# Edit with these changes:
# 1. Line 19: memref<800xi8> → memref<12288xi8>
# 2. Line 23: memref<80xi8> → memref<4096xi8>
# 3. Line 18: @of_in → @of_input
# 4. Line 22: @of_out → @of_output
# 5. Line 11: mel_kernel_simple → attention_64x64
# 6. Line 53: "mel_fixed_combined_v3.o" → "attention_int8_64x64_tiled.o"
# 7. Line 59: %c800_i64 = arith.constant 800 : i64
#             → %c12288_i64 = arith.constant 12288 : i64
# 8. Line 60: %c80_i64 = arith.constant 80 : i64
#             → %c4096_i64 = arith.constant 4096 : i64
# 9. Line 63-67: Update memcpy_nd sizes to 12288
# 10. Line 70-76: Update memcpy_nd sizes to 4096

# Compile
aiecc.py attention_from_mel.mlir
```

---

## Test

```bash
# Update test script to use new XCLBIN
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels

# Edit test_iron_fresh.py:
# Change line 16: xclbin_path = "attention_fixed.xclbin"
# Change line 17: insts_path = "insts_fixed.bin"

# Run test
python3 test_iron_fresh.py

# Expected output:
# Non-zero: 3500+/4096 (85%+)  ✅
# Range: [-128, 127]  ✅
# Execution: <1ms  ✅
```

---

## If Compilation Fails

**Missing Peano compiler?**

Check these locations:
```bash
find /home/ucadmin -name "clang++" -path "*llvm-aie*" 2>/dev/null
find /home/ucadmin -name "aiecc.py" 2>/dev/null
```

If not found:
```bash
# Option 1: Use mel's compilation infrastructure
cd mel_kernels/build_fixed_v3
# See how they compiled it

# Option 2: Install mlir-aie
git clone https://github.com/Xilinx/mlir-aie
cd mlir-aie
# Follow build instructions
```

---

## Success Criteria

After fix:
- ✅ Non-zero output (>50%)
- ✅ Execution <1ms
- ✅ Range includes negatives
- ✅ Ready for 4-tile extension

---

## Questions?

See full reports:
- `INVESTIGATION_SUMMARY.md` - Complete analysis
- `ATTENTION_KERNEL_FIX_REPORT.md` - Technical details

