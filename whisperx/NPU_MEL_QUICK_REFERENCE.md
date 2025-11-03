# NPU Mel Preprocessing - Quick Reference

**Status**: ⚠️ Ready for Recompilation
**Date**: November 2, 2025

---

## TL;DR

✅ **What Works**:
- NPU runtime code (`NPUMelPreprocessor`)
- Server integration (`server_dynamic.py`)
- Automatic fallback to CPU
- Test suite for validation

❌ **What Doesn't Work**:
- Current XCLBINs compiled BEFORE Oct 28 accuracy fixes
- Missing/incomplete instruction binaries
- Accuracy unknown (cannot test without working XCLBINs)

🎯 **Recommendation**: **DO NOT ENABLE YET** - Recompile XCLBINs first

---

## Critical Finding

**XCLBIN Timeline**:
```
Oct 27 15:20  → mel_fft.xclbin compiled (BEFORE fixes)
Oct 28 00:00  → FFT + mel filterbank fixes implemented in C code
Oct 28 01:00  → mel_int8_final.xclbin compiled (during fixes - partial?)
Nov 1  03:01  → C source code updated to latest with all fixes
```

**Conclusion**: XCLBINs are stale - need recompilation with fixed C code

---

## Files to Review

1. **NPU_MEL_STATUS.md** - Full investigation report (800+ lines)
2. **test_npu_mel_runtime.py** - Test suite (410 lines)
3. **server_dynamic.py** - Updated integration (lines 182-226)
4. **npu_mel_preprocessing.py** - Runtime class (lines 172-196)

---

## Next Steps (Priority Order)

### 1. Recompile XCLBINs (2-4 hours) 🔴 **CRITICAL**

**Location**: `npu/npu_optimization/mel_kernels/`

**Required**:
- Fixed C code: ✅ Already in place
  - `fft_fixed_point.c` (FFT scaling fix)
  - `mel_kernel_fft_fixed.c` (HTK mel filterbanks)
  - `mel_coeffs_fixed.h` (207KB coefficient tables)

**Commands** (example - adapt to your build system):
```bash
cd npu/npu_optimization/mel_kernels
rm -rf build/*
# Recompile with MLIR-AIE2 toolchain
make clean
make all
# Verify
ls -lh build/mel_fixed_new.xclbin  # Should be >2KB
ls -lh build/insts.bin             # Should be >0 bytes
```

---

### 2. Test Accuracy (30 min)

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
python3 test_npu_mel_runtime.py
```

**Expected**:
- NPU initialization: ✅
- Processing works: ✅
- Correlation with librosa: >0.95 ✅

---

### 3. Enable in Production (15 min)

**Only if correlation >0.95!**

```bash
# Server will auto-detect and use new XCLBIN
python3 server_dynamic.py

# Test
curl -X POST -F "file=@test_audio.wav" http://localhost:9004/transcribe
```

---

## Oct 28 Accuracy Fixes (Summary)

### Fix #1: FFT Scaling
- **Problem**: No scaling → 512x overflow
- **Fix**: Added `>>1` scaling per stage
- **Result**: Correlation 0.44 → 1.0000 ✅

### Fix #2: HTK Mel Filterbanks
- **Problem**: Linear binning (wrong)
- **Fix**: HTK triangular filters with proper mel-scale
- **Result**: Error <0.38% vs librosa ✅

**Combined**: Expected >95% correlation (from 4.68%)

---

## Performance Expectations

### Current (CPU)
```
Mel preprocessing: 30 ms (300 µs/frame)
```

### After Recompilation (NPU)
```
Mel preprocessing: 5 ms (50 µs/frame)
Speedup: 6x
Accuracy: >95%
```

### Full Pipeline Target
```
Current:  11x realtime (500 ms for 5.5s audio)
With NPU: 11.7x realtime (475 ms for 5.5s audio)
Target:   18-20x realtime (requires encoder/decoder on NPU)
```

---

## Quick Commands

### Test NPU Runtime
```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
python3 test_npu_mel_runtime.py
```

### Check NPU Device
```bash
ls -l /dev/accel/accel0
/opt/xilinx/xrt/bin/xrt-smi examine
```

### View XCLBINs
```bash
ls -lh npu/npu_optimization/mel_kernels/build/*.xclbin
```

### View Instruction Binaries
```bash
ls -lh npu/npu_optimization/mel_kernels/build/*.bin | grep -v " 0 "
```

---

## Hardware Status

✅ **AMD Phoenix NPU**: Detected at /dev/accel/accel0
✅ **XRT 2.20.0**: Installed and operational
✅ **Firmware**: 1.5.5.391 (latest)
✅ **Device**: AMD Ryzen 9 8945HS with XDNA1 NPU

---

## Risk Assessment

| Item | Risk | Mitigation |
|------|------|------------|
| **Recompilation** | Low | C code validated in Python tests |
| **Accuracy** | Low | Fixes proven to work (1.0000 FFT correlation) |
| **Integration** | Very Low | Automatic CPU fallback implemented |
| **Performance** | Low | Expected 6x speedup is conservative |

---

## Decision Matrix

### Enable NPU Now?
❌ **NO** - XCLBINs don't have Oct 28 fixes

### After Recompilation?
✅ **YES** - If correlation >0.95

### Full NPU Pipeline (encoder/decoder)?
⏳ **LATER** - Requires custom MLIR-AIE2 kernels (8-12 weeks)

---

## Contact Points

**Code**:
- `server_dynamic.py` - Server integration
- `npu_mel_preprocessing.py` - NPU runtime class
- `test_npu_mel_runtime.py` - Test suite

**Documentation**:
- `NPU_MEL_STATUS.md` - Full investigation report
- `BOTH_FIXES_COMPLETE_OCT28.md` - Accuracy fixes documentation
- `NPU_MEL_QUICK_REFERENCE.md` - This document

**Build**:
- `npu/npu_optimization/mel_kernels/` - Kernel source and build

---

**Last Updated**: November 2, 2025
**Next Review**: After XCLBIN recompilation
