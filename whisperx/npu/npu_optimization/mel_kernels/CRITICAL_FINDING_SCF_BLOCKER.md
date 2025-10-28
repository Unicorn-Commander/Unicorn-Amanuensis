# CRITICAL FINDING: SCF Loop Blocker in Core Compilation

**Date**: October 27, 2025
**Time**: 22:50 UTC
**Status**: ROOT CAUSE IDENTIFIED

---

## The Discovery

After extensive testing, I've identified the exact blocker preventing core body compilation:

**The `--aie-standard-lowering` pass CANNOT extract cores that contain `scf.for` loops.**

### Proof

**Test 1 - Simple Core (NO LOOP)**: ✅ **WORKS**
```mlir
%core_0_2 = aie.core(%tile_0_2) {
  func.call @my_kernel(%lock) : (index) -> ()
  aie.end
}
```
**Result**: Successfully extracts `func.func @core_0_2()` with body

**Test 2 - Core WITH LOOP**: ❌ **FAILS**
```mlir
%core_0_2 = aie.core(%tile_0_2) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %i = %c0 to %c10 step %c1 {
    func.call @my_kernel(%lock) : (index) -> ()
  }
  aie.end
}
```
**Result**: Only outputs declarations, NO core body function

### Why This Happens

1. `--aie-standard-lowering` expects simple operations in cores (arith, func.call, aie ops)
2. `scf.for` is a structured control flow operation that needs conversion to `cf` (control flow)
3. Cannot run `--convert-scf-to-cf` BEFORE extraction (crashes: "expected that op has no uses")
4. Cannot extract core body WITH scf operations

**This is a chicken-and-egg problem!**

---

## What This Means for Our MEL Kernel

Our `mel_physical.mlir` has this structure:
```mlir
%core_0_2 = aie.core(%tile_0_2) {
  scf.for %arg0 = %c0 to %c4294967294 step %c2 {
    aie.use_lock(...)
    aie.use_lock(...)
    func.call @mel_kernel_simple(...)
    aie.use_lock(...)
    aie.use_lock(...)
  }
  aie.end
} {link_with = "mel_kernel_simple.o"}
```

**This CANNOT be compiled with manual aie-opt passes because of the `scf.for` loop!**

---

## The Real Solution

Looking at the evidence:

1. ✅ `aiecc.py` exists and can compile cores with loops (matrix_transpose proof)
2. ❌ Python IRON API is broken (missing helper functions, circular imports)
3. ✅ Python can be fixed OR bypassed

**The actual blocker is NOT the core compiler - it's the Python environment!**

`aiecc.py` orchestrates the ENTIRE compilation pipeline, including:
- Extracting cores from MLIR (with proper pass ordering)
- Converting SCF to CF at the right time
- Calling Peano compiler
- Linking
- Generating CDO/PDI

We cannot replicate `aiecc.py` manually because we don't have all the pass details and ordering.

---

## Path Forward: Fix Python OR Use Alternative

### Option A: Fix Python Environment (RECOMMENDED - 1-2 hours)

**What's Broken**:
1. Missing `get_user_code_loc()` in util.py (ALREADY FIXED!)
2. Missing `make_maybe_no_args_decorator()` in util.py (ALREADY FIXED!)
3. Circular import `aie.extras.types` vs built-in `types`

**What's Left**:
- Fix the circular import (rename aie.extras.types to aie.extras.aietypes)
- OR set PYTHONPATH to prioritize aie module over built-in
- Then `aiecc.py` will work!

### Option B: Simplify Core to Remove Loop (2-3 hours)

Instead of:
```mlir
scf.for {
  acquire
  process
  release
}
```

Use external looping:
```mlir
core {
  acquire
  process
  release
  end
}
```

Loop in runtime sequence instead, calling kernel multiple times from host.

### Option C: Use Python IRON API from Scratch (1 week)

Build a completely new Python environment with fresh MLIR-AIE install.

---

## Recommendation

**FIX PYTHON** (Option A)

Why:
1. We've already fixed 2/3 of the issues
2. Only one import problem remains
3. Once fixed, we get the FULL toolchain
4. Can compile ANY pattern (loops, complex DMA, etc.)
5. 1-2 hours vs weeks of manual work

The circular import fix:
```bash
# Either rename the conflicting module
mv /home/ucadmin/mlir-aie-source/python/extras/types.py \
   /home/ucadmin/mlir-aie-source/python/extras/aietypes.py

# And update imports
find /home/ucadmin/mlir-aie-source/python -name "*.py" \
  -exec sed -i 's/from \.types import/from .aietypes import/g' {} \;
```

OR:

```bash
# Adjust PYTHONPATH to load aie before built-in
export PYTHONPATH=/home/ucadmin/mlir-aie-source/python:$PYTHONPATH
```

---

## Key Insight

**The "missing core compiler" was never missing!**

The core compiler is `aiecc.py` + all its orchestrated tools. We found all the individual tools (aie-opt, aie-translate, Peano), but we cannot use them manually because we don't know the exact pass ordering that handles SCF→CF conversion during core extraction.

**Fix Python → Get aiecc.py working → Compile cores with loops → Generate XCLBIN → Test → Success!**

---

## Files Created This Session

- `test_simple_core.mlir` - Test file that PROVED the scf.for blocker
- `mel_core_extracted.mlir` - Failed extraction (declarations only)
- `mel_core_no_link.mlir` - Failed (link_with wasn't the issue)
- `mel_core_from_original.mlir` - Failed (same result)

---

## Confidence Level

**VERY HIGH** - The root cause is definitively identified and proven with test cases.

The solution is clear: Fix Python circular import, then use aiecc.py.

---

**Next Step**: Fix circular import in Python, then compile with aiecc.py.

**ETA to Working XCLBIN**: 1-2 hours after Python fix.

---

Created: October 27, 2025 22:50 UTC
Blocker: SCF operations in cores prevent manual extraction
Solution: Fix Python → Use aiecc.py
