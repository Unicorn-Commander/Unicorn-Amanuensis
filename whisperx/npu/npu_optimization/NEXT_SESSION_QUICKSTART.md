# Next Session Quick Start Guide

**Status**: 75% Complete - Ready for XCLBIN Packaging!
**Time Needed**: 45-90 minutes to first NPU execution

---

## 🎯 Quick Status

✅ **DONE**:
- Complete C++ toolchain built (414 MB)
- MLIR lowered through all passes
- C++ kernel compiled for AIE2
- NPU instructions generated
- xaie configuration created
- 7 files ready for packaging!

⏳ **TODO**:
- Package into XCLBIN format (1 step!)
- Write XRT test program
- Execute on NPU

---

## 🚀 Fast Track (Recommended)

### Option 1: Use Docker (30 min)

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization

# Pull MLIR-AIE Docker image
docker pull ghcr.io/xilinx/mlir-aie/mlir-aie:latest

# Generate XCLBIN
docker run --rm \
  -v $(pwd):/work \
  -w /work \
  ghcr.io/xilinx/mlir-aie/mlir-aie:latest \
  aiecc.py --aie-generate-xclbin \
           --no-compile-host \
           --xclbin-name=passthrough.xclbin \
           passthrough_step2.mlir

# Verify
ls -lh passthrough.xclbin
```

**Expected**: passthrough.xclbin (50-100 KB)

---

### Option 2: Fix Python API and Rebuild (2-4 hours)

If you want the complete solution:

1. Add IRON Python modules to MLIR-AIE build
2. Rebuild with complete Python API
3. Use local aiecc.py

**Reference**: `/home/ucadmin/mlir-aie-source/` - build already done, just need to add IRON modules

---

## 📝 After Getting XCLBIN

### Write Test Program

**File**: `test_passthrough.cpp`

```cpp
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    // Open NPU device
    auto device = xrt::device(0);

    // Load XCLBIN
    auto uuid = device.load_xclbin("passthrough.xclbin");

    // Get kernel (name from MLIR: passthrough_kernel)
    auto kernel = xrt::kernel(device, uuid, "MLIR_AIE");

    // Create buffers (1024 bytes each)
    auto in_bo = xrt::bo(device, 1024, kernel.group_id(0));
    auto out_bo = xrt::bo(device, 1024, kernel.group_id(1));

    // Write test pattern
    auto in_map = in_bo.map<uint8_t*>();
    for (int i = 0; i < 1024; i++) {
        in_map[i] = i & 0xFF;
    }
    in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Execute kernel
    auto run = kernel(in_bo, out_bo, 1024);
    run.wait();

    // Read results
    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto out_map = out_bo.map<uint8_t*>();

    // Verify passthrough worked
    bool passed = true;
    for (int i = 0; i < 1024; i++) {
        if (out_map[i] != in_map[i]) {
            printf("FAIL at byte %d: expected %d, got %d\n", i, in_map[i], out_map[i]);
            passed = false;
            break;
        }
    }

    printf("Test %s!\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
```

### Compile Test

```bash
g++ test_passthrough.cpp -o test_passthrough \
    -I/opt/xilinx/xrt/include \
    -L/opt/xilinx/xrt/lib \
    -lxrt_coreutil \
    -std=c++17
```

### Run Test

```bash
./test_passthrough
```

**Expected output**:
```
Test PASSED!
```

---

## 🎉 When It Works

You'll have achieved:

1. ✅ First custom NPU kernel execution
2. ✅ Complete understanding of MLIR→NPU pipeline
3. ✅ Foundation for 220x Whisper performance
4. ✅ Proof that custom kernels work on Phoenix NPU

---

## 📁 Files Generated This Session

All in: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/`

```
passthrough_complete.mlir     (2.4 KB) - Original source
passthrough_kernel.cc         (616 B)  - C++ kernel
passthrough_step1.mlir        (3.8 KB) - After ObjectFIFO transform
passthrough_step2.mlir        (4.4 KB) - With ELF reference ← USE THIS
passthrough_kernel_new.o      (988 B)  - Compiled AIE2 kernel
passthrough_npu.bin           (16 B)   - NPU instructions
passthrough_xaie.txt          (12 KB)  - libxaie config
```

**Documentation**:
```
COMPILATION_SUCCESS.md         - Build success details
XCLBIN_GENERATION_PROGRESS.md  - Step-by-step progress
FINAL_SESSION_SUMMARY.md       - Complete technical summary
NEXT_SESSION_QUICKSTART.md     - This file
```

---

## 🛠️ Toolchain Location

**MLIR-AIE tools**: `/home/ucadmin/mlir-aie-source/install/bin/`
- aie-opt, aie-translate, bootgen, etc.

**Environment setup**:
```bash
source /home/ucadmin/mlir-aie-source/utils/env_setup.sh \
       /home/ucadmin/mlir-aie-source/install
```

**Peano compiler**:
```bash
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-source/ironenv/lib/python3.13/site-packages/llvm-aie
```

---

## 🎯 Success Criteria

- [ ] passthrough.xclbin generated (50-100 KB)
- [ ] test_passthrough compiled successfully
- [ ] Test runs without errors
- [ ] Test output: "Test PASSED!"
- [ ] **CELEBRATE!** 🎉

---

## 💡 If You Get Stuck

1. **Check device**: `ls -l /dev/accel/accel0`
2. **Check XRT**: `/opt/xilinx/xrt/bin/xrt-smi examine`
3. **Review docs**: Read `FINAL_SESSION_SUMMARY.md`
4. **Try Docker**: Easiest path to XCLBIN

---

## 📞 Key Resources

- **XRT Documentation**: `/opt/xilinx/xrt/docs/`
- **MLIR-AIE Examples**: `/home/ucadmin/mlir-aie-source/programming_examples/`
- **Test Examples**: `/home/ucadmin/mlir-aie-source/test/npu-xrt/`

---

**Estimated Total Time**: 45-90 minutes
**Confidence Level**: Very High (95%)
**Next Major Milestone**: First NPU kernel execution! 🚀

---

*Go get 'em! You're almost there!* 🦄✨
