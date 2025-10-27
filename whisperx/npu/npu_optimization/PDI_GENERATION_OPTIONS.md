# PDI Generation Options & Findings

**Date**: October 26, 2025
**Status**: Investigation Complete
**Location**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/`

---

## Summary

We have **multiple paths** to generate proper PDI files and complete our XCLBIN. After investigation, here are all discovered options with recommendations.

---

## Option 1: Build MLIR-AIE from Source (RECOMMENDED)

**Pros**:
- Official toolchain
- Complete control
- Generates PDI specific to our kernel
- Reusable for future kernels

**Cons**:
- Build time: ~30-60 minutes
- Disk space: ~2 GB

**Steps**:
```bash
# Clone MLIR-AIE repository
cd /home/ucadmin
git clone --recursive https://github.com/Xilinx/mlir-aie.git
cd mlir-aie

# Build with all tools
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_AIE_ENABLE_PYTHON=ON \
  -DMLIR_AIE_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=$(which python3) \
  -DCMAKE_INSTALL_PREFIX=/home/ucadmin/mlir-aie-install

make -j$(nproc)
make install

# Add to PATH
export PATH=/home/ucadmin/mlir-aie-install/bin:$PATH

# Generate PDI
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization
aie-translate \
  --aie-generate-xclbin \
  --xclbin-name=passthrough \
  passthrough_step3.mlir \
  -o passthrough_final.xclbin
```

**Timeline**: 1-2 hours total
**Success Probability**: 95%

---

## Option 2: Use Reference PDI Files (QUICK TEST)

**Discovery**: We found 16 large PDI files (200-270 KB) in our directory!

**Files**:
```
7f5ac85a-2023-0008-0005-416198770000.pdi  272 KB
7f5ac85a-2023-0008-0005-416198770001.pdi  252 KB
7f5ac85a-2023-0008-0005-416198770002.pdi  202 KB
... (13 more)
```

**Structure Verified**:
```
00000000  dd 00 00 00 44 33 22 11  88 77 66 55 cc bb aa 99  |....D3"..wfU....|
00000030  49 44 50 50 20 10 20 00                           |IDPP . .        |
000000a0  61 69 65 5f 69 6d 61 67  65 00 00 00 00 00 00 00  |aie_image.......|
00000150  43 44 4f 00                                       |CDO.            |
```

**Quick Test**:
```bash
# Try one of these PDIs in our XCLBIN
/opt/xilinx/xrt/bin/xclbinutil \
  --add-replace-section BITSTREAM:RAW:passthrough_npu.bin \
  --add-replace-section MEM_TOPOLOGY:JSON:mem_topology.json \
  --add-replace-section IP_LAYOUT:JSON:passthrough_ip_layout.json \
  --add-replace-section AIE_PARTITION:JSON:passthrough_aie_partition.json \
  --add-replace-section PDI:RAW:7f5ac85a-2023-0008-0005-416198770000.pdi \
  --add-replace-section GROUP_TOPOLOGY:JSON:group_topology.json \
  --force \
  --output test_reference_pdi.xclbin

# Test
python3 test_pyxrt_correct.py
```

**Pros**:
- Instant testing (no build required)
- Can validate XCLBIN loads
- May work for testing

**Cons**:
- PDI won't match our kernel
- Likely won't execute properly
- Just for validation testing

**Timeline**: 5 minutes
**Success Probability**: 50% (for testing only)

---

## Option 3: Extract from Working MLIR-AIE Examples

**Approach**: Look for MLIR-AIE examples in `/home/ucadmin/mlir-aie-source`

**Steps**:
```bash
# Search for passthrough or simple examples
find /home/ucadmin/mlir-aie-source -name "*passthrough*" -o -name "*simple*"

# Look for pre-compiled XCLBINs
find /home/ucadmin/mlir-aie-source -name "*.xclbin"

# Extract PDI if found
/opt/xilinx/xrt/bin/xclbinutil \
  --dump-section PDI:RAW:extracted.pdi \
  --input example.xclbin
```

**Timeline**: 15-30 minutes
**Success Probability**: 30% (depends on examples available)

---

## Option 4: Manual PDI Construction

**Approach**: Build PDI manually from our artifacts

**What We Have**:
- `passthrough_kernel_new.o` (988 B) - Compiled AIE2 kernel
- `passthrough_npu.bin` (16 B) - NPU instructions
- `passthrough_xaie.txt` (12 KB) - xaie configuration

**PDI Structure Required**:
```
Offset  Size   Content
------  -----  -------
0x00    32 B   PDI header
0x20    80 B   IDPP signature block
0xA0    24 B   aie_image section header
0xC0    ???    aie_image data (compiled kernel)
???     ???    CDO section
```

**CDO Commands Needed**:
- Tile enable commands
- Memory allocation
- DMA configurations
- Stream connections

**Difficulty**: Very High
**Timeline**: 8-16 hours
**Success Probability**: 40%

**Not Recommended**: Too time-consuming for uncertain outcome

---

## Option 5: AMD Vitis Tools

**Approach**: Use AMD Vitis toolchain if available

**Check**:
```bash
which vitis
which xsct
which bootgen  # Native binary, not Python wrapper
```

**If Available**:
- Vitis may include complete MLIR-AIE toolchain
- Can use Vitis HLS for PDI generation
- Official AMD tooling

**Status**: Need to investigate if installed
**Timeline**: Variable
**Success Probability**: High if tools available

---

## Comparison Matrix

| Option | Time | Difficulty | Success | Reusable | Recommended |
|--------|------|------------|---------|----------|-------------|
| **Build MLIR-AIE** | 1-2 hrs | Medium | 95% | ✅ Yes | **⭐ BEST** |
| Reference PDI Test | 5 min | Low | 50% | ❌ No | For validation only |
| Extract Example | 30 min | Low | 30% | ✅ Yes | Worth trying first |
| Manual PDI | 8-16 hrs | Very High | 40% | ❌ No | Not recommended |
| Vitis Tools | Variable | Low | High | ✅ Yes | If available |

---

## Recommended Action Plan

### Phase 1: Quick Investigation (15 min)
```bash
# 1. Search for MLIR-AIE examples
find /home/ucadmin/mlir-aie-source -type f -name "*.mlir" | grep -i pass

# 2. Search for pre-built XCLBINs
find /home/ucadmin -name "*.xclbin" 2>/dev/null | grep -v node_modules

# 3. Check for Vitis
which vitis xsct bootgen
```

### Phase 2: If No Examples Found, Build MLIR-AIE (1-2 hrs)
```bash
cd /home/ucadmin
git clone --recursive https://github.com/Xilinx/mlir-aie.git
cd mlir-aie
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMLIR_AIE_ENABLE_PYTHON=ON
make -j$(nproc)
make install
```

### Phase 3: Generate PDI (5 min)
```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization
/home/ucadmin/mlir-aie-install/bin/aie-translate \
  --aie-generate-xclbin \
  passthrough_step3.mlir \
  -o passthrough_final.xclbin
```

### Phase 4: Test on NPU (5 min)
```python
import pyxrt
device = pyxrt.device(0)
xclbin = pyxrt.xclbin("passthrough_final.xclbin")
uuid = device.register_xclbin(xclbin)
kernel = pyxrt.kernel(device, uuid, "DPU:passthrough")
print("🎉 SUCCESS!")
```

---

## Discovery: Large PDI Files

**Found**: 16 PDI files in our directory (200-270 KB each)

**Structure**: All have proper format:
- PDI header ✅
- IDPP signature ✅
- aie_image section ✅
- CDO configuration ✅

**Size Comparison**:
- Our incomplete PDI: 16 bytes
- mobilenet PDI: 8.7 KB
- These reference PDIs: 200-270 KB (much larger!)

**Hypothesis**: These might be from complex multi-kernel XCLBINs

**Testing Value**:
- Can validate our XCLBIN structure loads
- Won't execute our kernel (different kernel)
- Good for PyXRT registration testing

---

## Tools Inventory

### Available (Working)
```
/opt/xilinx/xrt/bin/xclbinutil              ✅ XCLBIN manipulation
/opt/xilinx/xrt/bin/xrt-smi                 ✅ NPU management
pyxrt module                                 ✅ Python NPU access
aie-opt (Python wrapper - needs fix)         ⚠️ MLIR optimization
aie-translate (Python wrapper - needs fix)   ⚠️ PDI generation
bootgen (Python wrapper - needs fix)         ⚠️ PDI packaging
```

### Needed
```
aie-translate (native binary)                ❌ Need from MLIR-AIE build
aie-opt (native binary)                      ❌ Need from MLIR-AIE build
bootgen (native binary)                      ❌ May be in Vitis
```

### In MLIR-AIE Source
```
/home/ucadmin/mlir-aie-source/               ⚠️ Source only, not built
```

---

## Success Criteria

**PDI Generation Successful When**:
1. PDI file generated
2. Size: 8-15 KB (for simple passthrough kernel)
3. Contains proper header structure
4. aie_image section present
5. CDO section present

**XCLBIN Complete When**:
1. Built with proper PDI
2. Size: ~11-16 KB total
3. PyXRT registers successfully
4. Kernel object created
5. Kernel executes on NPU

---

## Next Steps

**Immediate** (< 5 min):
```bash
# Quick search for examples
find /home/ucadmin/mlir-aie-source -name "*.mlir" | head -20
```

**Short-term** (1-2 hours):
```bash
# Build MLIR-AIE from source
cd /home/ucadmin
git clone --recursive https://github.com/Xilinx/mlir-aie.git
# ... build steps ...
```

**Testing** (5 min):
```bash
# Test with reference PDI
# (just to validate XCLBIN structure loads)
```

---

## Conclusion

**Best Path Forward**: Build MLIR-AIE from source

**Reasoning**:
- Only 1-2 hour investment
- Official, supported toolchain
- Generates correct PDI for our kernel
- Reusable for future development
- 95% success probability

**Alternative**: Check for existing examples first (15 min investment)

**Status**: Ready to proceed with build

---

**Generated**: October 26, 2025
**By**: Claude Code
**For**: Magic Unicorn Unconventional Technology & Stuff Inc.
**Recommendation**: Build MLIR-AIE, then generate PDI, then test on NPU
**ETA to 100%**: 2-3 hours maximum
