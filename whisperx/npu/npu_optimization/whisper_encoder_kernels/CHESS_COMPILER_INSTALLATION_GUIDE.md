# Chess Compiler Installation Guide for AMD Phoenix NPU (AIE2)

**Mission**: Install AMD Chess compiler toolchain to enable 32x32 matmul and multi-core XCLBIN compilation

**Impact**: Unlocks 6-8x combined performance improvement (32x32 matmul: 1.5-2x + multi-core: 4x)

**Hardware**: AMD Ryzen 9 8945HS with Phoenix NPU (AIE2 architecture)

**Date**: 2025-10-30

---

## Executive Summary

The Chess compiler (`chess-llvm-link`, `xchesscc`) is required for linking AIE kernels in multi-core configurations and 32x32 matmul operations. It is distributed as part of **AMD Vitis AIE Essentials** within the Ryzen AI Software package.

### Critical Path Discovery

The compilation toolchain looks for Chess compiler at:
```
${AIETOOLS_ROOT}/tps/lnx64/${target}/bin/LNa64bin/chess-llvm-link
```

Where:
- `AIETOOLS_ROOT` = Vitis AIE Essentials installation directory
- `target` = `aie_ml` for AIE2 (Phoenix NPU)

**Full expected path**: `/tools/ryzen_ai-1.3.0/vitis_aie_essentials/tps/lnx64/aie_ml/bin/LNa64bin/chess-llvm-link`

---

## Option 1: AMD Ryzen AI Software 1.3 Early Access (RECOMMENDED for AIE2)

This is the recommended path for Phoenix NPU (AIE2) development on Ubuntu.

### Prerequisites

✅ **Already Met**:
- Ubuntu 25.04 (kernel 6.14.0-34)
- XRT 2.20.0 installed
- MLIR-AIE source at `/home/ucadmin/mlir-aie-source`
- Python virtual environment ready

⚠️ **Required**:
- AMD account with Early Access approval
- ~5-10 GB free disk space (estimated)
- Free AIE Tools license

### Step 1: Register for Early Access

1. **Visit**: https://account.amd.com/en/member/ryzenai-sw-ea.html
2. **Sign in** with your AMD account (or create one)
3. **Request access** to the Ryzen AI SW Early Access Secure Site
4. **Wait for approval** (typically 1-2 business days)

### Step 2: Download Vitis AIE Essentials

Once approved:

1. **Login** to the Early Access portal
2. **Navigate** to "Ryzen AI Software 1.3 Early Access"
3. **Download**: `ryzen_ai-1.3.0ea1.tgz`
   - File type: VAIML Installer for Linux-based compilation
   - Size: ~3-8 GB (estimated, exact size not publicly documented)

### Step 3: Extract Vitis AIE Essentials

```bash
# Create installation directory (requires sudo)
sudo mkdir -p /tools
cd /tools

# Extract the downloaded package
sudo tar -xzvf ~/Downloads/ryzen_ai-1.3.0ea1.tgz
cd ryzen_ai-1.3.0

# Create dedicated directory for Vitis AIE Essentials
sudo mkdir vitis_aie_essentials

# Move the wheel package
sudo mv vitis_aie_essentials*.whl vitis_aie_essentials/
cd vitis_aie_essentials

# Unzip the wheel package (this extracts the actual tools)
sudo unzip vitis_aie_essentials*.whl
```

### Step 4: Obtain AIE Tools License

1. **Visit**: https://account.amd.com/en/forms/license/license-form.html
2. **Select**: "AI Engine Tools" or "Vitis Core Development Kit"
3. **Fill out** the license request form
4. **Download**: Your license file (named `Xilinx.lic`)
5. **Install** the license:

```bash
sudo mkdir -p /opt
sudo cp ~/Downloads/Xilinx.lic /opt/Xilinx.lic
sudo chmod 644 /opt/Xilinx.lic
```

**Note**: The license is typically free for development and evaluation purposes.

### Step 5: Configure Environment

Create an environment setup script:

```bash
cat > ~/aietools_setup.sh << 'EOF'
#!/bin/bash
#################################################################################
# Setup Vitis AIE Essentials for Phoenix NPU (AIE2)
#################################################################################

# Vitis AIE Essentials
export AIETOOLS_ROOT=/tools/ryzen_ai-1.3.0/vitis_aie_essentials
export PATH=$PATH:${AIETOOLS_ROOT}/bin

# License
export LM_LICENSE_FILE=/opt/Xilinx.lic

# XRT (already installed)
if [ -f /opt/xilinx/xrt/setup.sh ]; then
    source /opt/xilinx/xrt/setup.sh
fi

echo "AIETOOLS_ROOT: $AIETOOLS_ROOT"
echo "LM_LICENSE_FILE: $LM_LICENSE_FILE"
echo "Chess compiler: ${AIETOOLS_ROOT}/tps/lnx64/aie_ml/bin/LNa64bin/chess-llvm-link"
EOF

chmod +x ~/aietools_setup.sh
```

Add to your shell profile for automatic loading:

```bash
echo "source ~/aietools_setup.sh" >> ~/.bashrc
source ~/aietools_setup.sh
```

### Step 6: Verify Installation

```bash
# Check if xchesscc is accessible
which xchesscc
# Expected: /tools/ryzen_ai-1.3.0/vitis_aie_essentials/bin/xchesscc

# Check if chess-llvm-link exists
ls -lh ${AIETOOLS_ROOT}/tps/lnx64/aie_ml/bin/LNa64bin/chess-llvm-link
# Expected: File exists with execute permissions

# Verify license
ls -lh $LM_LICENSE_FILE
# Expected: /opt/Xilinx.lic exists

# Test basic xchesscc invocation
xchesscc --help 2>&1 | head -5
```

### Step 7: Test with MLIR-AIE

Activate your MLIR-AIE environment and test compilation:

```bash
cd /home/ucadmin/mlir-aie-source
source ironenv/bin/activate
source utils/env_setup.sh
source ~/aietools_setup.sh

# Test a simple programming example
cd programming_examples/basic/vector_scalar_mul
make clean
make
```

If successful, you should see Chess compiler being invoked during the build process.

---

## Option 2: AMD Vitis 2024.2 Full Installation (Alternative)

This is a larger installation that supports broader device families but is overkill for Phoenix NPU only.

### Characteristics

- **Size**: 50+ GB download, 100+ GB installed
- **Supports**: Versal, Alveo, VCK5000, Phoenix NPU
- **Use case**: If you need to develop for multiple AMD adaptive devices

### Download

1. Visit: https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html
2. Select: "Vitis Core Development Kit 2024.2"
3. Download: Full installer (~50 GB)

### Installation

```bash
# Run installer (requires root)
sudo ./Xilinx_Unified_2024.2_1126_2547_Lin64.bin

# Installation directory: /tools/Xilinx/Vitis/2024.2
```

### Environment Setup

```bash
export AIETOOLS_ROOT=/tools/Xilinx/Vitis/2024.2/aietools
export PATH=$PATH:${AIETOOLS_ROOT}/bin
export LM_LICENSE_FILE=/opt/Xilinx.lic
```

**Note**: This option is NOT recommended for Phoenix NPU-only development due to excessive size and unnecessary components.

---

## Option 3: Use Peano Instead (Chess-Free Development)

If you cannot obtain Chess compiler, you can use the open-source Peano compiler for AIE2 single-core development.

### Limitations

- ✅ **Supported**: Single-core AIE kernels
- ✅ **Supported**: Basic matmul operations (up to 16x16)
- ❌ **Not supported**: Multi-core linking (requires chess-llvm-link)
- ❌ **Not supported**: 32x32 matmul optimizations (requires xchesscc)
- ❌ **Performance**: Limited to single-core (~12-16x realtime instead of 50-65x)

### Already Installed

Peano is already installed via llvm-aie Python package in your MLIR-AIE environment.

### To Use Peano Only

```bash
cd /home/ucadmin/mlir-aie-source
source ironenv/bin/activate

# Verify Peano installation
python3 -c "import llvm_aie; print(llvm_aie.__file__)"

# Build examples without Chess
cd programming_examples/basic/matrix_multiplication/single_core
make
```

**Conclusion**: Peano is functional but limits performance gains. Chess compiler is required for the full 6-8x speedup target.

---

## Troubleshooting

### Issue: "chess-llvm-link: No such file or directory"

**Cause**: AIETOOLS_ROOT not set or Chess not installed

**Solution**:
```bash
# Check environment
echo $AIETOOLS_ROOT
# Should output: /tools/ryzen_ai-1.3.0/vitis_aie_essentials

# Verify Chess exists
ls ${AIETOOLS_ROOT}/tps/lnx64/aie_ml/bin/LNa64bin/chess-llvm-link

# Re-source environment
source ~/aietools_setup.sh
```

### Issue: License errors

**Symptom**: "FLEXlm error: -1,357" or "License checkout failed"

**Solution**:
```bash
# Check license file exists
ls -l $LM_LICENSE_FILE

# Verify license has correct permissions
sudo chmod 644 /opt/Xilinx.lic

# Check license contents (should have FEATURE/INCREMENT lines)
grep "FEATURE\|INCREMENT" /opt/Xilinx.lic
```

### Issue: "xchesscc not found"

**Solution**:
```bash
# Add to PATH
export PATH=$PATH:${AIETOOLS_ROOT}/bin

# Verify
which xchesscc
```

### Issue: Early Access request not approved

**Solution**:
- Check your email for AMD account notifications
- Allow 1-2 business days for approval
- Contact AMD support: aup@amd.com (for university researchers)
- Alternative: Use Peano-only workflow (limited performance)

### Issue: Kernel too old (Ubuntu 24.04)

**Solution**:
```bash
# Install Hardware Enablement (HWE) stack for kernel 6.11+
sudo apt update
sudo apt install --install-recommends linux-generic-hwe-24.04
sudo reboot
```

---

## Expected File Structure After Installation

```
/tools/ryzen_ai-1.3.0/
└── vitis_aie_essentials/
    ├── bin/
    │   ├── xchesscc                 # Chess compiler wrapper
    │   ├── chess-clang              # Chess C compiler
    │   └── [other tools]
    ├── tps/
    │   └── lnx64/
    │       ├── aie_ml/              # AIE2 (Phoenix NPU)
    │       │   └── bin/
    │       │       └── LNa64bin/
    │       │           └── chess-llvm-link   # ← Critical for multi-core
    │       ├── versal_prod/         # AIE1 (Versal)
    │       └── aie2p/               # AIE2P (future)
    ├── include/
    ├── data/
    └── lib/
```

---

## Integration with Matmul Compilation

Once Chess is installed, your 32x32 matmul compilation will proceed:

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/build_matmul_fixed

# Clean previous build
rm -rf _build _xclbin

# Ensure environment is loaded
source ~/aietools_setup.sh
source /home/ucadmin/mlir-aie-source/ironenv/bin/activate
source /home/ucadmin/mlir-aie-source/utils/env_setup.sh

# Build with Chess compiler
python build_matmul_32x32.py

# Expected: Full compilation including Chess linking phase
# Output: matmul_32x32.xclbin ready for 50-65x realtime performance
```

---

## Performance Validation

After installation, verify the performance improvements:

### Baseline (Current)
- 16x16 matmul, single-core
- Performance: ~8-10x realtime (estimated from initial testing)

### With 32x32 Matmul (Chess required)
- 32x32 matmul, single-core
- Performance: ~12-20x realtime (1.5-2x improvement)

### With Multi-Core (Chess required)
- 32x32 matmul, 4 cores
- Performance: ~50-65x realtime (6-8x combined improvement)

---

## Quick Reference Commands

### Daily Workflow

```bash
# Start development session
source ~/aietools_setup.sh
cd /home/ucadmin/mlir-aie-source
source ironenv/bin/activate
source utils/env_setup.sh

# Verify Chess is accessible
which xchesscc
ls ${AIETOOLS_ROOT}/tps/lnx64/aie_ml/bin/LNa64bin/chess-llvm-link

# Build your kernel
cd /path/to/your/project
make
```

### Environment Check

```bash
# Print all relevant environment variables
env | grep -E "AIETOOLS|XRT|LM_LICENSE"
```

---

## Estimated Installation Time

| Phase | Duration |
|-------|----------|
| AMD account creation & approval | 1-2 business days |
| Download ryzen_ai-1.3.0ea1.tgz | 10-30 minutes (depends on connection) |
| Extract and setup | 5-10 minutes |
| License request & download | 15-30 minutes |
| Environment configuration | 5 minutes |
| Verification testing | 10-15 minutes |
| **Total (excluding approval)** | **45-90 minutes** |
| **Total (including approval)** | **1-3 days** |

---

## Summary of Download Requirements

| Item | Size | Source |
|------|------|--------|
| ryzen_ai-1.3.0ea1.tgz | ~3-8 GB (est.) | AMD Early Access Portal |
| Xilinx.lic | ~10 KB | AMD License Portal |
| **Total disk space** | **~10 GB** | (including extracted files) |

---

## Support Resources

- **MLIR-AIE Documentation**: https://xilinx.github.io/mlir-aie/
- **GitHub Issues**: https://github.com/Xilinx/mlir-aie/issues
- **AMD University Program**: aup@amd.com
- **Early Access Portal**: https://account.amd.com/en/member/ryzenai-sw-ea.html
- **License Portal**: https://account.amd.com/en/forms/license/license-form.html

---

## Next Steps After Installation

1. ✅ Verify Chess compiler installation
2. ✅ Test basic MLIR-AIE example with Chess
3. ✅ Compile 32x32 matmul kernel
4. ✅ Benchmark single-core 32x32 performance
5. ✅ Implement multi-core configuration
6. ✅ Achieve 50-65x realtime target performance

---

**Document Version**: 1.0
**Last Updated**: 2025-10-30
**Author**: Team Lead: Chess Compiler Installation Research
**Status**: Ready for user execution
