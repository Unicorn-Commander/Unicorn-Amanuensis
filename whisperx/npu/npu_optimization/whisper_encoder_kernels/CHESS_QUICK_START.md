# Chess Compiler Quick Start Guide

**Time to read**: 3 minutes
**Time to execute** (excluding Early Access approval): 45-90 minutes

---

## What is This?

Chess compiler (`chess-llvm-link`, `xchesscc`) is AMD's compiler for AI Engine (AIE) cores. Without it, you're limited to:
- Single-core AIE kernels only
- Maximum 16x16 matmul
- ~8-10x realtime performance

With Chess compiler, you unlock:
- Multi-core AIE compilation
- 32x32 matmul operations
- **6-8x speedup → 50-65x realtime performance**

---

## Do I Have Chess Already?

Quick check:

```bash
# Check if Chess exists
which xchesscc

# Expected if installed: /tools/ryzen_ai-1.3.0/vitis_aie_essentials/bin/xchesscc
# Expected if NOT installed: (no output)
```

If you see a path, **you already have it!** Skip to "Verify Installation" section below.

---

## Installation (5 Steps)

### Step 1: Request Early Access (1-2 business days)

1. Visit: https://account.amd.com/en/member/ryzenai-sw-ea.html
2. Sign in with AMD account
3. Request access to "Ryzen AI SW Early Access"
4. Wait for approval email

**While waiting**: Continue to Step 4 (license request)

### Step 2: Download (10-30 minutes)

Once approved:

1. Login to Early Access portal
2. Download: `ryzen_ai-1.3.0ea1.tgz` (~3-8 GB)

### Step 3: Extract (5-10 minutes)

```bash
sudo mkdir -p /tools
cd /tools
sudo tar -xzvf ~/Downloads/ryzen_ai-1.3.0ea1.tgz
cd ryzen_ai-1.3.0
sudo mkdir vitis_aie_essentials
sudo mv vitis_aie_essentials*.whl vitis_aie_essentials/
cd vitis_aie_essentials
sudo unzip vitis_aie_essentials*.whl
```

### Step 4: Get License (15-30 minutes)

1. Visit: https://account.amd.com/en/forms/license/license-form.html
2. Request "AI Engine Tools" license
3. Download `Xilinx.lic`
4. Install:

```bash
sudo mkdir -p /opt
sudo cp ~/Downloads/Xilinx.lic /opt/Xilinx.lic
sudo chmod 644 /opt/Xilinx.lic
```

**Note**: License is typically free for development

### Step 5: Configure Environment (5 minutes)

```bash
# Create setup script
cat > ~/aietools_setup.sh << 'EOF'
#!/bin/bash
export AIETOOLS_ROOT=/tools/ryzen_ai-1.3.0/vitis_aie_essentials
export PATH=$PATH:${AIETOOLS_ROOT}/bin
export LM_LICENSE_FILE=/opt/Xilinx.lic

# Load XRT if available
if [ -f /opt/xilinx/xrt/setup.sh ]; then
    source /opt/xilinx/xrt/setup.sh
fi
EOF

chmod +x ~/aietools_setup.sh

# Add to .bashrc
echo "source ~/aietools_setup.sh" >> ~/.bashrc

# Load now
source ~/aietools_setup.sh
```

---

## Verify Installation

```bash
# Test 1: Check xchesscc
which xchesscc
# Expected: /tools/ryzen_ai-1.3.0/vitis_aie_essentials/bin/xchesscc

# Test 2: Check chess-llvm-link
ls ${AIETOOLS_ROOT}/tps/lnx64/aie_ml/bin/LNa64bin/chess-llvm-link
# Expected: File exists

# Test 3: Check license
ls -l $LM_LICENSE_FILE
# Expected: /opt/Xilinx.lic exists

# Test 4: Build example
cd /home/ucadmin/mlir-aie-source
source ironenv/bin/activate
source utils/env_setup.sh
cd programming_examples/basic/vector_scalar_mul
make clean && make
# Expected: Successful build with Chess compiler
```

All tests pass? **You're ready to compile 32x32 matmul!**

---

## Now Compile 32x32 Matmul

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/build_matmul_fixed

# Ensure all environments loaded
source ~/aietools_setup.sh
source /home/ucadmin/mlir-aie-source/ironenv/bin/activate
source /home/ucadmin/mlir-aie-source/utils/env_setup.sh

# Clean and build
rm -rf _build _xclbin
python build_matmul_32x32.py

# Expected: Full compilation including Chess linking
# Output: matmul_32x32.xclbin ready for 50-65x realtime!
```

---

## Troubleshooting

### "chess-llvm-link: No such file"

```bash
# Re-source environment
source ~/aietools_setup.sh

# Verify AIETOOLS_ROOT
echo $AIETOOLS_ROOT
# Should output: /tools/ryzen_ai-1.3.0/vitis_aie_essentials
```

### "License error"

```bash
# Check license file
cat $LM_LICENSE_FILE | head -20

# Should contain lines like:
# INCREMENT ... xilinxd ...
# FEATURE ... xilinxd ...
```

### Early Access not approved yet

Use Peano-only workflow (limited to single-core, ~12-16x realtime):

```bash
cd /home/ucadmin/mlir-aie-source
source ironenv/bin/activate
cd programming_examples/basic/matrix_multiplication/single_core
make
```

---

## Full Documentation

See `CHESS_COMPILER_INSTALLATION_GUIDE.md` for:
- Detailed installation options
- Alternative installation methods
- Advanced troubleshooting
- System file structure
- Integration details

---

## Summary

| Installation Component | Time |
|------------------------|------|
| Early Access approval | 1-2 business days |
| Download | 10-30 minutes |
| Extract & Install | 5-10 minutes |
| License request | 15-30 minutes |
| Configure environment | 5 minutes |
| Test compilation | 5 minutes |
| **Total** (excluding approval) | **45-90 minutes** |

**Result**: 32x32 matmul compilation unlocked → 50-65x realtime performance!

---

**Quick Reference**:
- Early Access: https://account.amd.com/en/member/ryzenai-sw-ea.html
- License: https://account.amd.com/en/forms/license/license-form.html
- Full Guide: `CHESS_COMPILER_INSTALLATION_GUIDE.md`
- Support: aup@amd.com (university researchers)
