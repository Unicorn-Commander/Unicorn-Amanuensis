# MLIR-AIE2 Kernel Compilation Blockers

**Date**: October 25, 2025
**Status**: 🚧 BLOCKED - Dependencies Required
**Priority**: CRITICAL PATH for 220x NPU acceleration

---

## Summary

We have successfully:
- ✅ Verified NPU hardware (Phoenix NPU at `/dev/accel/accel0`)
- ✅ Confirmed XRT 2.20.0 installation working
- ✅ Located MLIR-AIE toolchain (`/home/ucadmin/mlir-aie-prebuilt/`)
- ✅ Created corrected MLIR kernel templates (using `npu1` device)
- ✅ Identified compilation approach (use `aiecc.py` orchestrator)

We are BLOCKED on:
- ❌ Python dependencies incomplete (`aie.extras.util` module missing)
- ❌ Docker image access denied (requires authentication)
- ❌ Cannot install wheel file (invalid wheel filename)

---

## Blockers Detailed

### 1. Python Dependencies Missing

**Issue**: The `mlir-aie` Python package has incomplete module structure.

**Error**:
```
ModuleNotFoundError: No module named 'aie.extras.util'
```

**Root Cause**:
- `/home/ucadmin/mlir-aie-prebuilt/mlir_aie/python/aie/extras/` contains:
  - `meta.py` ✅
  - `types.py` ✅
  - `util.py` ❌ MISSING (should be here but isn't)
- However, `util.py` exists at `/home/ucadmin/mlir-aie-prebuilt/mlir_aie/python/aie/util.py`
- The package expects `aie.extras.util` but only provides `aie.util`

**Impact**: Cannot import IRON Python API or use `aiecc.py` with Python dependencies.

**Attempted Solutions**:
1. ❌ Set `PYTHONPATH=/home/ucadmin/mlir-aie-prebuilt/mlir_aie/python` - module structure issue
2. ❌ Install wheel file - invalid wheel filename error
3. ❌ Use Docker image - access denied

### 2. Docker Image Access Denied

**Issue**: Cannot pull official MLIR-AIE Docker image.

**Error**:
```bash
$ docker pull ghcr.io/xilinx/mlir-aie:latest
Error response from daemon: Head "https://ghcr.io/v2/xilinx/mlir-aie/manifests/latest": denied
```

**Root Cause**: Image requires GitHub Container Registry authentication.

**Impact**: Cannot use pre-configured Docker environment with all dependencies.

### 3. Wheel File Installation Failed

**Issue**: Cannot install the prebuilt wheel file.

**Error**:
```bash
$ pip3 install /home/ucadmin/mlir-aie-prebuilt/mlir-aie.whl
ERROR: mlir-aie.whl is not a valid wheel filename.
```

**Root Cause**: The wheel file is named `mlir-aie.whl` but pip expects a proper wheel filename with version info like `mlir_aie-0.0.1.2025030204+eb26c0c-py3-none-linux_x86_64.whl`.

**Impact**: Cannot install Python package properly.

---

## What We Have Ready

### 1. Corrected MLIR Kernel Files ✅

Located in: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/`

- `mlir_aie2_simple_test.mlir` - 69 lines, simple passthrough test
- `mlir_aie2_minimal_corrected.mlir` - Minimal kernel using `npu1`
- `mlir_aie2_minimal_1col.mlir` - Single-column partition version

**Key Features**:
- Uses correct device name: `aie.device(npu1)`
- Proper tile layout for Phoenix NPU (4×6 array)
- ObjectFIFO for data movement
- Runtime sequence with DMA configuration
- Based on working Xilinx examples

**Example** (`mlir_aie2_simple_test.mlir`):
```mlir
module @simple_test {
  aie.device(npu1) {
    // Shim tile for DMA
    %tile00 = aie.tile(0, 0)

    // Compute tile
    %tile02 = aie.tile(0, 2)

    // ObjectFIFOs with 512-byte buffers
    aie.objectfifo @inOF(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<512xui8>>
    aie.objectfifo @outOF(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<512xui8>>

    // Core computation
    %core02 = aie.core(%tile02) {
      // Process data here
      func.call @simpleKernel(%elemIn, %elemOut, %c512)
      aie.end
    }

    // Runtime DMA sequence
    aiex.runtime_sequence(...) { ... }
  }
}
```

### 2. MLIR-AIE Toolchain Binaries ✅

Located in: `/home/ucadmin/mlir-aie-prebuilt/mlir_aie/bin/`

**Available Tools**:
- `aie-opt` (139 MB) - MLIR optimizer with 500+ AIE passes
- `aie-translate` (55 MB) - MLIR to binary translator
- `aiecc.py` - Compilation orchestrator (needs Python deps)
- `aie-visualize` - Design visualization
- `aie-lsp-server` - Language server

**Status**: Binaries work, but `aiecc.py` requires Python modules.

### 3. NPU Hardware Verified ✅

```bash
$ /opt/xilinx/xrt/bin/xrt-smi examine

Device(s) Present
|BDF             |Name         |
|----------------|-------------|
|[0000:c7:00.1]  |NPU Phoenix  |

XRT Version: 2.20.0
NPU Firmware: 1.5.5.391
Device Node: /dev/accel/accel0
```

### 4. Documentation Complete ✅

- `MLIR_COMPILATION_REPORT.md` (1000+ lines) - Comprehensive analysis
- `NPU_OPTIMIZATION_STRATEGY.md` (28,000 words) - Full strategy
- `COMPILATION_STATUS.md` - Quick reference
- Platform configuration verified (npu1, 4×6 tile array)
- Working examples analyzed from Xilinx repository

---

## Solution Options

### Option A: Fix Python Dependencies (RECOMMENDED)

**Approach**: Create symlink or fix module structure

**Steps**:
```bash
# Option 1: Create symlink for missing util module
cd /home/ucadmin/mlir-aie-prebuilt/mlir_aie/python/aie/extras/
ln -s ../util.py util.py

# Test
export PYTHONPATH="/home/ucadmin/mlir-aie-prebuilt/mlir_aie/python:$PYTHONPATH"
python3 -c "import aie; from aie.iron import Device; print('✓ Working')"

# If successful, compile test kernel
export PATH="/home/ucadmin/mlir-aie-prebuilt/mlir_aie/bin:$PATH"
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization
aiecc.py --aie-generate-xclbin --xclbin-name=test.xclbin mlir_aie2_simple_test.mlir
```

**Estimated Time**: 30 minutes

**Pros**:
- Uses existing infrastructure
- No additional downloads
- Can test immediately

**Cons**:
- Might break with updates
- Hacky solution

### Option B: Docker with Authentication

**Approach**: Authenticate to GitHub Container Registry and pull image

**Steps**:
```bash
# Create GitHub Personal Access Token with read:packages scope
# Then authenticate Docker:
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Pull image
docker pull ghcr.io/xilinx/mlir-aie:latest

# Run compilation in container
docker run --rm -v $(pwd):/work ghcr.io/xilinx/mlir-aie:latest \
  aiecc.py --aie-generate-xclbin --xclbin-name=test.xclbin /work/mlir_aie2_simple_test.mlir
```

**Estimated Time**: 45 minutes (with token setup)

**Pros**:
- Official solution
- All dependencies included
- Clean environment

**Cons**:
- Requires GitHub token
- Larger download (~2GB)
- Container overhead

### Option C: Install from Source

**Approach**: Build MLIR-AIE from source with all dependencies

**Steps**:
```bash
git clone https://github.com/Xilinx/mlir-aie.git
cd mlir-aie
git submodule update --init --recursive

# Install dependencies
sudo apt install -y cmake ninja-build clang lld

# Build (takes 1-2 hours)
mkdir build && cd build
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON
ninja

# Install Python package
cd python && pip install -e .
```

**Estimated Time**: 2-3 hours

**Pros**:
- Complete control
- Latest version
- All Python modules included

**Cons**:
- Long build time
- Complex dependencies
- Requires disk space (~10GB)

### Option D: Use Pre-compiled Binaries Without Python (ALTERNATIVE)

**Approach**: Bypass Python API, use IRON Python scripts or raw MLIR only

**Steps**:
```bash
# Skip aiecc.py, use tools directly
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization

export PATH="/home/ucadmin/mlir-aie-prebuilt/mlir_aie/bin:$PATH"

# Step 1: Lower MLIR to AIE dialect
aie-opt --aie-lower-to-aie \
        --aie-assign-tile-ids \
        mlir_aie2_simple_test.mlir -o lowered.mlir

# Step 2: Generate XCLBIN
aie-translate --aie-generate-xclbin lowered.mlir -o test.xclbin
```

**Estimated Time**: 15 minutes

**Pros**:
- No Python dependencies needed
- Direct use of binaries
- Fast to test

**Cons**:
- May not work for complex kernels
- Limited to specific MLIR passes
- Peano compiler might be needed

---

## Recommendation

**Immediate Next Step**: Try **Option A** (Python fix) first - it's fastest and uses what we have.

If that fails, try **Option D** (direct binaries) as it avoids Python entirely.

If both fail, proceed with **Option B** (Docker) after getting GitHub token from user.

**Rationale**:
- Option A: 30 min investment, high success probability
- Option D: 15 min investment, moderate success probability
- Option B: 45 min investment, guaranteed success
- Option C: 2-3 hours investment, only if others fail

---

## Expected Outcomes After Unblocking

Once compilation works:

**Phase 1** (1 week):
1. Compile simple test kernel → `test.xclbin`
2. Load and run on NPU via XRT
3. Verify NPU execution (not just CPU fallback)
4. **Target**: Confirm NPU is actually processing data

**Phase 2** (2 weeks):
1. Compile mel spectrogram kernel
2. Integrate into preprocessing pipeline
3. Replace librosa CPU code with NPU kernel
4. **Target**: 20-30x realtime (from current 5.2x)

**Phase 3** (2-3 weeks):
1. Compile matrix multiply kernel
2. Use for encoder/decoder operations
3. Replace ONNX Runtime matmul
4. **Target**: 60-80x realtime

**Phase 4** (4-6 weeks):
1. Implement full encoder on NPU
2. Implement full decoder on NPU
3. End-to-end NPU pipeline
4. **Target**: 200-220x realtime

---

## Files Reference

**Kernel Files**:
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mlir_aie2_simple_test.mlir`
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mlir_aie2_minimal_corrected.mlir`
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mlir_aie2_minimal_1col.mlir`

**Documentation**:
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/MLIR_COMPILATION_REPORT.md`
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/NPU_OPTIMIZATION_STRATEGY.md`
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/COMPILATION_STATUS.md`

**Toolchain**:
- `/home/ucadmin/mlir-aie-prebuilt/mlir_aie/bin/` - Binaries
- `/home/ucadmin/mlir-aie-prebuilt/mlir_aie/python/aie/` - Python modules (incomplete)

---

## Next Actions

1. **User Decision**: Choose which solution option to pursue (A, B, C, or D)
2. **Implement Fix**: Follow steps for chosen option
3. **Test Compilation**: Compile `mlir_aie2_simple_test.mlir` to `test.xclbin`
4. **Verify NPU Execution**: Load XCLBIN and run on NPU via XRT
5. **Proceed to Phase 2**: Compile actual Whisper kernels

---

**Status**: Ready to proceed once Python dependencies resolved or alternative approach selected.
**Blockers**: Python module structure issue, Docker access, or build from source required.
**Impact**: Critical path to achieving 220x NPU acceleration target.
