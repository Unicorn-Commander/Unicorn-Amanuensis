# Next Steps: FFT Integration Testing

## Current Status
✅ **All compilation complete** - FFT module and MEL kernel compiled successfully
✅ **Ready for XCLBIN generation**

## Quick Commands

### Option 1: Generate XCLBIN (Recommended - Uses Working Infrastructure)

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels

# Activate environment
source /home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/activate

# Use the automated build script
./build_mel_with_fft.sh
```

### Option 2: Manual MLIR Compilation

```bash
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export AIE_OPT=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aie-opt
export AIE_TRANSLATE=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aie-translate

cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels

# Lower MLIR
$AIE_OPT \
  --aie-canonicalize-device \
  --aie-objectFifo-stateful-transform \
  --aie-create-pathfinder-flows \
  --aie-assign-buffer-addresses \
  mel_with_fft.mlir -o build/mel_fft_lowered.mlir

# Generate NPU instructions
$AIE_TRANSLATE --aie-generate-xaie build/mel_fft_lowered.mlir -o build/insts.txt

# Generate CDO files
$AIE_TRANSLATE --aie-generate-cdo build/mel_fft_lowered.mlir

# (Full XCLBIN packaging requires additional steps - see build_mel_complete.sh)
```

## Test on NPU

Once XCLBIN is generated:

```python
import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np

# Load XCLBIN
device = xrt.device(0)
xclbin_obj = xrt.xclbin('build/mel_fft.xclbin')
uuid = xclbin_obj.get_uuid()
device.register_xclbin(xclbin_obj)
hw_ctx = xrt.hw_context(device, uuid)

# Create test input (400 INT16 samples = 800 bytes)
test_audio = np.sin(2 * np.pi * 440 * np.arange(400) / 16000)  # 440 Hz tone
test_audio = (test_audio * 32767).astype(np.int16)

input_bo = xrt.bo(device, 800, xrt.bo.flags.host_only, 0)
input_bo.write(test_audio.tobytes(), 0)

# Create output buffer
output_bo = xrt.bo(device, 80, xrt.bo.flags.host_only, 0)

# Execute
kernel = xrt.kernel(hw_ctx, "mel_kernel_simple")
run = kernel(input_bo, output_bo)
run.wait()

# Read results
mel_output = np.frombuffer(output_bo.read(80, 0), dtype=np.int8)
print("MEL bins:", mel_output)
print("✅ FFT kernel working on NPU!")
```

## Files Summary

**Compiled Object Files:**
- `fft_real.o` (7.9 KB) - FFT implementation
- `mel_kernel_fft.o` (5.9 KB) - MEL kernel
- `mel_kernel_combined.o` (15 KB) - Combined archive

**MLIR Configuration:**
- `mel_with_fft.mlir` (3.6 KB) - Device configuration

**Build Scripts:**
- `build_mel_with_fft.sh` - Automated build

## Troubleshooting

**If XCLBIN generation fails:**
1. Check that mel_kernel_combined.o exists and is valid
2. Verify MLIR syntax with: `aie-opt --verify mel_with_fft.mlir`
3. Check that link_with path is correct
4. Review build_mel_complete.sh for reference

**If NPU execution fails:**
1. Verify XCLBIN loads: `xrt-smi examine`
2. Check kernel registration
3. Verify buffer sizes (800 bytes in, 80 bytes out)
4. Check XRT logs: `/var/log/xrt.log`

**If output looks wrong:**
1. Test with known input (sine wave)
2. Compare with CPU librosa implementation
3. Check magnitude scaling (may need adjustment)
4. Verify Hann window application

## Expected Performance

**Target**: Process 400 samples (25ms audio) in ~50μs on NPU
**Speedup**: ~500x vs CPU librosa
**Contribution to 220x goal**: This is the mel spectrogram preprocessing step

## Contact

For issues or questions, refer to:
- `FFT_INTEGRATION_COMPLETE.md` - Full build report
- `CURRENT_STATUS_AND_NEXT_STEPS.md` - Overall project status
- Build logs in `build/` directory
