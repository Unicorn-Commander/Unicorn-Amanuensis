# Quick Start: Optimized Mel Kernel on NPU

**Goal**: Run optimized mel filterbank kernel on AMD Phoenix NPU for 25-30% better accuracy

---

## Step 1: Build the Kernel (✅ COMPLETE)

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/
bash compile_mel_optimized.sh
```

**Expected output:**
```
✅ Compilation complete!
✅ mel_kernel_simple symbol found
✅ Helper function symbols found
```

**Output file**: `mel_optimized_combined.o` (53 KB)

---

## Step 2: Create MLIR File (TODO)

Copy the working MLIR template and update the link_with attribute:

```bash
# Copy working MLIR file
cp build_fixed/mel_fixed.mlir mel_optimized.mlir

# Update to link with optimized kernel
sed -i 's/mel_fixed_combined.o/mel_optimized_combined.o/g' mel_optimized.mlir
```

**Verify the change:**
```bash
grep "link_with" mel_optimized.mlir
# Should show: { link_with = "mel_optimized_combined.o" }
```

---

## Step 3: Generate XCLBIN (TODO)

Compile the MLIR file to XCLBIN using aiecc.py:

```bash
aiecc.py mel_optimized.mlir
```

**Expected output**: `mel_optimized.xclbin` (should be ~8-16 KB)

**If you get errors**, check:
- MLIR file syntax: `aie-opt --verify-diagnostics mel_optimized.mlir`
- Archive exists: `ls -lh mel_optimized_combined.o`
- Symbols present: `llvm-nm mel_optimized_combined.o | grep mel_kernel_simple`

---

## Step 4: Test on NPU (TODO)

Load and test the XCLBIN on actual NPU hardware:

```python
import xrt
import numpy as np

# Load NPU device
device = xrt.xrt_device(0)  # /dev/accel/accel0

# Load XCLBIN
xclbin_path = "mel_optimized.xclbin"
device.load_xclbin(xclbin_path)

# Create test audio (1 second @ 16kHz, 400 samples for 25ms frame)
audio_samples = np.random.randint(-32768, 32767, size=400, dtype=np.int16)
audio_bytes = audio_samples.tobytes()  # 800 bytes

# Allocate NPU buffers
input_bo = xrt.bo(device, 800, xrt.bo.flags.device_only, 0)
output_bo = xrt.bo(device, 80, xrt.bo.flags.device_only, 0)

# Copy input to NPU
input_bo.write(audio_bytes, 0)
input_bo.sync(xrt.xclDirection.HOST_TO_DEVICE, 800, 0)

# Run kernel
kernel = xrt.kernel(device, device.get_xclbin_uuid(), "mel_kernel_simple")
run = kernel(input_bo, output_bo)
run.wait()

# Read output from NPU
output_bo.sync(xrt.xclDirection.DEVICE_TO_HOST, 80, 0)
mel_features = np.frombuffer(output_bo.read(80, 0), dtype=np.int8)

print(f"Input: {len(audio_bytes)} bytes")
print(f"Output: {len(mel_features)} mel bins")
print(f"Mel features: {mel_features}")
```

---

## Step 5: Benchmark Performance (TODO)

Run full benchmark to measure NPU performance:

```python
import time
import numpy as np

# Load 30 seconds of audio (1200 frames of 25ms)
num_frames = 1200
total_audio_duration = 30.0  # seconds

# Measure NPU processing time
start = time.time()

for frame_idx in range(num_frames):
    # Generate or load 400 samples (25ms @ 16kHz)
    audio_samples = np.random.randint(-32768, 32767, size=400, dtype=np.int16)
    audio_bytes = audio_samples.tobytes()
    
    # Run on NPU (same as Step 4)
    input_bo.write(audio_bytes, 0)
    input_bo.sync(xrt.xclDirection.HOST_TO_DEVICE, 800, 0)
    run = kernel(input_bo, output_bo)
    run.wait()
    output_bo.sync(xrt.xclDirection.DEVICE_TO_HOST, 80, 0)
    mel_features = np.frombuffer(output_bo.read(80, 0), dtype=np.int8)

end = time.time()
processing_time = end - start

# Calculate realtime factor
rtf = total_audio_duration / processing_time
speedup = rtf

print(f"Processed {total_audio_duration}s of audio in {processing_time:.2f}s")
print(f"Realtime factor: {rtf:.1f}x")
print(f"Speedup: {speedup:.1f}x faster than realtime")
print(f"Target: 220x (Current: {speedup:.1f}x)")
```

**Expected performance**: 200-220x realtime (process 30s audio in ~0.13s)

---

## Step 6: Validate Accuracy (TODO)

Compare NPU mel features with CPU librosa output:

```python
import librosa
import numpy as np

# Generate test audio
sr = 16000
duration = 0.025  # 25ms
audio = np.random.randn(int(sr * duration))

# CPU reference (librosa)
mel_spec_cpu = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=512,
    hop_length=160,
    n_mels=80,
    fmin=0,
    fmax=8000
)
mel_features_cpu = mel_spec_cpu[:, 0]  # First frame

# NPU output (from Step 4)
mel_features_npu = mel_features  # From NPU

# Compare
correlation = np.corrcoef(mel_features_cpu, mel_features_npu)[0, 1]
print(f"Correlation: {correlation:.4f}")
print(f"Target: >0.95 (high accuracy)")

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(mel_features_cpu, label='CPU (librosa)')
plt.title('CPU Mel Features')
plt.subplot(1, 2, 2)
plt.plot(mel_features_npu, label='NPU (optimized)')
plt.title('NPU Mel Features')
plt.tight_layout()
plt.savefig('mel_comparison.png')
print("Saved: mel_comparison.png")
```

---

## Troubleshooting

### Build fails
```bash
# Check Peano compiler
ls -l /home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie/bin/clang++

# Check source files
ls -l mel_kernel_fft_optimized.c fft_fixed_point.c mel_filterbank_coeffs.h
```

### XCLBIN generation fails
```bash
# Validate MLIR syntax
aie-opt --verify-diagnostics mel_optimized.mlir

# Check for compilation errors in aiecc.py output
# Look for "error:" messages
```

### NPU kernel doesn't execute
```bash
# Check NPU device
ls -l /dev/accel/accel0
/opt/xilinx/xrt/bin/xrt-smi examine

# Check XCLBIN is valid
file mel_optimized.xclbin
# Should show: "data" or similar (binary file)

# Enable XRT debug logging
export XRT_LOG_LEVEL=debug
# Re-run Python test
```

### Performance is low (<100x)
- Check if kernel is actually running on NPU (not falling back to CPU)
- Verify buffer sizes match (800 bytes in, 80 bytes out)
- Check for DMA overhead (use batch processing)
- Profile with XRT tools: `xbutil top`

---

## Success Criteria

- ✅ Build completes without errors
- ✅ XCLBIN generates successfully
- ✅ NPU executes kernel without errors
- ✅ Output mel features are valid (80 INT8 values)
- ✅ Performance: 200-220x realtime
- ✅ Accuracy: >0.95 correlation with librosa

---

## Next Steps After Validation

1. **Integrate with WhisperX pipeline**
   - Replace CPU mel spectrogram with NPU version
   - Batch multiple audio frames for efficiency
   
2. **Optimize DMA transfers**
   - Use double-buffering
   - Pipeline compute and transfer
   
3. **Full pipeline on NPU**
   - Add Whisper encoder kernel
   - Add Whisper decoder kernel
   - Target: 220x realtime end-to-end

---

**Current Status**: ✅ Step 1 complete (build successful)
**Next**: Complete Step 2 (create MLIR file)
**Location**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/`
