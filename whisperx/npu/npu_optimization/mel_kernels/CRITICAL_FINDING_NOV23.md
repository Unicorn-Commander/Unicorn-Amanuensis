# CRITICAL FINDING: NPU Mel Kernels Output All Zeros

**Date**: November 23, 2025
**Status**: ❌ BLOCKING BUG DISCOVERED
**Severity**: CRITICAL - Prevents Path A NPU acceleration

---

## Summary

Both `mel_batch20.xclbin` and `mel_fixed_v3.xclbin` NPU mel spectrogram kernels **output all zeros** from the NPU hardware. This has been consistently observed across all testing sessions since November 1, 2025.

## Evidence

### Test Results (November 23, 2025)

```
Testing: build_batch20/mel_batch20.xclbin
Audio: 176000 samples (11.00s)

NPU output (INT8): min=0, max=0, mean=0.00
First 10 values: [0 0 0 0 0 0 0 0 0 0]
Processed 1098 frames in 2.6213s (4.20x realtime)
```

### Historical Evidence

From `BUG_FIX_REPORT_NOV22.md`:
```
INFO:npu_mel_preprocessing:   NPU output (INT8): min=0, max=0, mean=0.00
```

From `/tmp/server_fresh_mel.log` (November 23, 2025):
```
INFO:npu_mel_preprocessing:Processing 176000 samples (11.00s) into 1098 frames
INFO:npu_mel_preprocessing:   NPU output (INT8): min=0, max=0, mean=0.00
INFO:npu_mel_preprocessing:   First 10 values: [0 0 0 0 0 0 0 0 0 0]
```

## Root Cause Analysis

### What's NOT the Problem

❌ **Kernel source code** - The C code in `fft_fixed_point.c` and `mel_kernel_fft_fixed.c` has been reviewed and fixed multiple times
❌ **Scaling factors** - Fixed from >>30 to >>12 on November 22
❌ **XCLBIN selection** - Both batch20 and fixed_v3 show identical zero output
❌ **Python normalization** - The zeros come from NPU hardware, not post-processing

### What IS the Problem

✅ **One of the following**:

1. **DMA Configuration Issue**
   - Input buffer not being transferred to NPU (HOST → DEVICE)
   - Output buffer not being transferred from NPU (DEVICE → HOST)
   - Buffer group IDs incorrect (currently using 1, 3, 4)

2. **Instruction Binary Problem**
   - `insts.bin` / `insts_batch20.bin` not correctly encoded
   - NPU not executing the kernel despite `ERT_CMD_STATE_COMPLETED` status
   - Instruction opcode (3) may be incorrect

3. **Kernel Execution Issue**
   - Kernel loads but doesn't execute on NPU tiles
   - AIE2 tile configuration mismatch
   - Memory addressing issues in MLIR→XCLBIN compilation

4. **Hardware/Runtime Bug**
   - XRT 2.20.0 issue with Phoenix NPU
   - Firmware 1.5.5.391 compatibility problem
   - NPU tile initialization failure

## Code Location

**File**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_mel_preprocessing.py`

**Critical section** (lines 209-232):
```python
# Allocate buffers
instr_bo = xrt.bo(self.device, n_insts, xrt.bo.flags.cacheable, self.kernel.group_id(1))
input_bo = xrt.bo(self.device, input_size, xrt.bo.flags.host_only, self.kernel.group_id(3))
output_bo = xrt.bo(self.device, output_size, xrt.bo.flags.host_only, self.kernel.group_id(4))

# Write instructions
instr_bo.write(insts_bin, 0)
instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

# Write input data
input_data = frame_int16.tobytes()
input_bo.write(input_data, 0)
input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, input_size, 0)

# Execute kernel
opcode = 3  # NPU execution opcode
run = self.kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
state = run.wait(1000)  # 1 second timeout

if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
    raise RuntimeError(f"NPU kernel execution failed: {state}")

# Read output
output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, output_size, 0)
mel_bins = np.frombuffer(output_bo.read(output_size, 0), dtype=np.int8)
```

## Impact on Dual-Path Strategy

### Path A: ❌ BLOCKED

**faster-whisper + NPU mel kernel**
- Mel kernel outputs zeros → no usable features → transcription fails
- **Cannot achieve 28-30x target** without working mel kernel
- **Recommendation**: Disable NPU mel preprocessing until bug is fixed

### Path B: ✅ UNAFFECTED

**OpenAI Whisper + NPU attention**
- Uses librosa CPU mel preprocessing
- NPU acceleration only for attention layers
- **Can still achieve 20-30x target** with attention optimization

## Working Baseline Performance

**Faster-whisper (CPU only)**: **19x realtime**

```
Audio: test_audio_jfk.wav (11.00s)
Model: Whisper base (INT8)
Device: CPU
Processing time: 0.58s
Realtime factor: 19.0x
Transcription: "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
```

**Conclusion**: CPU-only faster-whisper already exceeds the "13-16x baseline" mentioned in documentation.

## Immediate Next Steps

### Option 1: Debug Mel Kernel DMA (1-2 days)
- Add detailed logging to `_process_frame_npu()`
- Test with passthrough kernel (input → output copy)
- Verify buffer sync directions
- Check XRT runtime logs (`/var/log/syslog`)
- Test different buffer group IDs

### Option 2: Deploy CPU Baseline (RECOMMENDED)
- Deploy faster-whisper with 19x performance
- Focus on Path B (attention acceleration)
- Achieve 25-35x with NPU attention
- Fix mel kernel later as optimization

### Option 3: Investigate XRT/Firmware (2-4 days)
- Check XRT 2.20.0 release notes for known issues
- Test with different firmware versions
- Compare with working attention kernel execution
- Contact AMD for NPU debugging support

## Recommendation

**DEPLOY PATH B NOW**:
1. Use faster-whisper baseline (19x proven working)
2. Add NPU attention acceleration (target: 25-35x)
3. Skip mel kernel until DMA bug is resolved
4. Document mel kernel as "experimental, known issue"

**Benefits**:
- Immediate deployment with working solution
- Incremental performance improvement path
- Reduced risk vs. debugging hardware issue
- User gets value now, optimization later

---

**Status**: Mel kernel zero-output confirmed across all tests
**Action**: Recommend Path B deployment, defer Path A debugging
**Timeline**: Path B ready for testing TODAY

**Update November 24, 2025**:
- ✅ Path B tested - see ../PATH_B_TEST_RESULTS_NOV24.md
- ✅ Confirmed mel_batch30.xclbin ALSO outputs zeros (same DMA bug)
- ✅ Previous "99.73% coverage" was Python normalization artifacts, not real NPU output
- ⚠️ ALL mel kernels affected: batch10, batch20, batch30, fixed_v3, optimized
- 🔧 Root cause: DMA buffer configuration or instruction binary encoding issue
- 📊 faster-whisper baseline: 18.27x realtime (production-ready alternative)
