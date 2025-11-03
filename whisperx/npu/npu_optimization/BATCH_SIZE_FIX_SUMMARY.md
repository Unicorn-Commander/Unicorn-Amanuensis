# Batch Size Mismatch Fix - November 1, 2025

## Problem
Python code had `BATCH_SIZE = 100` but MLIR kernel only processes 10 frames, causing mismatch.

## Root Cause
**File**: `npu_mel_processor_batch_final.py`
- Line 66: `BATCH_SIZE = 100` (incorrect)
- Expected: `BATCH_SIZE = 10` (matches mel_batch10.xclbin)

## Solution Implemented

### 1. Batch Size Fixed ✅
**Change**: Line 66
```python
# Before:
BATCH_SIZE = 100  # Fixed batch size matching MLIR kernel

# After:
BATCH_SIZE = 10  # Fixed batch size matching MLIR kernel
```

### 2. Buffer Sizes Auto-Updated ✅
The buffer sizes are **dynamically calculated** from `BATCH_SIZE`, so they automatically updated:

**Line 116-117**:
```python
# Before (with BATCH_SIZE=100):
self.input_buffer_size = self.BATCH_SIZE * self.FRAME_SIZE * 2  # 80,000 bytes
self.output_buffer_size = self.BATCH_SIZE * self.N_MELS         # 8,000 bytes

# After (with BATCH_SIZE=10):
self.input_buffer_size = self.BATCH_SIZE * self.FRAME_SIZE * 2  # 8,000 bytes (batch-10)
self.output_buffer_size = self.BATCH_SIZE * self.N_MELS         # 800 bytes (batch-10)
```

**Calculated values**:
- Input buffer: 10 frames × 400 samples × 2 bytes = **8,000 bytes (8 KB)**
- Output buffer: 10 frames × 80 mel bins = **800 bytes**

### 3. Documentation Updated ✅
Updated all references to batch-100 → batch-10:
- Module docstring (lines 2-25)
- Class docstring (lines 46-63)
- Method docstrings
- Comments throughout
- Default XCLBIN path: `build_batch100` → `build_batch10`
- Log messages: "Batch-100 Mode" → "Batch-10 Mode"

### 4. Server Restarted ✅
```bash
pkill -f server_dynamic
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
nohup python3 server_dynamic.py > /tmp/server_batch10_fixed.log 2>&1 &
```

**Server logs confirm**:
```
INFO:npu_runtime_unified:  [✓] Mel kernel loaded (BATCH-10 MODE): mel_batch10.xclbin
INFO:npu_runtime_unified:      Batch size: 10 frames per NPU call
INFO:npu_runtime_unified:      Expected speedup: 715x realtime
```

## Expected Performance Improvement

### Before Fix (Mismatch)
- Python tried to process 100 frames
- Kernel only processed 10 frames
- **90% data loss** per batch
- Performance: ~348x realtime (limited by mismatch)

### After Fix (Aligned)
- Python processes 10 frames
- Kernel processes 10 frames
- **100% data processed**
- Expected performance: **600-700x realtime**

**Performance gain**: ~2x improvement (348x → 700x)

## Files Modified

### Primary Changes
1. **npu_mel_processor_batch_final.py** (616 lines)
   - Backup created: `npu_mel_processor_batch_final.py.backup_batch_fix`
   - Line 66: BATCH_SIZE = 100 → 10
   - All docstrings and comments updated
   - Default XCLBIN path corrected

### Server Integration
2. **server_dynamic.py** (uses npu_runtime_unified.py)
   - No changes needed (automatically uses updated processor)
   - Server logs show "BATCH-10 MODE" correctly

## Verification

### Configuration Verified ✅
```bash
$ grep "BATCH_SIZE = " npu_mel_processor_batch_final.py
    BATCH_SIZE = 10  # Fixed batch size matching MLIR kernel
```

### Server Status ✅
```bash
$ ps aux | grep server_dynamic | grep -v grep
ucadmin    71311  1.2  0.5 1351780 461304 ?      Sl   18:06   0:01 python3 server_dynamic.py

$ curl http://localhost:9004/status | jq '.npu_runtime'
{
  "available": true,
  "mel_ready": true,
  "gelu_ready": true,
  "attention_ready": true
}
```

### Kernel Loaded ✅
Server initialization shows:
```
INFO:npu_runtime_unified:  [✓] Mel kernel loaded (BATCH-10 MODE): mel_batch10.xclbin
INFO:npu_runtime_unified:      Batch size: 10 frames per NPU call
INFO:npu_runtime_unified:      Expected speedup: 715x realtime
INFO:npu_runtime_unified:      Accuracy: >0.95 correlation with librosa
```

## Next Steps for Full Performance

To achieve the full 600-700x realtime performance:

1. **Run Performance Benchmark** (not yet done)
   - Process full audio file (e.g., 1h 44m test case)
   - Measure actual mel processing time
   - Verify 2x improvement over previous 348x

2. **Monitor Mel Processing Logs**
   - Check for "mel completed in X.XXXs" messages
   - Calculate realtime factor: audio_duration / processing_time
   - Target: >600x realtime

3. **Test with Various Audio Lengths**
   - Short (10-30s): Verify overhead minimal
   - Medium (1-5min): Verify consistent performance
   - Long (1h+): Verify sustained performance

## Technical Details

### Why Batch Size Matters
- **NPU Kernel**: Processes exactly 10 frames per invocation
- **Python Code**: Must match kernel batch size
- **Mismatch Impact**: Wasted NPU cycles, data loss, incorrect performance

### Memory Layout
```
Input Buffer (8KB):
  Frame 0: [400 samples × int16] = 800 bytes
  Frame 1: [400 samples × int16] = 800 bytes
  ...
  Frame 9: [400 samples × int16] = 800 bytes
  Total: 10 × 800 = 8,000 bytes

Output Buffer (800B):
  Frame 0: [80 mel bins × int8] = 80 bytes
  Frame 1: [80 mel bins × int8] = 80 bytes
  ...
  Frame 9: [80 mel bins × int8] = 80 bytes
  Total: 10 × 80 = 800 bytes
```

### Processing Flow
1. Python frames audio into 10-frame batches
2. Converts to int16 and writes to 8KB input buffer
3. Calls NPU kernel once per batch
4. NPU processes all 10 frames in parallel
5. Reads 800B output buffer
6. Converts int8 mel bins to float32
7. Repeats for next batch

## Summary

✅ **Batch size fixed**: 100 → 10
✅ **Buffer sizes updated**: 80KB/8KB → 8KB/800B
✅ **Documentation corrected**: All references to batch-10
✅ **Server restarted**: Logs confirm BATCH-10 MODE
✅ **Configuration verified**: NPU runtime initialized correctly

**Expected Result**: 2x performance improvement (348x → 700x realtime)

**Status**: Ready for performance benchmarking

---

**Fixed by**: Claude Code
**Date**: November 1, 2025
**Time**: 18:06 UTC
**Session**: Batch Size Mismatch Fix
