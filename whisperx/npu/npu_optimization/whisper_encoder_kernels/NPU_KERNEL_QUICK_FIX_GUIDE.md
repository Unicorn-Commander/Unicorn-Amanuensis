# NPU Kernel Quick Fix Guide

**Use this guide when any NPU kernel fails with `ERT_CMD_STATE_ERROR`**

---

## The Fix (Copy-Paste Template)

```python
# ============================================================
# STEP 1: Load instructions (ADD THIS!)
# ============================================================
insts_path = "build_YOUR_KERNEL/insts.bin"  # or main_sequence.bin
with open(insts_path, "rb") as f:
    insts = f.read()
n_insts = len(insts)

# ============================================================
# STEP 2: Allocate instruction buffer (ADD THIS!)
# ============================================================
instr_bo = xrt.bo(device, n_insts,
                  xrt.bo.flags.cacheable,   # MUST be cacheable
                  kernel.group_id(1))       # MUST be group_id(1)
instr_bo.write(insts, 0)
instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
              n_insts, 0)

# ============================================================
# STEP 3: Allocate input/output buffers (FIX THIS!)
# ============================================================
input_bo = xrt.bo(device, INPUT_SIZE,
                  xrt.bo.flags.host_only,   # MUST be host_only
                  kernel.group_id(3))       # MUST be group_id(3)

output_bo = xrt.bo(device, OUTPUT_SIZE,
                   xrt.bo.flags.host_only,  # MUST be host_only
                   kernel.group_id(4))      # MUST be group_id(4)

# ============================================================
# STEP 4: Fix kernel call (FIX THIS!)
# ============================================================
opcode = 3  # ADD THIS!
run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
#           ^^^^^^  ^^^^^^^^  ^^^^^^^  ^^^^^^^^  ^^^^^^^^^
#           ADD     ADD       ADD      EXISTING  EXISTING
```

---

## Common Errors and Fixes

### Error 1: `ERT_CMD_STATE_ERROR`

**Symptoms**:
```
❌ ERROR: kernel state ert_cmd_state.ERT_CMD_STATE_ERROR
```

**Fix**: Add instruction buffer (see template above)

---

### Error 2: "No compute units with connectivity"

**Symptoms**:
```
[XRT] WARNING: Kernel has no compute units with connectivity required
```

**Fix**: Use correct group_id values:
- group_id(1) for instructions
- group_id(3) for input
- group_id(4) for output

---

### Error 3: Kernel timeout or hang

**Symptoms**:
```
Kernel execution timeout
```

**Fix**:
1. Check XCLBIN path is correct
2. Check instruction path is correct
3. Increase timeout: `run.wait(5000)`

---

## Checklist (Use Before Running ANY Kernel)

### Files Present?
- [ ] XCLBIN file exists
- [ ] Instruction file exists (insts.bin or main_sequence.bin)
- [ ] NPU device accessible (`ls /dev/accel/accel0`)

### Code Correct?
- [ ] Instruction buffer allocated with `group_id(1)`
- [ ] Input buffer allocated with `group_id(3)`
- [ ] Output buffer allocated with `group_id(4)`
- [ ] Instruction buffer synced to device
- [ ] Kernel called with opcode (`opcode = 3`)
- [ ] All 5 arguments passed to kernel

### If Still Failing?
- [ ] Print buffer sizes to verify
- [ ] Check XRT version (`xrt-smi examine`)
- [ ] Test with working kernel (matmul_16x16)
- [ ] Compare with template in NPU_KERNEL_TESTING_TEMPLATE.md

---

## Buffer Group ID Reference

| Group ID | Purpose | Flags | Example Size |
|----------|---------|-------|--------------|
| **1** | Instructions | `cacheable` | 300 bytes |
| **3** | Input data | `host_only` | 512-12288 bytes |
| **4** | Output data | `host_only` | 256-4096 bytes |

**DO NOT USE OTHER VALUES ON PHOENIX NPU!**

---

## Kernel Call Signature

**WRONG**:
```python
run = kernel(input_bo, output_bo)  # ❌ Missing opcode and instructions!
```

**CORRECT**:
```python
opcode = 3
run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
#           arg1    arg2      arg3     arg4      arg5
```

**ALL 5 ARGUMENTS REQUIRED!**

---

## Working Example (Matmul 16×16)

```python
# Load XCLBIN and instructions
device = xrt.device(0)
xclbin = xrt.xclbin("build_matmul_fixed/matmul_16x16.xclbin")
uuid = xclbin.get_uuid()
device.register_xclbin(xclbin)
hw_ctx = xrt.hw_context(device, uuid)
kernel = xrt.kernel(hw_ctx, "MLIR_AIE")

with open("build_matmul_fixed/main_sequence.bin", "rb") as f:
    insts = f.read()
n_insts = len(insts)

# Allocate buffers
instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
instr_bo.write(insts, 0)
instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

input_bo = xrt.bo(device, 512, xrt.bo.flags.host_only, kernel.group_id(3))
output_bo = xrt.bo(device, 256, xrt.bo.flags.host_only, kernel.group_id(4))

# Prepare data
input_data = np.random.randint(-64, 64, 512, dtype=np.int8)
input_bo.write(input_data.tobytes(), 0)
input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 512, 0)

# Execute
opcode = 3
run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
state = run.wait(1000)

if state == xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
    print("✅ Success!")
else:
    print(f"❌ Failed: {state}")

# Read output
output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 256, 0)
output = np.frombuffer(output_bo.read(256, 0), dtype=np.int8)
print(f"Output: {output.reshape(16, 16)}")
```

---

## Quick Diagnostic Commands

```bash
# Check NPU device
ls -l /dev/accel/accel0

# Check XRT version
/opt/xilinx/xrt/bin/xrt-smi examine

# Check kernel files
ls -lh build_*/

# Test working kernel
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels
python3 test_matmul_16x16.py

# Test fixed attention kernel
python3 test_attention_64x64.py
```

---

## Need More Help?

1. **Read full documentation**:
   - ATTENTION_KERNEL_FIX_REPORT.md (detailed technical analysis)
   - NPU_KERNEL_TESTING_TEMPLATE.md (complete testing template)
   - ATTENTION_FIX_EXECUTIVE_SUMMARY.md (high-level overview)

2. **Compare with working kernel**:
   - test_matmul_16x16.py (proven working example)
   - test_attention_64x64.py (fixed and working)

3. **Check kernel compilation**:
   - Run compile script: `bash compile_YOUR_KERNEL.sh`
   - Verify XCLBIN generated: `ls -lh build_*/`
   - Check instructions: `ls -lh build_*/insts.bin`

---

**This guide saved the attention kernel in 45 minutes. Use it to save yours!**

---

**Created**: October 30, 2025
**Updated**: After successful attention_64x64 fix
**Version**: 1.0
