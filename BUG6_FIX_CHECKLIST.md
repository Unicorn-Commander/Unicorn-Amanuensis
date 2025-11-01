# Bug #6 Fix: Implementation Checklist
## NPU Callback Registration - Quick Reference

**Estimated Time**: 4 hours
**Difficulty**: 🟢 Low
**Impact**: 🔴 Critical
**Files to Modify**: 3

---

## Quick Overview

**Problem**: NPU callback infrastructure exists but is never wired to C++ encoder

**Solution**: Add 3 missing integration pieces:
1. Expose C++ callback function to Python
2. Wire callback to encoder layers
3. Load XRT app and register at startup

---

## Task Breakdown

### Task 1: Update cpp_runtime_wrapper.py (1 hour)

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp_runtime_wrapper.py`

**Location**: After `forward()` method (around line 350)

**Add Method**:
```python
def set_npu_callback(
    self,
    layer_handle: EncoderLayerHandle,
    callback_fn: callable,
    user_data: Optional[ctypes.c_void_p] = None
) -> bool:
    """Set NPU callback for encoder layer."""

    if not self.lib:
        raise CPPRuntimeError("Library not loaded")

    # Define callback type
    NPUMatmulCallback = ctypes.CFUNCTYPE(
        ctypes.c_int,  # return type
        ctypes.POINTER(ctypes.c_float),  # A matrix
        ctypes.POINTER(ctypes.c_float),  # B matrix
        ctypes.POINTER(ctypes.c_float),  # C matrix (output)
        ctypes.c_int,  # m
        ctypes.c_int,  # k
        ctypes.c_int,  # n
        ctypes.c_void_p  # user_data
    )

    # Define C function signature
    self.lib.encoder_layer_set_npu_callback.argtypes = [
        ctypes.c_void_p,  # EncoderLayerHandle
        NPUMatmulCallback,  # callback
        ctypes.c_void_p  # user_data
    ]
    self.lib.encoder_layer_set_npu_callback.restype = ctypes.c_int

    # Convert Python callback to C callback
    c_callback = NPUMatmulCallback(callback_fn)

    # Call C function
    result = self.lib.encoder_layer_set_npu_callback(
        layer_handle,
        c_callback,
        user_data if user_data else ctypes.c_void_p(0)
    )

    if result != 0:
        raise CPPRuntimeError(f"Failed to set NPU callback (error {result})")

    return True
```

**Test**:
```python
# In Python console
from xdna2.cpp_runtime_wrapper import CPPRuntimeWrapper
runtime = CPPRuntimeWrapper()
# Should have set_npu_callback method
assert hasattr(runtime, 'set_npu_callback')
```

---

### Task 2: Update encoder_cpp.py (30 minutes)

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/encoder_cpp.py`

**Location**: Replace `register_npu_callback()` method (around line 303)

**Update Method**:
```python
def register_npu_callback(self, npu_app: Any) -> bool:
    """
    Register NPU callback with XRT application.

    Args:
        npu_app: Loaded NPU application from XRT

    Returns:
        True if registration successful, False otherwise
    """
    if not self.npu_callback:
        logger.warning("NPU callback not initialized")
        return False

    try:
        # Step 1: Register callback with XRT application
        success = self.npu_callback.register_with_encoder(npu_app)
        if not success:
            logger.error("Failed to register callback with XRT app")
            return False

        # Step 2: Get callback function pointer
        callback_fn = self.npu_callback.get_callback_fn()
        if not callback_fn:
            logger.error("Failed to get callback function pointer")
            return False

        # Step 3: Wire callback to each encoder layer
        logger.info("[EncoderCPP] Wiring NPU callback to layers...")
        for i, layer_handle in enumerate(self.layers):
            try:
                self.runtime.set_npu_callback(layer_handle, callback_fn)
                logger.debug(f"  Layer {i}: callback registered")
            except Exception as e:
                logger.error(f"Failed to set callback for layer {i}: {e}")
                return False

        logger.info("[EncoderCPP] NPU callback registered for all layers")
        return True

    except Exception as e:
        logger.error(f"Failed to register NPU callback: {e}")
        return False
```

**Test**:
```python
# After creating encoder
assert encoder.register_npu_callback(npu_app) == True
```

---

### Task 3: Update server.py (1.5 hours)

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/server.py`

**Part A: Add XRT Loader Function** (before `initialize_encoder()`)

```python
def load_xrt_npu_application():
    """
    Load XRT NPU application for Whisper encoder.

    Returns:
        Loaded NPU application object

    Raises:
        Exception: If XRT loading fails
    """
    try:
        # Import XRT
        import pyxrt

        # Find .xclbin file
        xclbin_path = Path(__file__).parent / "cpp" / "build" / "whisper_encoder.xclbin"
        if not xclbin_path.exists():
            # Try alternative locations
            alt_paths = [
                Path("/opt/xilinx/xrt/kernels/whisper_encoder.xclbin"),
                Path(__file__).parent.parent / "kernels" / "whisper_encoder.xclbin",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    xclbin_path = alt_path
                    break
            else:
                raise FileNotFoundError(
                    f"Cannot find whisper_encoder.xclbin in expected locations"
                )

        logger.info(f"  Loading XRT from: {xclbin_path}")

        # Load XRT device
        device = pyxrt.device(0)
        uuid = device.load_xclbin(str(xclbin_path))

        # Create kernel handle
        app = pyxrt.kernel(device, uuid, "matmul_bfp16")

        logger.info("  XRT application loaded successfully")
        return app

    except ImportError:
        raise ImportError(
            "pyxrt not found. Install with: pip install /opt/xilinx/xrt/python/pyxrt-*.whl"
        )
    except Exception as e:
        logger.error(f"XRT loading failed: {e}")
        raise
```

**Part B: Update initialize_encoder()** (after weight loading, around line 175)

```python
def initialize_encoder():
    global cpp_encoder, python_decoder, feature_extractor, conv1d_preprocessor, model_a, metadata

    try:
        # ... existing code ...

        cpp_encoder.load_weights(weights)
        logger.info("  Weights loaded successfully")

        # ========== NEW: NPU Callback Registration ==========
        if cpp_encoder.use_npu:
            logger.info("[Init] Loading XRT NPU application...")
            try:
                npu_app = load_xrt_npu_application()
                logger.info("  XRT NPU application loaded")

                # Register NPU callback
                logger.info("[Init] Registering NPU callback...")
                if cpp_encoder.register_npu_callback(npu_app):
                    logger.info("  ✅ NPU callback registered successfully")
                else:
                    logger.error("  ❌ NPU callback registration failed")
                    logger.warning("  Falling back to CPU mode")
                    cpp_encoder.use_npu = False

            except Exception as e:
                logger.error(f"Failed to load XRT NPU application: {e}")
                logger.warning("  Falling back to CPU mode")
                cpp_encoder.use_npu = False
        # ===================================================

        # Initialize conv1d preprocessor (Bug #5 fix)
        # ... rest of code ...
```

**Test**:
```bash
# Start service and check logs
python -m uvicorn xdna2.server:app --host 127.0.0.1 --port 9050

# Should see:
# INFO:xdna2.server:[Init] Loading XRT NPU application...
# INFO:xdna2.server:  XRT application loaded successfully
# INFO:xdna2.server:[Init] Registering NPU callback...
# INFO:xdna2.encoder_cpp:[EncoderCPP] Wiring NPU callback to layers...
# INFO:xdna2.encoder_cpp:  ✅ NPU callback registered for all layers
```

---

### Task 4: Testing (1 hour)

**Test 1: Service Startup**
```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis
source /home/ccadmin/mlir-aie/ironenv/bin/activate
export PATH="/home/ccadmin/local/bin:$PATH"

# Start service
ENABLE_PIPELINE=false python -m uvicorn xdna2.server:app \
  --host 127.0.0.1 --port 9050 --log-level info

# Expected output:
# ✅ XRT application loaded
# ✅ NPU callback registered for all layers
# ✅ All systems initialized successfully
```

**Test 2: Health Check**
```bash
curl http://localhost:9050/health | jq
# Should return 200 OK with encoder stats
```

**Test 3: Basic Transcription**
```bash
cd tests
curl -X POST http://localhost:9050/v1/audio/transcriptions \
  -F "file=@audio/test_1s.wav" \
  | jq

# Expected:
# {
#   "text": "...",
#   "segments": [...],
#   "language": "en"
# }
# NO "NPU callback not set" error!
```

**Test 4: Performance**
```bash
# Measure latency
time curl -X POST http://localhost:9050/v1/audio/transcriptions \
  -F "file=@audio/test_1s.wav" -o /tmp/result.json

# Expected: < 100ms for 1s audio (400x realtime)
```

**Test 5: Integration Tests**
```bash
cd tests
pytest test_pipeline_integration.py -v

# Expected: All tests pass
```

---

## Validation Checklist

After implementing all changes:

- [ ] `cpp_runtime_wrapper.py` has `set_npu_callback()` method
- [ ] `encoder_cpp.py` wires callback to all layers
- [ ] `server.py` loads XRT app and registers callback
- [ ] Service starts without errors
- [ ] Logs show "NPU callback registered for all layers"
- [ ] Health endpoint returns 200 OK
- [ ] Basic transcription works (1s audio)
- [ ] No "NPU callback not set" errors
- [ ] Latency < 100ms for 1s audio
- [ ] Integration tests pass

---

## Expected Logs (Success)

```
INFO:xdna2.server:  Initialization Complete
INFO:xdna2.server:[Init] Loading XRT NPU application...
INFO:xdna2.server:  Loading XRT from: /path/to/whisper_encoder.xclbin
INFO:xdna2.server:  XRT application loaded successfully
INFO:xdna2.server:[Init] Registering NPU callback...
INFO:xdna2.encoder_cpp:[EncoderCPP] Wiring NPU callback to layers...
INFO:xdna2.encoder_cpp:  Layer 0: callback registered
INFO:xdna2.encoder_cpp:  Layer 1: callback registered
INFO:xdna2.encoder_cpp:  Layer 2: callback registered
INFO:xdna2.encoder_cpp:  Layer 3: callback registered
INFO:xdna2.encoder_cpp:  Layer 4: callback registered
INFO:xdna2.encoder_cpp:  Layer 5: callback registered
INFO:xdna2.encoder_cpp:  ✅ NPU callback registered for all layers
INFO:xdna2.server:✅ All systems initialized successfully!

# First request:
INFO:xdna2.server:[Request 1] Processing: test_1s.wav
INFO:xdna2.server:  [1/5] Loading audio...
INFO:xdna2.server:  [2/5] Computing mel spectrogram...
INFO:xdna2.server:  [2.5/5] Applying conv1d preprocessing...
INFO:xdna2.server:  [3/5] Running C++ encoder (NPU)...
INFO:xdna2.server:    Encoder time: 48.2ms
INFO:xdna2.server:    Realtime factor: 420.3x  ← SUCCESS!
INFO:xdna2.server:  [4/5] Running decoder...
INFO:xdna2.server:  [5/5] Complete!
```

---

## Troubleshooting

### Issue: "Cannot find whisper_encoder.xclbin"

**Solution**:
1. Check where .xclbin files are located
2. Update `xclbin_path` in `load_xrt_npu_application()`
3. Or create symlink: `ln -s /actual/path /expected/path`

### Issue: "pyxrt not found"

**Solution**:
```bash
# Find XRT Python wheel
find /opt/xilinx/xrt -name "pyxrt*.whl"

# Install it
pip install /opt/xilinx/xrt/python/pyxrt-*.whl
```

### Issue: "Callback registration failed"

**Check**:
1. Is XRT app loaded successfully?
2. Does `npu_callback.get_callback_fn()` return valid function?
3. Are all 6 layers getting callback registered?
4. Check C++ logs for errors

### Issue: Still getting "NPU callback not set"

**Debug**:
```python
# Add debug logging in encoder_cpp.py forward()
logger.debug(f"NPU enabled: {self.use_npu}")
logger.debug(f"NPU callback: {self.npu_callback}")
logger.debug(f"Callback registered: {callback_registered}")
```

---

## Post-Fix: Re-run Validation

After Bug #6 is fixed, run full validation:

```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/tests

# 1. Basic functionality (5 min)
./run_basic_tests.sh

# 2. Integration tests (10 min)
pytest test_pipeline_integration.py -v

# 3. Performance benchmarks (30 min)
./run_performance_tests.sh

# 4. Accuracy validation (15 min)
python3 validate_accuracy.py

# 5. Update Week 13 report (30 min)
# Add results to WEEK13_VALIDATION_REPORT.md
```

**Expected Results**:
- All basic tests pass
- Integration tests: 8/8 pass
- Sequential mode: 15.6 req/s (400-500x realtime)
- Pipeline mode: 67 req/s (+329%)
- Transcription accuracy: >95%

---

## Timeline

| Task | Time | Status |
|------|------|--------|
| 1. cpp_runtime_wrapper.py | 1 hour | ⏳ TODO |
| 2. encoder_cpp.py | 30 min | ⏳ TODO |
| 3. server.py | 1.5 hours | ⏳ TODO |
| 4. Testing | 1 hour | ⏳ TODO |
| **TOTAL** | **4 hours** | ⏳ TODO |

---

## Success Criteria

- [ ] Service starts without NPU errors
- [ ] Logs show callback registered for all 6 layers
- [ ] Health check returns 200 OK
- [ ] 1s audio transcribes successfully
- [ ] Latency < 100ms (400x realtime)
- [ ] No "NPU callback not set" errors
- [ ] Integration tests pass
- [ ] Performance meets 400-500x target

---

## Reference Files

- **Full Analysis**: `BUG6_ROOT_CAUSE_ANALYSIS.md`
- **Validation Report**: `WEEK13_VALIDATION_REPORT.md`
- **Executive Summary**: `WEEK13_EXECUTIVE_SUMMARY.md`
- **This Checklist**: `BUG6_FIX_CHECKLIST.md`

---

**Created**: November 1, 2025
**Estimated Completion**: 4 hours
**Next Step**: Start with Task 1 (cpp_runtime_wrapper.py)
