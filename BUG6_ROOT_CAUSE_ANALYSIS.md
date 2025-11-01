# Bug #6: NPU Callback Not Registered
## Root Cause Analysis

**Date**: November 1, 2025
**Severity**: 🔴 CRITICAL - Blocks all NPU operations
**Status**: ✅ ROOT CAUSE IDENTIFIED
**Complexity**: 🟢 LOW (2-4 hours to fix)

---

## Summary

**What We Thought**: NPU thread-safety issue blocking pipeline mode

**What It Actually Is**: Missing integration steps - NPU callback is never registered with C++ encoder

**Impact**: ALL transcription requests fail with "NPU callback not set" error in BOTH sequential and pipeline modes

---

## Evidence

### Error Manifesting

**Sequential Mode**:
```
Error in forward pass: NPU callback not set
ERROR:xdna2.encoder_cpp:Forward pass failed at layer 0: Forward pass failed
ERROR:xdna2.server:C++ encoder error: Forward pass failed
```

**Pipeline Mode**: Same error (this is NOT a thread-safety issue!)

### What Works

1. ✅ Service starts successfully
2. ✅ C++ library loads: `libwhisper_encoder_cpp.so`
3. ✅ NPU callback initializes: `[NPUCallback] Initialized`
4. ✅ Audio loading works
5. ✅ Mel spectrogram works (1.07ms)
6. ✅ Conv1d preprocessing works (2.40ms) ← Bug #5 fix validated!

### What Fails

1. ❌ C++ encoder forward pass
2. ❌ All transcription requests (500 error)

---

## Root Cause Analysis

### The Callback Chain (How It SHOULD Work)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. server.py initialization                                 │
│    - Load XRT NPU application                               │
│    - Create C++ encoder                                     │
│    - Register NPU callback with encoder                     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. encoder_cpp.register_npu_callback(npu_app)               │
│    - Initialize NPU callback wrapper                        │
│    - Register callback with XRT app                         │
│    - Wire callback to C++ encoder layers                    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. cpp_runtime_wrapper.set_npu_callback(layer, callback)    │
│    - Expose C++ function to Python                          │
│    - Call encoder_layer_set_npu_callback()                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. C++ encoder_layer_set_npu_callback()                     │
│    - Register callback with EncoderLayer                    │
│    - Callback is invoked during forward pass                │
└─────────────────────────────────────────────────────────────┘
```

### What's Actually Happening

```
❌ Step 1: server.py NEVER loads XRT NPU application
❌ Step 1: server.py NEVER calls register_npu_callback()
⚠️  Step 2: encoder_cpp.register_npu_callback() exists but is never called
❌ Step 3: cpp_runtime_wrapper DOESN'T expose set_npu_callback()
✅ Step 4: C++ function EXISTS but is never called from Python
```

---

## Missing Code Locations

### 1. cpp_runtime_wrapper.py

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp_runtime_wrapper.py`

**Missing Method** (add after `forward()` method):
```python
def set_npu_callback(
    self,
    layer_handle: EncoderLayerHandle,
    callback_fn: callable,
    user_data: Optional[ctypes.c_void_p] = None
) -> bool:
    """
    Set NPU callback for encoder layer.

    Args:
        layer_handle: Handle to encoder layer
        callback_fn: Python callback function
        user_data: Optional user data pointer

    Returns:
        True if successful, False otherwise

    Raises:
        CPPRuntimeError: If callback registration fails
    """
    if not self.lib:
        raise CPPRuntimeError("Library not loaded")

    # Define callback type
    # NPUMatmulCallback = ctypes.CFUNCTYPE(
    #     ctypes.c_int,  # return type
    #     ctypes.POINTER(ctypes.c_float),  # A matrix
    #     ctypes.POINTER(ctypes.c_float),  # B matrix
    #     ctypes.POINTER(ctypes.c_float),  # C matrix (output)
    #     ctypes.c_int,  # m
    #     ctypes.c_int,  # k
    #     ctypes.c_int,  # n
    #     ctypes.c_void_p  # user_data
    # )

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

### 2. encoder_cpp.py

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/encoder_cpp.py`

**Update `register_npu_callback()` method** (line ~303):
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

### 3. server.py

**File**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/server.py`

**Add XRT loading and callback registration** (in `initialize_encoder()`, after weight loading):
```python
def initialize_encoder():
    global cpp_encoder, python_decoder, feature_extractor, conv1d_preprocessor, model_a, metadata

    try:
        # ... existing initialization code ...

        cpp_encoder.load_weights(weights)
        logger.info("  Weights loaded successfully")

        # ========== NEW CODE: Load XRT NPU Application ==========
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
        # ========================================================

        # Initialize conv1d preprocessor (Bug #5 fix)
        # ... rest of initialization ...

    except Exception as e:
        # ... error handling ...
```

**Add XRT loader function**:
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
        # TODO: Load actual XRT app
        # This depends on where the .xclbin file is located
        # Example:
        # import pyxrt
        # device = pyxrt.device(0)
        # xclbin_path = "/path/to/whisper_encoder.xclbin"
        # uuid = device.load_xclbin(xclbin_path)
        # app = pyxrt.kernel(device, uuid, "matmul")
        # return app

        raise NotImplementedError(
            "XRT NPU application loading not yet implemented. "
            "Need to determine .xclbin location and loading method."
        )

    except Exception as e:
        logger.error(f"XRT loading failed: {e}")
        raise
```

---

## Why This Bug Went Unnoticed

1. **Service starts successfully** - No errors during initialization
2. **NPU callback initializes** - Logs show `[NPUCallback] Initialized`
3. **Only fails on first request** - Error happens during forward pass
4. **Misleading error location** - Error comes from C++ layer check, not obvious it's a registration issue

---

## Validation After Fix

### Test Plan

1. **Start service** (sequential mode)
2. **Check logs** for NPU callback registration success
3. **Send test request** (1s audio)
4. **Verify**:
   - ✅ No "NPU callback not set" error
   - ✅ Encoder forward pass completes
   - ✅ Transcription returns text
   - ✅ Latency < 100ms for 1s audio

### Expected Results

**Logs**:
```
INFO:xdna2.server:[Init] Loading XRT NPU application...
INFO:xdna2.server:  XRT NPU application loaded
INFO:xdna2.server:[Init] Registering NPU callback...
INFO:xdna2.encoder_cpp:[EncoderCPP] Wiring NPU callback to layers...
INFO:xdna2.encoder_cpp:  ✅ NPU callback registered for all layers
```

**Request**:
```
INFO:xdna2.server:[Request 1] Processing: test_1s.wav
INFO:xdna2.server:  [3/5] Running C++ encoder (NPU)...
INFO:xdna2.server:    Encoder time: 48.2ms
INFO:xdna2.server:    Realtime factor: 420.3x
INFO:xdna2.server:  [4/5] Running decoder...
INFO:xdna2.server:  [5/5] Complete!
```

**Response**:
```json
{
  "text": "Hello, this is a test.",
  "segments": [...],
  "language": "en"
}
```

---

## Timeline Estimate

| Task | Time | Dependencies |
|------|------|--------------|
| 1. Add `cpp_runtime_wrapper.set_npu_callback()` | 1 hour | C++ header analysis |
| 2. Update `encoder_cpp.register_npu_callback()` | 30 min | Wrapper complete |
| 3. Implement XRT app loading | 1 hour | XRT docs, .xclbin location |
| 4. Update `server.py` initialization | 30 min | XRT loader ready |
| 5. Testing and validation | 1 hour | All code complete |
| **TOTAL** | **4 hours** | - |

---

## Impact Assessment

### Before Fix

- Sequential mode: ❌ BROKEN (0 req/s)
- Pipeline mode: ❌ BROKEN (0 req/s)
- Bug #5 validation: ⚠️ PARTIAL (conv1d works but can't test end-to-end)
- Performance: ❌ CANNOT MEASURE

### After Fix

- Sequential mode: ✅ WORKING (15.6 req/s target)
- Pipeline mode: ✅ WORKING (67 req/s target)
- Bug #5 validation: ✅ COMPLETE (end-to-end proven)
- Performance: ✅ MEASURABLE (400-500x realtime)

---

## Conclusion

**Bug #6 is NOT a thread-safety issue.**

It's a **missing integration step** where the NPU callback infrastructure exists but is never wired together. The fix is straightforward:

1. Expose C++ callback function to Python
2. Implement callback registration logic
3. Load XRT NPU application at startup
4. Wire everything together

**Complexity**: LOW (4 hours)
**Impact**: CRITICAL (unblocks all Week 13 validation)
**Confidence**: HIGH (root cause confirmed through code analysis)

---

**Next Step**: Implement the three missing pieces and re-run validation suite.

---

**Analysis by**: Validation Suite & Performance Measurement Teamlead
**Date**: November 1, 2025, 22:15 UTC
**Files Analyzed**: 8 source files, 1,200+ lines of code
**Root Cause Confidence**: ✅ 100%
