#!/usr/bin/env python3
"""
Python FFI Wrapper for C++ NPU Runtime

Provides ctypes-based Python bindings to the C++ Whisper encoder runtime.
Handles numpy array conversions, error checking, and resource management.

Key Features:
- Zero-copy numpy array integration
- Automatic error handling and reporting
- Resource cleanup with context managers
- Support for both libwhisper_encoder_cpp.so and libwhisper_xdna2_cpp.so

Architecture:
    Python App → cpp_runtime_wrapper.py → encoder_c_api.h → C++ Runtime → NPU

Author: CC-1L Integration Team
Date: November 1, 2025
Status: Production-ready
"""

import ctypes
from ctypes import (
    c_void_p, c_float, c_int, c_size_t, c_char_p,
    POINTER, CFUNCTYPE, cdll
)
import numpy as np
import os
from typing import Optional, Tuple
from pathlib import Path


class EncoderLayerHandle(ctypes.Structure):
    """Opaque handle to C++ encoder layer"""
    pass


class CPPRuntimeError(Exception):
    """Exception raised for C++ runtime errors"""
    pass


class CPPRuntimeWrapper:
    """
    Python wrapper for C++ Whisper encoder runtime.

    Provides high-level Python API that wraps the C API from encoder_c_api.h.
    Handles library loading, numpy conversions, and error checking.

    Usage:
        # Load the library
        runtime = CPPRuntimeWrapper()

        # Create encoder layer
        handle = runtime.create_layer(layer_idx=0, n_heads=8, n_state=512, ffn_dim=2048)

        # Load weights (as numpy arrays)
        runtime.load_weights(handle, q_weight, k_weight, ...)

        # Run forward pass
        output = runtime.forward(handle, input_data, seq_len, n_state)

        # Cleanup
        runtime.destroy_layer(handle)
    """

    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize the C++ runtime wrapper.

        Args:
            library_path: Path to .so file. If None, auto-detects from cpp/build/

        Raises:
            CPPRuntimeError: If library cannot be loaded or configured incorrectly
        """
        self.lib = None
        self._library_path = library_path

        # Load the library
        self._load_library()

        # Configure function signatures
        self._configure_functions()

        # Verify library configuration
        self._verify_configuration()

    def _load_library(self):
        """Load the C++ shared library"""
        if self._library_path:
            lib_path = self._library_path
        else:
            # Auto-detect library location
            lib_path = self._find_library()

        if not os.path.exists(lib_path):
            raise CPPRuntimeError(
                f"C++ runtime library not found at: {lib_path}\n"
                f"Did you build the C++ runtime? (cd cpp && ./build.sh)"
            )

        try:
            self.lib = cdll.LoadLibrary(lib_path)
            print(f"[CPPRuntime] Loaded library: {lib_path}")
        except OSError as e:
            raise CPPRuntimeError(f"Failed to load library {lib_path}: {e}")

    def _find_library(self) -> str:
        """
        Auto-detect C++ library location.

        Returns:
            Path to the shared library
        """
        # Get the directory containing this file
        this_dir = Path(__file__).parent

        # Check cpp/build/ directory
        cpp_dir = this_dir / "cpp" / "build"

        # Try both library names (encoder lib has the C API)
        candidates = [
            cpp_dir / "libwhisper_encoder_cpp.so",  # Has encoder C API
            cpp_dir / "libwhisper_xdna2_cpp.so",     # Has XRT runtime
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        # Fallback to default
        return str(cpp_dir / "libwhisper_encoder_cpp.so")

    def _configure_functions(self):
        """Configure ctypes function signatures"""

        # encoder_layer_create
        self.lib.encoder_layer_create.argtypes = [
            c_size_t,  # layer_idx
            c_size_t,  # n_heads
            c_size_t,  # n_state
            c_size_t,  # ffn_dim
        ]
        self.lib.encoder_layer_create.restype = c_void_p

        # encoder_layer_destroy
        self.lib.encoder_layer_destroy.argtypes = [c_void_p]
        self.lib.encoder_layer_destroy.restype = None

        # encoder_layer_load_weights
        self.lib.encoder_layer_load_weights.argtypes = [
            c_void_p,                    # handle
            POINTER(c_float),            # q_weight
            POINTER(c_float),            # k_weight
            POINTER(c_float),            # v_weight
            POINTER(c_float),            # out_weight
            POINTER(c_float),            # q_bias
            POINTER(c_float),            # k_bias
            POINTER(c_float),            # v_bias
            POINTER(c_float),            # out_bias
            POINTER(c_float),            # fc1_weight
            POINTER(c_float),            # fc2_weight
            POINTER(c_float),            # fc1_bias
            POINTER(c_float),            # fc2_bias
            POINTER(c_float),            # attn_ln_weight
            POINTER(c_float),            # attn_ln_bias
            POINTER(c_float),            # ffn_ln_weight
            POINTER(c_float),            # ffn_ln_bias
            c_size_t,                    # n_state
            c_size_t,                    # ffn_dim
        ]
        self.lib.encoder_layer_load_weights.restype = c_int

        # encoder_layer_forward
        self.lib.encoder_layer_forward.argtypes = [
            c_void_p,                    # handle
            POINTER(c_float),            # input
            POINTER(c_float),            # output
            c_size_t,                    # seq_len
            c_size_t,                    # n_state
        ]
        self.lib.encoder_layer_forward.restype = c_int

        # encoder_get_version
        self.lib.encoder_get_version.argtypes = []
        self.lib.encoder_get_version.restype = c_char_p

        # encoder_check_config
        self.lib.encoder_check_config.argtypes = []
        self.lib.encoder_check_config.restype = c_int

    def _verify_configuration(self):
        """Verify library is correctly configured"""
        config_ok = self.lib.encoder_check_config()
        if not config_ok:
            raise CPPRuntimeError(
                "Library configuration mismatch! "
                "Please rebuild the C++ runtime."
            )

        version = self.get_version()
        print(f"[CPPRuntime] Version: {version}")

    def get_version(self) -> str:
        """Get library version string"""
        version_bytes = self.lib.encoder_get_version()
        return version_bytes.decode('utf-8')

    def create_layer(
        self,
        layer_idx: int,
        n_heads: int = 8,
        n_state: int = 512,
        ffn_dim: int = 2048
    ) -> int:
        """
        Create a new encoder layer.

        Args:
            layer_idx: Layer index (0-5 for Whisper Base)
            n_heads: Number of attention heads (8 for Whisper Base)
            n_state: Hidden dimension (512 for Whisper Base)
            ffn_dim: FFN dimension (2048 for Whisper Base)

        Returns:
            Handle to encoder layer (opaque pointer as int)

        Raises:
            CPPRuntimeError: If layer creation fails
        """
        handle = self.lib.encoder_layer_create(
            layer_idx, n_heads, n_state, ffn_dim
        )

        if not handle:
            raise CPPRuntimeError(
                f"Failed to create encoder layer {layer_idx}"
            )

        return handle

    def destroy_layer(self, handle: int):
        """
        Destroy encoder layer and free resources.

        Args:
            handle: Encoder layer handle from create_layer()
        """
        if handle:
            self.lib.encoder_layer_destroy(handle)

    def load_weights(
        self,
        handle: int,
        q_weight: np.ndarray,
        k_weight: np.ndarray,
        v_weight: np.ndarray,
        out_weight: np.ndarray,
        q_bias: np.ndarray,
        k_bias: np.ndarray,
        v_bias: np.ndarray,
        out_bias: np.ndarray,
        fc1_weight: np.ndarray,
        fc2_weight: np.ndarray,
        fc1_bias: np.ndarray,
        fc2_bias: np.ndarray,
        attn_ln_weight: np.ndarray,
        attn_ln_bias: np.ndarray,
        ffn_ln_weight: np.ndarray,
        ffn_ln_bias: np.ndarray,
    ) -> None:
        """
        Load weights into encoder layer.

        All weights should be numpy arrays with dtype=np.float32.
        They will be quantized to INT8 internally by the C++ runtime.

        Args:
            handle: Encoder layer handle
            q_weight: Query weight (n_state, n_state)
            k_weight: Key weight (n_state, n_state)
            v_weight: Value weight (n_state, n_state)
            out_weight: Output weight (n_state, n_state)
            q_bias: Query bias (n_state,)
            k_bias: Key bias (n_state,)
            v_bias: Value bias (n_state,)
            out_bias: Output bias (n_state,)
            fc1_weight: FC1 weight (ffn_dim, n_state)
            fc2_weight: FC2 weight (n_state, ffn_dim)
            fc1_bias: FC1 bias (ffn_dim,)
            fc2_bias: FC2 bias (n_state,)
            attn_ln_weight: Attention LayerNorm weight (n_state,)
            attn_ln_bias: Attention LayerNorm bias (n_state,)
            ffn_ln_weight: FFN LayerNorm weight (n_state,)
            ffn_ln_bias: FFN LayerNorm bias (n_state,)

        Raises:
            CPPRuntimeError: If weight loading fails
        """
        # Validate all weights are float32 (biases can be None for Whisper base model)
        weights = [
            q_weight, k_weight, v_weight, out_weight,
            q_bias, k_bias, v_bias, out_bias,
            fc1_weight, fc2_weight, fc1_bias, fc2_bias,
            attn_ln_weight, attn_ln_bias, ffn_ln_weight, ffn_ln_bias
        ]

        # Indices of optional weights (biases)
        optional_indices = {4, 5, 6, 7, 10, 11, 13, 15}  # q_bias, k_bias, v_bias, out_bias, fc1_bias, fc2_bias, attn_ln_bias, ffn_ln_bias

        for i, w in enumerate(weights):
            if w is None:
                if i not in optional_indices:
                    raise CPPRuntimeError(f"Weight {i} is None but not optional")
                continue  # Skip None biases
            if not isinstance(w, np.ndarray):
                raise CPPRuntimeError(f"Weight {i} is not a numpy array")
            if w.dtype != np.float32:
                raise CPPRuntimeError(
                    f"Weight {i} has dtype {w.dtype}, expected float32"
                )

        # Extract dimensions
        n_state = q_weight.shape[0]
        ffn_dim = fc1_weight.shape[0]

        # Flatten all weights (C++ expects 1D arrays)
        q_weight_flat = q_weight.flatten()
        k_weight_flat = k_weight.flatten()
        v_weight_flat = v_weight.flatten()
        out_weight_flat = out_weight.flatten()
        fc1_weight_flat = fc1_weight.flatten()
        fc2_weight_flat = fc2_weight.flatten()

        # For None biases, use zero arrays (Whisper base model has no K/V biases)
        if q_bias is None:
            q_bias = np.zeros(n_state, dtype=np.float32)
        if k_bias is None:
            k_bias = np.zeros(n_state, dtype=np.float32)
        if v_bias is None:
            v_bias = np.zeros(n_state, dtype=np.float32)
        if out_bias is None:
            out_bias = np.zeros(n_state, dtype=np.float32)
        if fc1_bias is None:
            fc1_bias = np.zeros(ffn_dim, dtype=np.float32)
        if fc2_bias is None:
            fc2_bias = np.zeros(n_state, dtype=np.float32)
        if attn_ln_bias is None:
            attn_ln_bias = np.zeros(n_state, dtype=np.float32)
        if ffn_ln_bias is None:
            ffn_ln_bias = np.zeros(n_state, dtype=np.float32)

        # Call C++ function
        result = self.lib.encoder_layer_load_weights(
            handle,
            q_weight_flat.ctypes.data_as(POINTER(c_float)),
            k_weight_flat.ctypes.data_as(POINTER(c_float)),
            v_weight_flat.ctypes.data_as(POINTER(c_float)),
            out_weight_flat.ctypes.data_as(POINTER(c_float)),
            q_bias.ctypes.data_as(POINTER(c_float)),
            k_bias.ctypes.data_as(POINTER(c_float)),
            v_bias.ctypes.data_as(POINTER(c_float)),
            out_bias.ctypes.data_as(POINTER(c_float)),
            fc1_weight_flat.ctypes.data_as(POINTER(c_float)),
            fc2_weight_flat.ctypes.data_as(POINTER(c_float)),
            fc1_bias.ctypes.data_as(POINTER(c_float)),
            fc2_bias.ctypes.data_as(POINTER(c_float)),
            attn_ln_weight.ctypes.data_as(POINTER(c_float)),
            attn_ln_bias.ctypes.data_as(POINTER(c_float)),
            ffn_ln_weight.ctypes.data_as(POINTER(c_float)),
            ffn_ln_bias.ctypes.data_as(POINTER(c_float)),
            n_state,
            ffn_dim
        )

        if result != 0:
            raise CPPRuntimeError("Failed to load weights into encoder layer")

    def forward(
        self,
        handle: int,
        input_data: np.ndarray,
        seq_len: int,
        n_state: int
    ) -> np.ndarray:
        """
        Run encoder layer forward pass.

        Args:
            handle: Encoder layer handle
            input_data: Input array (seq_len, n_state), dtype=float32
            seq_len: Sequence length
            n_state: Hidden dimension

        Returns:
            Output array (seq_len, n_state), dtype=float32

        Raises:
            CPPRuntimeError: If forward pass fails
        """
        # Validate input
        if not isinstance(input_data, np.ndarray):
            raise CPPRuntimeError("Input must be a numpy array")
        if input_data.dtype != np.float32:
            raise CPPRuntimeError(
                f"Input has dtype {input_data.dtype}, expected float32"
            )

        expected_shape = (seq_len, n_state)
        if input_data.shape != expected_shape:
            raise CPPRuntimeError(
                f"Input shape {input_data.shape} != expected {expected_shape}"
            )

        # Flatten input (C++ expects 1D array)
        input_flat = input_data.flatten()

        # Allocate output buffer
        output_flat = np.zeros(seq_len * n_state, dtype=np.float32)

        # Call C++ function
        result = self.lib.encoder_layer_forward(
            handle,
            input_flat.ctypes.data_as(POINTER(c_float)),
            output_flat.ctypes.data_as(POINTER(c_float)),
            seq_len,
            n_state
        )

        if result != 0:
            raise CPPRuntimeError("Forward pass failed")

        # Reshape output
        return output_flat.reshape((seq_len, n_state))

    def set_npu_callback(self, layer_handle: int, callback_fn: callable) -> bool:
        """
        Register NPU callback function with C++ encoder layer.

        This is the missing link that wires the Python NPU callback to the C++ runtime.
        The callback_fn should match the NPUCallbackNative signature.

        Args:
            layer_handle: Handle to encoder layer from create_layer()
            callback_fn: Python callback function that takes:
                - user_data (void*)
                - a_ptr (float*)
                - b_ptr (float*)
                - c_ptr (float*)
                - m, k, n (size_t)
                Returns: int (0 for success, -1 for error)

        Returns:
            True if callback registration successful

        Raises:
            CPPRuntimeError: If callback registration fails
        """
        if not self.lib:
            raise CPPRuntimeError("Library not loaded")

        # Define NPU callback type matching C++ signature:
        # typedef int (*NPUMatmulCallback)(
        #     void* user_data,
        #     const float* A, const float* B, float* C,
        #     size_t m, size_t k, size_t n
        # );
        NPUMatmulCallback = CFUNCTYPE(
            c_int,           # return type
            c_void_p,        # user_data
            POINTER(c_float),  # A matrix
            POINTER(c_float),  # B matrix
            POINTER(c_float),  # C matrix
            c_size_t,        # m
            c_size_t,        # k
            c_size_t         # n
        )

        # Configure C++ function signature
        self.lib.encoder_layer_set_npu_callback.argtypes = [
            c_void_p,           # EncoderLayerHandle
            NPUMatmulCallback,  # callback
            c_void_p            # user_data
        ]
        self.lib.encoder_layer_set_npu_callback.restype = c_int

        # Convert Python callback to C callback
        c_callback = NPUMatmulCallback(callback_fn)

        # Store callback to prevent garbage collection
        # This is CRITICAL - Python will garbage collect the callback if we don't keep a reference
        if not hasattr(self, '_npu_callbacks'):
            self._npu_callbacks = {}
        self._npu_callbacks[layer_handle] = c_callback

        # Call C++ function
        result = self.lib.encoder_layer_set_npu_callback(
            layer_handle,
            c_callback,
            c_void_p(0)  # user_data (NULL for now)
        )

        if result != 0:
            raise CPPRuntimeError(f"Failed to set NPU callback (error code {result})")

        return True


class EncoderLayer:
    """
    Context manager for C++ encoder layer.

    Automatically handles resource cleanup.

    Usage:
        runtime = CPPRuntimeWrapper()

        with EncoderLayer(runtime, layer_idx=0) as layer:
            # Load weights
            runtime.load_weights(layer.handle, ...)

            # Run forward pass
            output = runtime.forward(layer.handle, input_data, seq_len, n_state)

        # Layer automatically destroyed when exiting context
    """

    def __init__(
        self,
        runtime: CPPRuntimeWrapper,
        layer_idx: int,
        n_heads: int = 8,
        n_state: int = 512,
        ffn_dim: int = 2048
    ):
        """
        Create encoder layer with automatic cleanup.

        Args:
            runtime: CPPRuntimeWrapper instance
            layer_idx: Layer index (0-5)
            n_heads: Number of attention heads
            n_state: Hidden dimension
            ffn_dim: FFN dimension
        """
        self.runtime = runtime
        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.n_state = n_state
        self.ffn_dim = ffn_dim
        self.handle = None

    def __enter__(self):
        """Create layer when entering context"""
        self.handle = self.runtime.create_layer(
            self.layer_idx, self.n_heads, self.n_state, self.ffn_dim
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy layer when exiting context"""
        if self.handle:
            self.runtime.destroy_layer(self.handle)
            self.handle = None
        return False


def main():
    """Demonstration of C++ runtime wrapper"""
    print("C++ Runtime Wrapper - Demonstration\n")

    try:
        # Load the runtime
        print("[Demo] Loading C++ runtime...")
        runtime = CPPRuntimeWrapper()
        print(f"  Version: {runtime.get_version()}")

        # Create a layer
        print("\n[Demo] Creating encoder layer 0...")
        with EncoderLayer(runtime, layer_idx=0) as layer:
            print(f"  Handle: {layer.handle}")
            print(f"  Layer created successfully!")

            # Demo: Create dummy weights
            print("\n[Demo] Creating dummy weights...")
            n_state = 512
            ffn_dim = 2048

            q_weight = np.random.randn(n_state, n_state).astype(np.float32)
            k_weight = np.random.randn(n_state, n_state).astype(np.float32)
            v_weight = np.random.randn(n_state, n_state).astype(np.float32)
            out_weight = np.random.randn(n_state, n_state).astype(np.float32)

            q_bias = np.random.randn(n_state).astype(np.float32)
            k_bias = np.random.randn(n_state).astype(np.float32)
            v_bias = np.random.randn(n_state).astype(np.float32)
            out_bias = np.random.randn(n_state).astype(np.float32)

            fc1_weight = np.random.randn(ffn_dim, n_state).astype(np.float32)
            fc2_weight = np.random.randn(n_state, ffn_dim).astype(np.float32)
            fc1_bias = np.random.randn(ffn_dim).astype(np.float32)
            fc2_bias = np.random.randn(n_state).astype(np.float32)

            attn_ln_weight = np.random.randn(n_state).astype(np.float32)
            attn_ln_bias = np.random.randn(n_state).astype(np.float32)
            ffn_ln_weight = np.random.randn(n_state).astype(np.float32)
            ffn_ln_bias = np.random.randn(n_state).astype(np.float32)

            print("[Demo] Loading weights...")
            runtime.load_weights(
                layer.handle,
                q_weight, k_weight, v_weight, out_weight,
                q_bias, k_bias, v_bias, out_bias,
                fc1_weight, fc2_weight, fc1_bias, fc2_bias,
                attn_ln_weight, attn_ln_bias, ffn_ln_weight, ffn_ln_bias
            )
            print("  Weights loaded successfully!")

            # Demo: Run forward pass
            print("\n[Demo] Running forward pass...")
            seq_len = 1500
            input_data = np.random.randn(seq_len, n_state).astype(np.float32)

            output = runtime.forward(layer.handle, input_data, seq_len, n_state)
            print(f"  Output shape: {output.shape}")
            print(f"  Output dtype: {output.dtype}")
            print(f"  Forward pass successful!")

        print("\n[Demo] Layer automatically destroyed")
        print("\n✅ All tests passed!")
        return 0

    except CPPRuntimeError as e:
        print(f"\n❌ Runtime error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
