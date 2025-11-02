"""
XDNA2 C++ Backend Server with Multi-Stream Pipeline

Native FastAPI server using C++ NPU encoder for 400-500x realtime performance.
Drop-in replacement for xdna1/server.py with identical API.

Key Features:
- C++ encoder via encoder_cpp.WhisperEncoderCPP
- NPU callback integration
- Multi-stream pipelining for concurrent request processing
- Buffer pool + zero-copy optimization
- OpenAI-compatible /v1/audio/transcriptions endpoint

Performance Target (Sequential):
- 400-500x realtime (vs 220x Python XDNA1)
- ~60ms latency for 30s audio
- 15.6 req/s throughput

Performance Target (Pipeline):
- 67 req/s throughput (+329%)
- 15% NPU utilization (+1775%)
- 10-15 concurrent requests

Author: CC-1L Multi-Stream Pipeline Team
Date: November 1, 2025
Status: Week 9 Multi-Stream Implementation
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import whisperx
import os
import tempfile
import torch
import gc
import logging
import numpy as np
import time
import asyncio
from typing import Optional
from pathlib import Path
import sys

# Add parent directory to path for buffer_pool import
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import C++ encoder
from .encoder_cpp import WhisperEncoderCPP, create_encoder_cpp
from .cpp_runtime_wrapper import CPPRuntimeError

# Import buffer pool
from buffer_pool import GlobalBufferManager

# Import zero-copy mel utilities
from .mel_utils import compute_mel_spectrogram_zerocopy, validate_mel_contiguity

# Import conv1d preprocessing (Bug #5 fix)
from .whisper_conv1d import WhisperConv1dPreprocessor

# Import multi-stream pipeline
from request_queue import RequestQueue, QueuedRequest
from transcription_pipeline import TranscriptionPipeline

# Import batch processor (Week 19)
from .batch_processor import BatchProcessor

# Import faster-whisper decoder (Week 19 optimization)
from .faster_whisper_wrapper import FasterWhisperDecoder

# Import custom decoder (Week 19.5 architecture fix)
from .custom_whisper_decoder import CustomWhisperDecoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unicorn-Amanuensis XDNA2 C++ + Multi-Stream Pipeline",
    description="Speech-to-Text with C++ NPU Encoder + Concurrent Processing (67 req/s)",
    version="3.0.0"  # Version bump for pipeline support
)

# Model configuration
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
# Zero-Copy Optimization: Keep decoder on CPU (same as encoder)
# This eliminates 2ms CPU->GPU transfer overhead
# Encoder is on CPU (C++ NPU backend), decoder should match
DEVICE = "cpu"  # Force CPU for zero-copy optimization
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")  # CPU-optimized
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Pipeline configuration
ENABLE_PIPELINE = os.environ.get("ENABLE_PIPELINE", "true").lower() == "true"
NUM_LOAD_WORKERS = int(os.environ.get("NUM_LOAD_WORKERS", "4"))
NUM_DECODER_WORKERS = int(os.environ.get("NUM_DECODER_WORKERS", "4"))
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "100"))

# Week 19 Batch Processing Configuration
ENABLE_BATCHING = os.environ.get("ENABLE_BATCHING", "true").lower() == "true"
BATCH_MAX_SIZE = int(os.environ.get("BATCH_MAX_SIZE", "8"))  # 4-16
BATCH_MAX_WAIT_MS = int(os.environ.get("BATCH_MAX_WAIT_MS", "50"))  # 25-100ms

# Week 19 Decoder Optimization
# USE_FASTER_WHISPER: Enable faster-whisper (CTranslate2) decoder for 4-6× speedup
# - "true": Use faster-whisper (RECOMMENDED for production)
# - "false": Use WhisperX (legacy, slower)
USE_FASTER_WHISPER = os.environ.get("USE_FASTER_WHISPER", "true").lower() == "true"

# Week 19.5 Architecture Fix - Custom Decoder
# USE_CUSTOM_DECODER: Use CustomWhisperDecoder that accepts NPU features directly
# - "true": Use CustomWhisperDecoder (ELIMINATES CPU RE-ENCODING, 2.5-3.6× speedup!)
# - "false": Use faster-whisper or WhisperX (RE-ENCODES on CPU, wasteful)
USE_CUSTOM_DECODER = os.environ.get("USE_CUSTOM_DECODER", "true").lower() == "true"

# NPU/Hardware configuration
# User preference: "I really don't want CPU fallback" - defaults to no fallback
REQUIRE_NPU = os.environ.get("REQUIRE_NPU", "false").lower() == "true"  # Fail if NPU unavailable
ALLOW_FALLBACK = os.environ.get("ALLOW_FALLBACK", "false").lower() == "true"  # Allow fallback devices (default: false)
FALLBACK_DEVICE = os.environ.get("FALLBACK_DEVICE", "none")  # none, igpu, or cpu (default: none)
# Device priority when fallback enabled: npu -> igpu -> cpu
# To enable fallback: Set ALLOW_FALLBACK=true and FALLBACK_DEVICE=igpu or cpu

# Global instances (initialized at startup)
cpp_encoder: Optional[WhisperEncoderCPP] = None
python_decoder = None  # WhisperX decoder (fallback to Python for now)
feature_extractor = None  # Feature extractor from WhisperX model
conv1d_preprocessor: Optional[WhisperConv1dPreprocessor] = None  # Conv1d preprocessing (Bug #5 fix)
model_a = None  # Alignment model
metadata = None
buffer_manager: Optional[GlobalBufferManager] = None
pipeline: Optional[TranscriptionPipeline] = None  # Multi-stream pipeline
batch_processor: Optional[BatchProcessor] = None  # Week 19 batch processor

# Performance stats
startup_time = time.time()
request_count = 0
total_audio_seconds = 0.0
total_processing_time = 0.0


def load_xrt_npu_application():
    """
    Load XRT NPU application for Whisper encoder.

    Attempts to load the xclbin kernel from expected locations.
    This is part of Bug #6 fix - the missing NPU callback registration.

    Returns:
        Loaded NPU application object (stub for now)

    Raises:
        FileNotFoundError: If xclbin not found
        ImportError: If pyxrt not available
        Exception: If XRT loading fails
    """
    # Check for xclbin files in expected locations
    # __file__ = /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/server.py
    # kernels/ = /home/ccadmin/CC-1L/kernels/ (4 levels up from server.py)
    xclbin_candidates = [
        # Whisper-specific xclbins (preferred)
        Path(__file__).parent / "cpp" / "build" / "whisper_encoder.xclbin",
        Path(__file__).parent / "kernels" / "whisper_encoder.xclbin",
        Path(__file__).parent / "final.xclbin",
        Path(__file__).parent.parent / "kernels" / "whisper_encoder.xclbin",

        # Generic matmul kernels from CC-1L/kernels/common/ (actual kernel location)
        Path(__file__).parent.parent.parent.parent / "kernels" / "common" / "build_bf16_1tile" / "matmul_1tile_bf16.xclbin",
        Path(__file__).parent.parent.parent.parent / "kernels" / "common" / "build_bf16_2tile_FIXED" / "matmul_2tile_bf16_xdna2_FIXED.xclbin",
        Path(__file__).parent.parent.parent.parent / "kernels" / "common" / "build_bf16_4tile" / "matmul_4tile_bf16.xclbin",
        Path(__file__).parent.parent.parent.parent / "kernels" / "common" / "build_bfp16_1tile" / "matmul_1tile.xclbin",
        Path(__file__).parent.parent.parent.parent / "kernels" / "common" / "build_fixed_1tile" / "matmul_1tile.xclbin",
    ]

    xclbin_path = None
    for candidate in xclbin_candidates:
        if candidate.exists():
            xclbin_path = candidate
            break

    if not xclbin_path:
        # List tried paths for debugging
        tried = "\n    ".join(str(c) for c in xclbin_candidates)
        raise FileNotFoundError(
            f"Cannot find whisper_encoder.xclbin in expected locations:\n    {tried}\n"
            f"NPU acceleration requires compiled xclbin kernel."
        )

    logger.info(f"  Found xclbin: {xclbin_path}")

    # Try to import and use pyxrt (XRT Python bindings)
    try:
        import pyxrt as xrt
        logger.info(f"  Loading XRT device...")

        # Load XRT device (device 0)
        device = xrt.device(0)
        logger.info(f"  XRT device opened")

        # Load xclbin file into object (CRITICAL: Don't use load_xclbin() on XDNA2!)
        xclbin = xrt.xclbin(str(xclbin_path))
        logger.info(f"  xclbin object created")

        # Register xclbin with device (XDNA2 requires register_xclbin, not load_xclbin)
        device.register_xclbin(xclbin)
        logger.info(f"  xclbin registered successfully")

        # Get UUID
        uuid = xclbin.get_uuid()
        logger.info(f"  UUID: {uuid}")

        # Create hardware context
        context = xrt.hw_context(device, uuid)
        logger.info(f"  Hardware context created")

        # Get available kernels
        kernels = xclbin.get_kernels()
        kernel_names_available = [k.get_name() for k in kernels]
        logger.info(f"  Available kernels: {kernel_names_available}")

        # Try to get kernel handle
        # MLIR-AIE xclbins use "MLIR_AIE" as kernel name
        # Older xclbins may use matmul variants
        kernel_names_to_try = ["MLIR_AIE", "matmul_bfp16", "matmul_bf16", "matmul", "whisper_matmul"]
        kernel = None
        kernel_name = None
        for kname in kernel_names_to_try:
            try:
                kernel = xrt.kernel(context, kname)
                kernel_name = kname
                logger.info(f"  Loaded kernel: {kname}")
                break
            except:
                continue

        if not kernel:
            raise RuntimeError(
                f"Could not load any kernel from {kernel_names_to_try}. "
                f"Available kernels in xclbin: {kernel_names_available}"
            )

        # Create XRT app wrapper with real buffer operations
        class BufferWrapper:
            """
            Wrapper for XRTApp buffers to provide .write()/.read() interface.
            Compatible with mlir-aie setup_aie() buffer interface.
            """
            def __init__(self, xrt_app, idx):
                self.xrt_app = xrt_app
                self.idx = idx

            def write(self, data):
                """Write data and sync to device"""
                self.xrt_app.write_buffer(self.idx, data)

            def read(self):
                """Sync from device and read data"""
                return self.xrt_app.read_buffer(self.idx)

        class BuffersDict:
            """
            Dictionary-like interface for XRTApp buffers.
            Provides compatibility with npu_callback that expects .buffers[3].write()
            """
            def __init__(self, xrt_app):
                self.xrt_app = xrt_app
                self._wrappers = {}

            def __getitem__(self, idx):
                """Get buffer wrapper by index"""
                if idx not in self._wrappers:
                    self._wrappers[idx] = BufferWrapper(self.xrt_app, idx)
                return self._wrappers[idx]

        class XRTApp:
            """
            XRT Application with Real Buffer Operations.

            Implements real XRT buffer allocation, data transfer, and kernel execution
            for NPU-accelerated matrix operations. Replaces the stub implementation
            with full hardware support.

            Key Features:
            - Real XRT buffer object allocation using xrt.bo()
            - Host-to-device and device-to-host data synchronization
            - Kernel execution with xrt.run()
            - Buffer metadata tracking for size/shape validation
            - Error handling and logging

            Week 16 Implementation (Service Integration Team)
            """
            def __init__(self, device, context, kernel, kernel_name):
                """
                Initialize XRT application.

                Args:
                    device: XRT device handle
                    context: XRT hardware context
                    kernel: XRT kernel handle
                    kernel_name: Name of the loaded kernel
                """
                self.device = device
                self.context = context
                self.kernel = kernel
                self.kernel_name = kernel_name

                # Real XRT buffer objects (xrt.bo instances)
                self.xrt_buffers = {}

                # Buffer metadata (for size tracking and validation)
                self.buffer_metadata = {}

                # Create buffers dict for compatibility with npu_callback
                self.buffers = BuffersDict(self)

                logger.info(f"  XRTApp initialized with kernel: {kernel_name}")

            def register_buffer(self, idx, dtype, shape):
                """
                Allocate real XRT buffer object.

                Creates an actual XRT buffer object (xrt.bo) on the NPU with proper
                size calculation and group ID assignment. Buffers are allocated in
                host-accessible memory for easy data transfer.

                Args:
                    idx: Buffer index (0-based, corresponds to kernel argument index)
                    dtype: NumPy dtype for the buffer (e.g., np.float32, np.bfloat16)
                    shape: Tuple of buffer dimensions (e.g., (1500, 512))

                Raises:
                    RuntimeError: If buffer allocation fails
                """
                try:
                    # Calculate buffer size in bytes
                    size = int(np.prod(shape) * np.dtype(dtype).itemsize)

                    # Create XRT buffer object
                    # - host_only flag: Buffer is accessible from host (for data transfer)
                    # - group_id: Get memory bank assignment from kernel argument index
                    bo = xrt.bo(
                        self.device,
                        size,
                        xrt.bo.host_only,
                        self.kernel.group_id(idx)
                    )

                    # Store buffer object and metadata
                    self.xrt_buffers[idx] = bo
                    self.buffer_metadata[idx] = {
                        'dtype': dtype,
                        'shape': shape,
                        'size': size
                    }

                    logger.info(f"  Registered buffer {idx}: {dtype} {shape} ({size:,} bytes)")

                except Exception as e:
                    logger.error(f"Failed to register buffer {idx}: {e}")
                    raise RuntimeError(f"Buffer registration failed: {e}")

            def write_buffer(self, idx, data):
                """
                Write data to XRT buffer and sync to device.

                Copies data from host (CPU) memory to XRT buffer and synchronizes
                to NPU device memory. Data can be smaller than buffer size (for
                variable-length inputs), but cannot exceed it.

                Args:
                    idx: Buffer index
                    data: NumPy array to write (can be smaller than buffer, same dtype)

                Raises:
                    KeyError: If buffer index not registered
                    ValueError: If data exceeds buffer size or dtype mismatch
                    RuntimeError: If write or sync fails
                """
                if idx not in self.xrt_buffers:
                    raise KeyError(f"Buffer {idx} not registered")

                # Validate data
                metadata = self.buffer_metadata[idx]
                buffer_size = metadata['size']
                expected_dtype = metadata['dtype']

                # Allow variable-sized data (common for NPU matmul with different input sizes)
                # Just check that data doesn't exceed buffer capacity
                data_size = data.nbytes
                if data_size > buffer_size:
                    raise ValueError(
                        f"Data size {data_size} bytes exceeds buffer {idx} "
                        f"capacity {buffer_size} bytes"
                    )

                if data.dtype != expected_dtype:
                    logger.warning(
                        f"Converting data from {data.dtype} to {expected_dtype} "
                        f"for buffer {idx}"
                    )
                    data = data.astype(expected_dtype)

                try:
                    # Get buffer object
                    bo = self.xrt_buffers[idx]

                    # Flatten data for writing (XRT expects 1D byte arrays)
                    data_flat = np.ascontiguousarray(data.flatten())

                    # Write data to buffer (offset 0)
                    bo.write(data_flat, 0)

                    # Synchronize to device (host → NPU)
                    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

                    logger.debug(f"  Wrote {data_flat.nbytes:,} bytes to buffer {idx}")

                except Exception as e:
                    logger.error(f"Failed to write buffer {idx}: {e}")
                    raise RuntimeError(f"Buffer write failed: {e}")

            def read_buffer(self, idx):
                """
                Read data from XRT buffer after kernel execution.

                Synchronizes buffer from NPU device memory to host and reads the
                data into a NumPy array. Used to retrieve results after kernel
                execution.

                Args:
                    idx: Buffer index

                Returns:
                    NumPy array with buffer contents (dtype/shape from metadata)

                Raises:
                    KeyError: If buffer index not registered
                    RuntimeError: If sync or read fails
                """
                if idx not in self.xrt_buffers:
                    raise KeyError(f"Buffer {idx} not registered")

                try:
                    # Get buffer object and metadata
                    bo = self.xrt_buffers[idx]
                    metadata = self.buffer_metadata[idx]

                    # Synchronize from device (NPU → host)
                    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

                    # Read data from buffer
                    # bo.map() returns buffer's memory as bytes
                    # Use size from metadata to handle variable-length data
                    size = metadata['size']
                    data = np.frombuffer(
                        bytes(bo.map())[:size],
                        dtype=metadata['dtype']
                    ).reshape(metadata['shape'])

                    logger.debug(f"  Read {data.nbytes:,} bytes from buffer {idx}")

                    return data

                except Exception as e:
                    logger.error(f"Failed to read buffer {idx}: {e}")
                    raise RuntimeError(f"Buffer read failed: {e}")

            def run(self, input_buffers=None, output_buffers=None):
                """
                Execute NPU kernel with registered buffers.

                Builds the kernel argument list from registered buffers and executes
                the kernel on the NPU. Waits for completion before returning.

                Args:
                    input_buffers: List of input buffer indices (optional, for logging)
                    output_buffers: List of output buffer indices (optional, for logging)

                Returns:
                    True if execution successful

                Raises:
                    RuntimeError: If no buffers registered or kernel execution fails
                """
                if not self.xrt_buffers:
                    raise RuntimeError("No buffers registered")

                try:
                    # Build argument list (buffers in index order)
                    args = []
                    for idx in sorted(self.xrt_buffers.keys()):
                        args.append(self.xrt_buffers[idx])

                    logger.debug(
                        f"  Executing kernel {self.kernel_name} with "
                        f"{len(args)} buffers..."
                    )

                    # Execute kernel with buffer arguments
                    # Returns a run handle that we can wait on
                    run_handle = self.kernel(*args)

                    # Wait for kernel completion
                    run_handle.wait()

                    logger.debug(f"  Kernel {self.kernel_name} execution complete")

                    return True

                except Exception as e:
                    logger.error(f"Kernel execution failed: {e}")
                    raise RuntimeError(f"Kernel execution failed: {e}")

            def cleanup(self):
                """
                Clean up XRT resources.

                Releases all buffer objects. Should be called when done with the
                application to free NPU memory.
                """
                logger.debug(f"  Cleaning up {len(self.xrt_buffers)} buffers...")
                self.xrt_buffers.clear()
                self.buffer_metadata.clear()

        return XRTApp(device, context, kernel, kernel_name)

    except ImportError:
        raise ImportError(
            "pyxrt not found. Install with: "
            "pip install /opt/xilinx/xrt/python/pyxrt-*.whl"
        )


def initialize_encoder():
    """
    Initialize C++ encoder at startup.

    Returns:
        True if successful, False otherwise
    """
    global cpp_encoder, python_decoder, feature_extractor, conv1d_preprocessor, model_a, metadata

    try:
        logger.info("="*70)
        logger.info("  XDNA2 C++ Backend Initialization")
        logger.info("="*70)

        # Create C++ encoder
        logger.info("[Init] Creating C++ encoder...")
        cpp_encoder = create_encoder_cpp(
            num_layers=6,
            n_heads=8,
            n_state=512,
            ffn_dim=2048,
            use_npu=True
        )
        logger.info("  C++ encoder created successfully")

        # Load Whisper model for weights and decoder
        logger.info(f"[Init] Loading Whisper model: {MODEL_SIZE}")
        from transformers import WhisperModel
        whisper_model = WhisperModel.from_pretrained(f"openai/whisper-{MODEL_SIZE}")

        # Extract and load weights into C++ encoder
        logger.info("[Init] Loading weights into C++ encoder...")
        weights = {}
        for layer_idx in range(6):
            layer = whisper_model.encoder.layers[layer_idx]
            prefix = f"encoder.layers.{layer_idx}"

            # Attention weights
            weights[f"{prefix}.self_attn.q_proj.weight"] = layer.self_attn.q_proj.weight.data.cpu().numpy()
            weights[f"{prefix}.self_attn.k_proj.weight"] = layer.self_attn.k_proj.weight.data.cpu().numpy()
            weights[f"{prefix}.self_attn.v_proj.weight"] = layer.self_attn.v_proj.weight.data.cpu().numpy()
            weights[f"{prefix}.self_attn.out_proj.weight"] = layer.self_attn.out_proj.weight.data.cpu().numpy()

            # Attention biases (with null checks for Whisper base model)
            if layer.self_attn.q_proj.bias is not None:
                weights[f"{prefix}.self_attn.q_proj.bias"] = layer.self_attn.q_proj.bias.data.cpu().numpy()
            if layer.self_attn.k_proj.bias is not None:
                weights[f"{prefix}.self_attn.k_proj.bias"] = layer.self_attn.k_proj.bias.data.cpu().numpy()
            if layer.self_attn.v_proj.bias is not None:
                weights[f"{prefix}.self_attn.v_proj.bias"] = layer.self_attn.v_proj.bias.data.cpu().numpy()
            if layer.self_attn.out_proj.bias is not None:
                weights[f"{prefix}.self_attn.out_proj.bias"] = layer.self_attn.out_proj.bias.data.cpu().numpy()

            # FFN weights
            weights[f"{prefix}.fc1.weight"] = layer.fc1.weight.data.cpu().numpy()
            weights[f"{prefix}.fc2.weight"] = layer.fc2.weight.data.cpu().numpy()
            if layer.fc1.bias is not None:
                weights[f"{prefix}.fc1.bias"] = layer.fc1.bias.data.cpu().numpy()
            if layer.fc2.bias is not None:
                weights[f"{prefix}.fc2.bias"] = layer.fc2.bias.data.cpu().numpy()

            # LayerNorm
            weights[f"{prefix}.self_attn_layer_norm.weight"] = layer.self_attn_layer_norm.weight.data.cpu().numpy()
            if layer.self_attn_layer_norm.bias is not None:
                weights[f"{prefix}.self_attn_layer_norm.bias"] = layer.self_attn_layer_norm.bias.data.cpu().numpy()
            weights[f"{prefix}.final_layer_norm.weight"] = layer.final_layer_norm.weight.data.cpu().numpy()
            if layer.final_layer_norm.bias is not None:
                weights[f"{prefix}.final_layer_norm.bias"] = layer.final_layer_norm.bias.data.cpu().numpy()

        cpp_encoder.load_weights(weights)
        logger.info("  Weights loaded successfully")

        # ========== NEW: NPU Callback Registration (Bug #6 Fix) ==========
        if cpp_encoder.use_npu:
            logger.info("[Init] Loading XRT NPU application...")
            try:
                # Try to load XRT app for NPU acceleration
                npu_app = load_xrt_npu_application()
                logger.info("  XRT NPU application loaded successfully")

                # Register NPU callback with encoder
                logger.info("[Init] Registering NPU callback...")
                if cpp_encoder.register_npu_callback(npu_app):
                    logger.info("  ✅ NPU callback registered successfully")
                else:
                    logger.error("  ❌ NPU callback registration failed")
                    logger.warning("  Falling back to CPU mode")
                    cpp_encoder.use_npu = False

            except FileNotFoundError as e:
                error_msg = f"XRT app not found: {e}"
                logger.error(error_msg)

                # Check if NPU is required - fail if so
                if REQUIRE_NPU:
                    logger.error("  ❌ CRITICAL: NPU required but xclbin not found")
                    logger.error("  Set REQUIRE_NPU=false to allow fallback")
                    raise RuntimeError(
                        "NPU acceleration required (REQUIRE_NPU=true) but xclbin not available. "
                        "Check xclbin paths or set REQUIRE_NPU=false to continue."
                    )

                # Check if fallback is allowed - fail if not
                if not ALLOW_FALLBACK:
                    logger.error("  ❌ CRITICAL: Fallback disabled and NPU unavailable")
                    logger.error("  Set ALLOW_FALLBACK=true to enable fallback")
                    raise RuntimeError(
                        "NPU unavailable and fallback disabled (ALLOW_FALLBACK=false). "
                        "Enable fallback or fix xclbin path."
                    )

                # Fallback is allowed - check device preference
                if FALLBACK_DEVICE == "none":
                    logger.error("  ❌ CRITICAL: No fallback device configured")
                    logger.error("  Set FALLBACK_DEVICE=igpu or FALLBACK_DEVICE=cpu to enable fallback")
                    raise RuntimeError(
                        "NPU unavailable and no fallback device configured (FALLBACK_DEVICE=none). "
                        "Set FALLBACK_DEVICE=igpu or cpu to continue."
                    )

                # Log fallback
                logger.warning(f"  NPU acceleration disabled (xclbin not available)")
                logger.warning(f"  Falling back to {FALLBACK_DEVICE.upper()} mode")
                cpp_encoder.use_npu = False

            except ImportError as e:
                error_msg = f"XRT libraries not available: {e}"
                logger.error(error_msg)

                # Check if NPU is required - fail if so
                if REQUIRE_NPU:
                    logger.error("  ❌ CRITICAL: NPU required but pyxrt not installed")
                    logger.error("  Install pyxrt or set REQUIRE_NPU=false to allow fallback")
                    raise RuntimeError(
                        "NPU acceleration required (REQUIRE_NPU=true) but pyxrt not available. "
                        "Install pyxrt or set REQUIRE_NPU=false to continue."
                    )

                # Check if fallback is allowed - fail if not
                if not ALLOW_FALLBACK:
                    logger.error("  ❌ CRITICAL: Fallback disabled and NPU unavailable")
                    logger.error("  Set ALLOW_FALLBACK=true to enable fallback")
                    raise RuntimeError(
                        "NPU unavailable and fallback disabled (ALLOW_FALLBACK=false). "
                        "Enable fallback or install pyxrt."
                    )

                # Fallback is allowed - check device preference
                if FALLBACK_DEVICE == "none":
                    logger.error("  ❌ CRITICAL: No fallback device configured")
                    logger.error("  Set FALLBACK_DEVICE=igpu or FALLBACK_DEVICE=cpu to enable fallback")
                    raise RuntimeError(
                        "NPU unavailable and no fallback device configured (FALLBACK_DEVICE=none). "
                        "Set FALLBACK_DEVICE=igpu or cpu to continue."
                    )

                # Log fallback
                logger.warning(f"  NPU acceleration disabled (pyxrt not installed)")
                logger.warning(f"  Falling back to {FALLBACK_DEVICE.upper()} mode")
                cpp_encoder.use_npu = False

            except Exception as e:
                error_msg = f"Failed to load XRT NPU application: {e}"
                logger.error(error_msg)

                # Check if NPU is required - fail if so
                if REQUIRE_NPU:
                    logger.error("  ❌ CRITICAL: NPU required but loading failed")
                    logger.error("  Set REQUIRE_NPU=false to allow fallback")
                    raise RuntimeError(
                        f"NPU acceleration required (REQUIRE_NPU=true) but loading failed: {e}"
                    )

                # Check if fallback is allowed - fail if not
                if not ALLOW_FALLBACK:
                    logger.error("  ❌ CRITICAL: Fallback disabled and NPU unavailable")
                    logger.error("  Set ALLOW_FALLBACK=true to enable fallback")
                    raise RuntimeError(
                        f"NPU unavailable and fallback disabled (ALLOW_FALLBACK=false): {e}"
                    )

                # Fallback is allowed - check device preference
                if FALLBACK_DEVICE == "none":
                    logger.error("  ❌ CRITICAL: No fallback device configured")
                    logger.error("  Set FALLBACK_DEVICE=igpu or FALLBACK_DEVICE=cpu to enable fallback")
                    raise RuntimeError(
                        f"NPU unavailable and no fallback device configured (FALLBACK_DEVICE=none): {e}"
                    )

                # Log fallback
                logger.warning(f"  Falling back to {FALLBACK_DEVICE.upper()} mode")
                cpp_encoder.use_npu = False
        # =================================================================

        # Initialize conv1d preprocessor (Bug #5 fix)
        logger.info("[Init] Initializing conv1d preprocessor...")
        conv1d_preprocessor = WhisperConv1dPreprocessor(whisper_model)
        logger.info("  Conv1d preprocessor initialized (mel 80→512)")

        # Week 19.5: Choose decoder based on configuration
        if USE_CUSTOM_DECODER:
            # CustomWhisperDecoder (Week 19.5 Architecture Fix - BEST!)
            # Accepts NPU encoder features directly (NO CPU RE-ENCODING!)
            logger.info("[Init] Loading CustomWhisperDecoder (no CPU re-encoding)...")
            python_decoder = CustomWhisperDecoder(
                model_name=MODEL_SIZE,
                device=DEVICE
            )
            # Load WhisperX temporarily to extract feature extractor
            logger.info("[Init] Loading WhisperX for feature extractor...")
            temp_whisperx = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
            feature_extractor = temp_whisperx.model.feature_extractor
            logger.info("  ✅ CustomWhisperDecoder loaded (2.5-3.6× faster - ELIMINATES CPU RE-ENCODING!)")

        elif USE_FASTER_WHISPER:
            # faster-whisper (Week 19 - STILL RE-ENCODES!)
            logger.info("[Init] Loading faster-whisper decoder (CTranslate2)...")
            logger.warning("  ⚠️  faster-whisper RE-ENCODES audio on CPU (wasteful!)")
            logger.warning("  ⚠️  Set USE_CUSTOM_DECODER=true to eliminate re-encoding")
            python_decoder = FasterWhisperDecoder(
                model_name=MODEL_SIZE,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                num_workers=1
            )
            # Load WhisperX temporarily to extract feature extractor
            logger.info("[Init] Loading WhisperX for feature extractor...")
            temp_whisperx = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
            feature_extractor = temp_whisperx.model.feature_extractor
            logger.info("  faster-whisper decoder loaded (4-6× faster than WhisperX, but RE-ENCODES)")

        else:
            # WhisperX (legacy - STILL RE-ENCODES!)
            logger.info("[Init] Loading Python decoder (WhisperX - legacy mode)...")
            logger.warning("  ⚠️  WhisperX RE-ENCODES audio on CPU (wasteful!)")
            logger.warning("  ⚠️  Set USE_CUSTOM_DECODER=true to eliminate re-encoding")
            python_decoder = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
            # Extract feature extractor from model (Bug #1 fix)
            feature_extractor = python_decoder.model.feature_extractor
            logger.info("  WhisperX decoder loaded (slowest, RE-ENCODES)")

        # Load alignment model
        logger.info("[Init] Loading alignment model...")
        model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
        logger.info("  Alignment model loaded")

        # Print stats
        logger.info("\n" + "="*70)
        logger.info("  Initialization Complete")
        logger.info("="*70)
        logger.info(f"  Encoder: C++ with NPU (400-500x realtime)")
        logger.info(f"  Decoder: Python (WhisperX, for now)")
        logger.info(f"  Model: {MODEL_SIZE}")
        logger.info(f"  Device: {DEVICE}")
        logger.info("="*70 + "\n")

        return True

    except Exception as e:
        logger.error(f"Failed to initialize C++ encoder: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize encoder, buffer pools, and pipeline on service startup"""
    global buffer_manager, pipeline

    # Initialize C++ encoder
    success = initialize_encoder()
    if not success:
        logger.error("CRITICAL: Failed to initialize C++ encoder")
        logger.error("Service will not be able to process requests")
        raise RuntimeError("C++ encoder initialization failed")

    # Initialize buffer pools
    logger.info("="*70)
    logger.info("  Buffer Pool Initialization")
    logger.info("="*70)

    buffer_manager = GlobalBufferManager.instance()

    # Buffer pool configuration (user-configurable via environment)
    # Week 18 Fix: Make buffer sizes configurable for long-form audio support
    MAX_AUDIO_DURATION = int(os.getenv('MAX_AUDIO_DURATION', '30'))  # seconds (default: 30s)
    SAMPLE_RATE = 16000

    # Calculate buffer sizes dynamically
    MAX_AUDIO_SAMPLES = MAX_AUDIO_DURATION * SAMPLE_RATE
    MAX_MEL_FRAMES = (MAX_AUDIO_SAMPLES // 160) * 2  # hop_length=160, conv1d stride=2
    MAX_ENCODER_FRAMES = MAX_MEL_FRAMES

    logger.info(f"[BufferPool] Configured for audio up to {MAX_AUDIO_DURATION}s")
    logger.info(f"  Audio buffer: {MAX_AUDIO_SAMPLES:,} samples ({MAX_AUDIO_SAMPLES*4/1024/1024:.1f} MB per buffer)")
    logger.info(f"  Mel buffer: {MAX_MEL_FRAMES:,} frames ({MAX_MEL_FRAMES*80*4/1024/1024:.1f} MB per buffer)")
    logger.info(f"  Encoder buffer: {MAX_ENCODER_FRAMES:,} frames ({MAX_ENCODER_FRAMES*512*4/1024/1024:.1f} MB per buffer)")

    # Configure buffer pools based on calculated sizes
    buffer_manager.configure({
        'mel': {
            'size': MAX_MEL_FRAMES * 80 * 4,  # frames × mels × sizeof(float32)
            'count': 10,             # Pre-allocate 10 buffers
            'max_count': 20,         # Max 20 concurrent requests
            'dtype': np.float32,
            'shape': (MAX_MEL_FRAMES, 80),  # (frames, mels) - C-contiguous
            'zero_on_release': False # No need to zero (overwritten each time)
        },
        'audio': {
            'size': MAX_AUDIO_SAMPLES * 4,  # samples × sizeof(float32)
            'count': 5,              # Pre-allocate 5 buffers
            'max_count': 15,         # Max 15 concurrent requests
            'dtype': np.float32,
            'shape': (MAX_AUDIO_SAMPLES,),  # CRITICAL FIX: Must specify shape!
            'zero_on_release': False
        },
        'encoder_output': {
            'size': MAX_ENCODER_FRAMES * 512 * 4,  # frames × hidden × sizeof(float32)
            'count': 5,              # Pre-allocate 5 buffers
            'max_count': 15,         # Max 15 concurrent requests
            'dtype': np.float32,
            'shape': (MAX_ENCODER_FRAMES, 512),  # (frames, hidden)
            'zero_on_release': False
        }
    })

    # Calculate total pool memory
    stats = buffer_manager.get_stats()
    total_memory = 0
    for pool_name, pool_stats in stats.items():
        if pool_name == 'mel':
            pool_memory = pool_stats['total_buffers'] * (MAX_MEL_FRAMES * 80 * 4)
        elif pool_name == 'audio':
            pool_memory = pool_stats['total_buffers'] * (MAX_AUDIO_SAMPLES * 4)
        elif pool_name == 'encoder_output':
            pool_memory = pool_stats['total_buffers'] * (MAX_ENCODER_FRAMES * 512 * 4)
        else:
            pool_memory = 0
        total_memory += pool_memory

    logger.info(f"  Total pool memory: {total_memory / (1024*1024):.1f}MB")
    logger.info(f"  Per-buffer breakdown:")
    logger.info(f"    Mel: {stats['mel']['total_buffers']} × {MAX_MEL_FRAMES*80*4/1024/1024:.1f}MB = {stats['mel']['total_buffers']*MAX_MEL_FRAMES*80*4/1024/1024:.1f}MB")
    logger.info(f"    Audio: {stats['audio']['total_buffers']} × {MAX_AUDIO_SAMPLES*4/1024/1024:.1f}MB = {stats['audio']['total_buffers']*MAX_AUDIO_SAMPLES*4/1024/1024:.1f}MB")
    logger.info(f"    Encoder: {stats['encoder_output']['total_buffers']} × {MAX_ENCODER_FRAMES*512*4/1024/1024:.1f}MB = {stats['encoder_output']['total_buffers']*MAX_ENCODER_FRAMES*512*4/1024/1024:.1f}MB")
    logger.info("="*70 + "\n")

    # Initialize batch processor (Week 19)
    if ENABLE_BATCHING:
        logger.info("="*70)
        logger.info("  Week 19 Batch Processing Initialization")
        logger.info("="*70)

        batch_processor = BatchProcessor(
            max_batch_size=BATCH_MAX_SIZE,
            max_wait_ms=BATCH_MAX_WAIT_MS,
            encoder_callback=cpp_encoder.forward if cpp_encoder else None,
            decoder_callback=python_decoder,
            feature_extractor=feature_extractor,
            conv1d_preprocessor=conv1d_preprocessor,
            model_a=model_a,
            metadata=metadata,
            device=DEVICE,
            batch_size=BATCH_SIZE
        )

        # Start batch processing loop
        asyncio.create_task(batch_processor.process_batches())

        logger.info(f"  Mode: BATCHING (2-3× throughput improvement)")
        logger.info(f"  Max batch size: {BATCH_MAX_SIZE}")
        logger.info(f"  Max wait time: {BATCH_MAX_WAIT_MS}ms")
        logger.info(f"  Target throughput: 25-35× realtime")
        logger.info("="*70 + "\n")

    # Initialize multi-stream pipeline (if enabled and no batching)
    elif ENABLE_PIPELINE:
        logger.info("="*70)
        logger.info("  Multi-Stream Pipeline Initialization")
        logger.info("="*70)

        pipeline = TranscriptionPipeline(
            cpp_encoder=cpp_encoder,
            python_decoder=python_decoder,
            model_a=model_a,
            metadata=metadata,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            num_load_workers=NUM_LOAD_WORKERS,
            num_decoder_workers=NUM_DECODER_WORKERS,
            max_queue_size=MAX_QUEUE_SIZE
        )

        await pipeline.start()

        logger.info(f"  Mode: PIPELINE (concurrent)")
        logger.info(f"  Load workers: {NUM_LOAD_WORKERS}")
        logger.info(f"  Decoder workers: {NUM_DECODER_WORKERS}")
        logger.info(f"  Max queue size: {MAX_QUEUE_SIZE}")
        logger.info(f"  Target throughput: 67 req/s (+329%)")
        logger.info("="*70 + "\n")
    else:
        logger.info("="*70)
        logger.info("  Running in SEQUENTIAL mode (batching and pipeline disabled)")
        logger.info("  Set ENABLE_BATCHING=true or ENABLE_PIPELINE=true to enable optimization")
        logger.info("="*70 + "\n")

    logger.info("✅ All systems initialized successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown - stop pipeline, batch processor, and print statistics"""
    global buffer_manager, pipeline, batch_processor

    logger.info("\n" + "="*70)
    logger.info("  Service Shutdown")
    logger.info("="*70)

    # Stop batch processor (if running)
    if batch_processor:
        logger.info("[Shutdown] Batch processor statistics:")
        stats = await batch_processor.get_stats()
        logger.info(f"  Total requests: {stats['total_requests']}")
        logger.info(f"  Total batches: {stats['total_batches']}")
        logger.info(f"  Avg batch size: {stats['avg_batch_size']:.1f}")
        logger.info(f"  Avg wait time: {stats['avg_wait_time']*1000:.2f}ms")
        logger.info(f"  Avg processing time: {stats['avg_processing_time']:.3f}s")
        logger.info(f"  Total errors: {stats['total_errors']}")

    # Stop pipeline (if running)
    if pipeline:
        logger.info("[Shutdown] Stopping pipeline...")
        await pipeline.stop(drain_queues=True, timeout=30.0)
        logger.info("[Shutdown] Pipeline statistics:")
        await pipeline.print_stats()

    # Print buffer pool statistics
    if buffer_manager:
        logger.info("[Shutdown] Buffer pool statistics:")
        buffer_manager.print_stats()
        buffer_manager.shutdown()

    logger.info("="*70)
    logger.info("✅ Shutdown complete")
    logger.info("="*70 + "\n")


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Form(False),
    min_speakers: int = Form(None),
    max_speakers: int = Form(None)
):
    """
    Transcribe audio file with C++ NPU encoder and buffer pool optimization.

    OpenAI-compatible endpoint using:
    - C++ encoder for 400-500x realtime performance
    - Buffer pooling for 80% allocation overhead reduction
    - Zero-copy mel computation for minimal data copying
    - Multi-stream pipeline for +329% throughput (67 req/s)

    Args:
        file: Audio file (WAV, MP3, etc.)
        diarize: Enable speaker diarization
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers

    Returns:
        JSON response with transcription, segments, and words
    """
    global request_count, total_audio_seconds, total_processing_time

    if cpp_encoder is None or buffer_manager is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Service not initialized. C++ encoder or buffer pool not available."}
        )

    # Read audio data
    audio_data = await file.read()

    # BATCH PROCESSING MODE: Route through batch processor (Week 19)
    if ENABLE_BATCHING and batch_processor is not None:
        try:
            start_time = time.perf_counter()
            logger.info(f"[Batch Request {request_count + 1}] Processing: {file.filename}")

            # Load audio from bytes
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name

            try:
                audio = whisperx.load_audio(tmp_path)
                audio_duration = len(audio) / 16000  # 16kHz
            finally:
                os.unlink(tmp_path)

            # Submit to batch processor
            result = await batch_processor.submit_request(
                audio=audio,
                language=None  # Auto-detect
            )

            # Calculate total time
            total_time = time.perf_counter() - start_time
            overall_realtime = audio_duration / total_time if total_time > 0 else 0

            # Update stats
            request_count += 1
            total_audio_seconds += audio_duration
            total_processing_time += total_time

            logger.info(
                f"[Batch Request {request_count}] Complete: {total_time*1000:.1f}ms "
                f"({overall_realtime:.1f}× realtime)"
            )

            # Return result
            return {
                "text": result.text,
                "segments": result.segments,
                "language": result.language,
                "words": [],  # TODO: Add word-level timestamps if available
                "performance": {
                    "audio_duration_s": audio_duration,
                    "processing_time_s": result.processing_time,
                    "realtime_factor": audio_duration / result.processing_time if result.processing_time > 0 else 0,
                    "mode": "batching"
                }
            }

        except Exception as e:
            logger.error(f"[Batch Request] Error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Batch processing failed",
                    "details": str(e)
                }
            )

    # PIPELINE MODE: Route through multi-stream pipeline
    elif ENABLE_PIPELINE and pipeline is not None:
        import uuid as uuid_lib
        request_id = str(uuid_lib.uuid4())

        try:
            start_time = time.perf_counter()
            logger.info(f"[Pipeline Request {request_count + 1}] Processing: {file.filename} (ID: {request_id})")

            # Create queued request
            from request_queue import QueuedRequest
            queued_request = QueuedRequest(
                request_id=request_id,
                audio_data=audio_data,
                options={
                    'diarize': diarize,
                    'min_speakers': min_speakers,
                    'max_speakers': max_speakers,
                    'format': 'json'
                },
                priority=0
            )

            # Submit to pipeline (with 30s timeout)
            result = await pipeline.transcribe(queued_request, timeout=30.0)

            # Calculate total time
            total_time = time.perf_counter() - start_time

            # Estimate audio duration (rough approximation: 1 KB ~= 0.03s for 16kHz audio)
            audio_duration = len(audio_data) / (16000 * 2)  # 16kHz * 2 bytes per sample
            overall_realtime = audio_duration / total_time if total_time > 0 else 0

            # Update stats
            request_count += 1
            total_audio_seconds += audio_duration
            total_processing_time += total_time

            logger.info(
                f"[Pipeline Request {request_count}] Complete: {total_time*1000:.1f}ms "
                f"({overall_realtime:.1f}x realtime)"
            )

            # Return result (pipeline already has text, segments, words)
            return {
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
                "language": result.get("language", "en"),
                "words": result.get("words", []),
                "performance": {
                    "audio_duration_s": audio_duration,
                    "processing_time_s": total_time,
                    "realtime_factor": overall_realtime,
                    "mode": "pipeline"
                }
            }

        except asyncio.TimeoutError:
            logger.error(f"[Pipeline Request] Timeout after 30s for request {request_id}")
            return JSONResponse(
                status_code=504,
                content={"error": "Request timeout after 30s"}
            )

        except Exception as e:
            logger.error(f"[Pipeline Request] Error for request {request_id}: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Pipeline processing failed",
                    "details": str(e)
                }
            )

    # SEQUENTIAL MODE: Fallback to original implementation
    logger.info(
        f"[Sequential Request {request_count + 1}] Processing: {file.filename} "
        "(pipeline disabled)"
    )

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    # Buffer pool tracking for cleanup
    mel_buffer = None
    audio_buffer = None
    encoder_buffer = None

    try:
        start_time = time.perf_counter()
        logger.info(f"[Request {request_count + 1}] Processing: {file.filename}")

        # 1. Load audio (Python - WhisperX)
        logger.info("  [1/5] Loading audio...")
        audio = whisperx.load_audio(tmp_path)
        audio_duration = len(audio) / 16000  # WhisperX uses 16kHz
        logger.info(f"    Audio duration: {audio_duration:.2f}s")

        # Acquire audio buffer from pool
        audio_buffer = buffer_manager.acquire('audio')
        np.copyto(audio_buffer[:len(audio)], audio)  # Copy into pooled buffer
        logger.debug(f"    Acquired audio buffer from pool")

        # 2. Compute mel spectrogram with buffer pool + zero-copy optimization
        logger.info("  [2/5] Computing mel spectrogram (pooled + zero-copy)...")
        mel_start = time.perf_counter()

        # Acquire mel buffer from pool
        mel_buffer = buffer_manager.acquire('mel')

        # Zero-Copy + Buffer Pool Optimization:
        # Compute mel directly into pooled buffer with C-contiguous (time, channels) layout
        # Eliminates BOTH allocation AND transpose copy (~2ms total savings)
        # Returns a SLICE of mel_buffer with correct size for variable-length audio
        mel_np, actual_frames = compute_mel_spectrogram_zerocopy(
            audio_buffer[:len(audio)],
            feature_extractor,  # Use global feature_extractor (Bug #1 fix)
            output=mel_buffer  # Write directly to pooled buffer (may be larger than needed)
        )

        mel_time = time.perf_counter() - mel_start
        logger.info(f"    Mel computation: {mel_time*1000:.2f}ms ({actual_frames} frames, pooled + zero-copy)")

        # Validate mel is ready for C++ encoder (should never fail with zero-copy)
        validate_mel_contiguity(mel_np)

        # 2.5. Apply conv1d preprocessing (Bug #5 fix: mel 80→512)
        logger.info("  [2.5/5] Applying conv1d preprocessing (mel→embeddings)...")
        conv1d_start = time.perf_counter()
        embeddings = conv1d_preprocessor.process(mel_np)  # (n_frames, 80) → (n_frames//2, 512)
        conv1d_time = time.perf_counter() - conv1d_start
        logger.info(f"    Conv1d time: {conv1d_time*1000:.2f}ms ({mel_np.shape[0]} frames → {embeddings.shape[0]} frames)")

        # 3. Run C++ encoder (NPU-accelerated)
        logger.info("  [3/5] Running C++ encoder (NPU)...")
        encoder_start = time.perf_counter()

        # Acquire encoder output buffer from pool
        encoder_buffer = buffer_manager.acquire('encoder_output')

        # Run C++ encoder on embeddings (not raw mel!)
        # Note: Current encoder returns allocated buffer, future optimization would write to provided buffer
        encoder_output = cpp_encoder.forward(embeddings)  # (n_frames//2, 512) → (n_frames//2, 512)
        encoder_time = time.perf_counter() - encoder_start

        realtime_factor = audio_duration / encoder_time if encoder_time > 0 else 0
        logger.info(f"    Encoder time: {encoder_time*1000:.2f}ms")
        logger.info(f"    Realtime factor: {realtime_factor:.1f}x")

        # 4. Run Python decoder (CustomWhisperDecoder, faster-whisper, or WhisperX)
        logger.info("  [4/5] Running decoder...")
        decoder_start = time.perf_counter()

        # Check if using CustomWhisperDecoder (Week 19.5 fix)
        is_custom_decoder = hasattr(python_decoder, 'transcribe_from_features')

        # Check if using faster-whisper
        is_faster_whisper = hasattr(python_decoder, 'get_stats')

        if is_custom_decoder:
            # CustomWhisperDecoder (Week 19.5 Architecture Fix - FAST!)
            # Accepts NPU encoder output directly (NO RE-ENCODING!)
            logger.info("    Using CustomWhisperDecoder with NPU features (no re-encoding)")
            result = python_decoder.transcribe_from_features(
                encoder_output,  # ✅ USE NPU ENCODER OUTPUT DIRECTLY!
                language="en",   # Force English for speed
                word_timestamps=False  # Alignment will add these
            )
            decoder_backend = "CustomWhisperDecoder (no re-encoding)"

        elif is_faster_whisper:
            # faster-whisper decoder (Week 19 optimization - 4-6× faster)
            # WARNING: This re-encodes audio on CPU (wasteful!)
            logger.warning(
                "    Using faster-whisper - will RE-ENCODE audio on CPU! "
                "Switch to CustomWhisperDecoder to eliminate re-encoding."
            )
            result = python_decoder.transcribe(
                audio,  # ❌ RE-ENCODES (300-3,200ms wasted)
                language="en",
                word_timestamps=False,
                vad_filter=False
            )
            decoder_backend = "faster-whisper (with re-encoding)"

        else:
            # WhisperX decoder (legacy path)
            # WARNING: This re-encodes audio on CPU (wasteful!)
            logger.warning(
                "    Using WhisperX - will RE-ENCODE audio on CPU! "
                "Switch to CustomWhisperDecoder to eliminate re-encoding."
            )

            # Zero-Copy Optimization: encoder output is on CPU, decoder is on CPU
            # torch.from_numpy() creates a view (zero-copy)
            # .to(DEVICE) is a no-op when DEVICE='cpu' (zero-copy)
            encoder_output_torch = torch.from_numpy(encoder_output).unsqueeze(0).to(DEVICE)

            # Use WhisperX decoder
            # Note: This is a simplification - actual integration needs proper pipeline
            # For now, we'll use the full WhisperX pipeline but note encoder was C++
            result = python_decoder.transcribe(
                audio,  # ❌ RE-ENCODES (300-3,200ms wasted)
                batch_size=BATCH_SIZE
            )
            decoder_backend = "WhisperX (with re-encoding)"

        decoder_time = time.perf_counter() - decoder_start
        logger.info(f"    Decoder time: {decoder_time*1000:.2f}ms ({decoder_backend})")

        # 5. Align whisper output
        logger.info("  [5/5] Aligning...")
        align_start = time.perf_counter()
        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE)
        align_time = time.perf_counter() - align_start
        logger.info(f"    Alignment time: {align_time*1000:.2f}ms")

        # Optional: Speaker diarization
        if diarize and HF_TOKEN:
            logger.info("  [Optional] Speaker diarization...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
            diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        # Format response
        text = " ".join([segment["text"] for segment in result["segments"]])

        # Performance metrics
        total_time = time.perf_counter() - start_time
        overall_realtime = audio_duration / total_time if total_time > 0 else 0

        # Update stats
        request_count += 1
        total_audio_seconds += audio_duration
        total_processing_time += total_time

        logger.info(f"\n  Performance Summary:")
        logger.info(f"    Total time: {total_time*1000:.2f}ms")
        logger.info(f"    Overall realtime: {overall_realtime:.1f}x")
        logger.info(f"    Encoder contribution: {encoder_time/total_time*100:.1f}%")
        logger.info(f"  Request {request_count} complete\n")

        return {
            "text": text,
            "segments": result["segments"],
            "language": result.get("language", "en"),
            "words": result.get("word_segments", []),
            "performance": {
                "audio_duration_s": audio_duration,
                "processing_time_s": total_time,
                "realtime_factor": overall_realtime,
                "encoder_time_ms": encoder_time * 1000,
                "encoder_realtime_factor": realtime_factor,
                "mel_time_ms": mel_time * 1000,
                "decoder_time_ms": decoder_time * 1000,
                "align_time_ms": align_time * 1000,
                "mode": "sequential"
            }
        }

    except CPPRuntimeError as e:
        logger.error(f"C++ encoder error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "C++ encoder failed",
                "details": str(e)
            }
        )

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": "Processing failed",
                "details": str(e)
            }
        )

    finally:
        # CRITICAL: Always release buffers back to pool
        if mel_buffer is not None:
            buffer_manager.release('mel', mel_buffer)
            logger.debug("  Released mel buffer")

        if audio_buffer is not None:
            buffer_manager.release('audio', audio_buffer)
            logger.debug("  Released audio buffer")

        if encoder_buffer is not None:
            buffer_manager.release('encoder_output', encoder_buffer)
            logger.debug("  Released encoder buffer")

        os.unlink(tmp_path)
        gc.collect()


@app.get("/stats/batching")
async def batching_stats():
    """
    Get Week 19 batch processing statistics.

    Returns comprehensive statistics for batch processing including:
    - Total requests and batches
    - Average batch size
    - Average wait time and processing time
    - Throughput metrics
    - Error rates

    Returns:
        Batch processing statistics or indication that batching is disabled
    """
    if not ENABLE_BATCHING or batch_processor is None:
        return {
            "enabled": False,
            "mode": "pipeline" if ENABLE_PIPELINE else "sequential",
            "message": "Batching disabled. Set ENABLE_BATCHING=true to enable."
        }

    try:
        stats = await batch_processor.get_stats()
        config = batch_processor.get_config()

        # Calculate throughput
        if stats['avg_processing_time'] > 0:
            throughput_rps = stats['avg_batch_size'] / stats['avg_processing_time']
        else:
            throughput_rps = 0.0

        return {
            "enabled": True,
            "mode": "batching",
            "throughput_rps": round(throughput_rps, 2),
            "total_requests": stats['total_requests'],
            "total_batches": stats['total_batches'],
            "avg_batch_size": round(stats['avg_batch_size'], 2),
            "avg_wait_time_ms": round(stats['avg_wait_time'] * 1000, 2),
            "avg_processing_time_s": round(stats['avg_processing_time'], 3),
            "total_errors": stats['total_errors'],
            "queue_depth": stats['queue_depth'],
            "pending_results": stats['pending_results'],
            "configuration": {
                "max_batch_size": config['max_batch_size'],
                "max_wait_ms": config['max_wait_ms'],
                "device": config['device'],
                "encoder_enabled": config['encoder_enabled'],
                "decoder_enabled": config['decoder_enabled']
            }
        }

    except Exception as e:
        logger.error(f"Error getting batch stats: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to retrieve batch statistics",
                "details": str(e)
            }
        )


@app.get("/stats/pipeline")
async def pipeline_stats():
    """
    Get multi-stream pipeline statistics.

    Returns comprehensive statistics for all pipeline stages including:
    - Overall throughput (requests per second)
    - Queue depth and utilization
    - Active requests in flight
    - Per-stage processing times and queue depths
    - Error rates and timeout rates

    Returns:
        Pipeline statistics or indication that pipeline is disabled
    """
    if not ENABLE_PIPELINE or pipeline is None:
        return {
            "enabled": False,
            "mode": "sequential",
            "message": "Pipeline disabled. Set ENABLE_PIPELINE=true to enable."
        }

    try:
        stats = await pipeline.get_stats()

        # Calculate overall throughput
        queue_stats = stats.get('queue', {})
        total_processed = queue_stats.get('total_dequeued', 0)

        # Calculate requests per second (use Stage 3 completion rate)
        stage3_stats = stats.get('stage3', {})
        stage3_processed = stage3_stats.get('total_processed', 0)
        stage3_time = stage3_stats.get('total_time', 0.0)

        if stage3_time > 0:
            throughput_rps = stage3_processed / stage3_time
        else:
            throughput_rps = 0.0

        return {
            "enabled": True,
            "mode": "pipeline",
            "throughput_rps": round(throughput_rps, 2),
            "queue": {
                "depth": queue_stats.get('current_size', 0),
                "max_size": queue_stats.get('max_queue_size', 0),
                "utilization": queue_stats.get('utilization', 0.0),
                "total_enqueued": queue_stats.get('total_enqueued', 0),
                "total_dequeued": queue_stats.get('total_dequeued', 0),
                "avg_wait_time_ms": queue_stats.get('avg_wait_time', 0.0) * 1000
            },
            "active_requests": stats.get('pipeline', {}).get('pending_responses', 0),
            "stages": {
                "stage1_load_mel": {
                    "total_processed": stats.get('stage1', {}).get('total_processed', 0),
                    "avg_time_ms": stats.get('stage1', {}).get('avg_time', 0.0) * 1000,
                    "queue_depth": stats.get('stage1', {}).get('input_queue_size', 0),
                    "workers_active": stats.get('stage1', {}).get('workers_active', 0),
                    "error_rate": stats.get('stage1', {}).get('error_rate', 0.0)
                },
                "stage2_encoder": {
                    "total_processed": stats.get('stage2', {}).get('total_processed', 0),
                    "avg_time_ms": stats.get('stage2', {}).get('avg_time', 0.0) * 1000,
                    "queue_depth": stats.get('stage2', {}).get('input_queue_size', 0),
                    "workers_active": stats.get('stage2', {}).get('workers_active', 0),
                    "error_rate": stats.get('stage2', {}).get('error_rate', 0.0)
                },
                "stage3_decoder_align": {
                    "total_processed": stats.get('stage3', {}).get('total_processed', 0),
                    "avg_time_ms": stats.get('stage3', {}).get('avg_time', 0.0) * 1000,
                    "queue_depth": stats.get('stage3', {}).get('input_queue_size', 0),
                    "workers_active": stats.get('stage3', {}).get('workers_active', 0),
                    "error_rate": stats.get('stage3', {}).get('error_rate', 0.0)
                }
            }
        }

    except Exception as e:
        logger.error(f"Error getting pipeline stats: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to retrieve pipeline statistics",
                "details": str(e)
            }
        )


@app.get("/health/pipeline")
async def pipeline_health():
    """
    Check pipeline health status.

    Returns health information for each pipeline stage and overall health status.
    Useful for monitoring and alerting.

    Returns:
        Health status with per-stage details
    """
    if not ENABLE_PIPELINE or pipeline is None:
        return {
            "healthy": False,
            "mode": "sequential",
            "reason": "Pipeline disabled. Set ENABLE_PIPELINE=true to enable."
        }

    try:
        # Check overall pipeline health
        is_healthy = pipeline.is_healthy()

        # Get detailed stats for health assessment
        stats = await pipeline.get_stats()

        # Check each stage
        stage_health = {}

        for stage_name in ['stage1', 'stage2', 'stage3']:
            stage_stats = stats.get(stage_name, {})
            stage_running = stage_stats.get('is_running', False)
            workers_active = stage_stats.get('workers_active', 0)
            workers_total = stage_stats.get('workers_total', 0)
            error_rate = stage_stats.get('error_rate', 0.0)

            stage_healthy = (
                stage_running and
                workers_active > 0 and
                error_rate < 0.5  # Less than 50% error rate
            )

            stage_health[stage_name] = {
                "healthy": stage_healthy,
                "running": stage_running,
                "workers_active": workers_active,
                "workers_total": workers_total,
                "error_rate": error_rate
            }

        # Overall health is healthy if all stages are healthy
        all_healthy = all(s["healthy"] for s in stage_health.values())

        return {
            "healthy": all_healthy and is_healthy,
            "mode": "pipeline",
            "stages": stage_health,
            "message": "All stages healthy" if all_healthy else "One or more stages unhealthy"
        }

    except Exception as e:
        logger.error(f"Error checking pipeline health: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "healthy": False,
                "error": "Health check failed",
                "details": str(e)
            }
        )


@app.get("/health")
async def health():
    """
    Enhanced health check with C++ runtime and buffer pool status.

    Returns:
        Service health status including encoder and buffer pool statistics
    """
    if cpp_encoder is None or buffer_manager is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": "C++ encoder or buffer pool not initialized"
            }
        )

    try:
        encoder_stats = cpp_encoder.get_stats()
        buffer_stats = buffer_manager.get_stats()

        uptime = time.time() - startup_time
        avg_realtime = total_audio_seconds / total_processing_time if total_processing_time > 0 else 0

        # Check for buffer pool health issues
        warnings = []
        for pool_name, pool_stats in buffer_stats.items():
            if pool_stats.get('has_leaks', False):
                warnings.append(f"Pool '{pool_name}' has {pool_stats['leaked_buffers']} leaked buffers")
            if pool_stats.get('hit_rate', 1.0) < 0.90 and pool_stats.get('total_acquires', 0) > 10:
                warnings.append(f"Pool '{pool_name}' has low hit rate: {pool_stats['hit_rate']*100:.1f}%")

        return {
            "status": "healthy" if not warnings else "degraded",
            "service": "Unicorn-Amanuensis XDNA2 C++ + Buffer Pool",
            "version": "2.1.0",  # Incremented for buffer pool support
            "backend": "C++ encoder with NPU + Buffer pooling",
            "model": MODEL_SIZE,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "uptime_seconds": uptime,
            "encoder": {
                "type": "C++ with NPU",
                "runtime_version": encoder_stats.get("runtime_version", "unknown"),
                "num_layers": encoder_stats.get("num_layers", 6),
                "npu_enabled": encoder_stats.get("use_npu", False),
                "weights_loaded": encoder_stats.get("weights_loaded", False)
            },
            "buffer_pools": {
                pool_name: {
                    "hit_rate": pool_stats.get("hit_rate", 0.0),
                    "buffers_available": pool_stats.get("buffers_available", 0),
                    "buffers_in_use": pool_stats.get("buffers_in_use", 0),
                    "total_buffers": pool_stats.get("total_buffers", 0),
                    "has_leaks": pool_stats.get("has_leaks", False)
                }
                for pool_name, pool_stats in buffer_stats.items()
            },
            "performance": {
                "requests_processed": request_count,
                "total_audio_seconds": total_audio_seconds,
                "total_processing_seconds": total_processing_time,
                "average_realtime_factor": avg_realtime,
                "target_realtime_factor": 400
            },
            "warnings": warnings
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "degraded",
                "error": str(e)
            }
        )


@app.get("/stats")
async def stats():
    """
    Detailed performance statistics.

    Returns:
        Encoder statistics, buffer pool statistics, and performance metrics
    """
    if cpp_encoder is None or buffer_manager is None:
        return {"error": "Encoder or buffer pool not initialized"}

    encoder_stats = cpp_encoder.get_stats()
    buffer_stats = buffer_manager.get_stats()

    return {
        "service": "Unicorn-Amanuensis XDNA2 C++ + Buffer Pool",
        "version": "2.1.0",
        "encoder": encoder_stats,
        "buffer_pools": buffer_stats,
        "requests": {
            "total": request_count,
            "audio_seconds": total_audio_seconds,
            "processing_seconds": total_processing_time,
            "average_realtime": total_audio_seconds / total_processing_time if total_processing_time > 0 else 0
        }
    }


@app.get("/")
async def root():
    """Root endpoint with service information"""
    mode = "pipeline" if (ENABLE_PIPELINE and pipeline is not None) else "sequential"
    target_throughput = "67 req/s (+329%)" if mode == "pipeline" else "15.6 req/s"

    return {
        "service": "Unicorn-Amanuensis XDNA2 C++ + Multi-Stream Pipeline",
        "description": "Speech-to-Text with C++ NPU Encoder + Concurrent Processing",
        "version": "3.0.0",
        "backend": "C++ encoder (400-500x realtime) + Python decoder",
        "model": MODEL_SIZE,
        "mode": mode,
        "performance_target": target_throughput,
        "endpoints": {
            "/v1/audio/transcriptions": "POST - Transcribe audio (OpenAI-compatible)",
            "/health": "GET - Health check with encoder stats",
            "/health/pipeline": "GET - Pipeline health status",
            "/stats": "GET - Detailed performance statistics",
            "/stats/pipeline": "GET - Pipeline statistics",
            "/": "GET - This information"
        }
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Unicorn-Amanuensis XDNA2 C++ Backend...")
    uvicorn.run(app, host="0.0.0.0", port=9000)
