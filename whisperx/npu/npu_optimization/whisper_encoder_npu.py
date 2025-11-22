#!/usr/bin/env python3
"""
Whisper Encoder NPU Integration
================================
Chains custom NPU kernels for complete Whisper encoder layer execution
on AMD Phoenix NPU (XDNA1).

This module provides:
- WhisperEncoderNPU: Main class for NPU-accelerated encoder
- XRT buffer management with proper alignment
- Kernel chaining for attention and FFN blocks
- Support for all Whisper model sizes (tiny to large)

Usage:
    encoder = WhisperEncoderNPU(model_size="base")
    encoder.initialize()
    output = encoder.encoder_layer(hidden_states)

Author: Integration Architecture Team Lead
Date: November 18, 2025
Status: Design Phase - Skeleton Implementation
"""

import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for different Whisper model sizes"""
    hidden_dim: int
    num_heads: int
    head_dim: int
    ffn_dim: int
    num_layers: int

    @classmethod
    def from_size(cls, size: str) -> 'ModelConfig':
        """Get model configuration by size name"""
        configs = {
            "tiny": cls(384, 6, 64, 1536, 4),
            "base": cls(512, 8, 64, 2048, 6),
            "small": cls(768, 12, 64, 3072, 12),
            "medium": cls(1024, 16, 64, 4096, 24),
            "large": cls(1280, 20, 64, 5120, 32),
        }
        if size not in configs:
            raise ValueError(f"Unknown model size: {size}. Choose from {list(configs.keys())}")
        return configs[size]


class XRTBufferManager:
    """Manages XRT buffer allocation and data transfer for NPU kernels"""

    def __init__(self, device=None, kernel=None):
        """
        Initialize buffer manager.

        Args:
            device: XRT device handle
            kernel: XRT kernel handle
        """
        self.device = device
        self.kernel = kernel
        self.buffers: Dict[str, Dict] = {}
        self.total_allocated = 0

    def allocate(self, name: str, size_bytes: int, group_id: int = 0) -> Any:
        """
        Allocate an XRT buffer with 4KB alignment.

        Args:
            name: Buffer identifier
            size_bytes: Required size in bytes
            group_id: Kernel argument group ID

        Returns:
            XRT buffer object
        """
        # Round up to 4KB page boundary for DMA alignment
        aligned_size = ((size_bytes + 4095) // 4096) * 4096

        try:
            import xrt_binding as xrt
            bo = xrt.bo(self.device, aligned_size, xrt.bo.flags.normal, group_id)
        except ImportError:
            # Fallback for testing without XRT
            bo = np.zeros(aligned_size, dtype=np.uint8)
            logger.warning(f"XRT not available, using numpy buffer for {name}")

        self.buffers[name] = {
            'bo': bo,
            'size': aligned_size,
            'requested_size': size_bytes,
            'group_id': group_id
        }
        self.total_allocated += aligned_size

        logger.debug(f"Allocated buffer '{name}': {size_bytes} bytes (aligned to {aligned_size})")
        return bo

    def write(self, name: str, data: np.ndarray) -> None:
        """
        Write numpy array to XRT buffer.

        Args:
            name: Buffer identifier
            data: Data to write (will be converted to bytes)
        """
        if name not in self.buffers:
            raise KeyError(f"Buffer '{name}' not found")

        bo = self.buffers[name]['bo']

        try:
            import xrt_binding as xrt
            bo.write(data.tobytes())
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        except (ImportError, AttributeError):
            # Fallback for testing
            np.copyto(bo[:len(data.tobytes())], np.frombuffer(data.tobytes(), dtype=np.uint8))

    def read(self, name: str, dtype: np.dtype, shape: Tuple) -> np.ndarray:
        """
        Read XRT buffer to numpy array.

        Args:
            name: Buffer identifier
            dtype: Output data type
            shape: Output shape

        Returns:
            Numpy array with buffer contents
        """
        if name not in self.buffers:
            raise KeyError(f"Buffer '{name}' not found")

        bo = self.buffers[name]['bo']
        size_bytes = np.prod(shape) * np.dtype(dtype).itemsize

        try:
            import xrt_binding as xrt
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            return np.frombuffer(bo.read(size_bytes), dtype=dtype).reshape(shape)
        except (ImportError, AttributeError):
            # Fallback for testing
            return np.frombuffer(bo[:size_bytes].tobytes(), dtype=dtype).reshape(shape)

    def cleanup(self) -> None:
        """Release all allocated buffers"""
        self.buffers.clear()
        self.total_allocated = 0


class WhisperEncoderNPU:
    """
    NPU-accelerated Whisper encoder using custom MLIR-AIE2 kernels.

    This class chains softmax, GELU, LayerNorm, and MatMul kernels
    to execute complete encoder layers on the AMD Phoenix NPU.
    """

    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper Encoder NPU.

        Args:
            model_size: One of "tiny", "base", "small", "medium", "large"
        """
        self.model_size = model_size
        self.config = ModelConfig.from_size(model_size)

        # Kernel paths
        self.kernel_dir = Path(__file__).parent / "whisper_encoder_kernels" / "kernels_xdna1"

        # XCLBIN paths for each kernel
        self.xclbin_paths = {
            'softmax': self.kernel_dir / "build_softmax_bf16" / "softmax_bf16.xclbin",
            'softmax_parallel': self.kernel_dir / "build_softmax_parallel" / "softmax_parallel.xclbin",
            'gelu': self.kernel_dir / "build_gelu" / "gelu_bf16.xclbin",
            'layernorm': self.kernel_dir / "build_layernorm" / "layernorm_bf16.xclbin",
            # MatMul XCLBIN to be added when compiled
            'matmul': self.kernel_dir / "build_matmul" / "matmul_bf16.xclbin",
        }

        # XRT handles
        self.device = None
        self.kernels: Dict[str, Any] = {}

        # Buffer manager
        self.buffer_manager = None

        # Weights (loaded from model)
        self.weights: Dict[str, np.ndarray] = {}

        # State
        self.is_initialized = False

        logger.info(f"WhisperEncoderNPU created for {model_size} model")
        logger.info(f"  Hidden dim: {self.config.hidden_dim}")
        logger.info(f"  Num heads: {self.config.num_heads}")
        logger.info(f"  Num layers: {self.config.num_layers}")

    def initialize(self) -> bool:
        """
        Initialize NPU device and load all XCLBINs.

        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing WhisperEncoderNPU...")

            # Step 1: Open NPU device
            if not self._open_device():
                logger.error("Failed to open NPU device")
                return False

            # Step 2: Load XCLBINs
            if not self._load_xclbins():
                logger.warning("Some XCLBINs not loaded, functionality may be limited")

            # Step 3: Initialize buffer manager
            self.buffer_manager = XRTBufferManager(self.device)

            # Step 4: Allocate persistent buffers
            self._allocate_persistent_buffers()

            self.is_initialized = True
            logger.info("WhisperEncoderNPU initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def _open_device(self) -> bool:
        """Open XRT device for NPU access"""
        try:
            import xrt_binding as xrt

            # Find NPU device
            device_id = 0
            for i in range(xrt.device.get_num_devices()):
                dev = xrt.device(i)
                device_name = dev.get_info("name")
                if "NPU" in device_name or "accel" in device_name:
                    device_id = i
                    break

            self.device = xrt.device(device_id)
            logger.info(f"Opened NPU device {device_id}")
            return True

        except ImportError:
            logger.warning("XRT not available, using CPU simulation mode")
            self.device = "CPU_SIMULATION"
            return True
        except Exception as e:
            logger.error(f"Failed to open device: {e}")
            return False

    def _load_xclbins(self) -> bool:
        """Load all compiled XCLBIN kernels"""
        loaded_count = 0

        for name, path in self.xclbin_paths.items():
            if not path.exists():
                logger.warning(f"XCLBIN not found: {path}")
                continue

            try:
                if self.device == "CPU_SIMULATION":
                    # Simulation mode - just note the kernel exists
                    self.kernels[name] = {"path": path, "simulated": True}
                    logger.info(f"Registered kernel '{name}' (simulation mode)")
                else:
                    # Real XRT loading
                    import xrt_binding as xrt
                    xclbin = xrt.xclbin(str(path))
                    self.device.load_xclbin(xclbin)

                    # Get kernel handle
                    kernel_name = xclbin.get_kernels()[0].get_name()
                    kernel = xrt.kernel(self.device, xclbin.get_uuid(), kernel_name)

                    self.kernels[name] = kernel
                    logger.info(f"Loaded kernel '{name}' from {path}")

                loaded_count += 1

            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")

        logger.info(f"Loaded {loaded_count}/{len(self.xclbin_paths)} kernels")
        return loaded_count > 0

    def _allocate_persistent_buffers(self) -> None:
        """Allocate buffers that persist across all forward passes"""
        D = self.config.hidden_dim
        H = self.config.num_heads
        FFN = self.config.ffn_dim

        # Typical batch size and sequence length
        B = 8  # Optimal batch size for NPU
        T = 100  # Time steps per second of audio

        # Activation buffers (ping-pong for DMA overlap)
        activation_size = B * T * D * 2  # BF16 = 2 bytes
        self.buffer_manager.allocate('input_ping', activation_size, group_id=0)
        self.buffer_manager.allocate('input_pong', activation_size, group_id=0)
        self.buffer_manager.allocate('output_ping', activation_size, group_id=1)
        self.buffer_manager.allocate('output_pong', activation_size, group_id=1)

        # Intermediate buffers
        qkv_size = B * 3 * H * T * 64 * 2  # Q, K, V for all heads
        self.buffer_manager.allocate('qkv_buffer', qkv_size, group_id=2)

        scores_size = B * H * T * T * 2  # Attention scores
        self.buffer_manager.allocate('attn_scores', scores_size, group_id=2)

        ffn_size = B * T * FFN * 2  # FFN intermediate
        self.buffer_manager.allocate('ffn_hidden', ffn_size, group_id=2)

        # Instruction buffer (runtime sequence)
        self.buffer_manager.allocate('instructions', 4096, group_id=3)

        logger.info(f"Allocated {self.buffer_manager.total_allocated / 1024:.1f} KB in persistent buffers")

    def load_weights(self, weights_dict: Dict[str, np.ndarray]) -> None:
        """
        Load encoder layer weights.

        Args:
            weights_dict: Dictionary mapping weight names to numpy arrays
                Expected keys per layer:
                - 'layer_{i}.self_attn.q_proj.weight'
                - 'layer_{i}.self_attn.k_proj.weight'
                - 'layer_{i}.self_attn.v_proj.weight'
                - 'layer_{i}.self_attn.out_proj.weight'
                - 'layer_{i}.self_attn_layer_norm.weight'
                - 'layer_{i}.self_attn_layer_norm.bias'
                - 'layer_{i}.fc1.weight' (FFN up)
                - 'layer_{i}.fc2.weight' (FFN down)
                - 'layer_{i}.final_layer_norm.weight'
                - 'layer_{i}.final_layer_norm.bias'
        """
        self.weights = weights_dict

        # Convert to BF16 if not already
        for name, weight in self.weights.items():
            if weight.dtype != np.uint16:  # BF16 stored as uint16
                # Convert FP32 to BF16 by truncating mantissa
                self.weights[name] = self._fp32_to_bf16(weight)

        logger.info(f"Loaded {len(weights_dict)} weight tensors")

    def _fp32_to_bf16(self, data: np.ndarray) -> np.ndarray:
        """Convert FP32 array to BF16 (stored as uint16)"""
        # View as uint32, shift right by 16 to get upper 16 bits
        fp32_as_int = data.astype(np.float32).view(np.uint32)
        bf16_as_int = (fp32_as_int >> 16).astype(np.uint16)
        return bf16_as_int

    def _bf16_to_fp32(self, data: np.ndarray) -> np.ndarray:
        """Convert BF16 (stored as uint16) to FP32"""
        # Shift left by 16 to get FP32
        bf16_as_int = data.astype(np.uint32) << 16
        return bf16_as_int.view(np.float32)

    def execute_kernel(self, kernel_name: str, input_data: np.ndarray,
                       output_shape: Tuple, **kwargs) -> np.ndarray:
        """
        Execute a single NPU kernel.

        Args:
            kernel_name: Name of kernel to execute
            input_data: Input tensor
            output_shape: Expected output shape
            **kwargs: Additional kernel-specific parameters

        Returns:
            Output tensor from kernel execution
        """
        if kernel_name not in self.kernels:
            raise RuntimeError(f"Kernel '{kernel_name}' not loaded")

        kernel = self.kernels[kernel_name]

        # For simulation mode
        if isinstance(kernel, dict) and kernel.get('simulated'):
            return self._simulate_kernel(kernel_name, input_data, output_shape, **kwargs)

        # Real NPU execution
        return self._execute_on_npu(kernel_name, kernel, input_data, output_shape, **kwargs)

    def _execute_on_npu(self, kernel_name: str, kernel: Any,
                        input_data: np.ndarray, output_shape: Tuple,
                        **kwargs) -> np.ndarray:
        """Execute kernel on actual NPU hardware"""
        import xrt_binding as xrt

        # Prepare buffers
        input_size = input_data.nbytes
        output_size = np.prod(output_shape) * 2  # BF16

        bo_input = self.buffer_manager.allocate(f'{kernel_name}_in', input_size, 3)
        bo_output = self.buffer_manager.allocate(f'{kernel_name}_out', output_size, 4)

        # Load instructions
        instr_path = self.kernel_dir / f"build_{kernel_name}" / "insts.bin"
        if instr_path.exists():
            insts = np.fromfile(instr_path, dtype=np.uint8)
        else:
            insts = np.zeros(64, dtype=np.uint8)

        bo_instr = self.buffer_manager.buffers['instructions']['bo']
        bo_instr.write(insts.tobytes())
        bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Write input
        self.buffer_manager.write(f'{kernel_name}_in', input_data)

        # Execute kernel with validated 5-parameter pattern
        opcode = 3  # Standard NPU kernel opcode
        run = kernel(opcode, bo_instr, len(insts),
                    self.buffer_manager.buffers[f'{kernel_name}_in']['bo'],
                    self.buffer_manager.buffers[f'{kernel_name}_out']['bo'])
        run.wait()

        # Read output
        output = self.buffer_manager.read(f'{kernel_name}_out', np.uint16, output_shape)

        return output

    def _simulate_kernel(self, kernel_name: str, input_data: np.ndarray,
                         output_shape: Tuple, **kwargs) -> np.ndarray:
        """CPU simulation of kernel for testing"""
        # Convert BF16 to FP32 for computation
        if input_data.dtype == np.uint16:
            input_fp32 = self._bf16_to_fp32(input_data)
        else:
            input_fp32 = input_data.astype(np.float32)

        if kernel_name in ['softmax', 'softmax_parallel']:
            # Softmax simulation
            exp_x = np.exp(input_fp32 - np.max(input_fp32, axis=-1, keepdims=True))
            output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        elif kernel_name == 'gelu':
            # GELU simulation (tanh approximation)
            x = input_fp32
            output = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

        elif kernel_name == 'layernorm':
            # LayerNorm simulation
            eps = kwargs.get('eps', 1e-5)
            mean = np.mean(input_fp32, axis=-1, keepdims=True)
            var = np.var(input_fp32, axis=-1, keepdims=True)
            output = (input_fp32 - mean) / np.sqrt(var + eps)
            # Note: gamma/beta should be applied here

        elif kernel_name == 'matmul':
            # MatMul simulation
            weights = kwargs.get('weights')
            if weights is None:
                # Identity-like placeholder
                output = input_fp32
            else:
                if weights.dtype == np.uint16:
                    weights = self._bf16_to_fp32(weights)
                output = np.matmul(input_fp32, weights)
        else:
            # Unknown kernel - return zeros
            logger.warning(f"Unknown kernel '{kernel_name}', returning zeros")
            output = np.zeros(output_shape, dtype=np.float32)

        # Reshape to expected output
        output = output.reshape(output_shape)

        # Convert back to BF16
        return self._fp32_to_bf16(output)

    def encoder_layer(self, hidden_states: np.ndarray, layer_idx: int = 0) -> np.ndarray:
        """
        Execute one complete encoder layer on NPU.

        Args:
            hidden_states: Input tensor [B, T, D] in BF16 format
            layer_idx: Layer index for weight lookup

        Returns:
            Output tensor [B, T, D] in BF16 format
        """
        if not self.is_initialized:
            raise RuntimeError("WhisperEncoderNPU not initialized. Call initialize() first.")

        start_time = time.perf_counter()

        B, T, D = hidden_states.shape
        H = self.config.num_heads
        d_k = D // H
        FFN = self.config.ffn_dim

        # Store original for residual
        residual_1 = hidden_states.copy()

        # =========== SELF-ATTENTION BLOCK ===========

        # 1. Pre-attention LayerNorm
        normed = self.execute_kernel('layernorm', hidden_states, (B, T, D))

        # 2. Q, K, V Projections
        # TODO: Implement MatMul kernel calls when available
        # For now, use simulation
        Q = self.execute_kernel('matmul', normed, (B, T, D),
                               weights=self.weights.get(f'layer_{layer_idx}.q_proj.weight'))
        K = self.execute_kernel('matmul', normed, (B, T, D),
                               weights=self.weights.get(f'layer_{layer_idx}.k_proj.weight'))
        V = self.execute_kernel('matmul', normed, (B, T, D),
                               weights=self.weights.get(f'layer_{layer_idx}.v_proj.weight'))

        # 3. Reshape for multi-head attention [B, T, D] -> [B, H, T, d_k]
        Q = Q.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)

        # 4. Attention scores: Q @ K^T / sqrt(d_k)
        # Convert to FP32 for this computation
        Q_fp32 = self._bf16_to_fp32(Q)
        K_fp32 = self._bf16_to_fp32(K)
        scores = np.matmul(Q_fp32, K_fp32.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        scores_bf16 = self._fp32_to_bf16(scores)

        # 5. Softmax (parallel across heads)
        attn_weights = self.execute_kernel('softmax_parallel' if 'softmax_parallel' in self.kernels else 'softmax',
                                          scores_bf16, (B, H, T, T))

        # 6. Attention @ Values
        attn_fp32 = self._bf16_to_fp32(attn_weights)
        V_fp32 = self._bf16_to_fp32(V)
        context = np.matmul(attn_fp32, V_fp32)
        context_bf16 = self._fp32_to_bf16(context)

        # 7. Reshape and output projection
        context = context_bf16.transpose(0, 2, 1, 3).reshape(B, T, D)
        attn_output = self.execute_kernel('matmul', context, (B, T, D),
                                         weights=self.weights.get(f'layer_{layer_idx}.out_proj.weight'))

        # 8. Residual connection
        hidden_states = self._add_residual(residual_1, attn_output)

        # Store for second residual
        residual_2 = hidden_states.copy()

        # =========== FEED-FORWARD BLOCK ===========

        # 9. Pre-FFN LayerNorm
        normed = self.execute_kernel('layernorm', hidden_states, (B, T, D))

        # 10. FFN Up-Projection
        ffn_hidden = self.execute_kernel('matmul', normed, (B, T, FFN),
                                        weights=self.weights.get(f'layer_{layer_idx}.fc1.weight'))

        # 11. GELU activation
        ffn_hidden = self.execute_kernel('gelu', ffn_hidden, (B, T, FFN))

        # 12. FFN Down-Projection
        ffn_output = self.execute_kernel('matmul', ffn_hidden, (B, T, D),
                                        weights=self.weights.get(f'layer_{layer_idx}.fc2.weight'))

        # 13. Residual connection
        hidden_states = self._add_residual(residual_2, ffn_output)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Encoder layer {layer_idx} completed in {elapsed:.2f} ms")

        return hidden_states

    def _add_residual(self, residual: np.ndarray, output: np.ndarray) -> np.ndarray:
        """Add residual connection in BF16"""
        # Convert to FP32, add, convert back
        residual_fp32 = self._bf16_to_fp32(residual)
        output_fp32 = self._bf16_to_fp32(output)
        result = residual_fp32 + output_fp32
        return self._fp32_to_bf16(result)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Run full encoder stack (all layers).

        Args:
            hidden_states: Input tensor [B, T, D]

        Returns:
            Output tensor [B, T, D]
        """
        start_time = time.perf_counter()

        # Convert to BF16 if needed
        if hidden_states.dtype != np.uint16:
            hidden_states = self._fp32_to_bf16(hidden_states.astype(np.float32))

        # Process all encoder layers
        for i in range(self.config.num_layers):
            hidden_states = self.encoder_layer(hidden_states, layer_idx=i)
            logger.debug(f"Completed layer {i+1}/{self.config.num_layers}")

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Full encoder forward pass completed in {elapsed:.2f} ms")

        return hidden_states

    def benchmark(self, batch_size: int = 8, seq_length: int = 100,
                  num_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark encoder performance.

        Args:
            batch_size: Batch size for benchmark
            seq_length: Sequence length (time steps)
            num_iterations: Number of iterations to average

        Returns:
            Dictionary with timing metrics
        """
        logger.info(f"Benchmarking encoder: B={batch_size}, T={seq_length}, iters={num_iterations}")

        # Create test input
        test_input = np.random.randn(batch_size, seq_length, self.config.hidden_dim).astype(np.float32)
        test_input_bf16 = self._fp32_to_bf16(test_input)

        # Warmup
        _ = self.encoder_layer(test_input_bf16, layer_idx=0)

        # Benchmark single layer
        layer_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.encoder_layer(test_input_bf16, layer_idx=0)
            layer_times.append((time.perf_counter() - start) * 1000)

        avg_layer_time = np.mean(layer_times)
        std_layer_time = np.std(layer_times)

        # Benchmark full encoder
        full_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.forward(test_input)
            full_times.append((time.perf_counter() - start) * 1000)

        avg_full_time = np.mean(full_times)

        # Calculate metrics
        audio_duration_ms = seq_length * 10  # 10ms per time step
        rtf = avg_full_time / audio_duration_ms
        realtime_factor = 1 / rtf if rtf > 0 else float('inf')

        results = {
            'layer_time_ms': avg_layer_time,
            'layer_time_std_ms': std_layer_time,
            'full_encoder_time_ms': avg_full_time,
            'audio_duration_ms': audio_duration_ms,
            'rtf': rtf,
            'realtime_factor': realtime_factor,
            'batch_size': batch_size,
            'seq_length': seq_length,
            'num_layers': self.config.num_layers,
        }

        logger.info(f"Benchmark Results:")
        logger.info(f"  Single layer: {avg_layer_time:.2f} +/- {std_layer_time:.2f} ms")
        logger.info(f"  Full encoder ({self.config.num_layers} layers): {avg_full_time:.2f} ms")
        logger.info(f"  Realtime factor: {realtime_factor:.1f}x")

        return results

    def cleanup(self) -> None:
        """Release all resources"""
        if self.buffer_manager:
            self.buffer_manager.cleanup()

        self.kernels.clear()
        self.device = None
        self.is_initialized = False

        logger.info("WhisperEncoderNPU resources released")

    def get_system_info(self) -> Dict[str, Any]:
        """Get system and model information"""
        info = {
            'model_size': self.model_size,
            'hidden_dim': self.config.hidden_dim,
            'num_heads': self.config.num_heads,
            'num_layers': self.config.num_layers,
            'ffn_dim': self.config.ffn_dim,
            'is_initialized': self.is_initialized,
            'kernels_loaded': list(self.kernels.keys()),
            'buffers_allocated': self.buffer_manager.total_allocated if self.buffer_manager else 0,
            'weights_loaded': len(self.weights),
        }

        # Check kernel availability
        for name, path in self.xclbin_paths.items():
            info[f'kernel_{name}_available'] = path.exists()

        return info


def test_whisper_encoder_npu():
    """Test the WhisperEncoderNPU class"""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("=" * 70)
    print("WhisperEncoderNPU Test")
    print("=" * 70)

    # Create encoder
    encoder = WhisperEncoderNPU(model_size="base")

    # Initialize
    if not encoder.initialize():
        print("Failed to initialize encoder")
        return

    # Print system info
    info = encoder.get_system_info()
    print("\nSystem Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Run benchmark
    print("\nRunning benchmark...")
    results = encoder.benchmark(batch_size=1, seq_length=100, num_iterations=5)

    print("\nBenchmark Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Cleanup
    encoder.cleanup()

    print("\nTest completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_whisper_encoder_npu()
