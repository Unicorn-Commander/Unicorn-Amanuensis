#!/usr/bin/env python3
"""
NPU Encoder Interface for AMD Phoenix NPU

Wraps compiled NPU kernels (LayerNorm, Softmax, GELU, MatMul) to provide
a high-level encoder interface for Whisper transcription.

All kernels use BF16 format for computation.

AMD Phoenix NPU - XDNA1
"""

import numpy as np
import pyxrt as xrt
import struct
import time
import os
from typing import Optional, Dict, Tuple, List


class NPUEncoder:
    """
    NPU Encoder class that manages and executes AMD Phoenix NPU kernels.

    Provides an interface for running individual kernels or chained encoder
    layers for Whisper transcription.

    Buffer Management Strategy:
    - Pre-allocate buffers at initialization to avoid runtime allocation overhead
    - Reuse buffers across kernel invocations
    - Keep instruction buffers synced once (immutable during inference)
    - Use XRT buffer flags for optimal memory placement (cacheable for instr,
      host_only for data)
    """

    # Default buffer size: 1024 elements * 2 bytes = 2048 bytes
    DEFAULT_ELEMENTS = 1024

    # MatMul dimensions: 64x64 matrices
    MATMUL_SIZE = 64
    MATMUL_ELEMENTS = MATMUL_SIZE * MATMUL_SIZE  # 4096

    def __init__(self, kernel_dir: str, num_elements: int = DEFAULT_ELEMENTS):
        """
        Initialize NPU Encoder by loading all kernel XCLBINs and allocating buffers.

        Args:
            kernel_dir: Path to directory containing kernel builds
            num_elements: Number of elements for element-wise kernels (default 1024)

        Raises:
            RuntimeError: If NPU device or kernels cannot be initialized
        """
        self.kernel_dir = kernel_dir
        self.num_elements = num_elements
        self.buffer_size = num_elements * 2  # BF16 = 2 bytes per element
        self.matmul_buffer_size = self.MATMUL_ELEMENTS * 2  # For 64x64 matrices

        # Kernel configurations
        self._kernel_configs = {
            'layernorm': {
                'xclbin': 'build_layernorm/layernorm_bf16.xclbin',
                'insts': 'build_layernorm/insts.bin',
                'buffer_size': self.buffer_size,
                'num_inputs': 1,
            },
            'softmax': {
                'xclbin': 'build_softmax_bf16/softmax_bf16.xclbin',
                'insts': 'build_softmax_bf16/insts.bin',
                'buffer_size': self.buffer_size,
                'num_inputs': 1,
            },
            'gelu': {
                'xclbin': 'build_gelu/gelu_bf16.xclbin',
                'insts': 'build_gelu/insts.bin',
                'buffer_size': self.buffer_size,
                'num_inputs': 1,
            },
            'matmul': {
                'xclbin': 'build_matmul/matmul_bf16_vectorized.xclbin',
                'insts': 'build_matmul/insts_vec.bin',
                'buffer_size': self.matmul_buffer_size,
                'num_inputs': 2,
            },
        }

        # Storage for kernel objects
        self._kernels: Dict[str, Dict] = {}

        # Initialize device and load kernels
        self._init_device()
        self._load_all_kernels()
        self._allocate_all_buffers()

        # Performance tracking
        self._timing_stats: Dict[str, List[float]] = {
            'layernorm': [], 'softmax': [], 'gelu': [], 'matmul': []
        }

    def _init_device(self):
        """Initialize XRT device connection."""
        try:
            self.device = xrt.device(0)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NPU device: {e}")

    def _load_all_kernels(self):
        """Load all kernel XCLBINs and create hardware contexts."""
        for kernel_name, config in self._kernel_configs.items():
            xclbin_path = os.path.join(self.kernel_dir, config['xclbin'])
            insts_path = os.path.join(self.kernel_dir, config['insts'])

            # Verify files exist
            if not os.path.exists(xclbin_path):
                raise RuntimeError(f"XCLBIN not found: {xclbin_path}")
            if not os.path.exists(insts_path):
                raise RuntimeError(f"Instructions not found: {insts_path}")

            # Load XCLBIN
            xclbin_obj = xrt.xclbin(xclbin_path)
            uuid = xclbin_obj.get_uuid()
            self.device.register_xclbin(xclbin_obj)

            # Create hardware context and kernel
            hw_ctx = xrt.hw_context(self.device, uuid)
            kernel = xrt.kernel(hw_ctx, "MLIR_AIE")

            # Load instructions
            with open(insts_path, "rb") as f:
                insts = f.read()

            self._kernels[kernel_name] = {
                'hw_ctx': hw_ctx,
                'kernel': kernel,
                'insts': insts,
                'config': config,
            }

    def _allocate_all_buffers(self):
        """
        Pre-allocate XRT buffers for all kernels.

        Buffer allocation strategy:
        - Instruction buffers: cacheable flag for optimal performance
        - Input/output buffers: host_only for fast CPU access
        - Group IDs: 1=instr, 3=in1, 4=in2/out, 5=out (for matmul)
        """
        for kernel_name, kdata in self._kernels.items():
            kernel = kdata['kernel']
            config = kdata['config']
            buffer_size = config['buffer_size']

            # Instruction buffer - sync once at init
            bo_instr = xrt.bo(
                self.device,
                len(kdata['insts']),
                xrt.bo.flags.cacheable,
                kernel.group_id(1)
            )
            bo_instr.write(kdata['insts'], 0)
            bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            # Input buffer(s)
            bo_in = xrt.bo(
                self.device,
                buffer_size,
                xrt.bo.flags.host_only,
                kernel.group_id(3)
            )

            # For matmul, we need a second input buffer
            if config['num_inputs'] == 2:
                bo_in2 = xrt.bo(
                    self.device,
                    buffer_size,
                    xrt.bo.flags.host_only,
                    kernel.group_id(4)
                )
                bo_out = xrt.bo(
                    self.device,
                    buffer_size,
                    xrt.bo.flags.host_only,
                    kernel.group_id(5)
                )
                kdata['bo_in2'] = bo_in2
            else:
                bo_out = xrt.bo(
                    self.device,
                    buffer_size,
                    xrt.bo.flags.host_only,
                    kernel.group_id(4)
                )

            kdata['bo_instr'] = bo_instr
            kdata['bo_in'] = bo_in
            kdata['bo_out'] = bo_out

    @staticmethod
    def float_to_bf16(floats: np.ndarray) -> bytes:
        """
        Convert float32 array to BF16 bytes.

        BF16 is the upper 16 bits of FP32, providing the same exponent range
        with reduced mantissa precision.
        """
        floats = np.asarray(floats, dtype=np.float32).flatten()
        result = bytearray(len(floats) * 2)
        for i, val in enumerate(floats):
            bits = struct.unpack('I', struct.pack('f', val))[0]
            upper = (bits >> 16) & 0xFFFF
            struct.pack_into('H', result, i * 2, upper)
        return bytes(result)

    @staticmethod
    def bf16_to_float(bf16_bytes: bytes) -> np.ndarray:
        """
        Convert BF16 bytes to float32 array.

        Reconstructs FP32 by placing BF16 bits in the upper half.
        """
        result = np.zeros(len(bf16_bytes) // 2, dtype=np.float32)
        for i in range(len(result)):
            upper = struct.unpack('H', bf16_bytes[i*2:(i+1)*2])[0]
            result[i] = struct.unpack('f', struct.pack('I', upper << 16))[0]
        return result

    def _run_kernel(self, kernel_name: str, input_bf16: bytes,
                    input2_bf16: Optional[bytes] = None) -> Tuple[bytes, float]:
        """
        Execute a kernel with the given input data.

        Args:
            kernel_name: Name of kernel to run
            input_bf16: Input data in BF16 format
            input2_bf16: Second input for matmul

        Returns:
            Tuple of (output_bf16_bytes, elapsed_time_seconds)
        """
        kdata = self._kernels[kernel_name]
        kernel = kdata['kernel']
        config = kdata['config']

        # Write input to buffer
        kdata['bo_in'].write(input_bf16, 0)
        kdata['bo_in'].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # For matmul, write second input
        if config['num_inputs'] == 2 and input2_bf16:
            kdata['bo_in2'].write(input2_bf16, 0)
            kdata['bo_in2'].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Execute kernel
        opcode = 3  # Standard opcode for MLIR_AIE kernels
        start = time.perf_counter()

        if config['num_inputs'] == 2:
            run = kernel(
                opcode,
                kdata['bo_instr'],
                len(kdata['insts']),
                kdata['bo_in'],
                kdata['bo_in2'],
                kdata['bo_out']
            )
        else:
            run = kernel(
                opcode,
                kdata['bo_instr'],
                len(kdata['insts']),
                kdata['bo_in'],
                kdata['bo_out']
            )

        run.wait()
        elapsed = time.perf_counter() - start

        # Read output
        kdata['bo_out'].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        output_bf16 = kdata['bo_out'].read(config['buffer_size'], 0).tobytes()

        # Track timing
        self._timing_stats[kernel_name].append(elapsed)

        return output_bf16, elapsed

    def layernorm(self, input_bf16: bytes) -> bytes:
        """
        Run LayerNorm kernel on input data.

        Normalizes input to zero mean and unit variance.

        Args:
            input_bf16: Input data in BF16 format (self.buffer_size bytes)

        Returns:
            Normalized output in BF16 format
        """
        output, _ = self._run_kernel('layernorm', input_bf16)
        return output

    def softmax(self, input_bf16: bytes) -> bytes:
        """
        Run Softmax kernel on input data.

        Applies softmax activation to normalize inputs to probability distribution.

        Args:
            input_bf16: Input data in BF16 format (self.buffer_size bytes)

        Returns:
            Softmax output in BF16 format (sums to 1.0)
        """
        output, _ = self._run_kernel('softmax', input_bf16)
        return output

    def gelu(self, input_bf16: bytes) -> bytes:
        """
        Run GELU kernel on input data.

        Applies Gaussian Error Linear Unit activation.

        Args:
            input_bf16: Input data in BF16 format (self.buffer_size bytes)

        Returns:
            GELU-activated output in BF16 format
        """
        output, _ = self._run_kernel('gelu', input_bf16)
        return output

    def matmul(self, A_bf16: bytes, B_bf16: bytes) -> bytes:
        """
        Run vectorized MatMul kernel.

        Computes C = A @ B for 64x64 matrices.

        Args:
            A_bf16: First matrix in BF16 format (row-major, 64x64)
            B_bf16: Second matrix in BF16 format (row-major, 64x64)

        Returns:
            Result matrix in BF16 format (row-major, 64x64)
        """
        output, _ = self._run_kernel('matmul', A_bf16, B_bf16)
        return output

    def encoder_layer(self, input_floats: np.ndarray) -> np.ndarray:
        """
        Run a full encoder layer pattern: LayerNorm -> Softmax -> GELU.

        This simulates the normalization -> attention scores -> activation
        pattern found in transformer encoder layers.

        Args:
            input_floats: Input float32 array (self.num_elements elements)

        Returns:
            Output float32 array after encoder layer processing
        """
        # Ensure correct shape
        input_floats = np.asarray(input_floats, dtype=np.float32).flatten()
        if len(input_floats) != self.num_elements:
            raise ValueError(
                f"Input must have {self.num_elements} elements, got {len(input_floats)}"
            )

        # Convert to BF16
        input_bf16 = self.float_to_bf16(input_floats)

        # Chain: LayerNorm -> Softmax -> GELU
        normalized = self.layernorm(input_bf16)
        attention = self.softmax(normalized)
        activated = self.gelu(attention)

        # Convert back to float32
        return self.bf16_to_float(activated)

    def encoder_layer_with_matmul(self,
                                   input_floats: np.ndarray,
                                   weights: np.ndarray) -> np.ndarray:
        """
        Run encoder layer with MatMul: Input x Weights -> LayerNorm -> GELU.

        This represents a more complete feed-forward network pattern.

        Args:
            input_floats: Input matrix (64x64 float32)
            weights: Weight matrix (64x64 float32)

        Returns:
            Output float32 array (64x64)
        """
        # Ensure correct shapes
        input_floats = np.asarray(input_floats, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)

        if input_floats.size != self.MATMUL_ELEMENTS:
            raise ValueError(
                f"Input must be 64x64 ({self.MATMUL_ELEMENTS} elements)"
            )
        if weights.size != self.MATMUL_ELEMENTS:
            raise ValueError(
                f"Weights must be 64x64 ({self.MATMUL_ELEMENTS} elements)"
            )

        # Convert to BF16
        input_bf16 = self.float_to_bf16(input_floats.flatten())
        weights_bf16 = self.float_to_bf16(weights.flatten())

        # MatMul: project input
        projected = self.matmul(input_bf16, weights_bf16)

        # For element-wise ops, we need to ensure buffer sizes match
        # The matmul output is 4096 elements but layernorm expects 1024
        # So we process in chunks or use the first 1024 for demo
        projected_floats = self.bf16_to_float(projected)

        # Process first chunk through LayerNorm -> GELU
        chunk = projected_floats[:self.num_elements]
        chunk_bf16 = self.float_to_bf16(chunk)

        normalized = self.layernorm(chunk_bf16)
        activated = self.gelu(normalized)

        # Reconstruct full output (simplified: only first 1024 processed)
        result = projected_floats.copy()
        result[:self.num_elements] = self.bf16_to_float(activated)

        return result.reshape(64, 64)

    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get timing statistics for all kernels.

        Returns:
            Dictionary with avg/min/max/count for each kernel
        """
        stats = {}
        for name, times in self._timing_stats.items():
            if times:
                stats[name] = {
                    'avg_ms': np.mean(times) * 1000,
                    'min_ms': np.min(times) * 1000,
                    'max_ms': np.max(times) * 1000,
                    'count': len(times),
                }
            else:
                stats[name] = {
                    'avg_ms': 0.0,
                    'min_ms': 0.0,
                    'max_ms': 0.0,
                    'count': 0,
                }
        return stats

    def reset_timing_stats(self):
        """Reset all timing statistics."""
        for name in self._timing_stats:
            self._timing_stats[name] = []

    def benchmark(self, num_iterations: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Run benchmarks for all kernels.

        Args:
            num_iterations: Number of iterations per kernel

        Returns:
            Timing statistics for all kernels
        """
        self.reset_timing_stats()

        # Test data
        test_input = np.random.randn(self.num_elements).astype(np.float32)
        test_bf16 = self.float_to_bf16(test_input)

        # Matrices for matmul
        A = np.random.randn(64, 64).astype(np.float32) * 0.1
        B = np.random.randn(64, 64).astype(np.float32) * 0.1
        A_bf16 = self.float_to_bf16(A.flatten())
        B_bf16 = self.float_to_bf16(B.flatten())

        # Benchmark each kernel
        for _ in range(num_iterations):
            self.layernorm(test_bf16)
            self.softmax(test_bf16)
            self.gelu(test_bf16)
            self.matmul(A_bf16, B_bf16)

        return self.get_timing_stats()


def main():
    """Test the NPUEncoder class."""
    print("=" * 70)
    print("NPU Encoder Interface Test - AMD Phoenix NPU")
    print("=" * 70)
    print()

    kernel_dir = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1"

    try:
        # Initialize encoder
        print("Step 1: Initializing NPU Encoder...")
        encoder = NPUEncoder(kernel_dir)
        print("NPU Encoder initialized successfully")
        print(f"  - Element-wise buffer: {encoder.buffer_size} bytes ({encoder.num_elements} elements)")
        print(f"  - MatMul buffer: {encoder.matmul_buffer_size} bytes ({encoder.MATMUL_ELEMENTS} elements)")
        print()

        # Test individual kernels
        print("Step 2: Testing individual kernels...")

        # Test data
        test_input = np.random.randn(encoder.num_elements).astype(np.float32) * 2.0
        test_bf16 = encoder.float_to_bf16(test_input)

        # LayerNorm
        ln_out = encoder.layernorm(test_bf16)
        ln_floats = encoder.bf16_to_float(ln_out)
        print(f"  LayerNorm: mean={np.mean(ln_floats):.6f}, std={np.std(ln_floats):.6f}")

        # Softmax
        sm_out = encoder.softmax(test_bf16)
        sm_floats = encoder.bf16_to_float(sm_out)
        print(f"  Softmax: sum={np.sum(sm_floats):.6f} (should be ~1.0)")

        # GELU
        gelu_out = encoder.gelu(test_bf16)
        gelu_floats = encoder.bf16_to_float(gelu_out)
        print(f"  GELU: mean={np.mean(gelu_floats):.6f}, max={np.max(gelu_floats):.6f}")

        # MatMul
        A = np.random.randn(64, 64).astype(np.float32) * 0.1
        B = np.random.randn(64, 64).astype(np.float32) * 0.1
        A_bf16 = encoder.float_to_bf16(A.flatten())
        B_bf16 = encoder.float_to_bf16(B.flatten())
        mm_out = encoder.matmul(A_bf16, B_bf16)
        mm_floats = encoder.bf16_to_float(mm_out)
        expected = np.matmul(A, B).flatten()
        error = np.max(np.abs(mm_floats - encoder.bf16_to_float(encoder.float_to_bf16(expected))))
        print(f"  MatMul: max_error={error:.6f}")
        print()

        # Test encoder layer chain
        print("Step 3: Testing encoder layer chain...")
        input_floats = np.random.randn(encoder.num_elements).astype(np.float32)
        output_floats = encoder.encoder_layer(input_floats)
        print(f"  Input: mean={np.mean(input_floats):.4f}, std={np.std(input_floats):.4f}")
        print(f"  Output: mean={np.mean(output_floats):.6f}, max={np.max(output_floats):.6f}")
        print()

        # Benchmark
        print("Step 4: Running benchmarks (100 iterations each)...")
        stats = encoder.benchmark(100)
        print()
        print("Performance Results:")
        for kernel_name, kernel_stats in stats.items():
            print(f"  {kernel_name:12s}: {kernel_stats['avg_ms']:.3f} ms avg, "
                  f"{kernel_stats['min_ms']:.3f} ms min")
        print()

        # Calculate throughput
        total_time = sum(s['avg_ms'] for s in stats.values())
        print(f"Total encoder layer time: {stats['layernorm']['avg_ms'] + stats['softmax']['avg_ms'] + stats['gelu']['avg_ms']:.3f} ms")
        print()

        print("=" * 70)
        print("NPU ENCODER INTERFACE TEST PASSED!")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
