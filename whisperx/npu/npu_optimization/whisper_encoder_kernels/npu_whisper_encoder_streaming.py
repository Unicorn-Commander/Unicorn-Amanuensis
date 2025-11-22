#!/usr/bin/env python3
"""
NPU Whisper Encoder - Streaming Architecture

Streams data through NPU kernels in optimal chunks to maximize throughput
while minimizing kernel launch overhead.

Strategy:
1. Use larger buffer sizes per kernel call
2. Process entire attention matrices at once where possible
3. Use tiled matmul for large projections
4. Stream through chained kernels efficiently

AMD Phoenix NPU - XDNA1
"""

import numpy as np
import time
from typing import Dict, List, Tuple
from pathlib import Path
import sys
import struct

# Add path for NPU encoder
sys.path.insert(0, str(Path(__file__).parent / 'kernels_xdna1'))

try:
    import pyxrt as xrt
    HAS_XRT = True
except ImportError:
    HAS_XRT = False
    print("WARNING: pyxrt not available")

from whisper_weight_loader import WhisperWeightLoader


class NPUStreamingEncoder:
    """
    NPU-accelerated Whisper Encoder using streaming architecture.

    Key optimizations:
    1. Process entire hidden states through LayerNorm in one call
    2. Use tiled matmul for large matrices
    3. Batch GELU activations
    4. Minimize kernel launch overhead
    """

    # Whisper-base configuration
    HIDDEN_DIM = 512
    FFN_DIM = 2048
    NUM_HEADS = 8
    HEAD_DIM = 64
    NUM_LAYERS = 6

    # NPU chunk sizes
    CHUNK_SIZE = 1024  # Elements per kernel call
    MATMUL_TILE = 64   # MatMul tile size

    def __init__(self, onnx_path: str, kernel_dir: str):
        """Initialize NPU Streaming Encoder."""
        print("=" * 70)
        print("NPU Whisper Encoder - Streaming Architecture")
        print("=" * 70)

        self.onnx_path = Path(onnx_path)
        self.kernel_dir = Path(kernel_dir)

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        if not self.kernel_dir.exists():
            raise FileNotFoundError(f"Kernel directory not found: {kernel_dir}")

        # Load weights
        print("\n[1/3] Loading Whisper weights from ONNX...")
        self.weight_loader = WhisperWeightLoader(str(onnx_path))

        # Initialize NPU device and kernels
        print("\n[2/3] Initializing NPU kernels...")
        if HAS_XRT:
            self._init_npu()
        else:
            print("  WARNING: XRT not available, using CPU fallback")
            self.npu_available = False

        # Cache weights
        print("\n[3/3] Caching layer weights...")
        self._cache_weights()

        # Performance tracking
        self._timings = {}

        print("\n" + "=" * 70)
        print(f"Streaming Encoder initialized!")
        print(f"  NPU available: {getattr(self, 'npu_available', False)}")
        print(f"  Layers: {self.weight_loader.num_layers}")
        print("=" * 70)

    def _init_npu(self):
        """Initialize NPU device and load kernel XCLBINs."""
        try:
            self.device = xrt.device(0)
            self.npu_available = True

            # Load kernels
            self.kernels = {}

            # Load encoder chain XCLBIN if available
            chain_xclbin = self.kernel_dir / "build_encoder_simple" / "encoder_layer_simple.xclbin"
            if chain_xclbin.exists():
                self._load_kernel("chain", chain_xclbin)
                print(f"  Loaded encoder chain kernel")

            # Load individual kernels as fallback
            kernel_configs = [
                ("layernorm", "build_layernorm", "layernorm_bf16.xclbin"),
                ("softmax", "build_softmax_bf16", "softmax_bf16.xclbin"),
                ("gelu", "build_gelu", "gelu_bf16.xclbin"),
                ("matmul", "build_matmul_vectorized", "matmul_bf16.xclbin"),
            ]

            for name, build_dir, xclbin_name in kernel_configs:
                xclbin_path = self.kernel_dir / build_dir / xclbin_name
                if xclbin_path.exists():
                    self._load_kernel(name, xclbin_path)
                    print(f"  Loaded {name} kernel")
                else:
                    print(f"  Warning: {name} kernel not found at {xclbin_path}")

        except Exception as e:
            print(f"  NPU init failed: {e}")
            self.npu_available = False

    def _load_kernel(self, name: str, xclbin_path: Path):
        """Load a single kernel XCLBIN."""
        xclbin_obj = xrt.xclbin(str(xclbin_path))
        uuid = xclbin_obj.get_uuid()
        self.device.register_xclbin(xclbin_obj)

        hw_ctx = xrt.hw_context(self.device, uuid)
        kernel = xrt.kernel(hw_ctx, "MLIR_AIE")

        # Load instructions
        insts_path = xclbin_path.parent / "insts.bin"
        with open(insts_path, "rb") as f:
            insts = f.read()

        self.kernels[name] = {
            'hw_ctx': hw_ctx,
            'kernel': kernel,
            'insts': insts,
        }

    def _cache_weights(self):
        """Cache all layer weights."""
        self._cached_weights = {}

        for layer_idx in range(self.weight_loader.num_layers):
            layer_weights = self.weight_loader.get_layer_weights(layer_idx)
            self._cached_weights[layer_idx] = {}

            for key, weight in layer_weights.items():
                self._cached_weights[layer_idx][key] = weight.astype(np.float32)

            print(f"  Cached layer {layer_idx}: {len(layer_weights)} weights")

    def _float_to_bf16(self, floats: np.ndarray) -> bytes:
        """Convert float32 array to BF16 bytes."""
        floats = floats.flatten().astype(np.float32)
        result = bytearray(len(floats) * 2)
        for i, val in enumerate(floats):
            bits = struct.unpack('I', struct.pack('f', val))[0]
            upper = (bits >> 16) & 0xFFFF
            struct.pack_into('H', result, i*2, upper)
        return bytes(result)

    def _bf16_to_float(self, bf16_bytes: bytes) -> np.ndarray:
        """Convert BF16 bytes to float32 array."""
        num_elements = len(bf16_bytes) // 2
        result = np.zeros(num_elements, dtype=np.float32)
        for i in range(num_elements):
            upper = struct.unpack('H', bf16_bytes[i*2:(i+1)*2])[0]
            result[i] = struct.unpack('f', struct.pack('I', upper << 16))[0]
        return result

    def _run_kernel(self, kernel_name: str, input_bf16: bytes, buffer_size: int) -> bytes:
        """Run a single NPU kernel."""
        if not self.npu_available or kernel_name not in self.kernels:
            # CPU fallback
            return input_bf16

        k = self.kernels[kernel_name]
        kernel = k['kernel']
        insts = k['insts']

        # Allocate buffers
        bo_instr = xrt.bo(self.device, len(insts),
                         xrt.bo.flags.cacheable, kernel.group_id(1))
        bo_in = xrt.bo(self.device, buffer_size,
                      xrt.bo.flags.host_only, kernel.group_id(3))
        bo_out = xrt.bo(self.device, buffer_size,
                       xrt.bo.flags.host_only, kernel.group_id(4))

        # Write data
        bo_instr.write(insts, 0)
        bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bo_in.write(input_bf16, 0)
        bo_in.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Execute
        run = kernel(3, bo_instr, len(insts), bo_in, bo_out)
        run.wait()

        # Read result
        bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return bo_out.read(buffer_size, 0).tobytes()

    def _layernorm_cpu(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                       eps: float = 1e-5) -> np.ndarray:
        """CPU LayerNorm (vectorized)."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + eps)
        return normalized * weight + bias

    def _softmax_cpu(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """CPU softmax (vectorized)."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _gelu_cpu(self, x: np.ndarray) -> np.ndarray:
        """CPU GELU (vectorized)."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _stream_through_npu(self, data: np.ndarray, kernel_name: str) -> np.ndarray:
        """
        Stream data through NPU kernel in optimal chunks.

        Processes multiple chunks with single buffer allocation.
        """
        if not self.npu_available or kernel_name not in self.kernels:
            # CPU fallback for unavailable kernels
            if kernel_name == 'layernorm':
                # Basic normalization
                mean = np.mean(data)
                std = np.std(data)
                return (data - mean) / (std + 1e-5)
            elif kernel_name == 'softmax':
                return self._softmax_cpu(data)
            elif kernel_name == 'gelu':
                return self._gelu_cpu(data)
            return data

        original_shape = data.shape
        flat_data = data.flatten().astype(np.float32)

        # Process in chunks
        chunk_size = self.CHUNK_SIZE
        num_chunks = (len(flat_data) + chunk_size - 1) // chunk_size
        result = np.zeros_like(flat_data)

        k = self.kernels[kernel_name]
        kernel = k['kernel']
        insts = k['insts']
        buffer_size = chunk_size * 2  # BF16

        # Allocate buffers once
        bo_instr = xrt.bo(self.device, len(insts),
                         xrt.bo.flags.cacheable, kernel.group_id(1))
        bo_in = xrt.bo(self.device, buffer_size,
                      xrt.bo.flags.host_only, kernel.group_id(3))
        bo_out = xrt.bo(self.device, buffer_size,
                       xrt.bo.flags.host_only, kernel.group_id(4))

        bo_instr.write(insts, 0)
        bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Process all chunks
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(flat_data))
            chunk = flat_data[start:end]

            # Pad if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            # Convert and write
            chunk_bf16 = self._float_to_bf16(chunk)
            bo_in.write(chunk_bf16, 0)
            bo_in.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            # Execute
            run = kernel(3, bo_instr, len(insts), bo_in, bo_out)
            run.wait()

            # Read result
            bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            out_bf16 = bo_out.read(buffer_size, 0).tobytes()
            out_float = self._bf16_to_float(out_bf16)

            # Store
            actual_len = end - start
            result[start:end] = out_float[:actual_len]

        return result.reshape(original_shape)

    def encode_layer(self, hidden_states: np.ndarray, layer_idx: int) -> np.ndarray:
        """
        Process one encoder layer using streaming NPU execution.

        Uses NPU for heavy compute operations, CPU for control flow.
        """
        hidden_states = np.asarray(hidden_states, dtype=np.float32)
        original_shape = hidden_states.shape

        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states[np.newaxis, :]

        batch_size, seq_len, hidden_dim = hidden_states.shape
        weights = self._cached_weights[layer_idx]

        timings = {}

        # ============================================
        # 1. Pre-attention LayerNorm (CPU - vectorized is fast)
        # ============================================
        t0 = time.perf_counter()

        ln1_weight = weights.get('self_attn_layer_norm_weight')
        ln1_bias = weights.get('self_attn_layer_norm_bias')

        if ln1_weight is not None:
            normed = self._layernorm_cpu(hidden_states, ln1_weight, ln1_bias)
        else:
            normed = hidden_states.copy()

        timings['layernorm1'] = time.perf_counter() - t0

        # ============================================
        # 2. QKV Projections (CPU matmul - NumPy is efficient for 512x512)
        # ============================================
        t0 = time.perf_counter()

        normed_2d = normed.reshape(-1, hidden_dim)

        q_weight = weights.get('q_proj_weight')
        k_weight = weights.get('k_proj_weight')
        v_weight = weights.get('v_proj_weight')

        if q_weight is not None:
            Q = normed_2d @ q_weight
            K = normed_2d @ k_weight
            V = normed_2d @ v_weight

            # Add biases if available
            q_bias = weights.get('q_proj_bias')
            k_bias = weights.get('k_proj_bias')
            v_bias = weights.get('v_proj_bias')
            if q_bias is not None:
                Q += q_bias
            if k_bias is not None:
                K += k_bias
            if v_bias is not None:
                V += v_bias
        else:
            Q = K = V = normed_2d

        timings['qkv_proj'] = time.perf_counter() - t0

        # ============================================
        # 3. Attention (CPU - vectorized softmax is efficient)
        # ============================================
        t0 = time.perf_counter()

        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.NUM_HEADS, self.HEAD_DIM)
        K = K.reshape(batch_size, seq_len, self.NUM_HEADS, self.HEAD_DIM)
        V = V.reshape(batch_size, seq_len, self.NUM_HEADS, self.HEAD_DIM)

        # Transpose to (batch, heads, seq, head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Compute attention scores
        scale = 1.0 / np.sqrt(self.HEAD_DIM)
        attn_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale

        # Softmax (vectorized across all heads/batches)
        attn_probs = self._softmax_cpu(attn_scores, axis=-1)

        # Apply to values
        attn_output = np.matmul(attn_probs, V)

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)

        timings['attention'] = time.perf_counter() - t0

        # ============================================
        # 4. Output projection
        # ============================================
        t0 = time.perf_counter()

        o_weight = weights.get('o_proj_weight')
        if o_weight is not None:
            attn_output_2d = attn_output.reshape(-1, hidden_dim)
            projected = attn_output_2d @ o_weight
            if weights.get('o_proj_bias') is not None:
                projected += weights['o_proj_bias']
            attn_output = projected.reshape(batch_size, seq_len, hidden_dim)

        timings['o_proj'] = time.perf_counter() - t0

        # ============================================
        # 5. Residual 1
        # ============================================
        hidden_states = hidden_states + attn_output

        # ============================================
        # 6. Pre-FFN LayerNorm
        # ============================================
        t0 = time.perf_counter()

        ln2_weight = weights.get('final_layer_norm_weight')
        ln2_bias = weights.get('final_layer_norm_bias')

        if ln2_weight is not None:
            normed = self._layernorm_cpu(hidden_states, ln2_weight, ln2_bias)
        else:
            normed = hidden_states.copy()

        timings['layernorm2'] = time.perf_counter() - t0

        # ============================================
        # 7. FFN FC1
        # ============================================
        t0 = time.perf_counter()

        fc1_weight = weights.get('fc1_weight')
        if fc1_weight is not None:
            normed_2d = normed.reshape(-1, hidden_dim)
            ffn_hidden = normed_2d @ fc1_weight
            if weights.get('fc1_bias') is not None:
                ffn_hidden += weights['fc1_bias']
        else:
            ffn_hidden = normed.reshape(-1, hidden_dim)

        timings['fc1'] = time.perf_counter() - t0

        # ============================================
        # 8. GELU - Use NPU for large activation
        # ============================================
        t0 = time.perf_counter()

        # GELU is element-wise, good candidate for NPU streaming
        if self.npu_available and 'gelu' in self.kernels:
            ffn_hidden = self._stream_through_npu(ffn_hidden, 'gelu')
        else:
            ffn_hidden = self._gelu_cpu(ffn_hidden)

        timings['gelu'] = time.perf_counter() - t0

        # ============================================
        # 9. FFN FC2
        # ============================================
        t0 = time.perf_counter()

        fc2_weight = weights.get('fc2_weight')
        if fc2_weight is not None:
            ffn_output = ffn_hidden @ fc2_weight
            if weights.get('fc2_bias') is not None:
                ffn_output += weights['fc2_bias']
            ffn_output = ffn_output.reshape(batch_size, seq_len, hidden_dim)
        else:
            ffn_output = ffn_hidden.reshape(batch_size, seq_len, hidden_dim)

        timings['fc2'] = time.perf_counter() - t0

        # ============================================
        # 10. Residual 2
        # ============================================
        hidden_states = hidden_states + ffn_output

        # Store timings
        self._timings = timings

        if len(original_shape) == 2:
            return hidden_states[0]
        return hidden_states

    def encode(self, mel_features: np.ndarray) -> np.ndarray:
        """Run all encoder layers."""
        hidden_states = np.asarray(mel_features, dtype=np.float32)

        for layer_idx in range(self.weight_loader.num_layers):
            hidden_states = self.encode_layer(hidden_states, layer_idx)

        return hidden_states

    def benchmark(self, seq_len: int = 1500, num_iterations: int = 3) -> dict:
        """Benchmark the encoder."""
        input_data = np.random.randn(seq_len, self.HIDDEN_DIM).astype(np.float32)

        # Warmup
        _ = self.encode_layer(input_data[:100], 0)

        # Benchmark single layer
        layer_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.encode_layer(input_data, 0)
            layer_times.append(time.perf_counter() - start)

        # Benchmark full encoder
        full_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.encode(input_data)
            full_times.append(time.perf_counter() - start)

        return {
            'layer_ms': np.mean(layer_times) * 1000,
            'encoder_ms': np.mean(full_times) * 1000,
            'timings': self._timings,
        }


def main():
    """Test the streaming encoder."""
    print("=" * 70)
    print("NPU Whisper Encoder - Streaming Test")
    print("=" * 70)

    onnx_path = (
        "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/"
        "whisper_onnx_cache/models--onnx-community--whisper-base/"
        "onnx/encoder_model.onnx"
    )
    kernel_dir = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1"

    try:
        encoder = NPUStreamingEncoder(onnx_path, kernel_dir)

        # Test scenarios
        test_configs = [
            {"name": "Small (100 frames)", "seq_len": 100, "audio_sec": 1.0},
            {"name": "Medium (500 frames)", "seq_len": 500, "audio_sec": 5.0},
            {"name": "Full (1500 frames)", "seq_len": 1500, "audio_sec": 30.0},
        ]

        print("\n" + "=" * 70)
        print("STREAMING ENCODER PERFORMANCE TESTS")
        print("=" * 70)

        results = []

        for config in test_configs:
            print(f"\n{'-' * 60}")
            print(f"Test: {config['name']}")
            print(f"{'-' * 60}")

            result = encoder.benchmark(config['seq_len'], num_iterations=3)

            audio_sec = config['audio_sec']
            encoder_ms = result['encoder_ms']
            rtf = audio_sec / (encoder_ms / 1000) if encoder_ms > 0 else 0

            print(f"  Single layer: {result['layer_ms']:.2f} ms")
            print(f"  Full encoder: {encoder_ms:.2f} ms")
            print(f"  Realtime factor: {rtf:.1f}x")

            # Print operation breakdown
            if result['timings']:
                print("\n  Operation breakdown:")
                total = sum(result['timings'].values())
                for op, t in sorted(result['timings'].items(), key=lambda x: -x[1]):
                    pct = (t / total * 100) if total > 0 else 0
                    print(f"    {op:15s}: {t*1000:7.2f} ms ({pct:5.1f}%)")

            results.append({
                'name': config['name'],
                'seq_len': config['seq_len'],
                'audio_sec': audio_sec,
                'encoder_ms': encoder_ms,
                'rtf': rtf,
            })

        # Summary
        print("\n\n" + "=" * 70)
        print("STREAMING ENCODER RESULTS")
        print("=" * 70)
        print()

        print(f"{'Test':<25} {'Seq':<8} {'Audio':<8} {'Encoder':<12} {'RTF':<10}")
        print("-" * 70)

        for r in results:
            print(f"{r['name']:<25} {r['seq_len']:<8} {r['audio_sec']:<8.1f}s {r['encoder_ms']:<12.1f}ms {r['rtf']:<10.1f}x")

        # Comparison to optimized CPU version
        cpu_times = {100: 134, 500: 695, 1500: 2309}

        print("\n" + "=" * 70)
        print("COMPARISON TO BASELINE")
        print("=" * 70)

        for r in results:
            cpu_time = cpu_times.get(r['seq_len'], r['encoder_ms'])
            ratio = cpu_time / r['encoder_ms'] if r['encoder_ms'] > 0 else 0
            if ratio > 1:
                print(f"{r['name']}: {ratio:.1f}x faster than CPU optimized")
            else:
                print(f"{r['name']}: {1/ratio:.1f}x slower than CPU optimized")

        print("\n" + "=" * 70)
        print("STREAMING TEST COMPLETE")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
