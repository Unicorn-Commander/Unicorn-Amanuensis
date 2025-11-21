#!/usr/bin/env python3
"""
Optimized Whisper Encoder v2 - Hardware-Aware Implementation
Optimizations for AMD Phoenix NPU (XDNA1):
  - Buffer reuse (eliminate allocation overhead)
  - Chunked processing (fits in 2MB memory tiles)
  - Batched DMA transfers (reduce overhead)
"""

import pyxrt as xrt
import numpy as np
import struct
import time
from pathlib import Path
from typing import Optional
from attention_npu import MultiHeadAttentionNPU, FFNWithGELU

class WhisperEncoderOptimizedV2:
    """Hardware-optimized Whisper encoder for Phoenix NPU"""

    def __init__(self,
                 model_size="base",
                 xclbin_dir: Optional[Path] = None,
                 device_id: int = 0):
        """
        Initialize optimized Whisper encoder with buffer reuse

        Args:
            model_size: "base" (6 layers, 512 dims)
            xclbin_dir: Directory containing compiled XCLBINs
            device_id: NPU device ID
        """
        self.model_size = model_size

        # Whisper base configuration
        self.config = {
            "base": {"n_layers": 6, "n_dims": 512, "n_heads": 8},
        }[model_size]

        self.n_layers = self.config["n_layers"]
        self.n_dims = self.config["n_dims"]
        self.n_heads = self.config["n_heads"]
        self.head_dim = self.n_dims // self.n_heads
        self.ffn_dim = self.n_dims * 4  # 2048 for base

        # Setup NPU device
        self.device_id = device_id
        self.device = xrt.device(device_id)

        if xclbin_dir is None:
            xclbin_dir = Path(__file__).parent
        self.xclbin_dir = Path(xclbin_dir)

        # Load kernels
        self._load_kernels()

        # Initialize attention and FFN modules
        self.attention = MultiHeadAttentionNPU(
            n_dims=self.n_dims,
            n_heads=self.n_heads,
            device_id=device_id,
            xclbin_dir=xclbin_dir
        )

        self.ffn = FFNWithGELU(
            n_dims=self.n_dims,
            ffn_dim=self.ffn_dim
        )

        # Initialize random weights
        self._init_weights()

        # Hardware constraints (Phoenix NPU)
        self.memory_tile_capacity = 2 * 1024 * 1024  # 2 MB
        self.chunk_size = self._calculate_optimal_chunk_size()

        print(f"\n✅ Optimized Whisper {model_size} encoder initialized (v2)")
        print(f"   - {self.n_layers} layers with full attention + FFN")
        print(f"   - Attention: {self.n_heads} heads × {self.head_dim} dims")
        print(f"   - FFN: {self.n_dims} → {self.ffn_dim} → {self.n_dims}")
        print(f"   - Chunk size: {self.chunk_size} frames (fits in memory tiles)")

    def _calculate_optimal_chunk_size(self):
        """Calculate chunk size that fits in memory tiles"""
        # Per frame: 512 float32 = 2 KB
        # Need space for input + output + temp buffers
        bytes_per_frame = self.n_dims * 2  # BF16 format
        overhead_factor = 3  # Input + output + temp
        available_memory = self.memory_tile_capacity // overhead_factor
        chunk_size = available_memory // bytes_per_frame

        # Round down to multiple of 16 (for potential vectorization)
        chunk_size = (chunk_size // 16) * 16

        return min(chunk_size, 680)  # Cap at 680 frames

    def _load_kernels(self):
        """Load LayerNorm kernel and pre-allocate buffers"""
        print("\n🔧 Loading NPU kernels...")

        # Initialize flag first
        self.use_ln_npu = False

        layernorm_path = self.xclbin_dir / "build_layernorm_nosqrt/main.xclbin"

        if layernorm_path.exists():
            print(f"   Loading LayerNorm: {layernorm_path.name}")
            try:
                xclbin_obj = xrt.xclbin(str(layernorm_path))
                uuid = xclbin_obj.get_uuid()
                self.device.register_xclbin(xclbin_obj)

                self.ln_hw_ctx = xrt.hw_context(self.device, uuid)
                self.ln_kernel = xrt.kernel(self.ln_hw_ctx, "MLIR_AIE")

                # Pre-load instructions (reuse across all calls)
                insts_path = self.xclbin_dir / "build_layernorm_nosqrt/main_sequence.bin"
                with open(insts_path, "rb") as f:
                    self.ln_insts = f.read()

                # Pre-allocate buffers (REUSE for all calls!)
                self._preallocate_buffers()

                print(f"   ✅ LayerNorm loaded with pre-allocated buffers")
                self.use_ln_npu = True
            except Exception as e:
                print(f"   ⚠️  LayerNorm failed: {e}")
                self.use_ln_npu = False
        else:
            print(f"   ⚠️  LayerNorm not found")
            self.use_ln_npu = False

    def _preallocate_buffers(self):
        """Pre-allocate reusable buffers for LayerNorm"""
        # Always preallocate (even if we end up using CPU fallback)
        # This prevents AttributeError if use_ln_npu is False
        if not hasattr(self, 'ln_kernel'):
            return

        kernel = self.ln_kernel

        # Calculate max buffer size for chunked processing
        max_chunk_elements = self.chunk_size * self.n_dims
        buffer_size = max_chunk_elements * 2  # BF16 = 2 bytes per element

        # Allocate buffers once
        self.bo_instr = xrt.bo(
            self.device,
            len(self.ln_insts),
            xrt.bo.flags.cacheable,
            kernel.group_id(1)
        )

        self.bo_input = xrt.bo(
            self.device,
            buffer_size,
            xrt.bo.flags.host_only,
            kernel.group_id(3)
        )

        self.bo_output = xrt.bo(
            self.device,
            buffer_size,
            xrt.bo.flags.host_only,
            kernel.group_id(4)
        )

        # Write instructions once
        self.bo_instr.write(self.ln_insts, 0)
        self.bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        print(f"   ✅ Buffers pre-allocated: {buffer_size/1024:.1f} KB per buffer")

    def _init_weights(self):
        """Initialize random weights for testing"""
        print("\n📦 Initializing model weights...")

        self.weights = []
        for layer_idx in range(self.n_layers):
            layer_weights = {
                'W_q': np.random.randn(self.n_dims, self.n_dims).astype(np.float32) * 0.02,
                'W_k': np.random.randn(self.n_dims, self.n_dims).astype(np.float32) * 0.02,
                'W_v': np.random.randn(self.n_dims, self.n_dims).astype(np.float32) * 0.02,
                'W_o': np.random.randn(self.n_dims, self.n_dims).astype(np.float32) * 0.02,
                'W_ffn1': np.random.randn(self.n_dims, self.ffn_dim).astype(np.float32) * 0.02,
                'W_ffn2': np.random.randn(self.ffn_dim, self.n_dims).astype(np.float32) * 0.02,
            }
            self.weights.append(layer_weights)

        print(f"   ✅ Initialized {self.n_layers} layers of weights")

    def _bf16_to_float(self, bf16_bytes):
        """Convert BF16 bytes to float32"""
        result = np.zeros(len(bf16_bytes) // 2, dtype=np.float32)
        for i in range(len(result)):
            upper = struct.unpack('H', bf16_bytes[i*2:(i+1)*2])[0]
            result[i] = struct.unpack('f', struct.pack('I', upper << 16))[0]
        return result

    def _float_to_bf16(self, floats):
        """Convert float32 to BF16 bytes"""
        result = bytearray(len(floats) * 2)
        for i, val in enumerate(floats):
            bits = struct.unpack('I', struct.pack('f', val))[0]
            upper = (bits >> 16) & 0xFFFF
            struct.pack_into('H', result, i*2, upper)
        return bytes(result)

    def layernorm_npu_chunked(self, x_batch: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Chunked LayerNorm on NPU with buffer reuse

        Args:
            x_batch: (batch_size, n_dims) input tensor

        Returns:
            normalized: (batch_size, n_dims) normalized tensor
        """
        if not self.use_ln_npu:
            # CPU fallback
            mean = np.mean(x_batch, axis=1, keepdims=True)
            var = np.var(x_batch, axis=1, keepdims=True)
            return (x_batch - mean) / np.sqrt(var + eps)

        batch_size = x_batch.shape[0]

        # Flatten to 1D for processing
        x_flat = x_batch.flatten().astype(np.float32)

        # Convert to BF16 once for entire batch
        input_bf16 = self._float_to_bf16(x_flat)
        buffer_size = len(input_bf16)

        # Use pre-allocated buffers (NO allocation overhead!)
        kernel = self.ln_kernel

        # Write input data
        self.bo_input.write(input_bf16, 0)
        self.bo_input.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Execute kernel (instructions already loaded)
        run = kernel(3, self.bo_instr, len(self.ln_insts), self.bo_input, self.bo_output)
        run.wait()

        # Read output
        self.bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        output_bytes = self.bo_output.read(buffer_size, 0).tobytes()

        # Convert back to float32
        output_floats = self._bf16_to_float(output_bytes)

        # Reshape back to (batch_size, n_dims)
        return output_floats.reshape(batch_size, self.n_dims)

    def encoder_layer_chunked(self,
                              x: np.ndarray,
                              layer_idx: int,
                              verbose: bool = False) -> tuple:
        """
        Single encoder layer with chunked processing

        Args:
            x: (seq_len, n_dims) input
            layer_idx: layer index
            verbose: print timing

        Returns:
            (output, timing_dict)
        """
        seq_len = x.shape[0]
        timings = {}

        # Get weights for this layer
        W = self.weights[layer_idx]

        # Store input for residual
        residual = x.copy()

        # 1. Pre-attention LayerNorm (CHUNKED!)
        t0 = time.time()

        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        x_norm = np.zeros_like(x)

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, seq_len)

            # Process chunk on NPU (batched)
            x_norm[start:end] = self.layernorm_npu_chunked(x[start:end])

        timings['ln1'] = (time.time() - t0) * 1000

        # 2. Multi-head self-attention
        t0 = time.time()
        attn_output = self.attention.forward(
            x_norm, W['W_q'], W['W_k'], W['W_v'], W['W_o']
        )
        timings['attention'] = (time.time() - t0) * 1000

        # 3. Residual connection
        x = residual + attn_output

        # Store for second residual
        residual = x.copy()

        # 4. Pre-FFN LayerNorm (CHUNKED!)
        t0 = time.time()

        x_norm = np.zeros_like(x)
        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, seq_len)

            x_norm[start:end] = self.layernorm_npu_chunked(x[start:end])

        timings['ln2'] = (time.time() - t0) * 1000

        # 5. Feed-forward network
        t0 = time.time()
        ffn_output = self.ffn.forward(x_norm, W['W_ffn1'], W['W_ffn2'])
        timings['ffn'] = (time.time() - t0) * 1000

        # 6. Residual connection
        x = residual + ffn_output

        timings['total'] = sum(timings.values())

        if verbose:
            print(f"   Layer {layer_idx+1}: {timings['total']:.2f}ms "
                  f"(LN:{timings['ln1']+timings['ln2']:.1f} "
                  f"Attn:{timings['attention']:.1f} "
                  f"FFN:{timings['ffn']:.1f}) "
                  f"[{num_chunks} chunks]")

        return x, timings

    def forward(self,
                mel_features: np.ndarray,
                verbose: bool = True) -> np.ndarray:
        """
        Full encoder forward pass with chunked processing

        Args:
            mel_features: (seq_len, 80) mel spectrogram
            verbose: print progress

        Returns:
            encoder_output: (seq_len, n_dims)
        """
        if verbose:
            print(f"\n🚀 Running hardware-optimized Whisper encoder (v2)...")
            print(f"   Input shape: {mel_features.shape}")
            print(f"   Chunk size: {self.chunk_size} frames")

        start_time = time.time()

        # Input projection
        seq_len = mel_features.shape[0]
        x = np.zeros((seq_len, self.n_dims), dtype=np.float32)

        # Simple projection
        for i in range(seq_len):
            x[i] = np.sum(mel_features[i]) * np.ones(self.n_dims) / 80.0

        # Add positional encoding
        for pos in range(seq_len):
            for i in range(self.n_dims):
                if i % 2 == 0:
                    x[pos, i] += np.sin(pos / 10000 ** (i / self.n_dims))
                else:
                    x[pos, i] += np.cos(pos / 10000 ** (i / self.n_dims))

        # Run through all encoder layers with chunked processing
        all_timings = []
        for layer_idx in range(self.n_layers):
            x, timings = self.encoder_layer_chunked(x, layer_idx, verbose=verbose)
            all_timings.append(timings)

        # Final LayerNorm (CHUNKED!)
        if verbose:
            print(f"   Final LayerNorm (chunked)...")

        t0 = time.time()
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, seq_len)
            x[start:end] = self.layernorm_npu_chunked(x[start:end])

        final_ln_time = (time.time() - t0) * 1000

        total_time = (time.time() - start_time) * 1000

        if verbose:
            # Compute average per component
            avg_ln1 = np.mean([t['ln1'] for t in all_timings])
            avg_ln2 = np.mean([t['ln2'] for t in all_timings])
            avg_attn = np.mean([t['attention'] for t in all_timings])
            avg_ffn = np.mean([t['ffn'] for t in all_timings])
            avg_layer = np.mean([t['total'] for t in all_timings])

            print(f"\n✅ Encoder complete (v2 - Hardware Optimized):")
            print(f"   Total time: {total_time:.2f}ms")
            print(f"   Average per layer: {avg_layer:.2f}ms")
            print(f"      - LayerNorm: {avg_ln1+avg_ln2:.2f}ms")
            print(f"      - Attention: {avg_attn:.2f}ms")
            print(f"      - FFN: {avg_ffn:.2f}ms")
            print(f"   Final LayerNorm: {final_ln_time:.2f}ms")
            print(f"   Output shape: {x.shape}")

            # Calculate speedup vs v1
            num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
            print(f"   Chunks processed: {num_chunks}")

        return x


def main():
    """Test the hardware-optimized encoder"""
    print("="*70)
    print("Whisper Encoder v2 - Hardware-Optimized for Phoenix NPU")
    print("="*70)

    # Initialize encoder
    encoder = WhisperEncoderOptimizedV2(
        model_size="base",
        device_id=0
    )

    # Create test input
    print("\n📊 Creating test input...")
    seq_len = 100  # Smaller test first
    n_mels = 80
    mel_features = np.random.randn(seq_len, n_mels).astype(np.float32) * 0.1
    print(f"   Mel features shape: {mel_features.shape}")

    # Run encoder
    encoder_output = encoder.forward(mel_features, verbose=True)

    # Validate output
    print("\n📈 Output Statistics:")
    print(f"   Mean: {np.mean(encoder_output):.6f}")
    print(f"   Std: {np.std(encoder_output):.6f}")
    print(f"   Min: {np.min(encoder_output):.6f}")
    print(f"   Max: {np.max(encoder_output):.6f}")

    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70)


if __name__ == "__main__":
    main()
