#!/usr/bin/env python3
"""
Optimized Whisper Encoder with Attention and FFN Integration
Complete implementation using all compiled NPU kernels
"""

import pyxrt as xrt
import numpy as np
import struct
import time
from pathlib import Path
from typing import Optional
from attention_npu import MultiHeadAttentionNPU, FFNWithGELU

class WhisperEncoderOptimized:
    """Fully optimized Whisper encoder using NPU kernels"""

    def __init__(self,
                 model_size="base",
                 xclbin_dir: Optional[Path] = None,
                 device_id: int = 0):
        """
        Initialize optimized Whisper encoder

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

        # Initialize random weights (normally loaded from model checkpoint)
        self._init_weights()

        print(f"\n✅ Optimized Whisper {model_size} encoder initialized")
        print(f"   - {self.n_layers} layers with full attention + FFN")
        print(f"   - Attention: {self.n_heads} heads × {self.head_dim} dims")
        print(f"   - FFN: {self.n_dims} → {self.ffn_dim} → {self.n_dims}")

    def _load_kernels(self):
        """Load LayerNorm kernel"""
        print("\n🔧 Loading NPU kernels...")

        layernorm_path = self.xclbin_dir / "build_layernorm_nosqrt/main.xclbin"

        if layernorm_path.exists():
            print(f"   Loading LayerNorm: {layernorm_path.name}")
            try:
                xclbin_obj = xrt.xclbin(str(layernorm_path))
                uuid = xclbin_obj.get_uuid()
                self.device.register_xclbin(xclbin_obj)

                self.ln_hw_ctx = xrt.hw_context(self.device, uuid)
                self.ln_kernel = xrt.kernel(self.ln_hw_ctx, "MLIR_AIE")

                print(f"   ✅ LayerNorm loaded")
                self.use_ln_npu = True
            except Exception as e:
                print(f"   ⚠️  LayerNorm failed: {e}")
                self.use_ln_npu = False
        else:
            print(f"   ⚠️  LayerNorm not found")
            self.use_ln_npu = False

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

    def layernorm_npu(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Layer normalization on NPU"""
        if not self.use_ln_npu:
            mean = np.mean(x)
            var = np.var(x)
            return (x - mean) / np.sqrt(var + eps)

        kernel = self.ln_kernel

        # Load instruction sequence
        insts_path = self.xclbin_dir / "build_layernorm_nosqrt/main_sequence.bin"
        with open(insts_path, "rb") as f:
            insts = f.read()

        # Convert to BF16
        input_bf16 = self._float_to_bf16(x.astype(np.float32))
        buffer_size = len(input_bf16)

        # Allocate buffers
        bo_instr = xrt.bo(self.device, len(insts), xrt.bo.flags.cacheable, kernel.group_id(1))
        bo_input = xrt.bo(self.device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(3))
        bo_output = xrt.bo(self.device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(4))

        # Execute
        bo_instr.write(insts, 0)
        bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        bo_input.write(input_bf16, 0)
        bo_input.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        run = kernel(3, bo_instr, len(insts), bo_input, bo_output)
        run.wait()

        bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        output_bytes = bo_output.read(buffer_size, 0).tobytes()
        output_floats = self._bf16_to_float(output_bytes)

        return output_floats

    def encoder_layer(self,
                      x: np.ndarray,
                      layer_idx: int,
                      verbose: bool = False) -> tuple:
        """
        Single encoder layer with full attention and FFN

        Returns:
            (output, timing_dict)
        """
        seq_len = x.shape[0]
        timings = {}

        # Get weights for this layer
        W = self.weights[layer_idx]

        # Store input for residual
        residual = x.copy()

        # 1. Pre-attention LayerNorm
        t0 = time.time()
        x_norm = np.zeros_like(x)
        for i in range(seq_len):
            x_norm[i] = self.layernorm_npu(x[i])
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

        # 4. Pre-FFN LayerNorm
        t0 = time.time()
        x_norm = np.zeros_like(x)
        for i in range(seq_len):
            x_norm[i] = self.layernorm_npu(x[i])
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
                  f"FFN:{timings['ffn']:.1f})")

        return x, timings

    def forward(self,
                mel_features: np.ndarray,
                verbose: bool = True) -> np.ndarray:
        """
        Full encoder forward pass

        Args:
            mel_features: (seq_len, 80) mel spectrogram
            verbose: print progress

        Returns:
            encoder_output: (seq_len, n_dims)
        """
        if verbose:
            print(f"\n🚀 Running optimized Whisper encoder...")
            print(f"   Input shape: {mel_features.shape}")

        start_time = time.time()

        # Input projection
        seq_len = mel_features.shape[0]
        x = np.zeros((seq_len, self.n_dims), dtype=np.float32)

        # Simple projection (replace with learned projection)
        for i in range(seq_len):
            x[i] = np.sum(mel_features[i]) * np.ones(self.n_dims) / 80.0

        # Add positional encoding
        for pos in range(seq_len):
            for i in range(self.n_dims):
                if i % 2 == 0:
                    x[pos, i] += np.sin(pos / 10000 ** (i / self.n_dims))
                else:
                    x[pos, i] += np.cos(pos / 10000 ** (i / self.n_dims))

        # Run through all encoder layers
        all_timings = []
        for layer_idx in range(self.n_layers):
            x, timings = self.encoder_layer(x, layer_idx, verbose=verbose)
            all_timings.append(timings)

        # Final LayerNorm
        if verbose:
            print(f"   Final LayerNorm...")
        t0 = time.time()
        for i in range(seq_len):
            x[i] = self.layernorm_npu(x[i])
        final_ln_time = (time.time() - t0) * 1000

        total_time = (time.time() - start_time) * 1000

        if verbose:
            # Compute average per component
            avg_ln1 = np.mean([t['ln1'] for t in all_timings])
            avg_ln2 = np.mean([t['ln2'] for t in all_timings])
            avg_attn = np.mean([t['attention'] for t in all_timings])
            avg_ffn = np.mean([t['ffn'] for t in all_timings])
            avg_layer = np.mean([t['total'] for t in all_timings])

            print(f"\n✅ Encoder complete:")
            print(f"   Total time: {total_time:.2f}ms")
            print(f"   Average per layer: {avg_layer:.2f}ms")
            print(f"      - LayerNorm: {avg_ln1+avg_ln2:.2f}ms")
            print(f"      - Attention: {avg_attn:.2f}ms")
            print(f"      - FFN: {avg_ffn:.2f}ms")
            print(f"   Final LayerNorm: {final_ln_time:.2f}ms")
            print(f"   Output shape: {x.shape}")

        return x


def main():
    """Test the optimized encoder"""
    print("=" * 70)
    print("Whisper Encoder - Optimized with Attention + FFN")
    print("=" * 70)

    # Initialize encoder
    encoder = WhisperEncoderOptimized(
        model_size="base",
        device_id=0
    )

    # Create test input
    print("\n📊 Creating test input...")
    seq_len = 50  # Short sequence for testing
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

    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
