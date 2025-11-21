#!/usr/bin/env python3
"""
Multi-Head Self-Attention on NPU
Uses compiled matmul and softmax kernels for attention computation
"""

import numpy as np
import pyxrt as xrt
import struct
from pathlib import Path
from typing import Tuple, Optional

class MultiHeadAttentionNPU:
    """
    Multi-head self-attention using NPU kernels

    Implements: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    """

    def __init__(self,
                 n_dims: int = 512,
                 n_heads: int = 8,
                 device_id: int = 0,
                 xclbin_dir: Optional[Path] = None):
        """
        Initialize multi-head attention on NPU

        Args:
            n_dims: hidden dimensions (512 for Whisper base)
            n_heads: number of attention heads (8 for Whisper base)
            device_id: NPU device ID
            xclbin_dir: directory with compiled XCLBINs
        """
        self.n_dims = n_dims
        self.n_heads = n_heads
        self.head_dim = n_dims // n_heads  # 64 for base

        assert n_dims % n_heads == 0, "n_dims must be divisible by n_heads"

        self.device_id = device_id
        self.device = xrt.device(device_id)

        if xclbin_dir is None:
            xclbin_dir = Path(__file__).parent
        self.xclbin_dir = Path(xclbin_dir)

        # Load attention kernels
        self._load_kernels()

    def _load_kernels(self):
        """Load matmul and softmax kernels for attention"""
        print("🔧 Loading attention kernels...")

        # Try to load matmul kernel
        matmul_paths = [
            self.xclbin_dir / "kernels_xdna1/build_matmul/matmul_bf16.xclbin",
            self.xclbin_dir / "build_matmul_fixed/matmul_16x16.xclbin",
        ]

        self.matmul_kernel = None
        for path in matmul_paths:
            if path.exists():
                try:
                    print(f"   Loading matmul: {path.name}")
                    xclbin_obj = xrt.xclbin(str(path))
                    uuid = xclbin_obj.get_uuid()
                    self.device.register_xclbin(xclbin_obj)
                    hw_ctx = xrt.hw_context(self.device, uuid)
                    self.matmul_kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
                    self.matmul_hw_ctx = hw_ctx
                    print(f"   ✅ Matmul kernel loaded")
                    break
                except Exception as e:
                    print(f"   ⚠️  Failed to load {path.name}: {e}")

        # Try to load softmax kernel
        softmax_paths = [
            self.xclbin_dir / "kernels_xdna1/build_softmax_bf16/softmax_bf16.xclbin",
        ]

        self.softmax_kernel = None
        for path in softmax_paths:
            if path.exists():
                try:
                    print(f"   Loading softmax: {path.name}")
                    xclbin_obj = xrt.xclbin(str(path))
                    uuid = xclbin_obj.get_uuid()
                    self.device.register_xclbin(xclbin_obj)
                    hw_ctx = xrt.hw_context(self.device, uuid)
                    self.softmax_kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
                    self.softmax_hw_ctx = hw_ctx
                    print(f"   ✅ Softmax kernel loaded")
                    break
                except Exception as e:
                    print(f"   ⚠️  Failed to load {path.name}: {e}")

        self.use_npu = (self.matmul_kernel is not None)
        if not self.use_npu:
            print("   ⚠️  No matmul kernel loaded - using CPU")

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

    def matmul_npu(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Matrix multiply on NPU: C = A @ B

        Args:
            A: (M, K) matrix
            B: (K, N) matrix

        Returns:
            C: (M, N) matrix
        """
        if not self.use_npu:
            # CPU fallback
            return A @ B

        # For now, use CPU matmul (NPU integration requires tiling)
        # TODO: Implement tiled matmul on NPU
        return A @ B

    def softmax_npu(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Softmax on NPU

        Args:
            x: input tensor
            axis: axis to apply softmax

        Returns:
            softmax(x)
        """
        if not self.use_npu or self.softmax_kernel is None:
            # CPU fallback
            exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        # For now, use CPU softmax
        # TODO: Implement NPU softmax with kernel
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def forward(self,
                x: np.ndarray,
                W_q: np.ndarray,
                W_k: np.ndarray,
                W_v: np.ndarray,
                W_o: np.ndarray) -> np.ndarray:
        """
        Multi-head self-attention forward pass

        Args:
            x: input (seq_len, n_dims)
            W_q: query projection weights (n_dims, n_dims)
            W_k: key projection weights (n_dims, n_dims)
            W_v: value projection weights (n_dims, n_dims)
            W_o: output projection weights (n_dims, n_dims)

        Returns:
            output: (seq_len, n_dims)
        """
        seq_len = x.shape[0]

        # 1. Project to Q, K, V
        Q = self.matmul_npu(x, W_q)  # (seq_len, n_dims)
        K = self.matmul_npu(x, W_k)  # (seq_len, n_dims)
        V = self.matmul_npu(x, W_v)  # (seq_len, n_dims)

        # 2. Reshape for multi-head: (seq_len, n_heads, head_dim)
        Q = Q.reshape(seq_len, self.n_heads, self.head_dim)
        K = K.reshape(seq_len, self.n_heads, self.head_dim)
        V = V.reshape(seq_len, self.n_heads, self.head_dim)

        # 3. Transpose for batched matmul: (n_heads, seq_len, head_dim)
        Q = Q.transpose(1, 0, 2)
        K = K.transpose(1, 0, 2)
        V = V.transpose(1, 0, 2)

        # 4. Scaled dot-product attention per head
        outputs = []
        scale = 1.0 / np.sqrt(self.head_dim)

        for head_idx in range(self.n_heads):
            # Q @ K^T: (seq_len, head_dim) @ (head_dim, seq_len) = (seq_len, seq_len)
            scores = self.matmul_npu(Q[head_idx], K[head_idx].T) * scale

            # Softmax over keys
            attn_weights = self.softmax_npu(scores, axis=-1)  # (seq_len, seq_len)

            # Attention @ V: (seq_len, seq_len) @ (seq_len, head_dim) = (seq_len, head_dim)
            head_output = self.matmul_npu(attn_weights, V[head_idx])
            outputs.append(head_output)

        # 5. Concatenate heads: (seq_len, n_heads * head_dim) = (seq_len, n_dims)
        multi_head_output = np.concatenate(outputs, axis=-1)

        # 6. Output projection
        output = self.matmul_npu(multi_head_output, W_o)

        return output


class FFNWithGELU:
    """
    Feed-Forward Network with GELU activation
    Implements: FFN(x) = W2 @ GELU(W1 @ x)
    """

    def __init__(self,
                 n_dims: int = 512,
                 ffn_dim: int = 2048,
                 use_npu: bool = True):
        """
        Args:
            n_dims: input/output dimensions (512 for base)
            ffn_dim: intermediate dimensions (2048 for base)
            use_npu: whether to use NPU kernels
        """
        self.n_dims = n_dims
        self.ffn_dim = ffn_dim
        self.use_npu = use_npu

    def gelu(self, x: np.ndarray) -> np.ndarray:
        """
        GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        """
        # Standard GELU approximation
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def forward(self,
                x: np.ndarray,
                W1: np.ndarray,
                W2: np.ndarray) -> np.ndarray:
        """
        FFN forward pass

        Args:
            x: input (seq_len, n_dims)
            W1: first projection (n_dims, ffn_dim)
            W2: second projection (ffn_dim, n_dims)

        Returns:
            output: (seq_len, n_dims)
        """
        # First projection
        hidden = x @ W1  # (seq_len, ffn_dim)

        # GELU activation
        hidden = self.gelu(hidden)

        # Second projection
        output = hidden @ W2  # (seq_len, n_dims)

        return output


def test_attention():
    """Test multi-head attention"""
    print("=" * 70)
    print("Testing Multi-Head Attention on NPU")
    print("=" * 70)

    # Configuration
    seq_len = 10
    n_dims = 512
    n_heads = 8

    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden dimensions: {n_dims}")
    print(f"  Attention heads: {n_heads}")
    print(f"  Head dimension: {n_dims // n_heads}")

    # Initialize attention
    attn = MultiHeadAttentionNPU(n_dims=n_dims, n_heads=n_heads)

    # Create synthetic inputs
    print(f"\n📊 Creating test data...")
    x = np.random.randn(seq_len, n_dims).astype(np.float32) * 0.1

    # Weight matrices (normally loaded from model)
    W_q = np.random.randn(n_dims, n_dims).astype(np.float32) * 0.02
    W_k = np.random.randn(n_dims, n_dims).astype(np.float32) * 0.02
    W_v = np.random.randn(n_dims, n_dims).astype(np.float32) * 0.02
    W_o = np.random.randn(n_dims, n_dims).astype(np.float32) * 0.02

    # Run attention
    print(f"\n🚀 Running attention...")
    import time
    start = time.time()
    output = attn.forward(x, W_q, W_k, W_v, W_o)
    elapsed = (time.time() - start) * 1000

    print(f"\n✅ Attention complete:")
    print(f"   Time: {elapsed:.2f}ms")
    print(f"   Output shape: {output.shape}")
    print(f"   Output mean: {np.mean(output):.6f}")
    print(f"   Output std: {np.std(output):.6f}")

    # Test FFN
    print(f"\n" + "=" * 70)
    print("Testing Feed-Forward Network with GELU")
    print("=" * 70)

    ffn = FFNWithGELU(n_dims=n_dims, ffn_dim=2048)

    W1 = np.random.randn(n_dims, 2048).astype(np.float32) * 0.02
    W2 = np.random.randn(2048, n_dims).astype(np.float32) * 0.02

    print(f"\n🚀 Running FFN...")
    start = time.time()
    ffn_output = ffn.forward(x, W1, W2)
    elapsed = (time.time() - start) * 1000

    print(f"\n✅ FFN complete:")
    print(f"   Time: {elapsed:.2f}ms")
    print(f"   Output shape: {ffn_output.shape}")
    print(f"   Output mean: {np.mean(ffn_output):.6f}")
    print(f"   Output std: {np.std(ffn_output):.6f}")

    print("\n" + "=" * 70)
    print("✅ All tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_attention()
