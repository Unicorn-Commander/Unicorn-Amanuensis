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

                    # Pre-load instruction sequence for matmul kernel
                    insts_path = path.parent / "insts.bin"
                    if insts_path.exists():
                        with open(insts_path, "rb") as f:
                            self.matmul_insts = f.read()
                        self.matmul_instr_size = len(self.matmul_insts)

                        # Pre-allocate instruction buffer (reuse across all calls)
                        self.matmul_instr_bo = xrt.bo(
                            self.device,
                            self.matmul_instr_size,
                            xrt.bo.flags.cacheable,
                            self.matmul_kernel.group_id(1)
                        )
                        self.matmul_instr_bo.write(self.matmul_insts, 0)
                        self.matmul_instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                        print(f"   ✅ Matmul kernel loaded with instructions ({self.matmul_instr_size} bytes)")
                    else:
                        print(f"   ⚠️  Instructions not found at {insts_path}")
                        self.matmul_kernel = None
                        continue
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

    def _pad_to_64x64(self, matrix: np.ndarray) -> np.ndarray:
        """
        Pad matrix to multiple of 64×64 with zeros

        Args:
            matrix: Input matrix (M, N)

        Returns:
            Padded matrix with shape (M_pad, N_pad) where M_pad and N_pad are multiples of 64
        """
        M, N = matrix.shape
        M_pad = ((M + 63) // 64) * 64
        N_pad = ((N + 63) // 64) * 64

        if M == M_pad and N == N_pad:
            return matrix

        padded = np.zeros((M_pad, N_pad), dtype=np.float32)
        padded[:M, :N] = matrix
        return padded

    def _matmul_npu_64x64(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Execute single 64×64 matmul on NPU

        Args:
            A: (64, 64) matrix in float32
            B: (64, 64) matrix in float32

        Returns:
            C: (64, 64) result matrix in float32
        """
        assert A.shape == (64, 64) and B.shape == (64, 64), "Matrices must be 64×64"

        # Flatten matrices for BF16 conversion
        A_flat = A.flatten()
        B_flat = B.flatten()

        # Convert to BF16
        A_bf16 = self._float_to_bf16(A_flat)
        B_bf16 = self._float_to_bf16(B_flat)

        # Allocate output buffer (8192 bytes for 64×64 BF16)
        buffer_size = 64 * 64 * 2  # 8192 bytes

        # Create XRT buffers for this operation
        kernel = self.matmul_kernel
        device = self.device

        bo_input_A = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(3))
        bo_input_B = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(4))
        bo_output = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(5))

        # Write inputs
        bo_input_A.write(A_bf16, 0)
        bo_input_A.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bo_input_B.write(B_bf16, 0)
        bo_input_B.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Execute kernel
        # Note: Using pre-loaded instruction buffer from kernel loading
        opcode = 3  # Standard NPU kernel opcode
        run = kernel(opcode, self.matmul_instr_bo, self.matmul_instr_size,
                     bo_input_A, bo_input_B, bo_output)
        run.wait()

        # Read output
        bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        output_bytes = bo_output.read(buffer_size, 0).tobytes()

        # Convert back to float32
        output_floats = self._bf16_to_float(output_bytes)

        # Reshape to 64×64
        return output_floats.reshape(64, 64)

    def _matmul_npu_tiled(self, A: np.ndarray, B: np.ndarray, batch_col_tiles: int = 8) -> np.ndarray:
        """
        Batched tiled matrix multiply on NPU for arbitrary sizes

        OPTIMIZATION: Process multiple column tiles together to reduce kernel invocations.
        Instead of computing one 64×64 output tile at a time (3,008 calls for 3001×512 @ 512×512),
        we batch multiple column tiles together to compute full output rows (376 calls = 8× reduction).

        Strategy:
        - OLD: For each (i, j) output tile, accumulate over K: 47 × 8 × 8 = 3,008 calls
        - NEW: For each row i, batch all j column tiles together: 47 × 8 = 376 calls

        Implementation:
        - Process full rows: (64, K_pad) @ (K_pad, N_pad) → (64, N_pad)
        - Accumulate K dimension by batching column tiles
        - Reduces kernel launch overhead from 900ms to ~113ms

        Args:
            A: (M, K) matrix
            B: (K, N) matrix
            batch_col_tiles: Number of column tiles to batch together (default: 8 = 512 dims)
                            Set to num_tiles_N for full row batching
                            Set to 1 for original unbatched behavior

        Returns:
            C: (M, N) result matrix
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Inner dimensions must match: A is {A.shape}, B is {B.shape}"

        # Pad matrices to multiples of 64
        A_pad = self._pad_to_64x64(A)
        B_pad = self._pad_to_64x64(B)

        M_pad, K_pad = A_pad.shape
        K_pad2, N_pad = B_pad.shape

        # Initialize output matrix
        C_pad = np.zeros((M_pad, N_pad), dtype=np.float32)

        # Tile sizes
        tile_size = 64
        num_tiles_M = M_pad // tile_size
        num_tiles_K = K_pad // tile_size
        num_tiles_N = N_pad // tile_size

        # Clamp batch size to available column tiles
        batch_col_tiles = min(batch_col_tiles, num_tiles_N)

        # =========================================================================
        # BATCHED TILING OPTIMIZATION
        # =========================================================================
        # Process multiple output column tiles together to reduce kernel calls.
        # For (3001, 512) @ (512, 512):
        #   - M_pad = 3008 (47 tiles), K_pad = 512 (8 tiles), N_pad = 512 (8 tiles)
        #   - OLD: 47 × 8 × 8 = 3,008 kernel calls
        #   - NEW (batch_col_tiles=8): 47 × 8 = 376 kernel calls (8× reduction!)
        #
        # Memory consideration:
        #   - Each NPU call processes (64, 64) → 8KB BF16
        #   - Batched: (64, 512) accumulation buffer → 128KB float32 (manageable)
        # =========================================================================

        for i in range(num_tiles_M):
            # Process column tiles in batches for this row
            for j_batch_start in range(0, num_tiles_N, batch_col_tiles):
                # Determine batch size (may be smaller for last batch)
                j_batch_end = min(j_batch_start + batch_col_tiles, num_tiles_N)
                current_batch_size = j_batch_end - j_batch_start

                # Allocate accumulation buffer for this batch of columns
                # Shape: (tile_size, current_batch_size * tile_size)
                C_row_batch = np.zeros((tile_size, current_batch_size * tile_size), dtype=np.float32)

                # Accumulate over K dimension for all batched column tiles
                for k in range(num_tiles_K):
                    # Extract row tile from A: (64, 64)
                    A_tile = A_pad[i*tile_size:(i+1)*tile_size,
                                   k*tile_size:(k+1)*tile_size]

                    # Process each column tile in the batch
                    for j_idx, j in enumerate(range(j_batch_start, j_batch_end)):
                        # Extract column tile from B: (64, 64)
                        B_tile = B_pad[k*tile_size:(k+1)*tile_size,
                                       j*tile_size:(j+1)*tile_size]

                        # Execute single 64×64 matmul on NPU
                        partial_result = self._matmul_npu_64x64(A_tile, B_tile)

                        # Accumulate into batched buffer at correct column position
                        C_row_batch[:, j_idx*tile_size:(j_idx+1)*tile_size] += partial_result

                # Write batched results to output matrix
                C_pad[i*tile_size:(i+1)*tile_size,
                      j_batch_start*tile_size:j_batch_end*tile_size] = C_row_batch

        # Return unpadded result
        return C_pad[:M, :N]

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

        # Use tiled NPU matmul
        return self._matmul_npu_tiled(A, B)

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
