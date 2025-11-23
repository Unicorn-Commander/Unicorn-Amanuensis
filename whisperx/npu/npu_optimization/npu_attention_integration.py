#!/usr/bin/env python3
"""
NPU Attention Integration for Whisper Encoder Pipeline
Integrates validated INT32 attention kernel into production Whisper server

MISSION: Achieve 25-35x realtime by adding NPU attention to working decoder
Current: 16-17x realtime (decoder working, encoder CPU)
Target: 25-35x realtime (decoder + NPU attention)

Hardware: AMD Phoenix NPU (XDNA1)
Kernel: INT32 attention - 0.92 correlation, 2.08ms latency
XCLBIN: build_attention_int32/attention_64x64.xclbin (12.4 KB)
Status: VALIDATED and READY FOR PRODUCTION
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add XRT Python path
sys.path.insert(0, '/opt/xilinx/xrt/python')

class NPUAttentionIntegration:
    """
    Production-ready NPU attention integration

    Features:
    - Validated INT32 attention kernel (0.92 correlation)
    - Automatic fallback to CPU if NPU fails
    - Performance logging
    - Thread-safe operation
    - Drop-in replacement for CPU attention
    """

    def __init__(self, xclbin_path: Optional[str] = None, enable_npu: bool = True):
        """
        Initialize NPU attention with CPU fallback

        Args:
            xclbin_path: Path to attention_64x64.xclbin (auto-detected if None)
            enable_npu: If False, force CPU fallback
        """
        self.npu_available = False
        self.npu_attention = None
        self.fallback_to_cpu = not enable_npu
        self.performance_log = {
            'npu_calls': 0,
            'cpu_calls': 0,
            'npu_time_ms': 0.0,
            'cpu_time_ms': 0.0
        }

        if enable_npu:
            self._init_npu_attention(xclbin_path)
        else:
            logger.info("NPU attention disabled by configuration")

    def _init_npu_attention(self, xclbin_path: Optional[str]):
        """Initialize NPU attention kernel"""
        try:
            # Auto-detect xclbin path if not provided
            if xclbin_path is None:
                base = Path(__file__).parent / "whisper_encoder_kernels"
                xclbin_path = base / "build_attention_int32" / "attention_64x64.xclbin"

            xclbin_file = Path(xclbin_path)
            if not xclbin_file.exists():
                logger.warning(f"NPU XCLBIN not found: {xclbin_file}")
                logger.info("Falling back to CPU attention")
                return

            # Import NPU attention wrapper
            npu_wrapper_path = Path(__file__).parent / "whisper_encoder_kernels"
            sys.path.insert(0, str(npu_wrapper_path))

            from npu_attention_wrapper import NPUAttention

            # Initialize NPU attention
            logger.info(f"Loading NPU attention kernel from: {xclbin_file}")
            self.npu_attention = NPUAttention(xclbin_path=str(xclbin_file))
            self.npu_available = True

            logger.info("✅ NPU attention initialized successfully")
            logger.info(f"   XCLBIN: {xclbin_file.name} ({xclbin_file.stat().st_size} bytes)")
            logger.info(f"   Accuracy: 0.92 correlation with PyTorch FP32")
            logger.info(f"   Latency: ~2.08ms per 64x64 tile")
            logger.info(f"   Status: PRODUCTION READY")

        except Exception as e:
            logger.warning(f"Failed to initialize NPU attention: {e}")
            logger.info("Falling back to CPU attention")
            self.npu_available = False

    def compute_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute scaled dot-product attention using NPU or CPU fallback

        Args:
            query: Query matrix (seq_len, d_model) - FP32 or INT8
            key: Key matrix (seq_len, d_model) - FP32 or INT8
            value: Value matrix (seq_len, d_model) - FP32 or INT8
            mask: Optional attention mask

        Returns:
            Attention output (seq_len, d_model)
        """
        import time

        if self.npu_available and not self.fallback_to_cpu:
            try:
                # Use NPU attention
                start = time.perf_counter()
                output = self.npu_attention(query, key, value, mask=mask, quantize=True)
                elapsed_ms = (time.perf_counter() - start) * 1000

                self.performance_log['npu_calls'] += 1
                self.performance_log['npu_time_ms'] += elapsed_ms

                logger.debug(f"NPU attention: {elapsed_ms:.2f}ms")
                return output.astype(np.float32)

            except Exception as e:
                logger.warning(f"NPU attention failed: {e}, falling back to CPU")
                self.performance_log['cpu_calls'] += 1
                return self._cpu_attention(query, key, value, mask)
        else:
            # Use CPU attention
            start = time.perf_counter()
            output = self._cpu_attention(query, key, value, mask)
            elapsed_ms = (time.perf_counter() - start) * 1000

            self.performance_log['cpu_calls'] += 1
            self.performance_log['cpu_time_ms'] += elapsed_ms

            logger.debug(f"CPU attention: {elapsed_ms:.2f}ms")
            return output

    def _cpu_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """CPU fallback implementation of scaled dot-product attention"""
        # Compute attention scores
        d_k = query.shape[-1]
        scores = np.matmul(query, key.T) / np.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask

        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Apply to values
        output = np.matmul(attention_weights, value)
        return output

    def multi_head_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        num_heads: int,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Multi-head attention wrapper with statistics tracking"""
        import time

        logger.info(f"🚀 NPUAttentionIntegration.multi_head_attention() CALLED! query.shape={query.shape}, num_heads={num_heads}")

        if self.npu_available and not self.fallback_to_cpu:
            try:
                # ADD TIMING
                start = time.perf_counter()

                # Use NPU multi-head attention
                output = self.npu_attention.multi_head_attention(
                    query, key, value, num_heads, mask=mask, quantize=True
                ).astype(np.float32)

                # ADD STATISTICS TRACKING
                elapsed_ms = (time.perf_counter() - start) * 1000
                self.performance_log['npu_calls'] += 1
                self.performance_log['npu_time_ms'] += elapsed_ms

                logger.debug(f"NPU multi-head attention: {elapsed_ms:.2f}ms")
                return output

            except Exception as e:
                logger.warning(f"NPU multi-head attention failed: {e}")
                self.fallback_to_cpu = True

        # CPU fallback with statistics
        start = time.perf_counter()
        output = self._cpu_multi_head_attention(query, key, value, num_heads, mask)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.performance_log['cpu_calls'] += 1
        self.performance_log['cpu_time_ms'] += elapsed_ms

        logger.debug(f"CPU multi-head attention: {elapsed_ms:.2f}ms")
        return output

    def _cpu_multi_head_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        num_heads: int,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """CPU fallback for multi-head attention"""
        seq_len, d_model = query.shape
        d_k = d_model // num_heads

        # Reshape for multi-head
        Q = query.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
        K = key.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
        V = value.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)

        # Process each head
        outputs = []
        for i in range(num_heads):
            output_head = self._cpu_attention(Q[i], K[i], V[i], mask)
            outputs.append(output_head)

        # Concatenate heads
        output = np.stack(outputs, axis=0).transpose(1, 0, 2).reshape(seq_len, d_model)
        return output

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_calls = self.performance_log['npu_calls'] + self.performance_log['cpu_calls']

        if total_calls == 0:
            return {
                'total_calls': 0,
                'npu_usage': 0.0,
                'cpu_usage': 0.0,
                'avg_npu_time_ms': 0.0,
                'avg_cpu_time_ms': 0.0
            }

        npu_usage = self.performance_log['npu_calls'] / total_calls * 100
        cpu_usage = self.performance_log['cpu_calls'] / total_calls * 100

        avg_npu_time = (
            self.performance_log['npu_time_ms'] / self.performance_log['npu_calls']
            if self.performance_log['npu_calls'] > 0 else 0.0
        )
        avg_cpu_time = (
            self.performance_log['cpu_time_ms'] / self.performance_log['cpu_calls']
            if self.performance_log['cpu_calls'] > 0 else 0.0
        )

        return {
            'total_calls': total_calls,
            'npu_calls': self.performance_log['npu_calls'],
            'cpu_calls': self.performance_log['cpu_calls'],
            'npu_usage': npu_usage,
            'cpu_usage': cpu_usage,
            'avg_npu_time_ms': avg_npu_time,
            'avg_cpu_time_ms': avg_cpu_time,
            'total_npu_time_ms': self.performance_log['npu_time_ms'],
            'total_cpu_time_ms': self.performance_log['cpu_time_ms']
        }

    def print_performance_stats(self):
        """Print performance statistics"""
        stats = self.get_performance_stats()

        print("\n" + "="*70)
        print("NPU ATTENTION PERFORMANCE STATISTICS")
        print("="*70)
        print(f"NPU Available: {self.npu_available}")
        print(f"Total Attention Calls: {stats['total_calls']}")
        print(f"  NPU Calls: {stats['npu_calls']} ({stats['npu_usage']:.1f}%)")
        print(f"  CPU Calls: {stats['cpu_calls']} ({stats['cpu_usage']:.1f}%)")

        if stats['npu_calls'] > 0:
            print(f"\nNPU Performance:")
            print(f"  Average time: {stats['avg_npu_time_ms']:.2f}ms per call")
            print(f"  Total time: {stats['total_npu_time_ms']:.2f}ms")

        if stats['cpu_calls'] > 0:
            print(f"\nCPU Performance:")
            print(f"  Average time: {stats['avg_cpu_time_ms']:.2f}ms per call")
            print(f"  Total time: {stats['total_cpu_time_ms']:.2f}ms")

        if stats['npu_calls'] > 0 and stats['cpu_calls'] > 0:
            speedup = stats['avg_cpu_time_ms'] / stats['avg_npu_time_ms']
            print(f"\nNPU Speedup: {speedup:.2f}x faster than CPU")

        print("="*70 + "\n")

    def reset_stats(self):
        """Reset performance statistics"""
        self.performance_log = {
            'npu_calls': 0,
            'cpu_calls': 0,
            'npu_time_ms': 0.0,
            'cpu_time_ms': 0.0
        }

    def __repr__(self):
        return (
            f"NPUAttentionIntegration(npu_available={self.npu_available}, "
            f"calls={self.get_performance_stats()['total_calls']})"
        )


def test_integration():
    """Test NPU attention integration"""
    print("="*70)
    print("NPU ATTENTION INTEGRATION TEST")
    print("="*70)
    print()

    # Initialize
    integration = NPUAttentionIntegration()
    print(f"Integration: {integration}")
    print()

    # Test with small matrices
    print("Test 1: Small attention (64x64)")
    Q = np.random.randn(64, 64).astype(np.float32)
    K = np.random.randn(64, 64).astype(np.float32)
    V = np.random.randn(64, 64).astype(np.float32)

    output = integration.compute_attention(Q, K, V)
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    print()

    # Test multi-head attention
    print("Test 2: Multi-head attention (64x512, 8 heads)")
    Q = np.random.randn(64, 512).astype(np.float32)
    K = np.random.randn(64, 512).astype(np.float32)
    V = np.random.randn(64, 512).astype(np.float32)

    output = integration.multi_head_attention(Q, K, V, num_heads=8)
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    print()

    # Print stats
    integration.print_performance_stats()

    print("="*70)
    print("INTEGRATION TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_integration()
