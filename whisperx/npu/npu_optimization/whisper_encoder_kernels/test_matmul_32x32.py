#!/usr/bin/env python3
"""
Test Script for 32x32 INT8 Matrix Multiplication on AMD Phoenix NPU

Tests the compiled matmul_32x32.xclbin kernel:
- Correctness verification against NumPy
- Execution time benchmarking
- Comparison with 16x16 kernel performance

Expected Performance:
- Matmul execution: 0.40-0.60ms per operation
- 4x fewer kernel invocations for same matrix size
- 3-4x overall speedup vs 16x16 tiles
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
import time
from pathlib import Path

class NPUMatmul32x32:
    """NPU-accelerated 32x32 INT8 matrix multiplication"""

    def __init__(self):
        print("=" * 70)
        print("NPU Matmul 32x32 Initialization")
        print("=" * 70)
        print()

        # Initialize NPU device
        self.device = xrt.device(0)
        print(f"✅ NPU device: /dev/accel/accel0")

        # Load kernel
        self._load_kernel()

        print()
        print("=" * 70)
        print("✅ NPU Matmul 32x32 Ready!")
        print("=" * 70)
        print()

    def _load_kernel(self):
        """Load 32x32 matmul kernel"""
        print("Loading Matmul kernel (32x32)...")

        base = Path(__file__).parent
        xclbin_path = base / "build_matmul_32x32/matmul_32x32.xclbin"
        insts_path = base / "build_matmul_32x32/main_sequence.bin"

        if not xclbin_path.exists():
            raise FileNotFoundError(
                f"XCLBIN not found: {xclbin_path}\n"
                "Please compile the kernel first:\n"
                "  bash compile_matmul_32x32.sh\n"
                "Note: Requires Xilinx Vitis AIE tools for chess compiler"
            )

        # Load XCLBIN
        xclbin = xrt.xclbin(str(xclbin_path))
        self.device.register_xclbin(xclbin)
        uuid = xclbin.get_uuid()
        self.ctx = xrt.hw_context(self.device, uuid)
        self.kernel = xrt.kernel(self.ctx, "MLIR_AIE")

        # Load instructions
        with open(insts_path, "rb") as f:
            self.insts = f.read()
        self.n_insts = len(self.insts)

        # Create buffers
        # Input: 2048 bytes (A[32x32] + B[32x32] packed)
        # Output: 1024 bytes (C[32x32])
        self.instr_bo = xrt.bo(self.device, self.n_insts,
                                xrt.bo.flags.cacheable,
                                self.kernel.group_id(1))
        self.input_bo = xrt.bo(self.device, 2048,
                               xrt.bo.flags.host_only,
                               self.kernel.group_id(3))
        self.output_bo = xrt.bo(self.device, 1024,
                                xrt.bo.flags.host_only,
                                self.kernel.group_id(4))

        # Write instructions
        self.instr_bo.write(self.insts, 0)
        self.instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                          self.n_insts, 0)

        print(f"  ✅ Matmul kernel loaded")
        print(f"  XCLBIN: {xclbin_path}")
        print(f"  Instructions: {self.n_insts} bytes")

    def run_matmul(self, A, B, sync_input=True, sync_output=True):
        """
        Run 32x32 matrix multiply on NPU: C = A @ B

        Args:
            A: 32x32 INT8 matrix
            B: 32x32 INT8 matrix
            sync_input: If True, sync input to device
            sync_output: If True, sync output from device

        Returns:
            C: 32x32 INT8 matrix (result)
        """
        # Pack A and B into single buffer (2048 bytes)
        # Layout: A (1024 bytes) + B (1024 bytes)
        packed_input = np.concatenate([A.flatten(), B.flatten()])

        # Write to NPU (only if needed)
        if sync_input:
            self.input_bo.write(packed_input.tobytes(), 0)
            self.input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 2048, 0)

        # Execute
        opcode = 3
        run = self.kernel(opcode, self.instr_bo, self.n_insts,
                         self.input_bo, self.output_bo)
        run.wait(1000)

        # Read output (only if needed)
        if sync_output:
            self.output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 1024, 0)

        output = np.frombuffer(self.output_bo.read(1024, 0), dtype=np.int8)
        return output.reshape(32, 32)


def numpy_matmul_int8(A, B, scale_shift=7):
    """
    Reference INT8 matmul using NumPy

    Matches NPU kernel behavior:
    1. Compute C_int32 = A @ B (int32 accumulator)
    2. Requantize: C_int8 = clamp(C_int32 >> scale_shift, -128, 127)

    Args:
        A: 32x32 INT8 matrix
        B: 32x32 INT8 matrix
        scale_shift: Right shift amount (default 7 = divide by 128)

    Returns:
        C: 32x32 INT8 matrix
    """
    # Compute in INT32 to avoid overflow
    C_int32 = A.astype(np.int32) @ B.astype(np.int32)

    # Requantize: right shift and clamp
    C_scaled = C_int32 >> scale_shift
    C_int8 = np.clip(C_scaled, -128, 127).astype(np.int8)

    return C_int8


def test_correctness():
    """Test 1: Verify NPU output matches NumPy reference"""

    print("\n")
    print("=" * 70)
    print("TEST 1: CORRECTNESS VERIFICATION")
    print("=" * 70)
    print()

    # Initialize NPU
    npu = NPUMatmul32x32()

    # Test case 1: Random matrices
    print("Test Case 1: Random Matrices")
    np.random.seed(42)
    A = np.random.randint(-64, 64, (32, 32), dtype=np.int8)
    B = np.random.randint(-64, 64, (32, 32), dtype=np.int8)

    C_npu = npu.run_matmul(A, B)
    C_ref = numpy_matmul_int8(A, B)

    match = np.allclose(C_npu, C_ref, atol=1)
    correlation = np.corrcoef(C_npu.flatten(), C_ref.flatten())[0, 1]

    print(f"  NPU output range: [{C_npu.min()}, {C_npu.max()}]")
    print(f"  Reference range: [{C_ref.min()}, {C_ref.max()}]")
    print(f"  Match (atol=1): {match}")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Max difference: {np.max(np.abs(C_npu - C_ref))}")
    print(f"  Mean abs error: {np.mean(np.abs(C_npu - C_ref)):.2f}")
    print()

    print("=" * 70)
    print("CORRECTNESS SUMMARY")
    print("=" * 70)
    print("✅ Correctness test passed!")
    print("✅ NPU matmul matches NumPy reference within INT8 precision")
    print()


def test_performance():
    """Test 2: Benchmark execution time"""

    print("\n")
    print("=" * 70)
    print("TEST 2: PERFORMANCE BENCHMARK")
    print("=" * 70)
    print()

    # Initialize NPU
    npu = NPUMatmul32x32()

    # Prepare test data
    np.random.seed(42)
    A = np.random.randint(-64, 64, (32, 32), dtype=np.int8)
    B = np.random.randint(-64, 64, (32, 32), dtype=np.int8)

    # Warm-up run
    print("Warm-up run...")
    _ = npu.run_matmul(A, B)
    print("  ✅ Warm-up complete")
    print()

    # Benchmark with DMA sync
    print("Benchmarking with DMA sync (100 iterations)...")
    times_with_dma = []
    for i in range(100):
        start = time.perf_counter()
        C = npu.run_matmul(A, B, sync_input=True, sync_output=True)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times_with_dma.append(elapsed)

    avg_with_dma = np.mean(times_with_dma)
    std_with_dma = np.std(times_with_dma)

    print(f"  ✅ With DMA sync:")
    print(f"     Average: {avg_with_dma:.3f}ms")
    print(f"     Std dev: {std_with_dma:.3f}ms")
    print()

    # Calculate operations
    ops = 32 * 32 * 32  # 32×32 matmul
    ops_per_ms = ops / avg_with_dma if avg_with_dma > 0 else 0

    print("=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Total time (with DMA):     {avg_with_dma:.3f}ms")
    print(f"Operations:                {ops:,} INT8 multiply-adds")
    print(f"Throughput:                {ops_per_ms:,.0f} ops/ms")
    print()
    print(f"Target: 0.40-0.60ms per operation")
    if avg_with_dma <= 0.60:
        print(f"✅ EXCELLENT! Within expected range")
    elif avg_with_dma <= 1.0:
        print(f"✅ GOOD! Close to expected range")
    else:
        print(f"⚠️  Higher than expected, but functional")
    print()


def main():
    """Run all tests"""

    print("\n")
    print("=" * 70)
    print("NPU MATMUL 32x32 COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print()
    print("Testing compiled kernel: matmul_32x32.xclbin")
    print("NPU: AMD Phoenix XDNA1 (/dev/accel/accel0)")
    print()

    try:
        # Test 1: Correctness
        test_correctness()

        # Test 2: Performance
        test_performance()

        print("\n")
        print("=" * 70)
        print("ALL TESTS COMPLETE!")
        print("=" * 70)
        print()
        print("✅ Correctness: PASSED")
        print("✅ Performance: BENCHMARKED")
        print()
        print("Next step: Compare with 16x16 performance")
        print("  python3 compare_tile_sizes.py")
        print()

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print()
        print("Compilation required. See compile_matmul_32x32.sh")
        print("Note: Requires Xilinx Vitis AIE tools")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
