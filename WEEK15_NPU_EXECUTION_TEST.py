#!/usr/bin/env python3
"""
Week 15: NPU Kernel Execution Test - XDNA2 Strix Halo

Mission: Test ACTUAL NPU kernel execution (not just loading).

This script:
1. Loads xclbin to NPU (using Week 14 breakthrough API)
2. Creates XRT buffers for input/output data
3. Transfers data TO NPU
4. Executes the "MLIR_AIE" kernel
5. Transfers data FROM NPU
6. Validates results
7. Measures execution time

Hardware: ASUS ROG Flow Z13 GZ302EA (AMD Strix Halo)
NPU: AMD XDNA2 (50 TOPS, 32 tiles)
Kernel: matmul_1tile_bf16.xclbin (512x512 BF16 matrix multiply)

Author: Week 15 NPU Execution Team
Date: November 2, 2025
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Ensure XRT is available
try:
    import pyxrt as xrt
    print(f"[Init] XRT Python bindings loaded successfully")
except ImportError:
    print("[ERROR] pyxrt not found. Install with:")
    print("  source /opt/xilinx/xrt/setup.sh")
    print("  pip install /opt/xilinx/xrt/python/pyxrt-*.whl")
    sys.exit(1)


class NPUExecutionTest:
    """Test NPU kernel execution with real data"""

    def __init__(self, xclbin_path: Path):
        """Initialize NPU test with xclbin kernel"""
        self.xclbin_path = Path(xclbin_path)
        self.device = None
        self.xclbin = None
        self.context = None
        self.kernel = None
        self.bo_a = None  # Buffer object A
        self.bo_b = None  # Buffer object B
        self.bo_c = None  # Buffer object C

        # Kernel configuration (from matmul_1tile_bf16.mlir)
        # Runtime sequence expects: memref<262144xbf16>
        # This is 512x512 BF16 matrix (flattened)
        self.matrix_size = 512
        self.buffer_size = 262144  # 512 * 512 elements
        self.buffer_bytes = self.buffer_size * 2  # BF16 = 2 bytes per element

    def setup_npu(self):
        """Setup NPU hardware and load kernel"""
        print("\n" + "="*70)
        print("  NPU Hardware Setup")
        print("="*70)

        # Step 1: Open XRT device
        print("[1/5] Opening XRT device...")
        try:
            self.device = xrt.device(0)
            print("  Device opened: NPU 0")
        except Exception as e:
            print(f"  ERROR: Failed to open device: {e}")
            raise

        # Step 2: Load xclbin file into object
        print(f"[2/5] Loading xclbin: {self.xclbin_path.name}")
        try:
            self.xclbin = xrt.xclbin(str(self.xclbin_path))
            print(f"  xclbin object created")
        except Exception as e:
            print(f"  ERROR: Failed to load xclbin: {e}")
            raise

        # Step 3: Register xclbin with device (CRITICAL for XDNA2!)
        print("[3/5] Registering xclbin with NPU...")
        try:
            self.device.register_xclbin(self.xclbin)
            print("  xclbin registered successfully")
        except Exception as e:
            print(f"  ERROR: Failed to register xclbin: {e}")
            raise

        # Step 4: Create hardware context
        print("[4/5] Creating hardware context...")
        try:
            uuid = self.xclbin.get_uuid()
            self.context = xrt.hw_context(self.device, uuid)
            print(f"  Hardware context created")
            print(f"  UUID: {uuid}")
        except Exception as e:
            print(f"  ERROR: Failed to create context: {e}")
            raise

        # Step 5: Get kernel handle
        print("[5/5] Loading kernel...")
        try:
            # Try MLIR_AIE first (MLIR-AIE default kernel name)
            kernel_names = ["MLIR_AIE", "matmul_bf16", "matmul"]

            for kname in kernel_names:
                try:
                    self.kernel = xrt.kernel(self.context, kname)
                    print(f"  Kernel loaded: {kname}")
                    break
                except:
                    continue

            if not self.kernel:
                available = [k.get_name() for k in self.xclbin.get_kernels()]
                raise RuntimeError(
                    f"Could not load any kernel from {kernel_names}. "
                    f"Available kernels: {available}"
                )
        except Exception as e:
            print(f"  ERROR: Failed to load kernel: {e}")
            raise

        print("\n  NPU Setup Complete!")
        print("="*70)

    def create_buffers(self):
        """Create XRT buffer objects for input/output"""
        print("\n" + "="*70)
        print("  Buffer Creation")
        print("="*70)

        print(f"[Buffer Config]")
        print(f"  Matrix size: {self.matrix_size}x{self.matrix_size}")
        print(f"  Elements per buffer: {self.buffer_size}")
        print(f"  Bytes per buffer: {self.buffer_bytes} ({self.buffer_bytes/1024:.1f} KB)")
        print(f"  Total memory: {self.buffer_bytes * 3 / 1024:.1f} KB")

        try:
            # Create buffer objects (device memory)
            # Note: XDNA2 uses xrt.bo.host_only flag for NPU buffers
            print("\n[1/3] Creating buffer A (input)...")
            self.bo_a = xrt.bo(
                self.device,
                self.buffer_bytes,
                xrt.bo.host_only,
                self.kernel.group_id(0)  # First kernel argument
            )
            print("  Buffer A created")

            print("[2/3] Creating buffer B (input)...")
            self.bo_b = xrt.bo(
                self.device,
                self.buffer_bytes,
                xrt.bo.host_only,
                self.kernel.group_id(1)  # Second kernel argument
            )
            print("  Buffer B created")

            print("[3/3] Creating buffer C (output)...")
            self.bo_c = xrt.bo(
                self.device,
                self.buffer_bytes,
                xrt.bo.host_only,
                self.kernel.group_id(2)  # Third kernel argument
            )
            print("  Buffer C created")

            print("\n  All buffers created successfully!")
            print("="*70)

        except Exception as e:
            print(f"\n  ERROR: Failed to create buffers: {e}")
            raise

    def prepare_test_data(self):
        """Prepare test matrices for execution"""
        print("\n" + "="*70)
        print("  Test Data Preparation")
        print("="*70)

        # Create simple test matrices (small values to avoid overflow)
        # Using positive values only to avoid BF16 signed value bug
        print("[Test Data] Creating test matrices...")

        # Matrix A: Identity-like (diagonal ones, rest zeros)
        A = np.zeros((self.matrix_size, self.matrix_size), dtype=np.float32)
        np.fill_diagonal(A, 1.0)

        # Matrix B: Constant small values
        B = np.full((self.matrix_size, self.matrix_size), 0.5, dtype=np.float32)

        # Expected result: A @ B should be approximately B (since A is identity)
        expected = A @ B

        print(f"  Matrix A: {self.matrix_size}x{self.matrix_size} identity")
        print(f"  Matrix B: {self.matrix_size}x{self.matrix_size} constant (0.5)")
        print(f"  Expected C: Should be approximately B")

        # Convert to BF16 format (truncate to 16-bit bfloat)
        # BF16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
        def to_bf16_bytes(arr):
            """Convert float32 array to BF16 bytes"""
            # View as uint32, shift right 16 bits, cast to uint16
            uint32_view = arr.view(np.uint32)
            bf16_uint16 = (uint32_view >> 16).astype(np.uint16)
            return bf16_uint16.tobytes()

        self.data_a_bf16 = to_bf16_bytes(A.flatten())
        self.data_b_bf16 = to_bf16_bytes(B.flatten())
        self.expected_result = expected

        print(f"\n  BF16 conversion complete")
        print(f"  Buffer A size: {len(self.data_a_bf16)} bytes")
        print(f"  Buffer B size: {len(self.data_b_bf16)} bytes")
        print("="*70)

        return self.data_a_bf16, self.data_b_bf16, self.expected_result

    def transfer_to_npu(self):
        """Transfer input data to NPU buffers"""
        print("\n" + "="*70)
        print("  Data Transfer: Host → NPU")
        print("="*70)

        start_time = time.perf_counter()

        try:
            # Write data to buffer objects
            print("[1/3] Writing buffer A to NPU...")
            self.bo_a.write(self.data_a_bf16, 0)

            print("[2/3] Writing buffer B to NPU...")
            self.bo_b.write(self.data_b_bf16, 0)

            # Sync buffers to device memory
            print("[3/3] Syncing buffers to device memory...")
            self.bo_a.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self.bo_b.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            transfer_time = time.perf_counter() - start_time

            total_bytes = len(self.data_a_bf16) + len(self.data_b_bf16)
            bandwidth = total_bytes / transfer_time / 1e6  # MB/s

            print(f"\n  Transfer complete!")
            print(f"  Time: {transfer_time*1000:.2f}ms")
            print(f"  Bandwidth: {bandwidth:.1f} MB/s")
            print("="*70)

            return transfer_time

        except Exception as e:
            print(f"\n  ERROR: Failed to transfer data: {e}")
            raise

    def execute_kernel(self):
        """Execute kernel on NPU"""
        print("\n" + "="*70)
        print("  NPU Kernel Execution")
        print("="*70)

        print("[Kernel] Preparing execution...")

        try:
            # Execute kernel and measure time
            # Note: For MLIR-AIE kernels, call kernel directly with buffer objects
            print("[EXECUTE] Running kernel on NPU...")
            start_time = time.perf_counter()

            # Call kernel with buffer arguments
            # MLIR-AIE runtime sequence expects 3 memref arguments
            run = self.kernel(self.bo_a, self.bo_b, self.bo_c)
            run.wait()   # Wait for completion

            exec_time = time.perf_counter() - start_time

            # Calculate performance metrics
            # Matrix multiply: 512x512 @ 512x512 = 2 * 512^3 FLOPs
            flops = 2 * (self.matrix_size ** 3)
            gflops = flops / exec_time / 1e9

            print(f"\n  Kernel execution complete!")
            print(f"  Execution time: {exec_time*1000:.2f}ms")
            print(f"  Performance: {gflops:.1f} GFLOPS")
            print("="*70)

            return exec_time, gflops

        except Exception as e:
            print(f"\n  ERROR: Kernel execution failed: {e}")
            raise

    def transfer_from_npu(self):
        """Transfer output data from NPU"""
        print("\n" + "="*70)
        print("  Data Transfer: NPU → Host")
        print("="*70)

        start_time = time.perf_counter()

        try:
            # Sync buffer from device
            print("[1/2] Syncing buffer C from device...")
            self.bo_c.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

            # Read buffer data using map() method
            print("[2/2] Reading buffer C...")
            result_bf16_bytes = bytes(self.bo_c.map())[:self.buffer_bytes]

            transfer_time = time.perf_counter() - start_time

            bandwidth = self.buffer_bytes / transfer_time / 1e6  # MB/s

            print(f"\n  Transfer complete!")
            print(f"  Time: {transfer_time*1000:.2f}ms")
            print(f"  Bandwidth: {bandwidth:.1f} MB/s")
            print(f"  Data size: {len(result_bf16_bytes)} bytes")
            print("="*70)

            return result_bf16_bytes, transfer_time

        except Exception as e:
            print(f"\n  ERROR: Failed to transfer result: {e}")
            raise

    def validate_results(self, result_bf16_bytes):
        """Validate NPU computation results"""
        print("\n" + "="*70)
        print("  Result Validation")
        print("="*70)

        # Convert BF16 bytes back to float32
        def from_bf16_bytes(bf16_bytes):
            """Convert BF16 bytes to float32 array"""
            bf16_uint16 = np.frombuffer(bf16_bytes, dtype=np.uint16)
            # Shift left 16 bits and view as float32
            uint32 = bf16_uint16.astype(np.uint32) << 16
            return uint32.view(np.float32)

        result = from_bf16_bytes(result_bf16_bytes)
        result_matrix = result.reshape(self.matrix_size, self.matrix_size)

        # Compare with expected result
        expected_flat = self.expected_result.flatten()

        # Calculate error metrics
        abs_error = np.abs(result - expected_flat)
        rel_error = abs_error / (np.abs(expected_flat) + 1e-10)

        max_abs_error = np.max(abs_error)
        mean_abs_error = np.mean(abs_error)
        max_rel_error = np.max(rel_error) * 100  # Percentage
        mean_rel_error = np.mean(rel_error) * 100

        print(f"[Validation] Error Metrics:")
        print(f"  Max absolute error: {max_abs_error:.6f}")
        print(f"  Mean absolute error: {mean_abs_error:.6f}")
        print(f"  Max relative error: {max_rel_error:.2f}%")
        print(f"  Mean relative error: {mean_rel_error:.2f}%")

        # Sample results
        print(f"\n[Sample] First 5 elements:")
        print(f"  Expected: {expected_flat[:5]}")
        print(f"  Actual:   {result[:5]}")
        print(f"  Error:    {abs_error[:5]}")

        # Determine success
        # BF16 typically has ~1% error tolerance
        success = mean_rel_error < 5.0  # Allow 5% mean error for BF16

        print(f"\n[Result] Validation: {'PASS' if success else 'FAIL'}")
        if not success:
            print(f"  WARNING: Error exceeds 5% threshold")
            print(f"  This may indicate:")
            print(f"    - BF16 signed value bug (if using negative numbers)")
            print(f"    - Incorrect kernel configuration")
            print(f"    - Data format mismatch")

        print("="*70)

        return success, {
            'max_abs_error': max_abs_error,
            'mean_abs_error': mean_abs_error,
            'max_rel_error': max_rel_error,
            'mean_rel_error': mean_rel_error,
            'result_matrix': result_matrix
        }

    def cleanup(self):
        """Clean up resources"""
        print("\n[Cleanup] Releasing resources...")
        # XRT handles cleanup automatically when objects go out of scope
        self.bo_a = None
        self.bo_b = None
        self.bo_c = None
        self.kernel = None
        self.context = None
        self.xclbin = None
        self.device = None
        print("  Cleanup complete")


def main():
    """Run NPU execution test"""
    print("\n" + "="*70)
    print("  WEEK 15: NPU KERNEL EXECUTION TEST")
    print("  XDNA2 Strix Halo - First Actual Computation")
    print("="*70)
    print(f"  Date: {time.strftime('%B %d, %Y, %H:%M UTC', time.gmtime())}")
    print(f"  Mission: Test ACTUAL NPU kernel execution")
    print("="*70)

    # Find xclbin file
    # __file__ is in npu-services/unicorn-amanuensis/
    # Need to go up 3 levels to CC-1L root
    xclbin_path = Path(__file__).parent.parent.parent / \
                  "kernels/common/build_bf16_1tile/matmul_1tile_bf16.xclbin"

    if not xclbin_path.exists():
        print(f"\n[ERROR] xclbin not found: {xclbin_path}")
        print(f"  Expected location: kernels/common/build_bf16_1tile/")
        return 1

    print(f"\n[Found] xclbin: {xclbin_path}")

    # Run test
    test = NPUExecutionTest(xclbin_path)

    try:
        # Setup phase
        test.setup_npu()
        test.create_buffers()

        # Execution phase
        data_a, data_b, expected = test.prepare_test_data()

        transfer_to_time = test.transfer_to_npu()
        exec_time, gflops = test.execute_kernel()
        result_bytes, transfer_from_time = test.transfer_from_npu()

        # Validation phase
        success, metrics = test.validate_results(result_bytes)

        # Summary
        total_time = transfer_to_time + exec_time + transfer_from_time

        print("\n" + "="*70)
        print("  WEEK 15 TEST SUMMARY")
        print("="*70)
        print(f"\n[Status] Kernel Execution: {'SUCCESS' if success else 'FAILED'}")
        print(f"\n[Performance]")
        print(f"  Total time: {total_time*1000:.2f}ms")
        print(f"  - Transfer TO NPU: {transfer_to_time*1000:.2f}ms")
        print(f"  - Kernel execution: {exec_time*1000:.2f}ms")
        print(f"  - Transfer FROM NPU: {transfer_from_time*1000:.2f}ms")
        print(f"  Computation: {gflops:.1f} GFLOPS")
        print(f"\n[Accuracy]")
        print(f"  Mean error: {metrics['mean_rel_error']:.2f}%")
        print(f"  Max error: {metrics['max_rel_error']:.2f}%")
        print(f"  Validation: {'PASS' if success else 'FAIL'}")

        print(f"\n[Next Steps]")
        if success:
            print(f"  Week 15 COMPLETE - NPU execution validated!")
            print(f"  Next: Test with real Whisper encoder workload")
            print(f"  Next: Performance optimization (buffer reuse, etc.)")
            print(f"  Next: End-to-end transcription with NPU")
        else:
            print(f"  Debug error sources (check BF16 conversion)")
            print(f"  Verify kernel configuration")
            print(f"  Test with smaller matrices")

        print("="*70)

        # Cleanup
        test.cleanup()

        return 0 if success else 1

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

        # Cleanup on error
        try:
            test.cleanup()
        except:
            pass

        return 1


if __name__ == "__main__":
    sys.exit(main())
