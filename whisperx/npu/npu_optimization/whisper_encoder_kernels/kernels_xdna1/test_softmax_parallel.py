#!/usr/bin/env python3
"""
Test Parallel Softmax - 4 Tiles Processing Simultaneously
AMD Phoenix NPU - XDNA1
"""

import numpy as np
import pyxrt as xrt
import struct
import time

def bf16_to_float(bf16_bytes):
    """Convert BF16 bytes to float32"""
    result = np.zeros(len(bf16_bytes) // 2, dtype=np.float32)
    for i in range(len(result)):
        upper = struct.unpack('H', bf16_bytes[i*2:(i+1)*2])[0]
        result[i] = struct.unpack('f', struct.pack('I', upper << 16))[0]
    return result

def float_to_bf16(floats):
    """Convert float32 to BF16 bytes"""
    result = bytearray(len(floats) * 2)
    for i, val in enumerate(floats):
        bits = struct.unpack('I', struct.pack('f', val))[0]
        upper = (bits >> 16) & 0xFFFF
        struct.pack_into('H', result, i*2, upper)
    return bytes(result)

def softmax_ref(x):
    """Reference softmax implementation"""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

def main():
    print("=" * 70)
    print("Parallel Softmax BF16 - 4 Tiles Test - AMD Phoenix NPU")
    print("=" * 70)
    print()

    num_elements = 1024
    num_frames = 4
    num_iterations = 10
    buffer_size = num_elements * 2  # 2048 bytes per frame

    print(f"Test Configuration:")
    print(f"  Elements per frame: {num_elements}")
    print(f"  Parallel frames: {num_frames}")
    print(f"  Total elements: {num_elements * num_frames}")
    print(f"  Buffer size per frame: {buffer_size} bytes")
    print(f"  Iterations: {num_iterations}")
    print()

    xclbin_path = "build_softmax_parallel/softmax_parallel_4tile.xclbin"
    insts_path = "build_softmax_parallel/insts.bin"

    try:
        # Load XCLBIN
        print("Step 1: Loading XCLBIN...")
        device = xrt.device(0)
        xclbin_obj = xrt.xclbin(xclbin_path)
        uuid = xclbin_obj.get_uuid()
        device.register_xclbin(xclbin_obj)
        print("XCLBIN loaded successfully")
        print()

        # Create hardware context
        print("Step 2: Creating hardware context...")
        hw_ctx = xrt.hw_context(device, uuid)
        print("Hardware context created")
        print()

        # Get kernel
        print("Step 3: Getting kernel...")
        kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
        print("Kernel found: MLIR_AIE")
        print()

        # Load instruction sequence
        print("Step 4: Loading instruction sequence...")
        with open(insts_path, "rb") as f:
            insts = f.read()
        print(f"Instructions loaded: {len(insts)} bytes")
        print()

        # Prepare test data - 4 different frames
        print("Step 5: Preparing test data for 4 frames...")
        input_floats = []
        input_bf16 = []
        expected = []
        
        for i in range(num_frames):
            # Different data for each frame
            data = np.random.randn(num_elements).astype(np.float32) * (i + 1)
            input_floats.append(data)
            input_bf16.append(float_to_bf16(data))
            expected.append(softmax_ref(data))
        
        print(f"Created {num_frames} test frames")
        print()

        # Allocate 8 XRT buffers (4 input + 4 output)
        print("Step 6: Allocating 8 XRT buffers...")
        
        # Instruction buffer
        bo_instr = xrt.bo(device, len(insts), xrt.bo.flags.cacheable, kernel.group_id(1))
        bo_instr.write(insts, 0)
        bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        
        # Input buffers (4)
        bo_in = []
        for i in range(num_frames):
            bo = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(3 + i))
            bo_in.append(bo)
        
        # Output buffers (4)
        bo_out = []
        for i in range(num_frames):
            bo = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(7 + i))
            bo_out.append(bo)
        
        print(f"Allocated {num_frames} input and {num_frames} output buffers")
        print()

        # Run kernel
        print("Step 7: Running parallel kernel on NPU...")
        times = []
        opcode = 3

        for iteration in range(num_iterations):
            # Write input data to all 4 buffers
            for i in range(num_frames):
                bo_in[i].write(input_bf16[i], 0)
                bo_in[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            # Execute kernel with all 8 buffers
            start = time.perf_counter()
            run = kernel(opcode, bo_instr, len(insts),
                        bo_in[0], bo_in[1], bo_in[2], bo_in[3],
                        bo_out[0], bo_out[1], bo_out[2], bo_out[3])
            run.wait()
            end = time.perf_counter()
            times.append(end - start)

            # Read output from all 4 buffers
            output_floats = []
            for i in range(num_frames):
                bo_out[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                output_bytes = bo_out[i].read(buffer_size, 0).tobytes()
                output_floats.append(bf16_to_float(output_bytes))

        print("Kernel execution complete")
        print()

        # Performance metrics
        print("Step 8: Performance Measurements:")
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        per_frame_time = avg_time / num_frames

        print(f"  Average time (4 frames): {avg_time:.3f} ms")
        print(f"  Per-frame time: {per_frame_time:.3f} ms")
        print(f"  Std deviation: {std_time:.3f} ms")
        print(f"  Min time: {min_time:.3f} ms")
        print(f"  Max time: {max_time:.3f} ms")
        print()

        # Compare with batched kernel baseline
        batched_time = 5.4  # ms from previous test
        single_time = 1.565  # ms from single kernel test
        
        print("Performance Comparison:")
        print(f"  Single kernel: {single_time:.3f} ms per frame")
        print(f"  Batched (4x): {batched_time:.1f} ms total ({batched_time/4:.2f} ms/frame)")
        print(f"  Parallel (4 tiles): {avg_time:.3f} ms total ({per_frame_time:.3f} ms/frame)")
        print()
        
        if avg_time < batched_time:
            speedup = batched_time / avg_time
            print(f"  Parallel SPEEDUP: {speedup:.2f}x vs batched")
        else:
            print(f"  Note: Parallel slower than batched")
        print()

        # Verify accuracy for all frames
        print("Step 9: Verifying accuracy for all 4 frames...")
        all_pass = True
        
        for i in range(num_frames):
            max_error = np.max(np.abs(output_floats[i] - expected[i]))
            correlation = np.corrcoef(output_floats[i], expected[i])[0, 1]
            
            if max_error < 0.02 and correlation > 0.995:
                status = "PASS"
            else:
                status = "FAIL"
                all_pass = False
            
            print(f"  Frame {i}: max_error={max_error:.6f}, correlation={correlation:.6f} [{status}]")
        
        print()
        if all_pass:
            print("  All frames accuracy check PASSED")
        else:
            print("  Some frames FAILED accuracy check")
        print()

        # Throughput calculation
        total_ops = num_frames * num_elements
        throughput = total_ops / (avg_time / 1000)  # ops per second
        print(f"Throughput: {throughput/1e6:.2f} M softmax-elements/second")
        print()

        print("=" * 70)
        print("PARALLEL SOFTMAX KERNEL TEST COMPLETE!")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
