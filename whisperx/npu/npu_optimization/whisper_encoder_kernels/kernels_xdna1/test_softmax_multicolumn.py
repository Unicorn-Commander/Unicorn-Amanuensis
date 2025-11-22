#!/usr/bin/env python3
"""
Test Multi-Column Softmax - 4 Tiles Processing Simultaneously
AMD Phoenix NPU - XDNA1

ISSUE: Kernel compilation limitation discovered.
The MLIR defines 8 memref arguments but the compiled xclbin only supports 5 data slots.
This test documents the issue and provides diagnostic information.
"""

import numpy as np
import pyxrt as xrt
import struct
import time
import sys

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
    print("Multi-Column Softmax BF16 - 4 Tiles Test - AMD Phoenix NPU")
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
    print(f"Architecture:")
    print(f"  Column 0: ShimNOC (0,0), Compute (0,2), (0,3)")
    print(f"  Column 1: ShimNOC (1,0), Compute (1,2), (1,3)")
    print()

    xclbin_path = "build_softmax_multicolumn/softmax_multicolumn.xclbin"
    insts_path = "build_softmax_multicolumn/insts.bin"

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

        # Analyze kernel group_ids
        print("Step 4: Analyzing kernel arguments...")
        print("  Available group_ids:")
        for i in range(8):
            try:
                gid = kernel.group_id(i)
                print(f"    group_id({i}) = {gid}")
            except:
                break
        print()

        # Check for 9th group_id (would be needed for 8 data buffers)
        try:
            gid8 = kernel.group_id(8)
            print(f"  group_id(8) exists = {gid8}")
        except Exception as e:
            print(f"  group_id(8) NOT AVAILABLE - Error: {type(e).__name__}")
            print()
            print("  ISSUE: Kernel has only 8 group_ids (0-7)")
            print("  With groups 0-2 reserved (opcode, instr, len),")
            print("  only 5 data slots are available (groups 3-7)")
            print("  But the MLIR defines 8 memref arguments!")
        print()

        # Load instruction sequence
        print("Step 5: Loading instruction sequence...")
        with open(insts_path, "rb") as f:
            insts = f.read()
        print(f"Instructions loaded: {len(insts)} bytes")
        print()

        # Test different argument counts
        print("Step 6: Testing kernel call signatures...")

        # Instruction buffer
        bo_instr = xrt.bo(device, len(insts), xrt.bo.flags.cacheable, kernel.group_id(1))
        bo_instr.write(insts, 0)
        bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Allocate test buffers
        buffers = []
        for i in range(5):
            bo = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(3 + i % 5))
            buffers.append(bo)

        # Test with 5 buffers (max supported)
        opcode = 3
        print("  Testing with 5 data buffers...")
        sys.stdout.flush()
        try:
            run = kernel(opcode, bo_instr, len(insts),
                        buffers[0], buffers[1], buffers[2], buffers[3], buffers[4])
            run.wait()
            print("    5-buffer call: SUCCESS")
        except Exception as e:
            print(f"    5-buffer call: FAILED - {e}")

        # Test with 4 buffers (like 2-tile parallel)
        print("  Testing with 4 data buffers...")
        sys.stdout.flush()
        try:
            run = kernel(opcode, bo_instr, len(insts),
                        buffers[0], buffers[1], buffers[2], buffers[3])
            run.wait()
            print("    4-buffer call: SUCCESS")
        except Exception as e:
            print(f"    4-buffer call: FAILED - {e}")

        # NOTE: Testing with 8 buffers causes SEGFAULT (not catchable in Python)
        # The kernel only supports up to 5 data arguments
        print("  Testing with 8 data buffers (as MLIR designed)...")
        print("    SKIPPED - Would cause SEGFAULT")
        print("    (Confirmed by previous test runs)")
        print()
        print("  CONFIRMED: Kernel cannot accept 8 data buffers!")
        print()

        # Summary
        print("=" * 70)
        print("SUMMARY - KERNEL COMPILATION ISSUE")
        print("=" * 70)
        print()
        print("PROBLEM:")
        print("  The multi-column 4-tile kernel MLIR defines 8 memref arguments")
        print("  in the runtime_sequence, but the compiled xclbin only has")
        print("  8 group_ids total (0-7), providing only 5 data argument slots.")
        print()
        print("IMPACT:")
        print("  - Cannot pass 8 separate buffers for 4 input + 4 output")
        print("  - Kernel call with 8 arguments causes segfault")
        print("  - 4-tile multi-column parallelism cannot be tested")
        print()
        print("ROOT CAUSE:")
        print("  MLIR-AIE kernel compilation generates fixed argument count based")
        print("  on group_id allocation, which defaults to max 8 arguments total.")
        print()
        print("SOLUTIONS:")
        print("  1. Recompile kernel with extended argument support")
        print("     - May require custom aiecc.py options or MLIR changes")
        print("  2. Redesign MLIR to use combined buffers")
        print("     - Single input buffer for all 4 frames (8192 bytes)")
        print("     - Single output buffer for all 4 frames (8192 bytes)")
        print("     - This approach is used by the batched kernel successfully")
        print("  3. Use 2-column approach with 2 separate kernel calls")
        print("     - First call: column 0 (frames 0,1)")
        print("     - Second call: column 1 (frames 2,3)")
        print()
        print("COMPARISON WITH WORKING 2-TILE KERNEL:")
        print("  - 2-tile parallel: 4 data buffers (groups 3,4,5,6) - WORKS")
        print("  - 4-tile multi-column: 8 data buffers - FAILS")
        print()

        # Performance baseline for comparison
        single_time = 1.565  # ms from single kernel test
        two_tile_time = 1.548  # ms for 2 frames parallel (0.774 ms/frame)

        print("EXPECTED vs ACHIEVED:")
        print(f"  Single kernel: {single_time:.3f} ms per frame")
        print(f"  2-tile parallel: {two_tile_time:.3f} ms for 2 frames")
        print(f"  4-tile expected: ~1.6-2.0 ms for 4 frames")
        print(f"  4-tile achieved: CANNOT TEST due to compilation issue")
        print()

        print("STATUS: KERNEL NEEDS RECOMPILATION")
        print()
        print("=" * 70)
        print("MULTI-COLUMN SOFTMAX TEST COMPLETE")
        print("=" * 70)

        return 1  # Return error code to indicate issue

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
