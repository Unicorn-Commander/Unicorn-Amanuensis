#!/usr/bin/env python3
"""
XRT Buffer Allocation Strategy Testing
Tests all possible buffer allocation methods to find working combination
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
import time
from pathlib import Path

def test_allocation(device, kernel, config, test_name):
    """Test a specific buffer allocation configuration"""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")

    try:
        # Read instructions
        insts_path = config['insts_path']
        with open(insts_path, 'rb') as f:
            insts_data = f.read()
        n_insts = len(insts_data)

        print(f"Configuration:")
        print(f"  Instructions: {n_insts} bytes")
        print(f"  Input size: {config['input_size']} bytes")
        print(f"  Output size: {config['output_size']} bytes")
        print(f"  Instruction flags: {config['instr_flags']}")
        print(f"  Input flags: {config['input_flags']}")
        print(f"  Output flags: {config['output_flags']}")
        print(f"  Instruction group_id: {config['instr_gid']}")
        print(f"  Input group_id: {config['input_gid']}")
        print(f"  Output group_id: {config['output_gid']}")
        print()

        # Create buffers with specified configuration
        instr_bo = xrt.bo(
            device, n_insts,
            config['instr_flags'],
            kernel.group_id(config['instr_gid'])
        )

        input_bo = xrt.bo(
            device, config['input_size'],
            config['input_flags'],
            kernel.group_id(config['input_gid'])
        )

        output_bo = xrt.bo(
            device, config['output_size'],
            config['output_flags'],
            kernel.group_id(config['output_gid'])
        )

        print("✅ Buffers created successfully")

        # Write instructions
        instr_bo.write(insts_data, 0)
        instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

        # Write test input data
        test_input = np.random.randint(-64, 64, config['input_size'], dtype=np.int8)
        input_bo.write(test_input.tobytes(), 0)
        input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, config['input_size'], 0)

        print("✅ Data written to buffers")

        # Execute kernel
        print("Executing kernel...")
        opcode = 3
        start = time.perf_counter()
        run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
        run.wait(1000)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"✅ Execution completed: {elapsed_ms:.2f}ms")

        # Read output
        output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, config['output_size'], 0)
        output_data = np.frombuffer(output_bo.read(config['output_size'], 0), dtype=np.int8)

        # Analyze output
        nonzero_count = np.count_nonzero(output_data)
        nonzero_pct = 100.0 * nonzero_count / output_data.size

        print(f"\nResults:")
        print(f"  Non-zero values: {nonzero_count}/{output_data.size} ({nonzero_pct:.1f}%)")
        print(f"  Value range: [{output_data.min()}, {output_data.max()}]")
        print(f"  Mean: {output_data.mean():.2f}")
        print(f"  Std: {output_data.std():.2f}")

        # Verdict
        if nonzero_pct > 50:
            print(f"\n🎉 SUCCESS! This configuration works!")
            return True, {
                'nonzero_pct': nonzero_pct,
                'elapsed_ms': elapsed_ms,
                'output_stats': {
                    'min': int(output_data.min()),
                    'max': int(output_data.max()),
                    'mean': float(output_data.mean()),
                    'std': float(output_data.std())
                }
            }
        elif nonzero_pct > 0:
            print(f"\n⚠️  Partial success - some computation happening")
            return False, {'nonzero_pct': nonzero_pct}
        else:
            print(f"\n❌ All zeros - configuration doesn't work")
            return False, {'nonzero_pct': 0}

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False, {'error': str(e)}


def main():
    print("="*70)
    print("XRT BUFFER ALLOCATION STRATEGY TESTING")
    print("="*70)
    print()

    # Initialize NPU
    print("Initializing NPU...")
    device = xrt.device(0)

    # Load attention kernel
    base = Path(__file__).parent
    xclbin_path = base / "build_attention_64x64/attention_64x64.xclbin"
    insts_path = base / "build_attention_64x64/insts.bin"

    print(f"Loading: {xclbin_path}")
    xclbin = xrt.xclbin(str(xclbin_path))
    device.register_xclbin(xclbin)
    uuid = xclbin.get_uuid()
    hw_ctx = xrt.hw_context(device, uuid)
    kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
    print("✅ Kernel loaded")
    print()

    # Check available group IDs
    print("Checking kernel arguments and group IDs:")
    for i in range(6):
        try:
            gid = kernel.group_id(i)
            print(f"  Arg {i}: group_id = {gid}")
        except:
            break
    print()

    # Standard sizes for attention kernel
    INPUT_SIZE = 12288   # Q+K+V combined
    OUTPUT_SIZE = 4096   # 64x64 output

    # Test configurations to try
    test_configs = []

    # Test 1: BASELINE (current broken configuration)
    test_configs.append({
        'name': 'Baseline (group 1,2,3 with host_only)',
        'insts_path': insts_path,
        'input_size': INPUT_SIZE,
        'output_size': OUTPUT_SIZE,
        'instr_flags': xrt.bo.flags.cacheable,
        'input_flags': xrt.bo.flags.host_only,
        'output_flags': xrt.bo.flags.host_only,
        'instr_gid': 1,
        'input_gid': 2,
        'output_gid': 3
    })

    # Test 2: WORKING MEL PATTERN (group 1,3,4 with host_only)
    test_configs.append({
        'name': 'Working Mel Pattern (group 1,3,4)',
        'insts_path': insts_path,
        'input_size': INPUT_SIZE,
        'output_size': OUTPUT_SIZE,
        'instr_flags': xrt.bo.flags.cacheable,
        'input_flags': xrt.bo.flags.host_only,
        'output_flags': xrt.bo.flags.host_only,
        'instr_gid': 1,
        'input_gid': 3,
        'output_gid': 4
    })

    # Test 3: device_only flags
    test_configs.append({
        'name': 'Device Only (group 1,2,3)',
        'insts_path': insts_path,
        'input_size': INPUT_SIZE,
        'output_size': OUTPUT_SIZE,
        'instr_flags': xrt.bo.flags.cacheable,
        'input_flags': xrt.bo.flags.device_only,
        'output_flags': xrt.bo.flags.device_only,
        'instr_gid': 1,
        'input_gid': 2,
        'output_gid': 3
    })

    # Test 4: device_only with mel pattern
    test_configs.append({
        'name': 'Device Only (group 1,3,4)',
        'insts_path': insts_path,
        'input_size': INPUT_SIZE,
        'output_size': OUTPUT_SIZE,
        'instr_flags': xrt.bo.flags.cacheable,
        'input_flags': xrt.bo.flags.device_only,
        'output_flags': xrt.bo.flags.device_only,
        'instr_gid': 1,
        'input_gid': 3,
        'output_gid': 4
    })

    # Test 5: p2p flags
    test_configs.append({
        'name': 'P2P (group 1,2,3)',
        'insts_path': insts_path,
        'input_size': INPUT_SIZE,
        'output_size': OUTPUT_SIZE,
        'instr_flags': xrt.bo.flags.cacheable,
        'input_flags': xrt.bo.flags.p2p,
        'output_flags': xrt.bo.flags.p2p,
        'instr_gid': 1,
        'input_gid': 2,
        'output_gid': 3
    })

    # Test 6: No flags (let XRT auto-allocate)
    test_configs.append({
        'name': 'Auto Allocate (no flags, group 1,2,3)',
        'insts_path': insts_path,
        'input_size': INPUT_SIZE,
        'output_size': OUTPUT_SIZE,
        'instr_flags': xrt.bo.flags.cacheable,
        'input_flags': 0,  # No flags
        'output_flags': 0,  # No flags
        'instr_gid': 1,
        'input_gid': 2,
        'output_gid': 3
    })

    # Test 7: No flags with mel pattern
    test_configs.append({
        'name': 'Auto Allocate (no flags, group 1,3,4)',
        'insts_path': insts_path,
        'input_size': INPUT_SIZE,
        'output_size': OUTPUT_SIZE,
        'instr_flags': xrt.bo.flags.cacheable,
        'input_flags': 0,
        'output_flags': 0,
        'instr_gid': 1,
        'input_gid': 3,
        'output_gid': 4
    })

    # Test 8: Try group 0 for input
    test_configs.append({
        'name': 'Group 0 for input (0,1,2)',
        'insts_path': insts_path,
        'input_size': INPUT_SIZE,
        'output_size': OUTPUT_SIZE,
        'instr_flags': xrt.bo.flags.cacheable,
        'input_flags': xrt.bo.flags.host_only,
        'output_flags': xrt.bo.flags.host_only,
        'instr_gid': 1,
        'input_gid': 0,
        'output_gid': 2
    })

    # Test 9: Smaller sizes (same as mel)
    test_configs.append({
        'name': 'Small Size Like Mel (800/80 bytes)',
        'insts_path': insts_path,
        'input_size': 800,   # Same as mel
        'output_size': 80,    # Same as mel
        'instr_flags': xrt.bo.flags.cacheable,
        'input_flags': xrt.bo.flags.host_only,
        'output_flags': xrt.bo.flags.host_only,
        'instr_gid': 1,
        'input_gid': 3,
        'output_gid': 4
    })

    # Test 10: Mixed flags
    test_configs.append({
        'name': 'Mixed Flags (device input, host output)',
        'insts_path': insts_path,
        'input_size': INPUT_SIZE,
        'output_size': OUTPUT_SIZE,
        'instr_flags': xrt.bo.flags.cacheable,
        'input_flags': xrt.bo.flags.device_only,
        'output_flags': xrt.bo.flags.host_only,
        'instr_gid': 1,
        'input_gid': 2,
        'output_gid': 3
    })

    # Run all tests
    results = {}
    working_configs = []

    for i, config in enumerate(test_configs, 1):
        success, result = test_allocation(device, kernel, config,
                                          f"{i}/{len(test_configs)}: {config['name']}")
        results[config['name']] = result

        if success:
            working_configs.append(config)

        time.sleep(0.5)  # Brief pause between tests

    # Summary
    print("\n")
    print("="*70)
    print("SUMMARY OF ALL TESTS")
    print("="*70)
    print()

    for name, result in results.items():
        if 'error' in result:
            status = f"❌ ERROR: {result['error']}"
        elif result['nonzero_pct'] > 50:
            status = f"✅ SUCCESS ({result['nonzero_pct']:.1f}% non-zero)"
        elif result['nonzero_pct'] > 0:
            status = f"⚠️  PARTIAL ({result['nonzero_pct']:.1f}% non-zero)"
        else:
            status = "❌ FAILED (all zeros)"

        print(f"{name}:")
        print(f"  {status}")
        print()

    # Report working configurations
    if working_configs:
        print("="*70)
        print("✅ WORKING CONFIGURATIONS FOUND!")
        print("="*70)
        print()
        for config in working_configs:
            print(f"Configuration: {config['name']}")
            print(f"  instr_flags: {config['instr_flags']}")
            print(f"  input_flags: {config['input_flags']}")
            print(f"  output_flags: {config['output_flags']}")
            print(f"  group_ids: instr={config['instr_gid']}, "
                  f"input={config['input_gid']}, output={config['output_gid']}")
            print()
    else:
        print("="*70)
        print("❌ NO WORKING CONFIGURATIONS FOUND")
        print("="*70)
        print()
        print("This suggests the issue is NOT buffer allocation but kernel computation.")
        print()
        print("Next steps:")
        print("1. Examine attention kernel C++ source code")
        print("2. Review MLIR ObjectFIFO configuration")
        print("3. Compare with working mel kernel MLIR")
        print("4. Test with simpler attention computation")
        print()

if __name__ == "__main__":
    main()
