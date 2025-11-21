#!/usr/bin/env python3
"""
Profile Memory Access Patterns in NPU LayerNorm
Measure DMA transfer time vs actual kernel execution time
"""

import pyxrt as xrt
import numpy as np
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

def profile_layernorm_single_call(device, kernel, insts_path, data_size=512):
    """Profile a single LayerNorm call with detailed timing"""

    # Generate test data
    x = np.random.randn(data_size).astype(np.float32) * 0.1

    timings = {}

    # 1. Load instructions
    t0 = time.time()
    with open(insts_path, "rb") as f:
        insts = f.read()
    timings['load_instructions'] = (time.time() - t0) * 1000

    # 2. BF16 conversion (CPU)
    t0 = time.time()
    input_bf16 = float_to_bf16(x)
    timings['bf16_conversion_to'] = (time.time() - t0) * 1000

    buffer_size = len(input_bf16)

    # 3. Allocate buffers
    t0 = time.time()
    bo_instr = xrt.bo(device, len(insts), xrt.bo.flags.cacheable, kernel.group_id(1))
    bo_input = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(3))
    bo_output = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(4))
    timings['buffer_allocation'] = (time.time() - t0) * 1000

    # 4. Write instructions to buffer
    t0 = time.time()
    bo_instr.write(insts, 0)
    timings['write_instructions'] = (time.time() - t0) * 1000

    # 5. Sync instructions to device
    t0 = time.time()
    bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    timings['sync_instructions'] = (time.time() - t0) * 1000

    # 6. Write input data
    t0 = time.time()
    bo_input.write(input_bf16, 0)
    timings['write_input'] = (time.time() - t0) * 1000

    # 7. DMA input to device
    t0 = time.time()
    bo_input.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    timings['dma_input_to_device'] = (time.time() - t0) * 1000

    # 8. Execute kernel
    t0 = time.time()
    run = kernel(3, bo_instr, len(insts), bo_input, bo_output)
    timings['kernel_launch'] = (time.time() - t0) * 1000

    # 9. Wait for completion
    t0 = time.time()
    run.wait()
    timings['kernel_execution'] = (time.time() - t0) * 1000

    # 10. DMA output from device
    t0 = time.time()
    bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    timings['dma_output_from_device'] = (time.time() - t0) * 1000

    # 11. Read output
    t0 = time.time()
    output_bytes = bo_output.read(buffer_size, 0).tobytes()
    timings['read_output'] = (time.time() - t0) * 1000

    # 12. BF16 conversion back (CPU)
    t0 = time.time()
    output_floats = bf16_to_float(output_bytes)
    timings['bf16_conversion_from'] = (time.time() - t0) * 1000

    timings['total'] = sum(timings.values())

    return timings, output_floats

def profile_batched_calls(device, kernel, insts_path, num_frames=100):
    """Profile multiple sequential calls (current bottleneck)"""

    print(f"\n📊 Profiling {num_frames} sequential LayerNorm calls...")

    all_timings = []

    for i in range(num_frames):
        timings, _ = profile_layernorm_single_call(device, kernel, insts_path)
        all_timings.append(timings)

        if i % 10 == 0:
            print(f"   Frame {i}/{num_frames}... {timings['total']:.2f}ms")

    # Compute statistics
    avg_timings = {}
    for key in all_timings[0].keys():
        values = [t[key] for t in all_timings]
        avg_timings[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    return avg_timings

def analyze_memory_bottleneck(avg_timings):
    """Analyze where time is spent"""

    print("\n" + "="*70)
    print("📊 MEMORY ACCESS ANALYSIS")
    print("="*70)

    print(f"\n⏱️  Average Timing per Frame:")
    print(f"{'Component':<30} {'Mean':<10} {'Min':<10} {'Max':<10} {'%'}")
    print("-" * 70)

    total = avg_timings['total']['mean']

    # Group by category
    categories = {
        'CPU Overhead': [
            'load_instructions',
            'bf16_conversion_to',
            'bf16_conversion_from'
        ],
        'Buffer Management': [
            'buffer_allocation',
            'write_instructions',
            'write_input',
            'read_output'
        ],
        'DMA Transfers': [
            'sync_instructions',
            'dma_input_to_device',
            'dma_output_from_device'
        ],
        'NPU Kernel': [
            'kernel_launch',
            'kernel_execution'
        ]
    }

    for category, keys in categories.items():
        cat_time = sum(avg_timings[k]['mean'] for k in keys if k in avg_timings)
        print(f"\n{category}:")

        for key in keys:
            if key not in avg_timings:
                continue
            t = avg_timings[key]
            pct = (t['mean'] / total) * 100
            print(f"  {key:<28} {t['mean']:>8.3f}ms {t['min']:>8.3f}ms {t['max']:>8.3f}ms {pct:>5.1f}%")

        cat_pct = (cat_time / total) * 100
        print(f"  {'─'*28} {'─'*9} {'─'*9} {'─'*9} {'─'*6}")
        print(f"  {'Subtotal':<28} {cat_time:>8.3f}ms {' '*18} {cat_pct:>5.1f}%")

    print("\n" + "─" * 70)
    print(f"{'TOTAL':<30} {total:>8.3f}ms")
    print("=" * 70)

    # Analysis
    print(f"\n🔍 BOTTLENECK ANALYSIS:\n")

    cpu_time = sum(avg_timings[k]['mean'] for k in categories['CPU Overhead'] if k in avg_timings)
    buffer_time = sum(avg_timings[k]['mean'] for k in categories['Buffer Management'] if k in avg_timings)
    dma_time = sum(avg_timings[k]['mean'] for k in categories['DMA Transfers'] if k in avg_timings)
    kernel_time = sum(avg_timings[k]['mean'] for k in categories['NPU Kernel'] if k in avg_timings)

    cpu_pct = (cpu_time / total) * 100
    buffer_pct = (buffer_time / total) * 100
    dma_pct = (dma_time / total) * 100
    kernel_pct = (kernel_time / total) * 100

    print(f"1. CPU Overhead:      {cpu_time:>8.3f}ms ({cpu_pct:>5.1f}%)")
    print(f"   - BF16 conversion is on CPU")
    print(f"   - Can be reduced with batching\n")

    print(f"2. Buffer Management: {buffer_time:>8.3f}ms ({buffer_pct:>5.1f}%)")
    print(f"   - Allocating 3 buffers per call")
    print(f"   - Can reuse buffers across calls!\n")

    print(f"3. DMA Transfers:     {dma_time:>8.3f}ms ({dma_pct:>5.1f}%)")
    print(f"   - PCIe/memory bandwidth limited")
    print(f"   - Batching will amortize this cost\n")

    print(f"4. NPU Kernel:        {kernel_time:>8.3f}ms ({kernel_pct:>5.1f}%)")
    print(f"   - Actual compute on NPU")
    print(f"   - This is the ONLY useful work!\n")

    # Calculate potential speedup
    overhead = total - kernel_time
    print(f"\n💡 OPTIMIZATION POTENTIAL:\n")
    print(f"Current overhead per call: {overhead:.3f}ms")
    print(f"Actual kernel time:        {kernel_time:.3f}ms")
    print(f"Overhead ratio:            {overhead/kernel_time:.1f}x\n")

    print(f"With batched processing:")
    print(f"  - Allocate buffers once (not 3001 times)")
    print(f"  - Single DMA transfer for all frames")
    print(f"  - BF16 conversion amortized")
    print(f"  - Expected overhead: ~10ms total (not {overhead:.1f}ms × 3001)")
    print(f"  - Expected speedup: ~{(overhead * 3001) / (kernel_time * 3001 + 10):.0f}x\n")

def estimate_batched_performance(avg_timings, num_frames=3001, layers=6):
    """Estimate performance with batched processing"""

    print("="*70)
    print("📈 PERFORMANCE PROJECTION WITH BATCHING")
    print("="*70)

    # Current performance
    current_per_frame = avg_timings['total']['mean']
    current_per_layer = current_per_frame * num_frames * 2  # 2 LayerNorms per layer
    current_total = current_per_layer * layers

    print(f"\n🐌 CURRENT (Sequential):")
    print(f"   Per frame:      {current_per_frame:.3f}ms")
    print(f"   Per LayerNorm:  {current_per_frame * num_frames:.1f}ms ({num_frames} calls)")
    print(f"   Per layer:      {current_per_layer:.1f}ms (2 LayerNorms)")
    print(f"   Total (6 layers): {current_total:.1f}ms")

    # Batched performance
    kernel_time = avg_timings['kernel_execution']['mean']

    # Batched: allocate once, one DMA in, process all frames, one DMA out
    batched_allocation = 5  # ms (one-time)
    batched_dma_in = 50  # ms (all frames at once)
    batched_kernel = kernel_time * num_frames  # Still need to process each frame
    batched_dma_out = 50  # ms (all frames at once)
    batched_conversion = 10  # ms (batch conversion)

    batched_per_layernorm = batched_allocation + batched_dma_in + batched_kernel + batched_dma_out + batched_conversion
    batched_per_layer = batched_per_layernorm * 2
    batched_total = batched_per_layer * layers

    print(f"\n🚀 PROJECTED (Batched):")
    print(f"   Buffer allocation:  {batched_allocation:.1f}ms (one-time)")
    print(f"   DMA to device:      {batched_dma_in:.1f}ms (all {num_frames} frames)")
    print(f"   Kernel execution:   {batched_kernel:.1f}ms ({num_frames} frames)")
    print(f"   DMA from device:    {batched_dma_out:.1f}ms (all {num_frames} frames)")
    print(f"   BF16 conversion:    {batched_conversion:.1f}ms")
    print(f"   Per LayerNorm:      {batched_per_layernorm:.1f}ms")
    print(f"   Per layer:          {batched_per_layer:.1f}ms (2 LayerNorms)")
    print(f"   Total (6 layers):   {batched_total:.1f}ms")

    speedup = current_total / batched_total
    print(f"\n✨ EXPECTED SPEEDUP: {speedup:.1f}x")
    print(f"   Current: {current_total/1000:.1f}s")
    print(f"   Batched: {batched_total/1000:.1f}s")

    # Realtime factor
    audio_duration = 5.0  # seconds
    current_rtf = audio_duration / (current_total / 1000)
    batched_rtf = audio_duration / (batched_total / 1000)

    print(f"\n🎯 REALTIME FACTOR:")
    print(f"   Current: {current_rtf:.3f}x")
    print(f"   Batched: {batched_rtf:.2f}x")
    print(f"   Improvement: {batched_rtf/current_rtf:.1f}x")

    print("\n" + "="*70)

def main():
    print("="*70)
    print("Memory Access Profiling for NPU LayerNorm")
    print("="*70)

    # Initialize NPU
    device_id = 0
    device = xrt.device(device_id)

    xclbin_path = "build_layernorm_nosqrt/main.xclbin"
    insts_path = "build_layernorm_nosqrt/main_sequence.bin"

    print(f"\n📦 Loading XCLBIN: {xclbin_path}")
    xclbin_obj = xrt.xclbin(xclbin_path)
    uuid = xclbin_obj.get_uuid()
    device.register_xclbin(xclbin_obj)

    hw_ctx = xrt.hw_context(device, uuid)
    kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
    print("   ✅ Kernel loaded")

    # Profile a single call first
    print(f"\n🔬 Profiling single LayerNorm call...")
    single_timings, output = profile_layernorm_single_call(device, kernel, insts_path)

    print(f"\n📊 Single Call Breakdown:")
    for key, value in single_timings.items():
        if key != 'total':
            print(f"   {key:<30} {value:>8.3f}ms")
    print(f"   {'─'*30} {'─'*9}")
    print(f"   {'TOTAL':<30} {single_timings['total']:>8.3f}ms")

    # Profile 100 calls to get statistics
    avg_timings = profile_batched_calls(device, kernel, insts_path, num_frames=100)

    # Analyze bottlenecks
    analyze_memory_bottleneck(avg_timings)

    # Project batched performance
    estimate_batched_performance(avg_timings)

    print("\n✅ Profiling complete!")

if __name__ == "__main__":
    main()
