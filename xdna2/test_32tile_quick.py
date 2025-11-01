#!/usr/bin/env python3
"""Quick test of 32-tile encoder - targeting 17x realtime!"""

import numpy as np
import time
import sys
sys.path.insert(0, "xdna2")
sys.path.insert(0, "/opt/xilinx/xrt/python")

from runtime.whisper_xdna2_runtime import create_runtime

print("="*70)
print("  32-TILE ENCODER QUICK TEST")
print("  Target: 17x realtime (vs 4.82x with 4-tile)")
print("="*70)

# Create runtime with 32-tile kernel
print("\nInitializing 32-tile runtime...")
runtime = create_runtime(model_size="base", use_4tile=False)  # FALSE = 32-tile!

print("Loading encoder weights...")
runtime._load_encoder_weights()

# Test input
np.random.seed(42)
hidden_states = np.random.randn(512, 512).astype(np.float32)

print("\nWarmup run...")
_ = runtime._run_encoder(hidden_states)

print("\nTimed run (32-tile)...")
start = time.perf_counter()
output = runtime._run_encoder(hidden_states)
latency = (time.perf_counter() - start) * 1000

# Calculate realtime factor
audio_duration = 10.24  # seconds
realtime_factor = (audio_duration * 1000) / latency

print("\n" + "="*70)
print("  RESULTS")
print("="*70)
print(f"\n32-Tile Performance:")
print(f"  Latency: {latency:.2f} ms ({latency/1000:.3f} seconds)")
print(f"  Audio duration: {audio_duration:.2f} seconds")
print(f"  Realtime factor: {realtime_factor:.2f}x")

print(f"\nComparison:")
print(f"  4-tile (Phase 3):  4.82x realtime")
print(f"  32-tile (Phase 4): {realtime_factor:.2f}x realtime")
print(f"  Speedup: {realtime_factor / 4.82:.2f}x")

if realtime_factor >= 17:
    print(f"\n✅ TARGET ACHIEVED: {realtime_factor:.2f}x >= 17x!")
elif realtime_factor >= 10:
    print(f"\n⚠️  GOOD: {realtime_factor:.2f}x (below 17x but significant improvement)")
else:
    print(f"\n⚠️  BELOW TARGET: {realtime_factor:.2f}x < 17x")

print("\n" + "="*70)
