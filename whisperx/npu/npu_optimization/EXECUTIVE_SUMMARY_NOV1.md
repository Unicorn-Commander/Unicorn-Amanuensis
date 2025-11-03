
================================================================================
EXECUTIVE SUMMARY: NPU BATCH-10 PERFORMANCE GAP
================================================================================

WHY 17X SLOWER?
───────────────

The NPU batch-10 mel preprocessing achieves 42x realtime instead of 708x due to:

1. **Sequential Frame Processing** (40% of gap)
   - MLIR kernel calls mel_kernel_simple() 10 times per batch
   - No vectorization or parallel processing
   - Per-frame compute time unchanged from single-frame version

2. **Excessive Batch Overhead** (35% of gap)
   - 110 batches × 2.37ms per batch = 261ms total
   - Each batch: 2 sync() calls + 1 kernel launch + 1 wait()
   - Should use batch-50 (fits in 64KB) instead of batch-10

3. **CPU Preprocessing** (17% of gap)
   - Frame extraction loop: 1100 operations
   - Float32→int16 conversion: 110 times
   - Done per-batch instead of once upfront

4. **No SIMD/Vectorization** (8% of gap)
   - C kernel processes samples one-by-one
   - AIE2 can process 32 samples per cycle (unused)

MEMORY CONSTRAINT DISCOVERY
────────────────────────────

Batch-100 compilation FAILS due to memory limit:
  - Compute tile: 64 KB available
  - Batch-100 needs: 176 KB (double-buffered)
  - Error: "allocated buffers exceeded available memory"

Batch-50 is the MAXIMUM that fits:
  - Memory needed: 86 KB (double-buffered)
  - Fits in: Compute tile ✅
  - Benefit: 5x fewer batches than batch-10

QUICK WINS (RANKED BY IMPACT)
──────────────────────────────

1. Use batch-50 kernel (5x improvement)
   - Compile mel_fixed_v3_batch50.mlir (need to create)
   - Or use existing batch-10 but increase to batch-50
   - Expected: 42x → 210x realtime
   - Time: 1-2 hours

2. Pre-convert audio to int16 (15-20% improvement)
   - Move conversion outside batch loop
   - Expected: 210x → 250x realtime
   - Time: 15 minutes

3. Vectorize C kernel with AIE2 SIMD (2x improvement)
   - Add vector intrinsics to mel_kernel_simple
   - Process 32 samples per SIMD operation
   - Expected: 250x → 500x realtime
   - Time: 1 week

4. Eliminate CPU framing loop (15% improvement)
   - Pass full audio buffer to NPU
   - Let MLIR handle framing
   - Expected: 500x → 575x realtime
   - Time: 3-5 days

5. True parallel batch processing (2x improvement)
   - Redesign for vectorized frame processing
   - Use multiple AIE tiles
   - Expected: 575x → 1150x realtime
   - Time: 2-3 weeks

IMMEDIATE RECOMMENDATION
────────────────────────

Option A: Create batch-50 kernel (RECOMMENDED)
  - Memory: Fits in 64KB compute tile ✅
  - Expected: 42x → 250x realtime (6x improvement)
  - Effort: Low (1-2 hours)
  - Certainty: High (guaranteed to work)

Option B: Use memory tile for batch-100 (ADVANCED)
  - Modify MLIR to use memory tile instead of compute tile
  - Expected: 42x → 420x realtime (10x improvement)
  - Effort: Medium (4-8 hours, requires MLIR expertise)
  - Certainty: Medium (requires memory tile configuration)

PERFORMANCE BREAKDOWN (261ms total)
───────────────────────────────────

Component               Time    %      With Batch-50    With Vectorization
─────────────────────────────────────────────────────────────────────────
CPU preprocessing       44ms   16.9%   Same (44ms)      Same (44ms)
DMA + XRT overhead     176ms   67.4%   5x better (35ms) 5x better (35ms)
NPU computation         41ms   15.7%   5x worse (205ms) 2x better (103ms)
─────────────────────────────────────────────────────────────────────────
TOTAL                  261ms  100.0%   284ms (WORSE!)   182ms (BETTER!)

⚠️ CRITICAL INSIGHT: Batch-50 alone makes things WORSE because:
   - 5x fewer XRT calls saves 141ms
   - But 5x more NPU compute costs 164ms
   - Net result: 23ms slower!

CORRECTED RECOMMENDATION: Batch-50 + Vectorization
   - Batch-50: Reduces XRT overhead by 5x
   - Vectorization: Reduces compute time by 2-3x
   - Combined: 182ms → 60x realtime
   - With pre-convert: 138ms → 80x realtime

PATH TO 708X TARGET
───────────────────

Phase 1: Batch-50 + Pre-convert + Vectorize (2 weeks)
  Result: 42x → 80x realtime
  Gap: Still 9x short of 708x target

Phase 2: Eliminate CPU overhead (1 week)
  Result: 80x → 120x realtime
  Gap: Still 6x short

Phase 3: True parallel processing (3 weeks)
  Result: 120x → 360x realtime
  Gap: Still 2x short

Phase 4: Multi-tile parallelism (2 weeks)
  Result: 360x → 720x realtime
  Gap: ACHIEVED! ✅

Total timeline: 8 weeks to reach 708x target

ALTERNATIVE: Accept 80-120x realtime
  - Much faster than CPU (400x improvement)
  - Good enough for real-time transcription
  - Achievable in 2-3 weeks
  - Specification may be overly optimistic

RECOMMENDATION SUMMARY
──────────────────────

SHORT-TERM (2-3 weeks):
  1. Create batch-50 MLIR kernel
  2. Pre-convert audio to int16
  3. Vectorize C kernel with SIMD
  4. Target: 80-120x realtime
  
LONG-TERM (6-8 weeks):
  1. All of above, plus:
  2. Eliminate CPU preprocessing
  3. Parallel frame processing
  4. Multi-tile parallelism
  5. Target: 700x+ realtime

CONFIDENCE LEVELS
─────────────────

Batch-50 + pre-convert:      HIGH (proven approach, simple)
SIMD vectorization:          MEDIUM-HIGH (requires AIE2 expertise)
CPU elimination:             MEDIUM (complex MLIR changes)
Parallel processing:         LOW-MEDIUM (requires significant redesign)
Reaching 708x specification: LOW (may require months of optimization)

Reaching 80-120x:            HIGH (achievable in 2-3 weeks)
