# Phase 1 Day 3: Action Plan and Priorities

**Date**: November 3, 2025
**Session Focus**: Quick Wins - MatMul Testing & Decoder Fixes
**Estimated Time**: 6-8 hours
**Goal**: Demonstrate measurable performance improvements

---

## Mission Summary

Based on Day 2 findings, we have **2 high-value quick wins** ready to execute:

1. ✅ **Batched MatMul**: Already implemented, needs testing (10x speedup)
2. ⏳ **Decoder Fixes**: Documented, needs implementation (working transcription)

**Strategy**: Focus on deliverables that work NOW, defer complex issues to Week 2.

---

## Priority 1: Test Batched MatMul (2 hours)

### Goal
Validate the existing `npu_matmul_wrapper_batched.py` delivers 10x speedup.

### Tasks

#### Task 1.1: Simple Functionality Test (30 min)
```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels

# Create quick test script
cat > test_batched_quick.py <<'EOF'
#!/usr/bin/env python3
import numpy as np
import time
from npu_matmul_wrapper_batched import NPUMatmulBatched

print("Testing Batched MatMul...")

# Initialize
matmul = NPUMatmulBatched()

# Test 64×64
print("\n64×64 matrix:")
A = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
B = np.random.randint(-64, 64, (64, 64), dtype=np.int8)

start = time.time()
C = matmul(A, B)
elapsed = time.time() - start

print(f"  Result shape: {C.shape}")
print(f"  Time: {elapsed*1000:.2f}ms")
print(f"  Output range: [{C.min()}, {C.max()}]")

# Test 128×128
print("\n128×128 matrix:")
A = np.random.randint(-64, 64, (128, 128), dtype=np.int8)
B = np.random.randint(-64, 64, (128, 128), dtype=np.int8)

start = time.time()
C = matmul(A, B)
elapsed = time.time() - start

print(f"  Result shape: {C.shape}")
print(f"  Time: {elapsed*1000:.2f}ms")
print(f"  Output range: [{C.min()}, {C.max()}]")

print("\n✅ Batched matmul functional!")
EOF

chmod +x test_batched_quick.py
python3 test_batched_quick.py
```

**Expected Output**:
- 64×64: ~10ms (vs 34ms sequential)
- 128×128: ~50ms (vs 235ms sequential)

#### Task 1.2: Accuracy Validation (30 min)
```python
# Add to test script
import torch

# Test accuracy
A_np = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
B_np = np.random.randint(-64, 64, (64, 64), dtype=np.int8)

# NPU computation
C_npu = matmul(A_np, B_np)

# CPU reference
A_torch = torch.from_numpy(A_np).to(torch.int32)
B_torch = torch.from_numpy(B_np).to(torch.int32)
C_torch = (A_torch @ B_torch) >> 7  # Same scale shift
C_ref = C_torch.clamp(-128, 127).to(torch.int8).numpy()

# Compare
diff = np.abs(C_npu.astype(np.int16) - C_ref.astype(np.int16))
mae = diff.mean()
max_diff = diff.max()
match_rate = (diff <= 1).mean() * 100

print(f"\nAccuracy:")
print(f"  MAE: {mae:.2f}")
print(f"  Max diff: {max_diff}")
print(f"  Match rate (±1): {match_rate:.1f}%")

assert mae < 2.0, "MAE too high!"
assert match_rate > 95.0, "Match rate too low!"
print("✅ Accuracy validated!")
```

#### Task 1.3: Performance Benchmark (1 hour)
```bash
# Run comprehensive benchmark
python3 test_npu_matmul_wrapper.py

# Expected output:
# 16×16:    ~2ms
# 64×64:    ~10ms (3.4x faster)
# 128×128:  ~50ms (4.7x faster)
# 512×512:  ~1.5s (10x faster) ← KEY METRIC
```

### Success Criteria

- ✅ Batched version runs without errors
- ✅ Output matches sequential version (±1 tolerance)
- ✅ Speedup >5x for large matrices
- ✅ 512×512 completes in <2 seconds

---

## Priority 2: Fix Decoder Imports (30 min)

### Goal
Make `test_decoder_simple.py` runnable for diagnostics.

### Tasks

#### Task 2.1: Fix Import Paths
```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization

# Edit onnx_whisper_npu.py
# Change lines 24-28 from:
#   sys.path.insert(0, '/app/npu')
#   sys.path.insert(0, '/app/npu/npu_optimization')
# To:
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'npu'))
sys.path.insert(0, str(BASE_DIR / 'npu' / 'npu_optimization'))
```

#### Task 2.2: Run Diagnostic Tests
```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
python3 test_decoder_simple.py > decoder_diagnostic_results.txt 2>&1

# Review output
cat decoder_diagnostic_results.txt
```

### Success Criteria

- ✅ Tests run without import errors
- ✅ Can see ONNX model structure
- ✅ Can see encoder outputs
- ✅ Can see decoder step-by-step debugging

---

## Priority 3: Implement Decoder Fixes (3-4 hours)

### Goal
Get decoder producing coherent text instead of garbled output.

### Tasks

**Based on DECODER_PHASE1_PLAN.md (16,000 words)**

#### Task 3.1: Fix KV Cache Extraction (1 hour)

**Current Code** (lines 277-285 in onnx_whisper_npu.py):
```python
# Potentially wrong indices
for i in range(6):
    past_key_values.append((
        decoder_outputs[i*2 + 1],   # decoder.key
        decoder_outputs[i*2 + 2],   # decoder.value
        decoder_outputs[i*2 + 13],  # encoder.key ← SUSPICIOUS
        decoder_outputs[i*2 + 14]   # encoder.value ← SUSPICIOUS
    ))
```

**Fix** (verify output indices first):
```python
# First, inspect actual output structure
print(f"Total decoder outputs: {len(decoder_outputs)}")
for i, out in enumerate(decoder_outputs):
    print(f"  Output {i}: shape {out.shape}")

# Then extract with correct indices
# Expected structure:
# [0] logits
# [1-12] present.0-5.decoder.{key,value} (6 layers × 2)
# [13-24] present.0-5.encoder.{key,value} (6 layers × 2)

for i in range(6):
    past_key_values.append((
        decoder_outputs[1 + i*2],       # decoder.key
        decoder_outputs[1 + i*2 + 1],   # decoder.value
        decoder_outputs[13 + i*2],      # encoder.key
        decoder_outputs[13 + i*2 + 1]   # encoder.value
    ))
```

#### Task 3.2: Pre-compute Encoder K/V (1 hour)

**Add after encoder runs** (line 224):
```python
# After encoder
hidden_states = encoder_outputs[0]

# Pre-compute encoder K/V (stays constant for all decoder steps)
encoder_kv_cache = None
if use_past:
    # Run decoder once to extract encoder K/V
    initial_outputs = self.decoder_session.run(None, {
        'input_ids': decoder_input_ids,
        'encoder_hidden_states': hidden_states
    })

    # Extract and save encoder K/V
    encoder_kv_cache = []
    for i in range(6):
        encoder_kv_cache.append((
            initial_outputs[13 + i*2],      # encoder.key
            initial_outputs[13 + i*2 + 1]   # encoder.value
        ))

    # Get initial logits
    logits = initial_outputs[0]
    # ... process first token ...
```

**In generation loop**, reuse encoder K/V:
```python
for step in range(448):
    if use_past and past_key_values:
        inputs = {'input_ids': decoder_input_ids[:, -1:]}

        for i, (dec_k, dec_v) in enumerate(past_key_values):
            inputs[f'past_key_values.{i}.decoder.key'] = dec_k
            inputs[f'past_key_values.{i}.decoder.value'] = dec_v
            # Use pre-computed encoder K/V (don't recompute!)
            inputs[f'past_key_values.{i}.encoder.key'] = encoder_kv_cache[i][0]
            inputs[f'past_key_values.{i}.encoder.value'] = encoder_kv_cache[i][1]
```

#### Task 3.3: Extend Token Generation (30 min)

**Current**: Limited to 20 tokens
**Fix**: Generate up to 448 tokens (Whisper's max)

```python
# Current (line 239):
for step in range(448):  # Already correct!
    if decoder_input_ids.shape[1] >= 448:
        break  # Already correct!

    # Issue: May be exiting early due to wrong EOS detection
    # or incorrect token IDs
```

Check:
1. EOS token is 50257 (correct)
2. Start tokens are correct: [50258, 50259, 50360, 50365]
3. Token filtering works: `t < 50257`

#### Task 3.4: Validate with Test Audio (1 hour)

```python
# Test with simple audio
audio, sr = librosa.load("test_hello.wav", sr=16000)

# Should produce: "hello" or similar
result = whisper.transcribe_audio("test_hello.wav")

print(f"Text: {result['text']}")
print(f"Segments: {result['segments']}")

# Check
assert result['text'].strip(), "Empty output!"
assert not result['text'].startswith('['), "Placeholder output!"
assert len(result['text']) > 5, "Too short!"

print("✅ Decoder producing meaningful text!")
```

### Success Criteria

- ✅ Decoder produces text (not placeholders)
- ✅ Text is coherent (matches audio content roughly)
- ✅ No crashes or errors
- ✅ Can transcribe 30s audio

---

## Priority 4: Integration and Benchmarking (1-2 hours)

### Goal
Measure end-to-end performance with batched matmul.

### Tasks

#### Task 4.1: Integrate Batched MatMul

```python
# In whisperx encoder
from npu_matmul_wrapper_batched import NPUMatmulBatched

# Initialize once
self.npu_matmul = NPUMatmulBatched()

# Use in encoder forward pass
def forward(self, x):
    # QKV projection
    qkv = self.npu_matmul(x, self.qkv_weights)  # 10x faster!

    # Attention (use CPU for now - Week 2 will fix)
    attn_output = self.cpu_attention(qkv)

    # Output projection
    output = self.npu_matmul(attn_output, self.out_weights)  # 10x faster!

    # FFN
    hidden = self.npu_matmul(output, self.ffn_w1)  # 10x faster!
    output = self.npu_matmul(hidden, self.ffn_w2)  # 10x faster!

    return output
```

#### Task 4.2: Benchmark Full Pipeline

```bash
# Test with 30s audio
python3 -c "
from onnx_whisper_npu import ONNXWhisperNPU
import time

whisper = ONNXWhisperNPU()
whisper.initialize('base')

start = time.time()
result = whisper.transcribe_audio('test_30s.wav')
elapsed = time.time() - start

print(f'Text: {result[\"text\"][:100]}...')
print(f'Time: {elapsed:.2f}s')
print(f'RTF: {30/elapsed:.1f}x')
"
```

**Expected Results**:

Without batched matmul:
- Encoder: 2.20s
- Decoder: 2.50s
- Total: ~5s
- RTF: 6x

With batched matmul:
- Encoder: 0.22s (10x faster!)
- Decoder: 2.50s (unchanged)
- Total: ~3s
- RTF: 10x

With batched matmul + decoder fixes:
- Encoder: 0.22s
- Decoder: 2.50s (working but not optimized)
- Total: ~3s
- RTF: 10x

### Success Criteria

- ✅ Encoder is 5-10x faster with batched matmul
- ✅ Decoder produces coherent text
- ✅ Overall RTF >10x (improvement over baseline)
- ✅ No accuracy regression (text quality maintained)

---

## Schedule

### Morning Session (3-4 hours)
- ✅ Test batched matmul (2 hours)
- ✅ Fix decoder imports (30 min)
- ✅ Run decoder diagnostics (30 min)
- ☕ Break

### Afternoon Session (3-4 hours)
- ✅ Implement decoder fixes (2-3 hours)
- ✅ Integration testing (1 hour)
- ✅ Benchmark and document (1 hour)

---

## Deliverables

### Code Changes
1. **test_batched_quick.py** - Quick matmul validation script
2. **onnx_whisper_npu.py** - Fixed imports and decoder improvements
3. **decoder_diagnostic_results.txt** - Test output log

### Documentation
1. **BATCHED_MATMUL_RESULTS.md** - Performance benchmarks
2. **DECODER_FIX_RESULTS.md** - Implementation notes and test results
3. **PHASE1_DAY3_SUMMARY.md** - Session summary and findings

### Performance Data
1. Matmul benchmarks (16×16 through 512×512)
2. End-to-end transcription times
3. RTF improvements
4. Accuracy validation results

---

## Risks and Mitigation

### Risk 1: Batched MatMul Has Bugs
**Probability**: LOW (code looks solid)
**Mitigation**: Validate with accuracy tests, compare to sequential version
**Fallback**: Use sequential version (still works)

### Risk 2: Decoder Fixes More Complex Than Expected
**Probability**: MEDIUM
**Mitigation**: Follow 16,000-word plan, test incrementally
**Fallback**: Use faster-whisper as temporary solution

### Risk 3: Integration Issues
**Probability**: LOW
**Mitigation**: Test each component separately first
**Fallback**: Keep CPU fallbacks in place

---

## Success Metrics

**Minimum** (Must Achieve):
- ✅ Batched matmul validated and working
- ✅ Decoder runs without crashes
- ✅ 5x encoder speedup measured

**Good** (Target):
- ✅ 10x encoder speedup with batched matmul
- ✅ Decoder produces coherent text
- ✅ 10-15x realtime factor

**Excellent** (Stretch):
- ✅ 10x matmul speedup confirmed
- ✅ Decoder accuracy validated (low WER)
- ✅ 15-20x realtime factor
- ✅ Ready for KV cache implementation (Week 2)

---

## Next Steps (Day 4-5)

### If Successful Today:
**Day 4**: Implement KV cache (25x decoder speedup)
**Day 5**: End-to-end validation and Week 1 review

### If Blocked Today:
**Day 4**: Continue decoder fixes, investigate blockers
**Day 5**: Document findings, plan Week 2 priorities

---

## Notes for Team

1. **Focus on Quick Wins**: Don't try to fix everything today
2. **Test Incrementally**: Validate each component before integration
3. **Document Everything**: Capture benchmarks and findings
4. **Keep Fallbacks**: Don't remove working code
5. **Ask for Help**: If stuck >30 min, escalate

---

**Action Plan Created**: November 3, 2025
**Session Lead**: Encoder/Decoder Phase 1 Team
**Estimated Duration**: 6-8 hours
**Target**: Demonstrate 10x encoder speedup, working decoder

**Let's make it happen! 🚀**

**🦄 Magic Unicorn Unconventional Technology & Stuff Inc.**
