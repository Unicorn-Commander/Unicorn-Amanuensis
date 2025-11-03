# NPU Decoder Phase 1: Implementation Plan
## Fix Garbled Output & Implement KV Cache (Weeks 1-2)

**Date**: November 2, 2025
**Lead**: Implementation Lead for NPU Decoder Phase 1
**Mission**: Get decoder producing accurate text with KV cache optimization
**Timeline**: 2 weeks (80-96 hours)

---

## Phase Overview

**Current State**:
- ✅ Encoder working at 36.1x realtime (proven)
- ❌ Decoder produces garbled/placeholder text ("..." repeated)
- ❌ No KV cache optimization (O(n²) complexity)

**Target State**:
- ✅ Decoder produces accurate transcription (WER <20%)
- ✅ KV cache implemented and working
- ✅ 20-30x realtime performance (Phase 1 target)

**Success Metrics**:
- Minimum: Decoder outputs readable text (not garbled)
- Good: WER <20%, 20x realtime
- Excellent: WER <10%, 30x realtime with cache

---

## Week 1: Fix Garbled Decoder Output

### Day 1: Investigation & Diagnosis (8-10 hours)

#### Task 1.1: Run Diagnostic Test Suite (2 hours)
**File**: `test_decoder_simple.py` (already created)

**Command**:
```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
python3 test_decoder_simple.py > decoder_diagnostic_results.txt 2>&1
```

**Expected Output**:
- ONNX model input/output names
- Encoder hidden states validation
- Step-by-step decoder token generation
- Full transcription test results

**Deliverable**: `decoder_diagnostic_results.txt` with findings

---

#### Task 1.2: Analyze ONNX Model Structure (2 hours)

**Goal**: Verify we're using correct input/output names for encoder and decoder

**Questions to Answer**:
1. What are the exact input names for the decoder?
   - Expected: `input_ids`, `encoder_hidden_states`
   - Verify actual names from ONNX metadata

2. What are the decoder output indices?
   - Expected: [0] = logits, [1-24] = KV tensors
   - Map each index to its meaning

3. Is decoder_with_past_model.onnx loaded correctly?
   - Verify it exists at expected path
   - Check its input/output structure

**Deliverable**: Document with exact ONNX structure

---

#### Task 1.3: Validate Encoder Outputs (2 hours)

**Test Code**:
```python
# After encoder runs
hidden_states = encoder_outputs[0]

print(f"Shape: {hidden_states.shape}")  # Expected: (1, 1500, 512)
print(f"Range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
print(f"Mean: {hidden_states.mean():.4f}, Std: {hidden_states.std():.4f}")
print(f"NaN count: {np.isnan(hidden_states).sum()}")
print(f"Inf count: {np.isinf(hidden_states).sum()}")

# Check first few values
print(f"First values: {hidden_states[0, 0, :10]}")
```

**Success Criteria**:
- Shape is (1, N, 512) where N depends on audio length
- Values in reasonable range (-10 to +10)
- No NaN or Inf values
- Non-zero, non-constant values

**Deliverable**: Encoder validation report

---

#### Task 1.4: Debug Token Generation Loop (2-4 hours)

**Goal**: Understand why decoder generates garbled tokens

**Test Strategy**:
1. Run decoder for 10 steps with extensive logging
2. Check top-5 tokens at each step (not just argmax)
3. Verify token IDs are valid (0-51865 range)
4. Decode each token individually to see what it represents

**Key Questions**:
1. Are all generated tokens special tokens (>= 50257)?
2. Do logits look reasonable (not all zeros/ones)?
3. Is the decoder stuck in a loop (generating same token)?
4. Do the top-5 tokens make linguistic sense?

**Deliverable**: Step-by-step token generation analysis

---

### Day 2: Identify Root Cause (6-8 hours)

#### Task 2.1: Test Encoder-Decoder Connection (3-4 hours)

**Hypothesis**: Encoder outputs not connected to decoder properly

**Test 1: Verify Input Name**
```python
# Get actual decoder input names
decoder_inputs = {inp.name: inp for inp in decoder_session.get_inputs()}
print("Decoder expects:", list(decoder_inputs.keys()))

# Common variations:
# - "encoder_hidden_states" (standard)
# - "encoder_outputs"
# - "cross_attention_hidden_states"
# - "encoder_attention_mask"

# Try different names
for name in ["encoder_hidden_states", "encoder_outputs", "hidden_states"]:
    if name in decoder_inputs:
        print(f"✅ Found: {name}")
```

**Test 2: Verify Hidden States Match Expected Shape**
```python
decoder_input_info = decoder_inputs.get("encoder_hidden_states")
print(f"Expected shape: {decoder_input_info.shape}")
print(f"Actual shape: {hidden_states.shape}")

# Should match!
```

**Deliverable**: Confirmed correct input name for encoder hidden states

---

#### Task 2.2: Validate KV Cache Extraction (3-4 hours)

**Hypothesis**: KV cache indices are wrong (lines 277-285 in onnx_whisper_npu.py)

**Current Code**:
```python
for i in range(6):  # 6 decoder layers
    past_key_values.append((
        decoder_outputs[i*2 + 1],   # present.i.decoder.key
        decoder_outputs[i*2 + 2],   # present.i.decoder.value
        decoder_outputs[i*2 + 13],  # present.i.encoder.key  ← SUSPICIOUS
        decoder_outputs[i*2 + 14]   # present.i.encoder.value ← SUSPICIOUS
    ))
```

**Test Strategy**:
```python
# Print ALL decoder output shapes
print(f"Total decoder outputs: {len(decoder_outputs)}")
for i, output in enumerate(decoder_outputs):
    print(f"[{i}] {output.shape}")

# Expected structure for Whisper Base (6 layers):
# [0]: logits (1, seq_len, vocab_size)
# [1-12]: decoder KV (6 layers × 2 = 12 tensors)
# [13-24]: encoder KV (6 layers × 2 = 12 tensors)
# Total: 25 outputs

# Verify indices:
# Layer 0: decoder_k=[1], decoder_v=[2], encoder_k=[13], encoder_v=[14]
# Layer 1: decoder_k=[3], decoder_v=[4], encoder_k=[15], encoder_v=[16]
# Layer 2: decoder_k=[5], decoder_v=[6], encoder_k=[17], encoder_v=[18]
# ...
```

**Correct Formula** (if current is wrong):
```python
for i in range(6):
    past_key_values.append((
        decoder_outputs[1 + i*2],      # decoder.key
        decoder_outputs[1 + i*2 + 1],  # decoder.value
        decoder_outputs[13 + i*2],     # encoder.key
        decoder_outputs[13 + i*2 + 1]  # encoder.value
    ))
```

**Deliverable**: Corrected KV cache extraction code

---

### Day 3: Implement Fixes (8-10 hours)

#### Task 3.1: Fix Encoder-Decoder Connection (2-3 hours)

**Location**: `onnx_whisper_npu.py`, lines 270-273, 456-459

**Change**:
```python
# BEFORE
decoder_outputs = self.decoder_session.run(None, {
    'input_ids': decoder_input_ids,
    'encoder_hidden_states': hidden_states  # May be wrong name
})

# AFTER (use correct name from ONNX model)
decoder_outputs = self.decoder_session.run(None, {
    'input_ids': decoder_input_ids,
    'encoder_hidden_states': hidden_states  # Or whatever Test 2.1 found
})
```

**Validation**:
- Run decoder once
- Check if logits look more reasonable
- Verify no errors about missing inputs

**Deliverable**: Fixed encoder-decoder connection

---

#### Task 3.2: Fix KV Cache Indices (2-3 hours)

**Location**: `onnx_whisper_npu.py`, lines 277-285, 466-474

**Change**:
```python
# BEFORE (potentially wrong)
if use_past and len(decoder_outputs) == 25:
    past_key_values = []
    for i in range(6):
        past_key_values.append((
            decoder_outputs[i*2 + 1],
            decoder_outputs[i*2 + 2],
            decoder_outputs[i*2 + 13],  # Wrong!
            decoder_outputs[i*2 + 14]   # Wrong!
        ))

# AFTER (corrected based on Task 2.2 findings)
if use_past and len(decoder_outputs) == 25:
    past_key_values = []
    for i in range(6):
        # Decoder KV: indices 1-12 (6 layers × 2)
        # Encoder KV: indices 13-24 (6 layers × 2)
        past_key_values.append((
            decoder_outputs[1 + i*2],      # decoder.key
            decoder_outputs[2 + i*2],      # decoder.value
            decoder_outputs[13 + i*2],     # encoder.key
            decoder_outputs[14 + i*2]      # encoder.value
        ))
```

**Validation**:
- Print KV shapes from extraction
- Verify shapes match expected dimensions
- Run generation for 10 steps

**Deliverable**: Fixed KV cache extraction

---

#### Task 3.3: Add Extensive Logging (1-2 hours)

**Goal**: Make debugging easier for future issues

**Add to Decoder Loop** (lines 239-298):
```python
# At start of loop
if step % 10 == 0:  # Log every 10 steps
    logger.info(f"Step {step}: Generated {len(generated_tokens)} tokens so far")

# After token generation
if step < 5:  # Detailed logging for first 5 tokens
    logger.info(f"  Token {step}: {next_token_id} = '{tokenizer.decode([next_token_id])}'")

# After decoding
logger.info(f"Final text ({len(text_tokens)} tokens): '{text[:100]}...'")
```

**Deliverable**: Enhanced logging for debugging

---

### Day 4: Validation & Testing (8-10 hours)

#### Task 4.1: Test with Simple Audio (3-4 hours)

**Create Test Cases**:

**Test 1: Silence**
```python
# 5 seconds of silence
audio = np.zeros(16000 * 5, dtype=np.float32)
```
**Expected**: "[Audio processed but no speech detected]" or similar

**Test 2: Sine Wave**
```python
# 440 Hz tone (should be unintelligible)
t = np.linspace(0, 5, 16000 * 5)
audio = 0.5 * np.sin(2 * np.pi * 440 * t)
```
**Expected**: Gibberish or noise transcription

**Test 3: Real Speech** (if available)
```bash
# Use sample audio from test data
# Or record simple "Hello world" via microphone
```
**Expected**: Actual transcription of speech

**Success Criteria**:
- Decoder produces text (not placeholders)
- Text changes based on audio input
- No crashes or errors

**Deliverable**: Test results for 3+ audio samples

---

#### Task 4.2: Measure Word Error Rate (2-3 hours)

**If Real Speech Available**:

**Setup**:
```python
import jiwer

# Reference text (known correct transcription)
reference = "hello world this is a test"

# Hypothesis (decoder output)
hypothesis = result['text'].lower().strip()

# Calculate WER
wer = jiwer.wer(reference, hypothesis)
print(f"WER: {wer*100:.1f}%")
```

**Targets**:
- WER < 50%: Minimum success
- WER < 20%: Good success
- WER < 10%: Excellent success

**If No Real Speech**:
- Use existing test audio from WhisperX test suite
- Or download LibriSpeech samples

**Deliverable**: WER measurements for 5+ test files

---

#### Task 4.3: Test with Various Audio Lengths (2-3 hours)

**Test Cases**:
1. 5 seconds (short)
2. 15 seconds (medium)
3. 30 seconds (chunk boundary)
4. 45 seconds (multi-chunk)
5. 2 minutes (long)

**Check For**:
- Does chunked processing work? (>30s audio)
- Any crashes on long audio?
- Performance degradation?
- Accuracy consistent across lengths?

**Deliverable**: Length robustness test report

---

### Day 5: Documentation & Cleanup (6-8 hours)

#### Task 5.1: Create DECODER_FIX_COMPLETE.md (2-3 hours)

**Content**:
- Summary of problem
- Root causes identified
- Fixes implemented
- Before/after comparison
- Remaining issues (if any)

**Deliverable**: `DECODER_FIX_COMPLETE.md`

---

#### Task 5.2: Update Test Suite (2-3 hours)

**Create**:
- `test_decoder_accuracy.py` - WER testing
- `test_decoder_robustness.py` - Various audio lengths
- `test_decoder_edge_cases.py` - Error handling

**Deliverable**: Comprehensive decoder test suite

---

#### Task 5.3: Update Main Code with Fixes (2 hours)

**Apply Fixes to Both Decoder Paths**:
1. Chunked audio decoder (lines 183-354)
2. Short audio decoder (lines 356-530)

**Ensure Consistency**:
- Both use same KV cache extraction
- Both use same encoder connection
- Both have same logging

**Deliverable**: Updated `onnx_whisper_npu.py` with fixes

---

## Week 2: Implement KV Cache Optimization

### Day 6: Design KV Cache (8-10 hours)

#### Task 6.1: Understand KV Cache Theory (2 hours)

**Why KV Cache Matters**:

Without cache:
```
Step 1: Compute K,V for token 1
Step 2: Compute K,V for tokens 1,2 (recomputes token 1!)
Step 3: Compute K,V for tokens 1,2,3 (recomputes 1 and 2!)
...
Step 250: Compute K,V for all 250 tokens (recomputes 249!)

Total: 1 + 2 + 3 + ... + 250 = 31,375 K,V computations!
```

With cache:
```
Step 1: Compute K,V for token 1, cache it
Step 2: Compute K,V for token 2 only, use cached 1
Step 3: Compute K,V for token 3 only, use cached 1,2
...
Step 250: Compute K,V for token 250 only, use cached 1-249

Total: 250 K,V computations (125x fewer!)
```

**Performance Impact**:
- O(n²) without cache → O(n) with cache
- 250 tokens: 125x speedup potential
- Critical for real-time transcription

**Deliverable**: KV cache theory document

---

#### Task 6.2: Design Cache Structure (3-4 hours)

**Cache Requirements**:

1. **Storage**: Pre-allocated buffers for K,V matrices
2. **Capacity**: Support up to 448 tokens (Whisper max)
3. **Layers**: Separate cache for each of 6 decoder layers
4. **Types**: Self-attention (decoder) and cross-attention (encoder)

**Design**:

```python
class KVCache:
    """Key-Value cache for Whisper decoder"""

    def __init__(self, max_length=448, num_layers=6, hidden_dim=512, num_heads=8):
        """
        Args:
            max_length: Maximum sequence length (Whisper: 448)
            num_layers: Number of decoder layers (Base: 6)
            hidden_dim: Hidden dimension (Base: 512)
            num_heads: Number of attention heads (Base: 8)
        """
        self.max_length = max_length
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # 512 / 8 = 64

        # Decoder self-attention cache (grows with each token)
        # Shape: (num_layers, max_length, num_heads, head_dim)
        self.decoder_key_cache = np.zeros(
            (num_layers, max_length, num_heads, self.head_dim),
            dtype=np.float32
        )
        self.decoder_value_cache = np.zeros(
            (num_layers, max_length, num_heads, self.head_dim),
            dtype=np.float32
        )

        # Encoder cross-attention cache (constant, set once)
        # Shape: (num_layers, encoder_length, num_heads, head_dim)
        self.encoder_key_cache = None
        self.encoder_value_cache = None

        # Current cache length (how many tokens cached)
        self.cache_len = 0

    def set_encoder_cache(self, encoder_keys, encoder_values):
        """Set encoder K,V (called once after encoder runs)"""
        self.encoder_key_cache = encoder_keys
        self.encoder_value_cache = encoder_values

    def update_decoder_cache(self, layer_idx, new_key, new_value):
        """Add new decoder K,V for current token"""
        self.decoder_key_cache[layer_idx, self.cache_len] = new_key
        self.decoder_value_cache[layer_idx, self.cache_len] = new_value

    def get_decoder_cache(self, layer_idx):
        """Get all cached decoder K,V for a layer"""
        return (
            self.decoder_key_cache[layer_idx, :self.cache_len+1],
            self.decoder_value_cache[layer_idx, :self.cache_len+1]
        )

    def get_encoder_cache(self, layer_idx):
        """Get encoder K,V for a layer"""
        return (
            self.encoder_key_cache[layer_idx],
            self.encoder_value_cache[layer_idx]
        )

    def increment(self):
        """Move to next token position"""
        self.cache_len += 1
        if self.cache_len >= self.max_length:
            raise ValueError(f"Cache full (max {self.max_length} tokens)")

    def reset(self):
        """Clear cache for new sequence"""
        self.cache_len = 0
        self.decoder_key_cache[:] = 0
        self.decoder_value_cache[:] = 0
        self.encoder_key_cache = None
        self.encoder_value_cache = None
```

**Deliverable**: `kv_cache.py` with cache implementation

---

#### Task 6.3: Estimate Memory Usage (1 hour)

**Calculation**:

```
Decoder Cache (per layer):
- Keys: 448 tokens × 8 heads × 64 dim × 4 bytes = 917 KB
- Values: 448 tokens × 8 heads × 64 dim × 4 bytes = 917 KB
- Total per layer: 1.8 MB
- All 6 layers: 10.8 MB

Encoder Cache (per layer):
- Keys: 1500 frames × 8 heads × 64 dim × 4 bytes = 3.1 MB
- Values: 1500 frames × 8 heads × 64 dim × 4 bytes = 3.1 MB
- Total per layer: 6.2 MB
- All 6 layers: 37 MB

Grand Total: ~48 MB per audio file
```

**Acceptable?**: YES (48 MB is small compared to model size ~200 MB)

**Deliverable**: Memory usage analysis

---

#### Task 6.4: Plan Integration with ONNX (2-3 hours)

**Challenge**: ONNX Runtime handles KV cache internally

**Current Approach** (decoder_with_past_model.onnx):
- Inputs: past_key_values.{layer}.{type}.{kv}
- Outputs: present_key_values.{layer}.{type}.{kv}

**Our Cache Integration**:
```
Option A: Let ONNX manage cache (current approach)
  ✅ Simple: Just pass previous outputs as inputs
  ⚠️ Limited control over cache structure
  ⚠️ Cannot pre-compute encoder K,V separately

Option B: Manage cache in Python (our KVCache class)
  ✅ Full control over cache
  ✅ Can pre-compute encoder K,V
  ⚠️ Need to extract/inject K,V from ONNX outputs/inputs
  ⚠️ More complex code

Recommendation: Start with Option A (simpler)
              Migrate to Option B if needed for performance
```

**Deliverable**: Cache integration strategy

---

### Day 7-8: Implement KV Cache Integration (12-16 hours)

#### Task 7.1: Pre-compute Encoder K,V (4-6 hours)

**Goal**: Compute encoder K,V once after encoder, reuse for all decoder steps

**Current Behavior** (inefficient):
```python
# Every decoder step:
decoder_outputs = decoder_session.run(None, {
    'input_ids': decoder_input_ids,
    'encoder_hidden_states': hidden_states  # Recomputes encoder K,V!
})
# Encoder K,V extracted from outputs[13-24]
```

**Optimized Behavior**:
```python
# After encoder (one time):
first_decoder_outputs = decoder_session.run(None, {
    'input_ids': start_tokens,
    'encoder_hidden_states': hidden_states
})

# Extract encoder K,V (constant for all steps)
encoder_kv_cache = []
for i in range(6):
    encoder_kv_cache.append((
        first_decoder_outputs[13 + i*2],      # encoder.key
        first_decoder_outputs[13 + i*2 + 1]   # encoder.value
    ))

# In generation loop (reuse encoder K,V):
for step in range(max_steps):
    inputs = {'input_ids': decoder_input_ids[:, -1:]}

    for i, (dec_k, dec_v) in enumerate(decoder_kv_cache):
        inputs[f'past_key_values.{i}.decoder.key'] = dec_k
        inputs[f'past_key_values.{i}.decoder.value'] = dec_v
        # Reuse pre-computed encoder K,V
        inputs[f'past_key_values.{i}.encoder.key'] = encoder_kv_cache[i][0]
        inputs[f'past_key_values.{i}.encoder.value'] = encoder_kv_cache[i][1]

    decoder_outputs = decoder_with_past_session.run(None, inputs)
    # ... generate next token ...
```

**Performance Impact**:
- Encoder K,V computation: 1 time instead of 250 times
- Expected speedup: ~30% (encoder cross-attention is ~30% of decoder compute)

**Deliverable**: Pre-computed encoder K,V implementation

---

#### Task 7.2: Optimize Decoder K,V Updates (4-6 hours)

**Goal**: Efficiently update decoder K,V cache each step

**Current**:
```python
# Extract new decoder K,V from outputs
for i in range(6):
    new_decoder_kv.append((
        decoder_outputs[i*2 + 1],
        decoder_outputs[i*2 + 2]
    ))
```

**Optimized**:
```python
# Update our KVCache class
for i in range(6):
    new_k = decoder_outputs[i*2 + 1]
    new_v = decoder_outputs[i*2 + 2]
    kv_cache.update_decoder_cache(i, new_k, new_v)

kv_cache.increment()  # Move to next position

# Get cache for next iteration
for i in range(6):
    cached_k, cached_v = kv_cache.get_decoder_cache(i)
    inputs[f'past_key_values.{i}.decoder.key'] = cached_k
    inputs[f'past_key_values.{i}.decoder.value'] = cached_v
```

**Deliverable**: Efficient decoder K,V cache updates

---

#### Task 7.3: Integrate KVCache Class (4-6 hours)

**Modify** `onnx_whisper_npu.py`:

```python
# Import at top
from kv_cache import KVCache

# In transcribe_audio(), after encoder:
kv_cache = KVCache(
    max_length=448,
    num_layers=6,
    hidden_dim=512,
    num_heads=8
)

# Pre-compute encoder K,V
first_decoder_outputs = self.decoder_session.run(None, {
    'input_ids': decoder_input_ids,
    'encoder_hidden_states': hidden_states
})

encoder_kv = []
for i in range(6):
    encoder_kv.append((
        first_decoder_outputs[13 + i*2],
        first_decoder_outputs[13 + i*2 + 1]
    ))

kv_cache.set_encoder_cache(encoder_kv)

# Get first token
logits = first_decoder_outputs[0]
next_token_id = np.argmax(logits[0, -1, :])
# ... handle first token ...

# Generation loop
for step in range(1, 448):  # Start from 1 (already did step 0)
    # Build inputs with cache
    inputs = {'input_ids': decoder_input_ids[:, -1:]}

    for i in range(6):
        dec_k, dec_v = kv_cache.get_decoder_cache(i)
        enc_k, enc_v = kv_cache.get_encoder_cache(i)

        inputs[f'past_key_values.{i}.decoder.key'] = dec_k
        inputs[f'past_key_values.{i}.decoder.value'] = dec_v
        inputs[f'past_key_values.{i}.encoder.key'] = enc_k
        inputs[f'past_key_values.{i}.encoder.value'] = enc_v

    # Run decoder
    decoder_outputs = self.decoder_with_past_session.run(None, inputs)

    # Update cache
    for i in range(6):
        new_k = decoder_outputs[i*2 + 1]
        new_v = decoder_outputs[i*2 + 2]
        kv_cache.update_decoder_cache(i, new_k, new_v)

    kv_cache.increment()

    # ... rest of generation ...
```

**Deliverable**: Full KV cache integration

---

### Day 9: Testing & Validation (8-10 hours)

#### Task 9.1: Verify Cache Correctness (4-5 hours)

**Test**: Compare output with/without cache

```python
# Run WITHOUT cache
result_no_cache = transcribe_audio(test_file, use_kv_cache=False)

# Run WITH cache
result_with_cache = transcribe_audio(test_file, use_kv_cache=True)

# Compare
print(f"No cache: '{result_no_cache['text']}'")
print(f"With cache: '{result_with_cache['text']}'")

# Calculate similarity
from difflib import SequenceMatcher
similarity = SequenceMatcher(None, result_no_cache['text'], result_with_cache['text']).ratio()
print(f"Text similarity: {similarity*100:.1f}%")

# Should be >99% similar (minor differences OK due to numerical precision)
```

**Success Criteria**:
- Text similarity >99%
- WER difference <1%
- No crashes or errors

**Deliverable**: Cache correctness validation

---

#### Task 9.2: Measure Performance Improvement (4-5 hours)

**Benchmark Code**:

```python
import time

test_files = [
    "test_5s.wav",
    "test_30s.wav",
    "test_60s.wav"
]

for test_file in test_files:
    # Benchmark without cache
    start = time.time()
    result_no_cache = transcribe_audio(test_file, use_kv_cache=False)
    time_no_cache = time.time() - start

    # Benchmark with cache
    start = time.time()
    result_with_cache = transcribe_audio(test_file, use_kv_cache=True)
    time_with_cache = time.time() - start

    # Calculate speedup
    speedup = time_no_cache / time_with_cache

    print(f"\n{test_file}:")
    print(f"  No cache: {time_no_cache:.2f}s")
    print(f"  With cache: {time_with_cache:.2f}s")
    print(f"  Speedup: {speedup:.1f}x")
```

**Expected Speedups**:
- Short audio (5s): 10-15x
- Medium audio (30s): 20-25x
- Long audio (60s): 25-30x

**Deliverable**: Performance benchmarks

---

### Day 10: Documentation & Finalization (6-8 hours)

#### Task 10.1: Create KV_CACHE_IMPLEMENTATION.md (3-4 hours)

**Content**:
- KV cache theory explanation
- Design decisions
- Implementation details
- Performance results
- Code examples
- Future optimizations

**Deliverable**: Comprehensive KV cache documentation

---

#### Task 10.2: Create Phase 1 Summary Report (2-3 hours)

**Content**:
- Executive summary
- Problems fixed
- Performance achieved
- Code changes made
- Test results
- Next steps (Phase 2)

**Deliverable**: `PHASE1_COMPLETE.md`

---

#### Task 10.3: Update Phase 1 Test Suite (1-2 hours)

**Create**:
- `test_kv_cache.py` - Cache unit tests
- `test_decoder_with_cache.py` - Integration tests
- `benchmark_phase1.py` - Performance tests

**Deliverable**: Complete Phase 1 test suite

---

## Deliverables Summary

### Week 1 Deliverables:
1. ✅ `decoder_diagnostic_results.txt` - Test suite results
2. ✅ `DECODER_FIX_LOG.md` - Investigation findings
3. ✅ `DECODER_FIX_COMPLETE.md` - Fix summary
4. ✅ Updated `onnx_whisper_npu.py` - Fixed decoder
5. ✅ `test_decoder_accuracy.py` - Accuracy tests
6. ✅ `test_decoder_robustness.py` - Robustness tests

### Week 2 Deliverables:
7. ✅ `kv_cache.py` - KV cache implementation
8. ✅ `KV_CACHE_IMPLEMENTATION.md` - Documentation
9. ✅ Updated `onnx_whisper_npu.py` - With KV cache
10. ✅ `test_kv_cache.py` - Cache tests
11. ✅ `benchmark_phase1.py` - Performance benchmarks
12. ✅ `PHASE1_COMPLETE.md` - Phase 1 summary

---

## Success Criteria

### Minimum Success (Must Achieve):
- ✅ Decoder produces readable text (not garbled)
- ✅ WER <50% on test audio
- ✅ KV cache implemented (even if basic)

### Good Success (Target):
- ✅ WER <20% on test audio
- ✅ 20x speedup from KV cache
- ✅ 20-30x realtime full pipeline

### Excellent Success (Stretch):
- ✅ WER <10% (very accurate)
- ✅ 25x speedup from KV cache
- ✅ 30x realtime full pipeline
- ✅ Encoder K,V pre-computation implemented

---

## Risk Mitigation

### Risk 1: Decoder Fix Takes Longer Than Expected
**Mitigation**: Allocate extra 2-3 days buffer
**Fallback**: Use CPU decoder temporarily, focus on KV cache

### Risk 2: KV Cache Doesn't Improve Performance
**Mitigation**: Benchmark at each step, identify bottlenecks
**Fallback**: Accept lower speedup, move to Phase 2

### Risk 3: Accuracy Degrades with Cache
**Mitigation**: Extensive correctness testing
**Fallback**: Use cache only for long audio (>60s)

---

## Phase 1 Timeline

```
Week 1: Fix Decoder                  Week 2: KV Cache
├─ Day 1: Investigation (8-10h)     ├─ Day 6: Design (8-10h)
├─ Day 2: Root cause (6-8h)         ├─ Day 7-8: Implementation (12-16h)
├─ Day 3: Implement fixes (8-10h)   ├─ Day 9: Testing (8-10h)
├─ Day 4: Validation (8-10h)        └─ Day 10: Documentation (6-8h)
└─ Day 5: Documentation (6-8h)

Total: 80-96 person-hours (2 weeks full-time)
```

---

## Next Steps After Phase 1

**Phase 2 Preview** (Weeks 3-4):
- Implement batch processing (multiple audio files)
- Optimize DMA transfers (reduce overhead)
- Add beam search (improve accuracy)
- Target: 50x realtime

**Phase 3 Preview** (Weeks 5-6):
- Compile NPU mel kernels (mel preprocessing on NPU)
- Target: 80x realtime

**Phase 4 Preview** (Weeks 7-10):
- Full NPU encoder/decoder (custom kernels)
- Target: 220x realtime 🎯

---

**Phase 1 Start Date**: November 2, 2025
**Phase 1 Target Completion**: November 16, 2025 (2 weeks)
**Phase 1 Lead**: Implementation Lead for NPU Decoder

**Let's fix this decoder!** 🚀

**Magic Unicorn Unconventional Technology & Stuff Inc.** 🦄✨
