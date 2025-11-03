# NPU Decoder Phase 1 - Garbled Output Investigation

**Date**: November 2, 2025
**Lead**: Implementation Lead for NPU Decoder Phase 1
**Mission**: Fix garbled decoder output and implement KV cache for 220x performance

---

## Executive Summary

**Problem**: Decoder in `onnx_whisper_npu.py` produces garbled or placeholder text instead of accurate transcription.

**Current Behavior**:
```python
result = {'text': '... ... ... ...', 'segments': []}
# Should be: {'text': 'The quick brown fox...', 'segments': [...]}
```

**Status**: Investigation in progress

---

## File Under Investigation

**Location**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/onnx_whisper_npu.py`

**Total Lines**: 625
**Decoder Implementation**: Lines 220-530

---

## Initial Code Analysis

### Decoder Architecture (Lines 220-530)

The decoder has TWO separate code paths:

#### Path 1: Chunked Audio (Lines 183-354)
- Used for audio longer than 30 seconds
- Processes in 30-second chunks
- Has a decoder loop (lines 239-298)

#### Path 2: Short Audio (Lines 356-530)
- Used for audio shorter than 30 seconds
- Also has a decoder loop (lines 424-491)

**Both paths share similar decoder logic - need to investigate BOTH**

### Key Decoder Components Found

#### 1. Tokenizer Initialization (Lines 228, 408)
```python
from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
```
✅ **Status**: Correctly initialized

#### 2. Start Tokens (Lines 232, 417)
```python
decoder_input_ids = np.array([[50258, 50259, 50360, 50365]], dtype=np.int64)
# 50258 = <|startoftranscript|>
# 50259 = <|en|> (English)
# 50360 = <|transcribe|>
# 50365 = <|notimestamps|>
```
✅ **Status**: Correct Whisper start token sequence

#### 3. KV Cache Detection (Lines 235, 420)
```python
use_past = self.decoder_with_past_session is not None
past_key_values = None
```
✅ **Status**: KV cache support exists (decoder_with_past_model.onnx)
⚠️ **Issue**: May not be loading correctly or may have bugs

#### 4. Generation Loop (Lines 239-298 and 424-491)
```python
for step in range(448):  # Full Whisper capacity
    # Check maximum length
    if decoder_input_ids.shape[1] >= 448:
        logger.warning(f"⚠️ Reached maximum sequence length (448 tokens), stopping generation")
        break
```
✅ **Status**: Loop structure looks correct
⚠️ **Issue**: May be stopping too early or generating wrong tokens

#### 5. Encoder Hidden States (Lines 224, 400)
```python
hidden_states = encoder_outputs[0]
logger.info(f"✅ Encoder output: {hidden_states.shape}")
```
✅ **Status**: Encoder outputs are extracted
❓ **Question**: Are they being passed correctly to decoder?

---

## Investigation Steps

### Step 1: Check Encoder Outputs ✅ (Verified)

**Lines 220-224**:
```python
encoder_outputs = self.encoder_session.run(None, {
    'input_features': input_features
})
hidden_states = encoder_outputs[0]
logger.info(f"✅ Encoder output: {hidden_states.shape}")
```

**Expected Shape**: `(batch_size, 1500, hidden_dim)` for 30s audio
- batch_size = 1
- 1500 time steps (3000 mel frames / 2)
- hidden_dim = 512 for base model

**Status**: Need to verify actual shape with test run

---

### Step 2: Analyze Decoder Loop (CRITICAL)

**Lines 245-286** (with KV cache):
```python
if use_past and past_key_values is not None:
    # Efficient decoding with KV cache
    inputs = {'input_ids': decoder_input_ids[:, -1:]}

    for i, kv in enumerate(past_key_values):
        inputs[f'past_key_values.{i}.decoder.key'] = kv[0]
        inputs[f'past_key_values.{i}.decoder.value'] = kv[1]
        inputs[f'past_key_values.{i}.encoder.key'] = kv[2]
        inputs[f'past_key_values.{i}.encoder.value'] = kv[3]

    decoder_outputs = self.decoder_with_past_session.run(None, inputs)
```

**Potential Issues**:
1. ❓ **KV cache may not be initialized correctly** on first pass
2. ❓ **Past key values extraction** (lines 277-285) may have wrong indices
3. ❓ **Encoder K/V** may not be computed correctly

**Lines 268-275** (without KV cache - first pass):
```python
decoder_outputs = self.decoder_session.run(None, {
    'input_ids': decoder_input_ids,
    'encoder_hidden_states': hidden_states
})
logits = decoder_outputs[0]
```

**Potential Issues**:
1. ❓ **encoder_hidden_states** - is this the right input name?
2. ❓ **decoder_outputs** - are we getting the right outputs?

---

### Step 3: Check Token Generation (CRITICAL)

**Lines 287-296**:
```python
logits = decoder_outputs[0]
next_token_id = np.argmax(logits[0, -1, :])

# Check for end token
if next_token_id == 50257:  # <|endoftext|>
    break

# Add to sequence
decoder_input_ids = np.concatenate([
    decoder_input_ids,
    np.array([[next_token_id]], dtype=np.int64)
], axis=1)

generated_tokens.append(next_token_id)
```

**Potential Issues**:
1. ✅ **Token extraction looks correct** (argmax on last position)
2. ✅ **EOS detection correct** (50257 is right)
3. ⚠️ **May be generating only special tokens** (not real text)

---

### Step 4: Check Token Decoding (Lines 300-305)

```python
if generated_tokens:
    # Skip special tokens and decode
    text_tokens = [t for t in generated_tokens if t < 50257]
    if text_tokens:
        text = tokenizer.decode(text_tokens, skip_special_tokens=True)
        if not text.strip():
            text = "[Audio processed but no speech detected]"
    else:
        text = "[Audio processed but no text tokens generated]"
else:
    text = "[Audio processed but no tokens generated]"
```

**Potential Issues**:
1. ✅ **Special token filtering correct** (t < 50257)
2. ⚠️ **May not be generating any text tokens**
3. ⚠️ **Placeholder messages may be masking real issue**

---

## Suspected Root Causes (Priority Order)

### 1. 🔴 CRITICAL: Encoder Hidden States Not Connected Properly

**Hypothesis**: Decoder is not receiving encoder cross-attention properly

**Evidence Needed**:
- Print encoder hidden states shape
- Print decoder input names (ONNX session expects specific names)
- Check if `encoder_hidden_states` is the correct input name

**Test**:
```python
# Check ONNX decoder input names
print("Decoder inputs:", [input.name for input in self.decoder_session.get_inputs()])
print("Encoder outputs:", [output.name for output in self.encoder_session.get_outputs()])
```

**Expected Inputs**:
- `input_ids`: Decoder input tokens
- `encoder_hidden_states`: Encoder outputs for cross-attention

**Fix**: Use correct input names from ONNX model metadata

---

### 2. 🔴 CRITICAL: KV Cache Extraction Wrong

**Hypothesis**: Past key values are extracted with wrong indices (lines 277-285, 466-474)

**Current Code**:
```python
if use_past and len(decoder_outputs) == 25:
    past_key_values = []
    for i in range(6):  # 6 decoder layers
        past_key_values.append((
            decoder_outputs[i*2 + 1],   # present.i.decoder.key
            decoder_outputs[i*2 + 2],   # present.i.decoder.value
            decoder_outputs[i*2 + 13],  # present.i.encoder.key
            decoder_outputs[i*2 + 14]   # present.i.encoder.value
        ))
```

**Issues**:
- ❓ **Index calculations may be off** (i*2 + 13, i*2 + 14)
- ❓ **Expecting 25 outputs** (1 logits + 24 KV tensors)
- ❓ **May need different indices** for encoder vs decoder KV

**Test**:
```python
print(f"Decoder outputs count: {len(decoder_outputs)}")
for i, output in enumerate(decoder_outputs):
    print(f"Output {i}: shape {output.shape}")
```

**Fix**: Verify output indices match ONNX model structure

---

### 3. 🟡 HIGH: Encoder K/V Not Pre-computed

**Hypothesis**: Encoder K/V should be computed ONCE and reused, not recomputed every step

**Current Behavior**: Encoder K/V extracted from decoder outputs every step (inefficient)

**Optimal Behavior**:
- Compute encoder K/V ONCE after encoder
- Reuse same encoder K/V for all 250 decoder steps
- Only update decoder K/V each step

**Performance Impact**:
- Current: 250 steps × recompute encoder K/V = wasteful
- Optimal: 1 × compute encoder K/V + 250 × update decoder K/V = 250x faster

**Fix**: Pre-compute encoder K/V and keep them constant

---

### 4. 🟡 MEDIUM: Limited Token Generation

**Hypothesis**: Loop may be stopping too early or generating limited output

**Evidence Needed**:
- Count how many tokens are actually generated
- Check if loop exits early
- Verify EOS detection is working

**Test**:
```python
print(f"Generated {len(generated_tokens)} tokens")
print(f"Token IDs: {generated_tokens[:20]}")  # First 20 tokens
```

**Expected**: 50-200 tokens for meaningful transcription
**If seeing**: Only 5-20 tokens, may be stopping too early

---

### 5. 🟡 MEDIUM: Tokenizer Decoding Issue

**Hypothesis**: Tokens are being generated but not decoded correctly

**Evidence Needed**:
- Print raw token IDs before decoding
- Verify token IDs are in valid vocabulary range
- Check if tokens are all special tokens (>= 50257)

**Test**:
```python
print(f"Text tokens: {text_tokens}")
print(f"All tokens: {generated_tokens}")
print(f"Special token count: {len([t for t in generated_tokens if t >= 50257])}")
```

---

## Testing Strategy

### Test 1: Minimal Decoder Test (2 hours)

**Goal**: Verify decoder can generate ANY tokens

**Steps**:
1. Create simple 5-second "Hello world" audio
2. Run through encoder
3. Add extensive logging to decoder loop
4. Check token generation

**Expected Output**:
```
Encoder shape: (1, 250, 512)
Generated tokens: [50258, 50259, 50360, 50365, 15947, 1002, ...]
Text: "Hello world"
```

**If seeing**:
- Only special tokens → decoder not working
- Zeros or random tokens → encoder/decoder connection broken

---

### Test 2: Verify ONNX Model Inputs/Outputs (1 hour)

**Goal**: Ensure we're using correct input/output names

**Code**:
```python
# Inspect decoder model
print("\n=== DECODER MODEL ===")
print("Inputs:")
for inp in self.decoder_session.get_inputs():
    print(f"  {inp.name}: {inp.shape}")

print("Outputs:")
for out in self.decoder_session.get_outputs():
    print(f"  {out.name}: {out.shape}")

# Same for decoder_with_past
print("\n=== DECODER WITH PAST MODEL ===")
if self.decoder_with_past_session:
    print("Inputs:")
    for inp in self.decoder_with_past_session.get_inputs():
        print(f"  {inp.name}: {inp.shape}")
    print("Outputs:")
    for out in self.decoder_with_past_session.get_outputs():
        print(f"  {out.name}: {out.shape}")
```

---

### Test 3: Validate Encoder Outputs (30 min)

**Goal**: Verify encoder produces valid hidden states

**Code**:
```python
hidden_states = encoder_outputs[0]
print(f"Encoder hidden states: shape={hidden_states.shape}, dtype={hidden_states.dtype}")
print(f"Min: {hidden_states.min():.4f}, Max: {hidden_states.max():.4f}, Mean: {hidden_states.mean():.4f}")
print(f"Has NaN: {np.isnan(hidden_states).any()}, Has Inf: {np.isinf(hidden_states).any()}")
```

**Expected**:
- Shape: (1, 1500, 512) for 30s audio
- Values: Reasonable range (-10 to +10)
- No NaN or Inf

---

### Test 4: Step-by-Step Decoder Debug (4 hours)

**Goal**: Debug every decoder step to find where it breaks

**Code**:
```python
for step in range(448):
    print(f"\n=== Step {step} ===")
    print(f"decoder_input_ids shape: {decoder_input_ids.shape}")
    print(f"decoder_input_ids: {decoder_input_ids[0, -5:]}")  # Last 5 tokens

    if use_past and past_key_values:
        print(f"Using cached KV (decoder KV shape: {past_key_values[0][0].shape})")

    # Run decoder
    decoder_outputs = ...

    print(f"Logits shape: {logits.shape}")
    print(f"Logits min/max: {logits.min():.4f} / {logits.max():.4f}")

    next_token_id = np.argmax(logits[0, -1, :])
    print(f"Next token: {next_token_id}")

    # Decode single token to see what it is
    try:
        token_text = tokenizer.decode([next_token_id])
        print(f"Token text: '{token_text}'")
    except:
        print(f"Could not decode token {next_token_id}")

    if next_token_id == 50257:
        print("EOS reached")
        break

    generated_tokens.append(next_token_id)

    # Stop after 10 steps for debugging
    if step >= 10:
        break
```

---

## Expected Fixes

### Fix 1: Correct Encoder Hidden States Connection

```python
# BEFORE (potentially wrong)
decoder_outputs = self.decoder_session.run(None, {
    'input_ids': decoder_input_ids,
    'encoder_hidden_states': hidden_states  # May be wrong name
})

# AFTER (with correct input name)
# First check actual input names from ONNX model
decoder_inputs = {inp.name: None for inp in self.decoder_session.get_inputs()}
print(f"Decoder expects: {list(decoder_inputs.keys())}")

# Then use correct names
decoder_outputs = self.decoder_session.run(None, {
    'input_ids': decoder_input_ids,
    'encoder_outputs': hidden_states  # Or whatever the actual name is
})
```

---

### Fix 2: Correct KV Cache Indices

```python
# BEFORE (potentially wrong indices)
for i in range(6):
    past_key_values.append((
        decoder_outputs[i*2 + 1],   # May be wrong
        decoder_outputs[i*2 + 2],
        decoder_outputs[i*2 + 13],  # Definitely suspicious
        decoder_outputs[i*2 + 14]
    ))

# AFTER (with verified indices from ONNX output inspection)
# Expected output structure:
# 0: logits
# 1-12: present.0-5.decoder.{key,value} (6 layers × 2 = 12 outputs)
# 13-24: present.0-5.encoder.{key,value} (6 layers × 2 = 12 outputs)

for i in range(6):  # 6 decoder layers
    past_key_values.append((
        decoder_outputs[1 + i*2],      # present.i.decoder.key
        decoder_outputs[1 + i*2 + 1],  # present.i.decoder.value
        decoder_outputs[13 + i*2],     # present.i.encoder.key
        decoder_outputs[13 + i*2 + 1]  # present.i.encoder.value
    ))
```

---

### Fix 3: Pre-compute Encoder K/V

```python
# NEW: Pre-compute encoder K/V after encoder runs
encoder_kv_cache = None

# After encoder runs (line 224)
hidden_states = encoder_outputs[0]

# Extract encoder K/V on first decoder pass
if use_past:
    # Run decoder once to get encoder K/V
    initial_decoder_outputs = self.decoder_session.run(None, {
        'input_ids': decoder_input_ids,
        'encoder_hidden_states': hidden_states
    })

    # Extract encoder K/V (these stay constant!)
    encoder_kv_cache = []
    for i in range(6):
        encoder_kv_cache.append((
            initial_decoder_outputs[13 + i*2],      # encoder.key
            initial_decoder_outputs[13 + i*2 + 1]   # encoder.value
        ))

    # Start with initial logits
    logits = initial_decoder_outputs[0]
    # ... rest of first step ...

# Then in generation loop, reuse encoder_kv_cache
for step in range(448):
    if use_past and past_key_values:
        inputs = {'input_ids': decoder_input_ids[:, -1:]}

        for i, (dec_k, dec_v) in enumerate(past_key_values):
            inputs[f'past_key_values.{i}.decoder.key'] = dec_k
            inputs[f'past_key_values.{i}.decoder.value'] = dec_v
            # Use pre-computed encoder K/V
            inputs[f'past_key_values.{i}.encoder.key'] = encoder_kv_cache[i][0]
            inputs[f'past_key_values.{i}.encoder.value'] = encoder_kv_cache[i][1]
```

---

## Success Criteria for Phase 1

### Minimum Success (Must Achieve):
- ✅ Decoder produces readable text (not garbled)
- ✅ WER < 50% on simple test audio
- ✅ Can transcribe "Hello world" correctly

### Good Success (Target):
- ✅ WER < 20% on 30s test audio
- ✅ Text matches audio content accurately
- ✅ No more placeholder "..." output
- ✅ KV cache basic implementation working

### Excellent Success (Stretch):
- ✅ WER < 10% (very accurate)
- ✅ KV cache providing measurable speedup
- ✅ Can transcribe 2-minute audio without issues

---

## Timeline

**Day 1-2** (16 hours): Investigation and Testing
- Run all 4 diagnostic tests
- Identify root cause
- Document findings

**Day 3** (8 hours): Implement Fixes
- Fix encoder-decoder connection
- Fix KV cache indices
- Pre-compute encoder K/V

**Day 4-5** (16 hours): Validation and Testing
- Test with simple audio
- Test with 30s audio
- Measure WER
- Create test suite

**Total**: 40 hours (1 week full-time)

---

## Next Actions (Immediate)

1. ✅ Read this investigation log
2. ⏭️ Run Test 2 (ONNX model inspection)
3. ⏭️ Run Test 1 (minimal decoder test)
4. ⏭️ Run Test 3 (encoder validation)
5. ⏭️ Run Test 4 (step-by-step debugging)
6. ⏭️ Implement fixes based on findings
7. ⏭️ Validate fixes with test suite

---

**Investigation Status**: In Progress
**Next Update**: After Test 2 completion
**Implementation Lead**: NPU Decoder Phase 1 Team

**Magic Unicorn Unconventional Technology & Stuff Inc.** 🦄✨
