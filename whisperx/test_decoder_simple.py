#!/usr/bin/env python3
"""
Simple Decoder Test - Diagnose Garbled Output Issue

This script runs a minimal test to diagnose why the decoder is producing
garbled output instead of accurate transcription.

Usage:
    python3 test_decoder_simple.py
"""

import sys
import os
import numpy as np
import tempfile
import logging

# Add path for npu modules
sys.path.insert(0, '/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx')
sys.path.insert(0, '/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_audio(duration=5.0, sample_rate=16000):
    """Create simple test audio (sine wave)"""
    import numpy as np
    t = np.linspace(0, duration, int(sample_rate * duration))
    # 440 Hz sine wave (musical note A)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32), sample_rate

def test_onnx_model_structure():
    """Test 1: Inspect ONNX Model Structure"""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: ONNX Model Structure Inspection")
    logger.info("="*80)

    from npu.npu_optimization.onnx_whisper_npu import ONNXWhisperNPU

    whisper = ONNXWhisperNPU()
    if not whisper.initialize(model_size="base"):
        logger.error("Failed to initialize ONNX Whisper")
        return False

    logger.info("\n--- ENCODER MODEL ---")
    logger.info("Inputs:")
    for inp in whisper.encoder_session.get_inputs():
        logger.info(f"  {inp.name}: shape={inp.shape}, type={inp.type}")

    logger.info("\nOutputs:")
    for out in whisper.encoder_session.get_outputs():
        logger.info(f"  {out.name}: shape={out.shape}, type={out.type}")

    logger.info("\n--- DECODER MODEL ---")
    logger.info("Inputs:")
    for inp in whisper.decoder_session.get_inputs():
        logger.info(f"  {inp.name}: shape={inp.shape}, type={inp.type}")

    logger.info("\nOutputs:")
    for i, out in enumerate(whisper.decoder_session.get_outputs()):
        logger.info(f"  [{i}] {out.name}: shape={out.shape}, type={out.type}")

    if whisper.decoder_with_past_session:
        logger.info("\n--- DECODER WITH PAST MODEL ---")
        logger.info("Inputs:")
        for inp in whisper.decoder_with_past_session.get_inputs():
            logger.info(f"  {inp.name}: shape={inp.shape}, type={inp.type}")

        logger.info("\nOutputs:")
        for i, out in enumerate(whisper.decoder_with_past_session.get_outputs()):
            logger.info(f"  [{i}] {out.name}: shape={out.shape}, type={out.type}")

        logger.info(f"\n✅ Total decoder outputs: {len(whisper.decoder_session.get_outputs())}")
        logger.info(f"✅ Total decoder_with_past outputs: {len(whisper.decoder_with_past_session.get_outputs())}")

    return True

def test_encoder_outputs():
    """Test 2: Verify Encoder Produces Valid Outputs"""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Encoder Output Validation")
    logger.info("="*80)

    from npu.npu_optimization.onnx_whisper_npu import ONNXWhisperNPU

    whisper = ONNXWhisperNPU()
    if not whisper.initialize(model_size="base"):
        logger.error("Failed to initialize ONNX Whisper")
        return False

    # Create test audio
    audio, sr = create_test_audio(duration=5.0)
    logger.info(f"✅ Created test audio: {len(audio)} samples @ {sr}Hz ({len(audio)/sr:.1f}s)")

    # Extract mel features
    mel_features = whisper.extract_mel_features(audio, sr)
    logger.info(f"✅ Mel features extracted: shape={mel_features.shape}")

    # Prepare encoder input
    input_features = np.expand_dims(mel_features, axis=0)

    # Pad to expected length
    target_length = 3000
    if input_features.shape[2] < target_length:
        padding = target_length - input_features.shape[2]
        input_features = np.pad(input_features, ((0, 0), (0, 0), (0, padding)), mode='constant')

    logger.info(f"✅ Encoder input prepared: shape={input_features.shape}")

    # Run encoder
    encoder_outputs = whisper.encoder_session.run(None, {
        'input_features': input_features
    })

    hidden_states = encoder_outputs[0]
    logger.info(f"✅ Encoder hidden states: shape={hidden_states.shape}, dtype={hidden_states.dtype}")
    logger.info(f"   Stats: min={hidden_states.min():.4f}, max={hidden_states.max():.4f}, mean={hidden_states.mean():.4f}, std={hidden_states.std():.4f}")
    logger.info(f"   Has NaN: {np.isnan(hidden_states).any()}, Has Inf: {np.isinf(hidden_states).any()}")

    if np.isnan(hidden_states).any() or np.isinf(hidden_states).any():
        logger.error("❌ Encoder outputs contain NaN or Inf!")
        return False

    logger.info("✅ Encoder outputs look valid")
    return True

def test_decoder_step_by_step():
    """Test 3: Step-by-Step Decoder Debugging"""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Step-by-Step Decoder Debugging")
    logger.info("="*80)

    from npu.npu_optimization.onnx_whisper_npu import ONNXWhisperNPU
    from transformers import WhisperTokenizer

    whisper = ONNXWhisperNPU()
    if not whisper.initialize(model_size="base"):
        logger.error("Failed to initialize ONNX Whisper")
        return False

    # Load tokenizer
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")

    # Create test audio
    audio, sr = create_test_audio(duration=5.0)

    # Extract mel features and run encoder
    mel_features = whisper.extract_mel_features(audio, sr)
    input_features = np.expand_dims(mel_features, axis=0)

    target_length = 3000
    if input_features.shape[2] < target_length:
        padding = target_length - input_features.shape[2]
        input_features = np.pad(input_features, ((0, 0), (0, 0), (0, padding)), mode='constant')

    encoder_outputs = whisper.encoder_session.run(None, {
        'input_features': input_features
    })
    hidden_states = encoder_outputs[0]

    logger.info(f"✅ Encoder done: hidden_states.shape={hidden_states.shape}")

    # Start decoder
    decoder_input_ids = np.array([[50258, 50259, 50360, 50365]], dtype=np.int64)
    logger.info(f"✅ Start tokens: {decoder_input_ids[0]}")
    logger.info(f"   Decoded: {tokenizer.decode(decoder_input_ids[0], skip_special_tokens=False)}")

    use_past = whisper.decoder_with_past_session is not None
    logger.info(f"✅ KV cache available: {use_past}")

    past_key_values = None
    generated_tokens = []

    # Run first 10 steps with extensive logging
    for step in range(10):
        logger.info(f"\n--- STEP {step} ---")
        logger.info(f"decoder_input_ids shape: {decoder_input_ids.shape}")
        logger.info(f"decoder_input_ids (last 5): {decoder_input_ids[0, -5:]}")

        if use_past and past_key_values is not None:
            logger.info(f"Using KV cache (decoder KV[0] shape: {past_key_values[0][0].shape})")

            inputs = {'input_ids': decoder_input_ids[:, -1:]}

            for i, kv in enumerate(past_key_values):
                inputs[f'past_key_values.{i}.decoder.key'] = kv[0]
                inputs[f'past_key_values.{i}.decoder.value'] = kv[1]
                inputs[f'past_key_values.{i}.encoder.key'] = kv[2]
                inputs[f'past_key_values.{i}.encoder.value'] = kv[3]

            decoder_outputs = whisper.decoder_with_past_session.run(None, inputs)
            logits = decoder_outputs[0]

            # Update decoder KV cache
            new_past = []
            for i in range(6):
                new_past.append((
                    decoder_outputs[i*2 + 1],
                    decoder_outputs[i*2 + 2],
                    past_key_values[i][2],
                    past_key_values[i][3]
                ))
            past_key_values = new_past

        else:
            logger.info("First pass - extracting KV cache")

            decoder_outputs = whisper.decoder_session.run(None, {
                'input_ids': decoder_input_ids,
                'encoder_hidden_states': hidden_states
            })
            logits = decoder_outputs[0]

            logger.info(f"Decoder outputs count: {len(decoder_outputs)}")

            # Extract KV cache
            if use_past and len(decoder_outputs) == 25:
                logger.info("Extracting past_key_values from decoder outputs")
                past_key_values = []
                for i in range(6):
                    decoder_k = decoder_outputs[i*2 + 1]
                    decoder_v = decoder_outputs[i*2 + 2]
                    encoder_k = decoder_outputs[i*2 + 13]
                    encoder_v = decoder_outputs[i*2 + 14]

                    logger.info(f"  Layer {i}: dec_k={decoder_k.shape}, dec_v={decoder_v.shape}, enc_k={encoder_k.shape}, enc_v={encoder_v.shape}")

                    past_key_values.append((decoder_k, decoder_v, encoder_k, encoder_v))

        # Get next token
        logger.info(f"Logits shape: {logits.shape}")
        logger.info(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")

        next_token_logits = logits[0, -1, :]
        logger.info(f"Next token logits: min={next_token_logits.min():.4f}, max={next_token_logits.max():.4f}")

        # Top 5 tokens
        top_5_indices = np.argsort(next_token_logits)[-5:][::-1]
        logger.info(f"Top 5 token IDs: {top_5_indices}")
        logger.info(f"Top 5 logits: {next_token_logits[top_5_indices]}")

        for idx in top_5_indices:
            try:
                token_text = tokenizer.decode([idx])
                logger.info(f"  {idx}: '{token_text}' (logit={next_token_logits[idx]:.4f})")
            except:
                logger.info(f"  {idx}: [cannot decode] (logit={next_token_logits[idx]:.4f})")

        next_token_id = np.argmax(next_token_logits)
        logger.info(f"✅ Selected token: {next_token_id}")

        # Try to decode it
        try:
            token_text = tokenizer.decode([next_token_id])
            logger.info(f"   Token text: '{token_text}'")
        except Exception as e:
            logger.info(f"   Could not decode: {e}")

        # Check for EOS
        if next_token_id == 50257:
            logger.info("✅ EOS token reached")
            break

        # Add to sequence
        decoder_input_ids = np.concatenate([
            decoder_input_ids,
            np.array([[next_token_id]], dtype=np.int64)
        ], axis=1)

        generated_tokens.append(next_token_id)

    logger.info(f"\n✅ Generated {len(generated_tokens)} tokens: {generated_tokens}")

    # Decode full text
    text_tokens = [t for t in generated_tokens if t < 50257]
    logger.info(f"✅ Text tokens (filtered): {text_tokens}")

    if text_tokens:
        try:
            text = tokenizer.decode(text_tokens, skip_special_tokens=True)
            logger.info(f"✅ Decoded text: '{text}'")
        except Exception as e:
            logger.error(f"❌ Failed to decode: {e}")
    else:
        logger.warning("⚠️ No text tokens generated!")

    return True

def test_full_transcription():
    """Test 4: Full Transcription Test"""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Full Transcription Test")
    logger.info("="*80)

    from npu.npu_optimization.onnx_whisper_npu import ONNXWhisperNPU
    import soundfile as sf

    whisper = ONNXWhisperNPU()
    if not whisper.initialize(model_size="base"):
        logger.error("Failed to initialize ONNX Whisper")
        return False

    # Create test audio and save to file
    audio, sr = create_test_audio(duration=5.0)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio, sr)
        test_audio_path = tmp_file.name
        logger.info(f"✅ Created test audio file: {test_audio_path}")

    try:
        # Transcribe
        result = whisper.transcribe_audio(test_audio_path)

        logger.info(f"\n✅ Transcription complete!")
        logger.info(f"Text: '{result['text']}'")
        logger.info(f"Processing time: {result['processing_time']:.2f}s")
        logger.info(f"Real-time factor: {result['real_time_factor']:.3f}x")
        logger.info(f"NPU accelerated: {result['npu_accelerated']}")

        if result['text'].strip() and not result['text'].startswith('['):
            logger.info("✅ Decoder produced meaningful text!")
            return True
        else:
            logger.warning(f"⚠️ Decoder produced placeholder or empty text: '{result['text']}'")
            return False

    finally:
        # Cleanup
        os.unlink(test_audio_path)

def main():
    """Run all decoder diagnostic tests"""
    logger.info("\n" + "="*100)
    logger.info("🔍 NPU DECODER DIAGNOSTIC TEST SUITE")
    logger.info("="*100)
    logger.info("\nGoal: Diagnose why decoder produces garbled output\n")

    tests = [
        ("ONNX Model Structure", test_onnx_model_structure),
        ("Encoder Output Validation", test_encoder_outputs),
        ("Step-by-Step Decoder Debug", test_decoder_step_by_step),
        ("Full Transcription", test_full_transcription),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "✅ PASS" if success else "❌ FAIL"))
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, f"❌ CRASH: {e}"))

    logger.info("\n" + "="*100)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("="*100)

    for test_name, result in results:
        logger.info(f"{result:12} | {test_name}")

    logger.info("="*100)

if __name__ == "__main__":
    main()
