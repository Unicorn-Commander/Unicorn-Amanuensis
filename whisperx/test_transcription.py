#!/usr/bin/env python3
"""
Test transcription to debug why only first letter appears
"""

import os
import logging
import soundfile as sf
import numpy as np
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import WhisperProcessor
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_transcription():
    """Test INT8 model transcription"""
    
    # Load model
    model_path = "/home/ucadmin/openvino-models/whisper-large-v3-int8"
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    
    logger.info("Loading INT8 model...")
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        device="GPU",
        compile=True
    )
    
    # Create test audio (5 seconds of sine wave)
    sample_rate = 16000
    duration = 5
    t = np.linspace(0, duration, sample_rate * duration)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    
    logger.info(f"Test audio: {len(audio)} samples, {duration}s")
    
    # Process audio
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    logger.info(f"Input features shape: {inputs.input_features.shape}")
    
    # Generate with different parameters
    logger.info("Testing generation...")
    
    # Test 1: Basic generation
    predicted_ids = model.generate(inputs.input_features)
    text1 = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    logger.info(f"Test 1 (basic): '{text1}'")
    
    # Test 2: With max_new_tokens
    predicted_ids = model.generate(
        inputs.input_features,
        max_new_tokens=100
    )
    text2 = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    logger.info(f"Test 2 (max_new_tokens=100): '{text2}'")
    
    # Test 3: Force longer generation
    predicted_ids = model.generate(
        inputs.input_features,
        min_new_tokens=10,
        max_new_tokens=200,
        do_sample=False
    )
    text3 = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    logger.info(f"Test 3 (min=10, max=200): '{text3}'")
    
    # Test 4: Check token IDs
    logger.info(f"Token IDs generated: {predicted_ids[0][:20].tolist()}...")
    
    # Test with real audio file if exists
    test_audio = "/tmp/test.wav"
    if os.path.exists(test_audio):
        logger.info(f"\nTesting with real audio: {test_audio}")
        audio, sr = sf.read(test_audio)
        
        # Resample if needed
        if sr != 16000:
            logger.info(f"Resampling from {sr} to 16000")
            # Simple downsample (not ideal but quick)
            audio = audio[::sr//16000]
        
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        predicted_ids = model.generate(inputs.input_features, max_new_tokens=200)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logger.info(f"Real audio result: '{text}'")

if __name__ == "__main__":
    test_transcription()