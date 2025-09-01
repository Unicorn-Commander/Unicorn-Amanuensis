#!/usr/bin/env python3
"""
Chunked transcription for long audio files using Intel iGPU
Processes audio in 30-second chunks to fit within Whisper's token limits
"""

import numpy as np
import soundfile as sf
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor
import logging
import time
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transcribe_audio_chunked(audio_path, model_name="base", chunk_duration=30):
    """
    Transcribe long audio by processing in chunks
    
    Args:
        audio_path: Path to audio file
        model_name: "base" or "large-v3"
        chunk_duration: Duration of each chunk in seconds (max 30)
    """
    
    # Select model path
    if model_name == "large-v3":
        model_path = "/home/ucadmin/openvino-models/whisper-large-v3-int8"
    else:
        model_path = "/home/ucadmin/openvino-models/whisper-base-int8"
    
    logger.info(f"🚀 Loading {model_name} INT8 model on Intel iGPU...")
    
    # Load model on Intel GPU
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        device="GPU",
        ov_config={"PERFORMANCE_HINT": "THROUGHPUT"}
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    logger.info(f"✅ Model loaded on Intel iGPU")
    
    # Load audio
    logger.info(f"🎵 Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path)
    total_duration = len(audio) / sr
    logger.info(f"📊 Audio duration: {total_duration:.1f}s")
    
    # Process in chunks
    chunk_size = chunk_duration * sr
    n_chunks = int(np.ceil(len(audio) / chunk_size))
    logger.info(f"🔄 Processing in {n_chunks} chunks of {chunk_duration}s each")
    
    all_transcriptions = []
    start_time = time.time()
    
    for i in range(n_chunks):
        chunk_start = i * chunk_size
        chunk_end = min((i + 1) * chunk_size, len(audio))
        audio_chunk = audio[chunk_start:chunk_end]
        
        chunk_duration_actual = len(audio_chunk) / sr
        logger.info(f"📝 Chunk {i+1}/{n_chunks}: {chunk_duration_actual:.1f}s")
        
        # Process chunk
        inputs = processor(audio_chunk, sampling_rate=sr, return_tensors="pt")
        
        # Generate transcription with proper settings
        predicted_ids = model.generate(
            inputs.input_features,
            max_new_tokens=444,  # Max for Whisper
            language="en",
            task="transcribe",
            do_sample=False,
            num_beams=1
        )
        
        # Decode
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        all_transcriptions.append(transcription.strip())
        
        logger.info(f"   ✅ Transcribed: {transcription[:100]}...")
    
    # Combine all transcriptions
    full_transcription = " ".join(all_transcriptions)
    
    # Calculate performance
    total_time = time.time() - start_time
    speed = total_duration / total_time
    
    logger.info(f"🎉 Transcription complete!")
    logger.info(f"⚡ Performance: {speed:.1f}x realtime")
    logger.info(f"🎮 Device: Intel UHD Graphics 770")
    logger.info(f"🔧 Optimization: INT8 Quantization")
    
    return {
        "text": full_transcription,
        "duration": total_duration,
        "processing_time": total_time,
        "speed": speed,
        "n_chunks": n_chunks,
        "model": f"{model_name} (INT8)",
        "device": "Intel iGPU"
    }

if __name__ == "__main__":
    # Test with command line argument or default file
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/shafen_khan_call.wav"
    model = sys.argv[2] if len(sys.argv) > 2 else "base"
    
    result = transcribe_audio_chunked(audio_file, model)
    
    print("\n" + "="*60)
    print("📄 TRANSCRIPTION RESULT")
    print("="*60)
    print(f"Duration: {result['duration']:.1f}s")
    print(f"Speed: {result['speed']:.1f}x realtime")
    print(f"Chunks: {result['n_chunks']}")
    print(f"Model: {result['model']}")
    print(f"Device: {result['device']}")
    print("="*60)
    print("TEXT:")
    print(result['text'])
    print("="*60)