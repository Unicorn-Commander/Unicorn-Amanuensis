#!/usr/bin/env python3
"""
Transcribe Shafen Khan audio using chunked processing
Works around the 448 token limit by processing 30-second chunks
"""

import requests
import soundfile as sf
import numpy as np
import time
import sys

def transcribe_chunked(audio_path, server_url="http://127.0.0.1:9006", chunk_duration=30):
    """Transcribe long audio by splitting into chunks"""
    
    print(f"📂 Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path)
    total_duration = len(audio) / sr
    print(f"📊 Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    
    # Calculate chunks
    chunk_samples = chunk_duration * sr
    n_chunks = int(np.ceil(len(audio) / chunk_samples))
    print(f"🔄 Processing in {n_chunks} chunks of {chunk_duration} seconds")
    
    all_transcriptions = []
    start_time = time.time()
    
    for i in range(n_chunks):
        chunk_start = i * chunk_samples
        chunk_end = min((i + 1) * chunk_samples, len(audio))
        audio_chunk = audio[chunk_start:chunk_end]
        
        chunk_duration_actual = len(audio_chunk) / sr
        print(f"\n📝 Chunk {i+1}/{n_chunks}: {chunk_duration_actual:.1f}s", end="")
        
        # Save chunk to temporary file
        chunk_path = f"/tmp/chunk_{i}.wav"
        sf.write(chunk_path, audio_chunk, sr)
        
        # Send to server
        try:
            with open(chunk_path, 'rb') as f:
                files = {'audio': f}
                data = {'enable_diarization': 'false'}
                response = requests.post(f"{server_url}/transcribe", files=files, data=data)
                
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()
                if text:
                    all_transcriptions.append(text)
                    print(f" → {text[:50]}..." if len(text) > 50 else f" → {text}")
                else:
                    print(" → [empty]")
            else:
                print(f" → Error: {response.status_code}")
                
        except Exception as e:
            print(f" → Error: {e}")
    
    # Combine all transcriptions
    full_transcription = " ".join(all_transcriptions)
    
    # Calculate stats
    total_time = time.time() - start_time
    speed = total_duration / total_time
    
    print("\n" + "="*60)
    print("📄 TRANSCRIPTION COMPLETE")
    print("="*60)
    print(f"Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"Processing time: {total_time:.1f}s")
    print(f"Speed: {speed:.1f}x realtime")
    print(f"Chunks: {n_chunks}")
    print(f"Words transcribed: {len(full_transcription.split())}")
    print("="*60)
    print("\nFULL TRANSCRIPTION:")
    print("-"*60)
    print(full_transcription)
    print("-"*60)
    
    # Save to file
    output_file = audio_path.replace('.wav', '_transcription.txt').replace('.m4a', '_transcription.txt')
    with open(output_file, 'w') as f:
        f.write(f"Shafen Khan Call Transcription\n")
        f.write(f"Duration: {total_duration/60:.1f} minutes\n")
        f.write(f"Processing: {speed:.1f}x realtime on Intel iGPU\n")
        f.write("="*60 + "\n\n")
        f.write(full_transcription)
    
    print(f"\n✅ Transcription saved to: {output_file}")
    
    return full_transcription

if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/shafen_khan_call.wav"
    transcribe_chunked(audio_file)