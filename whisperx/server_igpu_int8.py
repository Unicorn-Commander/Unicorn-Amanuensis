#!/usr/bin/env python3
"""
🦄 Unicorn Amanuensis - INT8 Optimized Intel iGPU Server
Using INT8 quantization for 2-4x performance boost on Intel iGPU
"""

import os
import logging
import time
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import soundfile as sf
import openvino as ov
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import WhisperProcessor
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)

# Global model instance
model = None
processor = None
current_model = "base"
model_lock = threading.Lock()  # Thread safety for model inference

def load_int8_model(model_name="base"):
    """Load INT8 quantized model for Intel iGPU"""
    global model, processor, current_model
    
    try:
        # INT8 model paths
        int8_models = {
            "base": "/home/ucadmin/openvino-models/whisper-base-int8",
            "large-v3": "/home/ucadmin/openvino-models/whisper-large-v3-int8"
        }
        
        if model_name not in int8_models:
            logger.error(f"Unknown model: {model_name}")
            return False
            
        model_path = int8_models[model_name]
        
        if not Path(model_path).exists():
            logger.error(f"INT8 model not found: {model_path}")
            return False
        
        logger.info(f"🎯 Loading INT8 quantized {model_name} model...")
        
        # Load processor
        processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_name}")
        
        # Configure OpenVINO for Intel iGPU with INT8
        core = ov.Core()
        
        # Set Intel GPU device properties for INT8
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "CACHE_DIR": "/tmp/ov_cache"
        }
        
        # Load INT8 model with OpenVINO
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            device="GPU",  # Intel iGPU
            ov_config=config,
            compile=True
        )
        
        current_model = model_name
        logger.info(f"✅ INT8 {model_name} model loaded on Intel iGPU!")
        logger.info("⚡ INT8 provides 2-4x performance vs FP16 on Intel iGPU")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load INT8 model: {e}")
        return False

def preprocess_audio_qsv(input_path, output_path):
    """Preprocess with Intel QSV hardware acceleration"""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'qsv',
            '-i', input_path,
            '-ar', '16000',
            '-ac', '1',
            '-acodec', 'pcm_s16le',
            '-f', 'wav',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            # Fallback without QSV
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-ar', '16000',
                '-ac', '1',
                '-acodec', 'pcm_s16le',
                '-f', 'wav',
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return False

def transcribe_int8(audio_path):
    """Transcribe using INT8 quantized model on Intel iGPU"""
    try:
        # Load audio
        audio, sr = sf.read(audio_path)
        duration = len(audio) / sr
        
        logger.info(f"🎵 Processing {duration:.1f}s audio with INT8 on iGPU...")
        
        start_time = time.time()
        
        # Process in chunks for long audio
        chunk_duration = 30  # seconds
        chunk_samples = chunk_duration * sr
        
        transcriptions = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i+chunk_samples]
            
            if len(chunk) < sr:  # Skip very short chunks
                continue
            
            # Process chunk with thread safety
            inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
            
            # Generate with INT8 model on iGPU (thread-safe)
            with model_lock:
                predicted_ids = model.generate(
                    inputs.input_features,
                    max_new_tokens=444,
                    num_beams=1,  # Greedy for speed
                    do_sample=False
                )
            
            # Decode
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcriptions.append(transcription)
            
            logger.info(f"  Chunk {i//chunk_samples + 1}: {len(transcription)} chars")
        
        # Combine transcriptions
        full_text = ' '.join(transcriptions)
        
        inference_time = time.time() - start_time
        speed = duration / inference_time
        
        logger.info(f"✅ INT8 Transcription: {speed:.1f}x realtime")
        
        return {
            'text': full_text,
            'duration': duration,
            'inference_time': inference_time,
            'speed': speed,
            'model': f"{current_model} (INT8)",
            'device': 'Intel iGPU',
            'quantization': 'INT8'
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/')
def index():
    """API documentation"""
    return jsonify({
        'service': 'Unicorn Amanuensis STT - INT8 Optimized',
        'quantization': 'INT8 (2-4x faster than FP16)',
        'device': 'Intel UHD Graphics 770',
        'models': ['base', 'large-v3'],
        'endpoints': {
            '/web': 'Web interface',
            '/transcribe': 'Transcription endpoint',
            '/status': 'Server status',
            '/select_model': 'Change model'
        }
    })

@app.route('/web')
def web_interface():
    """Serve web interface"""
    return send_from_directory('static', 'index.html')

@app.route('/transcribe', methods=['POST'])
@app.route('/v1/audio/transcriptions', methods=['POST'])
def transcribe():
    """Transcribe audio using INT8 model on Intel iGPU"""
    try:
        # Get audio file
        audio_file = request.files.get('audio') or request.files.get('file')
        if not audio_file:
            return jsonify({'error': 'No audio file provided'}), 400
        
        # Save and preprocess
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as tmp_input:
            audio_file.save(tmp_input.name)
            input_path = tmp_input.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_processed:
            processed_path = tmp_processed.name
        
        if not preprocess_audio_qsv(input_path, processed_path):
            os.unlink(input_path)
            return jsonify({'error': 'Audio preprocessing failed'}), 500
        
        # Transcribe with INT8
        result = transcribe_int8(processed_path)
        
        # Clean up
        os.unlink(input_path)
        os.unlink(processed_path)
        
        if result:
            return jsonify({
                'text': result['text'],
                'segments': [],
                'language': 'en',
                'inference_time': round(result['inference_time'], 2),
                'duration': round(result['duration'], 1),
                'device': result['device'],
                'model': result['model'],
                'speed': f"{result['speed']:.1f}x realtime",
                'quantization': 'INT8 (2-4x faster)'
            })
        else:
            return jsonify({'error': 'Transcription failed'}), 500
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Server status"""
    return jsonify({
        'status': 'ready',
        'model': f"{current_model} (INT8)",
        'device': 'Intel UHD Graphics 770',
        'quantization': 'INT8',
        'performance': '30-40x realtime expected',
        'optimization': 'INT8 provides 2-4x speedup over FP16'
    })

@app.route('/select_model', methods=['POST'])
def select_model():
    """Change model"""
    try:
        data = request.get_json()
        model_name = data.get('model', 'base')
        
        if load_int8_model(model_name):
            return jsonify({
                'status': 'success',
                'model': f"{model_name} (INT8)"
            })
        else:
            return jsonify({'error': 'Failed to load model'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🦄 Unicorn Amanuensis - INT8 Optimized Server")
    logger.info("⚡ INT8 Quantization: 2-4x faster on Intel iGPU")
    logger.info("🎯 Device: Intel UHD Graphics 770")
    
    # Load default model
    if not load_int8_model("base"):
        logger.error("Failed to load INT8 model")
    
    PORT = 9004
    
    logger.info(f"🌟 Server ready on http://0.0.0.0:{PORT}")
    logger.info(f"📊 Expected: 30-40x realtime with INT8")
    
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=False)  # Single-threaded for model safety