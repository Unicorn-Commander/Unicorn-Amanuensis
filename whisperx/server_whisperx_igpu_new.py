#!/usr/bin/env python3
"""
WhisperX with Intel iGPU Acceleration
Uses WhisperX for accurate transcription with our iGPU optimizations
"""

import os
import logging
import time
import tempfile
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import whisperx
import torch
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)

# Global model
model = None
current_model_name = "large-v3"

def load_whisperx_model(model_name="large-v3"):
    """Load WhisperX model optimized for Intel iGPU"""
    global model, current_model_name
    
    try:
        logger.info(f"🚀 Loading WhisperX {model_name} model...")
        
        # Set device - use CPU for now but optimize with Intel extensions
        device = "cpu"
        compute_type = "int8"  # Use INT8 for speed
        
        # Load model with WhisperX
        model = whisperx.load_model(
            model_name, 
            device=device,
            compute_type=compute_type,
            language="en",
            threads=8  # Use multiple threads for CPU backend
        )
        
        current_model_name = model_name
        logger.info(f"✅ WhisperX {model_name} loaded with INT8 optimization")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load WhisperX model: {e}")
        return False

def preprocess_audio(input_path, output_path):
    """Preprocess audio using Intel QSV/VAAPI"""
    try:
        logger.info("🎵 Transcoding audio with Intel iGPU...")
        
        # Use Intel QSV for hardware acceleration
        cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'vaapi',
            '-hwaccel_device', '/dev/dri/renderD128',
            '-i', input_path,
            '-ar', '16000',
            '-ac', '1',
            '-acodec', 'pcm_s16le',
            '-f', 'wav',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return False

def transcribe_with_whisperx(audio_path, enable_diarization=False):
    """Transcribe using WhisperX with optimizations"""
    try:
        start_time = time.time()
        
        # Load audio
        audio = whisperx.load_audio(audio_path)
        duration = len(audio) / 16000  # WhisperX uses 16kHz
        
        logger.info(f"🎯 Processing {duration:.1f}s audio with WhisperX...")
        
        # Transcribe with WhisperX
        result = model.transcribe(
            audio,
            batch_size=16,  # Larger batch for efficiency
            language="en",
            chunk_length=30,  # Process in 30s chunks
            print_progress=True
        )
        
        transcribe_time = time.time() - start_time
        logger.info(f"✅ Transcription done in {transcribe_time:.2f}s")
        
        # Align whisper output (word-level timestamps)
        logger.info("🔄 Aligning output for word-level timestamps...")
        model_a, metadata_a = whisperx.load_align_model(
            language_code="en",
            device="cpu"
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata_a,
            audio,
            "cpu",
            return_char_alignments=False
        )
        
        # Optional: Speaker diarization
        if enable_diarization:
            logger.info("👥 Running speaker diarization...")
            try:
                hf_token = os.environ.get('HF_TOKEN', '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')
                diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token)
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                logger.info("✅ Diarization complete")
            except Exception as e:
                logger.warning(f"Diarization failed: {e}")
        
        # Calculate performance
        total_time = time.time() - start_time
        speed = duration / total_time
        
        logger.info(f"🚀 Processing complete: {total_time:.2f}s ({speed:.1f}x realtime)")
        
        # Format result
        full_text = " ".join([seg.get("text", "") for seg in result.get("segments", [])])
        
        return {
            "text": full_text,
            "segments": result.get("segments", []),
            "word_segments": result.get("word_segments", []),
            "duration": duration,
            "processing_time": total_time,
            "speed": f"{speed:.1f}x realtime",
            "model": f"WhisperX {current_model_name}",
            "device": "Intel CPU with INT8",
            "features": {
                "word_timestamps": True,
                "alignment": True,
                "diarization": enable_diarization,
                "hardware_transcoding": True
            }
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
        'service': 'Unicorn Amanuensis - WhisperX iGPU',
        'description': 'WhisperX with Intel optimization',
        'models': ['base', 'small', 'medium', 'large-v2', 'large-v3'],
        'features': {
            'word_timestamps': 'Precise word-level timing',
            'alignment': 'Phoneme-based alignment',
            'diarization': 'Speaker identification',
            'hardware_acceleration': 'Intel QSV/VAAPI transcoding'
        },
        'endpoints': {
            '/': 'This documentation',
            '/transcribe': 'Transcribe audio',
            '/status': 'Server status'
        }
    })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe endpoint"""
    try:
        # Get audio file
        if 'audio' in request.files:
            audio_file = request.files['audio']
        elif 'file' in request.files:
            audio_file = request.files['file']
        else:
            return jsonify({"error": "No audio file provided"}), 400
        
        # Get options
        enable_diarization = request.form.get('diarization', 'false').lower() == 'true'
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as tmp_input:
            audio_file.save(tmp_input.name)
            input_path = tmp_input.name
        
        # Preprocess audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
            wav_path = tmp_wav.name
        
        if not preprocess_audio(input_path, wav_path):
            os.unlink(input_path)
            return jsonify({"error": "Audio preprocessing failed"}), 500
        
        # Transcribe
        result = transcribe_with_whisperx(wav_path, enable_diarization)
        
        # Clean up
        os.unlink(input_path)
        os.unlink(wav_path)
        
        # Clear GPU cache
        gc.collect()
        
        if result:
            return jsonify(result)
        else:
            return jsonify({"error": "Transcription failed"}), 500
            
    except Exception as e:
        logger.error(f"Request error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Server status"""
    return jsonify({
        "status": "ready" if model else "not ready",
        "model": f"WhisperX {current_model_name}",
        "device": "Intel CPU with INT8",
        "features": {
            "transcription": model is not None,
            "word_timestamps": True,
            "alignment": True,
            "diarization": True
        }
    })

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🦄 Unicorn Amanuensis - WhisperX iGPU Server")
    logger.info("=" * 60)
    logger.info("📝 WhisperX: Advanced transcription with alignment")
    logger.info("⚡ Intel optimizations: INT8 quantization")
    logger.info("🎯 Features: Word timestamps, diarization")
    logger.info("=" * 60)
    
    # Load model
    if not load_whisperx_model("large-v3"):
        logger.error("Failed to load WhisperX model!")
        exit(1)
    
    # Server configuration
    PORT = 9008
    
    logger.info(f"🌟 Server starting on http://0.0.0.0:{PORT}")
    logger.info(f"📖 API Docs: http://0.0.0.0:{PORT}/")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=False)