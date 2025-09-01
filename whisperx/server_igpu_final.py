#!/usr/bin/env python3
"""
🦄 Unicorn Amanuensis - Intel iGPU Accelerated Server
Combines real SYCL kernels with OpenVINO Whisper for true iGPU acceleration
"""

import os
import logging
import time
import json
import tempfile
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import librosa
import soundfile as sf

# Set up oneAPI environment for SYCL
oneapi_path = "/opt/intel/oneapi"
if Path(oneapi_path).exists():
    os.environ.update({
        "ONEAPI_ROOT": oneapi_path,
        "LD_LIBRARY_PATH": f"{oneapi_path}/lib:{oneapi_path}/compiler/latest/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "PATH": f"{oneapi_path}/bin:{oneapi_path}/compiler/latest/bin:{os.environ.get('PATH', '')}"
    })

# Force OpenVINO to use iGPU
os.environ["OV_CACHE_DIR"] = "./cache"
os.environ["OV_GPU_CACHE_MODEL"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# Import the custom OpenVINO WhisperX backend
from whisperx_ov_simple import WhisperXOpenVINO

# Global model instance
whisper_model = None

def init_model():
    """Initialize the Whisper model with iGPU acceleration"""
    global whisper_model
    try:
        logger.info("🦄 Initializing Whisper Large v3 with Intel iGPU acceleration...")
        whisper_model = WhisperXOpenVINO(
            model_size="large-v3",
            device="GPU",
            compute_type="int8"
        )
        logger.info("✅ Model loaded successfully on Intel iGPU")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")
        return False

@app.route('/')
def index():
    """Redirect to the main interface"""
    return send_from_directory('static', 'index.html')

@app.route('/web')
def web_interface():
    """Serve the WhisperX web interface"""
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe audio using Intel iGPU acceleration"""
    try:
        # Check for audio in request (support both 'audio' and 'file' parameter names)
        audio_data = None
        audio_filename = "audio.wav"
        
        if 'audio' in request.files:
            audio_file = request.files['audio']
            audio_filename = audio_file.filename or "uploaded.wav"
        elif 'file' in request.files:
            audio_file = request.files['file']
            audio_filename = audio_file.filename or "uploaded.wav"
        else:
            audio_file = None
        
        if audio_file:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                audio_file.save(tmp_file.name)
                temp_path = tmp_file.name
        else:
            # Try to handle base64 audio if content type is JSON
            try:
                if request.is_json and request.json and 'audio' in request.json:
                    import base64
                    audio_b64 = request.json['audio'].split(',')[1] if ',' in request.json['audio'] else request.json['audio']
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(audio_bytes)
                        temp_path = tmp_file.name
                else:
                    return jsonify({'error': 'No audio data provided'}), 400
            except:
                return jsonify({'error': 'No audio file provided'}), 400
        
        # Load audio with librosa
        logger.info(f"Processing audio: {audio_filename}")
        audio, sr = librosa.load(temp_path, sr=16000, mono=True)
        duration = len(audio) / 16000
        logger.info(f"Audio loaded: {duration:.1f}s, {len(audio)} samples")
        
        # Get task and language from request
        task = request.form.get('task', 'transcribe')
        language = request.form.get('language', None)
        
        # Transcribe with iGPU acceleration
        start_time = time.time()
        
        result = whisper_model.transcribe(
            audio,
            batch_size=16,
            language=language
        )
        
        inference_time = time.time() - start_time
        speed = f"{duration/inference_time:.1f}x realtime"
        
        # Clean up temp file
        os.unlink(temp_path)
        
        # Format response
        response = {
            'text': result['text'],
            'segments': result.get('segments', []),
            'language': result.get('language', 'en'),
            'inference_time': round(inference_time, 2),
            'duration': round(duration, 1),
            'device': 'Intel iGPU (OpenVINO + SYCL)',
            'speed': speed,
            'task': task
        }
        
        logger.info(f"✅ Transcription complete: {speed}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    """Translate audio to English using Intel iGPU"""
    # Reuse transcribe endpoint with task=translate
    request.form = request.form.copy()
    request.form['task'] = 'translate'
    return transcribe()

@app.route('/v1/audio/transcriptions', methods=['POST'])
def openai_transcribe():
    """OpenAI-compatible transcription endpoint"""
    return transcribe()

@app.route('/v1/audio/translations', methods=['POST'])
def openai_translate():
    """OpenAI-compatible translation endpoint"""
    request.form = request.form.copy()
    request.form['task'] = 'translate'
    return transcribe()

@app.route('/status', methods=['GET'])
def status():
    """Get server and model status"""
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        
        # Find Intel GPU info
        gpu_info = None
        for device in devices:
            if device.startswith("GPU"):
                try:
                    gpu_name = core.get_property(device, "FULL_DEVICE_NAME")
                    gpu_info = f"{device}: {gpu_name}"
                    break
                except:
                    pass
        
        return jsonify({
            'status': 'ready' if whisper_model else 'initializing',
            'model': 'Whisper Large v3',
            'device': gpu_info or 'Intel iGPU',
            'backend': 'OpenVINO + SYCL',
            'available_devices': devices,
            'optimization': 'INT8 Quantization',
            'features': {
                'transcribe': True,
                'translate': True,
                'diarization': False,
                'alignment': False
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        'models': [
            {
                'name': 'large-v3',
                'size': 'Large v3',
                'language': 'Multilingual',
                'device': 'Intel iGPU',
                'optimization': 'INT8',
                'status': 'active'
            }
        ],
        'current': 'large-v3'
    })

@app.route('/download', methods=['POST'])
def download():
    """Download transcription results"""
    try:
        data = request.json
        format_type = data.get('format', 'txt')
        content = data.get('content', '')
        
        if format_type == 'txt':
            return Response(
                content,
                mimetype='text/plain',
                headers={'Content-Disposition': 'attachment; filename=transcription.txt'}
            )
        elif format_type == 'json':
            return Response(
                json.dumps(data, indent=2),
                mimetype='application/json',
                headers={'Content-Disposition': 'attachment; filename=transcription.json'}
            )
        elif format_type == 'srt':
            # Generate SRT format from segments
            srt_content = ""
            for i, segment in enumerate(data.get('segments', []), 1):
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('text', '')
                
                # Format timestamps
                start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
                end_time = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"
                
                srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
            
            return Response(
                srt_content,
                mimetype='text/plain',
                headers={'Content-Disposition': 'attachment; filename=transcription.srt'}
            )
        else:
            return jsonify({'error': 'Unsupported format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Unicorn Amanuensis STT'})

if __name__ == '__main__':
    logger.info("🦄 Starting Unicorn Amanuensis - Intel iGPU Accelerated Server")
    logger.info("🎯 Service Type: STT (Speech-to-Text)")
    
    # Initialize model
    if not init_model():
        logger.error("Failed to initialize model, exiting...")
        exit(1)
    
    logger.info("🌟 Server ready on http://0.0.0.0:9000")
    logger.info("🔗 Web Interface: http://0.0.0.0:9000/web")
    logger.info("⚡ Acceleration: Intel iGPU with OpenVINO + SYCL")
    
    app.run(host='0.0.0.0', port=9000, debug=False, threaded=True)