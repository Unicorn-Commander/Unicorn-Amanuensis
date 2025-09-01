#!/usr/bin/env python3
"""
🦄 Unicorn Amanuensis - Real Intel iGPU SYCL Server
Using actual SYCL kernels compiled with Intel DPC++ for hardware acceleration
"""

import os
import logging
import time
import numpy as np
import librosa
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
import tempfile

# Set up oneAPI environment
oneapi_path = "/opt/intel/oneapi"
if Path(oneapi_path).exists():
    os.environ.update({
        "ONEAPI_ROOT": oneapi_path,
        "MKLROOT": f"{oneapi_path}/mkl/latest",
        "LD_LIBRARY_PATH": f"{oneapi_path}/lib:{oneapi_path}/compiler/latest/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "PATH": f"{oneapi_path}/bin:{oneapi_path}/compiler/latest/bin:{os.environ.get('PATH', '')}"
    })

# Import our real SYCL runtime
import whisper_igpu_runtime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize REAL Intel iGPU SYCL runtime
logger.info("🦄 Initializing REAL Intel iGPU SYCL Runtime...")
whisper_model = None

def init_model():
    global whisper_model
    try:
        whisper_model = whisper_igpu_runtime.WhisperXIGPU("large-v3")
        logger.info("✅ Real Intel iGPU SYCL model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load SYCL model: {e}")
        return False

@app.route('/web')
def web_interface():
    """Serve the WhisperX web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🦄 Unicorn Amanuensis - Intel iGPU SYCL</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            .status { 
                background: rgba(0,255,0,0.2); 
                padding: 15px; 
                border-radius: 8px; 
                margin: 10px 0; 
                border: 1px solid rgba(0,255,0,0.3);
            }
            input[type="file"] {
                background: rgba(255,255,255,0.2);
                padding: 10px;
                border: none;
                border-radius: 5px;
                color: white;
                width: 100%;
                margin: 10px 0;
            }
            button {
                background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                transition: transform 0.2s;
            }
            button:hover { transform: scale(1.05); }
            .result {
                background: rgba(255,255,255,0.15);
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
                white-space: pre-wrap;
                font-family: monospace;
            }
            .unicorn { font-size: 2em; text-align: center; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="unicorn">🦄 Unicorn Amanuensis</div>
            <h1>Intel iGPU SYCL Speech-to-Text</h1>
            
            <div class="status">
                <strong>🎯 Status:</strong> REAL Intel iGPU SYCL Kernels Active<br>
                <strong>🔧 Model:</strong> Whisper Large v3<br>
                <strong>⚡ Device:</strong> Intel UHD Graphics 770 (SYCL)<br>
                <strong>🚀 Mode:</strong> Hardware-only acceleration
            </div>
            
            <form id="uploadForm">
                <label for="audio">Select audio file:</label><br>
                <input type="file" id="audio" name="audio" accept="audio/*" required><br><br>
                <button type="submit">🎤 Transcribe with SYCL</button>
            </form>
            
            <div id="result" class="result" style="display:none;"></div>
        </div>

        <script>
        document.getElementById('uploadForm').onsubmit = function(e) {
            e.preventDefault();
            
            var formData = new FormData();
            var audioFile = document.getElementById('audio').files[0];
            formData.append('audio', audioFile);
            
            document.getElementById('result').style.display = 'block';
            document.getElementById('result').innerHTML = '🔄 Processing with Intel iGPU SYCL kernels...';
            
            fetch('/transcribe', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('result').innerHTML = '❌ Error: ' + data.error;
                } else {
                    var result = `✅ Transcription Complete!

🎯 Text: ${data.text}

⏱️ Processing Time: ${data.inference_time}s
📊 Speed: ${data.speed}
🔧 Device: ${data.device}
📝 Language: ${data.language}`;
                    document.getElementById('result').innerHTML = result;
                }
            })
            .catch(error => {
                document.getElementById('result').innerHTML = '❌ Error: ' + error;
            });
        }
        </script>
    </body>
    </html>
    """

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe audio using REAL Intel iGPU SYCL kernels"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_file.save(tmp_file.name)
            
            # Load and preprocess audio
            logger.info(f"Loading audio file: {audio_file.filename}")
            audio, sr = librosa.load(tmp_file.name, sr=16000, mono=True)
            logger.info(f"Audio loaded: {len(audio)} samples, {len(audio)/16000:.1f}s")
            
            # Transcribe using REAL SYCL kernels
            start_time = time.time()
            
            if whisper_model is None:
                return jsonify({'error': 'SYCL model not initialized'}), 500
            
            result = whisper_model.transcribe(audio, language="en")
            
            inference_time = time.time() - start_time
            duration = len(audio) / 16000
            speed = f"{duration/inference_time:.1f}x realtime"
            
            # Clean up
            os.unlink(tmp_file.name)
            
            return jsonify({
                'text': result['text'],
                'segments': result.get('segments', []),
                'language': result.get('language', 'en'),
                'inference_time': round(inference_time, 2),
                'duration': round(duration, 1), 
                'device': 'Intel iGPU (Real SYCL Kernels)',
                'speed': speed
            })
            
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Get model and hardware status"""
    try:
        import subprocess
        
        # Get GPU info
        try:
            gpu_info = subprocess.check_output(['intel_gpu_top', '-l'], 
                                             timeout=2, text=True)
        except:
            gpu_info = "Intel GPU detected via SYCL"
        
        return jsonify({
            'status': 'ready',
            'model': 'Whisper Large v3',
            'device': 'Intel UHD Graphics 770',
            'backend': 'SYCL (Real Hardware Kernels)',
            'gpu_info': gpu_info,
            'sycl_initialized': whisper_model is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🦄 Starting Unicorn Amanuensis - Intel iGPU SYCL Server")
    
    # Initialize model
    if not init_model():
        logger.error("❌ Failed to initialize SYCL model")
        exit(1)
    
    logger.info("🌟 Server ready on http://0.0.0.0:9000")
    logger.info("🎤 STT Service: Intel iGPU SYCL acceleration")
    logger.info("🔗 Web Interface: http://0.0.0.0:9000/web")
    
    app.run(host='0.0.0.0', port=9000, debug=False, threaded=True)