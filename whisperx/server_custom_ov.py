#!/usr/bin/env python3
"""
Unicorn Amanuensis - Custom OpenVINO STT Server
TRUE iGPU acceleration without CTranslate2
"""

import os
import sys
import logging
import tempfile
import time
import gc
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Use our custom OpenVINO backend
sys.path.insert(0, os.path.dirname(__file__))
import whisperx_ov_simple as whisperx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unicorn Amanuensis - STT (Custom OpenVINO)",
    description="Speech-to-Text with TRUE Intel iGPU acceleration",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = "GPU"
COMPUTE_TYPE = "int8"
API_PORT = int(os.environ.get("API_PORT", "9000"))

# Load model with OpenVINO backend
logger.info(f"Loading Whisper {MODEL_SIZE} with OpenVINO iGPU backend...")
start_time = time.time()

try:
    model = whisperx.load_model(MODEL_SIZE, DEVICE, COMPUTE_TYPE)
    load_time = time.time() - start_time
    logger.info(f"✅ Model loaded in {load_time:.2f}s on {model.ov_device}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

def preprocess_audio_ffmpeg(input_path: str) -> str:
    """Preprocess audio with FFmpeg"""
    import uuid
    output_path = f"/tmp/audio_{uuid.uuid4().hex}.wav"
    
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        "-f", "wav", output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise Exception(f"FFmpeg error: {result.stderr}")
    
    return output_path

def transcribe_audio(audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe audio using OpenVINO iGPU"""
    
    logger.info(f"Transcribing: {audio_path}")
    
    # Preprocess
    preprocessed = None
    try:
        preprocessed = preprocess_audio_ffmpeg(audio_path)
        audio_file = preprocessed
    except Exception as e:
        logger.warning(f"Preprocessing failed: {e}")
        audio_file = audio_path
    
    # Load audio
    audio = whisperx.load_audio(audio_file)
    duration = len(audio) / 16000
    logger.info(f"Audio duration: {duration:.1f}s")
    
    # Transcribe with OpenVINO on iGPU
    start_time = time.time()
    result = model.transcribe(audio, language=language)
    inference_time = time.time() - start_time
    
    logger.info(f"Transcribed in {inference_time:.2f}s ({duration/inference_time:.1f}x realtime)")
    
    # Cleanup
    if preprocessed:
        try:
            os.unlink(preprocessed)
        except:
            pass
    
    return {
        "text": result["text"],
        "segments": result.get("segments", []),
        "language": result.get("language", language),
        "inference_time": inference_time,
        "duration": duration,
        "device": f"Intel iGPU ({model.ov_device})",
        "speed": f"{duration/inference_time:.1f}x realtime"
    }

@app.get("/")
async def root():
    return {
        "service": "Unicorn Amanuensis - STT",
        "service_type": "Speech-to-Text (STT)",
        "version": "4.0.0",
        "backend": "Custom OpenVINO (No CTranslate2)",
        "model": f"Whisper {MODEL_SIZE}",
        "device": f"Intel iGPU ({model.ov_device})",
        "capabilities": {
            "stt": {
                "model": f"Whisper {MODEL_SIZE}",
                "hardware": "Intel iGPU via OpenVINO",
                "optimization": "Native OpenVINO IR format",
                "features": ["transcription", "100+ languages", "chunked processing"]
            }
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "device": f"Intel iGPU ({model.ov_device})",
        "compute_type": COMPUTE_TYPE,
        "hardware": {
            "openvino_available": True,
            "openvino_device": model.ov_device,
            "va_api": False,
            "opencl": False,
            "details": {}
        },
        "features": {
            "diarization": False,
            "max_speakers": 10,
            "batch_size": 16,
            "alignment": False
        }
    }

@app.get("/gpu-status")
async def gpu_status():
    """GPU status endpoint for web interface"""
    return {
        "gpu_available": True,
        "gpu_name": "Intel UHD Graphics 770",
        "device": model.ov_device,
        "backend": "OpenVINO",
        "memory": {"used": 0, "total": 0},  # OpenVINO doesn't expose memory
        "utilization": 0  # Would need intel_gpu_top for real metrics
    }

@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    """Serve the proper WhisperX web interface"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        with open(index_path, 'r') as f:
            return HTMLResponse(content=f.read())
    else:
        # Fallback simple interface
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🦄 Unicorn Amanuensis - STT</title>
            <style>
                body { font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; }
                .status { background: #f0f9ff; padding: 20px; border-radius: 8px; margin: 20px 0; }
                h1 { color: #1e40af; }
                .device { color: #059669; font-weight: bold; }
                .backend { color: #dc2626; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>🦄 Unicorn Amanuensis STT</h1>
            <div class="status">
                <h2>TRUE Intel iGPU Acceleration</h2>
                <p>Model: Whisper """ + MODEL_SIZE + """</p>
                <p>Device: <span class="device">""" + model.ov_device + """</span></p>
                <p>Backend: <span class="backend">OpenVINO (No CTranslate2)</span></p>
                <p>Status: ✅ Ready</p>
            </div>
            
            <form action="/transcribe" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="audio/*,video/*" required>
                <button type="submit">Transcribe on iGPU</button>
            </form>
        </body>
        </html>
        """)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    # Save file
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = transcribe_audio(tmp_path, language)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
        gc.collect()

@app.post("/v1/audio/transcriptions")
async def openai_compatible(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form(default="json")
):
    response = await transcribe(file, language)
    result = response.body
    
    if isinstance(result, bytes):
        import json
        result = json.loads(result)
    
    if response_format == "text":
        return result["text"]
    
    return {"text": result["text"], "model": model}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)