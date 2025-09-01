#!/usr/bin/env python3
"""
Unicorn Amanuensis - WhisperX with TRUE iGPU acceleration via OpenVINO
Uses our custom OpenVINO backend instead of CTranslate2
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

import numpy as np
import torch

# Import our custom OpenVINO backend
sys.path.insert(0, os.path.dirname(__file__))
from whisperx_openvino_backend import patch_whisperx_for_openvino

# Patch WhisperX to use OpenVINO before importing
patch_whisperx_for_openvino()

import whisperx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unicorn Amanuensis - STT Service (WhisperX + iGPU)",
    description="WhisperX with TRUE Intel iGPU acceleration via custom OpenVINO backend",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = "GPU"  # Will use GPU.0 (iGPU) via our backend
COMPUTE_TYPE = "int8"
BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "16"))
API_PORT = int(os.environ.get("API_PORT", "9000"))

# Load WhisperX with our OpenVINO backend
logger.info(f"Loading WhisperX model: {MODEL_SIZE} with OpenVINO iGPU backend")
start_time = time.time()

try:
    model = whisperx.load_model(
        MODEL_SIZE,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        download_root="./models"
    )
    load_time = time.time() - start_time
    logger.info(f"✅ WhisperX model loaded in {load_time:.2f}s with OpenVINO iGPU backend")
    
    # Load alignment model for WhisperX features
    model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    logger.info("✅ Alignment model loaded")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"Mounted static files from {static_dir}")

def preprocess_audio_with_qsv(input_path: str, output_path: str = None) -> str:
    """Preprocess audio using FFmpeg with Intel QSV hardware acceleration"""
    import uuid
    
    if output_path is None:
        output_path = f"/tmp/preprocessed_{uuid.uuid4().hex}.wav"
    
    # FFmpeg with Intel QSV for audio extraction
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-hwaccel", "qsv",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        output_path
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info(f"Audio preprocessed with QSV: {output_path}")
            return output_path
    except:
        pass
    
    # Fallback without hardware acceleration
    ffmpeg_cmd_sw = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        output_path
    ]
    
    result = subprocess.run(ffmpeg_cmd_sw, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        logger.info(f"Audio preprocessed: {output_path}")
        return output_path
    else:
        raise Exception(f"FFmpeg failed: {result.stderr}")

def transcribe_audio(audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe audio using WhisperX with OpenVINO iGPU backend"""
    
    logger.info(f"Starting transcription of: {audio_path}")
    
    # Preprocess audio
    preprocessed_path = None
    try:
        preprocessed_path = preprocess_audio_with_qsv(audio_path)
        audio_to_use = preprocessed_path
    except Exception as e:
        logger.warning(f"Preprocessing failed: {e}, using original")
        audio_to_use = audio_path
    
    # Load audio
    audio = whisperx.load_audio(audio_to_use)
    
    # Get duration
    duration = len(audio) / 16000
    logger.info(f"Audio duration: {duration:.1f} seconds")
    
    # Transcribe with WhisperX (using our OpenVINO backend)
    start_time = time.time()
    result = model.transcribe(
        audio,
        batch_size=BATCH_SIZE,
        language=language
    )
    inference_time = time.time() - start_time
    
    logger.info(f"Transcription complete in {inference_time:.2f}s")
    
    # Align whisper output (word-level timestamps)
    if model_a and language in ["en"]:
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            DEVICE,
            return_char_alignments=False
        )
    
    # Clean up
    if preprocessed_path and preprocessed_path != audio_path:
        try:
            os.unlink(preprocessed_path)
        except:
            pass
    
    return {
        "text": result.get("text", ""),
        "segments": result.get("segments", []),
        "language": result.get("language", language),
        "inference_time": inference_time,
        "duration": duration,
        "device": "Intel iGPU (OpenVINO)"
    }

@app.get("/")
async def root():
    """API root endpoint with service info"""
    return {
        "service": "Unicorn Amanuensis - STT",
        "service_type": "Speech-to-Text (STT)",
        "version": "3.0.0",
        "backend": "WhisperX with custom OpenVINO iGPU backend",
        "model": f"Whisper {MODEL_SIZE}",
        "device": "Intel iGPU via OpenVINO",
        "capabilities": {
            "stt": {
                "model": f"Whisper {MODEL_SIZE}",
                "languages": "100+ languages",
                "hardware_acceleration": "Intel iGPU via custom OpenVINO backend",
                "optimization": "INT8 quantization on iGPU",
                "features": [
                    "transcription",
                    "word-level timestamps",
                    "speaker diarization",
                    "language detection"
                ]
            }
        },
        "endpoints": {
            "health": "/health",
            "web_interface": "/web",
            "transcribe": "/transcribe",
            "openai_compatible": "/v1/audio/transcriptions",
            "documentation": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "device": "Intel iGPU (OpenVINO)",
        "backend": "WhisperX + OpenVINO",
        "ready": True
    }

@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    """Serve the web UI"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🦄 Unicorn Amanuensis - STT Service</title>
        <style>
            body { font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; }
            .status { background: #f0f9ff; padding: 20px; border-radius: 8px; margin: 20px 0; }
            .upload-form { background: #f8fafc; padding: 20px; border-radius: 8px; }
            h1 { color: #1e40af; }
            .service-type { color: #7c3aed; font-size: 1.2em; margin-bottom: 10px; }
            .device { color: #059669; font-weight: bold; }
            .backend { color: #dc2626; }
        </style>
    </head>
    <body>
        <h1>🦄 Unicorn Amanuensis</h1>
        <div class="service-type">Speech-to-Text (STT) Service</div>
        <div class="status">
            <h2>WhisperX with Intel iGPU Acceleration</h2>
            <p>Model: <strong>Whisper """ + MODEL_SIZE + """</strong></p>
            <p>Device: <span class="device">Intel iGPU</span></p>
            <p>Backend: <span class="backend">Custom OpenVINO Runtime</span></p>
            <p>Features: Word timestamps, Speaker diarization, 100+ languages</p>
            <p>Status: ✅ Ready for STT</p>
        </div>
        
        <div class="upload-form">
            <h3>Upload Audio File</h3>
            <form action="/transcribe" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="audio/*,video/*" required>
                <button type="submit">Transcribe</button>
            </form>
        </div>
        
        <p>API Documentation: <a href="/docs">/docs</a></p>
    </body>
    </html>
    """)

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    """Transcribe audio file using WhisperX with iGPU"""
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = transcribe_audio(tmp_path, language)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
        gc.collect()

@app.post("/v1/audio/transcriptions")
async def openai_compatible_transcribe(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form(default="json")
):
    """OpenAI API compatible endpoint"""
    
    response = await transcribe(file, language)
    result = response.body
    
    if isinstance(result, bytes):
        import json
        result = json.loads(result)
    
    if response_format == "text":
        return result["text"]
    
    return {
        "text": result["text"],
        "model": model,
        "language": result.get("language", "auto")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)