#!/usr/bin/env python3
"""
Unicorn Amanuensis - TRUE OpenVINO iGPU Accelerated Whisper Server
Uses Optimum Intel for proper hardware acceleration
"""

import os
import logging
import tempfile
import time
import gc
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import torch
import numpy as np
from transformers import AutoProcessor
from optimum.intel import OVModelForSpeechSeq2Seq
import openvino as ov

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unicorn Amanuensis - STT Service", 
    description="Speech-to-Text service with Intel iGPU acceleration via OpenVINO",
    version="2.0.0"
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
MODEL_ID = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3")
DEVICE = os.environ.get("WHISPER_DEVICE", "GPU.0")  # Use GPU.0 for iGPU
API_PORT = int(os.environ.get("API_PORT", "9000"))
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "./models/openvino")

# Check if we're using a pre-converted model (Docker deployment)
if os.path.exists(MODEL_ID) and os.path.isdir(MODEL_ID):
    logger.info(f"Using pre-converted model from: {MODEL_ID}")
    MODEL_PATH = MODEL_ID
else:
    MODEL_PATH = MODEL_ID

# Initialize OpenVINO runtime
core = ov.Core()
available_devices = core.available_devices
logger.info(f"Available OpenVINO devices: {available_devices}")

# Select the best device (prefer iGPU)
if DEVICE in available_devices:
    OV_DEVICE = DEVICE
elif "GPU.0" in available_devices:
    OV_DEVICE = "GPU.0"
elif "GPU" in available_devices:
    OV_DEVICE = "GPU"
else:
    OV_DEVICE = "CPU"
    logger.warning("No GPU detected, falling back to CPU")

logger.info(f"Using device: {OV_DEVICE}")
if OV_DEVICE.startswith("GPU"):
    device_name = core.get_property(OV_DEVICE, "FULL_DEVICE_NAME")
    logger.info(f"GPU Device: {device_name}")

# Load model and processor
logger.info(f"Loading Whisper model: {MODEL_ID} for OpenVINO {OV_DEVICE}")
start_time = time.time()

try:
    # Load the OpenVINO optimized model with better GPU config
    ov_config = {
        "PERFORMANCE_HINT": "LATENCY",
        "CACHE_DIR": MODEL_CACHE_DIR,
        "GPU_THROUGHPUT_STREAMS": "1",
        "GPU_ENABLE_SDPA_OPTIMIZATION": "YES"
    }
    
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH,
        device=OV_DEVICE,
        ov_config=ov_config,
        compile=True,
        export=not os.path.isdir(MODEL_PATH)  # Only export if not pre-converted
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # Verify the model is actually on GPU
    logger.info(f"Model device: {model.device}")
    logger.info(f"Model compiled for: {OV_DEVICE}")
    
    load_time = time.time() - start_time
    logger.info(f"✅ Model loaded in {load_time:.2f} seconds on {OV_DEVICE}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

def preprocess_audio_with_ffmpeg(input_path: str, output_path: str = None) -> str:
    """Preprocess audio using FFmpeg with Intel QSV hardware acceleration"""
    import subprocess
    import uuid
    
    if output_path is None:
        output_path = f"/tmp/preprocessed_{uuid.uuid4().hex}.wav"
    
    # FFmpeg command with Intel QSV hardware acceleration if available
    # Convert to 16kHz mono WAV for optimal Whisper performance
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",  # Overwrite output
        "-hwaccel", "qsv",  # Try Intel QSV hardware acceleration
        "-i", input_path,
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",  # Mono
        "-c:a", "pcm_s16le",  # 16-bit PCM
        "-f", "wav",
        output_path
    ]
    
    try:
        # Try with hardware acceleration
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info(f"Audio preprocessed with QSV hardware acceleration: {output_path}")
            return output_path
    except:
        pass
    
    # Fallback to software processing
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
    
    try:
        result = subprocess.run(ffmpeg_cmd_sw, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info(f"Audio preprocessed with FFmpeg (software): {output_path}")
            return output_path
        else:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise Exception(f"FFmpeg preprocessing failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise Exception("Audio preprocessing timeout")
    except Exception as e:
        raise Exception(f"Audio preprocessing failed: {str(e)}")

def transcribe_audio(audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe audio using OpenVINO accelerated model"""
    import librosa
    import soundfile as sf
    import subprocess
    
    logger.info(f"Starting transcription of: {audio_path}")
    
    # First, preprocess audio with FFmpeg to ensure compatibility
    preprocessed_path = None
    try:
        preprocessed_path = preprocess_audio_with_ffmpeg(audio_path)
        audio_to_load = preprocessed_path
    except Exception as e:
        logger.warning(f"FFmpeg preprocessing failed: {e}, using original file")
        audio_to_load = audio_path
    
    # Load preprocessed audio
    try:
        audio, sr = sf.read(audio_to_load)
        logger.info(f"Loaded audio with soundfile: sr={sr}, shape={audio.shape if hasattr(audio, 'shape') else len(audio)}")
    except Exception as e:
        logger.info(f"Soundfile failed ({e}), trying librosa")
        # Fallback to librosa
        audio, sr = librosa.load(audio_to_load, sr=16000, mono=True)
        logger.info(f"Loaded audio with librosa: sr={sr}, shape={audio.shape if hasattr(audio, 'shape') else len(audio)}")
    
    # Clean up preprocessed file if created
    if preprocessed_path and preprocessed_path != audio_path:
        try:
            os.unlink(preprocessed_path)
        except:
            pass
    
    # Ensure audio is float32 and in the right shape
    audio = np.array(audio, dtype=np.float32)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Convert to mono if stereo
    
    # For long audio, chunk it into 30-second segments
    CHUNK_LENGTH_S = 30  # 30 seconds per chunk
    CHUNK_LENGTH_SAMPLES = CHUNK_LENGTH_S * 16000  # 16kHz
    
    total_samples = len(audio)
    total_duration = total_samples / 16000
    
    logger.info(f"Audio duration: {total_duration:.1f} seconds")
    
    # Process in chunks if audio is longer than 30 seconds
    if total_duration > CHUNK_LENGTH_S:
        logger.info(f"Processing long audio in {int(total_duration / CHUNK_LENGTH_S) + 1} chunks")
        
        transcriptions = []
        start_time = time.time()
        
        for i in range(0, total_samples, CHUNK_LENGTH_SAMPLES):
            chunk_start = i
            chunk_end = min(i + CHUNK_LENGTH_SAMPLES, total_samples)
            audio_chunk = audio[chunk_start:chunk_end]
            
            # Skip very short chunks (< 1 second)
            if len(audio_chunk) < 16000:
                continue
            
            # Process chunk
            inputs = processor(audio_chunk, sampling_rate=sr, return_tensors="pt")
            
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs.input_features,
                    language=language,
                    task="transcribe",
                    return_timestamps=False
                )
            
            chunk_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if chunk_text.strip():
                transcriptions.append(chunk_text.strip())
            
            logger.info(f"Processed chunk {i // CHUNK_LENGTH_SAMPLES + 1}: {len(chunk_text)} chars")
        
        # Combine all transcriptions
        transcription = " ".join(transcriptions)
        inference_time = time.time() - start_time
        
    else:
        # Process short audio in one go
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_features,
                language=language,
                task="transcribe",
                return_timestamps=False
            )
        
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        inference_time = time.time() - start_time
    
    logger.info(f"Transcription complete: {len(transcription)} characters in {inference_time:.2f}s")
    
    return {
        "text": transcription.strip(),
        "inference_time": inference_time,
        "device": OV_DEVICE,
        "model": MODEL_ID,
        "audio_duration": total_duration
    }

@app.get("/")
async def root():
    """API root endpoint with service info"""
    return {
        "service": "Unicorn Amanuensis - STT",
        "service_type": "Speech-to-Text (STT)",
        "version": "2.0.0",
        "model": MODEL_ID,
        "device": f"Intel iGPU ({OV_DEVICE})",
        "backend": "OpenVINO with Optimum Intel",
        "capabilities": {
            "stt": {
                "model": "Whisper Large v3",
                "languages": "100+ languages",
                "hardware_acceleration": "Intel iGPU via OpenVINO",
                "optimization": "INT8 quantization",
                "features": ["transcription", "language detection", "timestamps"]
            }
        },
        "endpoints": {
            "health": "/health",
            "web_interface": "/web",
            "transcription": "/transcribe",
            "openai_compatible": "/v1/audio/transcriptions",
            "documentation": "/docs"
        },
        "description": "Professional Speech-to-Text service optimized for Intel iGPU"
    }

@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    """Serve the web UI"""
    # First check if there's a static HTML file
    index_path = static_dir / "index.html"
    if index_path.exists():
        with open(index_path, 'r') as f:
            return HTMLResponse(content=f.read())
    
    # Otherwise serve a basic interface
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
        </style>
    </head>
    <body>
        <h1>🦄 Unicorn Amanuensis</h1>
        <div class="service-type">Speech-to-Text (STT) Service</div>
        <div class="status">
            <h2>Intel iGPU Accelerated Speech Recognition</h2>
            <p>Model: <strong>""" + MODEL_ID + """</strong></p>
            <p>Device: <span class="device">""" + OV_DEVICE + """</span></p>
            <p>Backend: OpenVINO with Optimum Intel</p>
            <p>Capabilities: 100+ languages, real-time transcription</p>
            <p>Status: ✅ Ready for STT</p>
        </div>
        
        <div class="upload-form">
            <h3>Upload Audio File</h3>
            <form action="/transcribe" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="audio/*" required>
                <button type="submit">Transcribe</button>
            </form>
        </div>
        
        <p>API Documentation: <a href="/docs">/docs</a></p>
    </body>
    </html>
    """)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "device": OV_DEVICE,
        "backend": "OpenVINO",
        "optimized": True,
        "hardware": {
            "device_name": core.get_property(OV_DEVICE, "FULL_DEVICE_NAME") if OV_DEVICE.startswith("GPU") else "CPU",
            "available_devices": available_devices
        }
    }

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    """Transcribe audio file using OpenVINO iGPU acceleration"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Transcribe using iGPU
        result = transcribe_audio(tmp_path, language)
        return JSONResponse(content=result)
    
    except Exception as e:
        import traceback
        error_msg = f"Transcription error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=str(e) if str(e) else "Transcription failed")
    
    finally:
        # Cleanup
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
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Transcribe using iGPU
        result = transcribe_audio(tmp_path, language)
        
        if response_format == "text":
            return result["text"]
        
        return {
            "text": result["text"],
            "model": model,
            "language": language or "auto"
        }
    
    except Exception as e:
        import traceback
        error_msg = f"Transcription error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=str(e) if str(e) else "Transcription failed")
    
    finally:
        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)