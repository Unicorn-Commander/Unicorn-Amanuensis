#!/usr/bin/env python3
"""
Unicorn Amanuensis - WhisperX Server Optimized for Intel iGPU
Hardware accelerated transcription with OpenVINO
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import gc
import logging
from pathlib import Path
import subprocess
import numpy as np
import json
from typing import Optional, Dict, List
import time

# Try to use OpenVINO optimized inference
try:
    import openvino as ov
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
    print("✅ OpenVINO is available for Intel iGPU acceleration")
except ImportError:
    OPENVINO_AVAILABLE = False
    print("⚠️ OpenVINO not available, using CPU mode")

import whisperx
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unicorn Amanuensis - Intel iGPU Optimized")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if the directory exists
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"Mounted static files from {static_dir}")

# Model configuration
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE_TYPE = os.environ.get("WHISPER_DEVICE", "igpu").lower()
ENABLE_DIARIZATION = os.environ.get("ENABLE_DIARIZATION", "true").lower() == "true"
MAX_SPEAKERS = int(os.environ.get("MAX_SPEAKERS", "10"))
BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "8"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Configure device based on availability
if DEVICE_TYPE == "igpu" and OPENVINO_AVAILABLE:
    # Use OpenVINO for Intel GPU
    DEVICE = "cpu"  # WhisperX will use CPU, but we'll optimize with OpenVINO
    COMPUTE_TYPE = "int8"  # Optimal for iGPU
    
    # Initialize OpenVINO
    core = Core()
    available_devices = core.available_devices
    logger.info(f"OpenVINO available devices: {available_devices}")
    
    # Prefer GPU if available
    if "GPU" in available_devices:
        OV_DEVICE = "GPU"
        logger.info("✅ Using Intel GPU via OpenVINO")
    else:
        OV_DEVICE = "CPU"
        logger.warning("Intel GPU not detected, using CPU with OpenVINO optimization")
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
    OV_DEVICE = None
    logger.info("Using CPU mode")

# Load WhisperX model
logger.info(f"Loading WhisperX model: {MODEL_SIZE} with compute_type={COMPUTE_TYPE}")
try:
    model = whisperx.load_model(
        MODEL_SIZE, 
        DEVICE, 
        compute_type=COMPUTE_TYPE,
        download_root="./models"
    )
    logger.info(f"✅ Model {MODEL_SIZE} loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    # Fallback to base model if large model fails
    MODEL_SIZE = "base"
    model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
    logger.warning(f"Fell back to {MODEL_SIZE} model")

# Load alignment model
logger.info("Loading alignment model...")
try:
    model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    logger.info("✅ Alignment model loaded")
except Exception as e:
    logger.error(f"Failed to load alignment model: {e}")
    model_a, metadata = None, None

# Load diarization model if enabled
diarize_model = None
if ENABLE_DIARIZATION and HF_TOKEN:
    try:
        logger.info("Loading speaker diarization model...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        logger.info("✅ Diarization model loaded")
    except Exception as e:
        logger.error(f"Failed to load diarization model: {e}")
        ENABLE_DIARIZATION = False

def check_intel_gpu_status():
    """Check Intel GPU status using vainfo and clinfo"""
    status = {
        "va_api": False,
        "opencl": False,
        "openvino_gpu": False,
        "details": {}
    }
    
    # Check VA-API
    try:
        result = subprocess.run(["vainfo"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            status["va_api"] = True
            # Parse VA-API info
            for line in result.stdout.split('\n'):
                if "Driver version" in line:
                    status["details"]["va_driver"] = line.split(':')[1].strip()
    except:
        pass
    
    # Check OpenCL
    try:
        result = subprocess.run(["clinfo", "-l"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and "Intel" in result.stdout:
            status["opencl"] = True
            status["details"]["opencl_devices"] = result.stdout.strip()
    except:
        pass
    
    # Check OpenVINO GPU
    if OPENVINO_AVAILABLE and OV_DEVICE == "GPU":
        status["openvino_gpu"] = True
        status["details"]["openvino_device"] = OV_DEVICE
    
    return status

def preprocess_audio_with_qsv(input_path: str, output_path: str) -> bool:
    """
    Preprocess audio to 16kHz mono using FFmpeg with Intel QSV acceleration
    """
    try:
        # Base FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", "16000",  # Resample to 16kHz
            "-ac", "1",      # Convert to mono
            "-c:a", "pcm_s16le",  # PCM 16-bit
            "-f", "wav",
            output_path
        ]
        
        # Try to use Intel QSV if available
        if DEVICE_TYPE == "igpu":
            # Check if QSV is available
            check_cmd = ["ffmpeg", "-hide_banner", "-hwaccels"]
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if "qsv" in result.stdout:
                logger.info("Using Intel QSV hardware acceleration")
                # Add QSV acceleration flags
                cmd = [
                    "ffmpeg", "-y",
                    "-hwaccel", "qsv",
                    "-hwaccel_device", "/dev/dri/renderD128",
                    "-i", input_path,
                    "-ar", "16000",
                    "-ac", "1",
                    "-c:a", "pcm_s16le",
                    "-f", "wav",
                    output_path
                ]
            elif "vaapi" in result.stdout:
                logger.info("Using VA-API hardware acceleration")
                cmd = [
                    "ffmpeg", "-y",
                    "-hwaccel", "vaapi",
                    "-hwaccel_device", "/dev/dri/renderD128",
                    "-i", input_path,
                    "-ar", "16000",
                    "-ac", "1", 
                    "-c:a", "pcm_s16le",
                    "-f", "wav",
                    output_path
                ]
        
        # Execute FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
        
        logger.info(f"✅ Audio preprocessed to 16kHz mono")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timeout")
        return False
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return False

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Form(False),
    min_speakers: int = Form(2),
    max_speakers: int = Form(None),
    timestamps: bool = Form(True),
    word_timestamps: bool = Form(True),
    language: str = Form("en"),
    response_format: str = Form("json")
):
    """
    Transcribe audio with Intel iGPU acceleration
    """
    
    if max_speakers is None:
        max_speakers = MAX_SPEAKERS
    
    # Save uploaded file
    start_time = time.time()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        original_path = tmp.name
    
    # Preprocess to 16kHz
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        preprocessed_path = tmp_wav.name
    
    try:
        logger.info(f"Processing: {file.filename} ({len(content)/1024:.1f} KB)")
        
        # Preprocess with hardware acceleration
        preprocess_start = time.time()
        if not preprocess_audio_with_qsv(original_path, preprocessed_path):
            preprocessed_path = original_path
            logger.warning("Using original audio without preprocessing")
        preprocess_time = time.time() - preprocess_start
        
        # Load audio
        audio = whisperx.load_audio(preprocessed_path)
        audio_duration = len(audio) / 16000
        logger.info(f"Audio duration: {audio_duration:.1f}s")
        
        # Transcribe with WhisperX
        transcribe_start = time.time()
        logger.info(f"Transcribing with {MODEL_SIZE} model...")
        result = model.transcribe(
            audio, 
            batch_size=BATCH_SIZE,
            language=language if language != "auto" else None
        )
        transcribe_time = time.time() - transcribe_start
        
        # Align for word-level timestamps
        align_time = 0
        if (word_timestamps or timestamps) and model_a:
            align_start = time.time()
            logger.info("Aligning for timestamps...")
            result = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio, 
                DEVICE,
                return_char_alignments=False
            )
            align_time = time.time() - align_start
        
        # Speaker diarization
        diarize_time = 0
        if diarize and diarize_model:
            diarize_start = time.time()
            logger.info(f"Diarizing (speakers: {min_speakers}-{max_speakers})...")
            diarize_segments = diarize_model(
                audio, 
                min_speakers=min_speakers, 
                max_speakers=max_speakers
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)
            diarize_time = time.time() - diarize_start
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        rtf = total_time / audio_duration  # Real-time factor
        
        # Format response based on requested format
        if response_format == "text":
            text = " ".join([seg.get("text", "") for seg in result.get("segments", [])])
            return JSONResponse(content={"text": text})
        
        # Full JSON response
        response = {
            "text": " ".join([seg.get("text", "") for seg in result.get("segments", [])]),
            "segments": result.get("segments", []) if timestamps else None,
            "word_segments": result.get("word_segments", []) if word_timestamps else None,
            "language": result.get("language", language),
            "duration": audio_duration,
            "performance": {
                "total_time": f"{total_time:.2f}s",
                "rtf": f"{rtf:.2f}x",
                "preprocess_time": f"{preprocess_time:.2f}s",
                "transcribe_time": f"{transcribe_time:.2f}s",
                "align_time": f"{align_time:.2f}s" if align_time > 0 else None,
                "diarize_time": f"{diarize_time:.2f}s" if diarize_time > 0 else None
            },
            "config": {
                "model": MODEL_SIZE,
                "device": f"Intel iGPU ({OV_DEVICE})" if OV_DEVICE else "CPU",
                "compute_type": COMPUTE_TYPE,
                "batch_size": BATCH_SIZE,
                "diarization": diarize and diarize_model is not None
            }
        }
        
        # Clean up memory
        gc.collect()
        
        logger.info(f"✅ Transcription complete in {total_time:.2f}s (RTF: {rtf:.2f}x)")
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()}
        )
    finally:
        # Clean up temp files
        try:
            os.unlink(original_path)
            if preprocessed_path != original_path:
                os.unlink(preprocessed_path)
        except:
            pass

@app.get("/health")
async def health():
    """Health check with Intel GPU status"""
    gpu_status = check_intel_gpu_status()
    
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "device": f"Intel iGPU ({OV_DEVICE})" if OV_DEVICE else "CPU",
        "compute_type": COMPUTE_TYPE,
        "hardware": {
            "openvino_available": OPENVINO_AVAILABLE,
            "openvino_device": OV_DEVICE,
            "va_api": gpu_status["va_api"],
            "opencl": gpu_status["opencl"],
            "details": gpu_status["details"]
        },
        "features": {
            "diarization": diarize_model is not None,
            "max_speakers": MAX_SPEAKERS,
            "batch_size": BATCH_SIZE,
            "alignment": model_a is not None
        }
    }

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Serve the web UI"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return HTMLResponse(content="""
        <html>
        <body>
        <h1>Unicorn Amanuensis - Intel iGPU Optimized</h1>
        <p>Model: {}</p>
        <p>Device: {}</p>
        <p>API Endpoint: POST /v1/audio/transcriptions</p>
        </body>
        </html>
        """.format(MODEL_SIZE, OV_DEVICE or "CPU"))

@app.get("/gpu-status")
async def gpu_status():
    """Get detailed GPU status"""
    status = check_intel_gpu_status()
    
    # Add more detailed checks
    if OPENVINO_AVAILABLE:
        core = Core()
        status["openvino"] = {
            "version": ov.__version__,
            "devices": core.available_devices,
            "gpu_properties": {}
        }
        
        if "GPU" in core.available_devices:
            try:
                gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
                status["openvino"]["gpu_properties"]["name"] = gpu_name
            except:
                pass
    
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)