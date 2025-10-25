from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import whisperx
import os
import tempfile
import torch
import gc
import logging
from pathlib import Path
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhisperX STT Service")

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
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Load model once at startup
logger.info(f"Loading WhisperX model: {MODEL_SIZE} on {DEVICE}")
model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)

# Load alignment model for English
logger.info("Loading alignment model...")
model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Form(False),
    min_speakers: int = Form(None),
    max_speakers: int = Form(None)
):
    """Transcribe audio file with optional speaker diarization"""
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        logger.info(f"Processing audio file: {file.filename}")

        # Load audio
        audio = whisperx.load_audio(tmp_path)

        # Get audio duration
        audio_data, sr = sf.read(tmp_path)
        duration = len(audio_data) / sr
        logger.info(f"Audio duration: {duration:.1f}s")

        # Transcribe with WhisperX
        # Use chunking for long audio (>60 seconds) to avoid memory issues
        if duration > 60:
            logger.info(f"🔪 Long audio detected ({duration:.1f}s), using chunked processing...")
            chunk_length = 30  # 30 seconds per chunk
            chunk_size = chunk_length * sr
            n_chunks = int(np.ceil(len(audio_data) / chunk_size))
            logger.info(f"📦 Processing in {n_chunks} chunks of {chunk_length}s each")

            all_segments = []

            for i in range(n_chunks):
                chunk_start = i * chunk_size
                chunk_end = min((i + 1) * chunk_size, len(audio_data))
                audio_chunk = audio_data[chunk_start:chunk_end]
                chunk_duration = len(audio_chunk) / sr
                time_offset = chunk_start / sr

                logger.info(f"🎯 Transcribing chunk {i+1}/{n_chunks} ({chunk_duration:.1f}s, offset: {time_offset:.1f}s)...")

                # Save chunk to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_chunk:
                    sf.write(tmp_chunk.name, audio_chunk, sr)
                    chunk_path = tmp_chunk.name

                try:
                    # Load chunk audio for WhisperX
                    chunk_audio = whisperx.load_audio(chunk_path)

                    # Transcribe chunk
                    chunk_result = model.transcribe(chunk_audio, batch_size=BATCH_SIZE)

                    # Adjust timestamps to account for chunk offset
                    for segment in chunk_result.get("segments", []):
                        segment["start"] += time_offset
                        segment["end"] += time_offset
                        all_segments.append(segment)

                    logger.info(f"✅ Chunk {i+1}/{n_chunks} done: {len(chunk_result.get('segments', []))} segments")

                finally:
                    # Clean up chunk file
                    os.unlink(chunk_path)

            # Combine all segments
            result = {
                "segments": all_segments,
                "language": "en"
            }
            logger.info(f"✅ All chunks processed: {len(all_segments)} total segments")
        else:
            # Short audio - process entire file at once
            logger.info("Transcribing...")
            result = model.transcribe(audio, batch_size=BATCH_SIZE)

        # Align whisper output
        logger.info("Aligning...")
        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE)
        
        # Optional: Speaker diarization
        if diarize and HF_TOKEN:
            logger.info("Performing speaker diarization...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
            diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Format response
        text = " ".join([segment["text"] for segment in result["segments"]])
        
        return {
            "text": text,
            "segments": result["segments"],
            "language": result.get("language", "en"),
            "words": result.get("word_segments", [])
        }
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise
        
    finally:
        os.unlink(tmp_path)
        gc.collect()

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE
    }

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Serve the web UI"""
    # Try templates directory first (updated UI)
    templates_dir = Path(__file__).parent / "templates"
    template_path = templates_dir / "index.html"

    if template_path.exists():
        return FileResponse(template_path)

    # Fallback to static directory
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        # Fallback to API info if no UI available
        return JSONResponse({
            "service": "WhisperX STT",
            "version": "1.0",
            "endpoints": {
                "/v1/audio/transcriptions": "POST - Transcribe audio",
                "/health": "GET - Health check",
                "/api": "GET - API information",
                "/models": "GET - List available models"
            }
        })

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "service": "WhisperX STT",
        "version": "1.0",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "endpoints": {
            "/": "GET - Web interface",
            "/v1/audio/transcriptions": "POST - Transcribe audio",
            "/health": "GET - Health check",
            "/models": "GET - List available models"
        }
    }

@app.get("/models")
async def list_models():
    """List available NPU models"""
    models_dir = Path(__file__).parent / "models"
    available_models = []

    # Check if NPU device exists
    npu_available = Path("/dev/accel/accel0").exists()
    device_type = "AMD Phoenix NPU (Bare-Metal)" if npu_available else "CPU/GPU"

    # Scan for NPU models
    if models_dir.exists():
        for model_path in models_dir.iterdir():
            if model_path.is_dir() and "npu" in model_path.name.lower():
                # Extract model name and size
                model_name = model_path.name
                if "whisperx-large-v3-npu" in model_name or "large-v3" in model_name:
                    available_models.append({
                        "id": "large-v3",
                        "name": "Large v3 (Most Accurate)",
                        "device": device_type,
                        "path": str(model_path),
                        "quantization": "INT8"
                    })
                elif "whisper-base-amd-npu-int8" in model_name or "base" in model_name:
                    available_models.append({
                        "id": "base",
                        "name": "Base (Balanced)",
                        "device": device_type,
                        "path": str(model_path),
                        "quantization": "INT8"
                    })
                elif "medium" in model_name:
                    available_models.append({
                        "id": "medium",
                        "name": "Medium",
                        "device": device_type,
                        "path": str(model_path),
                        "quantization": "INT8"
                    })
                elif "small" in model_name:
                    available_models.append({
                        "id": "small",
                        "name": "Small",
                        "device": device_type,
                        "path": str(model_path),
                        "quantization": "INT8"
                    })
                elif "tiny" in model_name:
                    available_models.append({
                        "id": "tiny",
                        "name": "Tiny (Fastest)",
                        "device": device_type,
                        "path": str(model_path),
                        "quantization": "INT8"
                    })

    # If no NPU models found, return defaults based on what's available
    if not available_models:
        if npu_available:
            available_models = [
                {
                    "id": "large-v3",
                    "name": "Large v3 (Most Accurate)",
                    "device": "AMD Phoenix NPU (Bare-Metal)",
                    "quantization": "INT8"
                },
                {
                    "id": "base",
                    "name": "Base (Balanced)",
                    "device": "AMD Phoenix NPU (Bare-Metal)",
                    "quantization": "INT8"
                }
            ]
        else:
            # Fallback for CPU/GPU
            available_models = [
                {
                    "id": "base",
                    "name": "Base",
                    "device": DEVICE.upper(),
                    "quantization": COMPUTE_TYPE
                }
            ]

    return {
        "models": available_models,
        "device": "AMD Phoenix NPU" if npu_available else DEVICE.upper(),
        "runtime": "Bare-Metal (Native)" if npu_available else "WhisperX",
        "backend": "XDNA1" if npu_available else "PyTorch"
    }
