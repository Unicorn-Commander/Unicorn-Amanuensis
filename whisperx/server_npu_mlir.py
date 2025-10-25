#!/usr/bin/env python3
"""
Unicorn Amanuensis - NPU-Accelerated STT Server
Uses custom MLIR-AIE2 runtime for AMD Phoenix NPU
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import tempfile
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
import time

# Add NPU runtime to path
npu_path = Path(__file__).parent / "npu"
sys.path.insert(0, str(npu_path))

# Import custom NPU runtime
from npu_runtime_aie2 import NPURuntime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unicorn Amanuensis - NPU STT Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"✅ Mounted static files from {static_dir}")

# Mount templates directory
templates_dir = Path(__file__).parent / "templates"
if templates_dir.exists():
    logger.info(f"✅ Templates directory found: {templates_dir}")

# Model configuration
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
USE_NPU = os.environ.get("USE_NPU", "true").lower() in ("true", "1", "yes")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Initialize NPU Runtime
logger.info("🚀 Initializing Unicorn Custom MLIR-AIE2 NPU Runtime...")
npu_runtime = NPURuntime()

if npu_runtime.is_available() and USE_NPU:
    logger.info("✅ NPU detected - using custom MLIR-AIE2 acceleration")

    # Load model
    model_path = os.environ.get('WHISPER_NPU_MODEL_PATH', f'/models/whisper-{MODEL_SIZE}-amd-npu-int8')
    if npu_runtime.load_model(model_path):
        logger.info(f"✅ Loaded NPU model: {MODEL_SIZE}")
    else:
        logger.warning("⚠️ NPU model loading failed, falling back to CPU")
        USE_NPU = False
else:
    logger.warning("⚠️ NPU not available - using CPU fallback")
    USE_NPU = False

# Fallback to standard WhisperX if NPU not available
if not USE_NPU:
    import whisperx
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")

    # Use standard Whisper model names for CPU fallback (NOT magicunicorn placeholders!)
    CPU_MODEL_SIZE = MODEL_SIZE if MODEL_SIZE in ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"] else "base"

    logger.info(f"Loading WhisperX fallback: {CPU_MODEL_SIZE} on {DEVICE}")
    model = whisperx.load_model(CPU_MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
    model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Form(False),
    min_speakers: int = Form(None),
    max_speakers: int = Form(None)
):
    """Transcribe audio file with NPU acceleration"""

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"🎙️ Processing audio file: {file.filename}")

        # Get audio info
        audio_data, sr = sf.read(tmp_path)
        duration = len(audio_data) / sr
        logger.info(f"📊 Audio duration: {duration:.1f}s, sample rate: {sr}Hz")

        start_time = time.time()

        # Transcribe with NPU or fallback
        if USE_NPU and npu_runtime.is_available():
            logger.info("⚡ Using NPU Custom MLIR-AIE2 Runtime")
            result = npu_runtime.transcribe(tmp_path)

            # Extract results
            text = result.get("text", "")
            segments = result.get("segments", [])
            processing_time = result.get("processing_time", time.time() - start_time)
            npu_accelerated = result.get("npu_accelerated", True)

            logger.info(f"✅ NPU transcription complete: {processing_time:.2f}s")
            logger.info(f"⚡ Real-time factor: {processing_time/duration:.4f}")
            logger.info(f"🚀 Speedup: {duration/processing_time:.1f}x realtime")

        else:
            logger.info("🖥️ Using CPU fallback (WhisperX)")

            # Load audio with WhisperX
            audio = whisperx.load_audio(tmp_path)

            # Transcribe
            if duration > 60:
                logger.info(f"🔪 Long audio ({duration:.1f}s), using chunked processing...")
                result = _transcribe_chunked(audio, audio_data, sr)
            else:
                result = model.transcribe(audio, batch_size=BATCH_SIZE)

            # Align
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE)

            # Diarization
            if diarize and HF_TOKEN:
                logger.info("👥 Performing speaker diarization...")
                diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
                diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
                result = whisperx.assign_word_speakers(diarize_segments, result)

            segments = result.get("segments", [])
            text = " ".join([seg["text"] for seg in segments])
            processing_time = time.time() - start_time
            npu_accelerated = False

        return {
            "text": text,
            "segments": segments,
            "language": result.get("language", "en"),
            "words": result.get("word_segments", result.get("words", [])),
            "processing_time": processing_time,
            "audio_duration": duration,
            "real_time_factor": processing_time / duration if duration > 0 else 0,
            "speedup": duration / processing_time if processing_time > 0 else 0,
            "npu_accelerated": npu_accelerated,
            "device_info": result.get("device_info", {"status": "cpu_fallback"})
        }

    except Exception as e:
        logger.error(f"❌ Error processing audio: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    finally:
        os.unlink(tmp_path)
        if not USE_NPU:
            import gc
            gc.collect()


def _transcribe_chunked(audio, audio_data, sr):
    """Chunk processing for WhisperX fallback"""
    chunk_length = 30
    chunk_size = chunk_length * sr
    n_chunks = int(np.ceil(len(audio_data) / chunk_size))

    all_segments = []

    for i in range(n_chunks):
        chunk_start = i * chunk_size
        chunk_end = min((i + 1) * chunk_size, len(audio_data))
        audio_chunk = audio_data[chunk_start:chunk_end]
        time_offset = chunk_start / sr

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_chunk:
            sf.write(tmp_chunk.name, audio_chunk, sr)
            chunk_path = tmp_chunk.name

        try:
            chunk_audio = whisperx.load_audio(chunk_path)
            chunk_result = model.transcribe(chunk_audio, batch_size=BATCH_SIZE)

            for segment in chunk_result.get("segments", []):
                segment["start"] += time_offset
                segment["end"] += time_offset
                all_segments.append(segment)
        finally:
            os.unlink(chunk_path)

    return {"segments": all_segments, "language": "en"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    device_info = npu_runtime.get_device_info() if USE_NPU else {"status": "cpu_fallback"}

    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "npu_enabled": USE_NPU,
        "npu_available": npu_runtime.is_available() if USE_NPU else False,
        "device_info": device_info,
        "runtime": "Custom MLIR-AIE2" if USE_NPU else "WhisperX CPU"
    }


@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Serve the web UI"""
    template_path = templates_dir / "index.html"

    if template_path.exists():
        return FileResponse(template_path)

    # Fallback to static
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    # Fallback to API info
    return JSONResponse({
        "service": "Unicorn Amanuensis - NPU STT",
        "version": "1.0",
        "npu_runtime": "Custom MLIR-AIE2",
        "endpoints": {
            "/v1/audio/transcriptions": "POST - Transcribe audio (NPU accelerated)",
            "/health": "GET - Health check",
            "/api": "GET - API information",
            "/models": "GET - List available models",
            "/npu/status": "GET - NPU device status"
        }
    })


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "service": "Unicorn Amanuensis",
        "version": "1.0",
        "model": MODEL_SIZE,
        "npu_enabled": USE_NPU,
        "npu_runtime": "Custom MLIR-AIE2" if USE_NPU else "None",
        "endpoints": {
            "/": "GET - Web interface",
            "/v1/audio/transcriptions": "POST - Transcribe audio",
            "/health": "GET - Health check",
            "/models": "GET - List available models",
            "/npu/status": "GET - NPU device status"
        }
    }


@app.get("/models")
async def list_models():
    """List available NPU models"""
    models_dir = Path(__file__).parent / "models"
    available_models = []

    # Check if NPU device exists
    npu_available = Path("/dev/accel/accel0").exists()
    device_type = "AMD Phoenix NPU (Custom MLIR-AIE2)" if npu_available else "CPU"

    # Scan for NPU models
    if models_dir.exists():
        for model_path in models_dir.iterdir():
            if model_path.is_dir() and ("npu" in model_path.name.lower() or "amd" in model_path.name.lower()):
                model_name = model_path.name

                if "large-v3" in model_name:
                    available_models.append({
                        "id": "large-v3",
                        "name": "Large v3 (Most Accurate)",
                        "device": device_type,
                        "path": str(model_path),
                        "quantization": "INT8",
                        "runtime": "Custom MLIR-AIE2"
                    })
                elif "base" in model_name:
                    available_models.append({
                        "id": "base",
                        "name": "Base (Balanced)",
                        "device": device_type,
                        "path": str(model_path),
                        "quantization": "INT8",
                        "runtime": "Custom MLIR-AIE2"
                    })

    if not available_models:
        available_models = [{
            "id": "base",
            "name": "Base (Default)",
            "device": device_type,
            "quantization": "INT8",
            "runtime": "Custom MLIR-AIE2" if npu_available else "CPU"
        }]

    return {
        "models": available_models,
        "device": "AMD Phoenix NPU" if npu_available else "CPU",
        "runtime": "Custom MLIR-AIE2" if npu_available else "WhisperX",
        "backend": "MLIR-AIE2" if npu_available else "PyTorch"
    }


@app.get("/npu/status")
async def npu_status():
    """Get detailed NPU device status"""
    if not USE_NPU:
        return {
            "enabled": False,
            "available": False,
            "reason": "NPU disabled or not available"
        }

    device_info = npu_runtime.get_device_info()

    return {
        "enabled": True,
        "available": npu_runtime.is_available(),
        "device_path": device_info.get("device_path", "/dev/accel/accel0"),
        "model_loaded": device_info.get("model_loaded", False),
        "aie2_driver": device_info.get("aie2_driver", False),
        "direct_runtime": device_info.get("direct_runtime", False),
        "mlir_kernels": device_info.get("mlir_kernels", "None"),
        "status": device_info.get("status", "unknown"),
        "runtime_version": "Custom MLIR-AIE2 v1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", "9000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
