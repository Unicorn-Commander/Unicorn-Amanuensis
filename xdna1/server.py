#!/usr/bin/env python3
"""
Production XDNA1 Server with Web GUI
=====================================

Integrates:
- WhisperXDNA1Runtime (sign-fixed mel kernel for 23.6× realtime)
- FastAPI web service
- Professional web GUI with hardware status
- NPU fallback to CPU WhisperX
- Performance monitoring

Author: Magic Unicorn Unconventional Technology & Stuff Inc.
Date: November 2025
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import tempfile
import gc
import logging
import time
from pathlib import Path
from typing import Optional

# Import XDNA1 runtime
from xdna1.runtime.whisper_xdna1_runtime import WhisperXDNA1Runtime, create_runtime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Unicorn-Amanuensis XDNA1 (Phoenix NPU)",
    description="Production speech-to-text service with AMD Phoenix XDNA1 NPU acceleration",
    version="1.0.0"
)

# Server configuration
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
DEVICE_FALLBACK = os.environ.get("FALLBACK_TO_CPU", "true").lower() == "true"
ENABLE_DIARIZATION = os.environ.get("ENABLE_DIARIZATION", "false").lower() == "true"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Initialize XDNA1 runtime (with CPU fallback)
logger.info("="*70)
logger.info("Initializing XDNA1 Runtime with Sign-Fixed Mel Kernel")
logger.info("="*70)

try:
    runtime = create_runtime(
        model_size=MODEL_SIZE,
        fallback_to_cpu=DEVICE_FALLBACK
    )
    logger.info("✅ XDNA1 Runtime initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize XDNA1 runtime: {e}")
    if not DEVICE_FALLBACK:
        raise
    logger.warning("Falling back to CPU WhisperX")
    runtime = None

# Get static directory path
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"Mounted static files from {static_dir}")
else:
    logger.warning(f"Static directory not found at {static_dir}")

# Performance tracking
class ServerStats:
    def __init__(self):
        self.transcriptions = 0
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0
        self.npu_transcriptions = 0
        self.cpu_fallback_count = 0

    def to_dict(self):
        avg_realtime = 0.0
        if self.total_processing_time > 0:
            avg_realtime = self.total_audio_duration / self.total_processing_time

        return {
            "transcriptions": self.transcriptions,
            "total_audio_duration": round(self.total_audio_duration, 2),
            "total_processing_time": round(self.total_processing_time, 2),
            "avg_realtime_factor": round(avg_realtime, 1),
            "npu_transcriptions": self.npu_transcriptions,
            "cpu_fallback_count": self.cpu_fallback_count
        }

server_stats = ServerStats()

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API documentation"""
    return {
        "service": "Unicorn-Amanuensis XDNA1 (Phoenix NPU)",
        "version": "1.0.0",
        "hardware": "AMD Phoenix XDNA1 NPU with Sign-Fixed Mel Kernel",
        "performance": "23.6× realtime (Whisper Base)",
        "endpoints": {
            "/web": "GET - Web GUI",
            "/v1/audio/transcriptions": "POST - OpenAI-compatible transcription",
            "/status": "GET - Hardware and service status",
            "/health": "GET - Health check",
            "/stats": "GET - Server statistics"
        }
    }

@app.get("/web")
async def web_gui():
    """Serve web GUI"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="Web GUI not found")

@app.get("/status")
async def status():
    """Get hardware and service status"""
    hardware_info = {
        "type": "npu" if (runtime and runtime.mel_processor and runtime.mel_processor.npu_available) else "cpu",
        "name": "AMD Phoenix XDNA1 NPU" if (runtime and runtime.mel_processor and runtime.mel_processor.npu_available) else "CPU",
        "details": {}
    }

    if runtime and runtime.mel_processor and runtime.mel_processor.npu_available:
        hardware_info["details"]["kernel"] = "mel_fixed_v3 (Sign-Fixed)"
        hardware_info["details"]["firmware"] = "XRT 2.20.0"
        hardware_info["details"]["performance"] = "23.6× realtime (Mel preprocessing)"

    return {
        "service": "Unicorn-Amanuensis XDNA1",
        "status": "operational",
        "model": MODEL_SIZE,
        "hardware": hardware_info,
        "performance": "23.6× realtime",
        "stats": server_stats.to_dict()
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "runtime": "XDNA1" if (runtime and runtime.mel_processor and runtime.mel_processor.npu_available) else "CPU Fallback",
        "timestamp": int(time.time())
    }

@app.get("/stats")
async def stats():
    """Get detailed server statistics"""
    stats_data = server_stats.to_dict()

    if runtime:
        stats_data["runtime_stats"] = runtime.get_statistics()

    return stats_data

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("base"),
    language: str = Form("en"),
    task: str = Form("transcribe"),
    diarize: bool = Form(False),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None)
):
    """
    OpenAI-compatible transcription endpoint

    Transcribe audio file using XDNA1 NPU-accelerated mel preprocessing
    with full WhisperX transcription pipeline.

    Args:
        file: Audio file (MP3, WAV, FLAC, OGG, M4A)
        model: Whisper model (tiny, base, small, medium, large-v3)
        language: Language code (auto, en, es, fr, etc.)
        task: "transcribe" or "translate"
        diarize: Enable speaker diarization (requires HF_TOKEN)
        min_speakers: Minimum speakers for diarization
        max_speakers: Maximum speakers for diarization

    Returns:
        JSON with transcription results and performance metrics
    """

    # Save uploaded file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Processing audio file: {file.filename}")
        logger.info(f"Model: {model}, Language: {language}, Task: {task}")

        # Use XDNA1 runtime if available, otherwise fall back to CPU
        if runtime:
            logger.info("Using XDNA1 Runtime with sign-fixed mel kernel")
            start_time = time.perf_counter()

            try:
                result = runtime.transcribe(
                    tmp_path,
                    language=language if language != "auto" else "en",
                    task=task,
                    use_npu_mel=True
                )

                # Add extra metadata
                result["model"] = model
                result["task"] = task
                result["hardware"] = "XDNA1 (Phoenix NPU)"
                result["sign_fix_enabled"] = True

                server_stats.transcriptions += 1
                server_stats.npu_transcriptions += 1
                server_stats.total_audio_duration += result.get("audio_duration_s", 0)
                server_stats.total_processing_time += result.get("elapsed_ms", 0) / 1000

                logger.info(f"✅ Transcription complete: {result.get('realtime_factor', 0):.1f}× realtime")

                # Return OpenAI-compatible response
                return {
                    "text": result.get("text", ""),
                    "segments": result.get("segments", []),
                    "language": result.get("language", language),
                    "model": model,
                    "elapsed_ms": result.get("elapsed_ms", 0),
                    "audio_duration_s": result.get("audio_duration_s", 0),
                    "realtime_factor": result.get("realtime_factor", 0),
                    "npu_mel_used": result.get("npu_mel_used", False),
                    "npu_generation": result.get("npu_generation", ""),
                    "sign_fix_enabled": result.get("sign_fix_enabled", False)
                }

            except Exception as e:
                logger.warning(f"XDNA1 transcription failed: {e}")
                logger.warning("Falling back to CPU WhisperX")
                server_stats.cpu_fallback_count += 1

                # Fall back to standard WhisperX
                if not DEVICE_FALLBACK:
                    raise

        # CPU fallback
        logger.info("Using CPU WhisperX (no NPU)")
        import whisperx

        start_time = time.perf_counter()

        # Load model
        device = "cpu"
        whisper_model = whisperx.load_model(model, device, compute_type="int8")

        # Load audio
        audio = whisperx.load_audio(tmp_path)
        audio_duration = len(audio) / 16000  # WhisperX uses 16kHz

        # Transcribe
        result = whisper_model.transcribe(audio)

        # Align
        model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)

        # Optional diarization
        if diarize and HF_TOKEN and ENABLE_DIARIZATION:
            try:
                logger.info("Performing speaker diarization...")
                diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=min_speakers or 1,
                    max_speakers=max_speakers or 4
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                logger.warning(f"Diarization failed: {e}")

        elapsed = time.perf_counter() - start_time
        realtime_factor = audio_duration / elapsed if elapsed > 0 else 0

        server_stats.transcriptions += 1
        server_stats.total_audio_duration += audio_duration
        server_stats.total_processing_time += elapsed

        logger.info(f"CPU transcription complete: {realtime_factor:.1f}× realtime")

        return {
            "text": " ".join([seg["text"] for seg in result["segments"]]),
            "segments": result.get("segments", []),
            "language": result.get("language", language),
            "model": model,
            "elapsed_ms": elapsed * 1000,
            "audio_duration_s": audio_duration,
            "realtime_factor": realtime_factor,
            "npu_mel_used": False,
            "npu_generation": "None (CPU)",
            "sign_fix_enabled": False
        }

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                gc.collect()
            except:
                pass

# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("")
    logger.info("="*70)
    logger.info("🦄 Unicorn-Amanuensis XDNA1 Server Starting")
    logger.info("="*70)
    logger.info(f"Model: {MODEL_SIZE}")
    logger.info(f"NPU Fallback: {'Enabled' if DEVICE_FALLBACK else 'Disabled'}")
    logger.info(f"Diarization: {'Enabled' if ENABLE_DIARIZATION else 'Disabled'}")
    logger.info("")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("")
    logger.info("="*70)
    logger.info("🦄 Unicorn-Amanuensis XDNA1 Server Shutting Down")
    logger.info("="*70)

    # Print final statistics
    if runtime:
        runtime.print_statistics()

    stats_dict = server_stats.to_dict()
    logger.info(f"Total Transcriptions: {stats_dict['transcriptions']}")
    logger.info(f"Total Audio Duration: {stats_dict['total_audio_duration']:.2f}s")
    logger.info(f"Total Processing Time: {stats_dict['total_processing_time']:.2f}s")
    if stats_dict['avg_realtime_factor'] > 0:
        logger.info(f"Average Realtime Factor: {stats_dict['avg_realtime_factor']:.1f}×")
    logger.info(f"NPU Transcriptions: {stats_dict['npu_transcriptions']}")
    logger.info(f"CPU Fallback Count: {stats_dict['cpu_fallback_count']}")
    logger.info("="*70)
    logger.info("")

    # Cleanup
    if runtime:
        runtime.cleanup()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT", "8000"))

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
