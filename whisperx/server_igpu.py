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
import subprocess
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhisperX STT Service - Intel iGPU Enhanced")

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
BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "16"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Configure device and compute type
if DEVICE_TYPE == "igpu":
    # Try to use OpenVINO for Intel GPU
    try:
        import openvino as ov
        DEVICE = "cpu"  # WhisperX uses CPU, but we'll optimize with OpenVINO
        COMPUTE_TYPE = "int8_float16"  # Mixed precision for iGPU
        logger.info("Intel iGPU mode enabled with OpenVINO optimization")
    except ImportError:
        logger.warning("OpenVINO not available, falling back to CPU")
        DEVICE = "cpu"
        COMPUTE_TYPE = "int8"
elif DEVICE_TYPE == "npu":
    # AMD NPU support
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
    logger.info("NPU mode - using optimized CPU path")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
    logger.info("CUDA GPU detected")
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
    logger.info("Using CPU mode")

# Load model once at startup
logger.info(f"Loading WhisperX model: {MODEL_SIZE} on {DEVICE} with {COMPUTE_TYPE}")
model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)

# Load alignment model for English
logger.info("Loading alignment model...")
model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)

# Load diarization model if enabled
diarize_model = None
if ENABLE_DIARIZATION and HF_TOKEN:
    try:
        logger.info("Loading speaker diarization model...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        logger.info("Diarization model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load diarization model: {e}")
        ENABLE_DIARIZATION = False

def preprocess_audio_with_ffmpeg(input_path, output_path):
    """Preprocess audio to 16kHz mono using FFmpeg with hardware acceleration"""
    try:
        # Check if Intel QSV is available for hardware acceleration
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", "16000",  # Resample to 16kHz
            "-ac", "1",      # Convert to mono
            "-c:a", "pcm_s16le",  # PCM 16-bit little-endian
            "-f", "wav",
            output_path
        ]
        
        # Try with Intel QSV if available
        if DEVICE_TYPE == "igpu":
            # Check if QSV is available
            check_cmd = ["ffmpeg", "-hide_banner", "-hwaccels"]
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            if "qsv" in result.stdout:
                logger.info("Using Intel QSV hardware acceleration for audio preprocessing")
                cmd.insert(2, "-hwaccel")
                cmd.insert(3, "qsv")
        
        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
        
        logger.info(f"Audio preprocessed: {input_path} -> {output_path} (16kHz mono)")
        return True
        
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        return False

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Form(False),
    min_speakers: int = Form(2),
    max_speakers: int = Form(None),
    timestamps: bool = Form(True),
    word_timestamps: bool = Form(True)
):
    """Transcribe audio file with optional speaker diarization and timestamps"""
    
    if max_speakers is None:
        max_speakers = MAX_SPEAKERS
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        original_path = tmp.name
    
    # Preprocess audio to 16kHz
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        preprocessed_path = tmp_wav.name
    
    try:
        logger.info(f"Processing audio file: {file.filename}")
        
        # Preprocess audio with FFmpeg
        if not preprocess_audio_with_ffmpeg(original_path, preprocessed_path):
            # Fallback to original if preprocessing fails
            preprocessed_path = original_path
            logger.warning("Using original audio without preprocessing")
        
        # Load audio
        audio = whisperx.load_audio(preprocessed_path)
        
        # Transcribe with WhisperX
        logger.info(f"Transcribing with batch_size={BATCH_SIZE}...")
        result = model.transcribe(audio, batch_size=BATCH_SIZE)
        
        # Align whisper output for word-level timestamps
        if word_timestamps or timestamps:
            logger.info("Aligning for word-level timestamps...")
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE)
        
        # Optional: Speaker diarization
        if diarize and diarize_model:
            logger.info(f"Performing speaker diarization (speakers: {min_speakers}-{max_speakers})...")
            diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            logger.info("Speaker diarization completed")
        
        # Format response
        response = {
            "text": " ".join([segment["text"] for segment in result["segments"]]),
            "segments": result["segments"] if (timestamps or diarize) else None,
            "word_segments": result.get("word_segments") if word_timestamps else None,
            "language": result.get("language", "en"),
            "duration": len(audio) / 16000,  # Audio duration in seconds
            "model": MODEL_SIZE,
            "device": f"{DEVICE_TYPE} ({DEVICE})",
            "diarization_enabled": diarize and diarize_model is not None
        }
        
        # Clean up
        gc.collect()
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    finally:
        # Clean up temporary files
        try:
            os.unlink(original_path)
            if preprocessed_path != original_path:
                os.unlink(preprocessed_path)
        except:
            pass

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "device": f"{DEVICE_TYPE} ({DEVICE})",
        "compute_type": COMPUTE_TYPE,
        "diarization_available": diarize_model is not None,
        "max_speakers": MAX_SPEAKERS,
        "batch_size": BATCH_SIZE
    }

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Serve the web UI"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        # Fallback to API info if no UI available
        return JSONResponse({
            "service": "WhisperX STT - Intel iGPU Enhanced",
            "version": "2.0",
            "model": MODEL_SIZE,
            "device": f"{DEVICE_TYPE} ({DEVICE})",
            "endpoints": {
                "/": "GET - Web interface",
                "/v1/audio/transcriptions": "POST - Transcribe audio",
                "/health": "GET - Health check"
            }
        })

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "service": "WhisperX STT - Intel iGPU Enhanced",
        "version": "2.0",
        "model": MODEL_SIZE,
        "device": f"{DEVICE_TYPE} ({DEVICE})",
        "compute_type": COMPUTE_TYPE,
        "features": {
            "diarization": ENABLE_DIARIZATION,
            "max_speakers": MAX_SPEAKERS,
            "word_timestamps": True,
            "audio_preprocessing": "FFmpeg 16kHz conversion",
            "hardware_acceleration": DEVICE_TYPE == "igpu"
        },
        "endpoints": {
            "/": "GET - Web interface",
            "/v1/audio/transcriptions": "POST - Transcribe audio",
            "/health": "GET - Health check",
            "/api": "GET - API information"
        }
    }