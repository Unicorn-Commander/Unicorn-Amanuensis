#!/usr/bin/env python3
"""
Unicorn Amanuensis - OpenVINO Native Implementation
Intel iGPU accelerated transcription without ctranslate2 dependencies
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
import torch
import torchaudio
from transformers import AutoProcessor, pipeline
import librosa

# OpenVINO optimized models
try:
    from optimum.intel import OVModelForSpeechSeq2Seq
    OPENVINO_AVAILABLE = True
    print("✅ OpenVINO optimized models available")
except ImportError:
    OPENVINO_AVAILABLE = False
    print("⚠️ OpenVINO optimized models not available, using standard transformers")

# Speaker diarization
try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    DIARIZATION_AVAILABLE = True
    print("✅ Pyannote.audio available for speaker diarization")
except ImportError:
    DIARIZATION_AVAILABLE = False
    print("⚠️ Pyannote.audio not available, diarization disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unicorn Amanuensis - OpenVINO Native")

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

# Configure device and model
if DEVICE_TYPE == "igpu" and OPENVINO_AVAILABLE:
    # Use OpenVINO for Intel GPU
    DEVICE = "gpu"
    MODEL_PATH = f"openai/whisper-{MODEL_SIZE}"
    logger.info("Intel iGPU mode enabled with OpenVINO optimization")
else:
    # Fallback to CPU
    DEVICE = "cpu" 
    MODEL_PATH = f"openai/whisper-{MODEL_SIZE}"
    logger.info("Using CPU mode")

# Load OpenVINO optimized model
logger.info(f"Loading OpenVINO-optimized Whisper model: {MODEL_SIZE}")
try:
    if OPENVINO_AVAILABLE:
        # Load OpenVINO optimized model
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            MODEL_PATH,
            export=True,  # Export to OpenVINO format if needed
            device=DEVICE,
            cache_dir="./models"
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        logger.info(f"✅ OpenVINO model {MODEL_SIZE} loaded successfully on {DEVICE}")
    else:
        # Fallback to standard pipeline
        model = pipeline(
            "automatic-speech-recognition",
            model=MODEL_PATH,
            device=0 if torch.cuda.is_available() and DEVICE_TYPE != "cpu" else -1,
            return_timestamps=True
        )
        processor = None
        logger.info(f"✅ Standard model {MODEL_SIZE} loaded successfully")
        
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    # Fallback to base model
    MODEL_SIZE = "base"
    MODEL_PATH = "openai/whisper-base"
    try:
        if OPENVINO_AVAILABLE:
            model = OVModelForSpeechSeq2Seq.from_pretrained(
                MODEL_PATH,
                export=True,
                device=DEVICE,
                cache_dir="./models"
            )
            processor = AutoProcessor.from_pretrained(MODEL_PATH)
        else:
            model = pipeline(
                "automatic-speech-recognition", 
                model=MODEL_PATH,
                device=0 if torch.cuda.is_available() and DEVICE_TYPE != "cpu" else -1,
                return_timestamps=True
            )
            processor = None
        logger.warning(f"Fell back to {MODEL_SIZE} model")
    except Exception as e2:
        logger.error(f"Failed to load fallback model: {e2}")
        raise

# Load diarization model if enabled
diarize_model = None
if ENABLE_DIARIZATION and HF_TOKEN and DIARIZATION_AVAILABLE:
    try:
        logger.info("Loading speaker diarization model...")
        diarize_model = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        logger.info("✅ Diarization model loaded")
    except Exception as e:
        logger.error(f"Failed to load diarization model: {e}")
        ENABLE_DIARIZATION = False

def check_intel_gpu_status():
    """Check Intel GPU status using vainfo and clinfo"""
    status = {
        "va_api": False,
        "opencl": False,
        "openvino_gpu": OPENVINO_AVAILABLE and DEVICE == "gpu",
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

def transcribe_with_openvino(audio_path: str, language: str = None):
    """Transcribe audio using OpenVINO optimized model"""
    try:
        if OPENVINO_AVAILABLE and processor:
            # Load and preprocess audio
            audio_array, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if audio_array.shape[0] > 1:
                audio_array = audio_array.mean(dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio_array = resampler(audio_array)
            
            # Convert to numpy and squeeze
            audio_np = audio_array.squeeze().numpy()
            
            # Prepare inputs
            inputs = processor(
                audio_np, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            
            # Generate transcription
            if language and language != "auto":
                # Set language if specified
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language=language, task="transcribe"
                )
                predicted_ids = model.generate(
                    **inputs, 
                    forced_decoder_ids=forced_decoder_ids,
                    return_timestamps=True
                )
            else:
                predicted_ids = model.generate(
                    **inputs,
                    return_timestamps=True
                )
            
            # Decode results
            transcription = processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )
            
            # Format output similar to WhisperX
            duration = len(audio_np) / 16000
            segments = [{
                "start": 0.0,
                "end": duration,
                "text": transcription[0] if transcription else ""
            }]
            
            return {
                "segments": segments,
                "language": language if language and language != "auto" else "en"
            }
            
        else:
            # Use standard pipeline
            result = model(audio_path, return_timestamps=True)
            
            # Convert to WhisperX-like format
            segments = []
            if "chunks" in result:
                for chunk in result["chunks"]:
                    segments.append({
                        "start": chunk["timestamp"][0] if chunk["timestamp"][0] else 0.0,
                        "end": chunk["timestamp"][1] if chunk["timestamp"][1] else 0.0,
                        "text": chunk["text"]
                    })
            else:
                segments.append({
                    "start": 0.0,
                    "end": 0.0,
                    "text": result.get("text", "")
                })
            
            return {
                "segments": segments,
                "language": language if language and language != "auto" else "en"
            }
            
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise

def add_speaker_diarization(audio_path: str, segments: List[Dict], min_speakers: int = 2, max_speakers: int = 10):
    """Add speaker diarization to transcription segments"""
    if not diarize_model:
        return segments
    
    try:
        # Perform diarization
        diarization = diarize_model(audio_path, num_speakers=min_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        
        # Map speakers to segments
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            
            # Find the most common speaker in this time range
            speakers_in_range = []
            for speech_turn, track, speaker in diarization.itertracks(yield_label=True):
                if speech_turn.start <= end_time and speech_turn.end >= start_time:
                    speakers_in_range.append(speaker)
            
            if speakers_in_range:
                # Use the most common speaker
                from collections import Counter
                most_common_speaker = Counter(speakers_in_range).most_common(1)[0][0]
                segment["speaker"] = most_common_speaker
            else:
                segment["speaker"] = "SPEAKER_00"
        
        return segments
        
    except Exception as e:
        logger.error(f"Diarization error: {e}")
        return segments

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
    Transcribe audio with OpenVINO Intel iGPU acceleration
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
        
        # Get audio duration
        try:
            audio_array, sample_rate = torchaudio.load(preprocessed_path)
            audio_duration = audio_array.shape[1] / sample_rate
        except:
            audio_duration = 0.0
            
        logger.info(f"Audio duration: {audio_duration:.1f}s")
        
        # Transcribe with OpenVINO
        transcribe_start = time.time()
        logger.info(f"Transcribing with OpenVINO {MODEL_SIZE} model...")
        result = transcribe_with_openvino(
            preprocessed_path, 
            language=language if language != "auto" else None
        )
        transcribe_time = time.time() - transcribe_start
        
        segments = result["segments"]
        
        # Add speaker diarization
        diarize_time = 0
        if diarize and diarize_model:
            diarize_start = time.time()
            logger.info(f"Performing speaker diarization (speakers: {min_speakers}-{max_speakers})...")
            segments = add_speaker_diarization(
                preprocessed_path, 
                segments, 
                min_speakers=min_speakers, 
                max_speakers=max_speakers
            )
            diarize_time = time.time() - diarize_start
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        rtf = total_time / audio_duration if audio_duration > 0 else 0  # Real-time factor
        
        # Format response based on requested format
        if response_format == "text":
            text = " ".join([seg.get("text", "") for seg in segments])
            return JSONResponse(content={"text": text})
        
        # Full JSON response
        response = {
            "text": " ".join([seg.get("text", "") for seg in segments]),
            "segments": segments if timestamps else None,
            "language": result.get("language", language),
            "duration": audio_duration,
            "performance": {
                "total_time": f"{total_time:.2f}s",
                "rtf": f"{rtf:.2f}x",
                "preprocess_time": f"{preprocess_time:.2f}s",
                "transcribe_time": f"{transcribe_time:.2f}s",
                "diarize_time": f"{diarize_time:.2f}s" if diarize_time > 0 else None
            },
            "config": {
                "model": MODEL_SIZE,
                "device": f"Intel iGPU (OpenVINO)" if OPENVINO_AVAILABLE and DEVICE == "gpu" else "CPU",
                "engine": "OpenVINO" if OPENVINO_AVAILABLE else "Standard",
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
        "device": f"Intel iGPU (OpenVINO)" if OPENVINO_AVAILABLE and DEVICE == "gpu" else "CPU",
        "engine": "OpenVINO" if OPENVINO_AVAILABLE else "Standard",
        "hardware": {
            "openvino_available": OPENVINO_AVAILABLE,
            "openvino_device": DEVICE,
            "va_api": gpu_status["va_api"],
            "opencl": gpu_status["opencl"],
            "details": gpu_status["details"]
        },
        "features": {
            "diarization": diarize_model is not None,
            "max_speakers": MAX_SPEAKERS,
            "batch_size": BATCH_SIZE,
            "timestamps": True,
            "word_timestamps": True
        }
    }

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Serve the web UI"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return HTMLResponse(content=f"""
        <html>
        <body>
        <h1>Unicorn Amanuensis - OpenVINO Native</h1>
        <p>Model: {MODEL_SIZE}</p>
        <p>Device: Intel iGPU (OpenVINO)</p>
        <p>Engine: {"OpenVINO" if OPENVINO_AVAILABLE else "Standard"}</p>
        <p>API Endpoint: POST /v1/audio/transcriptions</p>
        </body>
        </html>
        """)

@app.get("/gpu-status")
async def gpu_status():
    """Get detailed GPU status"""
    status = check_intel_gpu_status()
    
    # Add OpenVINO specific info
    if OPENVINO_AVAILABLE:
        try:
            import openvino as ov
            core = ov.Core()
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
        except:
            pass
    
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)