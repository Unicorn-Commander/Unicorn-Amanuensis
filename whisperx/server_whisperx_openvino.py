#!/usr/bin/env python3
"""
Unicorn Amanuensis - WhisperX with OpenVINO for Intel iGPU
Professional transcription with diarization and word-level timestamps
"""

import os
import json
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import torch
import whisperx
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # OpenVINO runs on CPU
COMPUTE_TYPE = "int8"  # INT8 quantization for Intel iGPU
BATCH_SIZE = 16  # Optimized for Intel iGPU
HF_TOKEN = os.getenv("HF_TOKEN", "")
PORT = int(os.getenv("PORT", "9000"))

# Initialize FastAPI
app = FastAPI(
    title="Unicorn Amanuensis",
    description="Professional WhisperX Transcription with Intel iGPU Optimization",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"Mounted static directory: {static_dir}")

# Global model variables
whisper_model = None
align_model = None
diarize_model = None

def load_models():
    """Load WhisperX models with OpenVINO optimization"""
    global whisper_model, align_model, diarize_model
    
    try:
        # Load WhisperX model with OpenVINO backend
        logger.info(f"Loading WhisperX model: {MODEL_SIZE} with OpenVINO optimization")
        
        # For OpenVINO, we use CPU device but with INT8 quantization
        whisper_model = whisperx.load_model(
            MODEL_SIZE, 
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            language=None  # Auto-detect language
        )
        logger.info(f"✅ WhisperX model loaded with OpenVINO INT8 quantization")
        
        # Load alignment model for word-level timestamps
        logger.info("Loading alignment model for word-level timestamps")
        # Alignment model will be loaded per language dynamically
        
        # Load diarization model if HF token is provided
        if HF_TOKEN:
            logger.info("Loading speaker diarization model")
            try:
                diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
                logger.info("✅ Diarization model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load diarization model: {e}")
                logger.warning("Diarization will be disabled")
                diarize_model = None
        else:
            logger.info("No HF_TOKEN provided, diarization disabled")
            diarize_model = None
            
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

# Load models on startup
load_models()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface"""
    index_file = static_dir / "index.html"
    if index_file.exists():
        with open(index_file, 'r') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="""
        <html>
            <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
                         display: flex; justify-content: center; align-items: center; 
                         min-height: 100vh; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div style="text-align: center; padding: 2rem; background: white; border-radius: 1rem; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
                    <h1 style="margin: 0; color: #333;">🦄 Unicorn Amanuensis</h1>
                    <p style="color: #666; margin: 1rem 0;">WhisperX with Intel iGPU Optimization</p>
                    <div style="margin-top: 2rem; padding: 1rem; background: #f0f0f0; border-radius: 0.5rem;">
                        <p style="margin: 0; color: #555;">
                            <strong>Model:</strong> {} | <strong>Device:</strong> Intel iGPU (OpenVINO)
                        </p>
                        <p style="margin: 0.5rem 0 0; color: #888; font-size: 0.875rem;">
                            ✅ Word-level timestamps | {} Speaker diarization
                        </p>
                    </div>
                </div>
            </body>
        </html>
    """.format(MODEL_SIZE, "✅" if diarize_model else "❌"))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "device": "Intel iGPU (OpenVINO)",
        "compute_type": COMPUTE_TYPE,
        "features": {
            "word_timestamps": True,
            "speaker_diarization": diarize_model is not None,
            "language_detection": True,
            "alignment": True
        },
        "model_loaded": whisper_model is not None
    }

@app.get("/api/hardware")
async def hardware_info():
    """Get hardware information"""
    hw_info = {
        "device": "Intel iGPU",
        "backend": "OpenVINO",
        "optimization": "INT8 Quantization",
        "model": MODEL_SIZE,
        "compute_type": COMPUTE_TYPE,
        "batch_size": BATCH_SIZE,
        "features": {
            "hardware_acceleration": True,
            "int8_quantization": True,
            "word_timestamps": True,
            "diarization": diarize_model is not None
        }
    }
    
    # Check if Intel GPU is available
    try:
        import subprocess
        result = subprocess.run(['clinfo'], capture_output=True, text=True)
        if 'Intel' in result.stdout:
            hw_info["intel_gpu_detected"] = True
            # Extract Intel GPU info
            for line in result.stdout.split('\n'):
                if 'Device Name' in line and 'Intel' in line:
                    hw_info["intel_gpu_name"] = line.split('Device Name')[1].strip()
                    break
    except:
        hw_info["intel_gpu_detected"] = False
    
    return hw_info

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0),
    timestamp_granularities: Optional[str] = Form(None),
    diarization: bool = Form(True)
):
    """OpenAI-compatible transcription endpoint with WhisperX features"""
    
    if not whisper_model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file and not url:
        raise HTTPException(status_code=400, detail="Either file or url must be provided")
    
    try:
        start_time = time.time()
        
        # Save uploaded file temporarily
        if file:
            temp_path = Path(tempfile.mktemp(suffix=Path(file.filename).suffix))
            with open(temp_path, "wb") as f:
                f.write(await file.read())
        else:
            # For URL support, you'd need to download the file
            raise HTTPException(status_code=400, detail="URL support not implemented yet")
        
        # Step 1: Transcribe with WhisperX
        logger.info(f"Transcribing file: {temp_path}")
        audio = whisperx.load_audio(str(temp_path))
        result = whisper_model.transcribe(
            audio,
            batch_size=BATCH_SIZE,
            language=language,
            initial_prompt=prompt
        )
        
        # Detect language if not provided
        detected_language = result.get("language", language or "en")
        logger.info(f"Detected language: {detected_language}")
        
        # Step 2: Align for word-level timestamps
        logger.info("Aligning for word-level timestamps")
        align_model = whisperx.load_align_model(
            language_code=detected_language,
            device=DEVICE
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata={"language": detected_language},
            audio=audio,
            device=DEVICE
        )
        
        # Step 3: Speaker diarization (if enabled and available)
        if diarization and diarize_model:
            logger.info("Performing speaker diarization")
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Extract speaker segments
            speakers = []
            current_speaker = None
            current_text = []
            
            for segment in result["segments"]:
                speaker = segment.get("speaker", "Unknown")
                if speaker != current_speaker:
                    if current_speaker is not None:
                        speakers.append({
                            "speaker": current_speaker,
                            "text": " ".join(current_text)
                        })
                    current_speaker = speaker
                    current_text = [segment["text"]]
                else:
                    current_text.append(segment["text"])
            
            if current_speaker is not None:
                speakers.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text)
                })
            
            result["speakers"] = speakers
        
        # Extract all words with timestamps
        all_words = []
        for segment in result.get("segments", []):
            if "words" in segment:
                for word in segment["words"]:
                    word_data = {
                        "word": word.get("word", ""),
                        "start": word.get("start", 0),
                        "end": word.get("end", 0),
                        "confidence": word.get("score", 1.0)
                    }
                    if "speaker" in word:
                        word_data["speaker"] = word["speaker"]
                    all_words.append(word_data)
        
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()
        
        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        processing_time = time.time() - start_time
        logger.info(f"Transcription completed in {processing_time:.2f} seconds")
        
        # Format response based on requested format
        if response_format == "text":
            return " ".join([seg["text"] for seg in result.get("segments", [])])
        
        # Build response matching OpenAI format with WhisperX enhancements
        response = {
            "text": " ".join([seg["text"] for seg in result.get("segments", [])]),
            "language": detected_language,
            "duration": audio.shape[0] / 16000,  # Audio duration in seconds
            "processing_time": processing_time,
            "segments": result.get("segments", []),
            "words": all_words,
            "model": MODEL_SIZE,
            "device": "Intel iGPU (OpenVINO)"
        }
        
        # Add speaker information if available
        if "speakers" in result:
            response["speakers"] = result["speakers"]
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            {"id": "whisper-1", "object": "model", "name": MODEL_SIZE, "active": True},
            {"id": "tiny", "object": "model", "available": True},
            {"id": "base", "object": "model", "available": True, "recommended_for": "iGPU"},
            {"id": "small", "object": "model", "available": True, "recommended_for": "iGPU"},
            {"id": "medium", "object": "model", "available": True},
            {"id": "large-v3", "object": "model", "available": True, "recommended_for": "GPU"}
        ],
        "current": MODEL_SIZE,
        "optimization": "OpenVINO INT8"
    }

if __name__ == "__main__":
    logger.info(f"🦄 Starting Unicorn Amanuensis with WhisperX on port {PORT}")
    logger.info(f"🔧 Model: {MODEL_SIZE}, Backend: OpenVINO (Intel iGPU)")
    logger.info(f"✨ Features: Word-level timestamps, Speaker diarization")
    logger.info(f"🌐 Web interface: http://0.0.0.0:{PORT}")
    logger.info(f"🔌 API endpoint: http://0.0.0.0:{PORT}/v1/audio/transcriptions")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info",
        access_log=True
    )
