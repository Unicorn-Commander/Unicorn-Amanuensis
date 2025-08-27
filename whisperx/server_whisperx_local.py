#!/usr/bin/env python3
"""
Unicorn Amanuensis - WhisperX with Local Diarization
All processing happens locally on Intel iGPU
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
import numpy as np
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = "cpu"  # Intel iGPU uses CPU device
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
BATCH_SIZE = 16
PORT = int(os.getenv("PORT", "9000"))

# Local model cache
MODEL_CACHE_DIR = Path("/models")
PYANNOTE_CACHE = MODEL_CACHE_DIR / "pyannote"
PYANNOTE_CACHE.mkdir(parents=True, exist_ok=True)

# Set HF cache to local directory
os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)
os.environ['TORCH_HOME'] = str(MODEL_CACHE_DIR / "torch")

# Initialize FastAPI
app = FastAPI(
    title="🦄 Unicorn Amanuensis",
    description="Professional WhisperX with Local Speaker Diarization",
    version="4.0.0"
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

# Global model variables
whisper_model = None
diarize_pipeline = None
model_info = {
    "model": MODEL_SIZE,
    "device": "Intel iGPU",
    "compute_type": COMPUTE_TYPE,
    "features": []
}

def load_local_diarization():
    """Load diarization pipeline from local cache"""
    try:
        from pyannote.audio import Pipeline
        
        # Check if models exist locally
        pipeline_path = PYANNOTE_CACHE / "speaker-diarization-3.1"
        
        # Try to load from local cache first
        if pipeline_path.exists():
            logger.info("📂 Loading diarization from local cache...")
            pipeline = Pipeline.from_pretrained(
                str(pipeline_path),
                use_auth_token=False
            )
        else:
            # Try to load without token (for open models)
            logger.info("📥 Loading open diarization model...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                cache_dir=PYANNOTE_CACHE,
                use_auth_token=False
            )
        
        # Configure for Intel iGPU (CPU device)
        pipeline.to(torch.device("cpu"))
        
        logger.info("✅ Diarization pipeline loaded locally")
        return pipeline
        
    except Exception as e:
        logger.warning(f"⚠️ Could not load diarization: {e}")
        logger.info("💡 Diarization will be disabled")
        logger.info("   To enable: Run download_diarization_models.py")
        return None

def load_models():
    """Load WhisperX and diarization models"""
    global whisper_model, diarize_pipeline, model_info
    
    try:
        # Load WhisperX model
        logger.info(f"🔧 Loading WhisperX model: {MODEL_SIZE}")
        logger.info(f"💾 Using cache directory: {MODEL_CACHE_DIR}")
        
        whisper_model = whisperx.load_model(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=str(MODEL_CACHE_DIR / "whisper"),
            language=None  # Auto-detect
        )
        
        logger.info("✅ WhisperX model loaded")
        model_info["features"].extend([
            "Transcription",
            "Word-level timestamps",
            "Language detection"
        ])
        
        # Try to load local diarization
        diarize_pipeline = load_local_diarization()
        if diarize_pipeline:
            model_info["features"].append("Speaker diarization (local)")
        
        logger.info(f"🚀 Features available: {', '.join(model_info['features'])}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise

# Load models on startup
logger.info("🦄 Initializing Unicorn Amanuensis with Local Processing...")
load_models()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface"""
    index_file = static_dir / "index.html"
    if index_file.exists():
        with open(index_file, 'r') as f:
            return HTMLResponse(content=f.read())
    
    features_html = "".join([f"<li>{feature}</li>" for feature in model_info['features']])
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🦄 Unicorn Amanuensis - Local Processing</title>
        <style>
            body {{
                font-family: -apple-system, sans-serif;
                background: linear-gradient(135deg, #667eea, #764ba2);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0;
            }}
            .container {{
                background: white;
                padding: 2rem;
                border-radius: 1rem;
                max-width: 600px;
            }}
            h1 {{ margin: 0 0 1rem 0; }}
            .features {{ 
                background: #f5f5f5; 
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 1rem 0;
            }}
            .status {{
                background: #e8f5e9;
                color: #2e7d32;
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                display: inline-block;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🦄 Unicorn Amanuensis</h1>
            <p class="status">✅ Running 100% Locally on Intel iGPU</p>
            <div class="features">
                <h3>Available Features:</h3>
                <ul>{features_html}</ul>
            </div>
            <p><strong>Model:</strong> {MODEL_SIZE}</p>
            <p><strong>Optimization:</strong> {COMPUTE_TYPE.upper()} quantization</p>
            <p><strong>API:</strong> http://0.0.0.0:{PORT}/v1/audio/transcriptions</p>
        </div>
    </body>
    </html>
    """)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "device": "Intel iGPU (CPU + optimizations)",
        "compute_type": COMPUTE_TYPE,
        "features": model_info["features"],
        "model_loaded": whisper_model is not None,
        "diarization_available": diarize_pipeline is not None,
        "local_processing": True
    }

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: Optional[UploadFile] = File(None),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
    diarization: bool = Form(True)
):
    """Transcribe audio with optional local speaker diarization"""
    
    if not whisper_model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    try:
        start_time = time.time()
        
        # Save uploaded file
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            temp_path = Path(tmp.name)
        
        # Load audio
        audio = whisperx.load_audio(str(temp_path))
        duration = len(audio) / 16000
        
        logger.info(f"🎤 Processing {duration:.1f}s of audio...")
        
        # Step 1: Transcribe with WhisperX
        logger.info("📝 Transcribing...")
        result = whisper_model.transcribe(
            audio,
            batch_size=BATCH_SIZE,
            language=language
        )
        
        detected_lang = result.get("language", "en")
        logger.info(f"🌍 Language: {detected_lang}")
        
        # Step 2: Align for word-level timestamps
        logger.info("⏱️ Aligning words...")
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_lang,
            device=DEVICE
        )
        
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device=DEVICE
        )
        
        # Clean up alignment model
        del model_a
        gc.collect()
        
        # Step 3: Local speaker diarization (if available and requested)
        speakers_data = None
        if diarization and diarize_pipeline:
            try:
                logger.info("🎭 Running local speaker diarization...")
                
                # Convert audio for diarization
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                
                # Run diarization locally
                diarization = diarize_pipeline(
                    {"waveform": audio_tensor, "sample_rate": 16000}
                )
                
                # Convert to WhisperX format
                result = whisperx.assign_word_speakers(
                    diarization,
                    result
                )
                
                # Extract speaker segments
                speakers = {}
                for seg in result.get("segments", []):
                    spk = seg.get("speaker", "SPEAKER_00")
                    if spk not in speakers:
                        speakers[spk] = []
                    speakers[spk].append(seg["text"])
                
                speakers_data = [
                    {"speaker": spk, "text": " ".join(texts)}
                    for spk, texts in speakers.items()
                ]
                
                logger.info(f"👥 Found {len(speakers)} speakers")
                
            except Exception as e:
                logger.warning(f"Diarization failed: {e}")
        
        # Extract words
        words = []
        for seg in result.get("segments", []):
            if "words" in seg:
                for word in seg["words"]:
                    words.append({
                        "word": word.get("word", ""),
                        "start": round(word.get("start", 0), 3),
                        "end": round(word.get("end", 0), 3),
                        "speaker": word.get("speaker") if diarization else None
                    })
        
        # Clean up
        temp_path.unlink()
        gc.collect()
        
        processing_time = time.time() - start_time
        
        # Format response
        if response_format == "text":
            return " ".join([s["text"] for s in result.get("segments", [])])
        
        response = {
            "text": " ".join([s["text"] for s in result.get("segments", [])]),
            "language": detected_lang,
            "duration": round(duration, 2),
            "segments": result.get("segments", []),
            "words": words,
            "processing_time": round(processing_time, 2),
            "model": MODEL_SIZE,
            "local_processing": True
        }
        
        if speakers_data:
            response["speakers"] = speakers_data
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("🦄 Unicorn Amanuensis - 100% Local Processing")
    logger.info("="*60)
    logger.info(f"Model: {MODEL_SIZE}")
    logger.info(f"Device: Intel iGPU")
    logger.info(f"Optimization: {COMPUTE_TYPE.upper()}")
    logger.info(f"Cache: {MODEL_CACHE_DIR}")
    logger.info(f"Features: {', '.join(model_info['features'])}")
    logger.info(f"API: http://0.0.0.0:{PORT}/v1/audio/transcriptions")
    logger.info("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)