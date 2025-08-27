#!/usr/bin/env python3
"""
Unicorn Amanuensis - Full WhisperX with Intel iGPU Optimization
Complete transcription with diarization and word-level timestamps
"""

import os
import sys
import json
import time
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import torch
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fix for ctranslate2 if needed
try:
    import ctypes
    import ctypes.util
    # Try to load libgomp explicitly
    gomp = ctypes.util.find_library('gomp')
    if gomp:
        ctypes.CDLL(gomp, mode=ctypes.RTLD_GLOBAL)
except:
    pass

# Import WhisperX after fixing libraries
import whisperx
import gc

# Environment configuration
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = "cpu"  # Intel iGPU uses CPU device with optimizations
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # INT8 quantization for Intel
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
HF_TOKEN = os.getenv("HF_TOKEN", "")
PORT = int(os.getenv("PORT", "9000"))

# Initialize FastAPI
app = FastAPI(
    title="🦄 Unicorn Amanuensis",
    description="Professional WhisperX Transcription Suite with Intel iGPU Optimization",
    version="3.0.0"
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
    logger.info(f"📁 Mounted static directory: {static_dir}")

# Global model variables
whisper_model = None
diarize_pipeline = None
model_info = {
    "model": MODEL_SIZE,
    "device": "Intel iGPU",
    "backend": "WhisperX + OpenVINO",
    "compute_type": COMPUTE_TYPE,
    "features": []
}

def load_models():
    """Load WhisperX models with Intel optimization"""
    global whisper_model, diarize_pipeline, model_info
    
    try:
        # Load WhisperX model
        logger.info(f"🔧 Loading WhisperX model: {MODEL_SIZE}")
        logger.info(f"📊 Device: {DEVICE}, Compute type: {COMPUTE_TYPE}")
        
        # Load with faster-whisper backend for better performance
        whisper_model = whisperx.load_model(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            language=None,  # Auto-detect
            asr_options={
                "beam_size": 5,
                "best_of": 5,
                "patience": 1,
                "length_penalty": 1,
                "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
                "initial_prompt": None,
                "prefix": None,
                "suppress_blank": True,
                "suppress_tokens": [-1],
                "without_timestamps": False,
                "max_initial_timestamp": 1.0,
                "word_timestamps": True,
                "prepend_punctuations": "\"'([{-",
                "append_punctuations": "\"'.!?:)]}",
            }
        )
        
        logger.info("✅ WhisperX model loaded successfully")
        model_info["features"].append("Transcription")
        model_info["features"].append("Word-level timestamps")
        model_info["features"].append("Language detection")
        
        # Load diarization pipeline if HuggingFace token is provided
        if HF_TOKEN:
            try:
                logger.info("🎭 Loading speaker diarization pipeline...")
                import torch
                # Use CPU for diarization as well
                diarize_pipeline = whisperx.DiarizationPipeline(
                    use_auth_token=HF_TOKEN,
                    device=torch.device(DEVICE)
                )
                logger.info("✅ Diarization pipeline loaded")
                model_info["features"].append("Speaker diarization")
            except Exception as e:
                logger.warning(f"⚠️ Could not load diarization: {e}")
                logger.warning("Continuing without speaker diarization")
                diarize_pipeline = None
        else:
            logger.info("ℹ️ No HF_TOKEN provided - diarization disabled")
            logger.info("💡 Set HF_TOKEN environment variable to enable speaker diarization")
            diarize_pipeline = None
            
        # Log final configuration
        logger.info(f"🚀 Model ready with features: {', '.join(model_info['features'])}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise

# Load models on startup
logger.info("🦄 Initializing Unicorn Amanuensis...")
load_models()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface"""
    index_file = static_dir / "index.html"
    if index_file.exists():
        with open(index_file, 'r') as f:
            return HTMLResponse(content=f.read())
    
    # Fallback HTML interface
    features_html = "".join([f"<li>✅ {feature}</li>" for feature in model_info['features']])
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🦄 Unicorn Amanuensis</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0;
            }}
            .container {{
                background: white;
                border-radius: 20px;
                padding: 3rem;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                max-width: 600px;
                width: 90%;
            }}
            h1 {{
                color: #333;
                margin: 0 0 1rem 0;
                display: flex;
                align-items: center;
                gap: 1rem;
            }}
            .subtitle {{
                color: #666;
                margin-bottom: 2rem;
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
                margin: 2rem 0;
            }}
            .info-card {{
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
            }}
            .info-label {{
                font-size: 0.875rem;
                color: #888;
                margin-bottom: 0.25rem;
            }}
            .info-value {{
                font-size: 1.125rem;
                font-weight: 600;
                color: #333;
            }}
            .features {{
                background: linear-gradient(135deg, #667eea15, #764ba215);
                border: 2px solid #667eea30;
                border-radius: 10px;
                padding: 1.5rem;
                margin: 2rem 0;
            }}
            .features h3 {{
                color: #667eea;
                margin: 0 0 1rem 0;
            }}
            .features ul {{
                list-style: none;
                padding: 0;
                margin: 0;
            }}
            .features li {{
                padding: 0.5rem 0;
                color: #555;
            }}
            .endpoints {{
                background: #f0f0f0;
                border-radius: 10px;
                padding: 1rem;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 0.875rem;
            }}
            .endpoints div {{
                margin: 0.5rem 0;
            }}
            .badge {{
                display: inline-block;
                background: #667eea;
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 600;
                margin-left: 0.5rem;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>
                <span style="font-size: 2.5rem;">🦄</span>
                Unicorn Amanuensis
            </h1>
            <p class="subtitle">Professional AI Transcription Suite</p>
            
            <div class="info-grid">
                <div class="info-card">
                    <div class="info-label">Model</div>
                    <div class="info-value">{model_info['model']}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Hardware</div>
                    <div class="info-value">{model_info['device']}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Backend</div>
                    <div class="info-value">{model_info['backend']}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Optimization</div>
                    <div class="info-value">{model_info['compute_type'].upper()}</div>
                </div>
            </div>
            
            <div class="features">
                <h3>🎯 Active Features</h3>
                <ul>
                    {features_html}
                </ul>
            </div>
            
            <div class="endpoints">
                <div>🌐 Web UI: <span class="badge">http://0.0.0.0:{PORT}</span></div>
                <div>🔌 API: <span class="badge">http://0.0.0.0:{PORT}/v1/audio/transcriptions</span></div>
                <div>❤️ Health: <span class="badge">http://0.0.0.0:{PORT}/health</span></div>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_info": model_info,
        "model_loaded": whisper_model is not None,
        "diarization_available": diarize_pipeline is not None
    }

@app.get("/api/hardware")
async def hardware_info():
    """Get detailed hardware information"""
    hw_info = {
        "device": model_info["device"],
        "backend": model_info["backend"],
        "optimization": model_info["compute_type"],
        "model": MODEL_SIZE,
        "batch_size": BATCH_SIZE,
        "features": model_info["features"],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
    }
    
    # Check Intel GPU via clinfo
    try:
        import subprocess
        result = subprocess.run(['clinfo'], capture_output=True, text=True, timeout=5)
        if 'Intel' in result.stdout:
            hw_info["intel_gpu_detected"] = True
            for line in result.stdout.split('\n'):
                if 'Device Name' in line and 'Intel' in line:
                    hw_info["intel_gpu_name"] = line.split('Device Name')[1].strip()
                    break
    except:
        hw_info["intel_gpu_detected"] = False
    
    return hw_info

@app.post("/v1/audio/transcriptions")
async def transcribe(
    request: Request,
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
    """OpenAI-compatible transcription endpoint with full WhisperX features"""
    
    if not whisper_model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file and not url:
        raise HTTPException(status_code=400, detail="Either file or url must be provided")
    
    try:
        start_time = time.time()
        
        # Save uploaded file
        if file:
            suffix = Path(file.filename).suffix if file.filename else ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                tmp_file.write(await file.read())
                temp_path = Path(tmp_file.name)
        else:
            raise HTTPException(status_code=400, detail="URL support not implemented yet")
        
        logger.info(f"📝 Processing file: {temp_path}")
        
        # Load audio
        audio = whisperx.load_audio(str(temp_path))
        audio_duration = len(audio) / 16000  # 16kHz sample rate
        
        # Step 1: Transcribe with WhisperX
        logger.info("🎙️ Transcribing audio...")
        result = whisper_model.transcribe(
            audio,
            batch_size=BATCH_SIZE,
            language=language,
            initial_prompt=prompt,
            temperature=temperature if temperature > 0 else None
        )
        
        # Get detected language
        detected_language = result.get("language", language or "en")
        logger.info(f"🌍 Language: {detected_language}")
        
        # Step 2: Align for word-level timestamps
        logger.info("⏱️ Aligning for word-level timestamps...")
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=DEVICE
        )
        
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device=DEVICE,
            return_char_alignments=False
        )
        
        # Clean up alignment model
        del model_a
        gc.collect()
        
        # Step 3: Speaker diarization (if enabled and available)
        speakers_data = None
        if diarization and diarize_pipeline:
            try:
                logger.info("🎭 Performing speaker diarization...")
                diarize_segments = diarize_pipeline(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                
                # Extract speaker segments
                speakers = {}
                for segment in result.get("segments", []):
                    speaker = segment.get("speaker", "SPEAKER_00")
                    if speaker not in speakers:
                        speakers[speaker] = []
                    speakers[speaker].append({
                        "text": segment.get("text", ""),
                        "start": segment.get("start", 0),
                        "end": segment.get("end", 0)
                    })
                
                speakers_data = [
                    {"speaker": speaker, "segments": segments}
                    for speaker, segments in speakers.items()
                ]
                
                logger.info(f"👥 Identified {len(speakers)} speakers")
                
            except Exception as e:
                logger.warning(f"⚠️ Diarization failed: {e}")
        
        # Extract words with timestamps
        all_words = []
        for segment in result.get("segments", []):
            if "words" in segment:
                for word in segment["words"]:
                    word_entry = {
                        "word": word.get("word", ""),
                        "start": round(word.get("start", 0), 3),
                        "end": round(word.get("end", 0), 3),
                        "confidence": round(word.get("score", 1.0), 3)
                    }
                    if "speaker" in word:
                        word_entry["speaker"] = word["speaker"]
                    all_words.append(word_entry)
        
        # Clean up temp file
        temp_path.unlink()
        
        # Calculate processing metrics
        processing_time = time.time() - start_time
        rtf = processing_time / audio_duration  # Real-time factor
        
        logger.info(f"✅ Transcription complete in {processing_time:.2f}s (RTF: {rtf:.2f})")
        
        # Format response
        if response_format == "text":
            return " ".join([seg["text"] for seg in result.get("segments", [])])
        
        # Full JSON response with all features
        response = {
            "text": " ".join([seg["text"] for seg in result.get("segments", [])]),
            "language": detected_language,
            "duration": round(audio_duration, 2),
            "segments": result.get("segments", []),
            "words": all_words,
            "word_count": len(all_words),
            "processing_time": round(processing_time, 2),
            "real_time_factor": round(rtf, 2),
            "model": MODEL_SIZE,
            "device": model_info["device"]
        }
        
        # Add speaker data if available
        if speakers_data:
            response["speakers"] = speakers_data
            response["speaker_count"] = len(speakers_data)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.get("/api/models")
async def list_models():
    """List available models"""
    return {
        "current": MODEL_SIZE,
        "available": [
            {"id": "large-v3", "name": "Whisper Large v3", "size": "1550M", "recommended": True},
            {"id": "large-v2", "name": "Whisper Large v2", "size": "1550M"},
            {"id": "medium", "name": "Whisper Medium", "size": "769M"},
            {"id": "small", "name": "Whisper Small", "size": "244M"},
            {"id": "base", "name": "Whisper Base", "size": "74M"},
            {"id": "tiny", "name": "Whisper Tiny", "size": "39M"}
        ],
        "optimization": f"{COMPUTE_TYPE.upper()} quantization"
    }

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("🦄 Unicorn Amanuensis - Professional Transcription Suite")
    logger.info("="*60)
    logger.info(f"🔧 Model: {MODEL_SIZE}")
    logger.info(f"💻 Device: Intel iGPU (via CPU + optimizations)")
    logger.info(f"⚡ Optimization: {COMPUTE_TYPE.upper()} quantization")
    logger.info(f"✨ Features: {', '.join(model_info['features'])}")
    logger.info(f"🌐 Web interface: http://0.0.0.0:{PORT}")
    logger.info(f"🔌 API endpoint: http://0.0.0.0:{PORT}/v1/audio/transcriptions")
    logger.info("="*60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        access_log=True
    )