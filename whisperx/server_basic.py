#!/usr/bin/env python3
"""
Unicorn Amanuensis - Basic Whisper Server
Simplified version without ctranslate2 dependency
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import whisper
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = int(os.getenv("PORT", "9000"))

# Initialize FastAPI
app = FastAPI(
    title="Unicorn Amanuensis",
    description="Professional AI Transcription Suite - Basic Version",
    version="1.0.0"
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

# Load Whisper model
logger.info(f"Loading Whisper model: {MODEL_SIZE} on {DEVICE}")
try:
    model = whisper.load_model(MODEL_SIZE, device=DEVICE)
    logger.info(f"✅ Model loaded successfully on {DEVICE}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

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
                    <p style="color: #666; margin: 1rem 0;">Professional AI Transcription Suite</p>
                    <div style="margin-top: 2rem; padding: 1rem; background: #f0f0f0; border-radius: 0.5rem;">
                        <p style="margin: 0; color: #555;">
                            Running on <strong>{}</strong> with model <strong>{}</strong>
                        </p>
                        <p style="margin: 0.5rem 0 0; color: #888; font-size: 0.875rem;">
                            Basic version without hardware acceleration
                        </p>
                    </div>
                </div>
            </body>
        </html>
    """.format(DEVICE.upper(), MODEL_SIZE))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "model_loaded": model is not None
    }

@app.get("/api/hardware")
async def hardware_info():
    """Get hardware information"""
    hw_info = {
        "device": DEVICE,
        "model": MODEL_SIZE,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        hw_info["cuda_device_name"] = torch.cuda.get_device_name(0)
        hw_info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
        hw_info["cuda_memory_reserved"] = torch.cuda.memory_reserved(0)
    
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
    timestamp_granularities: Optional[str] = Form(None)
):
    """OpenAI-compatible transcription endpoint"""
    
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file and not url:
        raise HTTPException(status_code=400, detail="Either file or url must be provided")
    
    try:
        start_time = time.time()
        
        # Save uploaded file temporarily
        if file:
            temp_path = Path(f"/tmp/{file.filename}")
            with open(temp_path, "wb") as f:
                f.write(await file.read())
        else:
            # For URL support, you'd need to download the file
            raise HTTPException(status_code=400, detail="URL support not implemented in basic version")
        
        # Transcribe with Whisper
        logger.info(f"Transcribing file: {temp_path}")
        result = model.transcribe(
            str(temp_path),
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            word_timestamps=True
        )
        
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()
        
        processing_time = time.time() - start_time
        logger.info(f"Transcription completed in {processing_time:.2f} seconds")
        
        # Format response based on requested format
        if response_format == "text":
            return result["text"]
        
        # Build response matching OpenAI format
        response = {
            "text": result["text"],
            "language": result.get("language", language or "unknown"),
            "duration": result.get("duration", 0),
            "processing_time": processing_time
        }
        
        # Add segments with timestamps
        if "segments" in result:
            response["segments"] = result["segments"]
        
        # Add words if available
        if "segments" in result and any("words" in seg for seg in result["segments"]):
            all_words = []
            for segment in result["segments"]:
                if "words" in segment:
                    all_words.extend(segment["words"])
            response["words"] = all_words
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            {"id": "whisper-1", "object": "model", "name": MODEL_SIZE},
            {"id": "tiny", "object": "model", "available": True},
            {"id": "base", "object": "model", "available": True},
            {"id": "small", "object": "model", "available": True},
            {"id": "medium", "object": "model", "available": True},
            {"id": "large-v3", "object": "model", "available": True}
        ]
    }

if __name__ == "__main__":
    logger.info(f"🦄 Starting Unicorn Amanuensis on port {PORT}")
    logger.info(f"🔧 Device: {DEVICE}, Model: {MODEL_SIZE}")
    logger.info(f"🌐 Web interface: http://0.0.0.0:{PORT}")
    logger.info(f"🔌 API endpoint: http://0.0.0.0:{PORT}/v1/audio/transcriptions")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info",
        access_log=True
    )