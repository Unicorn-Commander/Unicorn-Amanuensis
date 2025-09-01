#!/usr/bin/env python3
"""
Unicorn Amanuensis - Pure OpenVINO Server for Intel iGPU
Optimized for Intel Arc/Iris Xe/UHD Graphics
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import openvino as ov
from transformers import WhisperProcessor
import librosa
import soundfile as sf
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
PORT = int(os.getenv("PORT", "9003"))
DEVICE = os.getenv("WHISPER_DEVICE", "GPU")  # GPU for Intel iGPU

# Model paths
MODEL_BASE_PATH = Path("/app/models")
OPENVINO_MODEL_MAP = {
    "base": MODEL_BASE_PATH / "whisper-base-openvino",
    "small": MODEL_BASE_PATH / "whisper-small-openvino",
    "large-v3": MODEL_BASE_PATH / "whisper-large-v3-openvino",
}

# Initialize FastAPI
app = FastAPI(
    title="Unicorn Amanuensis - Intel iGPU Edition",
    description="OpenVINO-optimized Speech Recognition on Intel Graphics",
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

# Global OpenVINO runtime
core = ov.Core()
compiled_model = None
processor = None

def load_openvino_model():
    """Load OpenVINO model for Intel iGPU"""
    global compiled_model, processor
    
    model_path = OPENVINO_MODEL_MAP.get(MODEL_SIZE)
    if not model_path or not model_path.exists():
        logger.error(f"Model not found: {MODEL_SIZE} at {model_path}")
        raise ValueError(f"OpenVINO model not found for {MODEL_SIZE}")
    
    logger.info(f"🚀 Loading OpenVINO model: {MODEL_SIZE} for Intel {DEVICE}")
    
    # Load processor
    processor = WhisperProcessor.from_pretrained(str(model_path))
    
    # Load encoder model
    encoder_path = model_path / "openvino_encoder_model.xml"
    decoder_path = model_path / "openvino_decoder_model.xml"
    
    if not encoder_path.exists() or not decoder_path.exists():
        raise ValueError(f"OpenVINO model files not found in {model_path}")
    
    # Configure for Intel iGPU
    config = {}
    if DEVICE == "GPU":
        # Intel iGPU optimizations
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f16",
            "GPU_THROUGHPUT_STREAMS": "1",
            "CACHE_DIR": "/app/cache"
        }
        logger.info(f"🎮 Configuring for Intel iGPU with FP16 precision")
    
    # Compile models
    logger.info("Compiling encoder model...")
    encoder_model = core.read_model(str(encoder_path))
    compiled_encoder = core.compile_model(encoder_model, DEVICE, config)
    
    logger.info("Compiling decoder model...")
    decoder_model = core.read_model(str(decoder_path))
    compiled_decoder = core.compile_model(decoder_model, DEVICE, config)
    
    compiled_model = {
        "encoder": compiled_encoder,
        "decoder": compiled_decoder
    }
    
    # Check device info
    devices = core.available_devices
    logger.info(f"✅ Available devices: {devices}")
    
    if "GPU" in devices:
        gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
        logger.info(f"🎮 Intel GPU detected: {gpu_name}")
    
    logger.info(f"✅ OpenVINO model loaded successfully on {DEVICE}")
    return True

def transcribe_audio(audio_array: np.ndarray, language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe audio using OpenVINO model"""
    
    if compiled_model is None or processor is None:
        raise ValueError("Model not loaded")
    
    start_time = time.time()
    
    # Preprocess audio
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="np")
    input_features = inputs.input_features
    
    # Run encoder
    encoder_output = compiled_model["encoder"](input_features)[0]
    
    # Generate tokens (simplified - full implementation would include beam search)
    # For now, we'll use greedy decoding
    max_length = 448
    decoder_input_ids = np.array([[processor.tokenizer.bos_token_id]], dtype=np.int64)
    
    generated_tokens = []
    for _ in range(max_length):
        # Run decoder
        decoder_output = compiled_model["decoder"]([decoder_input_ids, encoder_output])
        logits = decoder_output[0]
        
        # Get next token (greedy)
        next_token = np.argmax(logits[0, -1, :])
        generated_tokens.append(int(next_token))
        
        # Check for EOS
        if next_token == processor.tokenizer.eos_token_id:
            break
        
        # Update decoder input
        decoder_input_ids = np.append(decoder_input_ids, [[next_token]], axis=1)
    
    # Decode tokens to text
    text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    processing_time = time.time() - start_time
    
    return {
        "text": text,
        "processing_time": processing_time,
        "model": MODEL_SIZE,
        "device": f"Intel {DEVICE}",
        "language": language or "auto"
    }

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        load_openvino_model()
        logger.info("="*60)
        logger.info("🦄 Unicorn Amanuensis - Intel iGPU Edition")
        logger.info("="*60)
        logger.info(f"✅ Model: Whisper {MODEL_SIZE} (OpenVINO)")
        logger.info(f"✅ Device: Intel {DEVICE}")
        logger.info(f"✅ Optimization: FP16 precision, OpenVINO 2024.0+")
        logger.info(f"🌐 API: http://0.0.0.0:{PORT}/v1/audio/transcriptions")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy" if compiled_model else "unhealthy",
        "model": MODEL_SIZE,
        "device": f"Intel {DEVICE}",
        "backend": "OpenVINO",
        "optimization": "FP16",
        "models_loaded": compiled_model is not None
    })

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[str] = Form(None)
):
    """OpenAI-compatible transcription endpoint"""
    
    if not compiled_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load and resample audio
        audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
        
        # Transcribe
        result = transcribe_audio(audio, language)
        
        # Clean up
        os.unlink(tmp_path)
        
        # Format response
        if response_format == "text":
            return result["text"]
        else:
            return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Unicorn Amanuensis - Intel iGPU Edition",
        "model": MODEL_SIZE,
        "device": f"Intel {DEVICE}",
        "backend": "OpenVINO",
        "api": "/v1/audio/transcriptions",
        "health": "/health"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)