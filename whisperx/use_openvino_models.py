#!/usr/bin/env python3
"""
Helper to automatically download and use OpenVINO-optimized Whisper models
These models are 3-5x faster on Intel iGPUs!
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

logger = logging.getLogger(__name__)

# Mapping of standard models to OpenVINO-optimized versions on HuggingFace
OPENVINO_MODEL_MAP = {
    "tiny": "unicorn-commander/whisper-tiny-openvino",
    "base": "unicorn-commander/whisper-base-openvino", 
    "small": "unicorn-commander/whisper-small-openvino",
    "medium": "unicorn-commander/whisper-medium-openvino",
    "large-v3": "unicorn-commander/whisper-large-v3-openvino",
}

# ONNX models for AMD NPU
NPU_MODEL_MAP = {
    "tiny": "unicorn-commander/whisper-tiny-onnx-npu",
    "base": "unicorn-commander/whisper-base-onnx-npu",
    "small": "unicorn-commander/whisper-small-onnx-npu",
}

def get_optimized_model_path(model_size: str, backend: str = "igpu"):
    """Get path to hardware-optimized model, downloading if needed"""
    
    cache_dir = Path.home() / ".cache" / "unicorn-amanuensis" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if backend == "igpu" and model_size in OPENVINO_MODEL_MAP:
        # Use OpenVINO model for Intel iGPU
        repo_id = OPENVINO_MODEL_MAP[model_size]
        model_dir = cache_dir / "openvino" / model_size
        
        if not model_dir.exists():
            logger.info(f"⬇️ Downloading OpenVINO model: {repo_id}")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                logger.info(f"✅ Downloaded to {model_dir}")
            except Exception as e:
                logger.warning(f"Failed to download OpenVINO model: {e}")
                return None
        
        return model_dir
    
    elif backend == "npu" and model_size in NPU_MODEL_MAP:
        # Use ONNX model for AMD NPU
        repo_id = NPU_MODEL_MAP[model_size]
        model_dir = cache_dir / "npu" / model_size
        
        if not model_dir.exists():
            logger.info(f"⬇️ Downloading NPU model: {repo_id}")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                logger.info(f"✅ Downloaded to {model_dir}")
            except Exception as e:
                logger.warning(f"Failed to download NPU model: {e}")
                return None
        
        return model_dir
    
    # Fallback to standard model
    return None

def load_optimized_whisper(model_size: str, backend: str):
    """Load hardware-optimized Whisper model"""
    
    model_path = get_optimized_model_path(model_size, backend)
    
    if model_path and backend == "igpu":
        # Load OpenVINO model
        try:
            from optimum.intel import OVModelForSpeechSeq2Seq
            from transformers import WhisperProcessor
            
            logger.info(f"🚀 Loading OpenVINO model from {model_path}")
            model = OVModelForSpeechSeq2Seq.from_pretrained(
                str(model_path),
                device="GPU",
                ov_config={"PERFORMANCE_HINT": "LATENCY"}
            )
            processor = WhisperProcessor.from_pretrained(str(model_path))
            
            return model, processor
            
        except Exception as e:
            logger.warning(f"Failed to load OpenVINO model: {e}")
    
    elif model_path and backend == "npu":
        # Load ONNX model for NPU compilation
        logger.info(f"🎯 Loading ONNX model for NPU from {model_path}")
        # NPU loading handled by npu_runtime.py
        return str(model_path)
    
    # Fallback to standard WhisperX
    logger.info(f"Using standard Whisper model: {model_size}")
    return None

# Quick test
if __name__ == "__main__":
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else "igpu"
    model = sys.argv[2] if len(sys.argv) > 2 else "base"
    
    print(f"Testing {backend} with {model} model...")
    result = load_optimized_whisper(model, backend)
    if result:
        print(f"✅ Successfully loaded optimized model!")
    else:
        print(f"⚠️ Using standard model")