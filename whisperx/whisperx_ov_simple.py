#!/usr/bin/env python3
"""
Simple WhisperX-compatible wrapper using OpenVINO for TRUE iGPU acceleration
This provides a drop-in replacement for whisperx.load_model that uses OpenVINO
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import openvino as ov
from transformers import WhisperProcessor
import torch

logger = logging.getLogger(__name__)

class WhisperXOpenVINO:
    """WhisperX-compatible model using OpenVINO backend for iGPU"""
    
    def __init__(self, model_size: str = "large-v3", device: str = "GPU", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        
        # Initialize OpenVINO
        self.core = ov.Core()
        available = self.core.available_devices
        logger.info(f"Available devices: {available}")
        
        # Select GPU device
        if "GPU.0" in available:
            self.ov_device = "GPU.0"
        elif "GPU" in available:
            self.ov_device = "GPU"
        else:
            self.ov_device = "CPU"
            logger.warning("No GPU found, using CPU")
        
        logger.info(f"Using device: {self.ov_device}")
        
        # Load processor
        model_id = f"openai/whisper-{model_size}"
        self.processor = WhisperProcessor.from_pretrained(model_id)
        
        # Check for pre-converted model
        ov_model_path = f"./models/whisper-{model_size}-ov"
        
        if not Path(ov_model_path).exists():
            logger.info(f"Converting {model_id} to OpenVINO format...")
            self._convert_model(model_id, ov_model_path)
        
        # Load OpenVINO model
        self._load_ov_model(ov_model_path)
    
    def _convert_model(self, model_id: str, output_path: str):
        """Convert Whisper to OpenVINO format"""
        from optimum.intel import OVModelForSpeechSeq2Seq
        
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Export to OpenVINO
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            export=True,
            compile=False
        )
        ov_model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)
        
        logger.info(f"Model converted and saved to {output_path}")
    
    def _load_ov_model(self, model_path: str):
        """Load OpenVINO model for inference"""
        from optimum.intel import OVModelForSpeechSeq2Seq
        
        # Force GPU-only execution with proper config
        ov_config = {
            "PERFORMANCE_HINT": "LATENCY",
            "CACHE_DIR": "./cache",
            "NUM_STREAMS": "1",
            "INFERENCE_PRECISION_HINT": "f16",  # Use FP16 for better GPU performance
            "GPU_ENABLE_SDPA_OPTIMIZATION": "YES",
            "EXCLUSIVE_ASYNC_REQUESTS": "YES"
        }
        
        # Set environment to force GPU
        import os
        os.environ["OV_CACHE_DIR"] = "./cache"
        os.environ["OV_GPU_CACHE_MODEL"] = "1"
        
        self.model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            device=self.ov_device,
            ov_config=ov_config,
            compile=True,
            # Force export settings for GPU
            export=False,  # Already exported
            trust_remote_code=True
        )
        
        logger.info(f"Model loaded and compiled for {self.ov_device}")
    
    def transcribe(self, audio: np.ndarray, batch_size: int = 16, 
                   language: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Transcribe audio using OpenVINO on iGPU"""
        
        # Force GPU execution
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA to avoid conflicts
        
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Process in chunks for long audio
        CHUNK_LENGTH = 30 * 16000  # 30 seconds
        total_samples = len(audio)
        
        if total_samples > CHUNK_LENGTH:
            # Process long audio in chunks
            transcriptions = []
            segments = []
            
            for i in range(0, total_samples, CHUNK_LENGTH):
                chunk = audio[i:min(i + CHUNK_LENGTH, total_samples)]
                
                # Skip very short chunks
                if len(chunk) < 16000:
                    continue
                
                # Process chunk
                inputs = self.processor(chunk, sampling_rate=16000, return_tensors="pt")
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        inputs.input_features,
                        language=language,
                        task="transcribe"
                    )
                
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                transcriptions.append(text)
                
                # Create segment
                segments.append({
                    "start": i / 16000,
                    "end": min(i + CHUNK_LENGTH, total_samples) / 16000,
                    "text": text
                })
            
            return {
                "text": " ".join(transcriptions),
                "segments": segments,
                "language": language or "en"
            }
        else:
            # Process short audio
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_features,
                    language=language,
                    task="transcribe"
                )
            
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return {
                "text": text,
                "segments": [{"start": 0, "end": len(audio) / 16000, "text": text}],
                "language": language or "en"
            }


def load_model(whisper_arch: str, device: str = "gpu", 
               compute_type: str = "int8", **kwargs):
    """Drop-in replacement for whisperx.load_model using OpenVINO"""
    return WhisperXOpenVINO(whisper_arch, device, compute_type)


# Alignment stub for compatibility
def load_align_model(language_code: str = None, device: str = "cuda", **kwargs):
    """Stub for alignment model (not needed for basic transcription)"""
    return None, None


def load_audio(file: str, sr: int = 16000):
    """Load audio file"""
    import librosa
    audio, _ = librosa.load(file, sr=sr, mono=True)
    return audio


def align(segments, model_a, metadata, audio, device, **kwargs):
    """Stub for alignment (return segments as-is)"""
    return {"segments": segments}