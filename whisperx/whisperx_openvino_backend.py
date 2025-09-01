#!/usr/bin/env python3
"""
Custom WhisperX backend using OpenVINO for Intel iGPU acceleration
This replaces the CTranslate2 backend with OpenVINO inference
"""

import os
import logging
import numpy as np
import torch
import openvino as ov
from pathlib import Path
from typing import Optional, Dict, Any, List
import whisperx
from whisperx.audio import load_audio, log_mel_spectrogram
from whisperx.types import TranscriptionResult
import warnings

logger = logging.getLogger(__name__)

class OpenVINOWhisperModel:
    """Custom Whisper model using OpenVINO backend instead of CTranslate2"""
    
    def __init__(self, model_path: str, device: str = "GPU", compute_type: str = "int8"):
        """Initialize OpenVINO-accelerated Whisper model"""
        
        self.device = device
        self.compute_type = compute_type
        
        # Initialize OpenVINO runtime
        self.core = ov.Core()
        available_devices = self.core.available_devices
        logger.info(f"Available OpenVINO devices: {available_devices}")
        
        # Select the best device
        if device in available_devices:
            self.ov_device = device
        elif "GPU.0" in available_devices:
            self.ov_device = "GPU.0"
        elif "GPU" in available_devices:
            self.ov_device = "GPU"
        else:
            self.ov_device = "CPU"
            logger.warning(f"Device {device} not found, using {self.ov_device}")
        
        logger.info(f"Using OpenVINO device: {self.ov_device}")
        
        # Load the OpenVINO model
        model_xml = Path(model_path) / "openvino_encoder_model.xml"
        model_bin = Path(model_path) / "openvino_encoder_model.bin"
        decoder_xml = Path(model_path) / "openvino_decoder_model.xml"
        decoder_bin = Path(model_path) / "openvino_decoder_model.bin"
        
        if model_xml.exists() and decoder_xml.exists():
            logger.info(f"Loading OpenVINO models from {model_path}")
            
            # Read and compile encoder
            self.encoder_model = self.core.read_model(model_xml, model_bin)
            self.compiled_encoder = self.core.compile_model(
                self.encoder_model, 
                self.ov_device,
                config={
                    "PERFORMANCE_HINT": "LATENCY",
                    "GPU_THROUGHPUT_STREAMS": "1" if "GPU" in self.ov_device else "AUTO"
                }
            )
            
            # Read and compile decoder
            self.decoder_model = self.core.read_model(decoder_xml, decoder_bin)
            self.compiled_decoder = self.core.compile_model(
                self.decoder_model,
                self.ov_device,
                config={
                    "PERFORMANCE_HINT": "LATENCY",
                    "GPU_THROUGHPUT_STREAMS": "1" if "GPU" in self.ov_device else "AUTO"
                }
            )
            
            logger.info(f"✅ OpenVINO models loaded and compiled for {self.ov_device}")
        else:
            raise FileNotFoundError(f"OpenVINO model files not found in {model_path}")
        
        # Create inference requests
        self.encoder_request = self.compiled_encoder.create_infer_request()
        self.decoder_request = self.compiled_decoder.create_infer_request()
        
        # Load tokenizer and config
        self._load_tokenizer(model_path)
        
    def _load_tokenizer(self, model_path: str):
        """Load tokenizer and model configuration"""
        from transformers import WhisperTokenizer, WhisperProcessor
        
        try:
            self.processor = WhisperProcessor.from_pretrained(model_path)
            self.tokenizer = self.processor.tokenizer
        except:
            # Fallback to loading from original model
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.tokenizer = self.processor.tokenizer
    
    def transcribe(self, audio: np.ndarray, batch_size: int = 16, 
                   language: Optional[str] = None, task: str = "transcribe",
                   chunk_length: int = 30, **kwargs) -> TranscriptionResult:
        """Transcribe audio using OpenVINO inference on iGPU"""
        
        # Prepare audio features
        features = log_mel_spectrogram(audio, padding=0)
        
        # Run encoder inference on iGPU
        self.encoder_request.infer({0: features})
        encoder_output = self.encoder_request.get_output_tensor(0).data
        
        # Generate tokens using decoder on iGPU
        tokens = self._generate_tokens(encoder_output, language, task)
        
        # Decode tokens to text
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Create result similar to WhisperX format
        result = {
            "text": text,
            "segments": self._create_segments(tokens, text),
            "language": language or "en"
        }
        
        return result
    
    def _generate_tokens(self, encoder_output: np.ndarray, 
                        language: Optional[str], task: str) -> List[int]:
        """Generate tokens using decoder with iGPU acceleration"""
        
        # Initialize decoder input with start tokens
        sot_sequence = self.tokenizer.prefix_tokens(task=task, language=language)
        tokens = list(sot_sequence)
        
        max_length = 448  # Whisper's max token length
        
        while len(tokens) < max_length:
            # Prepare decoder input
            decoder_input = np.array([tokens], dtype=np.int64)
            
            # Run decoder inference on iGPU
            self.decoder_request.infer({
                "input_ids": decoder_input,
                "encoder_hidden_states": encoder_output
            })
            
            # Get logits and select next token
            logits = self.decoder_request.get_output_tensor(0).data
            next_token = np.argmax(logits[0, -1, :])
            
            # Check for end of sequence
            if next_token == self.tokenizer.eos_token_id:
                break
                
            tokens.append(int(next_token))
        
        return tokens
    
    def _create_segments(self, tokens: List[int], text: str) -> List[Dict]:
        """Create segment information for compatibility"""
        return [{
            "start": 0.0,
            "end": 30.0,  # Placeholder
            "text": text,
            "tokens": tokens,
            "temperature": 0.0,
            "avg_logprob": 0.0,
            "compression_ratio": 1.0,
            "no_speech_prob": 0.0
        }]


def load_model_openvino(whisper_arch: str, device: str = "GPU", 
                        compute_type: str = "int8", 
                        download_root: str = None, **kwargs):
    """
    Load Whisper model with OpenVINO backend for iGPU acceleration
    This function replaces whisperx.load_model for OpenVINO inference
    """
    
    # Map model names to paths
    model_paths = {
        "large-v3": "/home/ucadmin/Unicorn-Amanuensis/models/whisper-large-v3-openvino",
        "base": "/home/ucadmin/Unicorn-Amanuensis/models/whisper-base-openvino",
    }
    
    model_path = model_paths.get(whisper_arch)
    
    if not model_path or not Path(model_path).exists():
        # Convert model if not available
        logger.info(f"OpenVINO model not found, converting {whisper_arch}...")
        model_path = convert_whisper_to_openvino(f"openai/whisper-{whisper_arch}")
    
    # Return our custom OpenVINO model
    return OpenVINOWhisperModel(model_path, device, compute_type)


def convert_whisper_to_openvino(model_id: str, output_dir: str = None) -> str:
    """Convert Whisper model to OpenVINO format on the fly"""
    from optimum.intel import OVModelForSpeechSeq2Seq
    from transformers import WhisperProcessor
    
    if output_dir is None:
        model_name = model_id.split("/")[-1]
        output_dir = f"/home/ucadmin/Unicorn-Amanuensis/models/{model_name}-openvino"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting {model_id} to OpenVINO format...")
    
    # Convert to OpenVINO
    ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        export=True,
        compile=False,
        device="GPU"
    )
    
    # Save model and processor
    ov_model.save_pretrained(output_dir)
    processor = WhisperProcessor.from_pretrained(model_id)
    processor.save_pretrained(output_dir)
    
    logger.info(f"✅ Model converted and saved to {output_dir}")
    
    return output_dir


# Monkey-patch WhisperX to use our OpenVINO backend
def patch_whisperx_for_openvino():
    """Replace WhisperX's CTranslate2 backend with OpenVINO"""
    import whisperx
    
    # Save original function
    whisperx._original_load_model = whisperx.load_model
    
    # Replace with our OpenVINO version
    whisperx.load_model = load_model_openvino
    
    logger.info("✅ WhisperX patched to use OpenVINO iGPU backend")


if __name__ == "__main__":
    # Test the OpenVINO backend
    patch_whisperx_for_openvino()
    
    # Now whisperx.load_model will use OpenVINO
    import whisperx
    
    model = whisperx.load_model("large-v3", device="GPU", compute_type="int8")
    print(f"Model loaded with OpenVINO backend: {model.ov_device}")