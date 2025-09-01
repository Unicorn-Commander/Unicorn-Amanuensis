#!/usr/bin/env python3
"""
Real Whisper Intel iGPU Implementation using whisper.cpp SYCL
No mocks, no simulations - actual transcription!
"""

import os
import subprocess
import json
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

# Set up environment for Intel iGPU
os.environ["ONEAPI_ROOT"] = "/opt/intel/oneapi"
os.environ["SYCL_DEVICE_FILTER"] = "gpu"
os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperIGPUReal:
    """Real Whisper implementation using Intel iGPU via SYCL"""
    
    # Paths to whisper.cpp SYCL binary and models
    WHISPER_SYCL_PATH = "/tmp/whisper.cpp/build_sycl/bin/whisper-cli"
    MODEL_BASE_PATH = "/tmp/whisper.cpp/models"
    
    # Model mapping
    MODEL_FILES = {
        "tiny": "ggml-tiny.bin",
        "base": "ggml-base.bin", 
        "small": "ggml-small.bin",
        "medium": "ggml-medium.bin",
        "large": "ggml-large-v3.bin",
        "large-v3": "ggml-large-v3.bin",
        "large-v2": "ggml-large-v2.bin"
    }
    
    def __init__(self, model_name: str = "base", device: str = "gpu"):
        """Initialize with real whisper.cpp SYCL"""
        self.model_name = model_name
        self.device = device
        
        # Verify SYCL binary exists
        if not Path(self.WHISPER_SYCL_PATH).exists():
            raise RuntimeError(f"Whisper SYCL binary not found at {self.WHISPER_SYCL_PATH}")
        
        # Get model path
        self.model_path = self._get_model_path(model_name)
        
        logger.info(f"🚀 Real Whisper Intel iGPU initialized")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Device: Intel iGPU via SYCL")
        logger.info(f"   Binary: {self.WHISPER_SYCL_PATH}")
    
    def _get_model_path(self, model_name: str) -> str:
        """Get the full path to model file"""
        model_file = self.MODEL_FILES.get(model_name)
        if not model_file:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = Path(self.MODEL_BASE_PATH) / model_file
        if not model_path.exists():
            # Try to download the model
            logger.info(f"Model {model_name} not found, downloading...")
            self._download_model(model_name)
        
        return str(model_path)
    
    def _download_model(self, model_name: str):
        """Download model if not present"""
        model_file = self.MODEL_FILES.get(model_name)
        if not model_file:
            return
        
        # Use whisper.cpp's download script
        download_cmd = [
            "bash", "-c",
            f"cd /tmp/whisper.cpp && ./models/download-ggml-model.sh {model_name.replace('-v3', '').replace('-v2', '')}"
        ]
        
        try:
            result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"Failed to download model: {result.stderr}")
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
    
    def transcribe_file(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe audio file using real whisper.cpp SYCL"""
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Get options
        language = kwargs.get('language', 'auto')
        word_timestamps = kwargs.get('word_timestamps', True)
        max_len = kwargs.get('max_len', 0)
        threads = kwargs.get('threads', 4)
        
        # Start timing
        start_time = time.time()
        
        # Get audio duration for RTF calculation
        duration = self._get_audio_duration(audio_path)
        
        logger.info(f"🎵 Processing {duration:.1f}s audio with {self.model_name} model")
        
        # Build command for real SYCL execution
        cmd = [
            self.WHISPER_SYCL_PATH,
            "-m", self.model_path,
            "-f", audio_path,
            "-t", str(threads),
            "--print-progress"
        ]
        
        if language != 'auto':
            cmd.extend(["-l", language])
        
        if max_len > 0:
            cmd.extend(["--max-len", str(max_len)])
        
        logger.info(f"⚡ Executing on Intel iGPU with SYCL...")
        
        # Run the real transcription
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env={**os.environ, 
                     "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
                     "SYCL_DEVICE_FILTER": "gpu"}
            )
            
            processing_time = time.time() - start_time
            
            if result.returncode != 0:
                logger.error(f"Whisper SYCL failed: {result.stderr}")
                raise RuntimeError(f"Transcription failed: {result.stderr}")
            
            # Parse the text output
            transcription_result = self._parse_text_output(result.stdout)
            
            # Calculate performance
            rtf = processing_time / duration if duration > 0 else 0
            
            logger.info(f"✅ Real transcription complete!")
            logger.info(f"   Processing time: {processing_time:.2f}s")
            logger.info(f"   Real-time factor: {1/rtf:.1f}x realtime")
            
            # Format result
            return {
                "text": transcription_result.get("text", ""),
                "segments": transcription_result.get("segments", []),
                "language": transcription_result.get("language", language),
                "duration": duration,
                "performance": {
                    "total_time": f"{processing_time:.2f}s",
                    "rtf": f"{1/rtf:.1f}x",
                    "engine": "Intel iGPU SYCL (Real)"
                },
                "config": {
                    "model": self.model_name,
                    "device": "Intel UHD Graphics 770",
                    "backend": "whisper.cpp SYCL"
                }
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Transcription timed out")
            raise RuntimeError("Transcription timed out after 10 minutes")
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
        finally:
            pass  # No temp files to clean up
    
    def _parse_text_output(self, text_output: str) -> Dict[str, Any]:
        """Parse whisper.cpp text output"""
        import re
        
        segments = []
        full_text = ""
        
        # Parse timestamped segments like [00:00:00.000 --> 00:00:04.640] text
        pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s+(.*)'
        
        for match in re.finditer(pattern, text_output):
            start_time = self._timestamp_to_seconds(match.group(1))
            end_time = self._timestamp_to_seconds(match.group(2))
            text = match.group(3).strip()
            
            segments.append({
                "start": start_time,
                "end": end_time,
                "text": text
            })
            
            full_text += text + " "
        
        # If no segments found, treat entire output as text
        if not segments and text_output.strip():
            full_text = text_output.strip()
            segments = [{"start": 0.0, "end": 0.0, "text": full_text}]
        
        return {
            "text": full_text.strip(),
            "segments": segments,
            "language": "en"
        }
    
    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert HH:MM:SS.mmm to seconds"""
        parts = timestamp.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    def _parse_whisper_output(self, json_path: str) -> Dict[str, Any]:
        """Parse whisper.cpp JSON output"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract segments with timestamps
            segments = []
            for seg in data.get("transcription", []):
                segment = {
                    "start": seg.get("offsets", {}).get("from", 0) / 1000.0,
                    "end": seg.get("offsets", {}).get("to", 0) / 1000.0,
                    "text": seg.get("text", "").strip()
                }
                
                # Add word-level timestamps if available
                if "timestamps" in seg:
                    segment["words"] = []
                    for token in seg.get("timestamps", {}).get("tokens", []):
                        segment["words"].append({
                            "start": token.get("offsets", {}).get("from", 0) / 1000.0,
                            "end": token.get("offsets", {}).get("to", 0) / 1000.0,
                            "text": token.get("text", ""),
                            "probability": token.get("p", 1.0)
                        })
                
                segments.append(segment)
            
            # Get full text
            full_text = " ".join(seg["text"] for seg in segments)
            
            return {
                "text": full_text,
                "segments": segments,
                "language": data.get("result", {}).get("language", "en")
            }
        except Exception as e:
            logger.error(f"Error parsing whisper output: {e}")
            return {"text": "", "segments": [], "language": "en"}
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', 
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0', audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0.0
    
    def transcribe_with_diarization(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe with speaker diarization using pyannote"""
        # First get regular transcription
        result = self.transcribe_file(audio_path, **kwargs)
        
        # Add diarization if requested
        if kwargs.get('diarization', False):
            try:
                from pyannote.audio import Pipeline
                import torch
                
                logger.info("🎭 Adding speaker diarization...")
                
                # Initialize diarization pipeline
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=os.environ.get("HF_TOKEN")
                )
                
                if torch.cuda.is_available():
                    pipeline.to(torch.device("cuda"))
                
                # Run diarization
                diarization = pipeline(audio_path)
                
                # Convert to segments
                speakers = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    speakers.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker
                    })
                
                result["speakers"] = speakers
                logger.info(f"✅ Identified {len(set(s['speaker'] for s in speakers))} speakers")
                
            except Exception as e:
                logger.warning(f"Diarization failed: {e}")
                result["speakers"] = []
        
        return result


def test_real_implementation():
    """Test the real implementation"""
    try:
        # Initialize real whisper
        whisper = WhisperIGPUReal('base')
        
        # Create test audio if needed
        test_audio = "/tmp/test_audio.wav"
        if not Path(test_audio).exists():
            # Generate test audio using TTS
            import subprocess
            subprocess.run([
                "espeak", "-w", test_audio,
                "This is a real test of the Intel iGPU transcription system."
            ], check=True)
        
        # Transcribe
        result = whisper.transcribe_file(test_audio)
        
        print(f"✅ Real implementation test successful!")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Performance: {result['performance']['rtf']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real implementation test failed: {e}")
        return False


if __name__ == "__main__":
    test_real_implementation()