#!/usr/bin/env python3
"""
NPU Runtime using ONNXRuntime with proper execution providers
"""

import os
import numpy as np
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import time
import wave

logger = logging.getLogger(__name__)

# Check for CPU-only mode
CPU_ONLY_MODE = os.environ.get('CPU_ONLY_MODE', '').lower() in ('1', 'true', 'yes')

class NPURuntime:
    """Main NPURuntime class using ONNXRuntime"""

    def __init__(self):
        if CPU_ONLY_MODE:
            logger.info("🖥️ NPURuntime: Running in CPU-only mode")
            self.available = False
            self._runtime = None
        else:
            self._runtime = ONNXNPURuntime()
            self.available = self._runtime.is_available()

    def is_available(self) -> bool:
        return self.available

    def load_model(self, model_path: str) -> bool:
        if not self.available:
            return False
        return self._runtime.load_model(model_path)

    def transcribe(self, audio_data) -> Dict[str, Any]:
        if not self.available:
            return {"error": "NPU not available", "text": "", "npu_accelerated": False}
        return self._runtime.transcribe(audio_data)

    def get_device_info(self) -> Dict[str, Any]:
        if not self.available or not self._runtime:
            return {"status": "not available"}
        return self._runtime.get_device_info()

class ONNXNPURuntime:
    """ONNX Runtime with NPU/CPU execution"""

    def __init__(self):
        self.device_path = "/dev/accel/accel0"
        self.encoder_session = None
        self.decoder_session = None
        self.model_loaded = False
        self.execution_provider = None
        self.available = self._check_device()

    def _check_device(self) -> bool:
        """Check if NPU device is available"""
        if CPU_ONLY_MODE:
            logger.info("🖥️ Running in CPU-only mode")
            return False

        if not Path(self.device_path).exists():
            logger.warning(f"❌ NPU device not found: {self.device_path}")
            return False

        logger.info(f"✅ NPU device found: {self.device_path}")
        return True

    def is_available(self) -> bool:
        """Check if NPU is available"""
        return self.available

    def load_model(self, model_path: str) -> bool:
        """Load ONNX Whisper models"""
        try:
            import onnxruntime as ort

            logger.info(f"🔄 Loading ONNX Whisper model: {model_path}")

            # Check for AMD NPU INT8 models
            npu_model_path = os.environ.get('WHISPER_NPU_MODEL_PATH', '/app/models/whisper-base-amd-npu-int8')

            if os.path.exists(npu_model_path):
                model_dir = Path(npu_model_path)
                logger.info(f"📁 Using AMD NPU INT8 models from: {npu_model_path}")
            elif model_path == "whisper-base":
                model_dir = Path("/app/models/whisper-base-onnx")
            else:
                model_dir = Path(model_path)

            # Look for ONNX models
            possible_paths = [
                (model_dir / "onnx/encoder_model_int8.onnx", model_dir / "onnx/decoder_model_int8.onnx"),
                (model_dir / "encoder_model_int8.onnx", model_dir / "decoder_model_int8.onnx"),
                (model_dir / "onnx/encoder_model.onnx", model_dir / "onnx/decoder_model.onnx"),
                (model_dir / "encoder_model.onnx", model_dir / "decoder_model.onnx"),
            ]

            encoder_path = None
            decoder_path = None

            for enc, dec in possible_paths:
                if enc.exists() and dec.exists():
                    encoder_path = enc
                    decoder_path = dec
                    break

            if not encoder_path or not decoder_path:
                logger.error(f"❌ ONNX models not found in {model_dir}")
                return False

            logger.info(f"📦 Loading ONNX models...")
            logger.info(f"   Encoder: {encoder_path}")
            logger.info(f"   Decoder: {decoder_path}")

            # Check available execution providers
            available_providers = ort.get_available_providers()
            logger.info(f"📋 Available execution providers: {available_providers}")

            # Try to use NPU/OpenVINO provider first, fallback to CPU
            if 'VitisAIExecutionProvider' in available_providers and self.available:
                providers = ['VitisAIExecutionProvider', 'CPUExecutionProvider']
                self.execution_provider = 'VitisAIExecutionProvider'
                logger.info("🚀 Using VitisAI Execution Provider (NPU)")
            elif 'OpenVINOExecutionProvider' in available_providers:
                providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
                self.execution_provider = 'OpenVINOExecutionProvider'
                logger.info("🚀 Using OpenVINO Execution Provider (supports INT8)")
            else:
                providers = ['CPUExecutionProvider']
                self.execution_provider = 'CPUExecutionProvider'
                logger.warning("⚠️ No accelerated EP available, using CPU fallback")

            # Create ONNX Runtime sessions
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.encoder_session = ort.InferenceSession(
                str(encoder_path),
                sess_options=sess_options,
                providers=providers
            )

            self.decoder_session = ort.InferenceSession(
                str(decoder_path),
                sess_options=sess_options,
                providers=providers
            )

            # Get model info
            encoder_inputs = self.encoder_session.get_inputs()
            decoder_inputs = self.decoder_session.get_inputs()

            logger.info(f"✅ Encoder loaded: {len(encoder_inputs)} inputs")
            logger.info(f"✅ Decoder loaded: {len(decoder_inputs)} inputs")
            logger.info(f"✅ Execution provider: {self.execution_provider}")

            self.model_loaded = True
            self.model_dir = model_dir
            return True

        except Exception as e:
            logger.error(f"❌ Model loading error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def transcribe(self, audio_data: Union[np.ndarray, bytes, str]) -> Dict[str, Any]:
        """Transcribe audio using ONNX Runtime"""
        if not self.model_loaded:
            return {"error": "Model not loaded", "text": "", "npu_accelerated": False}

        try:
            start_time = time.time()

            # Load audio
            if isinstance(audio_data, str) and os.path.exists(audio_data):
                logger.info(f"📁 Loading audio from: {audio_data}")
                with wave.open(audio_data, 'rb') as wav:
                    frames = wav.readframes(wav.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    sample_rate = wav.getframerate()
                    duration = len(audio_array) / sample_rate
                    logger.info(f"🎵 Audio loaded: {duration:.1f}s at {sample_rate}Hz")
            elif isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                duration = len(audio_array) / 16000
                sample_rate = 16000
            else:
                audio_array = audio_data.astype(np.float32)
                duration = len(audio_array) / 16000
                sample_rate = 16000

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            # Compute mel spectrogram (Whisper preprocessing)
            import librosa

            # Whisper uses 80 mel bins, 400-sample windows, 160-sample hop
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio_array,
                sr=16000,
                n_mels=80,
                n_fft=400,
                hop_length=160
            )
            log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)

            # Pad or trim to 30 seconds (3000 frames)
            target_frames = 3000
            if log_mel.shape[1] < target_frames:
                # Pad with zeros
                pad_width = target_frames - log_mel.shape[1]
                log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # Trim
                log_mel = log_mel[:, :target_frames]

            # Prepare encoder input (add batch dimension)
            encoder_input = log_mel.T[np.newaxis, :, :].astype(np.float32)  # Shape: (1, 3000, 80)

            logger.info(f"🎵 Mel spectrogram: {encoder_input.shape}")
            logger.info(f"⚡ Running {self.execution_provider} inference...")

            # Run encoder
            encoder_outputs = self.encoder_session.run(None, {
                self.encoder_session.get_inputs()[0].name: encoder_input
            })

            encoder_hidden_states = encoder_outputs[0]
            logger.info(f"✅ Encoder output: {encoder_hidden_states.shape}")

            # Simple greedy decoding (start token = 50258 for Whisper)
            decoder_input_ids = np.array([[50258]], dtype=np.int64)  # Start token
            max_length = 448
            generated_tokens = [50258]

            # Decoder loop
            for i in range(max_length):
                decoder_outputs = self.decoder_session.run(None, {
                    self.decoder_session.get_inputs()[0].name: decoder_input_ids,
                    self.decoder_session.get_inputs()[1].name: encoder_hidden_states
                })

                # Get logits for next token
                logits = decoder_outputs[0][0, -1, :]  # Last token logits
                next_token = int(np.argmax(logits))

                # Check for end token (50257 for Whisper)
                if next_token == 50257:
                    break

                generated_tokens.append(next_token)
                decoder_input_ids = np.array([generated_tokens], dtype=np.int64)

            processing_time = time.time() - start_time

            # Decode tokens to text
            try:
                from transformers import WhisperTokenizer
                tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
                text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            except:
                # Fallback if tokenizer not available
                text = f"Transcribed {len(generated_tokens)} tokens (tokenizer not available for decoding)"

            logger.info(f"✅ Transcription complete: '{text}'")
            logger.info(f"⏱️  Processing time: {processing_time:.2f}s ({duration/processing_time:.1f}x realtime)")

            return {
                "text": text,
                "segments": [{
                    "start": 0.0,
                    "end": duration,
                    "text": text
                }],
                "language": "en",
                "npu_accelerated": self.execution_provider in ['VitisAIExecutionProvider', 'OpenVINOExecutionProvider'],
                "processing_time": processing_time,
                "audio_duration": duration,
                "speedup": duration / processing_time if processing_time > 0 else 1000,
                "device_info": {
                    "execution_provider": self.execution_provider,
                    "npu_device": self.device_path if self.available else "N/A",
                    "status": "operational"
                }
            }

        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "text": f"Transcription failed: {str(e)}",
                "npu_accelerated": False
            }

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        return {
            "device_path": self.device_path,
            "device_available": self.available,
            "model_loaded": self.model_loaded,
            "execution_provider": self.execution_provider,
            "driver": "onnxruntime",
            "status": "operational" if self.model_loaded else "waiting"
        }

# Alias for compatibility
SimplifiedNPURuntime = ONNXNPURuntime
RealNPURuntime = ONNXNPURuntime
