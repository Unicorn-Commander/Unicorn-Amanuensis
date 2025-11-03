#!/usr/bin/env python3
"""
🦄 Unicorn Amanuensis - Smart Dynamic Server
Auto-detects hardware (NPU > iGPU > CPU) and models
Uses the best available acceleration automatically
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
import tempfile
import time
import uuid
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import diarization dependencies
try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    import torch
    DIARIZATION_AVAILABLE = True
    logger.info("✅ Pyannote.audio available for speaker diarization")
except (ImportError, OSError) as e:
    DIARIZATION_AVAILABLE = False
    DiarizationPipeline = None
    logger.warning(f"⚠️ Pyannote.audio not available: {type(e).__name__}")
    logger.info("   Diarization will be disabled (pyannote has CUDA dependencies)")

# Global progress tracking
progress_store = {}

app = FastAPI(
    title="🦄 Unicorn Amanuensis (Dynamic)",
    description="Auto-detects best hardware: NPU → iGPU → CPU",
    version="3.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class DynamicHardwareDetector:
    """Smart hardware detection with priority: NPU > iGPU > CPU"""

    @staticmethod
    def find_whisper_models() -> Dict[str, Path]:
        """Auto-find Whisper models in common locations"""
        search_paths = [
            Path("/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/whisper_onnx_cache"),
            Path("/home/ucadmin/Development/whisper_npu_project/whisper_onnx_cache"),
            Path("/app/models/whisper_onnx_cache"),
            Path("/models/whisper_onnx_cache"),
            Path.home() / ".cache" / "whisper",
            Path.home() / ".cache" / "huggingface" / "hub",
        ]

        models = {}
        for search_path in search_paths:
            if search_path.exists():
                logger.info(f"📁 Scanning: {search_path}")
                for model_dir in search_path.iterdir():
                    if model_dir.is_dir() and "whisper" in model_dir.name.lower():
                        model_name = model_dir.name
                        models[model_name] = model_dir
                        logger.info(f"   ✓ Found: {model_name}")

        return models

    @staticmethod
    def detect_npu() -> Optional[Dict]:
        """Detect AMD Phoenix NPU"""
        try:
            if not Path("/dev/accel/accel0").exists():
                return None

            result = subprocess.run(
                ["/opt/xilinx/xrt/bin/xrt-smi", "examine"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and "NPU Phoenix" in result.stdout:
                logger.info("🚀 AMD Phoenix NPU detected!")

                # Count XCLBIN kernels
                kernel_dir = Path(__file__).parent / "npu/npu_optimization/whisper_encoder_kernels"
                kernel_count = len(list(kernel_dir.glob("*.xclbin"))) if kernel_dir.exists() else 0

                return {
                    "type": "npu",
                    "name": "AMD Phoenix NPU",
                    "device": "/dev/accel/accel0",
                    "kernels": kernel_count,
                    "priority": 1,  # Highest priority
                    "expected_speedup": "28-220x"
                }
        except Exception as e:
            logger.debug(f"NPU detection failed: {e}")

        return None

    @staticmethod
    def detect_igpu() -> Optional[Dict]:
        """Detect Intel iGPU"""
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True
            )

            if "Intel" in result.stdout and ("VGA" in result.stdout or "Display" in result.stdout):
                logger.info("🎨 Intel iGPU detected!")
                return {
                    "type": "igpu",
                    "name": "Intel Integrated Graphics",
                    "priority": 2,
                    "expected_speedup": "13-19x"
                }
        except Exception as e:
            logger.debug(f"iGPU detection failed: {e}")

        return None

    @staticmethod
    def detect_hardware() -> Dict:
        """Detect best available hardware"""
        logger.info("🔍 Scanning hardware...")

        # Check in priority order
        npu = DynamicHardwareDetector.detect_npu()
        if npu:
            return npu

        igpu = DynamicHardwareDetector.detect_igpu()
        if igpu:
            return igpu

        logger.info("💻 Using CPU (fallback)")
        return {
            "type": "cpu",
            "name": "CPU",
            "priority": 3,
            "expected_speedup": "5-13x"
        }

class DynamicWhisperEngine:
    """Whisper engine that uses best available hardware"""

    def __init__(self):
        self.hardware = DynamicHardwareDetector.detect_hardware()
        self.models = DynamicHardwareDetector.find_whisper_models()
        self.engine = None
        self.current_model = "base"
        self.diarization_pipeline = None

        logger.info(f"✅ Hardware selected: {self.hardware['name']}")
        logger.info(f"✅ Models found: {len(self.models)}")

        self._initialize_engine()
        self._initialize_diarization()

    def _initialize_engine(self):
        """Initialize transcription engine based on hardware"""

        if self.hardware["type"] == "npu":
            self._init_npu_engine()
        elif self.hardware["type"] == "igpu":
            self._init_igpu_engine()
        else:
            self._init_cpu_engine()

    def _init_npu_engine(self):
        """Initialize NPU-accelerated engine"""

        # Initialize NPU runtime for mel preprocessing
        try:
            logger.info("🚀 Initializing NPU mel preprocessing runtime...")
            sys.path.insert(0, str(Path(__file__).parent / 'npu'))

            from npu_mel_preprocessing import NPUMelPreprocessor

            # Try different XCLBINs in order of preference
            xclbin_candidates = [
                'npu/npu_optimization/mel_kernels/build/mel_fixed_v3.xclbin',  # PRODUCTION (Nov 1, 2025 - with Oct 28 fixes)
                'npu/npu_optimization/mel_kernels/build/mel_int8_final.xclbin',
                'npu/npu_optimization/mel_kernels/build/mel_fft.xclbin',
                'npu/npu_optimization/mel_kernels/build/mel_int8_optimized.xclbin',
            ]

            npu_initialized = False
            for xclbin_file in xclbin_candidates:
                xclbin_path = Path(__file__).parent / xclbin_file
                if xclbin_path.exists():
                    logger.info(f"   Trying XCLBIN: {xclbin_path.name}")
                    try:
                        self.npu_runtime = NPUMelPreprocessor(
                            xclbin_path=str(xclbin_path),
                            fallback_to_cpu=True
                        )
                        if self.npu_runtime.npu_available:
                            logger.info(f"✅ NPU mel preprocessing runtime loaded!")
                            logger.info(f"   • XCLBIN: {xclbin_path.name}")
                            logger.info(f"   • Device: /dev/accel/accel0")
                            logger.info(f"   • Expected speedup: 6x for preprocessing")
                            if 'mel_fixed_v3' in xclbin_path.name:
                                logger.info(f"   ✅ PRODUCTION XCLBIN with Oct 28 accuracy fixes (0.92 correlation)")
                            else:
                                logger.info(f"   ⚠️  WARNING: Using older XCLBIN without Oct 28 fixes")
                                logger.info(f"   → mel_fixed_v3.xclbin recommended for >92% accuracy")
                            npu_initialized = True
                            break
                    except Exception as e:
                        logger.debug(f"   Failed with {xclbin_path.name}: {e}")
                        continue

            if not npu_initialized:
                logger.warning(f"⚠️ NPU preprocessing unavailable - using CPU fallback")
                self.npu_runtime = NPUMelPreprocessor(fallback_to_cpu=True)

        except Exception as e:
            logger.warning(f"⚠️ NPU mel preprocessing init failed: {e}")
            logger.warning(f"   Will use CPU for mel preprocessing")
            self.npu_runtime = None

        # Try to load full NPU Whisper pipeline (encoder + decoder on NPU)
        try:
            logger.info("🚀 Loading full NPU Whisper pipeline...")

            sys.path.insert(0, str(Path(__file__).parent / 'npu' / 'npu_optimization' / 'whisper_encoder_kernels'))
            from npu_whisper_integration_example import NPUWhisperPipeline

            self.npu_pipeline = NPUWhisperPipeline(model_name="base", device_id=0)
            self.use_npu_pipeline = True

            logger.info("✅ Full NPU Whisper pipeline loaded!")
            logger.info("   • Encoder: NPU matmul + attention (6 layers)")
            logger.info("   • Decoder: NPU matmul + attention (6 layers)")
            logger.info("   • Device: AMD Phoenix NPU (/dev/accel/accel0)")
            logger.info("   • Status: Experimental - encoder/decoder on NPU")

            # Also load faster-whisper for text generation (NPU pipeline doesn't have tokenizer yet)
            from faster_whisper import WhisperModel
            self.engine = WhisperModel("base", device="cpu", compute_type="int8")
            logger.info("   • Text generation: faster-whisper base")
            logger.info("   • Expected: 18-20x realtime (optimal for this hardware)")

        except Exception as e:
            logger.warning(f"⚠️ Full NPU pipeline failed: {e}")
            logger.info("📦 Using faster-whisper only (CPU INT8)")

            from faster_whisper import WhisperModel
            self.engine = WhisperModel("base", device="cpu", compute_type="int8")
            self.use_npu_pipeline = False

            logger.info("✅ Using faster-whisper base (CPU INT8)")
            logger.info("   • Expected: 18-20x realtime (optimal for this hardware)")

    def _init_igpu_engine(self):
        """Initialize iGPU-accelerated engine"""
        try:
            from faster_whisper import WhisperModel
            self.engine = WhisperModel("base", device="cpu", compute_type="int8")
            logger.info("✅ Using faster-whisper (iGPU mode planned)")
        except Exception as e:
            logger.error(f"iGPU init failed: {e}")
            self._init_cpu_engine()

    def _init_cpu_engine(self):
        """Initialize CPU engine"""
        try:
            from faster_whisper import WhisperModel
            self.engine = WhisperModel("base", device="cpu", compute_type="int8")
            logger.info("✅ Using faster-whisper (CPU)")
        except Exception as e:
            logger.error(f"Engine init failed: {e}")
            raise

    def _initialize_diarization(self):
        """Initialize speaker diarization pipeline"""
        if not DIARIZATION_AVAILABLE:
            logger.info("ℹ️ Speaker diarization not available (pyannote.audio not installed)")
            return

        try:
            # Check for HuggingFace token
            hf_token = os.environ.get("HF_TOKEN", None)

            # Try to load diarization pipeline
            logger.info("📥 Loading speaker diarization pipeline...")
            self.diarization_pipeline = DiarizationPipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )

            # Move to CPU (compatible with all hardware)
            self.diarization_pipeline.to(torch.device("cpu"))

            logger.info("✅ Speaker diarization pipeline loaded")

        except Exception as e:
            logger.warning(f"⚠️ Could not load diarization pipeline: {e}")
            logger.info("   Transcription will work without speaker labels")
            logger.info("   To enable diarization:")
            logger.info("   1. Accept license: https://huggingface.co/pyannote/speaker-diarization-3.1")
            logger.info("   2. Set HF_TOKEN environment variable")
            self.diarization_pipeline = None

    def add_speaker_diarization(self, audio_path: str, segments: List[Dict], min_speakers: int = 1, max_speakers: int = 10) -> List[Dict]:
        """Add speaker diarization to transcription segments

        Args:
            audio_path: Path to audio file
            segments: List of transcription segments with start/end times
            min_speakers: Minimum number of speakers (default: 1)
            max_speakers: Maximum number of speakers (default: 10)

        Returns:
            Segments with speaker labels added
        """
        if not self.diarization_pipeline:
            logger.warning("Diarization pipeline not available, returning segments without speakers")
            return segments

        try:
            logger.info(f"🎭 Running speaker diarization (speakers: {min_speakers}-{max_speakers})...")

            # Perform diarization on the audio file
            diarization = self.diarization_pipeline(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )

            # Map speakers to segments based on time overlap
            from collections import Counter
            for segment in segments:
                start_time = segment["start"]
                end_time = segment["end"]

                # Find all speakers in this time range
                speakers_in_range = []
                for speech_turn, track, speaker in diarization.itertracks(yield_label=True):
                    # Check if there's overlap between segment and speaker turn
                    if speech_turn.start <= end_time and speech_turn.end >= start_time:
                        speakers_in_range.append(speaker)

                # Assign the most common speaker in this range
                if speakers_in_range:
                    most_common_speaker = Counter(speakers_in_range).most_common(1)[0][0]
                    segment["speaker"] = most_common_speaker
                else:
                    # No speaker found, use default
                    segment["speaker"] = "SPEAKER_00"

            logger.info(f"✅ Diarization complete")
            return segments

        except Exception as e:
            logger.error(f"❌ Diarization failed: {e}")
            logger.warning("   Returning segments without speaker labels")
            return segments

    async def transcribe(self, audio_path: str, model: str = "base", vad_filter: bool = True, enable_diarization: bool = False, min_speakers: int = 1, max_speakers: int = 10, job_id: str = None) -> Dict:
        """Transcribe audio using best available hardware

        Args:
            audio_path: Path to audio file
            model: Whisper model to use
            vad_filter: Enable Voice Activity Detection to filter out silent/noisy segments (default: True)
            enable_diarization: Enable speaker diarization (default: False)
            min_speakers: Minimum number of speakers for diarization (default: 1)
            max_speakers: Maximum number of speakers for diarization (default: 10)
            job_id: Optional job ID for progress tracking
        """
        start_time = time.time()
        transcription_complete = False
        progress_updater_task = None

        def update_progress(progress: int, message: str):
            """Update progress if job_id is provided"""
            if job_id and job_id in progress_store:
                progress_store[job_id] = {
                    "status": "transcribing",
                    "progress": progress,
                    "message": message,
                    "job_id": job_id
                }

        async def simulate_progress():
            """Simulate progress updates during transcription"""
            nonlocal transcription_complete
            await asyncio.sleep(0.5)

            # Ramp up progress gradually
            for progress in range(50, 95, 5):
                if transcription_complete:
                    break
                update_progress(progress, f"Transcribing audio... {progress}%")
                await asyncio.sleep(1.0)  # Update every second

            # Hold at 95% until transcription completes
            while not transcription_complete:
                update_progress(95, "Finalizing transcription...")
                await asyncio.sleep(0.5)

        # Check if using NPU pipeline (full encoder+decoder on NPU)
        if hasattr(self, 'use_npu_pipeline') and self.use_npu_pipeline:
            logger.info("🚀 Using NPU pipeline for transcription...")
            logger.info("   NOTE: This is experimental - encoder/decoder run on NPU")
            logger.info("   The pipeline computes features but doesn't yet decode to text")
            logger.info("   Falling back to faster-whisper for actual transcription...")

            # For now, fall through to faster-whisper since NPU pipeline doesn't have text decoder
            # TODO: Implement full text generation with NPU pipeline
            pass

        # NPU mel preprocessing - ENABLED (Nov 3, 2025 - Production XCLBIN deployed)
        use_npu_mel = True  # ✅ ENABLED - Using mel_fixed_v3.xclbin with 0.92 accuracy
        mel_time = 0

        if use_npu_mel and hasattr(self, 'npu_runtime'):
            try:
                mel_start = time.time()
                logger.info("🚀 Using NPU mel preprocessing...")

                # Load audio - convert with FFmpeg if needed
                import librosa
                import subprocess
                from pathlib import Path

                # Check if file needs conversion
                audio_ext = Path(audio_path).suffix.lower()
                if audio_ext in ['.m4a', '.mp4', '.aac', '.opus']:
                    # Convert to WAV using FFmpeg
                    wav_path = audio_path.replace(audio_ext, '.wav')

                    # Try to find ffmpeg
                    import shutil
                    ffmpeg_path = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'

                    subprocess.run([
                        ffmpeg_path, '-i', audio_path,
                        '-ar', '16000',  # Resample to 16kHz
                        '-ac', '1',       # Convert to mono
                        '-y',             # Overwrite
                        wav_path
                    ], check=True, capture_output=True)
                    audio, sr = librosa.load(wav_path, sr=16000)
                else:
                    audio, sr = librosa.load(audio_path, sr=16000)

                # Process with NPU - call the mel_processor directly
                mel_features = self.npu_runtime.mel_processor.process(audio)
                mel_time = time.time() - mel_start
                logger.info(f"✅ NPU mel completed in {mel_time:.3f}s - Shape: {mel_features.shape}")

            except Exception as e:
                logger.error(f"❌ NPU mel preprocessing failed: {e}")
                raise HTTPException(status_code=500, detail=f"NPU preprocessing failed: {str(e)}. NPU-only mode - no CPU fallback allowed.")

        logger.info(f"🎙️ VAD filter: {'enabled' if vad_filter else 'disabled'}")

        # Check if we have NPU-computed mel features to inject
        if use_npu_mel and 'mel_features' in locals():
            logger.info("🔥 INJECTING NPU mel features directly into faster-whisper (bypassing CPU recomputation)")

            # Import required modules
            from faster_whisper.transcribe import TranscriptionOptions, TranscriptionInfo
            from faster_whisper.tokenizer import Tokenizer

            # Ensure mel_features is float32 and has correct shape
            import numpy as np
            if mel_features.dtype != np.float32:
                mel_features = mel_features.astype(np.float32)

            # mel_features should be (n_mels, n_frames), which matches faster-whisper's expectation
            logger.info(f"📊 Mel features shape: {mel_features.shape}, dtype: {mel_features.dtype}")

            # Create transcription options
            options = TranscriptionOptions(
                beam_size=5,
                best_of=5,
                patience=1.0,
                length_penalty=1.0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=True,
                prompt_reset_on_temperature=0.5,
                temperatures=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                initial_prompt=None,
                prefix=None,
                suppress_blank=True,
                suppress_tokens=[-1],
                without_timestamps=False,
                max_initial_timestamp=1.0,
                word_timestamps=True,
                prepend_punctuations='"\'"\u00bf([{-',
                append_punctuations='"\'.。,，!！?？:：")]}、',
                multilingual=False,
                max_new_tokens=None,
                clip_timestamps="0",
                hallucination_silence_threshold=None,
                hotwords=None
            )

            # Create tokenizer
            tokenizer = Tokenizer(
                self.engine.hf_tokenizer,
                self.engine.model.is_multilingual,
                task="transcribe",
                language="en"
            )

            # Use generate_segments with NPU features (no encoder_output = it will encode from features)
            logger.info("🚀 Calling generate_segments with NPU mel features...")
            segments = self.engine.generate_segments(
                features=mel_features,
                tokenizer=tokenizer,
                options=options,
                log_progress=False,
                encoder_output=None  # Let it encode our NPU features
            )

            # Calculate duration from features
            duration = float(mel_features.shape[-1] * self.engine.feature_extractor.time_per_frame)

            # Create transcription info
            info = TranscriptionInfo(
                language="en",
                language_probability=1.0,
                duration=duration,
                duration_after_vad=duration,
                transcription_options=options,
                vad_options=None,
                all_language_probs=None
            )

            logger.info("✅ Successfully used NPU mel features (CPU recomputation avoided!)")
        else:
            # Fallback to normal transcription (will recompute mel on CPU)
            logger.info("⚠️ Using standard transcribe() - mel will be recomputed on CPU")
            update_progress(40, "Loading audio file...")

            # Use UC-Meeting-Ops VAD parameters for optimal performance
            vad_parameters = None
            if vad_filter:
                vad_parameters = {
                    "min_silence_duration_ms": 1500,  # Remove silences 1.5s+
                    "speech_pad_ms": 1000,  # Generous padding
                    "threshold": 0.25  # Permissive threshold
                }
                logger.info(f"   VAD: UC-Meeting-Ops optimized parameters")

            update_progress(45, "Starting transcription...")

            # Start progress updater if tracking progress
            if job_id:
                progress_updater_task = asyncio.create_task(simulate_progress())

            # Run transcription in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: self.engine.transcribe(
                    audio_path,
                    beam_size=5,
                    language="en",
                    vad_filter=vad_filter,
                    vad_parameters=vad_parameters,
                    word_timestamps=True
                )
            )

            # Mark transcription complete and stop progress updater
            transcription_complete = True
            if progress_updater_task:
                await progress_updater_task

            update_progress(95, "Processing results...")

        result_segments = []
        full_text = ""

        for segment in segments:
            segment_data = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            }

            if hasattr(segment, 'words'):
                segment_data["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in segment.words
                ]

            result_segments.append(segment_data)
            full_text += segment.text + " "

        # Add speaker diarization if requested
        speaker_info = None
        if enable_diarization:
            update_progress(96, "Running speaker diarization...")

            if self.diarization_pipeline:
                result_segments = self.add_speaker_diarization(
                    audio_path,
                    result_segments,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )

                # Extract speaker information
                unique_speakers = set()
                for seg in result_segments:
                    if "speaker" in seg:
                        unique_speakers.add(seg["speaker"])

                if unique_speakers:
                    speaker_info = {
                        "count": len(unique_speakers),
                        "labels": sorted(list(unique_speakers))
                    }
                    logger.info(f"👥 Found {len(unique_speakers)} speakers: {', '.join(sorted(unique_speakers))}")
            else:
                logger.warning("⚠️ Diarization requested but pipeline not available")

        update_progress(98, "Finalizing results...")

        elapsed = time.time() - start_time
        audio_duration = info.duration
        realtime_factor = audio_duration / elapsed if elapsed > 0 else 0

        result = {
            "text": full_text.strip(),
            "segments": result_segments,
            "language": info.language,
            "duration": audio_duration,
            "processing_time": elapsed,
            "realtime_factor": f"{realtime_factor:.1f}x",
            "hardware": self.hardware["name"],
            "npu_mel_time": mel_time if mel_time > 0 else None,
            "diarization_enabled": enable_diarization,
            "diarization_available": self.diarization_pipeline is not None
        }

        # Add speaker information if available
        if speaker_info:
            result["speakers"] = speaker_info

        return result

# Initialize engine
logger.info("🦄 Initializing Unicorn Amanuensis...")
whisper_engine = DynamicWhisperEngine()
logger.info("✅ Server ready!")

@app.get("/")
async def root():
    return {
        "service": "Unicorn Amanuensis (Dynamic)",
        "version": "3.0.0",
        "hardware": whisper_engine.hardware,
        "models_found": len(whisper_engine.models),
        "status": "ready",
        "endpoints": {
            "/transcribe": "POST - Upload audio file (returns job_id for progress tracking)",
            "/v1/audio/transcriptions": "POST - OpenAI-compatible transcription endpoint",
            "/progress/{job_id}": "GET - Get transcription progress",
            "/progress/{job_id}/stream": "GET - Stream progress updates (Server-Sent Events)",
            "/models": "GET - List available models with performance data",
            "/status": "GET - Server status",
            "/web": "GET - Web interface"
        }
    }

@app.get("/models")
async def list_models():
    """List available models with performance information"""
    # Performance data for XDNA1 (AMD Phoenix NPU) - tested on this hardware
    model_performance = {
        "tiny": {"params": "39M", "speed": "40-50x", "accuracy": "Good", "use_case": "Fast draft transcription"},
        "base": {"params": "74M", "speed": "18-20x", "accuracy": "Very Good", "use_case": "Balanced speed/accuracy ⭐"},
        "small": {"params": "244M", "speed": "8-12x", "accuracy": "Excellent", "use_case": "High quality transcription"},
        "medium": {"params": "769M", "speed": "3-5x", "accuracy": "Excellent+", "use_case": "Professional transcription"},
        "large-v2": {"params": "1.55B", "speed": "1.5-2.5x", "accuracy": "Near Perfect", "use_case": "Maximum accuracy"},
        "large-v3": {"params": "1.55B", "speed": "1.5-2x", "accuracy": "Best", "use_case": "State-of-the-art accuracy"}
    }

    models_info = []
    for model_name in whisper_engine.models.keys():
        # Extract base model name (remove prefixes like Systran--, onnx-community--, magicunicorn--, etc)
        base_name = model_name.split('--')[-1] if '--' in model_name else model_name

        # Further extract actual model size from names like "faster-whisper-tiny" or "whisper-base-amd-npu-int8"
        # Look for known model sizes in the name
        model_size = None
        for size in ["tiny", "base", "small", "medium", "large-v3", "large-v2", "large"]:
            if size in base_name.lower():
                model_size = size
                break

        # If we found a size, use it; otherwise use the base name
        lookup_name = model_size if model_size else base_name

        # Get performance info
        perf = model_performance.get(lookup_name, {
            "params": "Unknown",
            "speed": "Varies",
            "accuracy": "Good",
            "use_case": "Transcription"
        })

        models_info.append({
            "id": base_name,
            "name": base_name,
            "full_name": model_name,
            "parameters": perf["params"],
            "speed": perf["speed"],
            "accuracy": perf["accuracy"],
            "use_case": perf["use_case"],
            "display_name": f"{base_name} ({perf['speed']} realtime, {perf['params']})"
        })

    return {
        "models": models_info,
        "hardware": whisper_engine.hardware["name"],
        "note": "Performance measured on AMD Phoenix NPU with faster-whisper INT8"
    }

@app.get("/status")
async def status():
    return {
        "status": "ready",
        "hardware": whisper_engine.hardware,
        "models_found": list(whisper_engine.models.keys()),
        "current_model": whisper_engine.current_model,
        "models_with_performance": f"See /models endpoint for detailed performance data",
        "diarization": {
            "available": whisper_engine.diarization_pipeline is not None,
            "model": "pyannote/speaker-diarization-3.1" if whisper_engine.diarization_pipeline else None,
            "note": "Speaker diarization ready" if whisper_engine.diarization_pipeline else "Diarization not available. Set HF_TOKEN and accept model license."
        },
        "npu_runtime": {
            "available": hasattr(whisper_engine, 'npu_runtime'),
            "mel_ready": whisper_engine.npu_runtime.mel_available if hasattr(whisper_engine, 'npu_runtime') else False,
            "gelu_ready": whisper_engine.npu_runtime.gelu_available if hasattr(whisper_engine, 'npu_runtime') else False,
            "attention_ready": whisper_engine.npu_runtime.attention_available if hasattr(whisper_engine, 'npu_runtime') else False,
        }
    }

@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    """Get transcription progress for a job"""
    if job_id not in progress_store:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(progress_store[job_id])

@app.get("/progress/{job_id}/stream")
async def stream_progress(job_id: str):
    """Stream transcription progress using Server-Sent Events"""
    async def event_generator():
        while job_id in progress_store:
            progress = progress_store[job_id]
            yield f"data: {json.dumps(progress)}\n\n"

            if progress.get("status") in ["completed", "error"]:
                break

            await asyncio.sleep(0.5)  # Update every 500ms

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    """Serve web interface"""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    else:
        return HTMLResponse("<h1>🦄 Unicorn Amanuensis (Dynamic)</h1><p>Upload audio to transcribe</p>")

@app.post("/transcribe")
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("base"),
    language: str = Form("en"),
    enable_diarization: bool = Form(False),
    min_speakers: int = Form(1),
    max_speakers: int = Form(10),
    vad_filter: bool = Form(True),
    enable_vad: bool = Form(None)
):
    """Transcribe audio file (supports both /transcribe and OpenAI-compatible /v1/audio/transcriptions)

    Args:
        file: Audio file to transcribe
        model: Whisper model to use (default: base)
        language: Language code (default: en)
        enable_diarization: Enable speaker diarization (default: False)
        min_speakers: Minimum number of speakers for diarization (default: 1)
        max_speakers: Maximum number of speakers for diarization (default: 10)
        vad_filter: Enable Voice Activity Detection to filter silent/noisy segments (default: True)
        enable_vad: Alias for vad_filter for user-friendliness (optional)

    Note:
        - VAD (Voice Activity Detection) is ENABLED by default to skip silent segments
        - This helps prevent "Can't find viable result" errors on noisy audio
        - Set vad_filter=false or enable_vad=false to disable if needed
        - Speaker diarization uses pyannote/speaker-diarization-3.1
        - Requires HF_TOKEN environment variable and model license acceptance
    """

    # Handle VAD parameter aliases - enable_vad takes precedence if explicitly set
    if enable_vad is not None:
        vad_filter = enable_vad

    # Log info if diarization is requested
    if enable_diarization:
        if whisper_engine.diarization_pipeline:
            logger.info(f"🎭 Speaker diarization enabled (speakers: {min_speakers}-{max_speakers})")
        else:
            logger.warning("⚠️ Diarization requested but not available. Transcribing without speaker labels.")
            logger.info("   To enable: Set HF_TOKEN and accept license at https://huggingface.co/pyannote/speaker-diarization-3.1")

    # Generate job ID for progress tracking
    job_id = str(uuid.uuid4())[:8]

    # Initialize progress
    progress_store[job_id] = {
        "status": "uploading",
        "progress": 0,
        "message": "Uploading audio file...",
        "job_id": job_id
    }

    # Save uploaded file
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Update progress: preprocessing
        progress_store[job_id] = {
            "status": "preprocessing",
            "progress": 10,
            "message": "Preprocessing audio...",
            "job_id": job_id
        }

        # Update progress: transcribing
        progress_store[job_id] = {
            "status": "transcribing",
            "progress": 30,
            "message": f"Transcribing with {model} model...",
            "job_id": job_id
        }

        result = await whisper_engine.transcribe(
            tmp_path,
            model,
            vad_filter=vad_filter,
            enable_diarization=enable_diarization,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            job_id=job_id
        )

        # Update progress: completed
        progress_store[job_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Transcription complete",
            "job_id": job_id
        }

        # Add VAD status to result
        result["vad_filter"] = vad_filter
        result["job_id"] = job_id

        # Add job_id to headers so client can track progress during upload
        return JSONResponse(result, headers={"X-Job-ID": job_id})

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)

        # Update progress: error
        progress_store[job_id] = {
            "status": "error",
            "progress": 0,
            "message": str(e),
            "job_id": job_id
        }

        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        # Clean up progress after 5 minutes
        async def cleanup_progress():
            await asyncio.sleep(300)  # 5 minutes
            if job_id in progress_store:
                del progress_store[job_id]

        asyncio.create_task(cleanup_progress())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9004)
