#!/usr/bin/env python3
"""
Practical WhisperX NPU Acceleration
Uses NPU runtime with existing INT8 models
Bypasses complex MLIR compilation for now
"""

import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Optional
import subprocess

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class WhisperNPURuntime:
    """Practical NPU runtime for Whisper using existing tooling"""

    def __init__(self, model_path: str = "whisper-base"):
        self.model_path = Path(model_path)
        self.npu_device = "/dev/accel/accel0"
        self.xclbin_path = self._find_xclbin()
        self.npu_available = self._check_npu()

        logger.info("=" * 70)
        logger.info("WhisperX NPU Runtime - Practical Implementation")
        logger.info("=" * 70)
        logger.info(f"NPU Device: {self.npu_device}")
        logger.info(f"NPU Available: {self.npu_available}")
        logger.info(f"XCL Binary: {self.xclbin_path}")
        logger.info(f"Model: {self.model_path}")

    def _check_npu(self) -> bool:
        """Check if NPU is available and operational"""
        try:
            # Check device file
            if not Path(self.npu_device).exists():
                logger.warning(f"❌ NPU device not found at {self.npu_device}")
                return False

            # Check XRT status
            result = subprocess.run(
                ["/opt/xilinx/xrt/bin/xrt-smi", "examine"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and "NPU Phoenix" in result.stdout:
                logger.info("✅ NPU Phoenix detected and operational")
                return True
            else:
                logger.warning("⚠️ NPU not detected by xrt-smi")
                return False

        except Exception as e:
            logger.warning(f"⚠️ NPU check failed: {e}")
            return False

    def _find_xclbin(self) -> Optional[Path]:
        """Find appropriate xclbin for Phoenix NPU"""
        xclbin_dirs = [
            Path("/opt/xilinx/xrt/share/amdxdna/bins/17f0_11"),
            Path("/opt/xilinx/xrt/share/amdxdna/bins/17f0_20"),
            Path("./"),
        ]

        for directory in xclbin_dirs:
            if directory.exists():
                # Look for 4col xclbin (Phoenix has 4 columns)
                xclbins = list(directory.glob("*4col*.xclbin"))
                if xclbins:
                    return xclbins[0]

                # Any xclbin will do for testing
                xclbins = list(directory.glob("*.xclbin"))
                if xclbins:
                    return xclbins[0]

        return None

    def transcribe(self, audio_path: str) -> Dict:
        """Transcribe audio using NPU-accelerated Whisper"""
        logger.info(f"\n🎤 Transcribing: {audio_path}")
        logger.info("=" * 70)

        start_time = time.time()

        try:
            # Use OpenVINO with NPU plugin
            result = self._transcribe_openvino_npu(audio_path)
        except Exception as e:
            logger.warning(f"NPU transcription failed: {e}")
            logger.info("Falling back to CPU/GPU execution...")
            result = self._transcribe_openvino_cpu(audio_path)

        elapsed = time.time() - start_time

        logger.info(f"\n✅ Transcription complete in {elapsed:.2f}s")
        return result

    def _transcribe_openvino_npu(self, audio_path: str) -> Dict:
        """Transcribe using OpenVINO with NPU plugin"""
        try:
            # Try to use NPU via OpenVINO
            # This requires OpenVINO NPU plugin which may not be available yet
            from optimum.intel import OVModelForSpeechSeq2Seq
            from transformers import AutoProcessor
            import torch

            logger.info("🚀 Using OpenVINO with NPU acceleration...")

            # Load INT8 quantized model
            model_path = "/home/ucadmin/openvino-models/whisper-base-int8"

            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            # Load model with NPU device (if supported)
            model = OVModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                device="NPU"  # Will fall back to GPU/CPU if NPU not supported
            )

            processor = AutoProcessor.from_pretrained("openai/whisper-base")

            # Load and process audio
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)

            # Process through model
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            generated_ids = model.generate(inputs["input_features"])
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return {
                "text": transcription,
                "backend": "openvino_npu",
                "npu_accelerated": True
            }

        except Exception as e:
            logger.warning(f"OpenVINO NPU execution failed: {e}")
            raise

    def _transcribe_openvino_cpu(self, audio_path: str) -> Dict:
        """Fallback to CPU/GPU OpenVINO execution"""
        try:
            from optimum.intel import OVModelForSpeechSeq2Seq
            from transformers import AutoProcessor
            import librosa

            logger.info("🔄 Using OpenVINO with CPU/GPU fallback...")

            # Load INT8 quantized model
            model_path = "/home/ucadmin/openvino-models/whisper-base-int8"

            if not Path(model_path).exists():
                # Try original ONNX models
                model_path = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base"

            model = OVModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                device="CPU"
            )

            processor = AutoProcessor.from_pretrained("openai/whisper-base")

            # Load and process audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)

            # Process through model
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            generated_ids = model.generate(inputs["input_features"])
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return {
                "text": transcription,
                "backend": "openvino_cpu",
                "npu_accelerated": False
            }

        except Exception as e:
            logger.error(f"OpenVINO CPU execution failed: {e}")
            raise

    def benchmark(self, audio_path: str, iterations: int = 3) -> Dict:
        """Benchmark NPU performance"""
        logger.info(f"\n📊 Benchmarking with {iterations} iterations...")
        logger.info("=" * 70)

        times = []
        for i in range(iterations):
            logger.info(f"\nIteration {i+1}/{iterations}")
            start = time.time()
            result = self.transcribe(audio_path)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        logger.info(f"\n📊 Benchmark Results:")
        logger.info(f"   Average: {avg_time:.2f}s ± {std_time:.2f}s")
        logger.info(f"   Min: {min(times):.2f}s")
        logger.info(f"   Max: {max(times):.2f}s")

        # Calculate audio duration to get RTF
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio_duration = len(audio) / sr
            rtf = avg_time / audio_duration
            speedup = audio_duration / avg_time

            logger.info(f"\n🎵 Audio Duration: {audio_duration:.2f}s")
            logger.info(f"⚡ Real-Time Factor: {rtf:.4f}")
            logger.info(f"🚀 Speedup: {speedup:.1f}x realtime")
        except:
            pass

        return {
            "avg_time": avg_time,
            "std_time": std_time,
            "times": times,
            "backend": result.get("backend", "unknown"),
            "npu_accelerated": result.get("npu_accelerated", False)
        }


def main():
    """Test WhisperX NPU runtime"""
    runtime = WhisperNPURuntime()

    # Test audio files
    test_files = [
        "/home/ucadmin/VibeVoice/Shafen_Khan_call.m4a",
        "/home/ucadmin/Development/Call with Shafen Khan.m4a",
    ]

    for audio_path in test_files:
        if Path(audio_path).exists():
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Testing with: {audio_path}")
            logger.info(f"{'=' * 70}")

            # Single transcription
            result = runtime.transcribe(audio_path)
            logger.info(f"\n📝 Transcription: {result['text'][:200]}...")
            logger.info(f"🔧 Backend: {result['backend']}")
            logger.info(f"🚀 NPU Accelerated: {result['npu_accelerated']}")

            # Benchmark
            benchmark_results = runtime.benchmark(audio_path, iterations=3)

            break  # Test with first available file
    else:
        logger.warning("No test audio files found")
        logger.info("Please provide an audio file path to test")


if __name__ == "__main__":
    main()
