"""
NPU Accelerator for WhisperX - Direct Machine Code Execution
===========================================================
Uses pre-generated NPU binaries without requiring Vitis
"""

import numpy as np
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import mmap

logger = logging.getLogger(__name__)

class NPUAccelerator:
    """NPU Accelerator using pre-compiled binaries"""
    
    def __init__(self):
        self.npu_binary_path = Path("whisperx_npu.bin")
        self.is_initialized = False
        self.use_emulation = False  # Start with hardware, fall back if needed
        
        # NPU specifications for AMD Phoenix
        self.specs = {
            "compute_units": 4,
            "vector_width": 1024,  # bits
            "int8_ops_per_cycle": 128,
            "frequency": 1.0e9,  # 1 GHz
            "memory_bandwidth": 136e9,  # 136 GB/s
            "int8_tops": 16  # 16 TOPS INT8
        }
        
        logger.info("🚀 NPU Accelerator initializing...")
        self._initialize()
    
    def _initialize(self):
        """Initialize NPU accelerator"""
        try:
            # Check for NPU device access - NO FALLBACK TO EMULATION
            if not os.path.exists('/dev/accel/accel0'):
                logger.error("❌ NPU device not found at /dev/accel/accel0")
                logger.error("❌ NPU hardware is REQUIRED - cannot proceed")
                raise RuntimeError("NPU device not found - transcription requires NPU hardware")
            
            # Check permissions
            if not os.access('/dev/accel/accel0', os.R_OK | os.W_OK):
                logger.error("❌ No permission to access NPU device")
                logger.error("❌ Add user to 'render' group: sudo usermod -a -G render $USER")
                raise RuntimeError("NPU device permission denied - add user to render group")
            
            logger.info("✅ NPU device accessible at /dev/accel/accel0")
            
            # Use REAL NPU implementation from our existing npu_runtime
            try:
                from npu_runtime import SimplifiedNPURuntime
                self.npu_runtime = SimplifiedNPURuntime()
                
                if not self.npu_runtime.open_device():
                    raise RuntimeError("Failed to open NPU device")
                
                logger.info("✅ NPU device opened successfully")
                
                # Load Whisper model on NPU
                if self.npu_runtime.load_model("whisper-base"):
                    logger.info("✅ Whisper model loaded on NPU")
                    self.use_emulation = False
                    self.is_initialized = True
                else:
                    raise RuntimeError("Failed to load Whisper model on NPU")
                    
            except ImportError as e:
                logger.error(f"❌ Failed to import SimplifiedNPURuntime: {e}")
                raise RuntimeError("NPU runtime module not available")
            except Exception as e:
                logger.error(f"❌ NPU initialization failed: {e}")
                raise RuntimeError(f"NPU initialization failed: {e}")
            
            logger.info("✅ NPU Accelerator ready - HARDWARE MODE ONLY")
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: NPU initialization failed: {e}")
            logger.error("❌ NPU hardware is REQUIRED for transcription")
            self.is_initialized = False
            raise
    
    def is_available(self) -> bool:
        """Check if NPU is available"""
        return self.is_initialized
    
    def execute_kernel(self, kernel_name: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute NPU kernel"""
        if not self.is_initialized:
            raise RuntimeError("NPU not initialized")
        
        if self.use_emulation:
            return self._execute_emulated(kernel_name, inputs)
        else:
            return self._execute_hardware(kernel_name, inputs)
    
    def _execute_emulated(self, kernel_name: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute kernel in emulation mode"""
        start_time = time.time()
        
        if kernel_name == "whisper_attention":
            # Emulate attention computation
            q = inputs.get("q")
            k = inputs.get("k") 
            v = inputs.get("v")
            
            # Q * K^T
            scores = np.matmul(q, k.T)
            
            # Scale
            scores = scores / np.sqrt(q.shape[-1])
            
            # Softmax
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            
            # Attention output
            output = np.matmul(attention_weights, v)
            
            elapsed = time.time() - start_time
            logger.debug(f"Attention kernel emulated in {elapsed*1000:.2f}ms")
            
            return {"output": output, "weights": attention_weights}
        
        elif kernel_name == "mel_spectrogram":
            # Emulate mel spectrogram computation
            audio = inputs.get("audio")
            
            # Simple FFT-based mel spectrogram (placeholder)
            # In real implementation, this would use the NPU's FFT kernel
            n_fft = 400
            hop_length = 160
            n_mels = 80
            
            # Simplified computation
            stft = np.fft.rfft(audio, n=n_fft)
            mel_spec = np.abs(stft[:n_mels])
            
            elapsed = time.time() - start_time
            logger.debug(f"Mel spectrogram emulated in {elapsed*1000:.2f}ms")
            
            return {"mel": mel_spec}
        
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")
    
    def _execute_hardware(self, kernel_name: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute kernel on actual NPU hardware"""
        if not hasattr(self, 'npu_runtime') or not self.npu_runtime:
            raise RuntimeError("NPU runtime not available - hardware required")
        
        try:
            # Use real NPU runtime - NO EMULATION
            if kernel_name == "whisper_transcribe":
                audio_data = inputs.get("audio", np.array([]))
                
                # Run REAL inference on NPU hardware
                result = self.npu_runtime.transcribe(audio_data)
                
                return {
                    "transcription": result.get("text", ""),
                    "segments": [],  # TODO: Add segment support
                    "processing_time": result.get("processing_time", 0),
                    "npu_accelerated": True,
                    "hardware_info": result.get("device_info", {})
                }
            else:
                # Other kernels must also use NPU
                raise NotImplementedError(f"NPU kernel '{kernel_name}' not implemented")
                
        except Exception as e:
            logger.error(f"❌ NPU hardware execution failed: {e}")
            raise RuntimeError(f"NPU hardware execution failed: {e}")
    
    def benchmark(self) -> Dict[str, float]:
        """Benchmark NPU performance"""
        results = {}
        
        # Test attention kernel
        q = np.random.randn(64, 64).astype(np.float32)
        k = np.random.randn(64, 64).astype(np.float32)
        v = np.random.randn(64, 64).astype(np.float32)
        
        start = time.time()
        self.execute_kernel("whisper_attention", {"q": q, "k": k, "v": v})
        results["attention_ms"] = (time.time() - start) * 1000
        
        # Test mel spectrogram
        audio = np.random.randn(16000).astype(np.float32)
        
        start = time.time()
        self.execute_kernel("mel_spectrogram", {"audio": audio})
        results["mel_spec_ms"] = (time.time() - start) * 1000
        
        # Calculate theoretical speedup
        results["theoretical_speedup"] = self.specs["int8_tops"] / 0.5  # vs 0.5 TOPS CPU
        
        return results