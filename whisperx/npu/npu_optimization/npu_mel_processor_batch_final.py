#!/usr/bin/env python3
"""
NPU Mel Spectrogram Processor - Batch Processing Final (20 frames)

This is the PRODUCTION BATCH PROCESSOR optimized for 20-frame chunks,
designed to work with the batch MLIR kernel (20 frames per NPU call).

Key Optimizations:
- Pre-allocated 16KB input buffer (20 frames × 800 bytes)
- Pre-allocated 1600B output buffer (20 frames × 80 bytes)
- Single sync per batch (not per frame)
- Processes audio in 20-frame chunks
- Handles partial batches (last batch may be < 20)
- Reuses XRT buffer objects for zero allocation overhead

Performance Target:
- Single-frame: 3.7ms/frame (20-30x realtime)
- Batch-10: 0.37ms/frame (600-700x realtime)
- Batch-20: 0.18ms/frame (1200-1500x realtime) ← 2x improvement
- Expected speedup: 2x faster than batch-10 processing

Hardware: AMD Phoenix NPU (XDNA1)
Kernel: mel_batch20.xclbin (20-frame batch processing)
Author: Magic Unicorn Unconventional Technology & Stuff Inc.
Date: November 1, 2025 (Upgraded from batch-10)
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import numpy as np
import time
import os
from pathlib import Path
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class NPUMelProcessorBatch:
    """
    Production batch NPU mel spectrogram processor (20 frames per call).

    This class processes audio in 20-frame batches using a custom MLIR kernel
    that processes all 20 frames in a single NPU call, reducing XRT overhead
    by 20x compared to single-frame processing.

    Architecture:
    - Input buffer: 16KB (20 frames × 400 samples × 2 bytes int16)
    - Output buffer: 1600B (20 frames × 80 mel bins × 1 byte int8)
    - Single kernel execution per batch
    - Partial batch support for audio edges

    Usage:
        processor = NPUMelProcessorBatch()
        mel_features = processor.process(audio_waveform)  # Returns [80, n_frames]
    """

    # Constants
    BATCH_SIZE = 20  # Fixed batch size matching MLIR kernel (upgraded from 10 for 2x performance)
    FRAME_SIZE = 400  # 25ms @ 16kHz
    HOP_LENGTH = 160  # 10ms @ 16kHz
    N_MELS = 80      # Whisper standard
    SAMPLE_RATE = 16000

    def __init__(
        self,
        xclbin_path: Optional[str] = None,
        fallback_to_cpu: bool = True,
        verbose: bool = True
    ):
        """
        Initialize batch NPU mel processor.

        Args:
            xclbin_path: Path to batch XCLBIN. If None, uses default location.
            fallback_to_cpu: If True, use librosa fallback if NPU fails
            verbose: If True, log detailed information
        """
        self.fallback_to_cpu = fallback_to_cpu
        self.verbose = verbose

        # Locate batch XCLBIN
        if xclbin_path is None:
            default_path = Path(__file__).parent / "mel_kernels" / "build_batch20" / "mel_batch20.xclbin"
            self.xclbin_path = str(default_path)
        else:
            self.xclbin_path = xclbin_path

        # Instructions binary path - extract batch size from xclbin filename
        xclbin_name = Path(self.xclbin_path).stem  # e.g., "mel_batch10" or "mel_batch100"
        insts_name = xclbin_name.replace("mel_", "insts_") + ".bin"  # e.g., "insts_batch10.bin"
        self.insts_path = Path(self.xclbin_path).parent / insts_name

        # NPU runtime objects
        self.device = None
        self.xclbin = None
        self.hw_ctx = None
        self.kernel = None
        self.npu_available = False

        # Pre-allocated XRT buffers (reused for entire session)
        self.instr_bo = None
        self.input_bo = None   # 16KB buffer (20 frames)
        self.output_bo = None  # 1600B buffer (20 outputs)
        self.insts_bin = None
        self.n_insts = 0

        # Buffer sizes
        self.input_buffer_size = self.BATCH_SIZE * self.FRAME_SIZE * 2  # 16,000 bytes (batch-20)
        self.output_buffer_size = self.BATCH_SIZE * self.N_MELS         # 1600 bytes (batch-20)

        # Performance metrics
        self.total_frames = 0
        self.total_batches = 0
        self.npu_time = 0.0
        self.cpu_time = 0.0
        self.kernel_time = 0.0
        self.transfer_time = 0.0

        # Initialize NPU
        self._initialize_npu()

    def _initialize_npu(self) -> bool:
        """
        Initialize NPU device and load batch XCLBIN with pre-allocated buffers.

        Returns:
            True if NPU initialized successfully, False otherwise
        """
        try:
            import pyxrt as xrt

            # Validation checks
            if not os.path.exists(self.xclbin_path):
                logger.warning(f"Batch XCLBIN not found: {self.xclbin_path}")
                if self.fallback_to_cpu:
                    logger.info("Falling back to CPU preprocessing")
                    return False
                else:
                    raise FileNotFoundError(f"XCLBIN not found: {self.xclbin_path}")

            if not os.path.exists(self.insts_path):
                logger.warning(f"Instructions not found: {self.insts_path}")
                if self.fallback_to_cpu:
                    logger.info("Falling back to CPU preprocessing")
                    return False
                else:
                    raise FileNotFoundError(f"Instructions not found: {self.insts_path}")

            if not os.path.exists("/dev/accel/accel0"):
                logger.warning("NPU device /dev/accel/accel0 not found")
                if self.fallback_to_cpu:
                    logger.info("Falling back to CPU preprocessing")
                    return False
                else:
                    raise RuntimeError("NPU device not available")

            if self.verbose:
                logger.info("="*70)
                logger.info("Initializing AMD Phoenix NPU (Batch-20 Mode)")
                logger.info("="*70)

            # Open device
            self.device = xrt.device(0)
            if self.verbose:
                logger.info(f"Device: /dev/accel/accel0 (AMD Phoenix NPU)")

            # Load and register XCLBIN
            self.xclbin = xrt.xclbin(self.xclbin_path)
            self.device.register_xclbin(self.xclbin)
            uuid = self.xclbin.get_uuid()
            xclbin_size = os.path.getsize(self.xclbin_path)
            if self.verbose:
                logger.info(f"XCLBIN: {Path(self.xclbin_path).name} ({xclbin_size / 1024:.1f} KB)")

            # Create hardware context
            self.hw_ctx = xrt.hw_context(self.device, uuid)

            # Get kernel
            self.kernel = xrt.kernel(self.hw_ctx, "MLIR_AIE")
            if self.verbose:
                logger.info(f"Kernel: MLIR_AIE (batch-20 mel spectrogram)")

            # Load instructions
            with open(self.insts_path, "rb") as f:
                self.insts_bin = f.read()
            self.n_insts = len(self.insts_bin)
            if self.verbose:
                logger.info(f"Instructions: {self.n_insts} bytes")

            # Pre-allocate instruction buffer (write once, reuse forever)
            self.instr_bo = xrt.bo(
                self.device,
                self.n_insts,
                xrt.bo.flags.cacheable,
                self.kernel.group_id(1)
            )
            self.instr_bo.write(self.insts_bin, 0)
            self.instr_bo.sync(
                xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                self.n_insts, 0
            )

            # Pre-allocate input buffer (16KB for 20 frames)
            self.input_bo = xrt.bo(
                self.device,
                self.input_buffer_size,
                xrt.bo.flags.host_only,
                self.kernel.group_id(3)
            )

            # Pre-allocate output buffer (1600B for 20 outputs)
            self.output_bo = xrt.bo(
                self.device,
                self.output_buffer_size,
                xrt.bo.flags.host_only,
                self.kernel.group_id(4)
            )

            self.npu_available = True

            if self.verbose:
                logger.info("="*70)
                logger.info("NPU Initialization Complete")
                logger.info("="*70)
                logger.info(f"Batch size: {self.BATCH_SIZE} frames")
                logger.info(f"Input buffer: {self.input_buffer_size / 1024:.1f} KB")
                logger.info(f"Output buffer: {self.output_buffer_size / 1024:.1f} KB")
                logger.info(f"Frame size: {self.FRAME_SIZE} samples (25ms)")
                logger.info(f"Hop length: {self.HOP_LENGTH} samples (10ms)")
                logger.info(f"Mel bins: {self.N_MELS}")
                logger.info("="*70)

            return True

        except Exception as e:
            logger.error(f"NPU initialization failed: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            if self.fallback_to_cpu:
                logger.info("Falling back to CPU preprocessing")
                self.npu_available = False
                return False
            else:
                raise

    def _process_batch_npu(self, frames: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Process a batch of frames on NPU (up to 20 frames).

        This method:
        1. Converts float32 frames to int16
        2. Writes batch to pre-allocated input buffer
        3. Executes kernel ONCE for entire batch
        4. Reads batch from pre-allocated output buffer
        5. Converts int8 output to float32

        Args:
            frames: [batch_size, 400] float32 audio frames
            batch_size: Actual number of frames in this batch (≤20)

        Returns:
            mel_outputs: [batch_size, 80] mel spectrograms (float32)
        """
        import pyxrt as xrt

        if batch_size == 0:
            return np.zeros((0, self.N_MELS), dtype=np.float32)

        try:
            # Step 1: Convert to int16 and flatten
            transfer_start = time.time()
            frames_int16 = np.clip(frames * 32767, -32768, 32767).astype(np.int16)
            frames_flat = frames_int16.flatten()

            # Calculate actual sizes for this batch
            input_bytes = batch_size * self.FRAME_SIZE * 2  # int16 = 2 bytes
            output_bytes = batch_size * self.N_MELS          # int8 = 1 byte

            # Step 2: Write to input buffer and sync to device
            self.input_bo.write(frames_flat.tobytes(), 0)
            self.input_bo.sync(
                xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                input_bytes, 0
            )
            transfer_time = time.time() - transfer_start

            # Step 3: Execute kernel for entire batch
            kernel_start = time.time()
            opcode = 3  # Mel computation opcode
            run = self.kernel(
                opcode,
                self.instr_bo,
                self.n_insts,
                self.input_bo,
                self.output_bo
            )

            # Wait for completion (10ms per frame + 1s safety margin)
            timeout_ms = 1000 + (batch_size * 10)
            run.wait(timeout_ms)
            kernel_time = time.time() - kernel_start

            # Step 4: Read output buffer and sync from device
            transfer_start = time.time()
            self.output_bo.sync(
                xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
                output_bytes, 0
            )
            mel_bins_all = np.frombuffer(
                self.output_bo.read(output_bytes, 0),
                dtype=np.int8
            )
            transfer_time += time.time() - transfer_start

            # Step 5: Reshape and convert to float32
            mel_outputs = mel_bins_all.reshape(batch_size, self.N_MELS).astype(np.float32)
            mel_outputs = mel_outputs / 127.0  # Normalize int8 [-128, 127] to float

            # Update metrics
            self.kernel_time += kernel_time
            self.transfer_time += transfer_time

            return mel_outputs

        except Exception as e:
            logger.error(f"NPU batch processing failed for {batch_size} frames: {e}")
            logger.error(f"  Input shape: {frames.shape}")
            logger.error(f"  Input bytes: {input_bytes}")
            logger.error(f"  Output bytes: {output_bytes}")
            raise

    def _process_batch_cpu(self, frames: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Process a batch of frames on CPU (fallback).

        Args:
            frames: [batch_size, 400] float32 frames
            batch_size: Number of frames to process

        Returns:
            mel_outputs: [batch_size, 80] mel spectrograms
        """
        try:
            import librosa
        except ImportError:
            logger.error("librosa not available for CPU fallback")
            raise

        mel_outputs = np.zeros((batch_size, self.N_MELS), dtype=np.float32)

        for i in range(batch_size):
            mel = librosa.feature.melspectrogram(
                y=frames[i],
                sr=self.SAMPLE_RATE,
                n_fft=512,
                hop_length=self.HOP_LENGTH,
                win_length=self.FRAME_SIZE,
                n_mels=self.N_MELS,
                fmin=0,
                fmax=self.SAMPLE_RATE // 2,
                htk=True,
                power=2.0
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_outputs[i] = mel_db[:, 0]

        return mel_outputs

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio to mel spectrogram using batch processing.

        This method:
        1. Frames the audio into 400-sample frames with 160-sample hop
        2. Processes frames in batches of 20 using NPU
        3. Handles partial batches (last batch may be < 20)
        4. Returns [n_mels, n_frames] array

        Args:
            audio: Audio samples (float32, mono, 16 kHz)

        Returns:
            mel_features: [n_mels, n_frames] mel spectrogram
        """
        # Calculate number of frames
        n_samples = len(audio)
        n_frames = (n_samples - self.FRAME_SIZE) // self.HOP_LENGTH + 1

        if n_frames == 0:
            logger.warning("Audio too short for even one frame")
            return np.zeros((self.N_MELS, 0), dtype=np.float32)

        # Allocate output
        mel_features = np.zeros((self.N_MELS, n_frames), dtype=np.float32)

        # Pre-allocate frame buffer (20 frames max)
        frames_buffer = np.zeros((self.BATCH_SIZE, self.FRAME_SIZE), dtype=np.float32)

        # Reset batch counter
        self.total_batches = 0

        # Process in batches of 20
        start_time = time.time()

        for batch_start in range(0, n_frames, self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, n_frames)
            batch_size = batch_end - batch_start

            # Extract frames for this batch
            for i in range(batch_size):
                frame_idx = batch_start + i
                start_idx = frame_idx * self.HOP_LENGTH
                end_idx = start_idx + self.FRAME_SIZE

                if end_idx <= n_samples:
                    frames_buffer[i] = audio[start_idx:end_idx]
                else:
                    # Pad last frame with zeros
                    frames_buffer[i].fill(0)
                    remaining = n_samples - start_idx
                    if remaining > 0:
                        frames_buffer[i, :remaining] = audio[start_idx:]

            # Process batch (NPU or CPU)
            if self.npu_available:
                batch_start_time = time.time()
                mel_batch = self._process_batch_npu(frames_buffer[:batch_size], batch_size)
                self.npu_time += time.time() - batch_start_time
            else:
                batch_start_time = time.time()
                mel_batch = self._process_batch_cpu(frames_buffer[:batch_size], batch_size)
                self.cpu_time += time.time() - batch_start_time

            # Store results (transpose to [n_mels, batch_size])
            mel_features[:, batch_start:batch_end] = mel_batch.T

            self.total_frames += batch_size
            self.total_batches += 1

        elapsed = time.time() - start_time

        # Calculate performance metrics
        audio_duration = n_samples / self.SAMPLE_RATE
        rtf = audio_duration / elapsed if elapsed > 0 else 0
        avg_batch_size = n_frames / self.total_batches if self.total_batches > 0 else 0

        if self.verbose:
            logger.info("="*70)
            logger.info("Processing Complete")
            logger.info("="*70)
            logger.info(f"Audio duration: {audio_duration:.2f}s ({n_samples} samples)")
            logger.info(f"Total frames: {n_frames}")
            logger.info(f"Total batches: {self.total_batches}")
            logger.info(f"Avg batch size: {avg_batch_size:.1f} frames")
            logger.info(f"Processing time: {elapsed:.4f}s")
            logger.info(f"Realtime factor: {rtf:.2f}x")
            logger.info(f"Backend: {'NPU' if self.npu_available else 'CPU'}")
            if self.npu_available and self.total_batches > 0:
                logger.info(f"  Kernel time: {self.kernel_time:.4f}s ({self.kernel_time/elapsed*100:.1f}%)")
                logger.info(f"  Transfer time: {self.transfer_time:.4f}s ({self.transfer_time/elapsed*100:.1f}%)")
                logger.info(f"  Time per frame: {elapsed/n_frames*1000:.3f}ms")
                logger.info(f"  Time per batch: {elapsed/self.total_batches*1000:.3f}ms")
            logger.info("="*70)

        return mel_features

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Alias for process() to allow function-style usage.

        Args:
            audio: Audio samples (float32, mono, 16 kHz)

        Returns:
            mel_features: [n_mels, n_frames] mel spectrogram
        """
        return self.process(audio)

    def get_performance_metrics(self) -> dict:
        """
        Get detailed performance metrics.

        Returns:
            metrics: Dictionary with performance statistics
        """
        avg_npu_time_per_frame = (self.npu_time / self.total_frames * 1000) if self.total_frames > 0 else 0
        avg_npu_time_per_batch = (self.npu_time / self.total_batches * 1000) if self.total_batches > 0 else 0

        return {
            "total_frames": self.total_frames,
            "total_batches": self.total_batches,
            "batch_size": self.BATCH_SIZE,
            "npu_time_total": self.npu_time,
            "cpu_time_total": self.cpu_time,
            "kernel_time_total": self.kernel_time,
            "transfer_time_total": self.transfer_time,
            "npu_time_per_frame_ms": avg_npu_time_per_frame,
            "npu_time_per_batch_ms": avg_npu_time_per_batch,
            "kernel_time_per_batch_ms": (self.kernel_time / self.total_batches * 1000) if self.total_batches > 0 else 0,
            "transfer_time_per_batch_ms": (self.transfer_time / self.total_batches * 1000) if self.total_batches > 0 else 0,
            "npu_available": self.npu_available,
            "buffer_input_size_kb": self.input_buffer_size / 1024,
            "buffer_output_size_kb": self.output_buffer_size / 1024
        }

    def reset_metrics(self):
        """Reset all performance metrics."""
        self.total_frames = 0
        self.total_batches = 0
        self.npu_time = 0.0
        self.cpu_time = 0.0
        self.kernel_time = 0.0
        self.transfer_time = 0.0

    def close(self):
        """Clean up NPU resources and close device."""
        if self.device:
            if self.verbose:
                logger.info("Closing NPU device and releasing resources...")
            self.device = None
            self.xclbin = None
            self.hw_ctx = None
            self.kernel = None
            self.instr_bo = None
            self.input_bo = None
            self.output_bo = None
            self.npu_available = False
            if self.verbose:
                logger.info("NPU resources released successfully")


# Convenience functions
def create_batch_processor(
    xclbin_path: Optional[str] = None,
    fallback_to_cpu: bool = True,
    verbose: bool = True
) -> NPUMelProcessorBatch:
    """
    Create batch NPU mel processor with default settings.

    Args:
        xclbin_path: Path to batch XCLBIN (optional)
        fallback_to_cpu: If True, use librosa fallback if NPU fails
        verbose: If True, log detailed information

    Returns:
        processor: Initialized batch NPU mel processor
    """
    return NPUMelProcessorBatch(
        xclbin_path=xclbin_path,
        fallback_to_cpu=fallback_to_cpu,
        verbose=verbose
    )


if __name__ == "__main__":
    # Quick test
    logger.info("NPU Mel Processor Batch-10 - Quick Test")
    logger.info("="*70)

    try:
        # Create processor
        processor = create_batch_processor(verbose=True)

        # Generate test audio (5 seconds)
        duration = 5.0
        n_samples = int(duration * 16000)
        t = np.linspace(0, duration, n_samples, dtype=np.float32)
        freq = 440 + 200 * np.sin(2 * np.pi * t)  # Frequency sweep
        audio = 0.5 * np.sin(2 * np.pi * freq * t)

        logger.info(f"Test audio: {duration}s ({n_samples} samples)")

        # Process
        mel_features = processor.process(audio)

        # Results
        logger.info("="*70)
        logger.info("Test Results")
        logger.info("="*70)
        logger.info(f"Output shape: {mel_features.shape} (mels, frames)")
        logger.info(f"Mel range: [{mel_features.min():.4f}, {mel_features.max():.4f}]")
        logger.info(f"Mel mean: {mel_features.mean():.4f}")

        # Performance metrics
        metrics = processor.get_performance_metrics()
        logger.info("="*70)
        logger.info("Performance Metrics")
        logger.info("="*70)
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")

        # Cleanup
        processor.close()

        logger.info("="*70)
        logger.info("Test completed successfully!")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
