#!/usr/bin/env python3
"""
Transcription Pipeline - Multi-Stage Speech-to-Text Pipeline

Implements a 3-stage pipelined architecture for concurrent transcription:

Stage 1: Audio Load + Mel Spectrogram (CPU-bound, 4 thread workers)
         - Load audio from bytes
         - Compute mel spectrogram with buffer pool + zero-copy
         - Output: Mel features → Stage 2

Stage 2: NPU Encoder (NPU-bound, 1 worker for serialization)
         - Run C++ encoder on NPU
         - Single worker to serialize NPU access
         - Output: Encoder embeddings → Stage 3

Stage 3: Decoder + Alignment (CPU-bound, 4 process workers)
         - Run Python decoder (WhisperX)
         - Align with WhisperX alignment model
         - Output: Final transcription text

Performance Target:
- Throughput: 15.6 → 67 req/s (+329%)
- NPU Utilization: 0.12% → 15% (+1775%)
- Concurrent Requests: 1 → 10-15
- Individual Latency: Unchanged (~60ms per request)

Author: CC-1L Multi-Stream Pipeline Team
Date: November 1, 2025
Status: Week 9 Implementation
"""

import asyncio
import time
import logging
import whisperx
import numpy as np
import torch
from typing import Optional, Dict, Any
from pathlib import Path

from request_queue import RequestQueue, QueuedRequest
from pipeline_workers import PipelineStage, WorkItem
from buffer_pool import GlobalBufferManager
from xdna2.mel_utils import compute_mel_spectrogram_zerocopy, validate_mel_contiguity
from xdna2.encoder_cpp import WhisperEncoderCPP
from xdna2.whisper_conv1d import WhisperConv1dPreprocessor

logger = logging.getLogger(__name__)


class TranscriptionPipeline:
    """
    Multi-stage transcription pipeline for concurrent request processing.

    Architecture:
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Stage 1:    │ --> │  Stage 2:    │ --> │  Stage 3:    │
    │  Load + Mel  │     │  Encoder     │     │  Decoder +   │
    │  (4 threads) │     │  (1 NPU)     │     │  Alignment   │
    │              │     │              │     │  (4 procs)   │
    └──────────────┘     └──────────────┘     └──────────────┘

    Features:
    - Request queue with priority scheduling
    - 3-stage pipeline with automatic routing
    - Buffer pool integration for zero-copy
    - Comprehensive statistics tracking
    - Graceful start/stop with queue draining
    - Error handling and recovery

    Usage:
        # Initialize pipeline
        pipeline = TranscriptionPipeline(
            cpp_encoder=encoder,
            python_decoder=decoder,
            model_a=align_model,
            metadata=metadata,
            num_load_workers=4,
            num_decoder_workers=4
        )

        # Start pipeline
        await pipeline.start()

        # Process request
        result = await pipeline.transcribe(request)

        # Stop pipeline
        await pipeline.stop()

    Thread Safety:
        All operations are thread-safe via asyncio primitives.
        Multiple requests can be processed concurrently.
    """

    def __init__(
        self,
        cpp_encoder: WhisperEncoderCPP,
        python_decoder: Any,
        model_a: Any,
        metadata: Any,
        device: str = "cpu",
        batch_size: int = 16,
        num_load_workers: int = 4,
        num_decoder_workers: int = 4,
        max_queue_size: int = 100,
        max_stage_queue_size: int = 20
    ):
        """
        Initialize transcription pipeline.

        Args:
            cpp_encoder: C++ encoder instance (WhisperEncoderCPP)
            python_decoder: Python decoder (WhisperX)
            model_a: Alignment model (WhisperX)
            metadata: Alignment metadata
            device: Device for decoder ('cpu' or 'cuda')
            batch_size: Batch size for decoder
            num_load_workers: Workers for Stage 1 (load + mel)
            num_decoder_workers: Workers for Stage 3 (decoder + align)
            max_queue_size: Maximum requests in main queue
            max_stage_queue_size: Maximum items in stage queues

        Raises:
            ValueError: If invalid configuration
        """
        if not cpp_encoder or not python_decoder:
            raise ValueError("cpp_encoder and python_decoder are required")

        self.cpp_encoder = cpp_encoder
        self.python_decoder = python_decoder
        self.model_a = model_a
        self.metadata = metadata
        self.device = device
        self.batch_size = batch_size

        # Extract feature extractor from whisperx model
        # whisperx.load_model() returns FasterWhisperPipeline
        # Feature extractor is at: python_decoder.model.feature_extractor
        self.feature_extractor = python_decoder.model.feature_extractor

        # Initialize conv1d preprocessor (Bug #5 fix)
        # WhisperX doesn't expose encoder.conv1/conv2, so we load from transformers
        logger.info("[Pipeline] Loading Whisper model for conv1d weights...")
        from transformers import WhisperModel
        whisper_model = WhisperModel.from_pretrained(f"openai/whisper-base")
        self.conv1d_preprocessor = WhisperConv1dPreprocessor(whisper_model)
        logger.info("[Pipeline] Conv1d preprocessor initialized")

        # Request queue
        self.request_queue = RequestQueue(max_queue_size=max_queue_size)

        # Buffer manager
        self.buffer_manager = GlobalBufferManager.instance()

        # Pipeline stages
        self.stage1: Optional[PipelineStage] = None
        self.stage2: Optional[PipelineStage] = None
        self.stage3: Optional[PipelineStage] = None

        # Configuration
        self.num_load_workers = num_load_workers
        self.num_decoder_workers = num_decoder_workers
        self.max_stage_queue_size = max_stage_queue_size

        # Router task
        self._router_task: Optional[asyncio.Task] = None
        self._running = False

        # Response futures for tracking requests
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._futures_lock = asyncio.Lock()

        logger.info("TranscriptionPipeline initialized")
        logger.info(f"  Load workers: {num_load_workers}")
        logger.info(f"  Decoder workers: {num_decoder_workers}")
        logger.info(f"  Max queue size: {max_queue_size}")
        logger.info(f"  Device: {device}")

    async def start(self):
        """Start all pipeline stages and routing"""
        if self._running:
            logger.warning("Pipeline already running")
            return

        logger.info("="*70)
        logger.info("  Starting Transcription Pipeline")
        logger.info("="*70)

        # Create Stage 1: Audio Load + Mel (CPU, threads)
        logger.info("[Pipeline] Creating Stage 1: Load + Mel...")
        self.stage1 = PipelineStage(
            name="LoadMel",
            process_func=self._process_load_mel,
            num_workers=self.num_load_workers,
            use_processes=False,  # Threads OK (I/O + NumPy)
            max_queue_size=self.max_stage_queue_size,
            worker_timeout=10.0  # 10s timeout for Stage 1
        )

        # Create Stage 2: Encoder (NPU, single worker)
        logger.info("[Pipeline] Creating Stage 2: Encoder...")
        self.stage2 = PipelineStage(
            name="Encoder",
            process_func=self._process_encoder,
            num_workers=1,  # Serialize NPU access
            use_processes=False,
            max_queue_size=self.max_stage_queue_size,
            worker_timeout=5.0  # 5s timeout for encoder
        )

        # Create Stage 3: Decoder + Alignment (CPU, processes for GIL)
        logger.info("[Pipeline] Creating Stage 3: Decoder + Alignment...")
        self.stage3 = PipelineStage(
            name="DecoderAlign",
            process_func=self._process_decoder_align,
            num_workers=self.num_decoder_workers,
            use_processes=False,  # Use threads for now (simpler, process pool has issues with WhisperX)
            max_queue_size=self.max_stage_queue_size,
            worker_timeout=30.0  # 30s timeout for decoder + align
        )

        # Start all stages
        logger.info("[Pipeline] Starting all stages...")
        await self.stage1.start()
        await self.stage2.start()
        await self.stage3.start()

        # Start router
        logger.info("[Pipeline] Starting router...")
        self._running = True
        self._router_task = asyncio.create_task(self._route_pipeline())

        logger.info("="*70)
        logger.info("  Pipeline Started Successfully")
        logger.info("="*70 + "\n")

    async def stop(self, drain_queues: bool = True, timeout: float = 30.0):
        """
        Stop all pipeline stages and routing.

        Args:
            drain_queues: Wait for queues to drain before stopping
            timeout: Maximum time to wait for draining (seconds)
        """
        if not self._running:
            logger.warning("Pipeline not running")
            return

        logger.info("="*70)
        logger.info("  Stopping Transcription Pipeline")
        logger.info("="*70)

        # Stop router
        self._running = False
        if self._router_task:
            self._router_task.cancel()
            try:
                await asyncio.wait_for(self._router_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Stop all stages (in reverse order)
        logger.info("[Pipeline] Stopping Stage 3...")
        if self.stage3:
            await self.stage3.stop(drain_queues=drain_queues, timeout=timeout)

        logger.info("[Pipeline] Stopping Stage 2...")
        if self.stage2:
            await self.stage2.stop(drain_queues=drain_queues, timeout=timeout)

        logger.info("[Pipeline] Stopping Stage 1...")
        if self.stage1:
            await self.stage1.stop(drain_queues=drain_queues, timeout=timeout)

        # Reject any pending response futures
        async with self._futures_lock:
            for request_id, future in self._response_futures.items():
                if not future.done():
                    future.set_exception(RuntimeError("Pipeline stopped"))
            self._response_futures.clear()

        logger.info("="*70)
        logger.info("  Pipeline Stopped Successfully")
        logger.info("="*70 + "\n")

    async def _route_pipeline(self):
        """
        Route work items between pipeline stages.

        This coroutine continuously moves completed work from one stage's
        output queue to the next stage's input queue.
        """
        logger.info("[Router] Started")

        while self._running:
            try:
                # Stage 1 → Stage 2
                if not self.stage1.output_queue.empty():
                    item = await asyncio.wait_for(
                        self.stage1.output_queue.get(),
                        timeout=0.1
                    )

                    # Check for errors from Stage 1
                    if isinstance(item.data, dict) and 'error' in item.data:
                        logger.error(f"[Router] Stage 1 error for {item.request_id}: {item.data['error']}")
                        await self._complete_request(item, error=item.data['error'])
                        continue

                    await self.stage2.input_queue.put(item)
                    logger.debug(f"[Router] {item.request_id}: Stage 1 → Stage 2")

                # Stage 2 → Stage 3
                if not self.stage2.output_queue.empty():
                    item = await asyncio.wait_for(
                        self.stage2.output_queue.get(),
                        timeout=0.1
                    )

                    # Check for errors from Stage 2
                    if isinstance(item.data, dict) and 'error' in item.data:
                        logger.error(f"[Router] Stage 2 error for {item.request_id}: {item.data['error']}")
                        await self._complete_request(item, error=item.data['error'])
                        continue

                    await self.stage3.input_queue.put(item)
                    logger.debug(f"[Router] {item.request_id}: Stage 2 → Stage 3")

                # Stage 3 → Complete
                if not self.stage3.output_queue.empty():
                    item = await asyncio.wait_for(
                        self.stage3.output_queue.get(),
                        timeout=0.1
                    )

                    # Check for errors from Stage 3
                    if isinstance(item.data, dict) and 'error' in item.data:
                        logger.error(f"[Router] Stage 3 error for {item.request_id}: {item.data['error']}")
                        await self._complete_request(item, error=item.data['error'])
                        continue

                    # Complete request successfully
                    await self._complete_request(item)
                    logger.debug(f"[Router] {item.request_id}: Stage 3 → Complete")

                # Small sleep to avoid busy loop
                await asyncio.sleep(0.001)

            except asyncio.TimeoutError:
                # No items to route, continue
                continue

            except asyncio.CancelledError:
                logger.info("[Router] Cancelled")
                break

            except Exception as e:
                logger.error(f"[Router] Unexpected error: {e}", exc_info=True)

        logger.info("[Router] Stopped")

    async def _complete_request(self, item: WorkItem, error: Optional[str] = None):
        """
        Complete a request by resolving its future.

        Args:
            item: Completed work item
            error: Error message if request failed
        """
        async with self._futures_lock:
            future = self._response_futures.pop(item.request_id, None)

        if future and not future.done():
            if error:
                future.set_exception(RuntimeError(error))
            else:
                future.set_result(item.data)

    def _process_load_mel(self, item: WorkItem) -> WorkItem:
        """
        Stage 1: Load audio and compute mel spectrogram.

        Args:
            item: WorkItem with audio_data in data dict

        Returns:
            WorkItem with mel features for Stage 2
        """
        request_id = item.request_id
        audio_data = item.data.get('audio_data')
        options = item.data.get('options', {})

        # Acquire buffers
        audio_buffer = None
        mel_buffer = None

        try:
            # Load audio (WhisperX handles this)
            audio_buffer = self.buffer_manager.acquire('audio')

            # For now, use WhisperX load_audio (TODO: optimize with buffer pool)
            import whisperx
            import tempfile
            import os

            # Write to temp file (WhisperX needs file path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name

            try:
                audio = whisperx.load_audio(tmp_path)
            finally:
                os.unlink(tmp_path)

            # Copy into buffer
            np.copyto(audio_buffer[:len(audio)], audio)

            # Acquire mel buffer
            mel_buffer = self.buffer_manager.acquire('mel')

            # Compute mel with zero-copy optimization (with variable-length support)
            # mel_np will be a SLICE of mel_buffer with the correct size
            mel_np, actual_frames = compute_mel_spectrogram_zerocopy(
                audio_buffer[:len(audio)],
                self.feature_extractor,
                output=mel_buffer
            )

            # Validate mel is C-contiguous
            validate_mel_contiguity(mel_np)

            logger.debug(
                f"[Stage1] Mel computed: {actual_frames} frames, "
                f"shape={mel_np.shape}, buffer={mel_buffer.shape}"
            )

            # Return work item for Stage 2
            return WorkItem(
                request_id=request_id,
                data={
                    'mel': mel_np,
                    'mel_buffer': mel_buffer,
                    'audio': audio_buffer[:len(audio)],
                    'audio_buffer': audio_buffer,
                    'options': options
                },
                metadata=item.metadata,
                stage=2
            )

        except Exception as e:
            # Release buffers on error
            if mel_buffer is not None:
                self.buffer_manager.release('mel', mel_buffer)
            if audio_buffer is not None:
                self.buffer_manager.release('audio', audio_buffer)

            logger.error(f"Stage 1 error for {request_id}: {e}", exc_info=True)

            return WorkItem(
                request_id=request_id,
                data={'error': f"Load/Mel failed: {str(e)}"},
                metadata={**item.metadata, 'error': True},
                stage=1
            )

    def _process_encoder(self, item: WorkItem) -> WorkItem:
        """
        Stage 2: Run encoder on mel spectrogram.

        Args:
            item: WorkItem with mel features from Stage 1

        Returns:
            WorkItem with encoder output for Stage 3
        """
        request_id = item.request_id
        mel = item.data.get('mel')
        mel_buffer = item.data.get('mel_buffer')
        audio = item.data.get('audio')
        audio_buffer = item.data.get('audio_buffer')
        options = item.data.get('options', {})

        encoder_buffer = None

        try:
            # Apply conv1d preprocessing (Bug #5 fix: mel 80→512)
            embeddings = self.conv1d_preprocessor.process(mel)  # (n_frames, 80) → (n_frames//2, 512)

            # Acquire encoder output buffer
            encoder_buffer = self.buffer_manager.acquire('encoder_output')

            # Run encoder (C++ + NPU) on embeddings (not raw mel!)
            encoder_output = self.cpp_encoder.forward(embeddings)  # (n_frames//2, 512) → (n_frames//2, 512)

            # Release mel buffer (no longer needed)
            self.buffer_manager.release('mel', mel_buffer)

            # Return work item for Stage 3
            return WorkItem(
                request_id=request_id,
                data={
                    'encoder_output': encoder_output,
                    'encoder_buffer': encoder_buffer,
                    'audio': audio,
                    'audio_buffer': audio_buffer,
                    'options': options
                },
                metadata=item.metadata,
                stage=3
            )

        except Exception as e:
            # Release buffers on error
            if encoder_buffer is not None:
                self.buffer_manager.release('encoder_output', encoder_buffer)
            if mel_buffer is not None:
                self.buffer_manager.release('mel', mel_buffer)

            logger.error(f"Stage 2 error for {request_id}: {e}", exc_info=True)

            return WorkItem(
                request_id=request_id,
                data={'error': f"Encoder failed: {str(e)}"},
                metadata={**item.metadata, 'error': True},
                stage=2
            )

    def _process_decoder_align(self, item: WorkItem) -> WorkItem:
        """
        Stage 3: Run decoder and alignment.

        Args:
            item: WorkItem with encoder output from Stage 2

        Returns:
            WorkItem with final transcription result
        """
        request_id = item.request_id
        encoder_output = item.data.get('encoder_output')
        encoder_buffer = item.data.get('encoder_buffer')
        audio = item.data.get('audio')
        audio_buffer = item.data.get('audio_buffer')
        options = item.data.get('options', {})

        try:
            # Decoder (WhisperX)
            # Note: For now we're using the full WhisperX pipeline since we can't
            # easily inject encoder output. In production, we'd modify WhisperX
            # or use a custom decoder that accepts encoder output directly.

            # Use the audio directly with WhisperX (it will re-run encoder internally)
            # This is suboptimal but functional for now
            result = self.python_decoder.transcribe(audio, batch_size=self.batch_size)

            # Alignment
            result = whisperx.align(
                result["segments"],
                self.model_a,
                self.metadata,
                audio,
                self.device
            )

            # Release buffers
            self.buffer_manager.release('encoder_output', encoder_buffer)
            self.buffer_manager.release('audio', audio_buffer)

            # Format response
            text = " ".join([seg["text"] for seg in result["segments"]])

            # Return final result
            return WorkItem(
                request_id=request_id,
                data={
                    'text': text,
                    'segments': result['segments'],
                    'words': result.get('word_segments', []),
                    'language': result.get('language', 'en')
                },
                metadata=item.metadata,
                stage=4  # Complete
            )

        except Exception as e:
            # Release buffers on error
            if encoder_buffer is not None:
                self.buffer_manager.release('encoder_output', encoder_buffer)
            if audio_buffer is not None:
                self.buffer_manager.release('audio', audio_buffer)

            logger.error(f"Stage 3 error for {request_id}: {e}", exc_info=True)

            return WorkItem(
                request_id=request_id,
                data={'error': f"Decoder/Align failed: {str(e)}"},
                metadata={**item.metadata, 'error': True},
                stage=3
            )

    async def transcribe(self, request: QueuedRequest, timeout: float = 60.0) -> Dict[str, Any]:
        """
        Process a transcription request through the pipeline.

        Args:
            request: Queued request to process
            timeout: Maximum time to wait for completion (seconds)

        Returns:
            Transcription result dict with text, segments, words

        Raises:
            RuntimeError: If request fails or times out
        """
        if not self._running:
            raise RuntimeError("Pipeline not running")

        # Create response future
        future = asyncio.Future()
        async with self._futures_lock:
            self._response_futures[request.request_id] = future

        try:
            # Create work item for Stage 1
            work_item = WorkItem(
                request_id=request.request_id,
                data={
                    'audio_data': request.audio_data,
                    'options': request.options
                },
                metadata={
                    'enqueued_at': request.enqueued_at,
                    'priority': request.priority
                },
                stage=1
            )

            # Put into Stage 1 input queue
            await self.stage1.input_queue.put(work_item)

            logger.info(f"[Pipeline] Request {request.request_id} submitted to Stage 1")

            # Wait for result
            result = await asyncio.wait_for(future, timeout=timeout)

            logger.info(f"[Pipeline] Request {request.request_id} completed successfully")

            return result

        except asyncio.TimeoutError:
            async with self._futures_lock:
                self._response_futures.pop(request.request_id, None)

            logger.error(f"[Pipeline] Request {request.request_id} timed out after {timeout}s")
            raise RuntimeError(f"Request timed out after {timeout}s")

        except Exception as e:
            async with self._futures_lock:
                self._response_futures.pop(request.request_id, None)

            logger.error(f"[Pipeline] Request {request.request_id} failed: {e}")
            raise

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline statistics.

        Returns:
            Dictionary with statistics for queue and all stages
        """
        stats = {
            'queue': await self.request_queue.get_stats(),
            'stage1': await self.stage1.get_stats() if self.stage1 else {},
            'stage2': await self.stage2.get_stats() if self.stage2 else {},
            'stage3': await self.stage3.get_stats() if self.stage3 else {},
            'pipeline': {
                'running': self._running,
                'pending_responses': len(self._response_futures)
            }
        }

        return stats

    async def print_stats(self):
        """Print comprehensive pipeline statistics"""
        stats = await self.get_stats()

        print("\n" + "="*70)
        print("  TRANSCRIPTION PIPELINE STATISTICS")
        print("="*70)

        # Queue stats
        print("\nRequest Queue:")
        queue_stats = stats['queue']
        print(f"  Total Enqueued:  {queue_stats['total_enqueued']}")
        print(f"  Total Dequeued:  {queue_stats['total_dequeued']}")
        print(f"  Current Size:    {queue_stats['current_size']}/{queue_stats['max_queue_size']}")
        print(f"  Avg Wait Time:   {queue_stats['avg_wait_time']*1000:.2f}ms")

        # Stage 1 stats
        if stats['stage1']:
            print("\nStage 1 (Load + Mel):")
            s1 = stats['stage1']
            print(f"  Total Processed: {s1['total_processed']}")
            print(f"  Avg Time:        {s1['avg_time']*1000:.2f}ms")
            print(f"  Queue Depth:     {s1['input_queue_size']}")

        # Stage 2 stats
        if stats['stage2']:
            print("\nStage 2 (Encoder):")
            s2 = stats['stage2']
            print(f"  Total Processed: {s2['total_processed']}")
            print(f"  Avg Time:        {s2['avg_time']*1000:.2f}ms")
            print(f"  Queue Depth:     {s2['input_queue_size']}")

        # Stage 3 stats
        if stats['stage3']:
            print("\nStage 3 (Decoder + Align):")
            s3 = stats['stage3']
            print(f"  Total Processed: {s3['total_processed']}")
            print(f"  Avg Time:        {s3['avg_time']*1000:.2f}ms")
            print(f"  Queue Depth:     {s3['input_queue_size']}")

        # Pipeline status
        pipeline = stats['pipeline']
        print("\nPipeline Status:")
        print(f"  Running:         {pipeline['running']}")
        print(f"  Pending:         {pipeline['pending_responses']}")

        print("="*70 + "\n")

    def is_healthy(self) -> bool:
        """Check if pipeline is healthy (all stages running)"""
        return (
            self._running and
            self.stage1.is_healthy() and
            self.stage2.is_healthy() and
            self.stage3.is_healthy()
        )


async def main():
    """Demonstration of transcription pipeline (requires full setup)"""
    print("Transcription Pipeline - Demonstration\n")
    print("NOTE: This demo requires full Whisper setup")
    print("      Run from server.py for complete integration\n")

    # This would require full initialization of encoder, decoder, etc.
    # For a complete demo, see server.py integration

    print("✅ Pipeline module loaded successfully")
    return 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(main()))
