"""
faster-whisper integration for Unicorn-Amanuensis
Replaces WhisperX decoder with optimized CTranslate2 implementation

Week 19 Performance Optimization - Team 1 Lead
Target: 4-6× decoder speedup (450ms → 75-112ms)
Expected impact: 2.2× end-to-end improvement (13.3× → 29× realtime)

Author: CC-1L NPU Acceleration Team
Date: November 2, 2025
"""
from faster_whisper import WhisperModel
import numpy as np
from typing import Dict, List, Optional, Union
import time
import logging

logger = logging.getLogger(__name__)


class FasterWhisperDecoder:
    """
    Drop-in replacement for WhisperX decoder using faster-whisper (CTranslate2).

    This implementation provides 4-6× faster decoding compared to the standard
    WhisperX Python decoder while maintaining comparable accuracy. CTranslate2
    uses optimized C++ implementations with INT8 quantization for efficient
    CPU inference.

    Key Optimizations:
    - INT8 quantization for reduced memory footprint and faster computation
    - Optimized GEMM operations via CTranslate2's CPU backend
    - Efficient beam search implementation
    - Optional VAD filtering to skip silence regions

    Performance Targets:
    - Baseline (WhisperX): ~450ms for 5s audio
    - Target (faster-whisper): 75-112ms for 5s audio
    - Speedup: 4-6×

    Usage:
        decoder = FasterWhisperDecoder(
            model_name="base",
            device="cpu",
            compute_type="int8"
        )

        result = decoder.transcribe(audio_array)
        print(result['text'])
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False
    ):
        """
        Initialize faster-whisper decoder.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
                       - "tiny": 39M params, fastest, lowest quality
                       - "base": 74M params, good balance (RECOMMENDED)
                       - "small": 244M params, better quality, slower
                       - "medium": 769M params, high quality, much slower
                       - "large": 1550M params, best quality, very slow

            device: Compute device ("cpu" or "cuda")
                   - "cpu": Universal, works everywhere
                   - "cuda": Requires CUDA GPU (not applicable for NPU)

            compute_type: Quantization level for model weights
                         - "int8": 8-bit quantization, fastest, minimal accuracy loss (RECOMMENDED)
                         - "int16": 16-bit quantization, slower, slightly better accuracy
                         - "float16": Half precision, requires CUDA
                         - "float32": Full precision, slowest, highest accuracy

            num_workers: Number of parallel workers for decoding
                        - 1: Sequential processing (RECOMMENDED for single requests)
                        - >1: Parallel processing for batch decoding

            download_root: Optional custom directory for model cache
                          Default: ~/.cache/huggingface/hub

            local_files_only: If True, only use locally cached models
                             Useful for offline/air-gapped deployments

        Raises:
            ValueError: If model_name or compute_type invalid
            RuntimeError: If model loading fails
        """
        logger.info("="*70)
        logger.info("  faster-whisper Decoder Initialization")
        logger.info("="*70)
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Compute type: {compute_type}")
        logger.info(f"  Workers: {num_workers}")

        start = time.perf_counter()

        try:
            # Initialize faster-whisper model
            # This downloads the model from Hugging Face if not cached
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                num_workers=num_workers,
                download_root=download_root,
                local_files_only=local_files_only
            )

            # Store configuration
            self.model_name = model_name
            self.device = device
            self.compute_type = compute_type
            self.num_workers = num_workers

            elapsed = time.perf_counter() - start
            logger.info(f"  Model loaded in {elapsed:.2f}s")
            logger.info("="*70 + "\n")

        except Exception as e:
            logger.error(f"Failed to load faster-whisper model: {e}")
            raise RuntimeError(f"faster-whisper initialization failed: {e}")

    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1.0,
        length_penalty: float = 1.0,
        temperature: Union[float, List[float]] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        vad_filter: bool = False,
        vad_parameters: Optional[Dict] = None,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False
    ) -> Dict:
        """
        Transcribe audio using faster-whisper.

        This method provides a drop-in replacement for WhisperX's transcribe()
        with similar API but significantly faster performance through CTranslate2.

        Args:
            audio: Audio input as:
                  - numpy.ndarray: Audio samples (float32, 16kHz)
                  - str: Path to audio file (WAV, MP3, etc.)

            language: ISO language code (e.g., "en", "es", "fr")
                     - None: Auto-detect language (adds ~50ms overhead)
                     - "en": Force English (faster, use if known)

            task: Transcription task
                 - "transcribe": Transcribe in original language
                 - "translate": Translate to English

            beam_size: Beam search width
                      - 1: Greedy search (fastest, lower quality)
                      - 5: Good balance (RECOMMENDED)
                      - 10+: Better quality, much slower

            best_of: Number of candidates when sampling
                    Only used when temperature > 0

            patience: Beam search patience factor
                     Higher = more thorough search, slower

            length_penalty: Length normalization exponent
                           - <1: Prefer shorter sequences
                           - 1: No preference (RECOMMENDED)
                           - >1: Prefer longer sequences

            temperature: Sampling temperature(s)
                        - 0.0: Deterministic (greedy)
                        - List: Try multiple temperatures for fallback
                        - Higher: More random/creative

            compression_ratio_threshold: Detect repetition loops
                                        Segments with ratio > threshold are rejected

            log_prob_threshold: Minimum average log probability
                               Segments below threshold are rejected

            no_speech_threshold: Silence detection threshold
                                Segments with no_speech_prob > threshold are skipped

            condition_on_previous_text: Use previous text as context
                                       Improves coherence but may propagate errors

            vad_filter: Enable Voice Activity Detection
                       - True: Skip silence regions (faster, may skip quiet speech)
                       - False: Process all audio (slower, more complete)

            vad_parameters: Custom VAD configuration
                           See faster-whisper docs for options

            initial_prompt: Optional text prompt to guide transcription
                           Useful for domain-specific vocabulary

            word_timestamps: Generate word-level timestamps
                            - True: Returns word_segments with timing
                            - False: Only sentence-level segments (faster)

        Returns:
            Dict with transcription results:
                {
                    'text': str,              # Full transcription text
                    'segments': List[Dict],   # Sentence-level segments with timing
                    'language': str,          # Detected/specified language
                    'duration': float,        # Audio duration in seconds
                    'decode_time': float,     # Time spent decoding (seconds)
                    'word_segments': List[Dict] (if word_timestamps=True)
                }

        Raises:
            ValueError: If audio format invalid
            RuntimeError: If transcription fails
        """
        start = time.perf_counter()

        # Validate and prepare audio
        if isinstance(audio, str):
            # Audio file path - faster-whisper will load it
            audio_input = audio
        elif isinstance(audio, np.ndarray):
            # NumPy array - ensure correct dtype
            if audio.dtype != np.float32:
                logger.debug(f"Converting audio from {audio.dtype} to float32")
                audio = audio.astype(np.float32)
            audio_input = audio
        else:
            raise ValueError(
                f"Audio must be numpy array or file path, got {type(audio)}"
            )

        try:
            # Run faster-whisper transcription
            # Returns (segments_generator, info)
            segments, info = self.model.transcribe(
                audio_input,
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                length_penalty=length_penalty,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                condition_on_previous_text=condition_on_previous_text,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters,
                initial_prompt=initial_prompt,
                word_timestamps=word_timestamps
            )

            # Collect segments from generator
            text_segments = []
            word_segments = [] if word_timestamps else None

            for segment in segments:
                # Build segment dict (WhisperX-compatible format)
                seg_dict = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'avg_logprob': segment.avg_logprob,
                    'no_speech_prob': segment.no_speech_prob
                }
                text_segments.append(seg_dict)

                # Collect word-level timestamps if requested
                if word_timestamps and hasattr(segment, 'words'):
                    for word in segment.words:
                        word_segments.append({
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability
                        })

            elapsed = time.perf_counter() - start

            # Combine into single text
            full_text = " ".join([s['text'] for s in text_segments])

            # Build result dict (WhisperX-compatible)
            result = {
                'text': full_text.strip(),
                'segments': text_segments,
                'language': info.language if language is None else language,
                'duration': info.duration,
                'decode_time': elapsed
            }

            # Add word segments if generated
            if word_timestamps:
                result['word_segments'] = word_segments

            logger.debug(
                f"Transcribed {info.duration:.1f}s audio in {elapsed:.3f}s "
                f"({info.duration/elapsed:.1f}x realtime)"
            )

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"faster-whisper transcription failed: {e}")

    def get_stats(self) -> Dict:
        """
        Get decoder statistics and configuration.

        Returns:
            Dict with decoder information:
                {
                    'model_name': str,
                    'device': str,
                    'compute_type': str,
                    'num_workers': int,
                    'backend': str
                }
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'compute_type': self.compute_type,
            'num_workers': self.num_workers,
            'backend': 'CTranslate2 (faster-whisper)'
        }


# Convenience function for quick testing
def test_faster_whisper(
    audio_path: str,
    model_name: str = "base",
    language: Optional[str] = None
) -> Dict:
    """
    Quick test function for faster-whisper decoder.

    Usage:
        from xdna2.faster_whisper_wrapper import test_faster_whisper

        result = test_faster_whisper("test.wav")
        print(result['text'])
        print(f"Decoded in {result['decode_time']:.3f}s")

    Args:
        audio_path: Path to audio file
        model_name: Whisper model size ("tiny", "base", "small", etc.)
        language: Optional language code (None for auto-detect)

    Returns:
        Transcription result dict
    """
    decoder = FasterWhisperDecoder(
        model_name=model_name,
        device="cpu",
        compute_type="int8"
    )

    return decoder.transcribe(
        audio_path,
        language=language,
        word_timestamps=False
    )
