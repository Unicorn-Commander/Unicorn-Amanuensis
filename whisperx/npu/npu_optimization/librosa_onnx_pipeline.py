#!/usr/bin/env python3
"""
Librosa → ONNX Runtime Integration Pipeline
============================================
Complete pipeline: librosa preprocessing + ONNX Runtime inference

Goal: Achieve 220x realtime transcription (UC-Meeting-Ops proven approach)

Architecture:
    Audio File → librosa (mel) → ONNX Encoder → ONNX Decoder → Text

This is what UC-Meeting-Ops uses to achieve 220x speedup!
- librosa for mel preprocessing (CPU, but accurate and fast)
- ONNX Runtime for Whisper inference (optimized, supports INT8)
- No custom NPU kernels needed for 220x target
"""

import numpy as np
import librosa
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import onnxruntime as ort

# Import our librosa preprocessor
import sys
sys.path.insert(0, str(Path(__file__).parent / "mel_kernels"))
from mel_preprocessing_librosa import LibrosaMelPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LibrosaONNXWhisper:
    """Complete pipeline: librosa preprocessing + ONNX Runtime inference"""

    def __init__(self,
                 model_path: str,
                 execution_provider: str = 'CPUExecutionProvider',
                 use_int8: bool = False):
        """
        Initialize Librosa + ONNX Whisper pipeline

        Args:
            model_path: Path to ONNX model directory
            execution_provider: ONNX Runtime execution provider
                Options: 'CPUExecutionProvider', 'OpenVINOExecutionProvider'
            use_int8: Use INT8 quantized models if available
        """
        self.model_path = Path(model_path)
        self.execution_provider = execution_provider
        self.use_int8 = use_int8

        # Initialize librosa preprocessor
        logger.info("Initializing librosa mel preprocessor...")
        self.preprocessor = LibrosaMelPreprocessor(
            sample_rate=16000,
            n_fft=512,  # Whisper uses 512 FFT
            hop_length=160,
            n_mels=80,
            fmin=0.0,
            fmax=8000.0
        )

        # Load ONNX models
        self._load_models()

        # Initialize tokenizer
        self._init_tokenizer()

        logger.info(f"LibrosaONNXWhisper initialized!")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Execution Provider: {execution_provider}")
        logger.info(f"  INT8: {use_int8}")

    def _load_models(self):
        """Load encoder and decoder ONNX models"""
        logger.info("Loading ONNX models...")

        # Determine model file names
        if self.use_int8:
            encoder_name = "encoder_model_int8.onnx"
            decoder_name = "decoder_model_int8.onnx"
            decoder_past_name = "decoder_with_past_model_int8.onnx"
        else:
            encoder_name = "encoder_model.onnx"
            decoder_name = "decoder_model.onnx"
            decoder_past_name = "decoder_with_past_model.onnx"

        encoder_path = self.model_path / encoder_name
        decoder_path = self.model_path / decoder_name
        decoder_past_path = self.model_path / decoder_past_name

        # Check files exist
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found: {encoder_path}")
        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder not found: {decoder_path}")

        # Setup execution providers
        available_providers = ort.get_available_providers()
        logger.info(f"Available providers: {available_providers}")

        if self.execution_provider not in available_providers:
            logger.warning(f"{self.execution_provider} not available, using CPUExecutionProvider")
            providers = ['CPUExecutionProvider']
        else:
            providers = [self.execution_provider, 'CPUExecutionProvider']

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load encoder
        logger.info(f"Loading encoder: {encoder_path}")
        self.encoder_session = ort.InferenceSession(
            str(encoder_path),
            sess_options=sess_options,
            providers=providers
        )

        # Load decoder
        logger.info(f"Loading decoder: {decoder_path}")
        self.decoder_session = ort.InferenceSession(
            str(decoder_path),
            sess_options=sess_options,
            providers=providers
        )

        # Load decoder with past (if available)
        if decoder_past_path.exists():
            logger.info(f"Loading decoder with past: {decoder_past_path}")
            self.decoder_with_past_session = ort.InferenceSession(
                str(decoder_past_path),
                sess_options=sess_options,
                providers=providers
            )
        else:
            logger.warning("Decoder with past not found, will use slower decoding")
            self.decoder_with_past_session = None

        # Log input/output info
        logger.info("\nEncoder inputs:")
        for inp in self.encoder_session.get_inputs():
            logger.info(f"  {inp.name}: {inp.shape} ({inp.type})")
        logger.info("Encoder outputs:")
        for out in self.encoder_session.get_outputs():
            logger.info(f"  {out.name}: {out.shape} ({out.type})")

        logger.info("\nDecoder inputs:")
        for inp in self.decoder_session.get_inputs():
            logger.info(f"  {inp.name}: {inp.shape} ({inp.type})")
        logger.info("Decoder outputs:")
        for out in self.decoder_session.get_outputs():
            logger.info(f"  {out.name}: {out.shape} ({out.type})")

    def _init_tokenizer(self):
        """Initialize Whisper tokenizer"""
        try:
            from transformers import WhisperTokenizer
            self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            self.tokenizer = None

    def transcribe(self, audio_file: str, max_tokens: int = 448) -> Dict[str, Any]:
        """
        Transcribe audio file

        Args:
            audio_file: Path to audio file
            max_tokens: Maximum tokens to generate

        Returns:
            Dict with transcription results and timing
        """
        total_start = time.perf_counter()
        timings = {}

        # 1. Load audio with librosa
        logger.info(f"Loading audio: {audio_file}")
        load_start = time.perf_counter()
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        duration = len(audio) / sr
        timings['audio_load'] = time.perf_counter() - load_start
        logger.info(f"  Duration: {duration:.2f}s ({len(audio)} samples)")

        # 2. Compute mel spectrogram with librosa
        logger.info("Computing mel spectrogram with librosa...")
        mel_start = time.perf_counter()
        mel_spec, proc_time = self.preprocessor.process_audio(audio)
        timings['mel_spectrogram'] = time.perf_counter() - mel_start
        logger.info(f"  Mel shape: {mel_spec.shape}")
        logger.info(f"  Librosa time: {proc_time:.4f}s ({duration/proc_time:.1f}x realtime)")

        # 3. Prepare input for encoder (Whisper expects batch dimension)
        # Shape: (batch, n_mels, time)
        mel_input = np.expand_dims(mel_spec, axis=0).astype(np.float32)

        # Pad to 3000 frames (30 seconds at 160 hop length)
        target_frames = 3000
        if mel_input.shape[2] < target_frames:
            padding = target_frames - mel_input.shape[2]
            mel_input = np.pad(mel_input, ((0, 0), (0, 0), (0, padding)), mode='constant')
        elif mel_input.shape[2] > target_frames:
            mel_input = mel_input[:, :, :target_frames]

        logger.info(f"  Encoder input shape: {mel_input.shape}")

        # 4. Run encoder with ONNX Runtime
        logger.info("Running ONNX encoder...")
        encoder_start = time.perf_counter()
        encoder_outputs = self.encoder_session.run(None, {'input_features': mel_input})
        hidden_states = encoder_outputs[0]
        timings['encoder'] = time.perf_counter() - encoder_start
        logger.info(f"  Hidden states shape: {hidden_states.shape}")
        logger.info(f"  Encoder time: {timings['encoder']:.4f}s")

        # 5. Run decoder with ONNX Runtime
        logger.info("Running ONNX decoder...")
        decoder_start = time.perf_counter()

        if self.tokenizer is not None:
            # Use proper tokenizer
            text, tokens_generated = self._decode_with_tokenizer(
                hidden_states, max_tokens
            )
            timings['decoder'] = time.perf_counter() - decoder_start
            logger.info(f"  Generated {tokens_generated} tokens in {timings['decoder']:.4f}s")
        else:
            # Fallback: simple decoding
            text, tokens_generated = self._decode_simple(hidden_states, max_tokens)
            timings['decoder'] = time.perf_counter() - decoder_start

        # 6. Calculate total time and metrics
        timings['total'] = time.perf_counter() - total_start

        # Calculate realtime factors
        rtf_total = timings['total'] / duration if duration > 0 else 0
        rtf_mel = timings['mel_spectrogram'] / duration if duration > 0 else 0
        rtf_encoder = timings['encoder'] / duration if duration > 0 else 0
        rtf_decoder = timings['decoder'] / duration if duration > 0 else 0

        realtime_factor = duration / timings['total'] if timings['total'] > 0 else 0

        result = {
            'text': text,
            'duration': duration,
            'timings': timings,
            'realtime_factors': {
                'total': realtime_factor,
                'mel': duration / timings['mel_spectrogram'],
                'encoder': duration / timings['encoder'],
                'decoder': duration / timings['decoder']
            },
            'tokens_generated': tokens_generated,
            'model_info': {
                'execution_provider': self.execution_provider,
                'int8': self.use_int8,
                'has_kv_cache': self.decoder_with_past_session is not None
            }
        }

        logger.info(f"\nTranscription complete!")
        logger.info(f"  Text: '{text}'")
        logger.info(f"  Total time: {timings['total']:.2f}s")
        logger.info(f"  Realtime factor: {realtime_factor:.1f}x")

        return result

    def _decode_with_tokenizer(self,
                               hidden_states: np.ndarray,
                               max_tokens: int) -> Tuple[str, int]:
        """
        Decode using Whisper tokenizer

        Args:
            hidden_states: Encoder output
            max_tokens: Maximum tokens to generate

        Returns:
            (text, tokens_generated)
        """
        # Start tokens for English transcription
        # 50258 = <|startoftranscript|>
        # 50259 = <|en|>
        # 50360 = <|transcribe|>
        # 50365 = <|notimestamps|>
        decoder_input_ids = np.array([[50258, 50259, 50360, 50365]], dtype=np.int64)

        generated_tokens = []
        use_past = self.decoder_with_past_session is not None
        past_key_values = None

        for step in range(max_tokens):
            if decoder_input_ids.shape[1] >= 448:
                logger.warning("Reached max sequence length (448 tokens)")
                break

            if use_past and past_key_values is not None:
                # Use decoder with KV cache (efficient)
                inputs = {'input_ids': decoder_input_ids[:, -1:]}

                # Add past key values
                for i, kv in enumerate(past_key_values):
                    inputs[f'past_key_values.{i}.decoder.key'] = kv[0]
                    inputs[f'past_key_values.{i}.decoder.value'] = kv[1]
                    inputs[f'past_key_values.{i}.encoder.key'] = kv[2]
                    inputs[f'past_key_values.{i}.encoder.value'] = kv[3]

                outputs = self.decoder_with_past_session.run(None, inputs)
                logits = outputs[0]

                # Update KV cache
                new_past = []
                for i in range(6):  # 6 decoder layers
                    new_past.append((
                        outputs[i*2 + 1],  # decoder key
                        outputs[i*2 + 2],  # decoder value
                        past_key_values[i][2],  # encoder key (unchanged)
                        past_key_values[i][3]   # encoder value (unchanged)
                    ))
                past_key_values = new_past
            else:
                # First pass - use regular decoder
                outputs = self.decoder_session.run(None, {
                    'input_ids': decoder_input_ids,
                    'encoder_hidden_states': hidden_states
                })
                logits = outputs[0]

                # Extract KV cache if available
                if use_past and len(outputs) == 25:  # 1 logits + 24 KV tensors
                    past_key_values = []
                    for i in range(6):
                        past_key_values.append((
                            outputs[i*2 + 1],   # decoder key
                            outputs[i*2 + 2],   # decoder value
                            outputs[i*2 + 13],  # encoder key
                            outputs[i*2 + 14]   # encoder value
                        ))

            # Get next token
            next_token_id = np.argmax(logits[0, -1, :])

            # Check for end token
            if next_token_id == 50257:  # <|endoftext|>
                break

            # Add to sequence
            decoder_input_ids = np.concatenate([
                decoder_input_ids,
                np.array([[next_token_id]], dtype=np.int64)
            ], axis=1)

            generated_tokens.append(next_token_id)

        # Decode tokens to text
        if generated_tokens:
            # Filter out special tokens
            text_tokens = [t for t in generated_tokens if t < 50257]
            if text_tokens:
                text = self.tokenizer.decode(text_tokens, skip_special_tokens=True)
            else:
                text = ""
        else:
            text = ""

        return text, len(generated_tokens)

    def _decode_simple(self,
                       hidden_states: np.ndarray,
                       max_tokens: int) -> Tuple[str, int]:
        """
        Simple decoding without tokenizer (fallback)

        Args:
            hidden_states: Encoder output
            max_tokens: Maximum tokens to generate

        Returns:
            (text, tokens_generated)
        """
        # Start tokens
        decoder_input_ids = np.array([[50258, 50259, 50360, 50365]], dtype=np.int64)

        # Run decoder once
        outputs = self.decoder_session.run(None, {
            'input_ids': decoder_input_ids,
            'encoder_hidden_states': hidden_states
        })

        # Simple placeholder output
        text = "[Audio processed - tokenizer not available for decoding]"
        return text, 0

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'execution_provider': self.execution_provider,
            'int8_enabled': self.use_int8,
            'has_kv_cache': self.decoder_with_past_session is not None,
            'model_path': str(self.model_path)
        }


def main():
    """Test the pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Test librosa + ONNX Whisper pipeline')
    parser.add_argument('--audio', type=str, required=True,
                       help='Path to audio file')
    parser.add_argument('--model', type=str,
                       default='/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/onnx',
                       help='Path to ONNX model directory')
    parser.add_argument('--provider', type=str, default='CPUExecutionProvider',
                       choices=['CPUExecutionProvider', 'OpenVINOExecutionProvider'],
                       help='ONNX Runtime execution provider')
    parser.add_argument('--int8', action='store_true',
                       help='Use INT8 quantized models')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = LibrosaONNXWhisper(
        model_path=args.model,
        execution_provider=args.provider,
        use_int8=args.int8
    )

    # Transcribe
    result = pipeline.transcribe(args.audio)

    # Print results
    print("\n" + "="*80)
    print("TRANSCRIPTION RESULTS")
    print("="*80)
    print(f"Text: {result['text']}")
    print(f"\nDuration: {result['duration']:.2f}s")
    print(f"Total time: {result['timings']['total']:.2f}s")
    print(f"Realtime factor: {result['realtime_factors']['total']:.1f}x")
    print("\nStage breakdown:")
    print(f"  Audio load:      {result['timings']['audio_load']:.4f}s")
    print(f"  Mel spectrogram: {result['timings']['mel_spectrogram']:.4f}s ({result['realtime_factors']['mel']:.1f}x)")
    print(f"  Encoder:         {result['timings']['encoder']:.4f}s ({result['realtime_factors']['encoder']:.1f}x)")
    print(f"  Decoder:         {result['timings']['decoder']:.4f}s ({result['realtime_factors']['decoder']:.1f}x)")
    print("="*80)


if __name__ == "__main__":
    main()
