#!/usr/bin/env python3
"""
Complete End-to-End Whisper Pipeline with NPU Encoder
Integrates: Audio → Mel → NPU Encoder → Decoder → Text
"""

import numpy as np
import torch
import librosa
import time
from pathlib import Path
from typing import Optional, Union
from whisper_encoder_optimized import WhisperEncoderOptimized

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  OpenAI Whisper not available - will use standalone mode")


class WhisperNPUPipeline:
    """
    Complete Whisper pipeline with NPU-accelerated encoder

    Pipeline: Audio → Mel Spectrogram → NPU Encoder → Whisper Decoder → Text
    """

    def __init__(self,
                 model_size: str = "base",
                 device_id: int = 0,
                 language: str = "en",
                 use_decoder: bool = True):
        """
        Initialize complete Whisper pipeline with NPU encoder

        Args:
            model_size: "base" (6 layers, 512 dims)
            device_id: NPU device ID
            language: target language for transcription
            use_decoder: whether to use Whisper decoder (requires whisper package)
        """
        self.model_size = model_size
        self.language = language
        self.use_decoder = use_decoder and WHISPER_AVAILABLE

        print("=" * 70)
        print("🚀 Whisper NPU Pipeline - Complete End-to-End System")
        print("=" * 70)

        # Initialize NPU encoder
        print("\n📦 Initializing NPU encoder...")
        self.encoder = WhisperEncoderOptimized(
            model_size=model_size,
            device_id=device_id
        )

        # Load Whisper decoder if available
        if self.use_decoder:
            print("\n📦 Loading Whisper decoder...")
            try:
                self.whisper_model = whisper.load_model(model_size)
                self.decoder = self.whisper_model.decoder
                self.tokenizer = whisper.tokenizer.get_tokenizer(
                    multilingual=True,
                    language=language,
                    task="transcribe"
                )
                print(f"   ✅ Whisper {model_size} decoder loaded")
            except Exception as e:
                print(f"   ⚠️  Failed to load decoder: {e}")
                self.use_decoder = False

        # Audio parameters (Whisper standard)
        self.sample_rate = 16000
        self.n_fft = 400
        self.hop_length = 160
        self.n_mels = 80

        print("\n✅ Pipeline initialized and ready!")
        print(f"   - NPU Encoder: {model_size} (6 layers, 512 dims)")
        print(f"   - Decoder: {'Whisper ' + model_size if self.use_decoder else 'Disabled'}")
        print(f"   - Language: {language}")

    def load_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Load audio file and resample to 16kHz

        Args:
            audio_path: path to audio file

        Returns:
            audio: (n_samples,) float32 array
        """
        print(f"\n🎵 Loading audio: {Path(audio_path).name}")

        # Load and resample audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        duration = len(audio) / self.sample_rate
        print(f"   Duration: {duration:.2f}s")
        print(f"   Sample rate: {sr} Hz")
        print(f"   Samples: {len(audio):,}")

        return audio

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram using librosa (Whisper-compatible)

        Args:
            audio: (n_samples,) audio array

        Returns:
            mel: (n_frames, n_mels) mel spectrogram
        """
        print("\n🎼 Computing mel spectrogram...")

        # Pad audio to 30 seconds if needed
        target_length = self.sample_rate * 30  # 30 seconds
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        elif len(audio) > target_length:
            # For longer audio, we'll process in chunks
            audio = audio[:target_length]  # Just use first 30s for now

        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=0,
            fmax=8000
        )

        # Convert to log scale (dB)
        mel = librosa.power_to_db(mel, ref=np.max)

        # Transpose to (time, mels) and normalize
        mel = mel.T
        mel = (mel + 80.0) / 80.0  # Normalize to roughly [0, 1]

        print(f"   Mel shape: {mel.shape}")
        print(f"   Mel range: [{mel.min():.3f}, {mel.max():.3f}]")

        return mel.astype(np.float32)

    def encode_npu(self, mel: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Encode mel spectrogram using NPU encoder

        Args:
            mel: (n_frames, n_mels) mel spectrogram
            verbose: print progress

        Returns:
            encoder_output: (n_frames, n_dims) encoder hidden states
        """
        if verbose:
            print("\n⚡ Encoding on NPU...")

        start_time = time.time()
        encoder_output = self.encoder.forward(mel, verbose=verbose)
        elapsed = time.time() - start_time

        if verbose:
            print(f"\n✅ NPU encoding complete: {elapsed:.2f}s")
            print(f"   Encoder output shape: {encoder_output.shape}")

        return encoder_output

    def decode_text(self, encoder_output: np.ndarray) -> dict:
        """
        Decode encoder output to text using Whisper decoder

        Args:
            encoder_output: (n_frames, n_dims) encoder hidden states

        Returns:
            result: dict with 'text' and metadata
        """
        if not self.use_decoder:
            return {
                'text': '[Decoder not available - NPU encoder only]',
                'language': self.language,
                'encoder_only': True
            }

        print("\n📝 Decoding to text...")

        start_time = time.time()

        # Convert to torch tensor
        encoder_output_torch = torch.from_numpy(encoder_output).unsqueeze(0)  # (1, n_frames, n_dims)

        # Get initial tokens
        initial_tokens = self.tokenizer.sot_sequence

        # Decode using Whisper decoder
        with torch.no_grad():
            # Simple greedy decoding
            tokens = initial_tokens
            max_length = 224

            for _ in range(max_length):
                # Get logits for next token
                tokens_tensor = torch.tensor([tokens]).long()
                logits = self.decoder(tokens_tensor, encoder_output_torch)

                # Get next token (greedy)
                next_token = logits[0, -1].argmax().item()

                # Check for end token
                if next_token == self.tokenizer.eot:
                    break

                tokens.append(next_token)

            # Decode tokens to text
            text = self.tokenizer.decode(tokens[len(initial_tokens):])

        elapsed = time.time() - start_time

        print(f"   Decoding time: {elapsed:.2f}s")
        print(f"   Generated {len(tokens)} tokens")

        return {
            'text': text,
            'language': self.language,
            'tokens': tokens,
            'decode_time': elapsed
        }

    def transcribe(self,
                   audio_path: Union[str, Path],
                   verbose: bool = True) -> dict:
        """
        Complete transcription pipeline

        Args:
            audio_path: path to audio file
            verbose: print detailed progress

        Returns:
            result: dict with transcription and timing info
        """
        if verbose:
            print("\n" + "=" * 70)
            print("🎯 Starting End-to-End Transcription")
            print("=" * 70)

        total_start = time.time()

        # Step 1: Load audio
        t0 = time.time()
        audio = self.load_audio(audio_path)
        audio_time = time.time() - t0

        # Step 2: Compute mel spectrogram
        t0 = time.time()
        mel = self.compute_mel_spectrogram(audio)
        mel_time = time.time() - t0

        # Step 3: NPU encoding
        t0 = time.time()
        encoder_output = self.encode_npu(mel, verbose=verbose)
        encode_time = time.time() - t0

        # Step 4: Decoding (if available)
        if self.use_decoder:
            result = self.decode_text(encoder_output)
            decode_time = result['decode_time']
        else:
            result = {
                'text': '[NPU encoder completed - decoder not configured]',
                'encoder_output_shape': encoder_output.shape,
                'encoder_only': True
            }
            decode_time = 0

        total_time = time.time() - total_start

        # Add timing info
        result['timing'] = {
            'audio_loading': audio_time,
            'mel_computation': mel_time,
            'npu_encoding': encode_time,
            'decoding': decode_time,
            'total': total_time
        }

        result['audio_duration'] = len(audio) / self.sample_rate
        result['realtime_factor'] = result['audio_duration'] / total_time if total_time > 0 else 0

        if verbose:
            self._print_summary(result)

        return result

    def _print_summary(self, result: dict):
        """Print transcription summary"""
        print("\n" + "=" * 70)
        print("📊 TRANSCRIPTION SUMMARY")
        print("=" * 70)

        print(f"\n📝 Transcription:")
        print(f"   {result['text']}")

        print(f"\n⏱️  Timing Breakdown:")
        timing = result['timing']
        print(f"   Audio loading:    {timing['audio_loading']*1000:>8.2f}ms")
        print(f"   Mel computation:  {timing['mel_computation']*1000:>8.2f}ms")
        print(f"   NPU encoding:     {timing['npu_encoding']*1000:>8.2f}ms")
        if timing['decoding'] > 0:
            print(f"   Decoding:         {timing['decoding']*1000:>8.2f}ms")
        print(f"   {'─' * 40}")
        print(f"   Total:            {timing['total']*1000:>8.2f}ms")

        print(f"\n🚀 Performance:")
        print(f"   Audio duration:   {result['audio_duration']:.2f}s")
        print(f"   Processing time:  {timing['total']:.2f}s")
        print(f"   Realtime factor:  {result['realtime_factor']:.2f}x")

        print("\n" + "=" * 70)


def main():
    """Test the complete pipeline"""
    import sys

    # Initialize pipeline
    pipeline = WhisperNPUPipeline(
        model_size="base",
        language="en",
        use_decoder=False  # Set to True if you have whisper installed
    )

    # Check for audio file argument
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        # Create synthetic audio for testing
        print("\n🎵 No audio file provided, creating synthetic test audio...")
        audio_path = "/tmp/test_audio_whisper.wav"

        # Generate 5 seconds of synthetic audio (440 Hz tone)
        sample_rate = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Save as WAV
        import scipy.io.wavfile
        scipy.io.wavfile.write(audio_path, sample_rate, (audio * 32767).astype(np.int16))
        print(f"   Created: {audio_path} ({duration}s, 440 Hz tone)")

    # Run transcription
    result = pipeline.transcribe(audio_path, verbose=True)

    print("\n✅ Pipeline test complete!")
    return result


if __name__ == "__main__":
    main()
