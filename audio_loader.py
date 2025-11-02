"""
Simple audio loader using scipy (no ffmpeg required)
Drop-in replacement for whisperx.load_audio for WAV files
"""
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

def load_audio(file_path, sr=16000):
    """
    Load audio from file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file (WAV format)
        sr: Target sample rate (default: 16000 Hz for Whisper)
    
    Returns:
        Audio array as float32 in range [-1, 1], shape (n_samples,)
    """
    # Load WAV file
    input_sr, audio = wav.read(file_path)
    
    # Convert to float32 in range [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
    
    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if input_sr != sr:
        # Calculate number of samples in output
        num_samples = int(len(audio) * sr / input_sr)
        audio = signal.resample(audio, num_samples)
    
    return audio.astype(np.float32)
