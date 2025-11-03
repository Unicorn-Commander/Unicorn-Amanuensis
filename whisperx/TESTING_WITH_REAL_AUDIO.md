# Testing Decoder with Real Audio

## Quick Start

### Option 1: Use HTTP API (Recommended)

```bash
# Start server
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
python3 server_production.py

# Test with audio file
curl -X POST \
  -F "file=@/path/to/your/audio.wav" \
  http://localhost:9004/transcribe
```

### Option 2: Python Script

```python
from npu.npu_optimization.onnx_whisper_npu import ONNXWhisperNPU

# Initialize
decoder = ONNXWhisperNPU()
decoder.initialize(model_size="base")

# Transcribe
result = decoder.transcribe_audio("/path/to/audio.wav")
print(result['text'])
```

### Option 3: Command Line

```bash
python3 test_kv_cache_fix.py  # Will try known audio paths
```

---

## Getting Test Audio

### Method 1: Record Your Own (Linux)

```bash
# Install sox if needed
sudo apt-get install sox

# Record 10 seconds
sox -d test_recording.wav rate 16k trim 0 10

# Record until stopped (Ctrl+C)
sox -d test_recording.wav rate 16k
```

### Method 2: Use Existing Files

Check these locations:
- `/home/ucadmin/VibeVoice/` - May have existing audio files
- `/home/ucadmin/Development/` - Development test files
- `/tmp/` - Temporary recordings

### Method 3: Download Public Domain Speech

```bash
# LibriVox audio book samples (public domain)
wget https://librivox.org/uploads/short/sample.mp3
ffmpeg -i sample.mp3 -ar 16000 test_speech.wav

# Common Voice dataset samples
# https://commonvoice.mozilla.org/
```

### Method 4: Text-to-Speech

```bash
# Using espeak-ng (install first)
sudo apt-get install espeak-ng

# Generate speech
espeak-ng "Hello, this is a test of the Whisper transcription system." \
  -w test_tts.wav --rate=150

# Convert to 16kHz mono
ffmpeg -i test_tts.wav -ar 16000 -ac 1 test_tts_16k.wav
```

### Method 5: Use System Sounds

```bash
# Find system audio files
find /usr/share/sounds -name "*.wav" | head -5

# Convert to proper format
ffmpeg -i /usr/share/sounds/some_sound.wav \
  -ar 16000 -ac 1 test_system.wav
```

---

## Audio Format Requirements

Whisper accepts:
- **Sample Rate**: 16000 Hz (preferred) or will auto-convert
- **Channels**: Mono (1 channel) preferred
- **Format**: WAV, MP3, M4A, FLAC, OGG
- **Duration**: Any length (will chunk if > 30 seconds)

### Convert Any Audio

```bash
# MP3 to WAV 16kHz mono
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# M4A to WAV
ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav

# Video to audio
ffmpeg -i video.mp4 -ar 16000 -ac 1 audio.wav
```

---

## Expected Outputs

### For Sine Wave Audio
```
Input: Pure 440 Hz sine wave (5 seconds)
Output: " [Music]"
Reason: Correctly identifies as non-speech
```

### For Formant Audio
```
Input: Speech-like formants (no phonemes)
Output: " [Music]" or "[Unintelligible]"
Reason: Structure similar to speech but no words
```

### For Real Speech
```
Input: "Hello, how are you today?"
Output: " Hello, how are you today?"
Tokens: ~8-10 tokens
```

### For Long Speech (60+ seconds)
```
Input: 60 seconds of conversation
Output: Full transcription with proper segments
Chunks: 2 (30s + 30s)
Total tokens: 100-300 depending on content
```

---

## Interpreting Results

### Good Signs ✅
- Text makes sense grammatically
- Punctuation is reasonable
- Speaker intent is clear
- WER < 10% (if you have ground truth)

### Warning Signs ⚠️
- Repeated phrases
- Nonsensical words
- Very short output for long audio
- All special tokens, no text

### Error Signs ❌
- "[Audio processed but no speech detected]" for clear speech
- Empty string output
- Crashes or exceptions
- Zero-dimension tensor errors

---

## Debugging Failed Transcriptions

### Check 1: Audio Quality
```bash
# Play the audio
ffplay test.wav

# Check format
ffprobe test.wav
```

Expected:
- Clear speech audible
- Sample rate: 16000 Hz
- Duration > 0.5 seconds

### Check 2: Mel Spectrogram
```python
import librosa
import matplotlib.pyplot as plt

audio, sr = librosa.load('test.wav', sr=16000)
mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
plt.imshow(librosa.power_to_db(mel), aspect='auto')
plt.show()
```

Should see:
- Clear patterns in spectrogram
- Not all zeros or all noise
- Frequency bands visible

### Check 3: Encoder Output
```python
decoder = ONNXWhisperNPU()
decoder.initialize()

# Check encoder output shape
mel_features = decoder.extract_mel_features(audio, 16000)
print(f"Mel shape: {mel_features.shape}")
# Expected: (80, time_steps)
```

### Check 4: Token Generation
```bash
# Run with debug logging
python3 test_kv_cache_fix.py 2>&1 | grep "Step\|token_id\|Decoded"
```

Look for:
- Multiple tokens generated (not just 1-2)
- Token IDs < 50257 (actual words)
- Reasonable decoded text

---

## Performance Benchmarks

### Target Performance

| Audio Length | Processing Time | RTF | Tokens |
|--------------|-----------------|-----|--------|
| 5s speech | 0.4s | 12x | 10-15 |
| 30s speech | 1.8s | 16x | 60-100 |
| 60s speech | 3.6s | 16x | 120-200 |

RTF = Real-time Factor (higher is better)

### Current Performance (Post-Fix)
- **Short audio**: 12.5x realtime ✅
- **Long audio**: 16.7x realtime ✅
- **Target (with NPU)**: 220x realtime 🎯

---

## Word Error Rate (WER) Testing

If you have ground truth transcription:

```python
from jiwer import wer

reference = "hello how are you doing today"
hypothesis = result['text'].lower().strip()

error_rate = wer(reference, hypothesis)
print(f"WER: {error_rate * 100:.2f}%")
```

**Good WER**: < 5%
**Acceptable WER**: 5-15%
**Needs improvement**: > 15%

---

## Common Issues

### Issue: "[Music]" for Speech
**Cause**: Audio is very unclear or too noisy
**Fix**:
- Check audio volume (should be -20dB to -3dB)
- Reduce background noise
- Ensure sample rate is 16kHz

### Issue: Very Short Output
**Cause**: Early end-of-text token generation
**Fix**:
- Check encoder output shape
- Verify KV cache accumulation
- May be intentional for short/silent audio

### Issue: Garbled Output
**Cause**: Token decoding issue
**Fix**:
- Verify `transformers` library installed
- Check tokenizer loading
- See debug output for actual token IDs

### Issue: Slow Performance
**Cause**: CPU-only inference
**Fix**:
- Verify OpenVINO is being used (check logs)
- Consider INT8 quantization
- Use NPU custom kernels (future)

---

## Next Steps After Validation

1. **Measure WER** on standard benchmarks (LibriSpeech)
2. **Optimize performance** with NPU kernels
3. **Add language detection** (currently English-only)
4. **Implement beam search** (currently greedy)
5. **Add temperature/top-p sampling** for better quality

---

## Contact & Support

**Location**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/`
**Documentation**: See `DECODER_TOKEN_GENERATION_FIX_COMPLETE.md`
**Test Scripts**:
- `test_kv_cache_fix.py`
- `test_long_audio.py`
- `test_speech_like_audio.py`

---

**Updated**: November 3, 2025
**Status**: Ready for real audio testing
