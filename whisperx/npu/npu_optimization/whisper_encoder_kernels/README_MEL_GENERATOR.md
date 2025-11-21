# Whisper Encoder Test Mel Spectrogram Generator

## Overview

This tool generates realistic test mel spectrogram data for Whisper encoder kernel testing. It supports both synthetic mel spectrogram generation and processing of real audio files.

## Generated Files

### Synthetic Test Data

1. **test_mel_synthetic.npy** (1.83 MB)
   - Shape: (80, 3000)
   - Format: Numpy array (unbatched)
   - Contains: Synthetic mel spectrogram with realistic frequency structure and temporal patterns
   - Value range: [-0.7182, 5.0986]
   - Statistics: Mean=0.0, Std=1.0 (normalized)

2. **test_mel_batched.npy** (1.83 MB)
   - Shape: (1, 80, 3000)
   - Format: Numpy array (batched)
   - Contains: Same synthetic mel spectrogram with batch dimension added
   - Ready for use with batch processing pipelines

### Real Audio Test Data

Generated from available audio files (in .npz format with metadata):

- **test_mel_jfk.npz** (261 KB)
- **test_mel_test_audio_jfk.npz** (261 KB)

Each .npz file contains:
- `mel_spec`: Actual mel spectrogram from real audio
- `metadata`: Dictionary with audio properties (sample rate, duration, etc.)

**Real audio statistics:**
- Shape: (80, 3000)
- Format: float32 (librosa format)
- Value range: [-80.0, 0.0] (dB scale)
- Mean: -70.2, Std: 16.9

## Usage Examples

### Loading Synthetic Test Data

```python
import numpy as np

# Load unbatched synthetic data
mel_spec = np.load('test_data/test_mel_synthetic.npy')
print(mel_spec.shape)  # (80, 3000)

# Load batched synthetic data
mel_spec_batched = np.load('test_data/test_mel_batched.npy')
print(mel_spec_batched.shape)  # (1, 80, 3000)
```

### Loading Real Audio Data

```python
import numpy as np

# Load real audio mel spectrogram
data = np.load('test_data/test_mel_jfk.npz', allow_pickle=True)
mel_spec = data['mel_spec']
metadata = data['metadata'].item()

print(f"Shape: {mel_spec.shape}")
print(f"Audio: {metadata['audio_name']}")
print(f"Duration: {metadata['audio_duration_sec']:.2f} seconds")
```

### Using with Whisper Encoder

```python
import numpy as np
from whisper.model import Whisper

# Load test data
mel_spec = np.load('test_data/test_mel_batched.npy')

# Create model
model = Whisper.from_pretrained('base')

# Encode mel spectrogram
with torch.no_grad():
    encoder_output = model.encoder(mel_spec)
    print(encoder_output.shape)
```

## Mel Spectrogram Specifications

### Whisper Base Encoder Requirements
- **Input shape**: (1, 80, 3000) with batch or (80, 3000) without
- **Mel bins**: 80 frequency bins
- **Time frames**: 3000 frames (30 seconds @ 100 Hz)
- **Sample rate**: 16 kHz
- **Hop length**: 160 samples (10 ms)
- **FFT size**: 400 samples (25 ms)

### Synthetic Data Generation Strategy

1. **Frequency Structure**
   - Formant-like frequency bands at relative positions
   - Simulates speech formants (F1, F2, F3, F4)
   - Uses Gaussian envelopes for smooth spectral shape

2. **Temporal Patterns**
   - Syllable-like amplitude modulation (~150 frame spacing)
   - Attack/decay envelopes simulating speech dynamics
   - Frequency modulation (pitch variation)
   - Realistic speech-like temporal structure

3. **Normalization**
   - Log scale conversion
   - Standardization (mean=0, std=1)
   - Clipping to [-10, 10] range

### Real Audio Processing

When loading real audio with librosa:
- Resampled to 16 kHz
- Computed mel spectrogram with standard Whisper parameters
- Converted to dB scale: [-80, 0] dB
- Padded or truncated to 3000 frames

## Script Details

### File Location
```
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/
whisper_encoder_kernels/test_mel_generator.py
```

### Output Directory
```
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/
whisper_encoder_kernels/test_data/
```

### Key Functions

1. **generate_synthetic_mel_spectrogram()**
   - Creates realistic synthetic mel spectrogram
   - Options for frequency structure and temporal patterns
   - Parameters: n_mels, n_frames, seed

2. **load_and_compute_mel_spectrogram()**
   - Loads audio file and computes mel spectrogram
   - Uses librosa for mel computation
   - Handles padding/truncation to target frames

3. **find_test_audio_files()**
   - Searches project for available audio files
   - Excludes virtual environments
   - Returns sorted list of audio paths

4. **print_mel_statistics()**
   - Displays comprehensive statistics
   - Shows shape, range, mean, std, percentiles

### Dependencies

- **Required**: numpy
- **Optional**: librosa (for real audio processing)

Install librosa:
```bash
pip install librosa
```

## Running the Generator

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels
python3 test_mel_generator.py
```

### Expected Output

```
============================================================
Whisper Encoder Test Mel Spectrogram Generator
============================================================

[1] Generating synthetic mel spectrogram...
[2] Searching for test audio files...
[3] Processing real audio files with librosa...
[4] Creating batched test data...

SUMMARY
============================================================
Synthetic mel shape:     (80, 3000)
Batched mel shape:       (1, 80, 3000)
Expected Whisper shape:  (1, 80, 3000) or (80, 3000)
============================================================
```

## Test Audio Files Found

The generator found the following test audio files in the project:

1. `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisper-cpp-igpu/bindings/go/samples/jfk.wav`
2. `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisper-cpp-igpu/samples/jfk.mp3`
3. `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/test_audio_jfk.wav`
4. `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/test_speech.wav`

All are JFK audio clips suitable for Whisper encoder testing.

## Performance Characteristics

### Memory Requirements
- Synthetic mel (80, 3000): 1.83 MB (float64)
- Batched mel (1, 80, 3000): 1.83 MB (float64)
- Real mel with metadata: 261 KB (float32, compressed)

### Computation Time
- Synthetic generation: <100ms
- Real audio processing: 1-5 seconds per file (includes librosa loading)
- Total script execution: ~10 seconds

## Integration with Testing

### Use Cases
1. **Unit Tests**: Verify encoder kernel functionality
2. **Regression Tests**: Ensure output correctness with known inputs
3. **Performance Benchmarks**: Measure kernel execution time
4. **Integration Tests**: Test encoder with realistic data
5. **CI/CD Pipelines**: Automated testing with generated data

### Example Test Integration

```python
import numpy as np
import unittest

class TestWhisperEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.synthetic_mel = np.load('test_data/test_mel_synthetic.npy')
        cls.batched_mel = np.load('test_data/test_mel_batched.npy')

    def test_encoder_output_shape(self):
        # Test with unbatched data
        output = self.encoder(self.synthetic_mel)
        self.assertEqual(output.shape[1:], (768,))  # Whisper base encoder output

    def test_encoder_batch_processing(self):
        # Test with batched data
        output = self.encoder(self.batched_mel)
        self.assertEqual(output.shape, (1, 1500, 768))
```

## Troubleshooting

### Librosa Import Error
```
ImportError: No module named 'librosa'
```
Solution: Install librosa
```bash
pip install librosa
```

### Audio File Not Found
The script continues even if audio files are not found. It will only generate synthetic data.

### Memory Issues
For very large batch processing, consider:
1. Loading data in chunks
2. Using float32 instead of float64
3. Reducing batch size

## Future Enhancements

Possible improvements:
- Multi-speaker test data
- Various noise levels (SNR)
- Language-specific audio (multilingual testing)
- Custom audio duration support
- Spectrogram augmentation (pitch shift, time stretch)
- Comparison plots (synthetic vs real)
