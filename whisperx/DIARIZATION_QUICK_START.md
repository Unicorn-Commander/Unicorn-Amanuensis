# Speaker Diarization - Quick Start Guide

## What You Get

When you enable diarization, you'll see **which speaker said what**:

```json
{
  "segments": [
    {"start": 0.0, "end": 2.0, "text": "Hello, how are you?", "speaker": "SPEAKER_00"},
    {"start": 2.0, "end": 4.5, "text": "I'm doing great!", "speaker": "SPEAKER_01"},
    {"start": 4.5, "end": 6.0, "text": "That's wonderful!", "speaker": "SPEAKER_00"}
  ],
  "speakers": {
    "count": 2,
    "labels": ["SPEAKER_00", "SPEAKER_01"]
  }
}
```

---

## Quick Enable (3 Minutes)

### Step 1: Accept License
Visit: https://huggingface.co/pyannote/speaker-diarization-3.1

Click **"Access repository"** and accept the terms.

### Step 2: Get Token
Visit: https://huggingface.co/settings/tokens

Create a **Read** token and copy it.

### Step 3: Set Environment
```bash
export HF_TOKEN='hf_your_token_here'
```

### Step 4: Restart Server
```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
python3 server_dynamic.py
```

### Step 5: Test It
```bash
curl -X POST \
  -F "file=@audio.wav" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4" \
  http://localhost:9004/transcribe
```

---

## API Parameters

### Enable Diarization
```bash
-F "enable_diarization=true"
```

### Specify Speaker Range
```bash
-F "min_speakers=2"    # Minimum 2 speakers
-F "max_speakers=5"    # Maximum 5 speakers
```

### Full Example
```bash
curl -X POST \
  -F "file=@meeting.wav" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4" \
  -F "model=base" \
  -F "vad_filter=true" \
  http://localhost:9004/transcribe | jq '.segments[] | {speaker, text}'
```

**Output**:
```json
{"speaker": "SPEAKER_00", "text": "Welcome to the meeting"}
{"speaker": "SPEAKER_01", "text": "Thanks for having me"}
{"speaker": "SPEAKER_00", "text": "Let's get started"}
```

---

## Response Format

### Without Diarization (Default)
```json
{
  "text": "Full transcription text",
  "segments": [
    {"start": 0.0, "end": 2.0, "text": "Segment text"}
  ],
  "diarization_enabled": false
}
```

### With Diarization
```json
{
  "text": "Full transcription text",
  "segments": [
    {"start": 0.0, "end": 2.0, "text": "Segment text", "speaker": "SPEAKER_00"}
  ],
  "speakers": {
    "count": 2,
    "labels": ["SPEAKER_00", "SPEAKER_01"]
  },
  "diarization_enabled": true,
  "diarization_available": true
}
```

---

## Check Status

```bash
curl http://localhost:9004/status | jq '.diarization'
```

**If Available**:
```json
{
  "available": true,
  "model": "pyannote/speaker-diarization-3.1",
  "note": "Speaker diarization ready"
}
```

**If Not Available**:
```json
{
  "available": false,
  "model": null,
  "note": "Diarization not available. Set HF_TOKEN and accept model license."
}
```

---

## Troubleshooting

### "Diarization not available"

**Check**:
1. Did you accept the license? https://huggingface.co/pyannote/speaker-diarization-3.1
2. Did you set HF_TOKEN? `echo $HF_TOKEN`
3. Did you restart the server?

**Fix**:
```bash
# Set token
export HF_TOKEN='hf_your_token_here'

# Restart server
pkill -f server_dynamic.py
python3 server_dynamic.py
```

### "OSError: libtorch_cuda.so"

This is expected - pyannote has CUDA dependencies but will still work on CPU.

**Fix** (optional, for cleaner logs):
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pyannote.audio
```

### Slow Processing

Diarization adds ~40-60% overhead.

**Example**:
- Without diarization: 3 seconds (20x realtime)
- With diarization: 5 seconds (12x realtime)

**Tip**: Use diarization only when you need speaker labels.

---

## Tips for Best Results

### 1. Specify Speaker Count
```bash
# If you know there are 2-3 speakers
-F "min_speakers=2"
-F "max_speakers=3"
```

Better speaker separation when you narrow the range.

### 2. Use Good Quality Audio
- Clear audio = better speaker detection
- Minimize background noise
- Use separate microphones if possible

### 3. Enable VAD
```bash
-F "vad_filter=true"  # Default, removes silence
```

Helps diarization focus on actual speech.

### 4. Test with Small Files First
```bash
# Test with 30 second clip first
ffmpeg -i long_audio.wav -t 30 test.wav
curl -X POST -F "file=@test.wav" -F "enable_diarization=true" ...
```

---

## GUI Integration

The GUI automatically supports diarization if you enable it in the checkbox:

1. Open: http://localhost:9004/web
2. Upload audio file
3. Check "Enable Diarization"
4. Set speaker range (optional)
5. Click "Transcribe"

Results will show speaker labels automatically.

---

## Use Cases

### Meeting Transcription
```bash
curl -X POST \
  -F "file=@meeting.wav" \
  -F "enable_diarization=true" \
  -F "min_speakers=3" \
  -F "max_speakers=8" \
  http://localhost:9004/transcribe
```

### Interview
```bash
curl -X POST \
  -F "file=@interview.wav" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=2" \
  http://localhost:9004/transcribe
```

### Podcast
```bash
curl -X POST \
  -F "file=@podcast.mp3" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4" \
  http://localhost:9004/transcribe
```

### Call Recording
```bash
curl -X POST \
  -F "file=@call.m4a" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=2" \
  http://localhost:9004/transcribe
```

---

## Advanced: Python Client

```python
import requests

# Upload file with diarization
with open("meeting.wav", "rb") as f:
    response = requests.post(
        "http://localhost:9004/transcribe",
        files={"file": f},
        data={
            "enable_diarization": "true",
            "min_speakers": 2,
            "max_speakers": 5
        }
    )

result = response.json()

# Print by speaker
for segment in result["segments"]:
    speaker = segment.get("speaker", "Unknown")
    text = segment["text"]
    print(f"[{speaker}]: {text}")
```

**Output**:
```
[SPEAKER_00]: Welcome everyone to the meeting
[SPEAKER_01]: Thanks for having me
[SPEAKER_02]: Great to be here
[SPEAKER_00]: Let's get started
```

---

## Performance Tips

### 1. Disable When Not Needed
```bash
# Default: diarization OFF
-F "enable_diarization=false"  # Or omit parameter
```

### 2. Use Smaller Model for Speed
```bash
-F "model=base"  # Fast, good quality
# vs
-F "model=large-v3"  # Slow, best quality
```

### 3. Process in Batches
```bash
# Process multiple files in parallel
for file in *.wav; do
  curl -X POST -F "file=@$file" -F "enable_diarization=true" \
    http://localhost:9004/transcribe > "${file%.wav}.json" &
done
wait
```

---

## That's It!

You now have speaker diarization working in `server_dynamic.py`!

**Quick recap**:
1. ✅ Accept license
2. ✅ Get HF_TOKEN
3. ✅ Set environment variable
4. ✅ Restart server
5. ✅ Add `enable_diarization=true` to requests

**Results**: 🎭 Speaker labels on every segment!

---

## Support

**Issues?** Check:
- Status endpoint: `curl http://localhost:9004/status`
- Server logs: Look for "Speaker diarization" messages
- Documentation: `DIARIZATION_IMPLEMENTATION_COMPLETE.md`

**Questions?** The implementation follows pyannote.audio best practices:
- https://github.com/pyannote/pyannote-audio
- https://huggingface.co/pyannote/speaker-diarization-3.1
