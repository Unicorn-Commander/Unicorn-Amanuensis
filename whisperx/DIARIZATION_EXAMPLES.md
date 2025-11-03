# Speaker Diarization - Usage Examples

## Real-World Examples

---

## Example 1: Two-Person Interview

### Setup
```bash
export HF_TOKEN='hf_your_token_here'
python3 server_dynamic.py
```

### API Call
```bash
curl -X POST \
  -F "file=@interview.wav" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=2" \
  http://localhost:9004/transcribe
```

### Response
```json
{
  "text": "Welcome to the show. Thanks for having me. So tell me about your new book. Well, it's a story about...",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Welcome to the show.",
      "speaker": "SPEAKER_00"
    },
    {
      "start": 2.5,
      "end": 4.0,
      "text": "Thanks for having me.",
      "speaker": "SPEAKER_01"
    },
    {
      "start": 4.0,
      "end": 7.0,
      "text": "So tell me about your new book.",
      "speaker": "SPEAKER_00"
    },
    {
      "start": 7.0,
      "end": 11.5,
      "text": "Well, it's a story about...",
      "speaker": "SPEAKER_01"
    }
  ],
  "speakers": {
    "count": 2,
    "labels": ["SPEAKER_00", "SPEAKER_01"]
  },
  "duration": 11.5,
  "processing_time": 4.2,
  "realtime_factor": "2.7x",
  "diarization_enabled": true,
  "diarization_available": true
}
```

### Pretty Print
```python
import requests

response = requests.post(
    "http://localhost:9004/transcribe",
    files={"file": open("interview.wav", "rb")},
    data={"enable_diarization": "true", "min_speakers": 2, "max_speakers": 2}
)

result = response.json()

print("\n=== INTERVIEW TRANSCRIPT ===\n")
for segment in result["segments"]:
    speaker = "Interviewer" if segment["speaker"] == "SPEAKER_00" else "Guest"
    print(f"{speaker}: {segment['text']}")
```

**Output**:
```
=== INTERVIEW TRANSCRIPT ===

Interviewer: Welcome to the show.
Guest: Thanks for having me.
Interviewer: So tell me about your new book.
Guest: Well, it's a story about...
```

---

## Example 2: Team Meeting (4 Speakers)

### API Call
```bash
curl -X POST \
  -F "file=@team_meeting.wav" \
  -F "enable_diarization=true" \
  -F "min_speakers=3" \
  -F "max_speakers=5" \
  -F "model=base" \
  http://localhost:9004/transcribe | jq -r '.segments[] | "[" + .speaker + "] " + .text'
```

### Output
```
[SPEAKER_00] Good morning everyone. Let's start the standup.
[SPEAKER_01] I finished the authentication module yesterday.
[SPEAKER_02] Great! I'm working on the database schema today.
[SPEAKER_03] I'll review the pull requests this afternoon.
[SPEAKER_00] Perfect. Any blockers?
[SPEAKER_01] No blockers on my end.
[SPEAKER_02] All good here.
[SPEAKER_03] Same here.
[SPEAKER_00] Excellent. Let's wrap up.
```

### Speaker Summary
```python
from collections import defaultdict
import requests

response = requests.post(
    "http://localhost:9004/transcribe",
    files={"file": open("meeting.wav", "rb")},
    data={"enable_diarization": "true", "min_speakers": 3, "max_speakers": 5}
)

result = response.json()

# Group by speaker
by_speaker = defaultdict(list)
for segment in result["segments"]:
    speaker = segment.get("speaker", "Unknown")
    by_speaker[speaker].append(segment["text"])

# Print summary
print("\n=== SPEAKER SUMMARY ===\n")
for speaker in sorted(by_speaker.keys()):
    texts = by_speaker[speaker]
    total_words = sum(len(text.split()) for text in texts)
    print(f"{speaker}:")
    print(f"  Segments: {len(texts)}")
    print(f"  Words: {total_words}")
    print(f"  Sample: {texts[0][:50]}...")
    print()
```

**Output**:
```
=== SPEAKER SUMMARY ===

SPEAKER_00:
  Segments: 3
  Words: 25
  Sample: Good morning everyone. Let's start the standup...

SPEAKER_01:
  Segments: 2
  Words: 18
  Sample: I finished the authentication module yesterday...

SPEAKER_02:
  Segments: 2
  Words: 16
  Sample: Great! I'm working on the database schema tod...

SPEAKER_03:
  Segments: 2
  Words: 14
  Sample: I'll review the pull requests this afternoon...
```

---

## Example 3: Phone Call Recording

### API Call
```bash
curl -X POST \
  -F "file=@call_recording.m4a" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=2" \
  -F "vad_filter=true" \
  http://localhost:9004/transcribe
```

### Response Processing
```python
import requests
import json

# Make request
with open("call_recording.m4a", "rb") as f:
    response = requests.post(
        "http://localhost:9004/transcribe",
        files={"file": f},
        data={
            "enable_diarization": "true",
            "min_speakers": 2,
            "max_speakers": 2,
            "vad_filter": "true"
        }
    )

result = response.json()

# Format as conversation
print("\n=== CALL TRANSCRIPT ===\n")
print(f"Duration: {result['duration']:.1f}s")
print(f"Processing time: {result['processing_time']:.1f}s")
print(f"Speakers: {result['speakers']['count']}")
print("\n--- Conversation ---\n")

for segment in result["segments"]:
    timestamp = f"[{segment['start']:.1f}s]"
    speaker = "Caller" if segment["speaker"] == "SPEAKER_00" else "Recipient"
    text = segment["text"]
    print(f"{timestamp} {speaker}: {text}")
```

**Output**:
```
=== CALL TRANSCRIPT ===

Duration: 45.2s
Processing time: 18.5s
Speakers: 2

--- Conversation ---

[0.0s] Caller: Hello, this is John from tech support.
[3.5s] Recipient: Hi John, I'm having trouble with my laptop.
[7.2s] Caller: I'd be happy to help. What seems to be the problem?
[11.8s] Recipient: It won't connect to the Wi-Fi network.
[15.5s] Caller: Okay, let's try a few things. First, can you...
```

---

## Example 4: Podcast (3 Hosts)

### API Call
```bash
curl -X POST \
  -F "file=@podcast_episode.mp3" \
  -F "enable_diarization=true" \
  -F "min_speakers=3" \
  -F "max_speakers=3" \
  http://localhost:9004/transcribe
```

### Export as Subtitles (SRT)
```python
import requests

response = requests.post(
    "http://localhost:9004/transcribe",
    files={"file": open("podcast.mp3", "rb")},
    data={"enable_diarization": "true", "min_speakers": 3, "max_speakers": 3}
)

result = response.json()

# Generate SRT with speaker names
speaker_names = {
    "SPEAKER_00": "Host",
    "SPEAKER_01": "Guest 1",
    "SPEAKER_02": "Guest 2"
}

with open("podcast.srt", "w") as f:
    for i, segment in enumerate(result["segments"], 1):
        speaker = speaker_names.get(segment.get("speaker", ""), "Unknown")
        start = segment["start"]
        end = segment["end"]
        text = f"{speaker}: {segment['text']}"

        # SRT format
        f.write(f"{i}\n")
        f.write(f"{format_time(start)} --> {format_time(end)}\n")
        f.write(f"{text}\n\n")

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

print("Subtitles saved to podcast.srt")
```

**Output File** (`podcast.srt`):
```
1
00:00:00,000 --> 00:00:03,500
Host: Welcome to the Tech Talk podcast!

2
00:00:03,500 --> 00:00:07,200
Guest 1: Thanks for having me on the show.

3
00:00:07,200 --> 00:00:11,800
Guest 2: Great to be here as well.

4
00:00:11,800 --> 00:00:16,500
Host: Today we're discussing artificial intelligence.
```

---

## Example 5: Batch Processing

### Process Multiple Files
```bash
#!/bin/bash

# Process all audio files in directory
for file in audio_files/*.wav; do
    echo "Processing: $file"

    curl -X POST \
        -F "file=@$file" \
        -F "enable_diarization=true" \
        -F "min_speakers=2" \
        -F "max_speakers=6" \
        http://localhost:9004/transcribe \
        > "transcripts/$(basename "$file" .wav).json"

    echo "Saved to: transcripts/$(basename "$file" .wav).json"
done
```

### Python Batch Processing
```python
import os
import requests
from pathlib import Path

def process_directory(input_dir, output_dir, min_speakers=2, max_speakers=6):
    """Process all audio files in directory with diarization"""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    audio_files = list(input_path.glob("*.wav")) + \
                  list(input_path.glob("*.mp3")) + \
                  list(input_path.glob("*.m4a"))

    for audio_file in audio_files:
        print(f"Processing: {audio_file.name}")

        with open(audio_file, "rb") as f:
            response = requests.post(
                "http://localhost:9004/transcribe",
                files={"file": f},
                data={
                    "enable_diarization": "true",
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers
                }
            )

        if response.status_code == 200:
            output_file = output_path / f"{audio_file.stem}.json"
            with open(output_file, "w") as f:
                json.dump(response.json(), f, indent=2)
            print(f"  ✓ Saved to: {output_file}")
        else:
            print(f"  ✗ Error: {response.status_code}")

# Run
process_directory("meetings/", "transcripts/")
```

---

## Example 6: Web GUI Integration

### HTML Form
```html
<!DOCTYPE html>
<html>
<head>
    <title>Transcription with Diarization</title>
</head>
<body>
    <h1>Upload Audio for Transcription</h1>

    <form id="upload-form">
        <input type="file" name="file" accept="audio/*" required>
        <br><br>

        <label>
            <input type="checkbox" name="enable_diarization" value="true" checked>
            Enable Speaker Diarization
        </label>
        <br><br>

        <label>Min Speakers:
            <input type="number" name="min_speakers" value="2" min="1" max="10">
        </label>
        <br>

        <label>Max Speakers:
            <input type="number" name="max_speakers" value="4" min="1" max="10">
        </label>
        <br><br>

        <button type="submit">Transcribe</button>
    </form>

    <div id="result"></div>

    <script>
        document.getElementById('upload-form').addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData(e.target);

            document.getElementById('result').innerHTML = 'Processing...';

            const response = await fetch('http://localhost:9004/transcribe', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            // Display results
            let html = '<h2>Transcript</h2>';

            if (result.speakers) {
                html += `<p>Speakers: ${result.speakers.count}</p>`;
            }

            html += '<div>';
            result.segments.forEach(segment => {
                const speaker = segment.speaker || 'Unknown';
                html += `<p><strong>[${speaker}]</strong> ${segment.text}</p>`;
            });
            html += '</div>';

            document.getElementById('result').innerHTML = html;
        });
    </script>
</body>
</html>
```

---

## Example 7: Real-time Monitoring

### Stream Progress
```python
import requests
import time
import json

# Upload file
with open("long_meeting.wav", "rb") as f:
    # Start transcription (returns job_id)
    response = requests.post(
        "http://localhost:9004/transcribe",
        files={"file": f},
        data={"enable_diarization": "true"}
    )

    job_id = response.json().get("job_id")
    print(f"Job started: {job_id}")

# Monitor progress
while True:
    progress = requests.get(f"http://localhost:9004/progress/{job_id}")
    data = progress.json()

    status = data.get("status")
    message = data.get("message", "")
    percent = data.get("progress", 0)

    print(f"[{percent}%] {message}")

    if status in ["completed", "error"]:
        break

    time.sleep(1)

if status == "completed":
    # Get final result
    result = requests.get(f"http://localhost:9004/transcribe/{job_id}")
    print("\nTranscription complete!")
    print(json.dumps(result.json(), indent=2))
```

**Output**:
```
Job started: a1b2c3d4
[10%] Preprocessing audio...
[30%] Transcribing with base model...
[50%] Transcribing audio... 50%
[70%] Transcribing audio... 70%
[95%] Processing results...
[96%] Running speaker diarization...
[98%] Finalizing results...
[100%] Transcription complete

Transcription complete!
{
  "segments": [...],
  "speakers": {"count": 3, "labels": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]}
}
```

---

## Example 8: Export Formats

### Export as Text
```python
def export_as_text(result, output_file):
    """Export diarized transcript as plain text"""
    with open(output_file, "w") as f:
        for segment in result["segments"]:
            speaker = segment.get("speaker", "Unknown")
            text = segment["text"]
            f.write(f"{speaker}: {text}\n")

export_as_text(result, "transcript.txt")
```

### Export as CSV
```python
import csv

def export_as_csv(result, output_file):
    """Export diarized transcript as CSV"""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Speaker", "Start", "End", "Text"])

        for segment in result["segments"]:
            writer.writerow([
                segment.get("speaker", "Unknown"),
                segment["start"],
                segment["end"],
                segment["text"]
            ])

export_as_csv(result, "transcript.csv")
```

### Export as JSON
```python
import json

def export_as_json(result, output_file):
    """Export full result as JSON"""
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

export_as_json(result, "transcript.json")
```

---

## Quick Reference

### Basic Usage
```bash
curl -X POST \
  -F "file=@audio.wav" \
  -F "enable_diarization=true" \
  http://localhost:9004/transcribe
```

### With Speaker Range
```bash
curl -X POST \
  -F "file=@audio.wav" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4" \
  http://localhost:9004/transcribe
```

### Full Options
```bash
curl -X POST \
  -F "file=@audio.wav" \
  -F "model=base" \
  -F "language=en" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=6" \
  -F "vad_filter=true" \
  http://localhost:9004/transcribe
```

### Check Status
```bash
curl http://localhost:9004/status | jq '.diarization'
```

---

## Tips for Best Results

1. **Specify Speaker Count**: Narrow the range for better accuracy
2. **Use Quality Audio**: Clear recording = better speaker detection
3. **Enable VAD**: Removes silence for better diarization
4. **Test First**: Try 30-second clips before long files
5. **Export Results**: Save JSON for later processing

---

## More Examples

See also:
- `test_diarization.py` - Test script with examples
- `DIARIZATION_QUICK_START.md` - Setup guide
- `DIARIZATION_IMPLEMENTATION_COMPLETE.md` - Full documentation
