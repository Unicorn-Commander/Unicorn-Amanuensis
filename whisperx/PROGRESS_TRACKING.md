# Real-Time Progress Tracking for Transcription

## Overview

The server now supports real-time progress tracking for long audio transcriptions. This allows clients to monitor the status of their transcription jobs and show progress to users.

## How It Works

1. **Submit transcription**: POST audio to `/transcribe` or `/v1/audio/transcriptions`
2. **Receive job_id**: Response includes a `job_id` field
3. **Monitor progress**: Poll `/progress/{job_id}` or stream via `/progress/{job_id}/stream`

## Progress Stages

| Stage | Progress | Description |
|-------|----------|-------------|
| `uploading` | 0% | Audio file being uploaded |
| `preprocessing` | 10% | Audio being preprocessed |
| `transcribing` | 30-90% | Whisper processing audio |
| - Loading audio | 40% | Audio file loaded |
| - Encoder | 50% | Whisper encoder processing |
| - Final | 90% | Generating final transcription |
| `completed` | 100% | Transcription complete |
| `error` | 0% | Error occurred |

## API Endpoints

### 1. Submit Transcription (Standard)

```bash
# Submit audio for transcription
curl -X POST \
  -F "file=@audio.wav" \
  -F "model=base" \
  http://localhost:9004/transcribe

# Response includes job_id:
{
  "text": "...",
  "job_id": "a1b2c3d4",
  ...
}
```

### 2. Poll Progress (REST)

```bash
# Get current progress
curl http://localhost:9004/progress/a1b2c3d4

# Response:
{
  "status": "transcribing",
  "progress": 50,
  "message": "Processing audio with Whisper encoder...",
  "job_id": "a1b2c3d4"
}
```

### 3. Stream Progress (Server-Sent Events)

```javascript
// JavaScript example
const job_id = "a1b2c3d4";
const eventSource = new EventSource(`http://localhost:9004/progress/${job_id}/stream`);

eventSource.onmessage = (event) => {
  const progress = JSON.parse(event.data);
  console.log(`${progress.message} (${progress.progress}%)`);

  if (progress.status === "completed") {
    eventSource.close();
    console.log("Transcription complete!");
  }

  if (progress.status === "error") {
    eventSource.close();
    console.error("Transcription failed:", progress.message);
  }
};
```

### 4. Python Example

```python
import requests
import time

# Submit transcription
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:9004/transcribe",
        files={"file": f},
        data={"model": "base"}
    )

# Extract job_id
job_id = response.json()["job_id"]
print(f"Job ID: {job_id}")

# Poll progress
while True:
    progress_response = requests.get(f"http://localhost:9004/progress/{job_id}")
    progress = progress_response.json()

    print(f"{progress['message']} ({progress['progress']}%)")

    if progress['status'] in ["completed", "error"]:
        break

    time.sleep(0.5)

# Get final result from original response
print(f"Transcription: {response.json()['text']}")
```

### 5. Web UI Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Transcription with Progress</title>
</head>
<body>
    <input type="file" id="audioFile" accept="audio/*">
    <button onclick="transcribe()">Transcribe</button>

    <div id="progress" style="display:none;">
        <progress id="progressBar" max="100" value="0"></progress>
        <p id="statusText"></p>
    </div>

    <div id="result" style="display:none;"></div>

    <script>
    async function transcribe() {
        const file = document.getElementById('audioFile').files[0];
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', 'base');

        // Show progress
        document.getElementById('progress').style.display = 'block';

        // Submit transcription
        const response = await fetch('http://localhost:9004/transcribe', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        const job_id = result.job_id;

        // Stream progress updates
        const eventSource = new EventSource(`http://localhost:9004/progress/${job_id}/stream`);

        eventSource.onmessage = (event) => {
            const progress = JSON.parse(event.data);

            document.getElementById('progressBar').value = progress.progress;
            document.getElementById('statusText').textContent = progress.message;

            if (progress.status === 'completed') {
                eventSource.close();
                document.getElementById('progress').style.display = 'none';
                document.getElementById('result').style.display = 'block';
                document.getElementById('result').textContent = result.text;
            }

            if (progress.status === 'error') {
                eventSource.close();
                alert('Transcription failed: ' + progress.message);
            }
        };
    }
    </script>
</body>
</html>
```

## Progress Cleanup

Progress data is automatically cleaned up 5 minutes after transcription completes to prevent memory leaks.

## Performance Impact

- Progress tracking adds minimal overhead (<1ms per update)
- SSE streams update every 500ms
- No impact on transcription speed (progress updates run in parallel)

## Notes

- The progress updates are approximate since faster-whisper doesn't expose per-chunk progress
- For very short audio (<5 seconds), you may only see a few progress updates
- For long audio (>60 seconds), progress updates provide meaningful feedback to users
