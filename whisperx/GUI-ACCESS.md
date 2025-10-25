# 🦄 Unicorn Amanuensis - GUI Access Guide

## Your Beautiful GUI is Ready! ✨

The NPU-enabled Amanuensis service includes a **professional web interface** with:

- 🦄 **Custom Unicorn Logo** (1024x1024 hi-res)
- 🎨 **Animated Gradient Background** (15s smooth animation)
- ✨ **Glassmorphism Effects** (modern blur effects)
- 🌓 **Dark/Light Mode Support**
- 📊 **Real-time Performance Charts**
- 🎵 **Audio Waveform Visualization** (WaveSurfer.js)
- 📁 **Drag & Drop Upload**
- ⚡ **Live Transcription Progress**

## Access URLs

Once the container is running, access the GUI at:

### Main Web Interface
```
http://localhost:9000/web
```

### Alternative Paths
```
http://localhost:9000/              # API documentation
http://localhost:9000/static/       # Static files
http://localhost:9000/status        # Server status JSON
```

## Available GUI Versions

The container includes multiple interface versions:

1. **`/static/index.html`** - Main production GUI
   - Full-featured interface
   - Real-time charts and waveforms
   - NPU performance metrics

2. **`/static/simple.html`** - Simplified interface
   - Clean, minimal design
   - Quick transcription only

3. **`/templates/index_pro.html`** - Professional suite
   - Advanced features
   - Batch processing
   - Export options

4. **`/templates/index_clean.html`** - Ultra-clean design
   - Minimalist aesthetic
   - Focus mode

## Features

### 1. Upload & Transcribe
- Drag & drop audio files
- Supports: WAV, MP3, M4A, FLAC, OGG
- Real-time upload progress

### 2. NPU Performance Display
Shows real-time metrics:
- ⚡ **Processing Speed**: e.g., "220x realtime"
- 🔥 **NPU Acceleration**: Active/Inactive indicator
- ⏱️ **Processing Time**: Actual vs audio duration
- 📊 **Throughput**: Tokens/second

### 3. Audio Visualization
- Waveform display using WaveSurfer.js
- Playback controls
- Timeline markers
- Click to seek

### 4. Results Display
- Formatted transcription text
- Word-level timestamps (if enabled)
- Speaker diarization labels
- Confidence scores
- Copy to clipboard

### 5. Settings Panel
- Model selection (base, small, medium, large, large-v3)
- Enable/disable diarization
- Word timestamps toggle
- Language selection

## Quick Start

1. **Start the service**:
   ```bash
   cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
   docker compose -f docker-compose-npu.yml up -d
   ```

2. **Open browser**:
   ```
   http://localhost:9000/web
   ```

3. **Upload audio**:
   - Drag file to drop zone
   - OR click "Choose File"
   - Watch NPU process it in real-time!

4. **View results**:
   - Transcription appears automatically
   - See NPU acceleration stats
   - Copy or download results

## Screenshots

The interface looks like this:

![Unicorn Amanuensis GUI](../Unicorn_Amanuensis_screenshot.png)

## Customization

### Change Default GUI
Edit `server_whisperx_npu.py` line 543:
```python
index_path = static_dir / "index.html"  # Change to "simple.html" for minimal UI
```

### Modify Logo
Replace `/app/static/unicorn-logo.png` with your own logo (1024x1024 recommended)

### Custom Branding
Edit the HTML files in `/app/templates/` or `/app/static/`

## API Endpoints (for developers)

The GUI uses these backend endpoints:

```bash
# Transcribe audio
POST /transcribe
Content-Type: multipart/form-data
Body: file=@audio.wav

# Get server status
GET /status

# Health check
GET /health

# List models
GET /models
```

## Troubleshooting

### GUI Not Loading
```bash
# Check container is running
docker ps | grep amanuensis-npu

# Check logs
docker logs amanuensis-npu

# Verify static files exist
docker exec amanuensis-npu ls -la /app/static/
```

### Logo Not Showing
```bash
# Verify logo file
docker exec amanuensis-npu ls -lh /app/static/unicorn-logo.png
# Should show: 691716 bytes (691KB)
```

### NPU Metrics Not Showing
- GUI shows NPU status from transcription results
- If NPU isn't working, metrics will show "CPU fallback"
- Check server logs for NPU initialization messages

## Performance

The GUI is lightweight:
- **Total size**: ~36KB HTML + 691KB logo
- **Load time**: <1s on localhost
- **Framework**: Vanilla JS + TailwindCSS (CDN)
- **Charts**: Chart.js (CDN)
- **Audio**: WaveSurfer.js (CDN)

No build step required - pure HTML/CSS/JS!

## Browser Compatibility

Tested and working on:
- ✅ Chrome/Chromium 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## Mobile Support

The interface is responsive and works on:
- 📱 **Phones**: Simplified layout
- 📱 **Tablets**: Full interface
- 🖥️ **Desktop**: Best experience

---

**Enjoy your beautiful NPU-powered transcription GUI!** 🦄✨
