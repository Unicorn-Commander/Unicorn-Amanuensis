# API Timing Specification

**Version**: 1.0.0
**Date**: November 2, 2025
**Status**: Production Ready

---

## Overview

This document specifies the timing data format returned by the Unicorn-Amanuensis transcription API when `include_timing=true` is specified. The timing data provides component-level performance breakdown for debugging and optimization.

---

## API Endpoint

### POST /v1/audio/transcriptions

**Standard Parameters**:
- `file` (UploadFile, required): Audio file to transcribe
- `diarize` (bool, optional): Enable speaker diarization
- `min_speakers` (int, optional): Minimum number of speakers
- `max_speakers` (int, optional): Maximum number of speakers

**Timing Parameter** (Week 19.6):
- `include_timing` (bool, optional, default: false): Include component timing breakdown

---

## Request Format

### Without Timing (Default)

```bash
curl -X POST http://localhost:9050/v1/audio/transcriptions \
  -F "file=@audio.wav"
```

### With Timing Breakdown

```bash
curl -X POST http://localhost:9050/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "include_timing=true"
```

---

## Response Format

### Standard Response (without timing)

```json
{
  "text": "Transcribed text content.",
  "segments": [
    {
      "start": 0.0,
      "end": 1.0,
      "text": "Transcribed text content."
    }
  ],
  "words": [],
  "language": "en",
  "performance": {
    "audio_duration_s": 1.0,
    "processing_time_s": 0.328,
    "realtime_factor": 3.05,
    "mode": "pipeline"
  }
}
```

### Response with Timing Breakdown

```json
{
  "text": "Transcribed text content.",
  "segments": [...],
  "words": [],
  "language": "en",
  "performance": {
    "audio_duration_s": 1.0,
    "processing_time_s": 0.328,
    "realtime_factor": 3.05,
    "mode": "pipeline"
  },
  "timing": {
    "stage1.total": {
      "mean": 150.2,
      "p50": 148.5,
      "p95": 155.0,
      "p99": 155.0,
      "min": 145.0,
      "max": 155.0,
      "count": 1
    },
    "stage1.audio_loading": {...},
    "stage1.mel_computation": {...},
    "stage2.total": {...},
    "stage2.conv1d_preprocessing": {...},
    "stage2.npu_encoder": {...},
    "stage3.total": {...},
    "stage3.decoder": {...},
    "stage3.alignment": {...},
    "stage3.postprocessing": {...}
  }
}
```

---

## Timing Object Specification

### Timing Field Structure

**Location**: Top-level `timing` field (only present when `include_timing=true`)

**Type**: Object (dictionary mapping component paths to statistics)

**Format**:
```typescript
{
  "timing": {
    [component_path: string]: ComponentTimingStats
  }
}
```

### ComponentTimingStats Object

**Type**: Object with statistical measures

**Fields**:
- `mean` (float): Mean time in milliseconds
- `p50` (float): Median (50th percentile) time in milliseconds
- `p95` (float): 95th percentile time in milliseconds
- `p99` (float): 99th percentile time in milliseconds
- `min` (float): Minimum time in milliseconds
- `max` (float): Maximum time in milliseconds
- `count` (int): Number of timing samples

**Example**:
```json
{
  "mean": 150.2,
  "p50": 148.5,
  "p95": 155.0,
  "p99": 155.0,
  "min": 145.0,
  "max": 155.0,
  "count": 1
}
```

**Notes**:
- All time values are in milliseconds
- For single requests, `count` will be 1 and all statistics will be equal
- For multiple requests, statistics accumulate across requests
- `p95` and `p99` require ≥20 and ≥100 samples respectively; otherwise equals `max`

---

## Component Hierarchy

### Component Path Format

**Format**: Dot-separated hierarchical path
**Example**: `stage1.audio_loading`

**Interpretation**:
- `stage1`: Top-level component (Stage 1)
- `stage1.audio_loading`: Substage within Stage 1

**Hierarchy Rules**:
- Parent timing includes all child timings
- Child timings are nested within parent context
- Components at same level are sequential (non-overlapping)

### Pipeline Components

#### Stage 1: Audio Loading + Mel Spectrogram

**Components**:
- `stage1.total`: Total time for Stage 1
  - `stage1.audio_loading`: Load audio from bytes + buffer acquisition
  - `stage1.mel_computation`: Mel spectrogram computation (zero-copy)

**Typical Times** (1s audio):
- `stage1.total`: 50-150ms
- `stage1.audio_loading`: 10-50ms
- `stage1.mel_computation`: 40-100ms

#### Stage 2: NPU Encoder

**Components**:
- `stage2.total`: Total time for Stage 2
  - `stage2.conv1d_preprocessing`: Conv1d preprocessing (mel 80→512)
  - `stage2.npu_encoder`: NPU encoder execution (6 layers)

**Typical Times** (1s audio):
- `stage2.total`: 15-30ms
- `stage2.conv1d_preprocessing`: 3-10ms
- `stage2.npu_encoder`: 10-20ms

#### Stage 3: Decoder + Alignment

**Components**:
- `stage3.total`: Total time for Stage 3
  - `stage3.decoder`: Decoder (CustomWhisper/faster-whisper/WhisperX)
  - `stage3.alignment`: WhisperX word-level alignment
  - `stage3.postprocessing`: Buffer release + text formatting

**Typical Times** (1s audio):
- `stage3.total`: 100-300ms
- `stage3.decoder`: 50-200ms
- `stage3.alignment`: 40-90ms
- `stage3.postprocessing`: 5-15ms

---

## Example Responses

### Example 1: Single Request (1s audio)

**Request**:
```bash
curl -X POST http://localhost:9050/v1/audio/transcriptions \
  -F "file=@audio_1s.wav" \
  -F "include_timing=true"
```

**Response**:
```json
{
  "text": " Ooh.",
  "segments": [
    {
      "start": 0.0,
      "end": 1.0,
      "text": " Ooh."
    }
  ],
  "words": [],
  "language": "en",
  "performance": {
    "audio_duration_s": 1.0,
    "processing_time_s": 0.328,
    "realtime_factor": 3.05,
    "mode": "pipeline"
  },
  "timing": {
    "stage1.total": {
      "mean": 150.2,
      "p50": 150.2,
      "p95": 150.2,
      "p99": 150.2,
      "min": 150.2,
      "max": 150.2,
      "count": 1
    },
    "stage1.audio_loading": {
      "mean": 50.1,
      "p50": 50.1,
      "p95": 50.1,
      "p99": 50.1,
      "min": 50.1,
      "max": 50.1,
      "count": 1
    },
    "stage1.mel_computation": {
      "mean": 100.1,
      "p50": 100.1,
      "p95": 100.1,
      "p99": 100.1,
      "min": 100.1,
      "max": 100.1,
      "count": 1
    },
    "stage2.total": {
      "mean": 20.5,
      "p50": 20.5,
      "p95": 20.5,
      "p99": 20.5,
      "min": 20.5,
      "max": 20.5,
      "count": 1
    },
    "stage2.conv1d_preprocessing": {
      "mean": 5.2,
      "p50": 5.2,
      "p95": 5.2,
      "p99": 5.2,
      "min": 5.2,
      "max": 5.2,
      "count": 1
    },
    "stage2.npu_encoder": {
      "mean": 15.3,
      "p50": 15.3,
      "p95": 15.3,
      "p99": 15.3,
      "min": 15.3,
      "max": 15.3,
      "count": 1
    },
    "stage3.total": {
      "mean": 157.8,
      "p50": 157.8,
      "p95": 157.8,
      "p99": 157.8,
      "min": 157.8,
      "max": 157.8,
      "count": 1
    },
    "stage3.decoder": {
      "mean": 100.0,
      "p50": 100.0,
      "p95": 100.0,
      "p99": 100.0,
      "min": 100.0,
      "max": 100.0,
      "count": 1
    },
    "stage3.alignment": {
      "mean": 50.0,
      "p50": 50.0,
      "p95": 50.0,
      "p99": 50.0,
      "min": 50.0,
      "max": 50.0,
      "count": 1
    },
    "stage3.postprocessing": {
      "mean": 7.8,
      "p50": 7.8,
      "p95": 7.8,
      "p99": 7.8,
      "min": 7.8,
      "max": 7.8,
      "count": 1
    }
  }
}
```

### Example 2: Multiple Requests (10 requests, statistics)

**Scenario**: 10 requests processed, timing accumulates

**Response** (last request):
```json
{
  "text": "Final transcription.",
  "segments": [...],
  "timing": {
    "stage1.total": {
      "mean": 152.3,
      "p50": 151.0,
      "p95": 160.5,
      "p99": 160.5,
      "min": 145.0,
      "max": 160.5,
      "count": 10
    },
    "stage1.audio_loading": {
      "mean": 52.1,
      "p50": 51.5,
      "p95": 55.0,
      "p99": 55.0,
      "min": 48.0,
      "max": 55.0,
      "count": 10
    },
    "stage1.mel_computation": {
      "mean": 100.2,
      "p50": 99.5,
      "p95": 105.5,
      "p99": 105.5,
      "min": 97.0,
      "max": 105.5,
      "count": 10
    }
    // ... other components with 10 samples
  }
}
```

**Interpretation**:
- `mean`: Average time across 10 requests
- `p50`: Median time (5th/6th request)
- `p95`: 95th percentile (not enough samples, equals max)
- `min`/`max`: Best/worst performance
- `count`: 10 samples collected

---

## Client Usage Examples

### Python Client

```python
import requests

# Make request with timing
response = requests.post(
    'http://localhost:9050/v1/audio/transcriptions',
    files={'file': open('audio.wav', 'rb')},
    data={'include_timing': 'true'}
)

result = response.json()

# Access timing data
if 'timing' in result:
    timing = result['timing']

    # Print stage totals
    print(f"Stage 1: {timing['stage1.total']['mean']:.1f}ms")
    print(f"Stage 2: {timing['stage2.total']['mean']:.1f}ms")
    print(f"Stage 3: {timing['stage3.total']['mean']:.1f}ms")

    # Find bottleneck
    stage_times = {
        'stage1': timing['stage1.total']['mean'],
        'stage2': timing['stage2.total']['mean'],
        'stage3': timing['stage3.total']['mean']
    }
    bottleneck = max(stage_times.items(), key=lambda x: x[1])
    print(f"Bottleneck: {bottleneck[0]} ({bottleneck[1]:.1f}ms)")
```

### JavaScript Client

```javascript
const formData = new FormData();
formData.append('file', audioFile);
formData.append('include_timing', 'true');

const response = await fetch('http://localhost:9050/v1/audio/transcriptions', {
  method: 'POST',
  body: formData
});

const result = await response.json();

// Access timing data
if (result.timing) {
  const timing = result.timing;

  // Calculate total time
  const totalTime =
    timing['stage1.total'].mean +
    timing['stage2.total'].mean +
    timing['stage3.total'].mean;

  console.log(`Total time: ${totalTime.toFixed(1)}ms`);

  // Breakdown by stage
  for (const [component, stats] of Object.entries(timing)) {
    if (component.endsWith('.total')) {
      console.log(`${component}: ${stats.mean.toFixed(1)}ms (${stats.count} samples)`);
    }
  }
}
```

### cURL + jq

```bash
# Request with timing
curl -X POST http://localhost:9050/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "include_timing=true" \
  | jq '.timing'

# Extract stage totals
curl -X POST http://localhost:9050/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "include_timing=true" \
  | jq '.timing | to_entries | map(select(.key | endswith(".total"))) | map({component: .key, mean: .value.mean})'

# Find bottleneck
curl -X POST http://localhost:9050/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "include_timing=true" \
  | jq '.timing | to_entries | map(select(.key | endswith(".total"))) | max_by(.value.mean)'
```

---

## Performance Considerations

### Overhead

**Timing System Overhead**:
- Per-measurement: ~1.56μs
- Per-request (10 components): ~0.016ms
- Percentage of request time: <0.1%

**Impact**:
- Negligible: <0.1% overhead on typical 60ms request
- Safe for production use
- Can be disabled by setting `include_timing=false` (default)

### Scalability

**Component Count**:
- Current: 10 components
- Maximum: 100+ supported
- Overhead: Linear with component count

**Sample Count**:
- Single request: 1 sample per component
- Multiple requests: Accumulates across requests
- Memory: 8 bytes per sample (float64)
- No automatic reset (manual reset via pipeline restart)

### Best Practices

**1. Only Enable When Needed**:
```bash
# Development/debugging: Enable timing
curl -F "include_timing=true" ...

# Production: Disable timing (default)
curl ...
```

**2. Monitor Overhead**:
- Check `timing` field size in response
- Large sample counts (>1000) may increase response time
- Consider periodic timing resets in long-running services

**3. Use for Debugging, Not Monitoring**:
- Timing is per-request, not aggregated
- Use external monitoring for production metrics
- Export timing to monitoring system via logging

---

## Version History

### v1.0.0 (November 2, 2025)

**Initial Release**:
- Component-level timing breakdown
- Hierarchical component structure
- Statistical aggregation (mean, p50, p95, p99)
- Optional timing via `include_timing` parameter
- 10 pipeline components instrumented

**Components**:
- Stage 1: audio_loading, mel_computation
- Stage 2: conv1d_preprocessing, npu_encoder
- Stage 3: decoder, alignment, postprocessing

**Statistics**:
- mean, p50, p95, p99, min, max, count

---

## Future Enhancements

### Planned Features (Week 20+)

**1. Timing Reset Endpoint** (Priority: P1):
```bash
POST /v1/admin/timing/reset
```
- Reset accumulated timing statistics
- Useful for long-running services

**2. Timing Export Endpoint** (Priority: P2):
```bash
GET /v1/admin/timing/export
```
- Export all timing data as JSON
- Include metadata (timestamp, sample count, etc.)

**3. Flamegraph Generation** (Priority: P2):
```bash
GET /v1/admin/timing/flamegraph
```
- Generate SVG flamegraph from timing data
- Visual performance analysis

**4. Real-time Monitoring** (Priority: P3):
```bash
WebSocket /v1/admin/timing/stream
```
- Stream timing updates via WebSocket
- Live performance monitoring

---

## Support & Contact

**Documentation**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/`
**Reports**: `WEEK19.6_TIMING_INSTRUMENTATION_REPORT.md`
**Tests**: `tests/test_component_timer.py`
**Implementation**: `xdna2/component_timer.py`

**Team**: Week 19.6 Team 2 (Performance Engineering)
**Status**: Production Ready
**Version**: 1.0.0

---

Built with precision timing by Magic Unicorn Unconventional Technology & Stuff Inc
