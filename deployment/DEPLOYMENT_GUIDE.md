# Unicorn-Amanuensis Deployment Guide
## Week 6 Production Deployment

**Service**: Unicorn-Amanuensis XDNA2 C++ Backend
**Version**: 2.0
**Target Performance**: 400-500x realtime
**Date**: November 1, 2025

---

## Quick Start (Manual Deployment)

For immediate validation without systemd:

```bash
# 1. Navigate to service directory
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis

# 2. Activate environment
source ~/mlir-aie/ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh

# 3. Start service
python3 -m uvicorn api:app --host 0.0.0.0 --port 9050
```

The service will be available at `http://localhost:9050`

---

## Systemd Installation (Production)

For persistent, production-grade deployment:

### 1. Copy Service File

```bash
sudo cp deployment/unicorn-amanuensis.service /etc/systemd/system/
```

### 2. Reload Systemd

```bash
sudo systemctl daemon-reload
```

### 3. Enable Service (Start on Boot)

```bash
sudo systemctl enable unicorn-amanuensis
```

### 4. Start Service

```bash
sudo systemctl start unicorn-amanuensis
```

### 5. Check Status

```bash
sudo systemctl status unicorn-amanuensis
```

### 6. View Logs

```bash
# Recent logs
sudo journalctl -u unicorn-amanuensis -n 50

# Follow logs in real-time
sudo journalctl -u unicorn-amanuensis -f
```

---

## Health Checks

### Automated Health Check

```bash
./deployment/health_check.sh
```

### Manual Health Checks

```bash
# Basic connectivity
curl http://localhost:9050/

# Health endpoint
curl http://localhost:9050/health

# Service statistics
curl http://localhost:9050/stats
```

---

## Service Management

### Start Service

```bash
sudo systemctl start unicorn-amanuensis
```

### Stop Service

```bash
sudo systemctl stop unicorn-amanuensis
```

### Restart Service

```bash
sudo systemctl restart unicorn-amanuensis
```

### Check Status

```bash
sudo systemctl status unicorn-amanuensis
```

### Disable Service (Don't Start on Boot)

```bash
sudo systemctl disable unicorn-amanuensis
```

---

## Testing the Service

### Test with Sample Audio

```bash
# Prepare test audio (if you have one)
curl -X POST http://localhost:9050/v1/audio/transcriptions \
  -F "file=@test_audio.wav" \
  -F "model=whisper-1"
```

### Expected Response

```json
{
  "text": "Transcribed text here...",
  "segments": [...],
  "language": "en",
  "performance": {
    "audio_duration_s": 30.0,
    "processing_time_s": 0.065,
    "realtime_factor": 461.5,
    "encoder_time_ms": 15.2,
    "encoder_realtime_factor": 1973.7
  }
}
```

---

## Configuration

Production configuration is in `config/production.yaml`:

```yaml
service:
  name: "unicorn-amanuensis"
  port: 9050
  workers: 4

npu:
  platform: "XDNA2_CPP"
  device_id: 0
  fallback_enabled: true

performance:
  max_concurrent_requests: 10
  request_timeout_seconds: 30
```

To use this configuration, set environment variable:

```bash
export UNICORN_CONFIG=/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/config/production.yaml
```

---

## Troubleshooting

### Service Won't Start

1. Check logs:
   ```bash
   sudo journalctl -u unicorn-amanuensis -n 100
   ```

2. Verify environment:
   ```bash
   source ~/mlir-aie/ironenv/bin/activate
   source /opt/xilinx/xrt/setup.sh
   python3 -c "import fastapi, uvicorn, whisperx; print('OK')"
   ```

3. Test manually:
   ```bash
   cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis
   python3 -m uvicorn api:app --host 127.0.0.1 --port 9050
   ```

### Port Already in Use

Check what's using port 9050:

```bash
sudo lsof -i :9050
```

Kill the process or use a different port:

```bash
python3 -m uvicorn api:app --host 0.0.0.0 --port 9051
```

### Import Errors

Ensure all dependencies are installed:

```bash
source ~/mlir-aie/ironenv/bin/activate
pip3 install -r requirements.txt
```

(Note: requirements.txt needs to be created for production)

### NPU Not Found

Verify NPU is available:

```bash
/opt/xilinx/xrt/bin/xrt-smi examine
```

Check for XDNA2 device.

---

## Performance Monitoring

### Real-Time Metrics

Monitor service performance:

```bash
watch -n 1 'curl -s http://localhost:9050/stats | python3 -m json.tool'
```

### Expected Performance

- **Realtime Factor**: 400-500x (target)
- **Latency**: ~50ms for 30s audio
- **NPU Utilization**: 2-3%
- **Memory**: <2GB per worker

---

## Security Considerations

### Production Deployment

1. **Firewall**: Restrict port 9050 access
2. **Authentication**: Add API key validation (not implemented in v2.0)
3. **Rate Limiting**: Configure in nginx/reverse proxy
4. **TLS**: Use reverse proxy (nginx) for HTTPS

### Example Nginx Configuration

```nginx
server {
    listen 443 ssl;
    server_name amanuensis.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:9050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Backup and Recovery

### Backup Configuration

```bash
tar -czf amanuensis-backup-$(date +%Y%m%d).tar.gz \
  config/ \
  deployment/ \
  xdna2/
```

### Restore from Backup

```bash
tar -xzf amanuensis-backup-20251101.tar.gz
sudo systemctl restart unicorn-amanuensis
```

---

## Upgrade Procedure

1. Stop service:
   ```bash
   sudo systemctl stop unicorn-amanuensis
   ```

2. Backup current version:
   ```bash
   cp -r /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis \
      /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis.bak
   ```

3. Deploy new version:
   ```bash
   # Update code
   git pull
   ```

4. Update dependencies:
   ```bash
   source ~/mlir-aie/ironenv/bin/activate
   pip3 install -r requirements.txt
   ```

5. Restart service:
   ```bash
   sudo systemctl start unicorn-amanuensis
   ```

6. Verify:
   ```bash
   ./deployment/health_check.sh
   ```

---

## Support

- **Documentation**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/`
- **Week 6 Report**: `WEEK6_COMPLETE.md` (to be created)
- **Issues**: GitHub Issues (when published)

---

**Deployed**: November 1, 2025
**Status**: Week 6 Production Ready
**Maintainer**: CC-1L Integration Team
