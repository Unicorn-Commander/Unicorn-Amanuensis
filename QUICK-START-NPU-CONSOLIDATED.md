# Quick Start - Unicorn Amanuensis with NPU (Consolidated)

**Date**: October 23, 2025
**Version**: NPU Consolidated v1.0

---

## 🆕 What's New: Consolidated NPU Support

Amanuensis now uses [unicorn-npu-core](https://github.com/Unicorn-Commander/unicorn-npu-core) - a shared library that provides NPU support across all Unicorn projects.

**Benefits**:
- ✅ Single command installation
- ✅ Automatic host system setup
- ✅ Consistent NPU support across projects
- ✅ Easy updates and bug fixes

---

## 🚀 Installation (One Command!)

```bash
bash scripts/install-amanuensis.sh
```

That's it! This script automatically:
1. Installs `unicorn-npu-core` library
2. Sets up NPU on host system (XRT, drivers, permissions)
3. Installs Amanuensis dependencies
4. Offers to download Whisper models

**Time**: ~5-10 minutes (first time), ~1-2 minutes (if NPU already set up)

---

## 📋 Manual Installation (Step-by-Step)

If you prefer manual control:

### Step 1: Install unicorn-npu-core

```bash
# From GitHub (once published)
pip install git+https://github.com/Unicorn-Commander/unicorn-npu-core.git

# OR from local development
cd /home/ucadmin/UC-1/unicorn-npu-core
pip install -e . --break-system-packages
```

### Step 2: Setup NPU on Host

```bash
# Using Python
python3 -m unicorn_npu.scripts.install_host

# OR directly
cd /home/ucadmin/UC-1/unicorn-npu-core
bash scripts/install-npu-host.sh
```

**Important**: Log out and log back in after this step!

### Step 3: Install Amanuensis

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis
pip install -r requirements.txt --break-system-packages
```

### Step 4: Download Whisper Models (Optional)

```bash
bash download_onnx_models.sh
```

---

## ⚡ Set NPU to Performance Mode

For best performance, set NPU to performance mode:

```bash
python3 -c "
from unicorn_npu import NPUDevice

npu = NPUDevice()
if npu.is_available():
    npu.set_power_mode('performance')
    print('✅ NPU set to performance mode')
    print(f'Power state: {npu.get_power_state()}')
else:
    print('❌ NPU not available')
"
```

---

## 🎯 Start the Service

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
python3 server_whisperx_npu.py
```

**Default Port**: 9000

**API Endpoints**:
- `POST /transcribe` - Main transcription endpoint
- `GET /health` - Health check
- `GET /status` - NPU status

---

## 🧪 Test the Service

```bash
# Check health
curl http://localhost:9000/health

# Transcribe audio
curl -X POST -F "file=@audio.wav" http://localhost:9000/transcribe
```

---

## 🔍 Verify NPU is Working

```python
from unicorn_npu import NPUDevice

npu = NPUDevice()

if npu.is_available():
    info = npu.get_device_info()
    print(f"✅ NPU Available")
    print(f"   Device: {info['device']}")
    print(f"   Type: {info['type']}")
    print(f"   Power: {npu.get_power_state()}")
else:
    print("❌ NPU not available")
```

---

## 📊 Expected Performance

With NPU acceleration:

| Model | Performance | Quality |
|-------|-------------|---------|
| whisper-base | **11.2x realtime** | Good |
| whisper-small | 8x realtime | Better |
| whisper-medium | 5x realtime | Excellent |
| whisper-large-v3 | 3x realtime | Best |

**Note**: With Unicorn Commander (whisper_npu_project), you can achieve **51x realtime** with distil-whisper!

---

## 🛠️ Troubleshooting

### NPU Not Detected

```bash
# Check device exists
ls -la /dev/accel/accel0

# Check permissions
groups | grep -E "(render|video)"

# Re-run host setup
python3 -m unicorn_npu.scripts.install_host
```

### Import Errors

```bash
# Reinstall unicorn-npu-core
pip install --force-reinstall unicorn-npu-core
```

### XRT Issues

```bash
# Check XRT installation
/opt/xilinx/xrt/bin/xrt-smi examine

# Verify XRT version
/opt/xilinx/xrt/bin/xrt-smi version
```

---

## 📚 Additional Documentation

- **Core Library**: [unicorn-npu-core README](https://github.com/Unicorn-Commander/unicorn-npu-core)
- **NPU Setup Details**: `NPU-SETUP.md`
- **Architecture**: `ARCHITECTURE.md`
- **XRT Installation**: `XRT-NPU-INSTALLATION.md`

---

## 🔄 Updating

### Update unicorn-npu-core

```bash
pip install --upgrade git+https://github.com/Unicorn-Commander/unicorn-npu-core.git
```

### Update Amanuensis

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis
git pull
pip install -r requirements.txt --break-system-packages
```

---

## 💡 Tips

1. **Always use performance mode** for production:
   ```bash
   python3 -c "from unicorn_npu import NPUDevice; NPUDevice().set_power_mode('performance')"
   ```

2. **Check NPU status before starting**:
   ```bash
   python3 -c "from unicorn_npu import NPUDevice; print(f'NPU: {NPUDevice().is_available()}')"
   ```

3. **Use Docker for production** (see main README.md)

4. **Monitor NPU power state**:
   ```bash
   /opt/xilinx/xrt/bin/xrt-smi examine | grep -i power
   ```

---

## 🦄 Related Projects

- **unicorn-npu-core**: Core NPU library (shared)
- **Unicorn-Orator**: TTS service with NPU
- **whisper_npu_project**: Unicorn Commander (51x realtime!)
- **amd-npu-utils**: NPU development toolkit

---

**Questions?** Check the [main README](README.md) or [NPU-SETUP.md](NPU-SETUP.md)

**Ready to go!** 🚀
