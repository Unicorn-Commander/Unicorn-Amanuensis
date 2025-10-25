# XRT + NPU Plugin Installation Guide

## For AMD Ryzen AI NPU (Phoenix/Hawk Point/Strix)

This guide covers installing XRT (Xilinx Runtime) and the NPU plugin for AMD Ryzen AI processors with integrated NPUs.

---

## 🎯 Prerequisites

### Hardware
- AMD Ryzen 9 8945HS (Phoenix) - **Tested ✓**
- AMD Ryzen AI 7/9 Series (Hawk Point) - Compatible
- AMD Ryzen AI 300 Series (Strix) - Compatible

### Software
- **OS**: Ubuntu 25.04
- **Kernel**: Linux 6.14+ (included in Ubuntu 25.04)
- **BIOS**: NPU/IPU must be enabled
  - Navigate to: BIOS → Advanced → CPU Configuration → IPU → Enabled

### Verify NPU Hardware

```bash
# Check for NPU in PCI devices
lspci | grep -i "signal processing"
# Should show: Signal processing controller

# Check device node
ls -la /dev/accel/accel0
# Should exist if driver is loaded
```

---

## 🚀 Option 1: Automated Install (Recommended)

Use the automated installer that handles prebuilts and fallback to source:

```bash
cd ~
curl -L -O https://raw.githubusercontent.com/Unicorn-Commander/npu-prebuilds/main/install-xrt-npu.sh
chmod +x install-xrt-npu.sh
./install-xrt-npu.sh
```

**Options:**
- `./install-xrt-npu.sh` - Try prebuilts, fallback to source
- `./install-xrt-npu.sh --from-source` - Always build from source
- `./install-xrt-npu.sh --clean` - Remove build directories after install

**Installation time:**
- With prebuilts: 5-10 minutes
- From source: 30-45 minutes

---

## 📦 Option 2: Manual Install with Prebuilts

### Step 1: Download Prebuilt Packages

```bash
mkdir -p ~/xrt-install && cd ~/xrt-install

# Download XRT base runtime
curl -L -O https://github.com/Unicorn-Commander/npu-prebuilds/releases/download/xrt-2.20.0/xrt_202520.2.20.0_25.04-amd64-base.deb

# Download XRT NPU runtime
curl -L -O https://github.com/Unicorn-Commander/npu-prebuilds/releases/download/xrt-2.20.0/xrt_202520.2.20.0_25.04-amd64-npu.deb

# Download XRT NPU plugin (includes DKMS driver)
curl -L -O https://github.com/Unicorn-Commander/npu-prebuilds/releases/download/xrt-2.20.0/xrt_plugin.2.20.0_25.04-amd64-amdxdna.deb

# Download checksums
curl -L -O https://github.com/Unicorn-Commander/npu-prebuilds/releases/download/xrt-2.20.0/SHA256SUMS.txt
```

### Step 2: Verify Downloads

```bash
sha256sum -c SHA256SUMS.txt
```

All packages should show "OK".

### Step 3: Install Packages

```bash
# Remove old XRT if present
sudo apt remove -y xrt-base xrt-npu xrt_plugin-amdxdna 2>/dev/null || true

# Install XRT packages
sudo apt install -y ./xrt_*.deb ./xrt_plugin*.deb
```

The plugin installation will:
- Build and install the AMDXDNA kernel module via DKMS
- Install NPU firmware
- Load the NPU driver

### Step 4: Setup Environment

```bash
# Source XRT environment (current session)
source /opt/xilinx/xrt/setup.sh

# Add to ~/.bashrc for automatic loading
echo 'source /opt/xilinx/xrt/setup.sh' >> ~/.bashrc
```

### Step 5: Verify Installation

```bash
# Check XRT version
xrt-smi version

# Check NPU detection
xrt-smi examine

# Expected output:
# Device(s) Present
# |BDF             |Name          |
# |----------------|--------------|
# |[0000:c7:00.1]  |RyzenAI-npu1  |
```

---

## 🛠️ Option 3: Build from Source

Only if prebuilts are unavailable or you need a custom build.

### Step 1: Install Build Dependencies

```bash
sudo apt update
sudo apt install -y \
    build-essential cmake git curl wget bc \
    libboost-all-dev libudev-dev libssl-dev \
    libprotobuf-dev protobuf-compiler \
    libncurses5-dev libelf-dev \
    ocl-icd-opencl-dev opencl-headers \
    python3-dev python3-pip dkms jq
```

### Step 2: Clone xdna-driver Repository

```bash
cd ~
git clone https://github.com/amd/xdna-driver.git
cd xdna-driver
git submodule update --init --recursive
```

This includes XRT 2.20 as a submodule.

### Step 3: Install XRT Build Dependencies

```bash
sudo ./tools/amdxdna_deps.sh
```

### Step 4: Build XRT (20-30 minutes)

```bash
cd xrt/build
./build.sh -npu -opt
cd ../..
```

### Step 5: Build XRT Plugin (5-10 minutes)

```bash
cd build
./build.sh -release
cd ..
```

### Step 6: Install Built Packages

```bash
cd ~/xdna-driver

# Install XRT
sudo apt install -y \
    xrt/build/Release/xrt_202520.2.20.0_25.04-amd64-base.deb \
    xrt/build/Release/xrt_202520.2.20.0_25.04-amd64-npu.deb

# Install XRT plugin
sudo apt install -y \
    build/Release/xrt_plugin.2.20.0_25.04-amd64-amdxdna.deb
```

### Step 7: Setup Environment

```bash
source /opt/xilinx/xrt/setup.sh
echo 'source /opt/xilinx/xrt/setup.sh' >> ~/.bashrc
```

---

## ✅ Post-Installation

### Configure NPU Power Mode

For maximum performance:

```bash
# Set high performance mode
xrt-smi configure --device 0 --power-mode high

# Verify setting
xrt-smi examine | grep -i power
```

Power modes:
- `low` - Power saving (for battery life)
- `medium` - Balanced (default)
- `high` - Maximum performance (for plugged-in use)

### Check NPU Status

```bash
# Full NPU information
xrt-smi examine

# NPU firmware version
xrt-smi examine | grep "NPU Firmware"

# XDNA driver status
lsmod | grep amdxdna
```

### Test NPU

```bash
# Basic test (should show NPU device)
xrt-smi examine | grep -A 5 "Device(s) Present"
```

---

## 🔧 Usage Examples

### Using xrt-smi

```bash
# Show all devices
xrt-smi examine

# Show specific device
xrt-smi examine --device 0

# Configure device
xrt-smi configure --device 0 --power-mode high

# Reset device
xrt-smi reset --device 0

# Get help
xrt-smi --help
```

### Python XRT API

```python
import xrt

# List NPU devices
devices = xrt.device.enumerate()
print(f"Found {len(devices)} NPU device(s)")

# Open NPU device
device = xrt.device(0)
print(f"Device name: {device.get_info(xrt.device_info.name)}")
```

---

## 🐛 Troubleshooting

### NPU Not Detected

**Symptom**: `xrt-smi examine` shows "0 devices found"

**Solutions**:

1. **Check BIOS**:
   - Ensure NPU/IPU is enabled in BIOS
   - BIOS → Advanced → CPU Configuration → IPU → Enabled
   - Reboot after changing

2. **Check XDNA driver**:
   ```bash
   # Check if driver is loaded
   lsmod | grep amdxdna

   # If not loaded, load manually
   sudo modprobe amdxdna

   # Check for errors
   dmesg | grep -i amdxdna
   ```

3. **Check PCI device**:
   ```bash
   lspci | grep -i "signal processing"
   ```

   If not shown, NPU is disabled in BIOS or not supported by hardware.

4. **Reinstall XRT plugin**:
   ```bash
   sudo apt reinstall xrt_plugin-amdxdna
   ```

### DKMS Build Failure

**Symptom**: "Failed to build amdxdna module"

**Solutions**:

1. **Install kernel headers**:
   ```bash
   sudo apt install -y linux-headers-$(uname -r)
   ```

2. **Rebuild DKMS module**:
   ```bash
   sudo dkms remove -m xrt-amdxdna -v 2.20.0 --all
   sudo dkms install -m xrt-amdxdna -v 2.20.0
   ```

3. **Check DKMS status**:
   ```bash
   sudo dkms status
   ```

### Version Mismatch

**Symptom**: "Unsatisfied dependencies: xrt-base (>= 2.20)"

**Solution**: Ensure all packages are the same version (2.20.0) and from the same build.

### Permission Denied on /dev/accel/accel0

**Symptom**: Cannot access NPU device

**Solution**:
```bash
# Check permissions
ls -la /dev/accel/accel0

# Should be: crw-rw-rw- root render
# If not, fix with:
sudo chmod 666 /dev/accel/accel0

# Add user to render group (permanent fix)
sudo usermod -aG render $USER
# Log out and log back in
```

### xrt-smi Command Not Found

**Solution**:
```bash
# Source XRT environment
source /opt/xilinx/xrt/setup.sh

# Or add to PATH manually
export PATH=/opt/xilinx/xrt/bin:$PATH

# Make permanent by adding to ~/.bashrc
echo 'export PATH=/opt/xilinx/xrt/bin:$PATH' >> ~/.bashrc
```

---

## 🗑️ Uninstallation

```bash
# Remove XRT packages
sudo apt remove -y xrt-base xrt-npu xrt_plugin-amdxdna

# Remove DKMS module
sudo dkms remove -m xrt-amdxdna -v 2.20.0 --all

# Remove XRT directory
sudo rm -rf /opt/xilinx/xrt

# Clean up bashrc (optional)
sed -i '/xilinx\/xrt/d' ~/.bashrc
```

---

## 📚 Additional Resources

- [XRT GitHub](https://github.com/Xilinx/XRT)
- [XDNA Driver GitHub](https://github.com/amd/xdna-driver)
- [NPU Development Toolkit](https://github.com/Unicorn-Commander/npu-prebuilds)
- [AMD Ryzen AI Documentation](https://www.amd.com/en/developer/resources/ryzen-ai-software.html)

---

## 📋 What Gets Installed

| Component | Location | Description |
|-----------|----------|-------------|
| XRT Runtime | `/opt/xilinx/xrt/` | Core runtime libraries |
| xrt-smi | `/opt/xilinx/xrt/bin/xrt-smi` | NPU management CLI |
| XDNA Driver | `/lib/modules/*/updates/dkms/` | amdxdna.ko kernel module |
| NPU Firmware | `/opt/xilinx/xrt/firmware/` | NPU firmware binaries |
| Python API | `/opt/xilinx/xrt/python/` | Python bindings |
| Headers | `/opt/xilinx/xrt/include/` | Development headers |
| Libraries | `/opt/xilinx/xrt/lib/` | Shared libraries |

---

## 💡 Tips

1. **Always source XRT environment** before using NPU tools:
   ```bash
   source /opt/xilinx/xrt/setup.sh
   ```

2. **Check NPU status regularly**:
   ```bash
   xrt-smi examine
   ```

3. **Use high performance mode** when plugged in:
   ```bash
   xrt-smi configure --device 0 --power-mode high
   ```

4. **Monitor NPU usage** with:
   ```bash
   watch -n 1 'xrt-smi examine'
   ```

---

**Last Updated**: October 8, 2025
**XRT Version**: 2.20.0
**Target Platform**: Ubuntu 25.04, AMD Ryzen 9 8945HS
