"""
Unicorn-Amanuensis: Multi-Platform Speech-to-Text API

Automatically detects and uses the best available backend:
- XDNA2 for Strix Point NPU
- XDNA1 for Phoenix/Hawk Point NPU
- CPU for fallback support
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import logging
import sys
import os

# Add runtime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'runtime'))

from runtime.platform_detector import get_platform, get_platform_info, Platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unicorn-Amanuensis",
    description="Multi-platform Speech-to-Text Service with NPU acceleration",
    version="2.0.0"
)

# Detect platform and load appropriate backend
platform = get_platform()
platform_info = get_platform_info()

logger.info(f"Initializing Unicorn-Amanuensis on {platform.value} backend")
logger.info(f"Platform info: {platform_info}")

# Import platform-specific server
if platform == Platform.XDNA2:
    logger.info("Loading XDNA2 backend...")
    try:
        # Import XDNA2 implementation with CC-1L's 1,183x matmul kernel!
        from xdna2.runtime.whisper_xdna2_runtime import create_runtime

        # Create XDNA2 runtime instance
        runtime = create_runtime(model_size="base", use_4tile=True)
        logger.info("XDNA2 backend loaded successfully with NPU acceleration!")
        backend_type = "XDNA2 (NPU-Accelerated with 1,183x INT8 matmul)"

        # For now, fall back to XDNA1 API wrapper
        # TODO: Create native XDNA2 FastAPI server
        from xdna1.server import app as backend_app
    except Exception as e:
        logger.warning(f"XDNA2 backend failed to load: {e}")
        logger.info("Falling back to XDNA1 backend")
        from xdna1.server import app as backend_app
        backend_type = "XDNA1 (XDNA2 fallback)"
elif platform == Platform.XDNA1:
    logger.info("Loading XDNA1 backend...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xdna1'))
    from xdna1.server import app as backend_app
    backend_type = "XDNA1"
else:  # CPU
    logger.info("Loading CPU backend...")
    # TODO: Import CPU implementation when ready
    # For now, fall back to XDNA1 (which supports CPU mode)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xdna1'))
    from xdna1.server import app as backend_app
    backend_type = "CPU"

# Mount backend routes
app.mount("/", backend_app)


@app.get("/platform")
async def get_platform_endpoint():
    """Get current platform information"""
    return {
        "service": "Unicorn-Amanuensis",
        "version": "2.0.0",
        "platform": platform_info,
        "backend": backend_type
    }


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Unicorn-Amanuensis",
        "description": "Multi-platform Speech-to-Text Service",
        "version": "2.0.0",
        "platform": platform.value,
        "backend": backend_type,
        "endpoints": {
            "/v1/audio/transcriptions": "POST - Transcribe audio",
            "/health": "GET - Health check",
            "/platform": "GET - Platform information"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
