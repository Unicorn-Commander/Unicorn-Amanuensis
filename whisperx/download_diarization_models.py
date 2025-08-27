#!/usr/bin/env python3
"""
Download pyannote diarization models for local use
These models run 100% locally after download
"""

import os
import torch
from pathlib import Path

print("🦄 Unicorn Amanuensis - Downloading Diarization Models")
print("=" * 60)
print("These models will be cached locally for offline use")
print("=" * 60)

# Set cache directory
cache_dir = Path("/models/pyannote")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(cache_dir)

print(f"\n📁 Cache directory: {cache_dir}")

# Models we need for speaker diarization
models_to_download = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0", 
    "speechbrain/spkrec-ecapa-voxceleb"
]

print("\n📥 Downloading models for local speaker diarization:")
print("   These run 100% locally after download\n")

try:
    # Try to import and download
    from pyannote.audio import Pipeline, Model
    from speechbrain.inference.speaker import EncoderClassifier
    
    for model_name in models_to_download:
        print(f"⬇️  Downloading: {model_name}")
        
        if "pyannote" in model_name:
            # For pyannote models, we can download without token for non-commercial use
            try:
                if "diarization" in model_name:
                    # This is the main pipeline
                    pipeline = Pipeline.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        use_auth_token=False
                    )
                    print(f"   ✅ Downloaded pipeline: {model_name}")
                else:
                    # This is a model component
                    model = Model.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        use_auth_token=False
                    )
                    print(f"   ✅ Downloaded model: {model_name}")
            except Exception as e:
                print(f"   ⚠️  Note: {model_name} may require accepting license terms")
                print(f"      Visit: https://huggingface.co/{model_name}")
                
        elif "speechbrain" in model_name:
            # SpeechBrain models are fully open
            classifier = EncoderClassifier.from_hparams(
                source=model_name,
                savedir=cache_dir / "speechbrain",
                run_opts={"device": "cpu"}
            )
            print(f"   ✅ Downloaded: {model_name}")
            
    print("\n✅ Models downloaded successfully!")
    print("📍 Models are cached locally at:", cache_dir)
    print("🚀 You can now use speaker diarization offline!")
    
except ImportError as e:
    print("\n⚠️ Missing dependencies. Installing...")
    os.system("pip install pyannote.audio speechbrain")
    print("\n Please run this script again after installation.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nNote: Some pyannote models require accepting license terms.")
    print("You can still use the service without diarization!")

print("\n" + "=" * 60)
print("Alternative: Manual download instructions:")
print("=" * 60)
print("""
If automatic download doesn't work, you can:

1. Download models manually from HuggingFace:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

2. Place them in: /models/pyannote/

3. The service will use them locally without internet connection.

Note: These models are for research/non-commercial use by default.
For commercial use, check the model licenses.
""")