#!/usr/bin/env python3
"""
Convert Whisper models to OpenVINO IR format for Intel iGPU acceleration
Uploads to HuggingFace Hub: magicunicorn organization
"""

import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, login
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from optimum.intel import OVModelForSpeechSeq2Seq
import argparse
import torch

def convert_and_upload(model_id: str, hf_token: str = None):
    """Convert Whisper model to OpenVINO and upload to HuggingFace"""
    
    model_name = model_id.split("/")[-1]
    output_dir = f"/tmp/{model_name}-openvino"
    repo_id = f"magicunicorn/{model_name}-openvino"
    
    print(f"🔄 Converting {model_id} to OpenVINO format...")
    print(f"📂 Output directory: {output_dir}")
    print(f"🤗 Target repository: {repo_id}")
    
    # Clean output directory
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True)
    
    try:
        # Load and convert model to OpenVINO
        print("Loading original model...")
        processor = WhisperProcessor.from_pretrained(model_id)
        
        print("Converting to OpenVINO IR format (this may take a while)...")
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            export=True,
            compile=False,
            device="GPU",  # Target Intel iGPU
            ov_config={"PERFORMANCE_HINT": "LATENCY"}
        )
        
        # Save the model
        print(f"Saving OpenVINO model to {output_dir}...")
        ov_model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        
        # Create model card
        model_card = f"""---
language:
- en
- zh  
- de
- es
- ru
- fr
- ja
- ko
- pt
- tr
- pl
- it
- nl
- sv
tags:
- whisper
- openvino
- intel-igpu
- speech-recognition
- automatic-speech-recognition
- unicorn-amanuensis
license: apache-2.0
pipeline_tag: automatic-speech-recognition
---

# {model_name} OpenVINO for Intel iGPU

This is an OpenVINO-optimized version of OpenAI's Whisper {model_name} model, specifically optimized for Intel integrated GPUs (iGPU).

## Model Details

- **Original Model**: `{model_id}`
- **Optimization**: OpenVINO IR format with FP16 precision
- **Target Hardware**: Intel Arc/Iris Xe/UHD Graphics
- **Framework**: OpenVINO 2024.0+
- **Supported Languages**: 100+ languages

## Performance

Optimized for Intel integrated GPUs with:
- Reduced latency through OpenVINO optimizations
- FP16 precision for balanced speed/accuracy
- Efficient memory usage for iGPU constraints

## Usage

### With Optimum Intel

```python
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import WhisperProcessor

# Load model and processor
model = OVModelForSpeechSeq2Seq.from_pretrained("magicunicorn/{model_name}-openvino")
processor = WhisperProcessor.from_pretrained("magicunicorn/{model_name}-openvino")

# Run inference
# ... your audio processing code ...
```

### With Unicorn Amanuensis

This model is automatically used when running Unicorn Amanuensis with Intel iGPU support:

```bash
docker run -d \\
  --name unicorn-amanuensis \\
  -e WHISPER_MODEL={model_name} \\
  -e WHISPER_DEVICE=igpu \\
  --device /dev/dri:/dev/dri \\
  unicorn-amanuensis:igpu
```

## Model Files

- `openvino_encoder_model.xml/bin` - Encoder model in OpenVINO IR format
- `openvino_decoder_model.xml/bin` - Decoder model in OpenVINO IR format  
- `openvino_decoder_with_past_model.xml/bin` - Decoder with KV cache
- `config.json` - Model configuration
- `tokenizer_config.json` - Tokenizer configuration
- `preprocessor_config.json` - Audio preprocessing configuration

## Credits

- Original Whisper model by OpenAI
- Conversion by Magic Unicorn / Unicorn Commander
- Part of the Unicorn Amanuensis STT suite

## License

Apache 2.0 - Same as the original Whisper model
"""
        
        # Save model card
        readme_path = Path(output_dir) / "README.md"
        readme_path.write_text(model_card)
        print(f"✅ Model card created")
        
        # Upload to HuggingFace
        if hf_token:
            print(f"\n📤 Uploading to HuggingFace Hub...")
            api = HfApi()
            
            # Login
            login(token=hf_token)
            
            # Create repo if it doesn't exist
            try:
                api.create_repo(repo_id=repo_id, exist_ok=True)
                print(f"✅ Repository ready: {repo_id}")
            except Exception as e:
                print(f"⚠️  Repository creation: {e}")
            
            # Upload files
            api.upload_folder(
                folder_path=output_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {model_name} OpenVINO model for Intel iGPU"
            )
            print(f"✅ Model uploaded to https://huggingface.co/{repo_id}")
        else:
            print("⚠️  No HuggingFace token provided, skipping upload")
            print(f"📂 Model saved locally at: {output_dir}")
        
        return output_dir, repo_id
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Convert Whisper models to OpenVINO format")
    parser.add_argument("--models", nargs="+", 
                       default=["openai/whisper-base", "openai/whisper-small", "openai/whisper-large-v3"],
                       help="Model IDs to convert")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"),
                       help="HuggingFace API token for uploading")
    
    args = parser.parse_args()
    
    print("🦄 Unicorn Amanuensis - OpenVINO Model Converter")
    print("=" * 50)
    
    successful = []
    failed = []
    
    for model_id in args.models:
        print(f"\n{'='*50}")
        print(f"Processing: {model_id}")
        print('='*50)
        
        output_dir, repo_id = convert_and_upload(model_id, args.hf_token)
        
        if output_dir:
            successful.append((model_id, repo_id))
        else:
            failed.append(model_id)
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Conversion Summary")
    print('='*50)
    
    if successful:
        print(f"\n✅ Successfully converted ({len(successful)}):")
        for model_id, repo_id in successful:
            print(f"  - {model_id} -> https://huggingface.co/{repo_id}")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}):")
        for model_id in failed:
            print(f"  - {model_id}")
    
    print("\n✨ Conversion complete!")

if __name__ == "__main__":
    main()