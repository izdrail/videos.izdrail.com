#!/usr/bin/env python3
"""
Script to pre-download TTS models during Docker build
- Kokoro-82M (EN)
- MMS-TTS Romanian (RO)
"""
import sys
from huggingface_hub import snapshot_download

MODELS = {
    "Kokoro-82M": "hexgrad/Kokoro-82M",
    "MMS-TTS Romanian": "facebook/mms-tts-ron",
}

def download_model(name: str, repo_id: str) -> None:
    print(f"\n▶ Pre-downloading {name}...")

    path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=["*.bin", "*.pt", "*.json", "*.onnx", "*.model"],
    )

    print(f"✔ {name} downloaded successfully")
    print(f"  Location: {path}")

def main():
    try:
        for name, repo_id in MODELS.items():
            download_model(name, repo_id)

        print("\n✅ All TTS models downloaded and cached")

    except Exception as e:
        print(f"\n❌ Model download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
