#!/usr/bin/env python3
"""
Script to pre-download the Kokoro-82M model during Docker build
"""
import sys
from huggingface_hub import snapshot_download

def main():
    print("Pre-downloading Kokoro-82M model...")

    try:
        # Download the model to the default cache directory
        model_path = snapshot_download(
            repo_id="hexgrad/Kokoro-82M",
        )
        print(f"Kokoro-82M model downloaded successfully!")
        print(f"Model location: {model_path}")

    except Exception as e:
        print(f"Error downloading Kokoro-82M model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()