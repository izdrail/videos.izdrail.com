#!/usr/bin/env python3
"""
Script to pre-download the XTTS model during Docker build
"""
import os
import sys
from TTS.api import TTS
import torch

def main():
    print("Pre-downloading XTTS model...")
    
    try:
        # Initialize TTS with the same model used in your app
        tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')
        print("XTTS model downloaded successfully!")
        
        # Optionally move to GPU if available (to cache GPU-specific model files)
        if torch.cuda.is_available():
            print("CUDA available, moving model to GPU...")
            tts.to("cuda")
            print("Model moved to GPU successfully!")
        else:
            print("CUDA not available, model will use CPU")
            
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()