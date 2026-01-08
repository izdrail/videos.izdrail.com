#!/usr/bin/env python3
"""
Script to pre-download the XTTS model during Docker build
"""
import os
import sys
from TTS.api import TTS
import torch
import traceback

# Fix for PyTorch 2.6+ secure loading - comprehensive TTS allowlist
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.shared_configs import BaseAudioConfig
    torch.serialization.add_safe_globals([
        XttsConfig, XttsAudioConfig, XttsArgs, 
        BaseDatasetConfig, BaseAudioConfig
    ])
except ImportError as e:
    print(f"[WARNING] Could not import TTS classes for PyTorch allowlist: {e}")

def main():
    print("Pre-downloading XTTS model...")
    
    try:
        # Set environment variable for license agreement
        os.environ['COQUI_TOS_AGREED'] = '1'
        
        # Initialize TTS with the same model used in your app
        # This will download the model files if they don't exist
        print("Downloading XTTS v2 model...")
        tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')
        print("XTTS model downloaded successfully!")
        
        # Clear some memory
        del tts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"FAILED to download XTTS model!")
        traceback.print_exc()
        sys.exit(1) # CRITICAL: Build must fail if model is not present

    print("Pre-downloading SpeechBrain models...")
    try:
        from speechbrain.pretrained import HIFIGAN, Tacotron2
        
        # Use persistent cache directory for SpeechBrain
        sb_cache = os.environ.get('SPEECHBRAIN_CACHE', '/opt/speechbrain_models')
        
        # Tacotron2
        Tacotron2.from_hparams(
            source="speechbrain/tts-tacotron2-ljspeech", 
            savedir=f"{sb_cache}/tts-tacotron2-ljspeech"
        )
        # HIFIGAN
        HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-ljspeech", 
            savedir=f"{sb_cache}/tts-hifigan-ljspeech"
        )
        print("SpeechBrain models downloaded successfully!")
    except Exception as e:
        print(f"Error downloading SpeechBrain models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()