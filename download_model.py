import os
import sys
import time
from TTS.api import TTS
import torch
import traceback

# Fix for PyTorch 2.4+ secure loading - comprehensive TTS allowlist
try:
    if hasattr(torch.serialization, 'add_safe_globals'):
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

# SpeechBrain compatibility fix for newer torchaudio
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []

def download_with_retry(func, name, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n▶ Attempting to download {name} (Attempt {attempt})...")
            func()
            print(f"✔ {name} downloaded successfully")
            return
        except Exception as e:
            print(f"⚠️  Attempt {attempt} failed for {name}: {e}")
            if attempt < max_retries:
                wait_time = attempt * 10
                print(f"🔄 Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

def main():
    print("Pre-downloading XTTS model...")
    
    try:
        # Set environment variable for license agreement
        os.environ['COQUI_TOS_AGREED'] = '1'
        
        def download_xtts():
            tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')
            del tts

        download_with_retry(download_xtts, "XTTS v2")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"FAILED to download XTTS model!")
        traceback.print_exc()
        sys.exit(1)

    print("\nPre-downloading SpeechBrain models...")
    try:
        from speechbrain.pretrained import HIFIGAN, Tacotron2
        sb_cache = os.environ.get('SPEECHBRAIN_CACHE', '/opt/speechbrain_models')
        
        def download_sb():
            Tacotron2.from_hparams(
                source="speechbrain/tts-tacotron2-ljspeech", 
                savedir=f"{sb_cache}/tts-tacotron2-ljspeech"
            )
            HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-ljspeech", 
                savedir=f"{sb_cache}/tts-hifigan-ljspeech"
            )

        download_with_retry(download_sb, "SpeechBrain models")

    except Exception as e:
        print(f"Error downloading SpeechBrain models: {e}")
        sys.exit(1)

    print("\nPre-downloading CLIP (open_clip) models...")
    try:
        import open_clip

        def download_clip():
            for weights in ["laion2b_s34b_b79k", "openai"]:
                try:
                    print(f"Downloading open_clip ViT-B-32 with {weights}...")
                    open_clip.create_model_and_transforms("ViT-B-32", pretrained=weights)
                except Exception as clip_err:
                    print(f"Warning downloading open_clip with {weights}: {clip_err}")

        download_with_retry(download_clip, "open_clip models")
    except Exception as e:
        print(f"Error downloading open_clip models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()