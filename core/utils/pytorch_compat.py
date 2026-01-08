"""
PyTorch Compatibility Utilities
Configures PyTorch secure loading for TTS models
"""
import torch

def setup_pytorch_allowlist():
    """
    Configure PyTorch 2.6+ secure loading to allow TTS model classes.
    This must be called before loading any TTS models.
    """
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.configs.shared_configs import BaseAudioConfig
        
        torch.serialization.add_safe_globals([
            XttsConfig, XttsAudioConfig, XttsArgs, 
            BaseDatasetConfig, BaseAudioConfig
        ])
        print("[PyTorch] Configured secure loading allowlist for TTS models")
    except ImportError as e:
        print(f"[WARNING] Could not import TTS classes for PyTorch allowlist: {e}")
