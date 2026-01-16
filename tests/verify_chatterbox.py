import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.tts.manager import TTSManager

class MockConfig:
    DEVICE = "cpu" # Use CPU for testing to avoid CUDA issues if not available
    TEMP_AUDIO_DIR = Path("temp/audio_cache")
    SUPPORTED_LANGUAGES = {}
    VOICE_SAMPLES_DIR = Path("voice_samples")
    STANDARD_VOICE_NAME = "af_heart"
    
    def __init__(self):
        self.TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def test_chatterbox():
    print("Testing Chatterbox Integration...")
    config = MockConfig()
    manager = TTSManager(config)
    
    # Force load chatterbox
    try:
        manager._load_engine("chatterbox")
    except Exception as e:
        print(f"Failed to load chatterbox: {e}")
        # If it fails due to missing package, that's expected if not installed yet
        if "No module named 'chatterbox'" in str(e):
             print("Please install chatterbox-tts: pip install chatterbox-tts")
        return

    input_text = "Hello, this is a test of Chatterbox TTS."
    output_path = manager.generate_speech(
        text=input_text,
        voice_id="Chatterbox Multilingual",
        engine="chatterbox",
        language="en"
    )
    
    if output_path and output_path.exists():
        print(f"SUCCESS: Audio generated at {output_path}")
        print(f"File size: {output_path.stat().st_size} bytes")
    else:
        print("FAILURE: Audio was not generated.")

if __name__ == "__main__":
    test_chatterbox()
