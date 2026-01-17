
import sys
import os
from pathlib import Path
from core.config import Config
from core.tts.manager import TTSManager

def test_tts():
    config = Config()
    tts = TTSManager(config)
    
    test_text = "This is a diagnostic test for the TTS system."
    voice = "sexy"
    lang = "en"
    
    print(f"🧪 Testing TTS for voice: {voice}, engine: {tts.engine}")
    try:
        path = tts.generate_speech(test_text, voice, lang)
        if path and path.exists():
            print(f"✅ TTS Success! Path: {path}")
            print(f"📏 File size: {path.stat().st_size} bytes")
        else:
            print(f"❌ TTS Failed! Return path: {path}")
            if path:
                print(f"❓ Path exists? {path.exists()}")
    except Exception as e:
        print(f"💥 TTS Exception: {e}")

if __name__ == "__main__":
    test_tts()
