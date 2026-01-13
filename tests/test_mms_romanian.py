import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from core.config import Config
from core.tts.manager import TTSManager

def test_mms_romanian():
    config = Config()
    tts = TTSManager(config)
    
    # Test 1: Explicit voice
    text1 = "Bună ziua! Acesta este un test explicit."
    voice1 = "MMS-TTS Romanian"
    language = "ro"
    
    # Test 2: Auto engine with generic voice
    text2 = "Acesta este un test auto cu voce standard."
    voice2 = "Standard Voice (Non-Cloned)"
    
    # Test 3: Auto engine with a "cloned" voice (should still use MMS for Romanian)
    text3 = "Acesta este un test auto cu o voce clonată, dar în limba română."
    voice3 = "Steven" # Simulating a cloned voice
    
    tests = [
        (text1, voice1, "explicit"),
        (text2, voice2, "auto_standard"),
        (text3, voice3, "auto_clone_fallback")
    ]
    
    for text, voice, label in tests:
        print(f"\n--- Testing {label} (Voice: {voice}) ---")
        try:
            # We pass engine="auto" to test the heuristic
            output_path = tts.generate_speech(text, voice, language, engine="auto")
            
            if output_path and output_path.exists():
                print(f"✅ [{label}] Success! Audio generated at: {output_path}")
                print(f"[{label}] Engine used: {tts.loaded_engine}")
            else:
                print(f"❌ [{label}] Failure: Audio path not returned or file does not exist.")
                
        except Exception as e:
            print(f"❌ [{label}] Error during generation: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_mms_romanian()
