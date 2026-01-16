import sys
import os
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.media.manager import MediaManager
from core.config import Config
from main import FFmpegVideoGenerator

def test_full_flow():
    config = Config()
    if not config.UNSPLASH_ACCESS_KEY:
        print("❌ SKIPPING: No Unsplash Key in Environment")
        return

    # 1. Test MediaManager finding Unsplash image
    print("Testing MediaManager...")
    mm = MediaManager(config)
    
    # We force Unsplash by preferred source? Logic is fuzzy but let's try direct method or preference
    # The get_random_media uses preferred_source
    image_path = mm.get_random_media(["mountain"], preferred_source="Unsplash")
    
    if not image_path:
        print("❌ MediaManager failed to find image on Unsplash")
        return
        
    print(f"✅ MediaManager found: {image_path}")
    
    if image_path.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
        print(f"⚠️ Warning: Expected image extension, got {image_path.suffix}")
        
    # 2. Test FFmpegVideoGenerator slide creation with this image
    # We need a dummy audio file
    dummy_audio = config.TEMP_DIR / "test_audio.wav"
    # Create valid dummy WAV
    import wave
    with wave.open(str(dummy_audio), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(b'\x00\x00' * 44100 * 5) # 5 seconds silence

    print("Testing FFmpeg Slide Creation...")
    generator = FFmpegVideoGenerator(config)
    output_slide = config.TEMP_DIR / "test_slide_unsplash.mp4"
    
    try:
        result = generator._create_slide_with_ffmpeg(
            sentence="Hello Unsplash",
            audio_path=dummy_audio,
            video_path=image_path,
            output_path=output_slide,
            slide_num=999,
            export_fps=30
        )
        
        if result and result.exists():
            print(f"✅ Slide created successfully: {result}")
            print(f"Size: {result.stat().st_size} bytes")
        else:
            print("❌ Slide creation failed")
            
    except Exception as e:
        print(f"❌ Exception during slide creation: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    if dummy_audio.exists(): dummy_audio.unlink()
    if output_slide.exists(): output_slide.unlink()
    # Don't delete downloaded image, it's in temp/videos cache

if __name__ == "__main__":
    test_full_flow()
