#!/usr/bin/env python3
import sys
import shutil
import sqlite3
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from core.config import Config

def purge_cache(target='all'):
    config = Config()
    
    if target in ['all', 'audio']:
        print(f"🧹 Purging Audio Cache: {config.TEMP_AUDIO_DIR}...")
        if config.TEMP_AUDIO_DIR.exists():
            shutil.rmtree(config.TEMP_AUDIO_DIR)
            config.TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            
            # Clear DB entries
            try:
                with sqlite3.connect("generation_cache.db") as conn:
                    conn.execute("DELETE FROM tts_cache")
                    print("✅ Database entries for TTS cleared.")
            except Exception as e:
                print(f"⚠️ Could not clear DB: {e}")
        print("✅ Audio cache purged.")

    if target in ['all', 'video']:
        video_cache = config.TEMP_DIR / "video_cache"
        print(f"🧹 Purging Video Cache: {video_cache}...")
        if video_cache.exists():
            shutil.rmtree(video_cache)
            video_cache.mkdir(parents=True, exist_ok=True)
            # Clear DB logs
            try:
                with sqlite3.connect("generation_cache.db") as conn:
                    conn.execute("DELETE FROM video_logs")
                    print("✅ Database entries for video logs cleared.")
            except Exception as e:
                print(f"⚠️ Could not clear DB: {e}")
        print("✅ Video cache purged.")

if __name__ == "__main__":
    target = 'all'
    if len(sys.argv) > 1:
        target = sys.argv[1]
    purge_cache(target)
