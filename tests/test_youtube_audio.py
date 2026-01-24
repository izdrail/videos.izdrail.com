
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.media.youtube_audio import YouTubeAudioLibraryAPI
from core.config import Config

def test_integration():
    config = Config()
    api = YouTubeAudioLibraryAPI(config)
    
    print("\n--- Testing Fetch ---")
    if api.tracks:
        print(f"✅ Library loaded with {len(api.tracks)} tracks.")
    else:
        print("❌ Library empty.")
        return

    print("\n--- Testing Search ---")
    query = "Epic"
    results = api.search(query)
    print(f"Search for '{query}' returned {len(results)} results.")
    if results:
        for r in results[:3]:
            print(f"  - {r.get('name')} (ID: {r.get('id')})")
    else:
        print("❌ No results found for 'Epic'.")

    print("\n--- Testing Download ---")
    if results:
        track = results[0]
        track_id = track.get('id')
        track_name = track.get('name')
        output_path = config.MUSIC_DIR / f"test_{track_id}.mp3"
        
        print(f"Downloading '{track_name}' (ID: {track_id})...")
        if api.download_track(track_id, output_path):
            print(f"✅ Successfully downloaded to {output_path}")
            # Clean up
            if output_path.exists():
                output_path.unlink()
        else:
            print("❌ Download failed.")

if __name__ == "__main__":
    test_integration()
