import sys
import os
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.media.unsplash import UnsplashAPI
from core.config import Config

def test_unsplash_api():
    config = Config()
    if not config.UNSPLASH_ACCESS_KEY:
        print("❌ SKIPPING: No Unsplash Key in Environment")
        return

    api = UnsplashAPI(config.UNSPLASH_ACCESS_KEY)
    
    # 1. Search
    print("Searching 'mountain'...")
    results = api.search_photos("mountain", per_page=1)
    if not results:
        print("❌ Search returned no results")
        return
        
    photo = results[0]
    print(f"✅ Found photo: {photo['id']} by {photo['description']}")
    print(f"URL: {photo['url']}")
    
    # 2. Download
    output_path = config.TEMP_DIR / f"test_unsplash_{photo['id']}.jpg"
    print(f"Downloading to {output_path}...")
    success = api.download_photo(photo['url'], output_path)
    
    if success and output_path.exists():
        print(f"✅ Download successful ({output_path.stat().st_size} bytes)")
        output_path.unlink() # Cleanup
        print("✅ Cleanup complete")
    else:
        print("❌ Download failed")

if __name__ == "__main__":
    test_unsplash_api()
