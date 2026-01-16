import os
import requests
import random
from pathlib import Path
from typing import List, Dict, Optional

class UnsplashAPI:
    """
    API Client for Unsplash to search and download photos.
    """
    def __init__(self, access_key: str = None):
        self.access_key = access_key or os.getenv("UNSPLASH_ACCESS_KEY")
        self.base_url = "https://api.unsplash.com"
        
    def search_photos(self, query: str, per_page: int = 20) -> List[Dict]:
        """
        Search for photos on Unsplash.
        Returns a list of dicts with 'url', 'id', 'user', 'description'.
        """
        if not self.access_key:
            print("⚠️ [Unsplash] No access key provided.")
            return []

        url = f"{self.base_url}/search/photos"
        params = {
            "query": query,
            "per_page": per_page,
            "client_id": self.access_key,
            "orientation": "portrait"  # Prefer portrait for mobile video background
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get("results", []):
                    # Use 'regular' size for balance, or 'full' for quality
                    img_url = item["urls"]["regular"] 
                    results.append({
                        "url": img_url,
                        "source": "Unsplash",
                        "id": item["id"],
                        "description": item.get("alt_description") or item.get("description") or "Unsplash Photo"
                    })
                return results
            else:
                print(f"❌ [Unsplash] Search failed ({response.status_code}): {response.text}")
                return []
        except Exception as e:
            print(f"❌ [Unsplash] Exception during search: {e}")
            return []

    def download_photo(self, url: str, output_path: Path) -> bool:
        """
        Download photo from URL and save to output_path.
        """
        try:
            # Unsplash requires triggering the download location endpoint for attribution tracking
            # But the URL we get from 'urls' is a direct image URL.
            # To properly comply, we should also hit the 'download_location' link provided in the API response,
            # but for this MVP 'download_photo' just grabs the bytes.
            
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                print(f"❌ [Unsplash] Download failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ [Unsplash] Download exception: {e}")
            return False
