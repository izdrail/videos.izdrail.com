"""
Giphy API Client
Provides access to Giphy GIF library
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
from .base import BaseMediaAPI


class GiphyAPI(BaseMediaAPI):
    """Giphy GIF search and download API"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("GIPHY_API_KEY", ""))

    def search_videos(
        self, query: str, orientation: str = "portrait", per_page: int = 15
    ) -> List[Dict]:
        """
        Search Giphy for GIFs

        Args:
            query: Search query
            orientation: Not used for Giphy
            per_page: Number of results (limit)

        Returns:
            List of GIF metadata dictionaries with 'url' and 'id' keys
        """
        if not self.api_key:
            print("[GiphyAPI] No API key configured")
            return []

        # Check cache
        cache_key = f"{query}_{per_page}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        try:
            url = "https://api.giphy.com/v1/gifs/search"
            params = {
                "api_key": self.api_key,
                "q": query,
                "limit": per_page,
                "rating": "g",
                "lang": "en",
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for gif in data.get("data", []):
                images = gif.get("images", {})
                # Prefer downsized or fixed_height_small for smaller downloads
                best_version = (
                    images.get("downsized_small")
                    or images.get("fixed_height_small")
                    or images.get("original", {})
                )
                if best_version:
                    results.append(
                        {
                            "url": best_version.get("mp4") or best_version.get("url"),
                            "id": gif.get("id"),
                            "width": int(best_version.get("width", 0)),
                            "height": int(best_version.get("height", 0)),
                            "thumbnail": images.get("fixed_height_small", {}).get(
                                "url"
                            ),
                        }
                    )

            # Cache results
            self.search_cache[cache_key] = results
            return results

        except Exception as e:
            print(f"[GiphyAPI] Search error: {e}")
            return []

    def download_video(self, video_url: str, output_path: Path) -> bool:
        """
        Download GIF from Giphy

        Args:
            video_url: URL of the GIF
            output_path: Path to save the GIF

        Returns:
            True if successful, False otherwise
        """
        return self._download_file(video_url, output_path)
