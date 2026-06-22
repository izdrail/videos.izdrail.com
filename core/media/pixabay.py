"""
Pixabay API Client
Provides access to Pixabay video library
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
from .base import BaseMediaAPI


class PixabayAPI(BaseMediaAPI):
    """Pixabay video search and download API"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("PIXABAY_API_KEY", ""))

    def search_videos(
        self, query: str, orientation: str = "portrait", per_page: int = 15
    ) -> List[Dict]:
        """
        Search Pixabay for videos

        Args:
            query: Search query
            orientation: Video orientation (not strictly supported by Pixabay API same as Pexels, but we can filter)
            per_page: Number of results per page

        Returns:
            List of video metadata dictionaries
        """
        if not self.api_key:
            print("[PixabayAPI] No API key configured")
            return []

        # Check cache
        cache_key = f"{query}_{orientation}_{per_page}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        try:
            url = "https://pixabay.com/api/videos/"
            params = {
                "key": self.api_key,
                "q": query,
                "video_type": "all",
                "per_page": per_page,
                "page": 1,
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for hit in data.get("hits", []):
                videos = hit.get("videos", {})
                # Choose best quality available (usually 'large' or 'medium' or 'small')
                # We prefer small/medium for file size unless large is required.
                # 'tiny', 'small', 'medium', 'large'

                # Priority: medium -> small -> large -> tiny
                best_video = (
                    videos.get("medium")
                    or videos.get("small")
                    or videos.get("large")
                    or videos.get("tiny")
                )

                if best_video:
                    results.append(
                        {
                            "url": best_video.get("url"),
                            "id": hit.get("id"),
                            "duration": hit.get("duration", 0),
                            "width": best_video.get("width"),
                            "height": best_video.get("height"),
                            "thumbnail": hit.get("previewURL")
                            or hit.get("userImageURL"),
                        }
                    )

            # Cache results
            self.search_cache[cache_key] = results
            return results

        except Exception as e:
            print(f"[PixabayAPI] Search error: {e}")
            return []

    def download_video(self, video_url: str, output_path: Path) -> bool:
        """
        Download video from Pixabay
        """
        return self._download_file(video_url, output_path)
