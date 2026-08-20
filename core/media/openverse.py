"""
Openverse API Client
Aggregates 800M+ openly licensed images and audio from openverse.org.
No API key is required for anonymous access (rate-limited).
"""

import time
from typing import List, Dict, Optional
from urllib.parse import urlparse

from .base import BaseMediaAPI, Media, MediaType


class OpenverseProvider(BaseMediaAPI):
    """Anonymous access to the Openverse catalog (images + audio)."""

    BASE = "https://api.openverse.org/v1"

    def __init__(self, api_key: Optional[str] = None):
        # Openverse allows anonymous requests; ``api_key`` is an optional token.
        super().__init__(api_key)
        if self.api_key:
            self.session.headers.update({"Authorization": f"Token {self.api_key}"})

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _ext(url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        path = urlparse(url).path.lower()
        for ext in (
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".mp4",
            ".webm",
            ".ogg",
            ".mp3",
            ".wav",
        ):
            if path.endswith(ext):
                return ext
        return None

    def _request(self, endpoint: str, params: Dict) -> Dict:
        """GET with simple exponential backoff on 429 / transient errors."""
        url = f"{self.BASE}/{endpoint}"
        backoff = 0.5
        for attempt in range(4):
            try:
                resp = self.session.get(url, params=params, timeout=15)
                if resp.status_code == 429:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 4.0)
                    continue
                if resp.status_code in (401, 403):
                    print(
                        f"[Openverse] Auth required ({resp.status_code}); "
                        f"anonymous access unavailable."
                    )
                    return {}
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt == 3:
                    print(f"[Openverse] Request failed: {e}")
                    return {}
                time.sleep(backoff)
                backoff = min(backoff * 2, 4.0)
        return {}

    # -- public API ------------------------------------------------------
    def search(
        self,
        query: str,
        media_type: MediaType = MediaType.ANY,
        limit: int = 20,
    ) -> List[Media]:
        """Search Openverse for openly licensed media.

        Openverse v1 exposes ``images`` and ``audio``; video is not a first-class
        endpoint, so ``VIDEO`` requests fall back to images (usable as static
        backgrounds). Returns a list of :class:`Media` objects.
        """
        if media_type == MediaType.AUDIO:
            endpoint = "audio"
        else:
            # IMAGE / VIDEO / ANY -> images (the only visual catalog available)
            endpoint = "images"

        cache_key = f"ov_{endpoint}_{query}_{limit}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        params = {
            "q": query,
            "page_size": min(limit, 50),
            "page": 1,
            "mature": "false",
        }
        data = self._request(endpoint, params)
        results: List[Media] = []
        for item in data.get("results", []) or []:
            url = item.get("url")
            if not url:
                continue
            license_val = item.get("license")
            results.append(
                Media(
                    url=url,
                    title=item.get("title"),
                    creator=item.get("creator"),
                    thumbnail_url=item.get("thumbnail"),
                    license=license_val,
                    license_version=item.get("license_version"),
                    attribution=item.get("attribution"),
                    license_url=item.get("license_url"),
                    width=item.get("width"),
                    height=item.get("height"),
                    duration=item.get("duration"),
                    ext=self._ext(url) or (".mp3" if endpoint == "audio" else ".jpg"),
                    provider="Openverse",
                    media_type=(
                        MediaType.AUDIO if endpoint == "audio" else MediaType.IMAGE
                    ),
                    source="Openverse",
                )
            )
            if len(results) >= limit:
                break

        self.search_cache[cache_key] = results
        return results

    def search_videos(
        self, query: str, orientation: str = "portrait", per_page: int = 15
    ) -> List[Dict]:
        """Manager-compatible search returning image backgrounds from Openverse."""
        media = self.search(query, media_type=MediaType.IMAGE, limit=per_page)
        return [m.to_dict() for m in media]

    def download_video(self, video_url: str, output_path) -> bool:
        return self._download_file(video_url, output_path)

    def capabilities(self) -> Dict:
        return {
            "supports_media_types": [MediaType.IMAGE, MediaType.AUDIO],
            "requires_key": False,
            "supports_license": True,
        }
