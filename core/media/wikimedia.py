"""
Wikimedia Commons API Client
Searches the Wikimedia Commons file repository (images, videos, audio) via the
MediaWiki Action API. No API key required.
"""

from typing import List, Dict, Optional
from urllib.parse import urlparse

from .base import BaseMediaAPI, Media, MediaType


class WikimediaProvider(BaseMediaAPI):
    """Read-only access to Wikimedia Commons openly licensed media."""

    API = "https://commons.wikimedia.org/w/api.php"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)

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
            ".tif",
            ".tiff",
            ".mp4",
            ".webm",
            ".ogv",
            ".ogg",
            ".mp3",
            ".wav",
        ):
            if path.endswith(ext):
                return ext
        return None

    @staticmethod
    def _em(value):
        """Extract the ``*.value`` wrapper MediaWiki often uses in extmetadata."""
        if isinstance(value, dict):
            return value.get("value")
        return value

    def _query(self, params: Dict) -> Dict:
        try:
            resp = self.session.get(self.API, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[Wikimedia] API error: {e}")
            return {}

    # -- public API ------------------------------------------------------
    def search(
        self,
        query: str,
        media_type: MediaType = MediaType.ANY,
        limit: int = 20,
    ) -> List[Media]:
        cache_key = f"wc_{query}_{media_type.value}_{limit}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        # 1) Find file pages in the File namespace (6).
        sr = self._query(
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srnamespace": 6,
                "srsearch": query,
                "srlimit": limit,
                "srwhat": "text",
            }
        )
        titles = [r.get("title") for r in sr.get("query", {}).get("search", [])]
        if not titles:
            self.search_cache[cache_key] = []
            return []

        # 2) Fetch imageinfo (url, size, mime, license) for all titles at once.
        ii = self._query(
            {
                "action": "query",
                "format": "json",
                "titles": "|".join(titles),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|mediatype|extmetadata",
                "iiurlwidth": 640,
            }
        )

        results: List[Media] = []
        pages = (ii.get("query", {}).get("pages", {}) or {}).values()
        for page in pages:
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("url")
            if not url:
                continue
            mime = (info.get("mime") or "").lower()
            if mime.startswith("video"):
                mt = MediaType.VIDEO
            elif mime.startswith("image"):
                mt = MediaType.IMAGE
            elif mime.startswith("audio"):
                mt = MediaType.AUDIO
            else:
                ext = self._ext(url) or ""
                mt = (
                    MediaType.VIDEO
                    if ext in (".mp4", ".webm", ".ogv")
                    else MediaType.IMAGE
                )
            if media_type != MediaType.ANY and mt != media_type:
                continue

            em = info.get("extmetadata") or {}
            artist = self._em(em.get("Artist")) or page.get("title")
            # Artist often contains wikitext/html; strip crude tags if present.
            if isinstance(artist, str) and ("[[" in artist or "<" in artist):
                artist = (page.get("title") or "").replace("File:", "")
            license_short = self._em(em.get("LicenseShortName"))
            license_url = self._em(em.get("LicenseUrl"))

            results.append(
                Media(
                    url=url,
                    title=page.get("title"),
                    creator=artist,
                    thumbnail_url=info.get("thumburl"),
                    license=license_short or "CC BY-SA",
                    license_version=self._em(em.get("LicenseVersion")),
                    attribution=artist,
                    license_url=license_url,
                    width=info.get("width"),
                    height=info.get("height"),
                    duration=None,
                    ext=self._ext(url),
                    provider="Wikimedia Commons",
                    media_type=mt,
                    source="Wikimedia",
                )
            )
            if len(results) >= limit:
                break

        self.search_cache[cache_key] = results
        return results

    def search_videos(
        self, query: str, orientation: str = "portrait", per_page: int = 15
    ) -> List[Dict]:
        media = self.search(query, media_type=MediaType.ANY, limit=per_page)
        return [m.to_dict() for m in media]

    def download_video(self, video_url: str, output_path) -> bool:
        return self._download_file(video_url, output_path)

    def capabilities(self) -> Dict:
        return {
            "supports_media_types": [MediaType.IMAGE, MediaType.VIDEO],
            "requires_key": False,
            "supports_license": True,
        }
