"""
NASA Image and Video Library Provider
Real JSON API, no API key required (an optional NASA_API_KEY only raises rate limits).
Docs: https://images.nasa.gov/docs/images.nasa.gov_api_docs.pdf

Two-stage fetch: the search endpoint returns items whose ``href`` points at a
per-item asset manifest (a plain JSON array of file URLs); we follow it to pick
the best video rendition. NASA-created media is free to use; occasional partner
content needs a per-item check (see the research doc).
"""

from typing import List, Dict, Optional
from urllib.parse import urlparse

from .base import BaseMediaAPI, Media, MediaType


class NASAProvider(BaseMediaAPI):
    """Key-less access to images.nasa.gov video (and image) assets."""

    SEARCH_URL = "https://images-api.nasa.gov/search"

    def capabilities(self) -> Dict:
        return {
            "supports_media_types": [MediaType.VIDEO, MediaType.IMAGE],
            "requires_key": False,
            "supports_license": True,
        }

    # -- helpers ---------------------------------------------------------
    def _search_items(self, query: str, media_type: str, limit: int) -> List[Dict]:
        try:
            resp = self.session.get(
                self.SEARCH_URL,
                params={"q": query, "media_type": media_type, "page_size": min(limit, 100)},
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json().get("collection", {}).get("items", []) or []
        except Exception as e:
            print(f"[NASA] search error: {e}")
            return []

    def _asset_urls(self, manifest_url: str) -> List[str]:
        try:
            resp = self.session.get(manifest_url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            return [u for u in data if isinstance(u, str)] if isinstance(data, list) else []
        except Exception as e:
            print(f"[NASA] asset manifest error: {e}")
            return []

    @staticmethod
    def _pick_video(asset_urls: List[str]) -> Optional[str]:
        """Prefer an .mp4 that is not a tiny preview; fall back to any video."""
        videos = [
            u for u in asset_urls
            if urlparse(u).path.lower().endswith((".mp4", ".mov", ".webm", ".m4v"))
        ]
        if not videos:
            return None
        for u in videos:
            name = urlparse(u).path.lower()
            if "orig" in name or "1080" in name or "720" in name:
                return u
        return videos[0]

    # -- unified discovery API -------------------------------------------
    def search(
        self, query: str, media_type: MediaType = MediaType.ANY, limit: int = 20
    ) -> List[Media]:
        mt = "video" if media_type in (MediaType.ANY, MediaType.VIDEO) else "image"
        results: List[Media] = []
        for item in self._search_items(query, mt, limit):
            data = (item.get("data") or [{}])[0]
            manifest = item.get("href")
            links = item.get("links") or []
            thumb = links[0].get("href") if links else None
            if not manifest:
                continue
            asset_urls = self._asset_urls(manifest)
            url = self._pick_video(asset_urls) if mt == "video" else (asset_urls[0] if asset_urls else None)
            if not url:
                continue
            path = urlparse(url).path.lower()
            ext = next((e for e in (".mp4", ".mov", ".webm", ".m4v", ".jpg", ".png") if path.endswith(e)), None)
            results.append(
                Media(
                    url=url,
                    title=data.get("title"),
                    creator=data.get("center") or "NASA",
                    thumbnail_url=thumb,
                    license="NASA media usage guidelines (free to use; verify partner content)",
                    license_url="https://www.nasa.gov/multimedia/guidelines/index.html",
                    attribution=f"NASA/{data.get('center') or 'NASA'}",
                    ext=ext,
                    media_type=MediaType.VIDEO if mt == "video" else MediaType.IMAGE,
                    source="NASA",
                )
            )
            if len(results) >= limit:
                break
        return results

    # -- bandit pipeline dict API -----------------------------------------
    def search_videos(
        self, query: str, orientation: str = "portrait", per_page: int = 15
    ) -> List[Dict]:
        out = []
        for m in self.search(query, MediaType.VIDEO, per_page):
            out.append(
                {
                    "url": m.url,
                    "ext": m.ext or ".mp4",
                    "thumbnail": m.thumbnail_url,
                    "title": m.title,
                    "width": None,
                    "height": None,
                }
            )
        return out

    def download_video(self, video_url: str, output_path) -> bool:
        return self._download_file(video_url, output_path, timeout=120)
