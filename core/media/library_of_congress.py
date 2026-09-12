"""
Library of Congress JSON API Provider
No API key required (rate-limited; be polite).
Docs: https://www.loc.gov/apis/json-and-yaml/ and
      https://www.loc.gov/apis/json-and-yaml/requests/endpoints/

Much of the LoC film/video collection is public domain or "no known
restrictions"; each item carries a rights field we surface as the license.
"""

from typing import List, Dict, Optional
from urllib.parse import urlparse

from .base import BaseMediaAPI, Media, MediaType


class LibraryOfCongressProvider(BaseMediaAPI):
    """Key-less access to loc.gov film/video items via the JSON API."""

    SEARCH_URL = "https://www.loc.gov/search/"

    def capabilities(self) -> Dict:
        return {
            "supports_media_types": [MediaType.VIDEO, MediaType.IMAGE],
            "requires_key": False,
            "supports_license": True,
        }

    def _search(self, query: str, limit: int, video_only: bool = True) -> List[Dict]:
        params = {
            "fo": "json",
            "q": query,
            "c": min(limit, 100),
        }
        if video_only:
            params["fa"] = "format:film,video"
        try:
            resp = self.session.get(self.SEARCH_URL, params=params, timeout=25)
            resp.raise_for_status()
            return resp.json().get("results", []) or []
        except Exception as e:
            print(f"[LoC] search error: {e}")
            return []

    @staticmethod
    def _video_files(item: Dict) -> List[Dict]:
        """Flatten resources->files and keep video file entries."""
        files = []
        for res in item.get("resources") or []:
            for group in res.get("files") or []:
                for f in group if isinstance(group, list) else [group]:
                    if not isinstance(f, dict):
                        continue
                    url = f.get("url") or ""
                    mime = (f.get("mimetype") or "").lower()
                    if mime.startswith("video") or urlparse(url).path.lower().endswith(
                        (".mp4", ".mov", ".webm", ".m4v", ".mpg", ".mpeg")
                    ):
                        files.append(f)
        return files

    def search(
        self, query: str, media_type: MediaType = MediaType.ANY, limit: int = 20
    ) -> List[Media]:
        results: List[Media] = []
        for item in self._search(query, limit, video_only=True):
            vids = self._video_files(item)
            if not vids:
                continue
            best = vids[0]
            url = best.get("url")
            if not url:
                continue
            image_urls = item.get("image_url") or []
            thumb = image_urls[-1] if image_urls else None
            rights = item.get("rights") or item.get("rights_advisory") or "See item page for rights information"
            path = urlparse(url).path.lower()
            ext = next((e for e in (".mp4", ".mov", ".webm", ".m4v", ".mpg", ".mpeg") if path.endswith(e)), ".mp4")
            results.append(
                Media(
                    url=url,
                    title=item.get("title"),
                    creator=(item.get("contributor") or [None])[0]
                    if isinstance(item.get("contributor"), list)
                    else item.get("contributor"),
                    thumbnail_url=thumb,
                    license=rights if isinstance(rights, str) else "See item page for rights information",
                    license_url=item.get("id"),
                    width=best.get("width"),
                    height=best.get("height"),
                    ext=ext,
                    media_type=MediaType.VIDEO,
                    source="LibraryOfCongress",
                )
            )
            if len(results) >= limit:
                break
        return results

    def search_videos(
        self, query: str, orientation: str = "portrait", per_page: int = 15
    ) -> List[Dict]:
        return [
            {
                "url": m.url,
                "ext": m.ext or ".mp4",
                "thumbnail": m.thumbnail_url,
                "title": m.title,
                "width": m.width,
                "height": m.height,
            }
            for m in self.search(query, MediaType.VIDEO, per_page)
        ]

    def download_video(self, video_url: str, output_path) -> bool:
        return self._download_file(video_url, output_path, timeout=120)
