"""
Mixkit (Envato) Provider
No API key and no official API: scrapes the public search pages
(https://mixkit.co/free-stock-video/<slug>/).
License: Mixkit License - free for commercial use, no attribution required
(https://mixkit.co/license/).

Curated, high-quality B-roll. Markup verified 2026-09-12:
cards carry ``<video class="item-grid-video-player__video" src=".../ID-360.mp4">``
plus an ``item-grid-video-player__overlay-video-title`` span. The 1080p file
follows the same URL shape with ``-1080.mp4``.
"""

import re
from typing import List, Dict, Optional

from .base import BaseMediaAPI, Media, MediaType


class MixkitProvider(BaseMediaAPI):
    """Key-less scraped access to Mixkit's free stock video catalogue."""

    BASE = "https://mixkit.co"
    VIDEO_RE = re.compile(
        r'<video[^>]+src="(https://assets\.mixkit\.co/videos/(\d+)/[^"]+-360\.mp4)"',
        re.IGNORECASE,
    )
    TITLE_RE = re.compile(
        r'item-grid-video-player__overlay-video-title">\s*([^<]+?)\s*<', re.IGNORECASE
    )

    def capabilities(self) -> Dict:
        return {
            "supports_media_types": [MediaType.VIDEO],
            "requires_key": False,
            "supports_license": True,
        }

    def _fetch_page(self, query: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-") or "nature"
        try:
            resp = self.session.get(f"{self.BASE}/free-stock-video/{slug}/", timeout=25)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"[Mixkit] page fetch error: {e}")
            return ""

    @staticmethod
    def _parse(html: str, limit: int) -> List[Media]:
        results: List[Media] = []
        videos = MixkitProvider.VIDEO_RE.findall(html)
        titles = MixkitProvider.TITLE_RE.findall(html)
        seen = set()
        for idx, (preview_url, vid_id) in enumerate(videos):
            if vid_id in seen:
                continue
            seen.add(vid_id)
            # Upgrade the 360p preview to the 1080p file (same URL shape).
            full_url = preview_url.replace("-360.mp4", "-1080.mp4")
            thumb = f"https://assets.mixkit.co/videos/{vid_id}/{vid_id}-thumb-720-0.jpg"
            title = titles[idx].strip() if idx < len(titles) else None
            results.append(
                Media(
                    url=full_url,
                    title=title,
                    creator="Mixkit (Envato)",
                    thumbnail_url=thumb,
                    license="Mixkit License (free for commercial use, no attribution required)",
                    license_url="https://mixkit.co/license/",
                    width=1920,
                    height=1080,
                    ext=".mp4",
                    media_type=MediaType.VIDEO,
                    source="Mixkit",
                )
            )
            if len(results) >= limit:
                break
        return results

    def search(
        self, query: str, media_type: MediaType = MediaType.ANY, limit: int = 20
    ) -> List[Media]:
        if media_type not in (MediaType.ANY, MediaType.VIDEO):
            return []
        return self._parse(self._fetch_page(query), limit)

    def search_videos(
        self, query: str, orientation: str = "portrait", per_page: int = 15
    ) -> List[Dict]:
        return [
            {
                "url": m.url,
                "ext": ".mp4",
                "thumbnail": m.thumbnail_url,
                "title": m.title,
                "width": m.width,
                "height": m.height,
            }
            for m in self.search(query, MediaType.VIDEO, per_page)
        ]

    def download_video(self, video_url: str, output_path) -> bool:
        return self._download_file(video_url, output_path, timeout=120)
