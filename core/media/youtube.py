"""
YouTube API Client
Provides access to YouTube video search/download via yt-dlp (no API key needed).

yt-dlp must be installed and available on PATH. Search/metadata is obtained by
invoking yt-dlp as a subprocess; downloads are performed the same way. All errors
(missing binary, no network) are handled gracefully and return empty results.
"""

import json
import os
import subprocess
from typing import List, Dict, Optional
from pathlib import Path
from .base import BaseMediaAPI, Media, MediaType


class YouTubeAPI(BaseMediaAPI):
    """YouTube search/download via the yt-dlp CLI."""

    BAD_KEYWORDS = [
        "lyrics",
        "official video",
        "music video",
        "interview",
        "commentary",
        "podcast",
        "vlog",
        "review",
        "reaction",
        "shorts",
        "live",
        "stream",
        "tutorial",
        "how to",
        "gameplay",
        "walkthrough",
        "let's play",
    ]

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)

    # -- helpers ---------------------------------------------------------
    def _yt_dlp_available(self) -> bool:
        from shutil import which

        return which("yt-dlp") is not None

    def _run(self, cmd: List[str], timeout: int = 30):
        """Run yt-dlp; return decoded stdout or None on any failure."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode == 0:
                return result.stdout
        except FileNotFoundError:
            print("[YouTubeAPI] yt-dlp not found on PATH; skipping.")
        except subprocess.TimeoutExpired:
            print("[YouTubeAPI] yt-dlp timed out")
        except Exception as e:
            print(f"[YouTubeAPI] yt-dlp error: {e}")
        return None

    # -- public API ------------------------------------------------------
    def search(
        self,
        query: str,
        media_type: MediaType = MediaType.ANY,
        limit: int = 10,
    ) -> List[Media]:
        cache_key = f"yt_{query}_{media_type.value}_{limit}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        if not self._yt_dlp_available():
            self.search_cache[cache_key] = []
            return []

        clean_query = query
        for bk in self.BAD_KEYWORDS:
            clean_query += f' -"{bk}"'

        cmd = [
            "yt-dlp",
            f"ytsearch{limit}:{clean_query}",
            "--flat-playlist",
            "--dump-json",
            "--no-download",
            "--no-warnings",
            "--quiet",
            "--extractor-args",
            "youtubetab:approximate_date",
        ]
        out = self._run(cmd, timeout=30)
        results: List[Media] = []
        if out:
            for line in out.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                video_id = entry.get("id") or entry.get("url")
                title = entry.get("title") or ""
                if not video_id or not title:
                    continue
                if any(bk in title.lower() for bk in self.BAD_KEYWORDS):
                    continue
                # Skip ultra-short clips for background footage.
                if isinstance(entry.get("duration"), int) and entry["duration"] < 15:
                    continue

                results.append(
                    Media(
                        url=entry.get("webpage_url")
                        or f"https://www.youtube.com/watch?v={video_id}",
                        title=title,
                        creator=entry.get("uploader") or entry.get("channel"),
                        thumbnail_url=entry.get("thumbnail"),
                        license=None,
                        license_version=None,
                        attribution=entry.get("uploader"),
                        license_url=None,
                        width=entry.get("width"),
                        height=entry.get("height"),
                        duration=entry.get("duration"),
                        ext=".mp4",
                        provider="YouTube",
                        media_type=MediaType.VIDEO,
                        source="YouTube",
                    )
                )
                if len(results) >= limit:
                    break

        self.search_cache[cache_key] = results
        return results

    def search_videos(
        self, query: str, orientation: str = "portrait", per_page: int = 5
    ) -> List[Dict]:
        media = self.search(query, media_type=MediaType.VIDEO, limit=per_page)
        return [m.to_dict() for m in media]

    def download_video(self, video_url: str, output_path: Path) -> bool:
        if not self._yt_dlp_available():
            return False
        try:
            cmd = [
                "yt-dlp",
                "-f",
                "worst[ext=mp4]/worst",
                "-o",
                str(output_path),
                "--no-warnings",
                "--quiet",
                "--socket-timeout",
                "15",
                "--retries",
                "2",
                video_url,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            return result.returncode == 0 and Path(output_path).exists()
        except Exception as e:
            print(f"[YouTubeAPI] Download error: {e}")
            return False

    def capabilities(self) -> Dict:
        return {
            "supports_media_types": [MediaType.VIDEO, MediaType.AUDIO],
            "requires_key": False,
            "supports_license": False,
        }
