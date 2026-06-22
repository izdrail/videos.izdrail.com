"""
YouTube API Client
Provides access to YouTube video downloads via yt-dlp
"""

import os
import subprocess
from typing import List, Dict, Optional
from pathlib import Path
from .base import BaseMediaAPI


class YouTubeAPI(BaseMediaAPI):
    """YouTube video search and download via yt-dlp"""

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

    def search_videos(
        self, query: str, orientation: str = "portrait", per_page: int = 5
    ) -> List[Dict]:
        cache_key = f"{query}_{per_page}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        try:
            clean_query = query
            for bk in self.BAD_KEYWORDS:
                clean_query += f' -"{bk}"'

            cmd = [
                "yt-dlp",
                f"ytsearch{per_page}:{clean_query}",
                "--flat-playlist",
                "--print",
                "%(title)s",
                "--print",
                "%(id)s",
                "--print",
                "%(duration)s",
                "--no-warnings",
                "--quiet",
                "--extractor-args",
                "youtubetab:approximate_date",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                results = []

                for i in range(0, len(lines), 3):
                    if i + 2 < len(lines):
                        title = lines[i].strip()
                        video_id = lines[i + 1].strip()
                        duration_str = lines[i + 2].strip()

                        if not video_id or not title:
                            continue

                        if any(bk in title.lower() for bk in self.BAD_KEYWORDS):
                            continue

                        try:
                            duration = int(duration_str)
                        except ValueError:
                            duration = 0

                        if duration < 15:
                            continue

                        results.append(
                            {
                                "url": f"https://www.youtube.com/watch?v={video_id}",
                                "id": video_id,
                                "title": title,
                                "duration": duration,
                            }
                        )

                self.search_cache[cache_key] = results
                return results

        except subprocess.TimeoutExpired:
            print("[YouTubeAPI] Search timed out")
        except Exception as e:
            print(f"[YouTubeAPI] Search error: {e}")

        return []

    def download_video(self, video_url: str, output_path: Path) -> bool:
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
            return result.returncode == 0 and output_path.exists()

        except Exception as e:
            print(f"[YouTubeAPI] Download error: {e}")
            return False
