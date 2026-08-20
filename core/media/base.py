"""
Base Media API Interface
Abstract base class for media source APIs (Pexels, Giphy, YouTube, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Optional
from pathlib import Path
import requests


class MediaType(Enum):
    """Kind of media a provider can return."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    ANY = "any"


@dataclass
class Media:
    """Unified media record returned by providers.

    Carries licensing/attribution metadata so the UI can display credits and so
    downstream code can prefer openly-licensed content. ``source`` is the raw
    provider key (e.g. ``"Openverse"``) while ``provider`` is the human label.
    """

    url: str
    title: Optional[str] = None
    creator: Optional[str] = None
    thumbnail_url: Optional[str] = None
    license: Optional[str] = None
    license_version: Optional[str] = None
    attribution: Optional[str] = None
    license_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    ext: Optional[str] = None
    provider: Optional[str] = None
    media_type: MediaType = MediaType.ANY
    source: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["media_type"] = self.media_type.value
        return d

    @classmethod
    def from_dict(cls, d: Dict, provider: Optional[str] = None) -> "Media":
        raw_mt = d.get("media_type", "any")
        if isinstance(raw_mt, MediaType):
            media_type = raw_mt
        else:
            try:
                media_type = MediaType(raw_mt)
            except ValueError:
                media_type = MediaType.ANY
        return cls(
            url=d.get("url"),
            title=d.get("title"),
            creator=d.get("creator") or d.get("uploader"),
            thumbnail_url=d.get("thumbnail_url") or d.get("thumbnail"),
            license=d.get("license"),
            license_version=d.get("license_version"),
            attribution=d.get("attribution"),
            license_url=d.get("license_url"),
            width=d.get("width"),
            height=d.get("height"),
            duration=d.get("duration"),
            ext=d.get("ext"),
            provider=provider or d.get("provider"),
            media_type=media_type,
            source=d.get("source") or provider,
        )


class BaseMediaAPI(ABC):
    """Abstract base class for media source APIs"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        self.search_cache: Dict[str, List[Dict]] = {}

    def capabilities(self) -> Dict:
        """Describe what this provider supports so the manager can route requests.

        ``requires_key`` reflects whether a usable API key is configured; key-less
        open providers return ``False`` and are always eligible. Override for
        finer control (e.g. which media types a provider can serve).
        """
        return {
            "supports_media_types": [
                MediaType.IMAGE,
                MediaType.VIDEO,
                MediaType.AUDIO,
            ],
            "requires_key": bool(self.api_key),
            "supports_license": False,
        }

    @abstractmethod
    def search_videos(
        self, query: str, orientation: str = "portrait", per_page: int = 15
    ) -> List[Dict]:
        """
        Search for videos matching the query

        Args:
            query: Search query
            orientation: Video orientation (portrait, landscape, square)
            per_page: Number of results per page

        Returns:
            List of video metadata dictionaries
        """
        pass

    @abstractmethod
    def download_video(self, video_url: str, output_path: Path) -> bool:
        """
        Download a video from URL to output path

        Args:
            video_url: URL of the video to download
            output_path: Path where video should be saved

        Returns:
            True if download successful, False otherwise
        """
        pass

    def _download_file(self, url: str, output_path: Path, timeout: int = 30) -> bool:
        """
        Generic file download helper

        Args:
            url: URL to download from
            output_path: Path to save file
            timeout: Request timeout in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            # Write to temporary file first
            temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Move to final location
            temp_path.replace(output_path)
            return output_path.exists()

        except Exception as e:
            print(f"[{self.__class__.__name__}] Download failed: {e}")
            # Clean up temp file if it exists
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    pass
            return False
