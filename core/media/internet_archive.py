"""
Internet Archive API Client
Searches the Internet Archive (archive.org) for openly licensed movies, audio and
images via the advancedsearch + metadata APIs. No API key required.
"""

from typing import List, Dict, Optional
from urllib.parse import quote, urlparse

from .base import BaseMediaAPI, Media, MediaType


class InternetArchiveProvider(BaseMediaAPI):
    """Read-only discovery of Internet Archive media."""

    SEARCH = "https://archive.org/advancedsearch.php"
    META = "https://archive.org/metadata"

    VIDEO_FORMATS = {
        "MPEG4",
        "h.264",
        "Ogg Video",
        "512Kb MPEG4",
        "MPEG2",
        "DivX",
        "AVI",
        "QuickTime",
    }
    IMAGE_FORMATS = {"JPEG", "PNG", "GIF", "TIFF", "BMP", "Item Image"}

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _ext(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        p = name.lower()
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
            ".avi",
            ".mov",
            ".mpeg",
            ".mp3",
            ".ogg",
            ".wav",
        ):
            if p.endswith(ext):
                return ext
        return None

    @staticmethod
    def _classify(mediatype: Optional[str], name: Optional[str]) -> MediaType:
        mt = (mediatype or "").lower()
        if mt == "movies" or mt == "video":
            return MediaType.VIDEO
        if mt == "audio":
            return MediaType.AUDIO
        if mt == "image":
            return MediaType.IMAGE
        ext = InternetArchiveProvider._ext(name) or ""
        if ext in (".mp4", ".webm", ".ogv", ".avi", ".mov", ".mpeg"):
            return MediaType.VIDEO
        if ext in (".mp3", ".ogg", ".wav"):
            return MediaType.AUDIO
        if ext:
            return MediaType.IMAGE
        return MediaType.ANY

    def _pick_file(self, files: List[Dict], media_type: MediaType) -> Optional[Dict]:
        """Choose a downloadable file appropriate for the requested media type."""
        preferred = (
            self.VIDEO_FORMATS
            if media_type == MediaType.VIDEO
            else self.IMAGE_FORMATS
            if media_type == MediaType.IMAGE
            else self.VIDEO_FORMATS | self.IMAGE_FORMATS
        )
        candidates = []
        for f in files:
            fmt = f.get("format", "")
            if fmt in preferred:
                candidates.append(f)
        if not candidates:
            # Fall back to any file with a recognised media extension.
            for f in files:
                if self._ext(f.get("name")):
                    candidates.append(f)
        if not candidates:
            return None

        # Prefer the largest by reported size (proxy for quality).
        def _size(f: Dict) -> int:
            try:
                return int(f.get("size", 0))
            except (TypeError, ValueError):
                return 0

        return max(candidates, key=_size)

    # -- public API ------------------------------------------------------
    def search(
        self,
        query: str,
        media_type: MediaType = MediaType.ANY,
        limit: int = 20,
    ) -> List[Media]:
        cache_key = f"ia_{query}_{media_type.value}_{limit}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        # Map requested type to an archive.org mediatype filter when possible.
        ia_type = ""
        if media_type == MediaType.VIDEO:
            ia_type = "movies"
        elif media_type == MediaType.AUDIO:
            ia_type = "audio"
        elif media_type == MediaType.IMAGE:
            ia_type = "image"

        q = query
        if ia_type:
            q = f"({query}) AND mediatype:{ia_type}"

        try:
            resp = self.session.get(
                self.SEARCH,
                params={
                    "q": q,
                    "fl[]": "identifier,title,mediatype,creator,license,date",
                    "rows": limit,
                    "page": 1,
                    "output": "json",
                },
                timeout=20,
            )
            resp.raise_for_status()
            docs = resp.json().get("response", {}).get("docs", [])
        except Exception as e:
            print(f"[InternetArchive] Search error: {e}")
            self.search_cache[cache_key] = []
            return []

        results: List[Media] = []
        for doc in docs:
            identifier = doc.get("identifier")
            if not identifier:
                continue
            media = self._build_media(identifier, doc, media_type)
            if media:
                results.append(media)
            if len(results) >= limit:
                break

        self.search_cache[cache_key] = results
        return results

    def _build_media(
        self, identifier: str, doc: Dict, media_type: MediaType
    ) -> Optional[Media]:
        """Fetch item metadata to resolve a concrete downloadable file."""
        try:
            meta = self.session.get(
                f"{self.META}/{quote(identifier)}", timeout=20
            ).json()
        except Exception:
            return None

        files = meta.get("files", []) or []
        file_doc = self._pick_file(files, media_type)
        if not file_doc:
            return None
        name = file_doc.get("name")
        if not name:
            return None

        # Use the doc's mediatype (more reliable than per-file) for classification
        # unless we explicitly requested audio (archive items are often mixed).
        mt = self._classify(doc.get("mediatype"), name)
        if media_type != MediaType.ANY and mt != media_type:
            return None

        base = f"https://archive.org/download/{quote(identifier)}"
        url = f"{base}/{quote(name)}"
        thumb = f"{base}/{quote(identifier)}.jpg"

        meta_md = meta.get("metadata", {}) or {}
        license_val = meta_md.get("license") or doc.get("license")
        creator = meta_md.get("creator") or doc.get("creator") or "Internet Archive"
        if isinstance(creator, list):
            creator = ", ".join(creator)

        return Media(
            url=url,
            title=meta_md.get("title") or doc.get("title") or identifier,
            creator=creator,
            thumbnail_url=thumb,
            license=license_val,
            license_version=None,
            attribution=creator,
            license_url=None,
            width=None,
            height=None,
            duration=None,
            ext=self._ext(name) or ".mp4",
            provider="Internet Archive",
            media_type=mt,
            source="InternetArchive",
        )

    def search_videos(
        self, query: str, orientation: str = "portrait", per_page: int = 15
    ) -> List[Dict]:
        # Background footage wants visuals: prefer video, allow image, drop audio.
        media = self.search(query, media_type=MediaType.VIDEO, limit=per_page)
        if len(media) < per_page:
            extra = self.search(
                query, media_type=MediaType.IMAGE, limit=per_page - len(media)
            )
            media = media + extra
        return [m.to_dict() for m in media if m.media_type != MediaType.AUDIO]

    def download_video(self, video_url: str, output_path) -> bool:
        return self._download_file(video_url, output_path)

    def capabilities(self) -> Dict:
        return {
            "supports_media_types": [
                MediaType.IMAGE,
                MediaType.VIDEO,
                MediaType.AUDIO,
            ],
            "requires_key": False,
            "supports_license": True,
        }
