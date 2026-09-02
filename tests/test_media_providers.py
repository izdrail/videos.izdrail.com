"""
Unit tests for the open media providers and MediaManager unified search.

Network-dependent paths are exercised with mocked HTTP/CLI layers so the suite
runs offline. The MediaManager test imports spacy (via neuron_extractor) and is
skipped automatically when that dependency is unavailable.
"""

import json

import pytest
import requests
from unittest.mock import MagicMock, patch

from core.media.base import Media, MediaType


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


def _get(provider, response):
    provider.session.get = MagicMock(return_value=response)


# ---------------------------------------------------------------------------
# Openverse
# ---------------------------------------------------------------------------


def test_openverse_images():
    from core.media.openverse import OpenverseProvider
    from core.media.base import MediaType

    p = OpenverseProvider()
    _get(
        p,
        FakeResponse(
            {
                "results": [
                    {
                        "id": "1",
                        "title": "Sunset",
                        "creator": "Alice",
                        "url": "https://openverse.org/sunset.jpg",
                        "thumbnail": "https://openverse.org/sunset_t.jpg",
                        "license": "cc-by",
                        "license_version": "4.0",
                        "attribution": "Alice",
                        "license_url": "https://lic",
                        "width": 1000,
                        "height": 800,
                    }
                ]
            }
        ),
    )
    media = p.search("sunset", MediaType.IMAGE, limit=5)
    assert len(media) == 1
    m = media[0]
    assert m.url.endswith(".jpg")
    assert m.license == "cc-by"
    assert m.license_version == "4.0"
    assert m.media_type == MediaType.IMAGE
    # Manager-compatible dicts
    as_dicts = p.search_videos("sunset")
    assert as_dicts and "url" in as_dicts[0]


def test_openverse_auth_required_returns_empty():
    from core.media.openverse import OpenverseProvider

    p = OpenverseProvider()
    _get(p, FakeResponse({}, status_code=401))
    assert p.search("sunset") == []


# ---------------------------------------------------------------------------
# Wikimedia Commons
# ---------------------------------------------------------------------------


def test_wikimedia_video_and_license():
    from core.media.wikimedia import WikimediaProvider
    from core.media.base import MediaType

    p = WikimediaProvider()

    def fake_get(url, params=None, timeout=20):
        params = params or {}
        if params.get("list") == "search":
            return FakeResponse(
                {"query": {"search": [{"title": "File:Eiffel Tower.webm"}]}}
            )
        if params.get("prop") == "imageinfo":
            return FakeResponse(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "File:Eiffel Tower.webm",
                                "imageinfo": [
                                    {
                                        "url": "https://upload.wikimedia.org/x/Eiffel.webm",
                                        "thumburl": "https://upload.wikimedia.org/x/Eiffel.jpg",
                                        "width": 1920,
                                        "height": 1080,
                                        "mime": "video/webm",
                                        "extmetadata": {
                                            "LicenseShortName": {
                                                "value": "CC BY-SA 4.0"
                                            },
                                            "Artist": {"value": "Photographer"},
                                            "LicenseUrl": {"value": "https://lic"},
                                        },
                                    }
                                ],
                            }
                        }
                    }
                }
            )
        return FakeResponse({})

    p.session.get = MagicMock(side_effect=fake_get)
    media = p.search("Eiffel Tower", MediaType.VIDEO, limit=5)
    assert media, "expected at least one Wikimedia result"
    m = media[0]
    assert m.url.endswith(".webm")
    assert m.license == "CC BY-SA 4.0"
    assert m.creator == "Photographer"
    assert m.media_type == MediaType.VIDEO


# ---------------------------------------------------------------------------
# Internet Archive
# ---------------------------------------------------------------------------


def test_internet_archive_movies():
    from core.media.internet_archive import InternetArchiveProvider
    from core.media.base import MediaType

    p = InternetArchiveProvider()

    def fake_get(url, params=None, timeout=20):
        if "advancedsearch.php" in url:
            return FakeResponse(
                {
                    "response": {
                        "docs": [
                            {
                                "identifier": "space_shuttle_01",
                                "title": "Shuttle",
                                "mediatype": "movies",
                                "creator": "NASA",
                            }
                        ]
                    }
                }
            )
        if "/metadata/" in url:
            return FakeResponse(
                {
                    "metadata": {"title": "Shuttle", "creator": "NASA"},
                    "files": [
                        {"name": "shuttle.mp4", "format": "MPEG4", "size": 12345}
                    ],
                }
            )
        return FakeResponse({})

    p.session.get = MagicMock(side_effect=fake_get)
    media = p.search("space shuttle", MediaType.VIDEO, limit=5)
    assert media, "expected at least one Internet Archive result"
    m = media[0]
    assert "archive.org/download" in m.url
    assert m.ext == ".mp4"
    assert m.media_type == MediaType.VIDEO


# ---------------------------------------------------------------------------
# YouTube (yt-dlp subprocess)
# ---------------------------------------------------------------------------


def test_youtube_search_parses_json(monkeypatch):
    import subprocess
    from core.media.youtube import YouTubeAPI
    from core.media.base import MediaType

    payload = json.dumps(
        {
            "id": "abc",
            "title": "Funny Cats",
            "uploader": "CatChan",
            "duration": 30,
            "thumbnail": "https://t.jpg",
            "webpage_url": "https://www.youtube.com/watch?v=abc",
        }
    )

    class FakeRun:
        returncode = 0
        stdout = payload + "\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeRun())
    p = YouTubeAPI()
    monkeypatch.setattr(p, "_yt_dlp_available", lambda: True)

    media = p.search("funny cats", MediaType.VIDEO, limit=5)
    assert media
    assert media[0].url == "https://www.youtube.com/watch?v=abc"
    assert media[0].media_type == MediaType.VIDEO


def test_youtube_missing_binary_returns_empty(monkeypatch):
    import subprocess
    from core.media.youtube import YouTubeAPI

    def _raise(*a, **k):
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "run", _raise)
    p = YouTubeAPI()
    assert p.search("anything") == []


# ---------------------------------------------------------------------------
# MediaManager unified search (needs spacy for neuron_extractor import)
# ---------------------------------------------------------------------------


class FakeProvider:
    def __init__(self, items):
        self._items = items

    def capabilities(self):
        return {
            "supports_media_types": [MediaType.IMAGE, MediaType.VIDEO],
            "requires_key": False,
            "supports_license": True,
        }

    def search(self, query, media_type=MediaType.ANY, limit=50):
        return self._items


def test_manager_search_dedup_and_order():
    pytest.importorskip("spacy")
    from core.media.manager import MediaManager

    mm = MediaManager(config=None)
    a = Media(url="https://x/a.jpg", source="Openverse", title="A")
    dup = Media(url="https://x/a.jpg", source="Wikimedia", title="A")  # same URL
    c = Media(url="https://x/c.jpg", source="Openverse", title="C")
    mm.apis = {
        "Openverse": FakeProvider([a, c]),
        "Wikimedia": FakeProvider([dup]),
    }
    mm.preferred_order = ["Openverse", "Wikimedia"]
    res = mm.search("cat", MediaType.IMAGE, limit=50, min_results=50)
    urls = [m.url for m in res]
    # Duplicate URL collapsed to a single entry.
    assert urls.count("https://x/a.jpg") == 1
    assert len(res) == 2


def test_manager_search_skips_unsupported_type():
    pytest.importorskip("spacy")
    from core.media.manager import MediaManager

    class ImageOnlyProvider(FakeProvider):
        def capabilities(self):
            return {"supports_media_types": [MediaType.IMAGE], "requires_key": False}

    mm = MediaManager(config=None)
    img = Media(url="https://x/i.jpg", source="Openverse", media_type=MediaType.IMAGE)
    mm.apis = {"Openverse": ImageOnlyProvider([img])}
    mm.preferred_order = ["Openverse"]
    res = mm.search("cat", MediaType.VIDEO, limit=50, min_results=50)
    assert res == []  # Openverse only serves images here -> skipped for VIDEO
