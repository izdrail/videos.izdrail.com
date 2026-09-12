"""
Offline tests for the key-less providers added 2026-09 (NASA, Library of
Congress, Mixkit). All HTTP is mocked; no network access required.
"""

import pytest

from core.media.base import MediaType
from core.media.nasa import NASAProvider
from core.media.library_of_congress import LibraryOfCongressProvider
from core.media.mixkit import MixkitProvider


class FakeResp:
    def __init__(self, payload=None, text="", status=200):
        self._payload = payload
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    """Routes GETs by URL substring to canned payloads."""

    def __init__(self, routes):
        self.routes = routes
        self.headers = {}
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        for needle, resp in self.routes.items():
            if needle in url:
                return resp
        return FakeResp(status=404)


# --------------------------------------------------------------------- NASA
def test_nasa_capabilities_keyless():
    caps = NASAProvider().capabilities()
    assert caps["requires_key"] is False
    assert MediaType.VIDEO in caps["supports_media_types"]


def test_nasa_search_videos_two_stage():
    search_payload = {
        "collection": {
            "items": [
                {
                    "href": "https://images-assets.nasa.gov/video/abc/collection.json",
                    "data": [{"title": "Earth from orbit", "center": "JSC"}],
                    "links": [{"href": "https://images-assets.nasa.gov/video/abc/abc~thumb.jpg"}],
                }
            ]
        }
    }
    manifest_payload = [
        "https://images-assets.nasa.gov/video/abc/abc~orig.mp4",
        "https://images-assets.nasa.gov/video/abc/abc~small.mp4",
    ]
    nasa = NASAProvider()
    nasa.session = FakeSession(
        {
            "images-api.nasa.gov": FakeResp(search_payload),
            "collection.json": FakeResp(manifest_payload),
        }
    )
    vids = nasa.search_videos("earth", per_page=5)
    assert len(vids) == 1
    assert vids[0]["url"] == "https://images-assets.nasa.gov/video/abc/abc~orig.mp4"
    assert vids[0]["ext"] == ".mp4"
    assert vids[0]["title"] == "Earth from orbit"
    assert vids[0]["thumbnail"].endswith("~thumb.jpg")


def test_nasa_search_failure_returns_empty():
    nasa = NASAProvider()
    nasa.session = FakeSession({})
    assert nasa.search_videos("anything") == []


# -------------------------------------------------------- Library of Congress
def test_loc_capabilities_keyless():
    caps = LibraryOfCongressProvider().capabilities()
    assert caps["requires_key"] is False


def test_loc_search_videos_parses_resources():
    payload = {
        "results": [
            {
                "id": "https://www.loc.gov/item/123/",
                "title": "Historic newsreel",
                "rights": "No known restrictions",
                "image_url": ["https://tile.loc.gov/img/small.jpg", "https://tile.loc.gov/img/large.jpg"],
                "resources": [
                    {
                        "files": [
                            [
                                {"url": "https://tile.loc.gov/vid/clip.mp4", "mimetype": "video/mp4", "width": 1920, "height": 1080},
                                {"url": "https://tile.loc.gov/vid/clip.jpg", "mimetype": "image/jpeg"},
                            ]
                        ]
                    }
                ],
            },
            {"id": "https://www.loc.gov/item/456/", "title": "No video here", "resources": []},
        ]
    }
    loc = LibraryOfCongressProvider()
    loc.session = FakeSession({"loc.gov/search": FakeResp(payload)})
    vids = loc.search_videos("history", per_page=5)
    assert len(vids) == 1
    assert vids[0]["url"] == "https://tile.loc.gov/vid/clip.mp4"
    assert vids[0]["width"] == 1920
    assert vids[0]["thumbnail"] == "https://tile.loc.gov/img/large.jpg"


# -------------------------------------------------------------------- Mixkit
MIXKIT_HTML = """
<html><body>
<video src="https://assets.mixkit.co/videos/2213/2213-360.mp4" playsinline loop muted preload="none" class="item-grid-video-player__video"></video>
<span class="item-grid-video-player__overlay-video-title">
  Waterfall in forest
</span>
<video src="https://assets.mixkit.co/videos/4075/4075-360.mp4" class="item-grid-video-player__video"></video>
<span class="item-grid-video-player__overlay-video-title"> Countryside meadow </span>
</body></html>
"""


def test_mixkit_capabilities_keyless_video_only():
    caps = MixkitProvider().capabilities()
    assert caps["requires_key"] is False
    assert caps["supports_media_types"] == [MediaType.VIDEO]


def test_mixkit_parse_upgrades_to_1080():
    mk = MixkitProvider()
    mk.session = FakeSession({"mixkit.co/free-stock-video": FakeResp(text=MIXKIT_HTML)})
    vids = mk.search_videos("waterfall", per_page=5)
    assert len(vids) == 2
    assert vids[0]["url"] == "https://assets.mixkit.co/videos/2213/2213-1080.mp4"
    assert vids[0]["title"] == "Waterfall in forest"
    assert vids[0]["thumbnail"] == "https://assets.mixkit.co/videos/2213/2213-thumb-720-0.jpg"
    assert vids[0]["ext"] == ".mp4"


def test_mixkit_http_error_returns_empty():
    mk = MixkitProvider()
    mk.session = FakeSession({})
    assert mk.search_videos("nothing") == []


# ------------------------------------------------------- manager registration
def test_manager_registers_new_providers():
    pytest.importorskip("spacy")
    from core.media.manager import MediaManager

    mgr = MediaManager(config=None)
    for name in ("NASA", "LibraryOfCongress", "Mixkit"):
        assert name in mgr.apis
        assert name in mgr.preferred_order
