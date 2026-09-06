"""
Unit tests for media_scoring.py pipeline
"""

import time
import pytest
from PIL import Image
from pathlib import Path

from media_scoring import (
    clip_relevance_score,
    compute_quality_score,
    rerank_pooled_candidates,
)
from core.media.pexels import PexelsAPI
from core.media.manager import MediaManager


def test_clip_relevance_score_fallback():
    """Test fallback behavior on invalid or empty thumbnail URL/inputs."""
    # Empty text -> 0.5
    assert clip_relevance_score("", "https://example.com/test.jpg") == 0.5

    # Non-existent image URL -> 0.5
    assert clip_relevance_score("a cat running on grass", "https://invalid-domain-xyz.com/image.jpg") == 0.5

    # None thumbnail -> 0.5
    assert clip_relevance_score("a dog", None) == 0.5


def test_clip_relevance_score_with_pil_image():
    """Test CLIP scoring with direct PIL Image input."""
    img = Image.new("RGB", (224, 224), color=(0, 128, 255))
    score = clip_relevance_score("a blue sky or ocean", img)
    assert 0.0 <= score <= 1.0


def test_compute_quality_score_aspect_orientation():
    """Test resolution and orientation penalty in compute_quality_score."""
    # Proper portrait clip (1080x1920) matching target (1080x1920)
    portrait_clip = {"width": 1080, "height": 1920}
    portrait_score = compute_quality_score(portrait_clip, target_w=1080, target_h=1920)

    # 4K Landscape clip (3840x2160) - higher resolution but wrong orientation
    landscape_4k_clip = {"width": 3840, "height": 2160}
    landscape_score = compute_quality_score(landscape_4k_clip, target_w=1080, target_h=1920)

    # Orientation penalty must ensure portrait clip outranks landscape clip
    assert portrait_score > landscape_score
    assert 0.0 <= portrait_score <= 1.0
    assert 0.0 <= landscape_score <= 1.0


def test_compute_quality_score_invalid_metadata():
    """Test fallback when media metadata is missing width/height."""
    assert compute_quality_score({}, target_w=1080, target_h=1920) == 0.5
    assert compute_quality_score({"width": 0, "height": 0}) == 0.5


def test_rerank_pooled_candidates_basic():
    """Test pooling candidates across sources and scoring with rerank_pooled_candidates."""
    now = time.time()
    candidates_by_source = {
        "Pexels": [
            {"url": "http://pexels.com/v1.mp4", "width": 1080, "height": 1920, "id": 1},
            {"url": "http://pexels.com/v2.mp4", "width": 720, "height": 1280, "id": 2},
        ],
        "Pixabay": [
            {"url": "http://pixabay.com/v3.mp4", "width": 1920, "height": 1080, "id": 3},
            {"url": "http://pixabay.com/v4.mp4", "width": 1080, "height": 1920, "id": 4},
        ],
    }

    # Preferred source boost for Pexels
    results = rerank_pooled_candidates(
        narration_text="a portrait nature background video",
        candidates_by_source=candidates_by_source,
        used_urls=set(),
        preferred_source="Pexels",
        now=now,
        top_k=2,
    )

    assert len(results) == 2
    for item in results:
        assert "_source" in item
        assert "_score" in item
        assert isinstance(item["_score"], float)

    # First item should have _source field attached
    assert results[0]["_source"] in ("Pexels", "Pixabay")


def test_rerank_pooled_candidates_excludes_used_urls():
    """Test used_urls exclusion in rerank_pooled_candidates."""
    used = {"http://pexels.com/used.mp4"}
    candidates_by_source = {
        "Pexels": [
            {"url": "http://pexels.com/used.mp4", "width": 1080, "height": 1920},
            {"url": "http://pexels.com/fresh.mp4", "width": 1080, "height": 1920},
        ]
    }

    results = rerank_pooled_candidates(
        narration_text="forest trees",
        candidates_by_source=candidates_by_source,
        used_urls=used,
        top_k=5,
    )

    assert len(results) == 1
    assert results[0]["url"] == "http://pexels.com/fresh.mp4"


def test_pexels_api_target_resolution_selection():
    """Test PexelsAPI selects video closest to target resolution (1080x1920)."""
    pexels = PexelsAPI(api_key="mock_key")

    # Mock response data with multiple resolution files
    mock_data = {
        "videos": [
            {
                "id": 101,
                "duration": 15,
                "image": "http://pexels.com/thumb.jpg",
                "video_files": [
                    {"link": "http://pexels.com/small.mp4", "width": 320, "height": 568},
                    {"link": "http://pexels.com/target_hd.mp4", "width": 1080, "height": 1920},
                    {"link": "http://pexels.com/4k.mp4", "width": 2160, "height": 3840},
                ],
            }
        ]
    }

    # Intercept session get
    class MockResponse:
        def raise_for_status(self): pass
        def json(self): return mock_data

    pexels.session.get = lambda url, params, timeout: MockResponse()

    results = pexels.search_videos("nature", target_width=1080, target_height=1920)
    assert len(results) == 1
    # Must pick target_hd (1080x1920) file, NOT small (320x568)
    assert results[0]["url"] == "http://pexels.com/target_hd.mp4"
    assert results[0]["width"] == 1080
    assert results[0]["height"] == 1920
