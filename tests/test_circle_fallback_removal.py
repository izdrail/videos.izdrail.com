"""Regression coverage for removal of bundled circle fallback videos."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from main import FFmpegVideoGenerator


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_circle_fallback_assets_are_removed():
    assert not (ROOT / "circle_overlays").exists()
    for name in ("girl-1.mp4", "guy-talking.mp4", "ranger-talking.mp4"):
        assert not any(ROOT.rglob(name))


def test_no_visual_returns_explicit_empty_state():
    generator = FFmpegVideoGenerator.__new__(FFmpegVideoGenerator)
    generator.keyword_extractor = SimpleNamespace(
        sanitize_keyword=lambda value: value,
        enrich_keyword_context=lambda keyword, sentence: keyword,
        generate_fallback_keywords=lambda keyword: [],
    )
    generator.media_manager = MagicMock()
    generator.media_manager.get_random_media.return_value = (None, None)
    generator.sd_manager = None

    assert generator.get_background_video("missing", "No matching footage") is None
