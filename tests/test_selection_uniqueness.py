import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.media.identity import canonical_url, candidate_identity, path_identity, unique_slide_assets
from core.nlp.keyword_extractor import KeywordExtractor


def test_canonical_url_removes_fragment_and_tracking_but_keeps_asset_query():
    a = canonical_url("HTTPS://CDN.EXAMPLE/video.mp4?id=42&utm_source=x#t=4")
    b = canonical_url("https://cdn.example/video.mp4?id=42")
    assert a == b
    assert candidate_identity({"url": a}, "SNN") == candidate_identity({"url": b}, "Other")


def test_provider_id_is_strong_identity():
    assert candidate_identity({"id": 7, "url": "https://a/x"}, "Pexels") == candidate_identity({"id": "7", "url": "https://b/y"}, "pexels")


def test_unique_slide_assets_collapses_symlink_alias(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    alias = tmp_path / "alias.mp4"
    alias.symlink_to(video)
    result, removed = unique_slide_assets({1: video, 2: alias, 3: None})
    assert result == {1: video, 3: None}
    assert removed == 1
    assert path_identity(video) == path_identity(alias)


def test_unique_slide_assets_small_and_empty_pools():
    assert unique_slide_assets({}) == ({}, 0)
    only = Path("/missing/a.mp4")
    assert unique_slide_assets({1: only, 2: only}) == ({1: only}, 1)


def test_snn_beam_rejects_exact_duplicate_without_embeddings(monkeypatch):
    extractor = KeywordExtractor.__new__(KeywordExtractor)
    extractor.used_keywords = set()
    extractor.used_embeddings = []
    extractor.semantic_threshold = 0.8
    monkeypatch.setattr(extractor, "_embedding", lambda value: None)
    monkeypatch.setattr(extractor, "_keyword_engagement", lambda *args: 1.0)
    result = extractor.optimize_keyword_sequence(
        ["one", "two"], {0: ["city"], 1: ["city", "forest"]}, use_snn=True
    )
    assert result == {0: "city", 1: "forest"}


def test_final_dedup_is_request_local_under_concurrency(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"x")
    def select():
        return unique_slide_assets({1: video, 2: video})
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: select(), range(8)))
    assert all(value == ({1: video}, 1) for value in outputs)
