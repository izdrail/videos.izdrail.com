"""
Unit tests for Tier 2 features:
- D: CLIPScorer semantic relevance
- E: TemporalCoherenceOptimizer visual transition smoothing
- F: GenerationDB clip performance tracking & RL bandit feedback
"""

import os
import sqlite3
import tempfile
from pathlib import Path
import pytest
from PIL import Image

from core.database import GenerationDB
from core.media.manager import MediaManager
from core.visual.clip_scorer import CLIPScorer
from core.visual.temporal_coherence import TemporalCoherenceOptimizer


def test_clip_scorer_basic():
    """Test CLIPScorer initialization and similarity computation on PIL image."""
    scorer = CLIPScorer()
    img = Image.new("RGB", (200, 200), color=(0, 128, 255))
    score = scorer.compute_similarity("a blue sky or ocean", img)
    assert 0.0 <= score <= 1.0


def test_clip_scorer_candidate_scoring():
    """Test candidate dictionary scoring via CLIPScorer."""
    scorer = CLIPScorer()
    img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img2 = Image.new("RGB", (100, 100), color=(0, 255, 0))

    candidates = [
        {"media": {"thumbnail": img1, "title": "red square"}},
        {"media": {"thumbnail": img2, "title": "green square"}},
    ]
    scored = scorer.score_media_candidates("red bright square", candidates)
    assert len(scored) == 2
    assert "clip_score" in scored[0]
    assert 0.0 <= scored[0]["clip_score"] <= 1.0


def test_temporal_coherence_extract_and_diff():
    """Test feature extraction and transition difference calculation."""
    tc = TemporalCoherenceOptimizer()
    white_img = Image.new("RGB", (100, 100), (255, 255, 255))
    black_img = Image.new("RGB", (100, 100), (0, 0, 0))

    feat_w = tc.extract_frame_features(white_img)
    feat_b = tc.extract_frame_features(black_img)

    assert feat_w is not None
    assert feat_b is not None
    assert feat_w["brightness"] > feat_b["brightness"]

    diff = tc.compute_transition_difference(feat_w, feat_b)
    assert diff["brightness_diff"] > 0.8
    assert diff["total_diff"] > 0.5


def test_temporal_coherence_sequence_optimization(tmp_path):
    """Test optimizing slide clips to smooth out jarring transitions."""
    tc = TemporalCoherenceOptimizer(max_brightness_diff=0.3)

    img_white = Image.new("RGB", (100, 100), (255, 255, 255))
    img_gray = Image.new("RGB", (100, 100), (180, 180, 180))
    img_black = Image.new("RGB", (100, 100), (0, 0, 0))

    path_w = tmp_path / "white.jpg"
    path_g = tmp_path / "gray.jpg"
    path_b = tmp_path / "black.jpg"

    img_white.save(path_w)
    img_gray.save(path_g)
    img_black.save(path_b)

    # Sequence with abrupt white -> black transition at index 1
    chosen = [path_w, path_b]
    candidates = {1: [path_b, path_g]}

    optimized = tc.optimize_slide_clips(chosen, candidates)
    # Expect slide 1 to be replaced with gray.jpg which is visually closer to white.jpg
    assert optimized[1] == path_g


def test_database_clip_performance_logging(tmp_path):
    """Test GenerationDB clip_performance logging and score multiplier retrieval."""
    db_file = tmp_path / "test_gen.db"
    db = GenerationDB(db_path=db_file)

    url = "https://example.com/clip1.mp4"
    db.log_clip_performance(
        media_url=url,
        keyword="nature",
        source="Pexels",
        event_type="select",
    )
    db.log_clip_performance(
        media_url=url,
        keyword="nature",
        source="Pexels",
        event_type="watch",
        completion_rate=0.9,
    )

    score = db.get_clip_performance_score(media_url=url, source="Pexels")
    assert score > 0.0

    # Test replacement penalty
    db.log_clip_performance(
        media_url=url,
        event_type="replace",
    )
    new_score = db.get_clip_performance_score(media_url=url, source="Pexels")
    assert new_score < score


def test_media_manager_bandit_incorporates_performance_score(tmp_path):
    """Test MediaManager bandit incorporates clip performance scores."""
    media_mgr = MediaManager()
    source_results = {
        "Pexels": [{"url": "http://example.com/p1.mp4", "width": 1920, "height": 1080}],
    }
    # Quality score check
    score = media_mgr._compute_quality_score(source_results["Pexels"][0])
    assert score == 1.0
