"""
Unit tests for the Quality & Selection Logic Overhaul.

These tests cover:
  * Point 1: substring-matching bug in rank_keywords (word boundaries)
  * Point 2: beam-search sequence optimizer (complete, non-crashing)
  * Point 6: availability fallback in MediaManager.is_keyword_available

The modules under test import heavy optional dependencies (spacy, torch, ...).
When those are unavailable the relevant tests are skipped so the suite can still
collect in a minimal environment.
"""

import pytest

np = pytest.importorskip("numpy")
spacy = pytest.importorskip("spacy")

from core.nlp.keyword_extractor import KeywordExtractor  # noqa: E402
from core.media.manager import MediaManager  # noqa: E402


# ---------------------------------------------------------------------------
# Point 1 — substring matching
# ---------------------------------------------------------------------------


def test_rank_keywords_does_not_match_substring():
    """'day' as a stock category must NOT boost 'holiday' / 'birthday'."""
    ke = KeywordExtractor()
    ke.used_keywords.clear()
    ranked = ke.rank_keywords(["holiday", "sunset"])
    # "sunset" matches the stock category exactly; "holiday" no longer inherits
    # the spurious "day" boost, so it must rank first.
    assert ranked[0] == "sunset"


def test_rank_keywords_word_boundary_for_actions():
    """'running' must not be matched by the 'run' visual action."""
    ke = KeywordExtractor()
    ke.used_keywords.clear()
    # The visual_actions set contains "running", so this also guards the
    # word-boundary match used for action detection.
    ranked = ke.rank_keywords(["surrounding", "running"])
    assert ranked[0] == "running"


# ---------------------------------------------------------------------------
# Point 2 — beam search
# ---------------------------------------------------------------------------


def test_optimize_keyword_sequence_returns_complete_mapping():
    """Beam search returns one (possibly None) keyword per sentence index."""
    ke = KeywordExtractor()
    # Neutralise neural + embedding calls so the test is deterministic & cheap.
    ke._keyword_engagement = lambda ctx, kw, use_snn=False: float(len(kw))
    ke._keyword_similarity = lambda a, b: 0.0
    ke._embedding = lambda kw: None  # _unique_against becomes a pure exact-check
    ke.used_keywords.clear()

    candidates_map = {
        0: ["cat", "dog"],
        1: ["dog", "fish"],
        2: ["fish", "cat"],
    }
    chosen = ke.optimize_keyword_sequence(["a", "b", "c"], candidates_map, beam_width=4)
    assert isinstance(chosen, dict)
    assert set(chosen.keys()) == {0, 1, 2}
    for idx, kw in chosen.items():
        assert kw is None or kw in candidates_map[idx]


def test_optimize_keyword_sequence_prefers_higher_engagement():
    """With a deterministic engagement signal the highest-scoring candidate wins."""
    ke = KeywordExtractor()
    ke._keyword_engagement = lambda ctx, kw, use_snn=False: {
        "good": 1.0,
        "bad": 0.1,
    }.get(kw, 0.5)
    ke._keyword_similarity = lambda a, b: 0.0
    ke._embedding = lambda kw: None
    ke.used_keywords.clear()

    candidates_map = {0: ["bad", "good"]}
    chosen = ke.optimize_keyword_sequence(["x"], candidates_map, beam_width=4)
    assert chosen[0] == "good"


# ---------------------------------------------------------------------------
# Point 6 — availability fallback (mocked media APIs)
# ---------------------------------------------------------------------------


class _FakeAPI:
    def __init__(self, results):
        self._results = results

    def search_videos(self, query, per_page=10):
        return self._results


def test_is_keyword_available_true_when_source_has_results():
    mm = MediaManager(config=None)
    fake = _FakeAPI([{"url": "http://x/1.mp4"}])
    mm.apis = {"Fake": fake}
    mm.preferred_order = ["Fake"]
    assert mm.is_keyword_available("river") is True


def test_is_keyword_available_false_when_no_results():
    mm = MediaManager(config=None)
    fake = _FakeAPI([])
    mm.apis = {"Fake": fake}
    mm.preferred_order = ["Fake"]
    assert mm.is_keyword_available("nonexistent") is False
