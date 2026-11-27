"""CLIPScorer must never read an unbounded response body."""
import sys, types
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))


def test_http_load_uses_bounded_fetcher_not_requests_get():
    identity = types.ModuleType("core.media.identity")
    identity.canonical_url = lambda x: x
    identity.candidate_identity = lambda *a: None
    with mock.patch.dict(sys.modules, {"core.media.identity": identity}):
        import media_scoring
        from core.visual.clip_scorer import CLIPScorer

        sentinel = object()
        with mock.patch.object(
            media_scoring, "_fetch_thumbnail_image", return_value=sentinel
        ) as fetch, mock.patch(
            "requests.get", side_effect=AssertionError("unbounded requests.get")
        ):
            assert CLIPScorer().load_image("https://example.test/v.mp4") is sentinel
            fetch.assert_called_once_with("https://example.test/v.mp4")
