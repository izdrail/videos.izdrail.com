"""
Unit tests for OllamaClient retry, caching, and fallback logic.
"""

import sys
import os
from unittest.mock import MagicMock, patch
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.nlp.ollama_client import (
    OllamaClient,
    DEFAULT_FALLBACK_KEYWORDS,
    DEFAULT_FALLBACK_MOOD,
    DEFAULT_FALLBACK_SCRIPT,
)


class TestOllamaClientRetry:
    """Tests for retry with exponential backoff."""

    @patch("core.nlp.ollama_client.time.sleep")
    @patch("core.nlp.ollama_client.requests.post")
    def test_successful_request_returns_json(self, mock_post, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "hello"}
        mock_post.return_value = mock_resp

        client = OllamaClient(
            model="test", url="http://fake/api/generate", max_retries=3
        )
        result = client.post("test prompt")

        assert result == {"response": "hello"}
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()

    @patch("core.nlp.ollama_client.time.sleep")
    @patch("core.nlp.ollama_client.requests.post")
    def test_retries_on_connection_error(self, mock_post, mock_sleep):
        from requests.exceptions import ConnectionError

        mock_post.side_effect = ConnectionError("refused")

        client = OllamaClient(
            model="test", url="http://fake/api/generate", max_retries=3, base_delay=1.0
        )
        result = client.post("test prompt")

        assert result is None
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0]

    @patch("core.nlp.ollama_client.time.sleep")
    @patch("core.nlp.ollama_client.requests.post")
    def test_retries_on_timeout(self, mock_post, mock_sleep):
        from requests.exceptions import Timeout

        mock_post.side_effect = Timeout("timed out")

        client = OllamaClient(
            model="test", url="http://fake/api/generate", max_retries=2, base_delay=0.5
        )
        result = client.post("prompt")

        assert result is None
        assert mock_post.call_count == 2
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [0.5]

    @patch("core.nlp.ollama_client.time.sleep")
    @patch("core.nlp.ollama_client.requests.post")
    def test_retries_on_server_error(self, mock_post, mock_sleep):
        mock_resp_500 = MagicMock()
        mock_resp_500.status_code = 500
        mock_resp_500.text = "Internal Server Error"

        mock_resp_ok = MagicMock()
        mock_resp_ok.status_code = 200
        mock_resp_ok.json.return_value = {"response": "recovered"}

        mock_post.side_effect = [mock_resp_500, mock_resp_ok]

        client = OllamaClient(
            model="test", url="http://fake/api/generate", max_retries=3, base_delay=0.1
        )
        result = client.post("prompt")

        assert result == {"response": "recovered"}
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(0.1)

    @patch("core.nlp.ollama_client.time.sleep")
    @patch("core.nlp.ollama_client.requests.post")
    def test_no_retry_on_client_error(self, mock_post, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"
        mock_post.return_value = mock_resp

        client = OllamaClient(
            model="test", url="http://fake/api/generate", max_retries=3
        )
        result = client.post("prompt")

        assert result is None
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()

    @patch("core.nlp.ollama_client.time.sleep")
    @patch("core.nlp.ollama_client.requests.post")
    def test_unexpected_exception_breaks_immediately(self, mock_post, mock_sleep):
        mock_post.side_effect = ValueError("unexpected")

        client = OllamaClient(
            model="test", url="http://fake/api/generate", max_retries=3
        )
        result = client.post("prompt")

        assert result is None
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()


class TestOllamaClientCaching:
    """Tests for response caching."""

    @patch("core.nlp.ollama_client.requests.post")
    def test_cache_hit_avoids_api_call(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "cached result"}
        mock_post.return_value = mock_resp

        client = OllamaClient(model="test", url="http://fake/api/generate")

        r1 = client.post("same prompt")
        r2 = client.post("same prompt")

        assert r1 == r2 == {"response": "cached result"}
        assert mock_post.call_count == 1
        assert client.stats["cache_hits"] == 1

    @patch("core.nlp.ollama_client.requests.post")
    def test_different_prompts_get_different_cache_keys(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "ok"}
        mock_post.return_value = mock_resp

        client = OllamaClient(model="test", url="http://fake/api/generate")

        client.post("prompt A")
        client.post("prompt B")

        assert mock_post.call_count == 2

    @patch("core.nlp.ollama_client.requests.post")
    def test_cache_eviction_on_max_size(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "ok"}
        mock_post.return_value = mock_resp

        client = OllamaClient(
            model="test", url="http://fake/api/generate", cache_max_size=3
        )

        for i in range(5):
            client.post(f"prompt {i}")

        assert len(client._cache) == 3
        assert mock_post.call_count == 5

    def test_clear_cache(self):
        client = OllamaClient(model="test", url="http://fake/api/generate")
        client._cache["key"] = "value"
        client.clear_cache()
        assert len(client._cache) == 0


class TestOllamaClientFallback:
    """Tests for fallback behavior."""

    def test_post_or_fallback_returns_fallback_on_failure(self):
        with patch("core.nlp.ollama_client.requests.post", side_effect=ConnectionError):
            client = OllamaClient(
                model="test", url="http://fake/api/generate", max_retries=1
            )
            result = client.post_or_fallback("prompt", fallback=["default_kws"])

            assert result == ["default_kws"]

    @patch("core.nlp.ollama_client.requests.post")
    def test_post_or_fallback_returns_result_on_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "real data"}
        mock_post.return_value = mock_resp

        client = OllamaClient(model="test", url="http://fake/api/generate")
        result = client.post_or_fallback("prompt", fallback="fallback_value")

        assert result == {"response": "real data"}


class TestOllamaClientStats:
    """Tests for stats tracking."""

    @patch("core.nlp.ollama_client.requests.post")
    def test_stats_track_calls(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "ok"}
        mock_post.return_value = mock_resp

        client = OllamaClient(model="test", url="http://fake/api/generate")
        client.post("prompt 1")
        client.post("prompt 2")
        client.post("prompt 1")  # cached

        stats = client.stats
        assert stats["total_calls"] == 2
        assert stats["cache_hits"] == 1
        assert stats["failures"] == 0

    @patch("core.nlp.ollama_client.time.sleep")
    @patch("core.nlp.ollama_client.requests.post")
    def test_stats_track_retries(self, mock_post, mock_sleep):
        from requests.exceptions import ConnectionError

        mock_post.side_effect = ConnectionError("refused")

        client = OllamaClient(
            model="test", url="http://fake/api/generate", max_retries=2
        )
        client.post("prompt")

        stats = client.stats
        assert stats["total_calls"] == 1
        assert stats["retries"] == 2
        assert stats["failures"] == 1


class TestOllamaClientURLModelUpdates:
    """Tests for set_url and set_model."""

    def test_set_url_clears_cache(self):
        client = OllamaClient(model="test", url="http://old/api/generate")
        client._cache["key"] = "value"

        client.set_url("http://new/api/generate")

        assert client.url == "http://new/api/generate"
        assert len(client._cache) == 0

    def test_set_model(self):
        client = OllamaClient(model="old_model", url="http://fake/api/generate")
        client.set_model("new_model")
        assert client.model == "new_model"

    def test_set_url_noop_if_same(self):
        client = OllamaClient(model="test", url="http://same/api/generate")
        client._cache["key"] = "value"
        client.set_url("http://same/api/generate")
        assert "key" in client._cache


class TestKeywordExtractorFallback:
    """Integration tests for keyword extractor fallback on Ollama failure."""

    @patch("core.nlp.ollama_client.requests.post", side_effect=ConnectionError)
    def test_extract_keywords_uses_fallback(self, mock_post):
        try:
            import spacy  # noqa: F401
        except ImportError:
            return  # spacy not installed; skip gracefully

        from core.nlp.keyword_extractor import OllamaKeywordExtractor

        extractor = OllamaKeywordExtractor(model="test", url="http://fake/api/generate")
        extractor._client.max_retries = 1
        result = extractor.extract_keywords("Some text about technology", top_n=3)

        assert isinstance(result, list)
        assert len(result) <= 3
        assert all(isinstance(kw, str) for kw in result)

    @patch("core.nlp.ollama_client.requests.post", side_effect=ConnectionError)
    def test_extract_mood_keyword_uses_fallback(self, mock_post):
        try:
            import spacy  # noqa: F401
        except ImportError:
            return

        from core.nlp.keyword_extractor import OllamaKeywordExtractor

        extractor = OllamaKeywordExtractor(model="test", url="http://fake/api/generate")
        extractor._client.max_retries = 1
        result = extractor.extract_mood_keyword("A happy upbeat video about cooking")

        assert isinstance(result, str)
        assert len(result) > 0

    @patch("core.nlp.ollama_client.requests.post", side_effect=ConnectionError)
    def test_generate_script_from_text_returns_original_on_failure(self, mock_post):
        try:
            import spacy  # noqa: F401
        except ImportError:
            return

        from core.nlp.keyword_extractor import OllamaKeywordExtractor

        extractor = OllamaKeywordExtractor(model="test", url="http://fake/api/generate")
        extractor._client.max_retries = 1
        original = "This is my original script text."
        result = extractor.generate_script_from_text(original)

        assert result == original

    @patch("core.nlp.ollama_client.requests.post", side_effect=ConnectionError)
    def test_generate_topic_script_uses_fallback(self, mock_post):
        try:
            import spacy  # noqa: F401
        except ImportError:
            return

        from core.nlp.keyword_extractor import OllamaKeywordExtractor

        extractor = OllamaKeywordExtractor(model="test", url="http://fake/api/generate")
        extractor._client.max_retries = 1
        result = extractor.generate_topic_script("artificial intelligence")

        assert isinstance(result, str)
        assert len(result) > 0


class TestDefaultFallbacks:
    """Tests for default fallback constants."""

    def test_fallback_keywords_are_valid(self):
        assert isinstance(DEFAULT_FALLBACK_KEYWORDS, list)
        assert len(DEFAULT_FALLBACK_KEYWORDS) > 0
        assert all(isinstance(kw, str) for kw in DEFAULT_FALLBACK_KEYWORDS)

    def test_fallback_mood_is_string(self):
        assert isinstance(DEFAULT_FALLBACK_MOOD, str)
        assert len(DEFAULT_FALLBACK_MOOD) > 0

    def test_fallback_script_is_string(self):
        assert isinstance(DEFAULT_FALLBACK_SCRIPT, str)
        assert len(DEFAULT_FALLBACK_SCRIPT) > 0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
